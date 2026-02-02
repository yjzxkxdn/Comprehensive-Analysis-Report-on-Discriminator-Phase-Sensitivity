"""
测试判别器对相位不同但频谱相同音频的敏感性。
使用希尔伯特变换生成解析信号，修改相位，生成频谱相同但相位不同的音频对。
计算同一判别器对这些音频的特征相似度，按层深度分析。
"""

import os
from pathlib import Path
import sys
import json
import time
import warnings
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_discriminators import load_all_discriminators, DiscriminatorFeatureExtractor


def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """
    使用FFT实现希尔伯特变换。
    输入: 实信号 x, 形状 (..., N)
    输出: 解析信号 (复信号), 形状 (..., N)
    """
    N = x.shape[-1]

    # FFT
    X = torch.fft.fft(x, dim=-1)

    # 创建希尔伯特变换滤波器
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        # 偶数长度
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        # 奇数长度
        h[0] = 1
        h[1:(N+1)//2] = 2

    # 应用滤波器
    X = X * h

    # IFFT
    analytic = torch.fft.ifft(X, dim=-1)
    return analytic


def create_phase_shifted_signals(original_audio: torch.Tensor,
                                 phase_shifts: List[float] = None,
                                 method: str = 'hilbert') -> Dict[str, torch.Tensor]:
    """
    创建频谱相同但相位不同的信号。

    参数:
        original_audio: 原始音频, 形状 (B, 1, T) 或 (1, T)
        phase_shifts: 相位偏移列表 (弧度), 默认为 [0, π/4, π/2, π, 3π/2]
        method: 方法, 'hilbert' 或 'fft'

    返回:
        字典, 键为描述, 值为相位偏移后的音频
    """
    if phase_shifts is None:
        phase_shifts = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]

    if len(original_audio.shape) == 2:
        # (1, T) -> (1, 1, T)
        original_audio = original_audio.unsqueeze(1)
    elif len(original_audio.shape) == 3:
        # (B, 1, T)
        pass
    else:
        raise ValueError(f"输入音频形状应为 (B, 1, T) 或 (1, T), 但得到 {original_audio.shape}")

    results = {}

    if method == 'hilbert':
        # 方法1: 希尔伯特变换
        for phase_shift in phase_shifts:
            analytic_signal = hilbert_transform(original_audio)

            # 修改相位: 乘以 exp(j*phase_shift)
            # 创建复数相位因子张量
            phase_factor = torch.tensor(np.exp(1j * phase_shift), dtype=torch.complex64, device=analytic_signal.device)
            shifted_analytic = analytic_signal * phase_factor

            # 取实部作为新的音频信号
            shifted_audio = shifted_analytic.real

            desc = f"相位偏移_{phase_shift:.3f}rad"
            results[desc] = shifted_audio

    elif method == 'fft':
        # 方法2: FFT方法 (保持幅度不变，修改相位)
        for phase_shift in phase_shifts:
            # FFT
            X = torch.fft.fft(original_audio, dim=-1)

            # 获取幅度和相位
            magnitude = torch.abs(X)
            phase = torch.angle(X)

            # 修改相位: 添加常数相位偏移
            new_phase = phase + phase_shift

            # 重建信号
            shifted_X = magnitude * torch.exp(1j * new_phase)
            shifted_audio = torch.fft.ifft(shifted_X, dim=-1).real

            desc = f"相位偏移_{phase_shift:.3f}rad"
            results[desc] = shifted_audio

    else:
        raise ValueError(f"未知方法: {method}")

    # 添加原始音频作为参考
    results["原始音频_0.0rad"] = original_audio

    return results


def create_test_audio_set(sample_rate=44100, device='cpu',
                         audio_paths: Union[str, List[str], Path, List[Path]] = None,
                         target_durations: List[float] = None):
    """
    创建测试音频集，从本地文件读取音频。

    参数:
        sample_rate: 目标采样率
        device: 设备
        audio_paths: 音频文件路径或路径列表。如果为None，使用默认示例文件
        target_durations: 目标持续时间列表（秒）。如果为None，使用原始音频长度

    返回:
        audio_tests: 音频张量列表, 每个形状 (1, 1, T)
        test_descriptions: 描述列表
    """
    audio_tests = []
    test_descriptions = []
    
    # 处理输入路径
    if audio_paths is None:
        # 如果没有提供路径，使用一些常见测试文件位置
        default_paths = [
            "test_audio/sine_440hz.wav",
            "test_audio/music_sample.wav",
            "test_audio/speech_sample.wav"
        ]
        # 过滤存在的文件
        audio_paths = [p for p in default_paths if Path(p).exists()]
        
        if not audio_paths:
            # 如果没有找到文件，创建示例音频
            return create_fallback_audio(sample_rate, device)
    
    elif isinstance(audio_paths, (str, Path)):
        audio_paths = [audio_paths]
    
    # 确保所有路径都是Path对象
    audio_paths = [Path(p) if isinstance(p, str) else p for p in audio_paths]
    
    for audio_path in audio_paths:
        if not audio_path.exists():
            warnings.warn(f"文件不存在: {audio_path}, 跳过")
            continue
        
        try:
            # 使用librosa加载音频，支持多种格式
            audio, orig_sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
            
            # 转换为torch张量
            audio_tensor = torch.from_numpy(audio).float().to(device)
            
            if target_durations is None:
                # 使用整个音频
                segments = [audio_tensor]
                durations = [len(audio) / sample_rate]
            else:
                # 根据目标持续时间创建多个片段
                segments = []
                durations = []
                for target_duration in target_durations:
                    target_samples = int(target_duration * sample_rate)
                    if len(audio_tensor) >= target_samples:
                        # 取前target_samples个样本
                        segment = audio_tensor[:target_samples]
                        segments.append(segment)
                        durations.append(target_duration)
                    else:
                        # 如果音频太短，进行零填充
                        padding = target_samples - len(audio_tensor)
                        segment = torch.nn.functional.pad(
                            audio_tensor, 
                            (0, padding),
                            mode='constant',
                            value=0
                        )
                        segments.append(segment)
                        durations.append(len(audio_tensor) / sample_rate)  # 实际有效时长
            
            # 处理每个片段
            for i, (segment, duration) in enumerate(zip(segments, durations)):
                # 添加通道和批次维度: (T) -> (1, 1, T)
                segment = segment.unsqueeze(0).unsqueeze(0)
                
                # 归一化到[-1, 1]范围，避免削波
                max_val = torch.max(torch.abs(segment))
                if max_val > 0:
                    segment = segment / max_val
                
                audio_tests.append(segment)
                
                # 创建描述
                if len(segments) > 1:
                    desc = f"{audio_path.stem}_片段{i+1}_{duration:.2f}s"
                else:
                    desc = f"{audio_path.stem}_{duration:.2f}s"
                test_descriptions.append(desc)
                
        except Exception as e:
            warnings.warn(f"无法加载文件 {audio_path}: {str(e)}")
            continue
    
    if not audio_tests:
        # 如果没有成功加载任何文件，创建回退音频
        return create_fallback_audio(sample_rate, device)
    
    return audio_tests, test_descriptions


def create_fallback_audio(sample_rate, device):
    """创建回退测试音频，当没有找到音频文件时使用"""
    audio_tests = []
    test_descriptions = []
    
    durations = [0.5, 1.0]
    
    for duration in durations:
        t_samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, t_samples, device=device)
        
        # 正弦波
        sine_wave = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0)
        audio_tests.append(sine_wave)
        test_descriptions.append(f"正弦波_440Hz_{duration}s")
        
        # 方波
        square_wave = torch.sign(torch.sin(2 * np.pi * 220 * t))
        square_wave = square_wave.unsqueeze(0).unsqueeze(0)
        audio_tests.append(square_wave)
        test_descriptions.append(f"方波_220Hz_{duration}s")
        
        # 白噪声
        noise = torch.randn(1, 1, t_samples, device=device) * 0.3
        audio_tests.append(noise)
        test_descriptions.append(f"白噪声_{duration}s")
    
    return audio_tests, test_descriptions



def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """计算两个张量展平后的余弦相似度。"""
    v1 = vec1.flatten()
    v2 = vec2.flatten()
    return (torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)).item()


def compute_layerwise_similarity(features1, features2, disc_type):
    """
    计算两个特征图集合之间的逐层相似度。

    返回:
        layer_similarities: 字典列表, 每个字典包含层信息和相似度
        depth_stats: 按深度分组的统计信息
    """
    fmap_rs1, fmap_gs1 = features1[disc_type]
    fmap_rs2, fmap_gs2 = features2[disc_type]

    layer_similarities = []

    # 计算真实音频特征图的相似度 (fmap_rs)
    for scale_idx, (scale_rs1, scale_rs2) in enumerate(zip(fmap_rs1, fmap_rs2)):
        for layer_idx, (layer_rs1, layer_rs2) in enumerate(zip(scale_rs1, scale_rs2)):
            sim = cosine_similarity(layer_rs1, layer_rs2)
            layer_info = {
                'disc_type': disc_type,
                'scale_idx': scale_idx,
                'layer_idx': layer_idx,
                'similarity': sim.item() if hasattr(sim, 'item') else sim,
                'feature_type': 'real',
                'depth': layer_idx  # 简单深度: 层索引
            }
            layer_similarities.append(layer_info)

    # 可选: 计算生成音频特征图的相似度 (fmap_gs)
    # 为简化，这里只使用真实音频特征

    # 按深度分析
    depth_stats = {}
    for layer_info in layer_similarities:
        depth = layer_info['depth']
        if depth not in depth_stats:
            depth_stats[depth] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'similarities': []
            }

        sim = layer_info['similarity']
        depth_stats[depth]['count'] += 1
        depth_stats[depth]['sum'] += sim
        depth_stats[depth]['min'] = min(depth_stats[depth]['min'], sim)
        depth_stats[depth]['max'] = max(depth_stats[depth]['max'], sim)
        depth_stats[depth]['similarities'].append(sim)

    # 计算平均值
    for depth in depth_stats:
        depth_stats[depth]['mean'] = depth_stats[depth]['sum'] / depth_stats[depth]['count']
        depth_stats[depth]['std'] = np.std(depth_stats[depth]['similarities']) if depth_stats[depth]['count'] > 1 else 0.0

    return layer_similarities, depth_stats


def compute_phase_sensitivity(extractor, original_audio, phase_shifts=None, method='hilbert'):
    """
    计算判别器对相位变化的敏感性。

    返回:
        phase_similarities: 相位偏移 -> 判别器类型 -> 相似度统计
        layer_analysis: 详细的逐层分析
    """
    if phase_shifts is None:
        phase_shifts = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]

    # 生成相位偏移信号
    phase_signals = create_phase_shifted_signals(original_audio, phase_shifts, method)

    # 获取原始音频的特征
    original_features = extractor.get_feature_maps(original_audio, original_audio)

    # 初始化结果结构
    phase_similarities = {}
    layer_analysis = {}

    for phase_desc, shifted_audio in phase_signals.items():
        if phase_desc == "原始音频_0.0rad":
            continue  # 跳过原始音频

        # 获取相位偏移音频的特征
        shifted_features = extractor.get_feature_maps(shifted_audio, shifted_audio)

        phase_similarities[phase_desc] = {}
        layer_analysis[phase_desc] = {}

        for disc_type in ['msd', 'mpd', 'mssbstftd', 'mssbcqtd']:
            if disc_type not in original_features or disc_type not in shifted_features:
                continue

            # 计算逐层相似度
            layer_sims, depth_stats = compute_layerwise_similarity(
                {disc_type: original_features[disc_type]},
                {disc_type: shifted_features[disc_type]},
                disc_type
            )

            # 计算该判别器的总体相似度
            all_sims = [layer['similarity'] for layer in layer_sims]
            mean_sim = np.mean(all_sims) if all_sims else 0.0
            std_sim = np.std(all_sims) if len(all_sims) > 1 else 0.0

            phase_similarities[phase_desc][disc_type] = {
                'mean': mean_sim,
                'std': std_sim,
                'min': np.min(all_sims) if all_sims else 0.0,
                'max': np.max(all_sims) if all_sims else 0.0,
                'count': len(all_sims)
            }

            # 保存层分析
            layer_analysis[phase_desc][disc_type] = {
                'layer_similarities': layer_sims,
                'depth_stats': depth_stats
            }

    return phase_similarities, layer_analysis


def print_phase_sensitivity_results(phase_similarities, layer_analysis):
    """打印相位敏感性结果。"""
    print(f"\n{'='*80}")
    print("相位敏感性分析结果")
    print(f"{'='*80}")

    # 打印总体相似度表格
    print(f"\n不同相位偏移下的平均特征相似度:")
    print(f"{'相位偏移':<20} {'MSD':<10} {'MPD':<10} {'MSSBSTFTD':<12} {'MSSBCQTD':<12}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

    for phase_desc in sorted(phase_similarities.keys()):
        row = f"{phase_desc:<20}"
        for disc_type in ['msd', 'mpd', 'mssbstftd', 'mssbcqtd']:
            if disc_type in phase_similarities[phase_desc]:
                mean_sim = phase_similarities[phase_desc][disc_type]['mean']
                row += f" {mean_sim:<10.4f}"
            else:
                row += f" {'-':<10}"
        print(row)

    # 按判别器类型分析
    print(f"\n{'='*80}")
    print("按判别器类型分析")
    print(f"{'='*80}")

    for disc_type in ['msd', 'mpd', 'mssbstftd', 'mssbcqtd']:
        print(f"\n{disc_type}:")

        # 收集该判别器所有相位偏移的相似度
        all_sims = []
        for phase_desc, disc_dict in phase_similarities.items():
            if disc_type in disc_dict:
                all_sims.append(disc_dict[disc_type]['mean'])

        if all_sims:
            mean_all = np.mean(all_sims)
            std_all = np.std(all_sims)
            print(f"  平均相似度 (所有相位偏移): {mean_all:.4f} ± {std_all:.4f}")
            print(f"  范围: [{np.min(all_sims):.4f}, {np.max(all_sims):.4f}]")

            # 相位敏感性指标: 相似度随相位偏移的变化程度
            phase_sensitivity = std_all / mean_all if mean_all > 0 else 0.0
            print(f"  相位敏感性 (变异系数): {phase_sensitivity:.4f}")

    # 按深度分析
    print(f"\n{'='*80}")
    print("按网络深度分析 (以MSD为例)")
    print(f"{'='*80}")

    # 选择一个相位偏移和判别器进行深度分析
    example_phase = list(phase_similarities.keys())[0] if phase_similarities else None
    if example_phase and 'msd' in layer_analysis.get(example_phase, {}):
        depth_stats = layer_analysis[example_phase]['msd']['depth_stats']

        print(f"\n相位偏移: {example_phase}, 判别器: MSD")
        print(f"{'深度 (层索引)':<20} {'平均相似度':<15} {'标准差':<15} {'样本数':<10}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*10}")

        for depth in sorted(depth_stats.keys()):
            stats = depth_stats[depth]
            print(f"{depth:<20} {stats['mean']:<15.4f} {stats['std']:<15.4f} {stats['count']:<10}")


def print_detailed_layer_similarities(layer_analysis, phase_desc=None, disc_type=None):
    """
    打印详细的逐层相似度信息。

    参数:
        layer_analysis: 层分析字典
        phase_desc: 指定的相位偏移描述，如果为None则使用第一个
        disc_type: 指定的判别器类型，如果为None则使用msd
    """
    if not layer_analysis:
        print("没有层分析数据")
        return

    if phase_desc is None:
        phase_desc = list(layer_analysis.keys())[0]

    if disc_type is None:
        disc_type = 'msd'

    if phase_desc not in layer_analysis or disc_type not in layer_analysis[phase_desc]:
        print(f"找不到 {phase_desc} 中的 {disc_type} 数据")
        return

    analysis = layer_analysis[phase_desc][disc_type]
    layer_similarities = analysis['layer_similarities']

    print(f"\n{'='*80}")
    print(f"逐层相似度详情: {phase_desc}, 判别器: {disc_type}")
    print(f"{'='*80}")

    # 按尺度和层排序
    layer_similarities.sort(key=lambda x: (x.get('scale_idx', 0), x.get('layer_idx', 0)))

    print(f"\n{'尺度':<5} {'层索引':<8} {'相似度':<12} {'深度':<6}")
    print(f"{'-'*5} {'-'*8} {'-'*12} {'-'*6}")

    for layer_info in layer_similarities:
        scale_idx = layer_info.get('scale_idx', 0)
        layer_idx = layer_info.get('layer_idx', 0)
        similarity = layer_info.get('similarity', 0.0)
        depth = layer_info.get('depth', layer_idx)

        print(f"{scale_idx:<5} {layer_idx:<8} {similarity:<12.6f} {depth:<6}")

    # 统计信息
    sim_values = [layer_info.get('similarity', 0.0) for layer_info in layer_similarities]
    if sim_values:
        print(f"\n统计信息:")
        print(f"  层数: {len(sim_values)}")
        print(f"  平均相似度: {np.mean(sim_values):.6f}")
        print(f"  标准差: {np.std(sim_values):.6f}")
        print(f"  最小相似度: {np.min(sim_values):.6f}")
        print(f"  最大相似度: {np.max(sim_values):.6f}")

        # 按深度分组统计
        depth_groups = {}
        for layer_info in layer_similarities:
            depth = layer_info.get('depth', layer_info.get('layer_idx', 0))
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(layer_info.get('similarity', 0.0))

        print(f"\n按深度统计:")
        print(f"{'深度':<8} {'平均相似度':<12} {'标准差':<12} {'层数':<6}")
        print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*6}")
        for depth in sorted(depth_groups.keys()):
            depth_sims = depth_groups[depth]
            print(f"{depth:<8} {np.mean(depth_sims):<12.6f} {np.std(depth_sims):<12.6f} {len(depth_sims):<6}")


def save_results_to_json(phase_similarities, layer_analysis, audio_desc, output_file='phase_sensitivity_results.json'):
    """保存结果到JSON文件。"""
    # 转换为可JSON序列化的格式
    serializable_results = {
        'audio_description': audio_desc,
        'phase_similarities': {},
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 保存相位相似度
    for phase_desc, disc_dict in phase_similarities.items():
        serializable_results['phase_similarities'][phase_desc] = {}
        for disc_type, stats in disc_dict.items():
            serializable_results['phase_similarities'][phase_desc][disc_type] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'count': int(stats['count'])
            }

    # 保存层分析（简化版，只保存统计信息）
    serializable_results['layer_analysis_summary'] = {}
    for phase_desc, disc_dict in layer_analysis.items():
        serializable_results['layer_analysis_summary'][phase_desc] = {}
        for disc_type, analysis in disc_dict.items():
            # 只保存深度统计摘要
            depth_summary = {}
            for depth, stats in analysis['depth_stats'].items():
                depth_summary[str(depth)] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'count': int(stats['count']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                }
            serializable_results['layer_analysis_summary'][phase_desc][disc_type] = depth_summary

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")


def main():
    """主函数。"""
    print("判别器相位敏感性分析")
    print(f"{'='*80}")
    audio_paths = r"E:\测试音频\女-凄美地 - Vocal.wav"

    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 选择模型 (使用pupucodec作为示例)
    exp_dir = 'experiments/pupuvocoder_large'
    print(f"加载模型: {exp_dir}")

    # 加载判别器
    discriminators = load_all_discriminators(exp_dir)
    if not discriminators:
        print(f"错误: 无法加载判别器")
        return

    print(f"成功加载 {len(discriminators)} 个判别器: {list(discriminators.keys())}")

    # 创建特征提取器
    extractor = DiscriminatorFeatureExtractor(discriminators, device=device)

    # 创建测试音频集
    print(f"\n创建测试音频集...")
    audio_tests, audio_descriptions = create_test_audio_set(
        sample_rate=44100, device=device, audio_paths= audio_paths,target_durations=[0.5]
    )

    print(f"创建了 {len(audio_tests)} 个测试音频:")
    for i, desc in enumerate(audio_descriptions):
        print(f"  {i+1:2d}. {desc}")

    # 相位偏移列表
    phase_shifts = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    print(f"\n测试相位偏移: {phase_shifts}")

    all_results = {}

    # 对每个测试音频进行分析
    for audio_idx, (audio, audio_desc) in enumerate(zip(audio_tests, audio_descriptions)):
        print(f"\n{'='*80}")
        print(f"分析音频 {audio_idx+1}: {audio_desc}")
        print(f"{'='*80}")

        # 计算相位敏感性
        phase_similarities, layer_analysis = compute_phase_sensitivity(
            extractor, audio, phase_shifts, method='hilbert'
        )

        # 打印结果
        #print_phase_sensitivity_results(phase_similarities, layer_analysis)

        # 打印详细的逐层相似度（仅第一个音频）
        if audio_idx == 0:
            # 选择第一个非零相位偏移
            non_zero_phases = [p for p in layer_analysis.keys()]
            for example_phase in non_zero_phases[2:4]:
                # 对所有判别器类型打印逐层相似度
                for disc_type in ['msd', 'mpd', 'mssbstftd', 'mssbcqtd']:
                    if disc_type in layer_analysis.get(example_phase, {}):
                        print_detailed_layer_similarities(layer_analysis, example_phase, disc_type)

        # 保存结果
        all_results[audio_desc] = {
            'phase_similarities': phase_similarities,
            'layer_analysis': layer_analysis
        }

        # 保存到文件
        output_file = f'phase_sensitivity_{audio_idx+1}.json'
        save_results_to_json(phase_similarities, layer_analysis, audio_desc, output_file)

    # 生成汇总报告
    print(f"\n{'='*80}")
    print("相位敏感性分析汇总报告")
    print(f"{'='*80}")

    # 计算跨音频的平均相似度
    overall_stats = {disc_type: [] for disc_type in ['msd', 'mpd', 'mssbstftd', 'mssbcqtd']}

    for audio_desc, results in all_results.items():
        phase_similarities = results['phase_similarities']

        for phase_desc, disc_dict in phase_similarities.items():
            for disc_type, stats in disc_dict.items():
                overall_stats[disc_type].append(stats['mean'])

    print(f"\n跨所有音频和相位偏移的平均相似度:")
    for disc_type in ['msd', 'mpd', 'mssbstftd', 'mssbcqtd']:
        if overall_stats[disc_type]:
            mean_sim = np.mean(overall_stats[disc_type])
            std_sim = np.std(overall_stats[disc_type])
            print(f"  {disc_type:<12}: {mean_sim:.4f} ± {std_sim:.4f}")

    print(f"\n测试完成!")


if __name__ == '__main__':
    main()