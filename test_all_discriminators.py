"""
测试加载所有四个预训练模型的判别器。
"""

import os
import sys
import torch

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_discriminators import load_all_discriminators, DiscriminatorFeatureExtractor


def test_experiment(exp_dir, name):
    """测试一个实验目录的判别器。"""
    print(f"\n{'='*60}")
    print(f"测试 {name}: {exp_dir}")
    print(f"{'='*60}")

    if not os.path.exists(exp_dir):
        print(f"实验目录不存在: {exp_dir}")
        return None


    discriminators = load_all_discriminators(exp_dir)
    if not discriminators:
        print("没有成功加载任何判别器")
        return None

    # 创建测试音频
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 采样率44.1kHz，0.5秒音频（避免内存过大）
    sample_rate = 44100
    duration = 0.5
    t_samples = int(sample_rate * duration)
    batch_size = 1

    print(f"创建测试音频: {batch_size} batch, {duration}秒, {t_samples}样本")
    y = torch.randn(batch_size, 1, t_samples).to(device)
    y_hat = torch.randn(batch_size, 1, t_samples).to(device)

    # 创建特征提取器
    extractor = DiscriminatorFeatureExtractor(discriminators, device=device)

    # 测试前向传播
    print("测试前向传播...")
    features = extractor.get_feature_maps(y, y_hat)

    print(f"成功加载 {len(discriminators)} 个判别器:")
    for disc_type, (fmap_rs, fmap_gs) in features.items():
        print(f"  {disc_type}: {len(fmap_rs)}个特征图组, 每组{len(fmap_rs[0]) if fmap_rs else 0}层")

    # 测试感知损失
    loss = extractor.perceptual_loss(y, y_hat)
    print(f"感知损失: {loss}")

    return discriminators




def main():
    """测试所有四个实验目录。"""
    experiments = [
        ("experiments/pupucodec", "PupuCodec Base"),
        ("experiments/pupucodec_large", "PupuCodec Large"),
        ("experiments/pupuvocoder", "PupuVocoder Base"),
        ("experiments/pupuvocoder_large", "PupuVocoder Large"),
    ]

    results = {}
    for exp_dir, name in experiments:
        disc = test_experiment(exp_dir, name)
        results[name] = disc is not None

    print(f"\n{'='*60}")
    print("测试总结:")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "成功" if success else "失败"
        print(f"{name}: {status}")


if __name__ == '__main__':
    main()