"""
独立加载预训练的判别器，用于其他项目的感知损失。
支持四个判别器：MSD, MPD, MSSBSTFTD, MSSBCQTD。
"""

import os
import json
import torch
import torch.nn as nn

# 尝试导入safetensors，如果失败则回退到torch.load

from safetensors.torch import load_file


from models.vocoders.gan.discriminator.mpd import MultiPeriodDiscriminator
from models.vocoders.gan.discriminator.msd import MultiScaleDiscriminator
from models.vocoders.gan.discriminator.mssbcqtd import MultiScaleSubbandCQTDiscriminator
from models.vocoders.gan.discriminator.mssbstftd import MultiBandDiscriminator as MultiScaleSubbandSTFTDiscriminator



def dict_to_obj(d):
    """递归将字典转换为对象，允许点访问属性。"""
    if isinstance(d, dict):
        class Obj:
            pass
        obj = Obj()
        for k, v in d.items():
            setattr(obj, k, dict_to_obj(v))
        return obj
    elif isinstance(d, list):
        return [dict_to_obj(x) for x in d]
    else:
        return d


def load_config(config_path):
    """加载args.json配置文件并转换为对象。"""
    # 尝试使用utf-8-sig处理可能的BOM
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    try:
        config_dict = json.loads(content)
    except json.JSONDecodeError as e:
        # 如果标准JSON解析失败，尝试使用json5（如果可用）处理尾随逗号等
        try:
            import json5
            config_dict = json5.loads(content)
        except ImportError:
            # 如果json5不可用，尝试简单的修复：移除尾随逗号
            import re
            # 简单的尾随逗号修复（不完美）
            content = re.sub(r',\s*([\]}])', r'\1', content)
            config_dict = json.loads(content)
    cfg = dict_to_obj(config_dict)
    return cfg


def build_discriminator(cfg, disc_type):
    """根据类型构建判别器实例。"""
    if disc_type == "mpd":
        return MultiPeriodDiscriminator(cfg)
    elif disc_type == "msd":
        return MultiScaleDiscriminator(cfg)
    elif disc_type == "mssbstftd":
        return MultiScaleSubbandSTFTDiscriminator(cfg)
    elif disc_type == "mssbcqtd":
        return MultiScaleSubbandCQTDiscriminator(cfg)
    else:
        raise ValueError


def load_discriminator_weights(discriminator, weight_path):
    """加载权重文件 safetensors或pth。"""
    if weight_path.endswith('.safetensors'):
        state_dict = load_file(weight_path)
        # 检查是否有'module.'前缀，如果有则去掉
        if any(key.startswith('module.') for key in state_dict.keys()):
            # 去掉'module.'前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
    else:
        state_dict = torch.load(weight_path, map_location='cpu')
    discriminator.load_state_dict(state_dict)
    discriminator.eval()
    return discriminator


def load_all_discriminators(exp_dir, model_name='pupucodec'):
    """
    加载一个实验目录中的所有四个判别器。
    exp_dir: 实验目录，例如 'experiments/pupucodec'
    model_name: 'pupucodec' 或 'pupuvocoder'，用于确定配置文件路径。
    返回一个字典，键为判别器类型，值为加载好的判别器实例。
    """
    config_path = os.path.join(exp_dir, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'配置文件不存在: {config_path}')

    cfg = load_config(config_path)

    # 判别器顺序，参考args.json中的model.discriminators列表
    disc_order = cfg.model.discriminators
    print(f'判别器顺序: {disc_order}')

    discriminators = {}

    checkpoint_dir = os.path.join(exp_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        print(f'  错误: checkpoint目录不存在: {checkpoint_dir}')
        return discriminators

    checkpoint_items = []

    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            if any(fname.startswith('model_') and fname.endswith('.safetensors') for fname in os.listdir(item_path)):
                checkpoint_items.append(('dir', item_path))
        elif os.path.isfile(item_path) and os.path.getsize(item_path) > 100 * 1024 * 1024:  # 大于100MB的文件可能是打包检查点
            checkpoint_items.append(('file', item_path))

    if not checkpoint_items:
        print(f'  错误: 找不到有效的checkpoint')
        return discriminators

    # 使用第一个找到的checkpoint（通常只有一个）
    checkpoint_type, checkpoint_path = checkpoint_items[0]
    print(f'  使用checkpoint: {checkpoint_path} ({checkpoint_type})')

    if checkpoint_type == 'dir':
        # 标准目录结构：每个判别器有单独的safetensors文件
        for i, disc_type in enumerate(disc_order, start=1):
            print(f'加载 {disc_type}...')
            try:
                disc = build_discriminator(cfg, disc_type)
            except Exception as e:
                print(f'  跳过 {disc_type}: {e}')
                continue

            weight_path = os.path.join(checkpoint_path, f'model_{i}.safetensors')
            if not os.path.exists(weight_path):
                print(f'  警告: 找不到权重文件: {weight_path}')
                continue

            print(f'  使用权重文件: {weight_path}')
            try:
                disc = load_discriminator_weights(disc, weight_path)
                discriminators[disc_type] = disc
            except Exception as e:
                print(f'  加载权重失败: {e}')

    elif checkpoint_type == 'file':
        # 打包的检查点文件（如.zip或.pth）
        print(f'  注意: 使用打包检查点文件，需要特殊处理')
        # 尝试加载整个检查点并提取判别器
        try:
            # 尝试使用torch.load加载整个检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 检查检查点格式
            if isinstance(checkpoint, dict) and 'discriminators' in checkpoint:
                # 这是完整的训练检查点，包含generator和discriminators
                disc_dict = checkpoint.get('discriminators', {})
                for disc_type in disc_order:
                    if disc_type in disc_dict:
                        print(f'加载 {disc_type}...')
                        try:
                            disc = build_discriminator(cfg, disc_type)
                            disc.load_state_dict(disc_dict[disc_type])
                            disc.eval()
                            discriminators[disc_type] = disc
                            print(f'  从打包检查点加载成功')
                        except Exception as e:
                            print(f'  加载失败: {e}')
                    else:
                        print(f'  跳过 {disc_type}: 在检查点中未找到')
            else:
                print(f'  错误: 不支持的检查点格式')
        except Exception as e:
            print(f'  加载打包检查点失败: {e}')

    print(f'成功加载 {len(discriminators)} 个判别器: {list(discriminators.keys())}')
    return discriminators


class DiscriminatorFeatureExtractor:
    """
    封装所有判别器，提供特征提取功能。
    可以用于计算感知损失。
    """
    def __init__(self, discriminators, device='cpu'):
        self.discriminators = discriminators
        self.device = device
        for disc in self.discriminators.values():
            disc.to(device)

    def get_feature_maps(self, y, y_hat):
        """
        输入真实音频y和生成音频y_hat，形状为 (B, 1, T)。
        返回一个字典，包含每个判别器的特征图列表。
        特征图格式: 每个判别器返回 (fmap_rs, fmap_gs) 两个列表，每个列表包含各层的特征图。
        """
        with torch.no_grad():
            features = {}
            for disc_type, disc in self.discriminators.items():
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
                features[disc_type] = (fmap_rs, fmap_gs)
            return features

    def perceptual_loss(self, y, y_hat, loss_fn=nn.L1Loss()):
        """
        计算特征图之间的L1损失，类似于训练器中的feature损失。
        """
        total_loss = 0.0
        with torch.no_grad():
            for disc_type, disc in self.discriminators.items():
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
                disc_loss = 0.0
                for dr, dg in zip(fmap_rs, fmap_gs):
                    for rl, gl in zip(dr, dg):
                        disc_loss = disc_loss + loss_fn(rl, gl)
                total_loss = total_loss + disc_loss
        return total_loss


class SelectiveDiscriminatorFeatureExtractor(DiscriminatorFeatureExtractor):
    """
    选择性特征提取器，只选择每个判别器中有价值的特征层。
    基于经验选择中间层，避免过多计算和冗余特征。
    """
    def __init__(self, discriminators, device='cpu'):
        super().__init__(discriminators, device)

    def _select_msd_layers(self, fmap_rs, fmap_gs):
        """
        MSD: 3个尺度，每个尺度8层（7个conv + 1个conv_post）
        选择每个尺度的第3、5、7层（0-based索引: 2, 4, 6）
        跳过第一层（过于低级）和conv_post（过于高级）
        """
        selected_rs = []
        selected_gs = []
        for scale_rs, scale_gs in zip(fmap_rs, fmap_gs):
            # 每个尺度有8层，选择第3、5、7层（索引2,4,6）
            indices = [2, 4, 6]
            selected_rs.append([scale_rs[i] for i in indices])
            selected_gs.append([scale_gs[i] for i in indices])
        return selected_rs, selected_gs

    def _select_mpd_layers(self, fmap_rs, fmap_gs):
        """
        MPD: 8个周期，每个周期6层（5个conv + 1个conv_post）
        选择每个周期的第2、4层（0-based索引: 1, 3）
        跳过第一层（过于低级）和conv_post（过于高级）
        """
        selected_rs = []
        selected_gs = []
        for period_rs, period_gs in zip(fmap_rs, fmap_gs):
            # 每个周期有6层，选择第2、4层（索引1,3）
            indices = [1, 3]
            selected_rs.append([period_rs[i] for i in indices])
            selected_gs.append([period_gs[i] for i in indices])
        return selected_rs, selected_gs

    def _select_mssbstftd_layers(self, fmap_rs, fmap_gs):
        """
        MSSBSTFTD: 3个尺度，每个尺度21层（5频带×4层 + 1个conv_post）
        选择每个频带的第2、3层（0-based索引: 1, 2），加上conv_post（索引-1）
        总共每个尺度: 5×2 + 1 = 11层
        """
        selected_rs = []
        selected_gs = []
        for scale_rs, scale_gs in zip(fmap_rs, fmap_gs):
            # fmap结构：前20层是5个频带各4层，最后1层是conv_post
            scale_selected_rs = []
            scale_selected_gs = []

            # 处理每个频带的4层
            for band_start in range(0, 20, 4):
                # 选择每个频带的第2、3层（索引1,2）
                indices = [1, 2]
                for i in indices:
                    scale_selected_rs.append(scale_rs[band_start + i])
                    scale_selected_gs.append(scale_gs[band_start + i])

            # 添加conv_post（最后一层）
            scale_selected_rs.append(scale_rs[-1])
            scale_selected_gs.append(scale_gs[-1])

            selected_rs.append(scale_selected_rs)
            selected_gs.append(scale_selected_gs)
        return selected_rs, selected_gs

    def _select_mssbcqtd_layers(self, fmap_rs, fmap_gs):
        """
        MSSBCQTD: 3个尺度，每个尺度5层（5个conv）
        选择每个尺度的第2、4层（0-based索引: 1, 3）
        """
        selected_rs = []
        selected_gs = []
        for scale_rs, scale_gs in zip(fmap_rs, fmap_gs):
            # 每个尺度有5层，选择第2、4层（索引1,3）
            indices = [1, 3]
            selected_rs.append([scale_rs[i] for i in indices])
            selected_gs.append([scale_gs[i] for i in indices])
        return selected_rs, selected_gs

    def get_feature_maps(self, y, y_hat):
        """
        选择性获取特征图，只返回有价值的层。
        """
        with torch.no_grad():
            features = {}
            for disc_type, disc in self.discriminators.items():
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)

                # 根据判别器类型选择层
                if disc_type == 'msd':
                    selected_rs, selected_gs = self._select_msd_layers(fmap_rs, fmap_gs)
                elif disc_type == 'mpd':
                    selected_rs, selected_gs = self._select_mpd_layers(fmap_rs, fmap_gs)
                elif disc_type == 'mssbstftd':
                    selected_rs, selected_gs = self._select_mssbstftd_layers(fmap_rs, fmap_gs)
                elif disc_type == 'mssbcqtd':
                    selected_rs, selected_gs = self._select_mssbcqtd_layers(fmap_rs, fmap_gs)
                else:
                    # 未知类型，返回所有层
                    selected_rs, selected_gs = fmap_rs, fmap_gs

                features[disc_type] = (selected_rs, selected_gs)
            return features

    def perceptual_loss(self, y, y_hat, loss_fn=nn.L1Loss()):
        """
        计算选择性特征图之间的L1损失。
        """
        total_loss = 0.0
        with torch.no_grad():
            for disc_type, disc in self.discriminators.items():
                y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)

                # 根据判别器类型选择层
                if disc_type == 'msd':
                    selected_rs, selected_gs = self._select_msd_layers(fmap_rs, fmap_gs)
                elif disc_type == 'mpd':
                    selected_rs, selected_gs = self._select_mpd_layers(fmap_rs, fmap_gs)
                elif disc_type == 'mssbstftd':
                    selected_rs, selected_gs = self._select_mssbstftd_layers(fmap_rs, fmap_gs)
                elif disc_type == 'mssbcqtd':
                    selected_rs, selected_gs = self._select_mssbcqtd_layers(fmap_rs, fmap_gs)
                else:
                    # 未知类型，使用所有层
                    selected_rs, selected_gs = fmap_rs, fmap_gs

                disc_loss = 0.0
                for dr, dg in zip(selected_rs, selected_gs):
                    for rl, gl in zip(dr, dg):
                        disc_loss = disc_loss + loss_fn(rl, gl)
                total_loss = total_loss + disc_loss
        return total_loss


if __name__ == '__main__':
    # 示例用法：加载pupucodec的判别器并测试前向传播
    import sys

    # 选择实验目录
    exp_dir = 'experiments/pupucodec'  # 改为pupuvocoder以加载另一个模型
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]

    print(f'加载实验目录: {exp_dir}')

    discriminators = load_all_discriminators(exp_dir)
    print('成功加载所有判别器')

    # 移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = DiscriminatorFeatureExtractor(discriminators, device=device)

    # 创建随机测试音频（假设采样率44.1kHz，1秒音频）
    batch_size = 2
    sample_rate = 44100
    duration = 1.0
    t_samples = int(sample_rate * duration)
    y = torch.randn(batch_size, 1, t_samples).to(device)
    y_hat = torch.randn(batch_size, 1, t_samples).to(device)

    # 测试特征提取
    features = extractor.get_feature_maps(y, y_hat)
    print('特征提取成功')
    for disc_type, (fmap_rs, fmap_gs) in features.items():
        print(f'{disc_type}: 真实特征图数量 {len(fmap_rs)}, 生成特征图数量 {len(fmap_gs)}')

    # 测试感知损失
    loss = extractor.perceptual_loss(y, y_hat)
    print(f'感知损失值: {loss}')

