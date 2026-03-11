# 重新定义
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math
from torchvision.models import resnet18


def safe_log(x, eps=1e-5):
    x = torch.clamp(x, min=eps)
    return torch.log(x)


def safe_exp(x, max_val=15):
    x = torch.clamp(x, max=max_val)
    return torch.exp(x)


def check_nan(tensor, name):
    if not torch.isfinite(tensor).all():
        print(f"[NaN or Inf] detected in {name}: min={tensor.min().item()}, max={tensor.max().item()}")


class TextureMartingaleModule(nn.Module):
    def __init__(self, in_channels, dilation=1, theta=1.0, include_features=None):
        super().__init__()
        self.dilation = dilation
        self.theta = theta
        self.include_features = include_features or ["contrast", "energy", "entropy", "homogeneity"]
        self.in_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        d = self.dilation
        k = 3 + 2 * (d - 1)
        pad = d
        glcm_feats = []
        for c in range(C):
            channel = x[:, c:c + 1, :, :]
            unfolded = F.unfold(channel, kernel_size=k, dilation=d, padding=pad)
            K = unfolded.shape[1]
            patches = unfolded.view(B, K, H, W)
            feats = self.compute_glcm_features(patches)
            channel_feat = []
            for feat_name, feat_value in feats.items():
                if feat_name in self.include_features:
                    log_feat = safe_log(feat_value)
                    M = safe_exp(self.theta * log_feat - 0.5 * self.theta ** 2)
                    M = torch.clamp(M, min=1e-4, max=1e4)
                    M = torch.where(torch.isfinite(M), M, torch.full_like(M, 1.0))
                    channel_feat.append(M)
                    # 额外检查
                    check_nan(M, f"Martingale-{feat_name}-channel{c}")
            glcm_feats.append(torch.stack(channel_feat, dim=1))
        out = torch.cat(glcm_feats, dim=1)
        check_nan(out, "TextureMartingaleModule-out")
        return out

    def compute_glcm_features(self, patches):
        mean = patches.mean(dim=1, keepdim=True)
        std = patches.std(dim=1, keepdim=True).clamp(min=1e-3)  # 防止除0
        normed = (patches - mean) / std
        contrast = (normed ** 2).mean(dim=1)
        energy = (patches ** 2).mean(dim=1)

        clamped = torch.clamp(patches, min=1e-6)
        entropy = -(clamped * torch.log(clamped)).mean(dim=1)

        homogeneity = 1.0 / ((1.0 + (patches - mean).abs()).mean(dim=1) + 1e-6)
        return {
            "contrast": contrast,
            "energy": energy,
            "entropy": entropy,
            "homogeneity": homogeneity,
        }


# 构建轻量级transformer模块，并根据所设计的纹理鞅
# 用纹理鞅引导注意力聚焦于纹理显著区域
#
class LightTransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.texture_proj = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, x, texture=None):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x_norm = self.norm1(x_flat)

        if texture is not None:
            texture_map = self.texture_proj(texture)  # [B, 1, H, W]
            texture_map = F.interpolate(texture_map, size=(H, W), mode="bilinear", align_corners=False)
            texture_bias = texture_map.flatten(2).transpose(1, 2)  # [B, N, 1]
            texture_bias = texture_bias.expand(-1, N, C)  # Match attention output shape [B, N, C]
        else:
            texture_bias = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=None)
        attn_out = attn_out + texture_bias
        x = x_flat + attn_out
        x = x + self.ffn(self.norm2(x))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class CascadedMartingaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        # ResNet18结构调整：需要扩展到高通道，可仿照ResNet50的每层通道
        # 这里只做演示，如果你的主干就是ResNet50，建议直接用resnet50（这样和右侧完全对齐）

        # 层定义
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256

        # 1. 每一层的 martingale module
        self.tm1 = TextureMartingaleModule(in_channels=3, dilation=1)
        self.tm2 = TextureMartingaleModule(in_channels=64, dilation=1)
        self.tm3 = TextureMartingaleModule(in_channels=256, dilation=1)
        self.tm4 = TextureMartingaleModule(in_channels=512, dilation=1)

        # 2. 降维到和主干一样（比如 [3*4, 64*4, 256*4, 512*4] -> [12, 256, 512, 1024]）
        self.reduce_tm1 = nn.Conv2d(3 * 4, 12, 1)
        self.reduce_tm2 = nn.Conv2d(64 * 4, 64, 1)
        self.reduce_tm3 = nn.Conv2d(256 * 4, 256, 1)
        self.reduce_tm4 = nn.Conv2d(512 * 4, 512, 1)

        # 3. 融合（cat后和输出 shape 完全一致！）
        self.fuse1 = nn.Conv2d(64 + 12, 64, 1)
        self.fuse2 = nn.Conv2d(64 + 64, 256, 1)
        self.fuse3 = nn.Conv2d(128 + 256, 512, 1)
        self.fuse4 = nn.Conv2d(256 + 512, 1024, 1)

    def forward(self, x0):
        x = x0
        x1 = self.layer0(x)  # 64
        m1 = self.tm1(x)
        m1 = self.reduce_tm1(m1)
        x1f = self.fuse1(torch.cat([x1, F.interpolate(m1, x1.shape[2:])], dim=1))
        # x1f = self.trans1(x1f)  # [B, 64, 128, 128]

        x2 = self.layer1(x1)  # 64
        m2 = self.tm2(x1f)
        m2 = self.reduce_tm2(m2)
        x2f = self.fuse2(torch.cat([x2, F.interpolate(m2, x2.shape[2:])], dim=1))
        # x2f = self.trans2(x2f)  # [B, 256, 64, 64]

        x3 = self.layer2(x2)  # 128
        m3 = self.tm3(x2f)
        m3 = self.reduce_tm3(m3)
        x3f = self.fuse3(torch.cat([x3, F.interpolate(m3, x3.shape[2:])], dim=1))
        # x3f = self.trans3(x3f)  # [B, 512, 32, 32]

        x4 = self.layer3(x3)  # 256
        m4 = self.tm4(x3f)
        m4 = self.reduce_tm4(m4)
        x4f = self.fuse4(torch.cat([x4, F.interpolate(m4, x4.shape[2:])], dim=1))
        # x4f = self.trans4(x4f)  # [B, 1024, 16, 16]

        return x1f, x2f, x3f, x4f


"""
另一条支路
UNetTransformerBranch 的目标：
- 在编码器阶段设计一条独立于主干 ResNet + 纹理鞅 的全局建模支路；
- 使用轻量 Transformer 提取多尺度全局上下文特征；
- 输出尺寸与主干网络的中高层对齐，用于后续融合；
- 提升感受野、建模长距离上下文依赖，为解码器提供更丰富的表征能力；
- 特别适用于医学图像或纹理敏感任务中的结构增强与语义表达。
"""


class UNetTransformerBranch(nn.Module):
    """
    - 四层输出，分别对齐ResNet主干1/2, 1/4, 1/8, 1/16分辨率
    - 用轻量Transformer块进行全局建模
    """

    def __init__(self, in_channels=3, embed_dim=64, depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 8)):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # 第一层输出（1/2分辨率）
        self.layer1 = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        # 1/4
        self.down2 = nn.MaxPool2d(2)
        self.trans2 = nn.Sequential(*[
            LightTransformerBlock(64, num_heads[0])
            for _ in range(depths[0])
        ])
        self.out_proj_2 = nn.Conv2d(64, 256, kernel_size=1)

        # 1/8
        self.down3 = nn.MaxPool2d(2)
        self.trans3 = nn.Sequential(*[
            LightTransformerBlock(256, num_heads[1])
            for _ in range(depths[1])
        ])
        self.out_proj_3 = nn.Conv2d(256, 512, kernel_size=1)

        # 1/16
        self.down4 = nn.MaxPool2d(2)
        self.trans4 = nn.Sequential(*[
            LightTransformerBlock(512, num_heads[2])
            for _ in range(depths[2])
        ])
        self.out_proj_4 = nn.Conv2d(512, 1024, kernel_size=1)

    def forward(self, x):
        # x: [B, 3, H, W] (e.g., 256x256)
        x1 = self.stem(x)  # [B, 64, 128, 128]
        x1 = self.layer1(x1)  # [B, 64, 128, 128]

        x2 = self.down2(x1)  # [B, 64, 64, 64]
        x2 = self.trans2(x2)
        x2 = self.out_proj_2(x2)  # [B, 256, 64, 64]

        x3 = self.down3(x2)  # [B, 256, 32, 32]
        x3 = self.trans3(x3)
        x3 = self.out_proj_3(x3)  # [B, 512, 32, 32]

        x4 = self.down4(x3)  # [B, 512, 16, 16]
        x4 = self.trans4(x4)
        x4 = self.out_proj_4(x4)  # [B, 1024, 16, 16]

        return x1, x2, x3, x4


"""
下面的深度特征融合模块
"""
class EnhancedDPCA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size
        k = int(abs((math.log2(channels) / gamma) + b / gamma))
        k = k if k % 2 == 1 else k + 1
        self.adapt_conv1d = nn.Conv1d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        # Residual FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        # Learnable channel gate (可学习门控)
        self.gate = nn.Parameter(torch.randn(1, channels))
        # BatchNorm
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, Oi, Di):
        # Oi, Di: [B, C, H, W]
        B, C, H, W = Oi.shape

        # 1. GAP
        gap_O = F.adaptive_avg_pool2d(Oi, 1).view(B, C)
        gap_D = F.adaptive_avg_pool2d(Di, 1).view(B, C)

        # 2. Stack + Adaptive Conv1D
        stack = torch.stack([gap_O, gap_D], dim=1)  # [B, 2, C]
        conv_out = self.adapt_conv1d(stack).squeeze(1)  # [B, C]

        # 3. Residual FFN + skip
        ffn_out = self.ffn(conv_out)
        residual = conv_out + ffn_out  # 残差连接

        # 4. Learnable Channel Gate
        gated = residual + self.gate  # 通道门控

        # 5. BatchNorm + Sigmoid
        attn = gated.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        attn = self.bn(attn)
        mask = torch.sigmoid(attn)

        # 6. Channel-wise 乘法
        fused = Oi * mask
        return fused


class FinalDPCAFusion(nn.Module):
    def __init__(self, channels=1024):
        super().__init__()
        self.dpca = EnhancedDPCA(channels=channels)

    def forward(self, feat1, feat2):
        fused = self.dpca(feat1, feat2)
        return fused


"""
跳跃连接
CAB
"""


class CABModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cnn_feat, trans_feat):
        z_cnn = self.avg_pool(cnn_feat)
        z_trans = self.avg_pool(trans_feat)
        s_cnn = self.fc2(self.relu(self.fc1(z_cnn)))
        s_trans = self.fc2(self.relu(self.fc1(z_trans)))
        attn = self.sigmoid(s_cnn + s_trans)
        out = cnn_feat * attn
        return out


"""
开始设计简单解码器部分
# ===================== 用法示意 =======================
# --- 1. 编码器输出 ---
# (假定已经获得四层主干、四层全局支路、四层skip，每层都用DualBranchDynamicSkip融合)
# 假设：
# res_texture_feats = (x1f, x2f, x3f, x4f)  # 主干四层输出
# trans_feats      = (t1,  t2,  t3,  t4)    # transformer分支四层输出
# skip0 = DualBranchDynamicSkip(64)(x1f, t1)
# skip1 = DualBranchDynamicSkip(256)(x2f, t2)
# skip2 = DualBranchDynamicSkip(512)(x3f, t3)
# skip3 = DualBranchDynamicSkip(1024)(x4f, t4)

# --- 2. 最深层融合 ---
# deep_feat = FinalDPCAFusion(1024)(x4f, t4)   # 增强DPCA融合

# --- 3. 解码器 ---
# decoder = DecoderWithDualBranchSkip()
# pred_mask = decoder(deep_feat, skip3, skip2, skip1, skip0)
# print(pred_mask.shape)  # [B, 1, 128, 128]  二值分割图

# =======================
"""
"""
更改之后的
Pyramid SCCA + 递归跳跃融合解码器（DualBranchSkipDecoder with Pyramid SCCA）

"""


# 解码器部分，直接cat，上采样
# class SimpleCatDecoder(nn.Module):
#     def __init__(self, skip_channels=[1024, 512, 256, 64], out_channels=1):
#         super().__init__()
#         # 假定输入顺序：skip3, skip2, skip1, skip0
#         self.conv1 = nn.Conv2d(skip_channels[0] + 1024, 512, 3, padding=1)  # skip3 + t4 -> 512
#         self.conv2 = nn.Conv2d(512 + skip_channels[1], 256, 3, padding=1)  # 上采样后 + skip2 -> 256
#         self.conv3 = nn.Conv2d(256 + skip_channels[2], 128, 3, padding=1)  # 上采样后 + skip1 -> 128
#         self.conv4 = nn.Conv2d(128 + skip_channels[3], 64, 3, padding=1)  # 上采样后 + skip0 -> 64
#         self.head = nn.Conv2d(64, out_channels, 1)
#
#     def forward(self, skip0, skip1, skip2, skip3, t4):
#         # 1. skip3 + t4
#         x = torch.cat([skip3, t4], dim=1)  # [B, 1024+1024, 16, 16]
#         x = F.relu(self.conv1(x))  # [B, 512, 16, 16]
#         # 2. 上采样+cat skip2
#         x = F.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x, skip2], dim=1)  # [B, 512+512, 32, 32]
#         x = F.relu(self.conv2(x))  # [B, 256, 32, 32]
#         # 3. 上采样+cat skip1
#         x = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x, skip1], dim=1)  # [B, 256+256, 64, 64]
#         x = F.relu(self.conv3(x))  # [B, 128, 64, 64]
#         # 4. 上采样+cat skip0
#         x = F.interpolate(x, size=skip0.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([x, skip0], dim=1)  # [B, 128+64, 128, 128]
#         x = F.relu(self.conv4(x))  # [B, 64, 128, 128]
#         # 5. 最后上采样到 256x256
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 64, 256, 256]
#         out = self.head(x)  # [B, 1, 256, 256]
#         return torch.sigmoid(out)


# # 加入了FCU框架
# class FCUModule(nn.Module):
#     def __init__(self, conv_channels, trans_channels):
#         super().__init__()
#         assert len(conv_channels) == len(trans_channels)
#         self.fuse_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(c + t, c, kernel_size=1),
#                 nn.BatchNorm2d(c),
#                 nn.ReLU(inplace=True)
#             ) for c, t in zip(conv_channels, trans_channels)
#         ])
#         self.trans_fuse_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(c + t, t, kernel_size=1),
#                 nn.BatchNorm2d(t),
#                 nn.ReLU(inplace=True)
#             ) for c, t in zip(conv_channels, trans_channels)
#         ])
#
#     def forward(self, conv_feats, trans_feats):
#         out_conv, out_trans = [], []
#         for i in range(len(conv_feats)):
#             c_feat, t_feat = conv_feats[i], trans_feats[i]
#             t_to_c = self.fuse_blocks[i](torch.cat([c_feat, t_feat], dim=1))
#             c_to_t = self.trans_fuse_blocks[i](torch.cat([c_feat, t_feat], dim=1))
#             out_conv.append(t_to_c)
#             out_trans.append(c_to_t)
#         return out_conv, out_trans
# class PyramidSCCA(nn.Module):
#     def __init__(self, channels, dilations=[1, 2, 3], out_channels=None):
#         super().__init__()
#         if out_channels is None:
#             out_channels = channels
#         self.sccas = nn.ModuleList([
#             nn.Conv2d(channels, out_channels, kernel_size=3, padding=d, dilation=d)
#             for d in dilations
#         ])
#         self.merge = nn.Conv2d(out_channels * len(dilations), out_channels, 1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         feats = [scca(x) for scca in self.sccas]
#         x = torch.cat(feats, dim=1)
#         x = self.merge(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
#
# class DualBranchPyramidDecoder(nn.Module):
#     def __init__(self, channels_list, out_channels=1):
#         super().__init__()
#         self.scca4 = PyramidSCCA(channels_list[0])
#         self.scca3 = PyramidSCCA(channels_list[1])
#         self.scca2 = PyramidSCCA(channels_list[2])
#         self.scca1 = PyramidSCCA(channels_list[3])
#
#         self.cab4 = CABModule(channels_list[0])
#         self.cab3 = CABModule(channels_list[1])
#         self.cab2 = CABModule(channels_list[2])
#         self.cab1 = CABModule(channels_list[3])
#
#         self.up_conv3 = nn.Conv2d(channels_list[0], channels_list[1], 1)
#         self.up_conv2 = nn.Conv2d(channels_list[1], channels_list[2], 1)
#         self.up_conv1 = nn.Conv2d(channels_list[2], channels_list[3], 1)
#         self.head = nn.Conv2d(channels_list[3], out_channels, 1)
#
#     def forward(self, O4, O3, O2, O1, D4):
#         O4s = self.scca4(O4)
#         O3s = self.scca3(O3)
#         O2s = self.scca2(O2)
#         O1s = self.scca1(O1)
#
#         D3 = self.cab4(O4s, D4)
#         D3_up = F.interpolate(self.up_conv3(D3), scale_factor=2, mode='bilinear', align_corners=False)
#
#         D2 = self.cab3(O3s, D3_up)
#         D2_up = F.interpolate(self.up_conv2(D2), scale_factor=2, mode='bilinear', align_corners=False)
#
#         D1 = self.cab2(O2s, D2_up)
#         D1_up = F.interpolate(self.up_conv1(D1), scale_factor=2, mode='bilinear', align_corners=False)
#
#         out_feat = self.cab1(O1s, D1_up)
#         out = self.head(out_feat)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
#         return torch.sigmoid(out)
#
#
# class MedSegNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = CascadedMartingaleEncoder()
#         self.trans_branch = UNetTransformerBranch()
#
#         self.fcu = FCUModule(
#             conv_channels=[64, 256, 512],
#             trans_channels=[64, 256, 512]
#         )
#
#         self.cab0 = CABModule(64)
#         self.cab1 = CABModule(256)
#         self.cab2 = CABModule(512)
#         self.cab3 = CABModule(1024)
#
#         self.decoder = DualBranchPyramidDecoder(
#             channels_list=[1024, 512, 256, 64],
#             out_channels=1
#         )
#
#     def forward(self, x):
#         x1f, x2f, x3f, x4f = self.encoder(x)
#         t1, t2, t3, t4 = self.trans_branch(x)
#
#         [x1f, x2f, x3f], [t1, t2, t3] = self.fcu([x1f, x2f, x3f], [t1, t2, t3])
#
#         skip0 = self.cab0(x1f, t1)
#         skip1 = self.cab1(x2f, t2)
#         skip2 = self.cab2(x3f, t3)
#         skip3 = self.cab3(x4f, t4)
#
#         out = self.decoder(
#             O4=skip3,
#             O3=skip2,
#             O2=skip1,
#             O1=skip0,
#             D4=x4f  # 深层特征直接送入解码器
#         )
#         return out


# 精简版
class FCUModule(nn.Module):
    def __init__(self, conv_channels, trans_channels):
        super().__init__()
        assert len(conv_channels) == len(trans_channels)
        self.fuse_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + t, c, kernel_size=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c, t in zip(conv_channels, trans_channels)
        ])
        self.trans_fuse_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + t, t, kernel_size=1),
                nn.BatchNorm2d(t),
                nn.ReLU(inplace=True)
            ) for c, t in zip(conv_channels, trans_channels)
        ])

    def forward(self, conv_feats, trans_feats):
        out_conv, out_trans = [], []
        for i in range(len(conv_feats)):
            c_feat, t_feat = conv_feats[i], trans_feats[i]
            t_to_c = self.fuse_blocks[i](torch.cat([c_feat, t_feat], dim=1))
            c_to_t = self.trans_fuse_blocks[i](torch.cat([c_feat, t_feat], dim=1))
            out_conv.append(t_to_c)
            out_trans.append(c_to_t)
        return out_conv, out_trans


class PyramidSCCA(nn.Module):
    def __init__(self, channels, dilations=[1, 2, 3], out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.sccas = nn.ModuleList([
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.merge = nn.Conv2d(out_channels * len(dilations), out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feats = [scca(x) for scca in self.sccas]
        x = torch.cat(feats, dim=1)
        x = self.merge(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DualBranchPyramidDecoder(nn.Module):
    def __init__(self, channels_list, out_channels=1):
        super().__init__()
        self.scca4 = PyramidSCCA(channels_list[0])
        self.scca3 = PyramidSCCA(channels_list[1])
        self.scca2 = PyramidSCCA(channels_list[2])
        self.scca1 = PyramidSCCA(channels_list[3])

        self.cab4 = CABModule(channels_list[0])
        self.cab3 = CABModule(channels_list[1])
        self.cab2 = CABModule(channels_list[2])
        self.cab1 = CABModule(channels_list[3])

        self.up_conv3 = nn.Conv2d(channels_list[0], channels_list[1], 1)
        self.up_conv2 = nn.Conv2d(channels_list[1], channels_list[2], 1)
        self.up_conv1 = nn.Conv2d(channels_list[2], channels_list[3], 1)
        self.head = nn.Conv2d(channels_list[3], out_channels, 1)

    def forward(self, O4, O3, O2, O1, D4):
        O4s = self.scca4(O4)
        O3s = self.scca3(O3)
        O2s = self.scca2(O2)
        O1s = self.scca1(O1)

        D3 = self.cab4(O4s, D4)
        D3_up = F.interpolate(self.up_conv3(D3), scale_factor=2, mode='bilinear', align_corners=False)

        D2 = self.cab3(O3s, D3_up)
        D2_up = F.interpolate(self.up_conv2(D2), scale_factor=2, mode='bilinear', align_corners=False)

        D1 = self.cab2(O2s, D2_up)
        D1_up = F.interpolate(self.up_conv1(D1), scale_factor=2, mode='bilinear', align_corners=False)

        out_feat = self.cab1(O1s, D1_up)
        out = self.head(out_feat)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


class MedSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CascadedMartingaleEncoder()
        self.trans_branch = UNetTransformerBranch()

        self.fcu = FCUModule(
            conv_channels=[64, 256, 512],
            trans_channels=[64, 256, 512]
        )

        self.cab0 = CABModule(64)
        self.cab1 = CABModule(256)
        self.cab2 = CABModule(512)
        self.cab3 = CABModule(1024)

        self.decoder = DualBranchPyramidDecoder(
            channels_list=[1024, 512, 256, 64],
            out_channels=1
        )

    def forward(self, x):
        x1f, x2f, x3f, x4f = self.encoder(x)
        t1, t2, t3, t4 = self.trans_branch(x)

        [x1f, x2f, x3f], [t1, t2, t3] = self.fcu([x1f, x2f, x3f], [t1, t2, t3])

        skip0 = self.cab0(x1f, t1)
        skip1 = self.cab1(x2f, t2)
        skip2 = self.cab2(x3f, t3)
        skip3 = self.cab3(x4f, t4)

        out = self.decoder(
            O4=skip3,
            O3=skip2,
            O2=skip1,
            O1=skip0,
            D4=x4f  # 深层特征直接送入解码器
        )
        return out


if __name__ == '__main__':
    model = MedSegNet()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print("Output shape:", out.shape)
    # Step 4: 打印输出信息
    # print("Output shape:", out.shape)  # 通常为 [B, 1, H, W]，如 [1, 1, 128, 128]
    print("Output range: min {:.4f}, max {:.4f}".format(out.min().item(), out.max().item()))

# if __name__ == "__main__":
#     model_1 = UNetTransformerBranch()
#     model_2 = CascadedMartingaleEncoder()
#     x = torch.randn(4, 3, 256, 256)
#     y_1 = model_1(x)
#     y_2 = model_2(x)
#     print(' UNetTransformerBranch')
#     for i, out in enumerate(y_1):
#         print(f"x{1 + i} shape:", out.shape)
#     print('CascadedMartingaleEncoder')
#     for i, out in enumerate(y_2):
#         print(f"x{i + 1}f shape:", out.shape)
# def main():
#     # Step 1: 初始化模型
#     model = MedSegNet()
#     model.eval()  # 测试模式，关闭Dropout/BatchNorm的训练行为
#
#     # Step 2: 构建假输入数据 [B, 3, H, W]
#     input_tensor = torch.randn(1, 3, 256, 256)  # 可根据需要调整输入尺寸
#
#     # Step 3: 模型前向传播
#     with torch.no_grad():  # 测试时关闭梯度计算
#         output = model(input_tensor)
#
#     # Step 4: 打印输出信息
#     print("Output shape:", output.shape)  # 通常为 [B, 1, H, W]，如 [1, 1, 128, 128]
#     print("Output range: min {:.4f}, max {:.4f}".format(output.min().item(), output.max().item()))


# def count_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# print(f"Total trainable parameters: {count_params(model):,}")

# if __name__ == "__main__":
#
#     model_1 = UNetTransformerBranch()
#     model_2 = CascadedMartingaleEncoder()
#     x = torch.randn(4, 3, 256, 256)
#     y_1 = model_1(x)
#     y_2 = model_2(x)
#     print(' UNetTransformerBranch')
#     for i, out in enumerate(y_1):
#         print(f"x{1 + i} shape:", out.shape)
#     print('CascadedMartingaleEncoder')
#     for i, out in enumerate(y_2):
#         print(f"x{i + 1}f shape:", out.shape)
# def main():
#     # Step 1: 初始化模型
#     model = MedSegNet()
#     model.eval()  # 测试模式，关闭Dropout/BatchNorm的训练行为
#
#     # Step 2: 构建假输入数据 [B, 3, H, W]
#     input_tensor = torch.randn(4, 3, 256, 256)  # 可根据需要调整输入尺寸
#
#     # Step 3: 模型前向传播
#     with torch.no_grad():  # 测试时关闭梯度计算
#         output = model(input_tensor)
#
#     # Step 4: 打印输出信息
#     print("Output shape:", output.shape)  # 通常为 [B, 1, H, W]，如 [1, 1, 128, 128]
#     print("Output range: min {:.4f}, max {:.4f}".format(output.min().item(), output.max().item()))
#
#
# # def count_params(model):
# #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# #
# #
# # print(f"Total trainable parameters: {count_params(model):,}")
#
# if __name__ == "__main__":
#     main()
