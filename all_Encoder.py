import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# 纹理鞅模块
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
                    log_feat = torch.log(feat_value + 1e-6)
                    M = torch.exp(self.theta * log_feat - 0.5 * self.theta ** 2)
                    channel_feat.append(M)
            glcm_feats.append(torch.stack(channel_feat, dim=1))
        out = torch.cat(glcm_feats, dim=1)
        return out

    def compute_glcm_features(self, patches):
        mean = patches.mean(dim=1, keepdim=True)
        std = patches.std(dim=1, keepdim=True) + 1e-6
        normed = (patches - mean) / std
        contrast = (normed ** 2).mean(dim=1)
        energy = (patches ** 2).mean(dim=1)
        entropy = -(patches * torch.log(patches + 1e-6)).mean(dim=1)
        homogeneity = 1.0 / (1.0 + (patches - mean).abs()).mean(dim=1)
        return {
            "contrast": contrast,
            "energy": energy,
            "entropy": entropy,
            "homogeneity": homogeneity,
        }


# 纹理引导Transformer
class TextureGuidedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias_channels):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj_bias = nn.Linear(bias_channels, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, texture_martingale):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        tm_flat = texture_martingale.flatten(2).transpose(1, 2)  # [B, HW, bias_channels]
        bias = self.proj_bias(tm_flat)  # [B, HW, C]
        q = k = self.norm(x_flat + bias)
        out, _ = self.attn(q, k, x_flat)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class CascadedMartingaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=False)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [B, 64, H/2, W/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [B, 256, H/4, W/4]
        self.layer2 = backbone.layer2  # [B, 512, H/8, W/8]
        self.layer3 = backbone.layer3  # [B, 1024, H/16, W/16]

        self.tm1 = TextureMartingaleModule(in_channels=3, dilation=1)
        self.tm2 = TextureMartingaleModule(in_channels=64, dilation=1)
        self.tm3 = TextureMartingaleModule(in_channels=256, dilation=1)
        self.tm4 = TextureMartingaleModule(in_channels=512, dilation=1)

        # 1x1卷积降维
        self.reduce_tm1 = nn.Conv2d(3 * 4, 12, 1)
        self.reduce_tm2 = nn.Conv2d(64 * 4, 12, 1)
        self.reduce_tm3 = nn.Conv2d(256 * 4, 12, 1)
        self.reduce_tm4 = nn.Conv2d(512 * 4, 12, 1)

        # 特征融合降通道
        self.fuse1 = nn.Conv2d(64 + 12, 64, 1)
        self.fuse2 = nn.Conv2d(256 + 12, 256, 1)
        self.fuse3 = nn.Conv2d(512 + 12, 512, 1)
        self.fuse4 = nn.Conv2d(1024 + 12, 1024, 1)

        # Transformer blocks
        self.trans1 = TextureGuidedTransformerBlock(dim=64, num_heads=4, bias_channels=12)
        self.trans2 = TextureGuidedTransformerBlock(dim=256, num_heads=8, bias_channels=12)
        self.trans3 = TextureGuidedTransformerBlock(dim=512, num_heads=8, bias_channels=12)
        self.trans4 = TextureGuidedTransformerBlock(dim=1024, num_heads=8, bias_channels=12)

    def forward(self, x0):
        # Stage 1
        x1 = self.layer0(x0)
        m1 = self.tm1(x0)
        m1 = self.reduce_tm1(m1)
        m1_up = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x1f = self.fuse1(torch.cat([x1, m1_up], dim=1))
        x1f = self.trans1(x1f, m1_up)



        # Stage 2
        x2 = self.layer1(x1f)
        m2 = self.tm2(x1)
        m2 = self.reduce_tm2(m2)
        m2_up = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x2f = self.fuse2(torch.cat([x2, m2_up], dim=1))
        x2f = self.trans2(x2f, m2_up)

        # Stage 3
        x3 = self.layer2(x2f)
        m3 = self.tm3(x2)
        m3 = self.reduce_tm3(m3)
        m3_up = F.interpolate(m3, size=x3.shape[2:], mode="bilinear", align_corners=False)
        x3f = self.fuse3(torch.cat([x3, m3_up], dim=1))
        x3f = self.trans3(x3f, m3_up)

        # Stage 4
        x4 = self.layer3(x3f)
        m4 = self.tm4(x3)
        m4 = self.reduce_tm4(m4)
        m4_up = F.interpolate(m4, size=x4.shape[2:], mode="bilinear", align_corners=False)
        x4f = self.fuse4(torch.cat([x4, m4_up], dim=1))
        x4f = self.trans4(x4f, m4_up)

        return x1f, x2f, x3f, x4f


if __name__ == "__main__":
    model = CascadedMartingaleEncoder()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    for i, out in enumerate(y):
        print(f"x{i + 1} shape:", out.shape)
