import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet50


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


class DPCAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b
        self.kernel_size = self.get_adaptive_kernel(channels)
        self.conv1d = nn.Conv1d(2, 1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def get_adaptive_kernel(self, C):
        return max(3, int(abs((math.log2(C) + self.b) / self.gamma)))

    def forward(self, Oi, Di):
        B, C, H, W = Oi.shape
        gap_O = F.adaptive_avg_pool2d(Oi, 1).view(B, C)
        gap_D = F.adaptive_avg_pool2d(Di, 1).view(B, C)
        concat = torch.stack([gap_O, gap_D], dim=1)  # [B, 2, C]
        attn = self.conv1d(concat).squeeze(1)        # [B, C]
        attn = self.sigmoid(attn).view(B, C, 1, 1)
        fused = Oi * attn + Di
        return fused


class CascadedMartingaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=False)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.tm1 = TextureMartingaleModule(in_channels=3)
        self.tm2 = TextureMartingaleModule(in_channels=64)
        self.tm3 = TextureMartingaleModule(in_channels=256)
        self.tm4 = TextureMartingaleModule(in_channels=512)

        self.reduce_tm1 = nn.Conv2d(3 * 4, 12, 1)
        self.reduce_tm2 = nn.Conv2d(64 * 4, 12, 1)
        self.reduce_tm3 = nn.Conv2d(256 * 4, 12, 1)
        self.reduce_tm4 = nn.Conv2d(512 * 4, 12, 1)

        self.dpca1 = DPCAModule(channels=64)
        self.dpca2 = DPCAModule(channels=256)
        self.dpca3 = DPCAModule(channels=512)
        self.dpca4 = DPCAModule(channels=1024)

        self.fuse1 = nn.Conv2d(64, 64, 1)
        self.fuse2 = nn.Conv2d(256, 256, 1)
        self.fuse3 = nn.Conv2d(512, 512, 1)
        self.fuse4 = nn.Conv2d(1024, 1024, 1)

    def forward(self, x):
        x1 = self.layer0(x)
        m1 = self.reduce_tm1(self.tm1(x))
        x1f = self.fuse1(self.dpca1(x1, F.interpolate(m1, size=x1.shape[2:], mode="bilinear")))

        x2 = self.layer1(x1f)
        m2 = self.reduce_tm2(self.tm2(x1))
        x2f = self.fuse2(self.dpca2(x2, F.interpolate(m2, size=x2.shape[2:], mode="bilinear")))

        x3 = self.layer2(x2f)
        m3 = self.reduce_tm3(self.tm3(x2))
        x3f = self.fuse3(self.dpca3(x3, F.interpolate(m3, size=x3.shape[2:], mode="bilinear")))

        x4 = self.layer3(x3f)
        m4 = self.reduce_tm4(self.tm4(x3))
        x4f = self.fuse4(self.dpca4(x4, F.interpolate(m4, size=x4.shape[2:], mode="bilinear")))

        return x1f, x2f, x3f, x4f


if __name__ == '__main__':
    model = CascadedMartingaleEncoder()
    x = torch.randn(1, 3, 256, 256)
    outputs = model(x)
    for i, feat in enumerate(outputs):
        print(f"x{i+1}f shape: {feat.shape}")