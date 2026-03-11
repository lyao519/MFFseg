# 重新定义
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            channel = x[:, c:c + 1, :, :]  # 每个通道单独计算
            unfolded = F.unfold(channel, kernel_size=k, dilation=d, padding=pad)
            # unfold后: [B, k*k, H*W]
            K = unfolded.shape[1]
            patches = unfolded.view(B, K, H, W)
            # GLCM特征
            feats = self.compute_glcm_features(patches)
            # 将特征用martingale映射（图片中的exp公式）
            channel_feat = []
            for feat_name, feat_value in feats.items():
                if feat_name in self.include_features:
                    log_feat = torch.log(feat_value + 1e-6)
                    M = torch.exp(self.theta * log_feat - 0.5 * self.theta ** 2)
                    channel_feat.append(M)
            # shape: [4, B, H, W]
            glcm_feats.append(torch.stack(channel_feat, dim=1))
        # 拼接所有通道，输出shape: [B, C*4, H, W]
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


class CascadedMartingaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [B, 64, H/2, W/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [B, 256, H/4, W/4]
        self.layer2 = backbone.layer2  # [B, 512, H/8, W/8]
        self.layer3 = backbone.layer3  # [B, 1024, H/16, W/16]

        self.tm1 = TextureMartingaleModule(in_channels=3, dilation=1)
        self.tm2 = TextureMartingaleModule(in_channels=64, dilation=1)
        self.tm3 = TextureMartingaleModule(in_channels=256, dilation=1)
        self.tm4 = TextureMartingaleModule(in_channels=512, dilation=1)

        # 你可用1x1 conv降维，不然通道会很大！（C*4）
        self.reduce_tm1 = nn.Conv2d(3 * 4, 12, 1)
        self.reduce_tm2 = nn.Conv2d(64 * 4, 12, 1)
        self.reduce_tm3 = nn.Conv2d(256 * 4, 12, 1)
        self.reduce_tm4 = nn.Conv2d(512 * 4, 12, 1)

        self.fuse1 = nn.Conv2d(64 + 12, 64, 1)
        self.fuse2 = nn.Conv2d(256 + 12, 256, 1)
        self.fuse3 = nn.Conv2d(512 + 12, 512, 1)
        self.fuse4 = nn.Conv2d(1024 + 12, 1024, 1)

    def forward(self, x0):
        # 输入图像
        x = x0
        x1 = self.layer0(x)
        m1 = self.tm1(x)
        m1 = self.reduce_tm1(m1)
        x1f = self.fuse1(torch.cat([x1, F.interpolate(m1, x1.shape[2:])], dim=1))

        x2 = self.layer1(x1)
        m2 = self.tm2(x1)
        m2 = self.reduce_tm2(m2)
        x2f = self.fuse2(torch.cat([x2, F.interpolate(m2, x2.shape[2:])], dim=1))

        x3 = self.layer2(x2)
        m3 = self.tm3(x2)
        m3 = self.reduce_tm3(m3)
        x3f = self.fuse3(torch.cat([x3, F.interpolate(m3, x3.shape[2:])], dim=1))

        x4 = self.layer3(x3)
        m4 = self.tm4(x3)
        m4 = self.reduce_tm4(m4)
        x4f = self.fuse4(torch.cat([x4, F.interpolate(m4, x4.shape[2:])], dim=1))

        return x1f, x2f, x3f, x4f


if __name__ == "__main__":
    model = CascadedMartingaleEncoder()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    for i, out in enumerate(y):
        print(f"x{i + 1} shape:", out.shape)
    #     torch.save(out, f"feature_output_x{i + 1}.pt")
    # print("Feature maps saved as 'feature_output_x1.pt' to 'x4.pt'")







