import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureMartingaleModule(nn.Module):
    def __init__(self, window_sizes=[3, 5, 7, 9], theta=1.0, sigma2=1.0,
                 include_features=["contrast", "energy", "entropy", "homogeneity"]):
        super(TextureMartingaleModule, self).__init__()
        self.window_sizes = window_sizes
        self.theta = theta
        self.include_features = include_features

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Tensor of shape [B, C * len(window_sizes) * num_features, H, W]
        """
        B, C, H, W = x.shape
        outputs = []

        for c in range(C):  # 对每个通道独立处理
            channel_feat = x[:, c:c + 1, :, :]  # [B, 1, H, W]
            channel_outputs = []

            for win_size in self.window_sizes:
                pad = win_size // 2
                patches = F.unfold(channel_feat, kernel_size=win_size, padding=pad)
                K = win_size * win_size
                patches = patches.view(B, K, H, W)

                feats = self.compute_glcm_features(patches)

                for feat_name, feat_value in feats.items():
                    if feat_name in self.include_features:
                        log_feat = torch.log(feat_value + 1e-6)
                        M = torch.exp(self.theta * log_feat - 0.5 * self.theta ** 2)
                        channel_outputs.append(M)

            channel_output = torch.stack(channel_outputs, dim=1)  # [B, F, H, W]
            outputs.append(channel_output)

        final = torch.cat(outputs, dim=1)  # [B, C * F, H, W]
        return final

    def compute_glcm_features(self, patches):
        """
        patches: [B, K, H, W] where K = window_size*window_size
        Returns: dict of GLCM-style features
        """
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


class UNetEncoderWithTexture(nn.Module):
    def __init__(self, in_channels=3, base_channels=64,
                 window_sizes=[3, 5, 7, 9], glcm_features=4):
        super().__init__()
        self.texture = TextureMartingaleModule(window_sizes=window_sizes)
        texture_channels = in_channels * len(window_sizes) * glcm_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + texture_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        texture_feat = self.texture(x)  # [B, extra_channels, H, W]
        x = torch.cat([x, texture_feat], dim=1)
        return self.conv1(x)
