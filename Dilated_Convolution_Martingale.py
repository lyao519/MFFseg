import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class TextureMartingaleModule(nn.Module):
    def __init__(self, dilations=[1, 2, 3, 4], theta=1.0, sigma2=1.0,
                 include_features=["contrast", "energy", "entropy", "homogeneity"]):
        super(TextureMartingaleModule, self).__init__()
        self.dilations = dilations
        self.theta = theta
        self.include_features = include_features

    def forward(self, x):
        B, C, H, W = x.shape
        pyramid_outputs = []
        for dilation in self.dilations:
            kernel_size = 3 + 2 * (dilation - 1)
            pad = dilation
            scale_outputs = []
            for c in range(C):
                channel_feat = x[:, c:c + 1, :, :]
                unfolded = F.unfold(channel_feat, kernel_size=kernel_size, dilation=dilation, padding=pad)
                K = unfolded.shape[1]
                patches = unfolded.view(B, K, H, W)
                feats = self.compute_glcm_features(patches)
                feature_maps = []
                for feat_name, feat_value in feats.items():
                    if feat_name in self.include_features:
                        log_feat = torch.log(feat_value + 1e-6)
                        M = torch.exp(self.theta * log_feat - 0.5 * self.theta ** 2)
                        feature_maps.append(M)
                scale_outputs.append(torch.stack(feature_maps, dim=1))
            pyramid_scale = torch.cat(scale_outputs, dim=1)
            pyramid_outputs.append(pyramid_scale)
        return pyramid_outputs

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


class SEFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEFusion, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        squeeze = self.global_pool(x).view(B, C)
        excitation = self.fc2(self.relu(self.fc1(squeeze)))
        excitation = self.sigmoid(excitation).view(B, C, 1, 1)
        return x * excitation


class DynamicScaleSelector(nn.Module):
    def __init__(self, num_scales, feature_dim):
        super(DynamicScaleSelector, self).__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        weights = self.attn(inputs[0])
        weighted_sum = 0
        for i, feat in enumerate(inputs):
            weighted_sum += feat * weights[:, i:i + 1]
        return weighted_sum


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.initial = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return x1, x2, x3, x4


class UNetDecoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))

        self.final = nn.Conv2d(256, 1, 1)

    def forward(self, x1, x2, x3, x4):
        d3 = self.up3(x4)
        d3 = self.conv3(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        d2 = self.conv2(torch.cat([d2, x2], dim=1))
        d1 = self.up1(d2)
        d1 = self.conv1(torch.cat([d1, x1], dim=1))
        return self.final(d1)


class ResUNetWithTexture(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.texture = UNetEncoderWithTexture(in_channels=in_channels)
        self.decoder = UNetDecoder()

    def forward(self, x):
        texture_feat = self.texture(x)
        x1, x2, x3, x4 = self.encoder(x + texture_feat)  # optional fusion method
        out = self.decoder(x1, x2, x3, x4)
        return out


class UNetEncoderWithTexture(nn.Module):
    def __init__(self, in_channels=3, base_channels=64,
                 dilations=[1, 2, 3, 4], glcm_features=4):
        super().__init__()
        self.dilations = dilations
        self.texture = TextureMartingaleModule(dilations=dilations)
        texture_channels = in_channels * glcm_features

        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + texture_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            ) for _ in dilations
        ])

        self.fusion = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.se_block = SEFusion(base_channels)
        self.selector = DynamicScaleSelector(num_scales=len(dilations), feature_dim=base_channels)

    def forward(self, x):
        texture_pyramid = self.texture(x)
        branch_outputs = []
        for i, texture_feat in enumerate(texture_pyramid):
            inp = torch.cat([x, texture_feat], dim=1)
            out = self.conv_branches[i](inp)
            branch_outputs.append(out)
        dynamic_fused = self.selector(branch_outputs)
        dynamic_fused = self.fusion(dynamic_fused)
        dynamic_fused = self.se_block(dynamic_fused)
        return dynamic_fused
