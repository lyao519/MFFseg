# # coding: utf-8
# """
# ResUNet with Texture Martingale Cascade:
# At each layer, perform:
# - Standard convolution: x1 = conv(x0), x2 = conv(x1), x3 = conv(x2), x4 = conv(x3)
# - Texture martingale: m1 = TM(x0), m2 = TM(concat(x1, m1)), m3 = TM(concat(x2, m2)), m4 = TM(concat(x3, m3))
# - Final output: concat(x4, m4)
# - Fuse xi with mi via concat + 1x1 conv + BN + ReLU + optional SE
#
# GLCM-based martingale feature maps: contrast, entropy, energy, homogeneity
# """
#
# from __future__ import annotations
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50
#
# # ------------------------------
# # SE Block
# # ------------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
# # ------------------------------
# # Texture Martingale via GLCM Features
# # ------------------------------
# class TextureMartingaleModule(nn.Module):
#     def __init__(self, dilations=[1, 2, 3, 4], theta=1.0, include_features=None):
#         super().__init__()
#         self.dilations = dilations
#         self.theta = theta
#         self.include_features = include_features or ["contrast", "energy", "entropy", "homogeneity"]
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         pyramid_outputs = []
#         for d in self.dilations:
#             k = 3 + 2 * (d - 1)
#             pad = d
#             scale_outputs = []
#             for c in range(C):
#                 feat = x[:, c:c + 1, :, :]
#                 unfolded = F.unfold(feat, kernel_size=k, dilation=d, padding=pad)
#                 _, K, L = unfolded.shape
#                 patch_H = int(H)
#                 patch_W = int(W)
#                 patches = unfolded.view(B, K, patch_H, patch_W)
#                 feats = self.compute_glcm_features(patches)
#                 feature_maps = []
#                 for name, val in feats.items():
#                     if name in self.include_features:
#                         log_feat = torch.log(val + 1e-6)
#                         M = torch.exp(self.theta * log_feat - 0.5 * self.theta**2)
#                         feature_maps.append(M)
#                 scale_outputs.append(torch.stack(feature_maps, dim=1))
#             pyramid_scale = torch.cat(scale_outputs, dim=1)
#             pyramid_outputs.append(pyramid_scale)
#         return F.interpolate(pyramid_outputs[0], size=(H, W), mode="bilinear", align_corners=False)
#
#     def compute_glcm_features(self, patches):
#         mean = patches.mean(dim=1, keepdim=True)
#         std = patches.std(dim=1, keepdim=True) + 1e-6
#         normed = (patches - mean) / std
#         contrast = (normed ** 2).mean(dim=1)
#         energy = (patches ** 2).mean(dim=1)
#         entropy = -(patches * torch.log(patches + 1e-6)).mean(dim=1)
#         homogeneity = 1.0 / (1.0 + (patches - mean).abs()).mean(dim=1)
#         return {
#             "contrast": contrast,
#             "energy": energy,
#             "entropy": entropy,
#             "homogeneity": homogeneity,
#         }
#
# # ------------------------------
# # Cascaded Encoder using TM features + SE + BN + ReLU
# # ------------------------------
# class CascadedMartingaleEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = resnet50()
#         self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
#         self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#
#         self.tm1 = TextureMartingaleModule([1])
#         self.tm2 = TextureMartingaleModule([2])
#         self.tm3 = TextureMartingaleModule([3])
#         self.tm4 = TextureMartingaleModule([4])
#
#         self.fuse1 = self._make_fuse_layer(64 + 12, 64)
#         self.fuse2 = self._make_fuse_layer(256 + 12, 256)
#         self.fuse3 = self._make_fuse_layer(512 + 12, 512)
#         self.fuse4 = self._make_fuse_layer(1024 + 12, 1024)
#
#     def _make_fuse_layer(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             SEBlock(out_ch)
#         )
#
#     def forward(self, x):
#         x0 = x
#         x1 = self.layer0(x0)
#         m1 = self.tm1(x0)
#         m1_up = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
#         x1f = self.fuse1(torch.cat([x1, m1_up], dim=1))
#
#         x2 = self.layer1(x1f)
#         m2 = self.tm2(torch.cat([x2, m1_up], dim=1))
#         m2_up = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         x2f = self.fuse2(torch.cat([x2, m2_up], dim=1))
#
#         x3 = self.layer2(x2f)
#         m3 = self.tm3(torch.cat([x3, m2_up], dim=1))
#         m3_up = F.interpolate(m3, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         x3f = self.fuse3(torch.cat([x3, m3_up], dim=1))
#
#         x4 = self.layer3(x3f)
#         m4 = self.tm4(torch.cat([x4, m3_up], dim=1))
#         m4_up = F.interpolate(m4, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         x4f = self.fuse4(torch.cat([x4, m4_up], dim=1))
#
#         return x1f, x2f, x3f, x4f
#
# if __name__ == "__main__":
#     model = CascadedMartingaleEncoder()
#     x = torch.randn(2, 3, 256, 256)
#     y = model(x)
#     for i, out in enumerate(y):
#         print(f"x{i+1} shape:", out.shape)


# coding: utf-8
# """
# ResUNet with Texture Martingale Cascade:
# At each layer, perform:
# - Standard convolution: x1 = conv(x0), x2 = conv(x1), x3 = conv(x2), x4 = conv(x3)
# - Texture martingale: m1 = TM(x0), m2 = TM(concat(x1, m1)), m3 = TM(concat(x2, m2)), m4 = TM(concat(x3, m3))
# - Final output: concat(x4, m4)
# - Fuse xi with mi via concat + 1x1 conv + BN + ReLU + optional SE
#
# GLCM-based martingale feature maps: contrast, entropy, energy, homogeneity
# """
#
# from __future__ import annotations
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50
#
# # ------------------------------
# # SE Block
# # ------------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
# # ------------------------------
# # Texture Martingale via GLCM Features
# # ------------------------------
# class TextureMartingaleModule(nn.Module):
#     def __init__(self, dilations=[1, 2, 3, 4], theta=1.0, include_features=None):
#         super().__init__()
#         self.dilations = dilations
#         self.theta = theta
#         self.include_features = include_features or ["contrast", "energy", "entropy", "homogeneity"]
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         pyramid_outputs = []
#         for d in self.dilations:
#             k = 3 + 2 * (d - 1)
#             pad = d
#             scale_outputs = []
#             for c in range(C):
#                 feat = x[:, c:c + 1, :, :]
#                 unfolded = F.unfold(feat, kernel_size=k, dilation=d, padding=pad)
#                 _, K, L = unfolded.shape
#                 patch_H = int(H)
#                 patch_W = int(W)
#                 patches = unfolded.view(B, K, patch_H, patch_W)
#                 feats = self.compute_glcm_features(patches)
#                 feature_maps = []
#                 for name, val in feats.items():
#                     if name in self.include_features:
#                         log_feat = torch.log(val + 1e-6)
#                         M = torch.exp(self.theta * log_feat - 0.5 * self.theta**2)
#                         feature_maps.append(M)
#                 scale_outputs.append(torch.stack(feature_maps, dim=1))
#             pyramid_scale = torch.cat(scale_outputs, dim=1)
#             pyramid_outputs.append(pyramid_scale)
#         return F.interpolate(pyramid_outputs[0], size=(H, W), mode="bilinear", align_corners=False)
#
#     def compute_glcm_features(self, patches):
#         mean = patches.mean(dim=1, keepdim=True)
#         std = patches.std(dim=1, keepdim=True) + 1e-6
#         normed = (patches - mean) / std
#         contrast = (normed ** 2).mean(dim=1)
#         energy = (patches ** 2).mean(dim=1)
#         entropy = -(patches * torch.log(patches + 1e-6)).mean(dim=1)
#         homogeneity = 1.0 / (1.0 + (patches - mean).abs()).mean(dim=1)
#         return {
#             "contrast": contrast,
#             "energy": energy,
#             "entropy": entropy,
#             "homogeneity": homogeneity,
#         }
#
# # ------------------------------
# # Cascaded Encoder using TM features + SE + BN + ReLU
# # ------------------------------
# class CascadedMartingaleEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = resnet50()
#         self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
#         self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#
#         self.tm1 = TextureMartingaleModule([1])
#         self.tm2 = TextureMartingaleModule([2])
#         self.tm3 = TextureMartingaleModule([3])
#         self.tm4 = TextureMartingaleModule([4])
#
#         self.fuse1 = self._make_fuse_layer(64 + 12, 64)
#         self.fuse2 = self._make_fuse_layer(256 + 12, 256)
#         self.fuse3 = self._make_fuse_layer(512 + 12, 512)
#         self.fuse4 = self._make_fuse_layer(1024 + 12, 1024)
#
#     def _make_fuse_layer(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             SEBlock(out_ch)
#         )
#
#     def forward(self, x):
#         x0 = x
#         x1 = self.layer0(x0)
#         m1 = self.tm1(x0)
#         m1_up = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
#         x1f = self.fuse1(torch.cat([x1, m1_up], dim=1))
#
#         x2 = self.layer1(x1f)
#         m2 = self.tm2(torch.cat([x2, m1_up], dim=1))
#         m2_up = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         x2f = self.fuse2(torch.cat([x2, m2_up], dim=1))
#
#         x3 = self.layer2(x2f)
#         m3 = self.tm3(torch.cat([x3, m2_up], dim=1))
#         m3_up = F.interpolate(m3, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         x3f = self.fuse3(torch.cat([x3, m3_up], dim=1))
#
#         x4 = self.layer3(x3f)
#         m4 = self.tm4(torch.cat([x4, m3_up], dim=1))
#         m4_up = F.interpolate(m4, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         x4f = self.fuse4(torch.cat([x4, m4_up], dim=1))
#
#         return x1f, x2f, x3f, x4f

# coding: utf-8
"""
ResUNet with Texture Martingale Cascade:
At each layer, perform:
- Standard convolution: x1 = conv(x0), x2 = conv(x1), x3 = conv(x2), x4 = conv(x3)
- Texture martingale: m1 = TM(x0), m2 = TM(concat(x1, m1)), m3 = TM(concat(x2, m2)), m4 = TM(concat(x3, m3))
- Final output: concat(x4, m4)
- Fuse xi with mi via concat + 1x1 conv + BN + ReLU + optional SE

GLCM-based martingale feature maps: contrast, entropy, energy, homogeneity
"""

# from __future__ import annotations
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50
#
# # ------------------------------
# # SE Block
# # ------------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
# # ------------------------------
# # Texture Martingale via GLCM Features
# # ------------------------------
# class TextureMartingaleModule(nn.Module):
#     def __init__(self, dilations=[1, 2, 3, 4], theta=1.0, include_features=None):
#         super().__init__()
#         self.dilations = dilations
#         self.theta = theta
#         self.include_features = include_features or ["contrast", "energy", "entropy", "homogeneity"]
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         pyramid_outputs = []
#         for d in self.dilations:
#             k = 3 + 2 * (d - 1)
#             pad = d
#             scale_outputs = []
#             for c in range(C):
#                 feat = x[:, c:c + 1, :, :]
#                 unfolded = F.unfold(feat, kernel_size=k, dilation=d, padding=pad)
#                 _, K, L = unfolded.shape
#                 # compute output spatial size for unfold
#                 H_out = int((H + 2 * pad - d * (k - 1) - 1) + 1)
#                 W_out = int((W + 2 * pad - d * (k - 1) - 1) + 1)
#                 patches = unfolded.view(B, K, H_out, W_out)
#                 feats = self.compute_glcm_features(patches)
#                 feature_maps = []
#                 for name, val in feats.items():
#                     if name in self.include_features:
#                         log_feat = torch.log(val + 1e-6)
#                         M = torch.exp(self.theta * log_feat - 0.5 * self.theta**2)
#                         feature_maps.append(M)
#                 scale_outputs.append(torch.stack(feature_maps, dim=1))
#             pyramid_scale = torch.cat(scale_outputs, dim=1)
#             pyramid_outputs.append(pyramid_scale)
#         return F.interpolate(pyramid_outputs[0], size=(H, W), mode="bilinear", align_corners=False)
#
#     def compute_glcm_features(self, patches):
#         mean = patches.mean(dim=1, keepdim=True)
#         std = patches.std(dim=1, keepdim=True) + 1e-6
#         normed = (patches - mean) / std
#         contrast = (normed ** 2).mean(dim=1)
#         energy = (patches ** 2).mean(dim=1)
#         entropy = -(patches * torch.log(patches + 1e-6)).mean(dim=1)
#         homogeneity = 1.0 / (1.0 + (patches - mean).abs()).mean(dim=1)
#         return {
#             "contrast": contrast,
#             "energy": energy,
#             "entropy": entropy,
#             "homogeneity": homogeneity,
#         }
#
# # ------------------------------
# # Cascaded Encoder using TM features + SE + BN + ReLU
# # ------------------------------
# class CascadedMartingaleEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = resnet50()
#         self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
#         self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#
#         self.tm1 = TextureMartingaleModule([1])
#         self.tm2 = TextureMartingaleModule([2])
#         self.tm3 = TextureMartingaleModule([3])
#         self.tm4 = TextureMartingaleModule([4])
#
#         self.fuse1 = self._make_fuse_layer(64 + 12, 64)
#         self.fuse2 = self._make_fuse_layer(256 + 12, 256)
#         self.fuse3 = self._make_fuse_layer(512 + 12, 512)
#         self.fuse4 = self._make_fuse_layer(1024 + 12, 1024)
#
#         # 只在TM输入拼接后降为12通道，fuse层永远主干+12
#         self.reduce_tm2 = nn.Conv2d(256 + 12, 12, 1)
#         self.reduce_tm3 = nn.Conv2d(512 + 12, 12, 1)
#         self.reduce_tm4 = nn.Conv2d(1024 + 12, 12, 1)
#
#     def _make_fuse_layer(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             SEBlock(out_ch)
#         )
#
#     def forward(self, x):
#         x0 = x
#         x1 = self.layer0(x0)
#         m1 = self.tm1(x0)  # [B, 12, H, W]
#         m1_up = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
#         x1f = self.fuse1(torch.cat([x1, m1_up], dim=1))  # [B, 76, H, W]
#
#         x2 = self.layer1(x1f)
#         m1_to_x2 = F.interpolate(m1, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         tm2_in = torch.cat([x2, m1_to_x2], dim=1)
#         tm2_in = self.reduce_tm2(tm2_in)  # [B, 12, H, W]
#         m2 = self.tm2(tm2_in)
#         m2_up = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         x2f = self.fuse2(torch.cat([x2, m2_up], dim=1))  # [B, 268, H, W]
#
#         x3 = self.layer2(x2f)
#         m2_to_x3 = F.interpolate(m2, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         tm3_in = torch.cat([x3, m2_to_x3], dim=1)
#         tm3_in = self.reduce_tm3(tm3_in)
#         m3 = self.tm3(tm3_in)
#         m3_up = F.interpolate(m3, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         x3f = self.fuse3(torch.cat([x3, m3_up], dim=1))  # [B, 524, H, W]
#
#         x4 = self.layer3(x3f)
#         m3_to_x4 = F.interpolate(m3, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         tm4_in = torch.cat([x4, m3_to_x4], dim=1)
#         tm4_in = self.reduce_tm4(tm4_in)
#         m4 = self.tm4(tm4_in)
#         m4_up = F.interpolate(m4, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         x4f = self.fuse4(torch.cat([x4, m4_up], dim=1))  # [B, 1036, H, W]
#
#         return x1f, x2f, x3f, x4f
#
# if __name__ == "__main__":
#     model = CascadedMartingaleEncoder()
#     x = torch.randn(1, 3, 256, 256)
#     y = model(x)
#     for i, out in enumerate(y):
#         print(f"x{i+1} shape:", out.shape)
#         torch.save(out, f"feature_output_x{i+1}.pt")
#     print("Feature maps saved as 'feature_output_x1.pt' to 'x4.pt'")

# from __future__ import annotations
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50
#
#
# # ------------------------------
# # SE Block
# # ------------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
#
# # ------------------------------
# # Texture Martingale via GLCM Features
# # ------------------------------
# class TextureMartingaleModule(nn.Module):
#     def __init__(self, dilations=[1], theta=1.0, include_features=None):
#         super().__init__()
#         self.dilations = dilations
#         self.theta = theta
#         self.include_features = include_features or ["contrast", "energy", "entropy", "homogeneity"]
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         pyramid_outputs = []
#         for d in self.dilations:
#             k = 3 + 2 * (d - 1)
#             pad = d
#             scale_outputs = []
#             for c in range(C):
#                 feat = x[:, c:c + 1, :, :]
#                 unfolded = F.unfold(feat, kernel_size=k, dilation=d, padding=pad)
#                 _, K, L = unfolded.shape
#                 # compute output spatial size for unfold
#                 H_out = int((H + 2 * pad - d * (k - 1) - 1) + 1)
#                 W_out = int((W + 2 * pad - d * (k - 1) - 1) + 1)
#                 patches = unfolded.view(B, K, H_out, W_out)
#                 feats = self.compute_glcm_features(patches)
#                 feature_maps = []
#                 for name, val in feats.items():
#                     if name in self.include_features:
#                         log_feat = torch.log(val + 1e-6)
#                         M = torch.exp(self.theta * log_feat - 0.5 * self.theta ** 2)
#                         feature_maps.append(M)
#                 scale_outputs.append(torch.stack(feature_maps, dim=1))
#             pyramid_scale = torch.cat(scale_outputs, dim=1)
#             pyramid_outputs.append(pyramid_scale)
#         return F.interpolate(pyramid_outputs[0], size=(H, W), mode="bilinear", align_corners=False)
#
#     def compute_glcm_features(self, patches):
#         mean = patches.mean(dim=1, keepdim=True)
#         std = patches.std(dim=1, keepdim=True) + 1e-6
#         normed = (patches - mean) / std
#         contrast = (normed ** 2).mean(dim=1)
#         energy = (patches ** 2).mean(dim=1)
#         entropy = -(patches * torch.log(patches + 1e-6)).mean(dim=1)
#         homogeneity = 1.0 / (1.0 + (patches - mean).abs()).mean(dim=1)
#         return {
#             "contrast": contrast,
#             "energy": energy,
#             "entropy": entropy,
#             "homogeneity": homogeneity,
#         }
#
#
# # ------------------------------
# # Cascaded Encoder using TM features + SE + BN + ReLU
# # ------------------------------
# class CascadedMartingaleEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = resnet50()
#         self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
#         self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#
#         self.tm1 = TextureMartingaleModule([1])
#         self.tm2 = TextureMartingaleModule([2])
#         self.tm3 = TextureMartingaleModule([3])
#         self.tm4 = TextureMartingaleModule([4])
#
#         # self.fuse1 = self._make_fuse_layer(64 + 12, 64)
#         # self.fuse2 = self._make_fuse_layer(256 + 12, 256)
#         # self.fuse3 = self._make_fuse_layer(512 + 12, 512)
#         # self.fuse4 = self._make_fuse_layer(1024 + 12, 1024)
#
#         # 降维层：拼接后通道→64+12, 256+12, 512+12, 1024+12，防止通道爆炸
#         self.reduce_tm2 = nn.Conv2d(256 + 12, 12, 1)
#         self.reduce_tm3 = nn.Conv2d(512 + 12, 12, 1)
#         self.reduce_tm4 = nn.Conv2d(1024 + 12, 12, 1)
#         self.reduce_fuse2 = nn.Conv2d(256 + 48, 256 + 12, 1)
#         self.reduce_fuse3 = nn.Conv2d(512 + 48, 512 + 12, 1)
#         self.reduce_fuse4 = nn.Conv2d(1024 + 48, 1024 + 12, 1)
#
#     def _make_fuse_layer(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             SEBlock(out_ch)
#         )
#
#     def forward(self, x):
#         x0 = x
#         x1 = self.layer0(x0)
#         m1 = self.tm1(x0)
#         m1_up = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
#         print('m1_up:', m1_up.shape)
#         x1f = self.fuse1(torch.cat([x1, m1_up], dim=1))
#
#         x2 = self.layer1(x1f)
#         m1_to_x2 = F.interpolate(m1, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         tm2_in = torch.cat([x2, m1_to_x2], dim=1)
#         print('tm2_in:', tm2_in.shape)
#         tm2_in = self.reduce_tm2(tm2_in)
#         print('tm2_in(reduced):', tm2_in.shape)
#         m2 = self.tm2(tm2_in)
#         m2_up = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         print('m2_up:', m2_up.shape)
#         x2f = self.fuse2(torch.cat([x2, m2_up], dim=1))
#         print('x2f:', x2f.shape)
#
#         x3 = self.layer2(x2f)
#         m2_to_x3 = F.interpolate(m2, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         tm3_in = torch.cat([x3, m2_to_x3], dim=1)
#         print('tm3_in:', tm3_in.shape)
#         tm3_in = self.reduce_tm3(tm3_in)
#         print('tm3_in(reduced):', tm3_in.shape)
#         m3 = self.tm3(tm3_in)
#         m3_up = F.interpolate(m3, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         print('m3_up:', m3_up.shape)
#         x3f = self.fuse3(torch.cat([x3, m3_up], dim=1))
#         print('x3f:', x3f.shape)
#
#         x4 = self.layer3(x3f)
#         m3_to_x4 = F.interpolate(m3, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         tm4_in = torch.cat([x4, m3_to_x4], dim=1)
#         print('tm4_in:', tm4_in.shape)
#         tm4_in = self.reduce_tm4(tm4_in)
#         print('tm4_in(reduced):', tm4_in.shape)
#         m4 = self.tm4(tm4_in)
#         m4_up = F.interpolate(m4, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         print('m4_up:', m4_up.shape)
#         x4f = self.fuse4(torch.cat([x4, m4_up], dim=1))
#         print('x4f:', x4f.shape)
#
#         return x1f, x2f, x3f, x4f
#
#
# if __name__ == "__main__":
#     model = CascadedMartingaleEncoder()
#     x = torch.randn(1, 3, 256, 256)
#     y = model(x)
#     for i, out in enumerate(y):
#         print(f"x{i + 1} shape:", out.shape)
#     #     torch.save(out, f"feature_output_x{i + 1}.pt")
#     # print("Feature maps saved as 'feature_output_x1.pt' to 'x4.pt'")


# coding: utf-8
# from __future__ import annotations
# """
# ResUNet with Texture Martingale Cascade:
# At each layer, perform:
# - Standard convolution: x1 = conv(x0), x2 = conv(x1), x3 = conv(x2), x4 = conv(x3)
# - Texture martingale: m1 = TM(x0), m2 = TM(concat(x1, m1)), m3 = TM(concat(x2, m2)), m4 = TM(concat(x3, m3))
# - Final output: concat(x4, m4)
# - Fuse xi with mi via concat + 1x1 conv + BN + ReLU + optional SE
#
# GLCM-based martingale feature maps: contrast, entropy, energy, homogeneity
# """
#
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50
#
# # ------------------------------
# # SE Block
# # ------------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y
#
# # ------------------------------
# # Texture Martingale via GLCM Features
# # ------------------------------
# class TextureMartingaleModule(nn.Module):
#     def __init__(self, dilations=[1], theta=1.0, include_features=None):  # 正确，只用[1]
#         super().__init__()
#         self.dilations = dilations
#         self.theta = theta
#         self.include_features = include_features or ["contrast", "energy", "entropy", "homogeneity"]
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         d = self.dilations[0]  # 只用单一dilation
#         k = 3 + 2 * (d - 1)
#         pad = d
#         scale_outputs = []
#         for c in range(C):
#             feat = x[:, c:c + 1, :, :]
#             unfolded = F.unfold(feat, kernel_size=k, dilation=d, padding=pad)
#             _, K, L = unfolded.shape
#             H_out = int((H + 2 * pad - d * (k - 1) - 1) + 1)
#             W_out = int((W + 2 * pad - d * (k - 1) - 1) + 1)
#             patches = unfolded.view(B, K, H_out, W_out)
#             feats = self.compute_glcm_features(patches)
#             feature_maps = []
#             for name, val in feats.items():
#                 if name in self.include_features:
#                     log_feat = torch.log(val + 1e-6)
#                     M = torch.exp(self.theta * log_feat - 0.5 * self.theta**2)
#                     feature_maps.append(M)
#             scale_outputs.append(torch.stack(feature_maps, dim=1))
#         pyramid_scale = torch.cat(scale_outputs, dim=1)
#         return F.interpolate(pyramid_scale, size=(H, W), mode="bilinear", align_corners=False)
#
#
#
#     def compute_glcm_features(self, patches):
#         mean = patches.mean(dim=1, keepdim=True)
#         std = patches.std(dim=1, keepdim=True) + 1e-6
#         normed = (patches - mean) / std
#         contrast = (normed ** 2).mean(dim=1)
#         energy = (patches ** 2).mean(dim=1)
#         entropy = -(patches * torch.log(patches + 1e-6)).mean(dim=1)
#         homogeneity = 1.0 / (1.0 + (patches - mean).abs()).mean(dim=1)
#         return {
#             "contrast": contrast,
#             "energy": energy,
#             "entropy": entropy,
#             "homogeneity": homogeneity,
#         }
#
# # ------------------------------
# # Cascaded Encoder using TM features + SE + BN + ReLU
# # ------------------------------
# class CascadedMartingaleEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = resnet50()
#         self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
#         self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
#         self.layer2 = backbone.layer2
#         self.layer3 = backbone.layer3
#
#         self.tm1 = TextureMartingaleModule([1])
#         self.tm2 = TextureMartingaleModule([2])
#         self.tm3 = TextureMartingaleModule([3])
#         self.tm4 = TextureMartingaleModule([4])
#
#         self.fuse1 = self._make_fuse_layer(64 + 12, 64)
#         self.fuse2 = self._make_fuse_layer(256 + 12, 256)
#         self.fuse3 = self._make_fuse_layer(512 + 12, 512)
#         self.fuse4 = self._make_fuse_layer(1024 + 12, 1024)
#
#         self.reduce_tm2 = nn.Conv2d(256 + 12, 12, 1)
#         self.reduce_tm3 = nn.Conv2d(512 + 12, 12, 1)
#         self.reduce_tm4 = nn.Conv2d(1024 + 12, 12, 1)
#
#     def _make_fuse_layer(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             SEBlock(out_ch)
#         )
#
#     def forward(self, x):
#         x0 = x
#         x1 = self.layer0(x0)
#         m1 = self.tm1(x0)
#         m1_up = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
#         print('m1_up:', m1_up.shape)
#         x1f = self.fuse1(torch.cat([x1, m1_up], dim=1))
#
#         x2 = self.layer1(x1f)
#         m1_to_x2 = F.interpolate(m1, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         tm2_in = torch.cat([x2, m1_to_x2], dim=1)
#         print('tm2_in:', tm2_in.shape)
#         tm2_in = self.reduce_tm2(tm2_in)
#         print('tm2_in(reduced):', tm2_in.shape)
#         m2 = self.tm2(tm2_in)
#         m2_up = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
#         print('m2_up:', m2_up.shape)
#         x2f = self.fuse2(torch.cat([x2, m2_up], dim=1))
#         print('x2f:', x2f.shape)
#
#         x3 = self.layer2(x2f)
#         m2_to_x3 = F.interpolate(m2, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         tm3_in = torch.cat([x3, m2_to_x3], dim=1)
#         print('tm3_in:', tm3_in.shape)
#         tm3_in = self.reduce_tm3(tm3_in)
#         print('tm3_in(reduced):', tm3_in.shape)
#         m3 = self.tm3(tm3_in)
#         m3_up = F.interpolate(m3, size=x3.shape[2:], mode="bilinear", align_corners=False)
#         print('m3_up:', m3_up.shape)
#         x3f = self.fuse3(torch.cat([x3, m3_up], dim=1))
#         print('x3f:', x3f.shape)
#
#         x4 = self.layer3(x3f)
#         m3_to_x4 = F.interpolate(m3, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         tm4_in = torch.cat([x4, m3_to_x4], dim=1)
#         print('tm4_in:', tm4_in.shape)
#         tm4_in = self.reduce_tm4(tm4_in)
#         print('tm4_in(reduced):', tm4_in.shape)
#         m4 = self.tm4(tm4_in)
#         m4_up = F.interpolate(m4, size=x4.shape[2:], mode="bilinear", align_corners=False)
#         print('m4_up:', m4_up.shape)
#         x4f = self.fuse4(torch.cat([x4, m4_up], dim=1))
#         print('x4f:', x4f.shape)
#
#         return x1f, x2f, x3f, x4f
#
# if __name__ == "__main__":
#     model = CascadedMartingaleEncoder()
#     x = torch.randn(1, 3, 256, 256)
#     y = model(x)
#     for i, out in enumerate(y):
#         print(f"x{i+1} shape:", out.shape)
#         torch.save(out, f"feature_output_x{i+1}.pt")
#     print("Feature maps saved as 'feature_output_x1.pt' to 'x4.pt'")


# coding: utf-8


from __future__ import annotations
"""
ResUNet with Texture Martingale Cascade:
At each layer, perform:
- Standard convolution: x1 = conv(x0), x2 = conv(x1), x3 = conv(x2), x4 = conv(x3)
- Texture martingale: m1 = TM(x0), m2 = TM(concat(x1, m1)), m3 = TM(concat(x2, m2)), m4 = TM(concat(x3, m3))
- Final output: concat(x4, m4)
- Fuse xi with mi via concat + 1x1 conv + BN + ReLU + optional SE

GLCM-based martingale feature maps: contrast, entropy, energy, homogeneity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# ------------------------------
# SE Block
# ------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ------------------------------
# Texture Martingale via GLCM Features
# ------------------------------
class TextureMartingaleModule(nn.Module):
    def __init__(self, dilations=1, theta=1.0, include_features=None):  # 只用[1]
        super().__init__()
        self.dilations = dilations
        self.theta = theta
        self.include_features = include_features or ["contrast", "energy", "entropy", "homogeneity"]

    def forward(self, x):
        print(f"[TM DEBUG] self.dilations = {self.dilations}")
        B, C, H, W = x.shape
        d = self.dilations[0]
        k = 3 + 2 * (d - 1)
        pad = d
        scale_outputs = []
        for c in range(C):
            feat = x[:, c:c + 1, :, :]
            unfolded = F.unfold(feat, kernel_size=k, dilation=d, padding=pad)
            _, K, L = unfolded.shape
            H_out = int((H + 2 * pad - d * (k - 1) - 1) + 1)
            W_out = int((W + 2 * pad - d * (k - 1) - 1) + 1)
            patches = unfolded.view(B, K, H_out, W_out)
            feats = self.compute_glcm_features(patches)
            feature_maps = []
            for name, val in feats.items():
                if name in self.include_features:
                    log_feat = torch.log(val + 1e-6)
                    M = torch.exp(self.theta * log_feat - 0.5 * self.theta ** 2)
                    feature_maps.append(M)
            scale_outputs.append(torch.stack(feature_maps, dim=1))
        pyramid_scale = torch.cat(scale_outputs, dim=1)
        return F.interpolate(pyramid_scale, size=(H, W), mode="bilinear", align_corners=False)

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

# ------------------------------
# Cascaded Encoder using TM features + SE + BN + ReLU
# ------------------------------
class CascadedMartingaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.tm1 = TextureMartingaleModule([1])
        self.tm2 = TextureMartingaleModule([2])
        self.tm3 = TextureMartingaleModule([3])
        self.tm4 = TextureMartingaleModule([4])

        self.fuse1 = self._make_fuse_layer(64 + 12, 64)
        self.fuse2 = self._make_fuse_layer(256 + 12, 256)
        self.fuse3 = self._make_fuse_layer(512 + 12, 512)
        self.fuse4 = self._make_fuse_layer(1024 + 12, 1024)

        self.reduce_tm2 = nn.Conv2d(256 + 12, 12, 1)
        self.reduce_tm3 = nn.Conv2d(512 + 12, 12, 1)
        self.reduce_tm4 = nn.Conv2d(1024 + 12, 12, 1)

    def _make_fuse_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch)
        )

    def forward(self, x):
        x0 = x
        x1 = self.layer0(x0)
        m1 = self.tm1(x0)
        m1_up = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        print('m1_up:', m1_up.shape)
        x1f = self.fuse1(torch.cat([x1, m1_up], dim=1))

        x2 = self.layer1(x1f)
        m1_to_x2 = F.interpolate(m1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        tm2_in = torch.cat([x2, m1_to_x2], dim=1)
        print('tm2_in:', tm2_in.shape)
        tm2_in = self.reduce_tm2(tm2_in)
        print('tm2_in(reduced):', tm2_in.shape)
        m2 = self.tm2(tm2_in)
        m2_up = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        print('m2_up:', m2_up.shape)
        x2f = self.fuse2(torch.cat([x2, m2_up], dim=1))
        print('x2f:', x2f.shape)

        x3 = self.layer2(x2f)
        m2_to_x3 = F.interpolate(m2, size=x3.shape[2:], mode="bilinear", align_corners=False)
        tm3_in = torch.cat([x3, m2_to_x3], dim=1)
        print('tm3_in:', tm3_in.shape)
        tm3_in = self.reduce_tm3(tm3_in)
        print('tm3_in(reduced):', tm3_in.shape)
        m3 = self.tm3(tm3_in)
        m3_up = F.interpolate(m3, size=x3.shape[2:], mode="bilinear", align_corners=False)
        print('m3_up:', m3_up.shape)
        x3f = self.fuse3(torch.cat([x3, m3_up], dim=1))
        print('x3f:', x3f.shape)

        x4 = self.layer3(x3f)
        m3_to_x4 = F.interpolate(m3, size=x4.shape[2:], mode="bilinear", align_corners=False)
        tm4_in = torch.cat([x4, m3_to_x4], dim=1)
        print('tm4_in:', tm4_in.shape)
        tm4_in = self.reduce_tm4(tm4_in)
        print('tm4_in(reduced):', tm4_in.shape)
        m4 = self.tm4(tm4_in)
        m4_up = F.interpolate(m4, size=x4.shape[2:], mode="bilinear", align_corners=False)
        print('m4_up:', m4_up.shape)
        x4f = self.fuse4(torch.cat([x4, m4_up], dim=1))
        print('x4f:', x4f.shape)

        return x1f, x2f, x3f, x4f

if __name__ == "__main__":
    model = CascadedMartingaleEncoder()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    for i, out in enumerate(y):
        print(f"x{i+1} shape:", out.shape)
        torch.save(out, f"feature_output_x{i+1}.pt")
    print("Feature maps saved as 'feature_output_x1.pt' to 'x4.pt'")


