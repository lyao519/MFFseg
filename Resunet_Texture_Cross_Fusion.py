# # coding: utf-8
# """
# ResUNet with Texture‑Martingale branch, multi‑scale feature fusion (MSF) and
# cross‑attention (CA) for medical image segmentation.
#
# Key additions vs. user‑provided baseline
# ---------------------------------------
# 1. **MultiScaleFusion** – upsamples encoder features to the highest spatial
#    resolution, concatenates along channels then compresses with 1×1 + 3×3 conv.
# 2. **CrossAttentionBlock** – lightweight MHSA operating on flattened spatial
#    tokens; used to inject texture cues into deep encoder stage (x4).
# 3. **Refactored ResUNetWithTexture** – now called *ResUNetTextureMSCA*; plugs
#    MSF and CA while keeping the original ResNet50 encoder & U‑Net decoder.
#
# The code is kept self‑contained; copy into a file and `import` normally.
# """
#
# from __future__ import annotations
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50
#
# # -------------------------------------------------------------
# # Existing modules (unchanged)
# # -------------------------------------------------------------
#
# class TextureMartingaleModule(nn.Module):
#     """Extract multi‑dilation GLCM‑style texture features (log‑martingale).
#     Returns a list of feature tensors – one per dilation.
#     Each element shape: (B, in_channels*|features|, H, W)
#     """
#
#     def __init__(
#         self,
#         dilations: list[int] = [1, 2, 3, 4],
#         theta: float = 1.0,
#         sigma2: float = 1.0,
#         include_features: list[str] | None = None,
#     ) -> None:
#         super().__init__()
#         if include_features is None:
#             include_features = ["contrast", "energy", "entropy", "homogeneity"]
#         self.dilations = dilations
#         self.theta = theta
#         self.include_features = include_features
#
#     # ‑‑ helper: compute simple statistics instead of full GLCM 4‑dir matrix
#     def compute_glcm_features(self, patches: torch.Tensor):
#         mean = patches.mean(dim=1, keepdim=True)
#         std = patches.std(dim=1, keepdim=True) + 1e‑6
#         normed = (patches - mean) / std
#         contrast = (normed ** 2).mean(dim=1)
#         energy = (patches ** 2).mean(dim=1)
#         entropy = -(patches * torch.log(patches + 1e‑6)).mean(dim=1)
#         homogeneity = 1.0 / (1.0 + (patches - mean).abs()).mean(dim=1)
#         return {
#             "contrast": contrast,
#             "energy": energy,
#             "entropy": entropy,
#             "homogeneity": homogeneity,
#         }
#
#     def forward(self, x: torch.Tensor):
#         B, C, H, W = x.shape
#         pyramid = []
#         for dilation in self.dilations:
#             k = 3 + 2 * (dilation - 1)
#             pad = dilation
#             per_channel_out = []
#             for c in range(C):
#                 feat = x[:, c : c + 1]
#                 unfolded = F.unfold(feat, kernel_size=k, dilation=dilation, padding=pad)
#                 K = unfolded.shape[1]
#                 patches = unfolded.view(B, K, H, W)
#                 feats = self.compute_glcm_features(patches)
#                 maps = []
#                 for name, value in feats.items():
#                     if name in self.include_features:
#                         log_feat = torch.log(value + 1e‑6)
#                         M = torch.exp(self.theta * log_feat - 0.5 * self.theta ** 2)
#                         maps.append(M)
#                 per_channel_out.append(torch.stack(maps, dim=1))  # (B, |F|, H, W)
#             pyramid.append(torch.cat(per_channel_out, dim=1))  # (B, C*|F|, H, W)
#         return pyramid  # list[Tensor]
#
# class SEFusion(nn.Module):
#     """Squeeze‑and‑Excitation used after fusion."""
#
#     def __init__(self, in_channels: int, reduction: int = 16) -> None:
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels // reduction)
#         self.fc2 = nn.Linear(in_channels // reduction, in_channels)
#         self.act = nn.ReLU(inplace=True)
#         self.gate = nn.Sigmoid()
#
#     def forward(self, x: torch.Tensor):
#         b, c, _, _ = x.shape
#         w = self.pool(x).view(b, c)
#         w = self.gate(self.fc2(self.act(self.fc1(w)))).view(b, c, 1, 1)
#         return x * w
#
# class DynamicScaleSelector(nn.Module):
#     """Softmax attention over K parallel scale branches."""
#
#     def __init__(self, num_scales: int, feature_dim: int) -> None:
#         super().__init__()
#         self.attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(feature_dim, feature_dim // 4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feature_dim // 4, num_scales, 1),
#             nn.Softmax(dim=1),
#         )
#
#     def forward(self, inputs: list[torch.Tensor]):
#         # assume all inputs have same C,H,W
#         weights = self.attn(inputs[0])  # (B, K, 1, 1)
#         out = 0.0
#         for i, feat in enumerate(inputs):
#             out = out + feat * weights[:, i : i + 1]
#         return out
#
# # -------------------------------------------------------------
# # 🆕  New modules: Multi‑Scale Fusion & Cross Attention
# # -------------------------------------------------------------
#
# class MultiScaleFusion(nn.Module):
#     """Fuse list of encoder feature maps to a common high‑resolution map.
#
#     Steps: 1) align channels with 1×1 conv 2) upsample to resolution of the
#     first feature (x1) 3) concatenate and fuse via 3×3 conv.
#     """
#
#     def __init__(self, in_channels: list[int], out_channels: int):
#         super().__init__()
#         self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
#         self.fuse = nn.Sequential(
#             nn.Conv2d(out_channels * len(in_channels), out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, feats: list[torch.Tensor]):
#         # feats = [x1, x2, x3, x4]
#         base_h, base_w = feats[0].shape[-2:]
#         upsampled = []
#         for f, conv in zip(feats, self.lateral_convs):
#             y = conv(f)
#             if y.shape[-2:] != (base_h, base_w):
#                 y = F.interpolate(y, size=(base_h, base_w), mode="bilinear", align_corners=False)
#             upsampled.append(y)
#         fused = self.fuse(torch.cat(upsampled, dim=1))
#         return fused  # (B, out_channels, H, W)
#
# class CrossAttentionBlock(nn.Module):
#     """Lightweight MHSA operating on flattened spatial tokens."""
#
#     def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
#         super().__init__()
#         self.q_proj = nn.Conv2d(dim, dim, 1)
#         self.k_proj = nn.Conv2d(dim, dim, 1)
#         self.v_proj = nn.Conv2d(dim, dim, 1)
#         self.mhsa = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
#         self.proj = nn.Conv2d(dim, dim, 1)
#         self.norm = nn.BatchNorm2d(dim)
#
#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
#         B, C, H, W = q.shape
#         q_tok = self.q_proj(q).flatten(2).transpose(1, 2)  # (B, HW, C)
#         k_tok = self.k_proj(k).flatten(2).transpose(1, 2)
#         v_tok = self.v_proj(v).flatten(2).transpose(1, 2)
#         attn_out, _ = self.mhsa(q_tok, k_tok, v_tok)
#         attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
#         out = self.proj(attn_out) + q  # residual
#         return self.norm(out)
#
# # -------------------------------------------------------------
# # Backbone + Decoder (unchanged)
# # -------------------------------------------------------------
#
# class ResNet50Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         m = resnet50(weights="IMAGENET1K_V1")
#         self.initial = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # 1/4
#         self.enc1 = m.layer1  # 256, 1/4
#         self.enc2 = m.layer2  # 512, 1/8
#         self.enc3 = m.layer3  # 1024, 1/16
#         self.enc4 = m.layer4  # 2048, 1/32
#
#     def forward(self, x):
#         x0 = self.initial(x)
#         x1 = self.enc1(x0)
#         x2 = self.enc2(x1)
#         x3 = self.enc3(x2)
#         x4 = self.enc4(x3)
#         return x1, x2, x3, x4
#
# class UNetDecoder(nn.Module):
#     def __init__(self, base_channels=64):
#         super().__init__()
#         self.up3 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(2048, 1024, 3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(inplace=True),
#         )
#         self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
#         )
#         self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
#         )
#         self.final = nn.Conv2d(256, 1, 1)
#
#     def forward(self, x1, x2, x3, x4):
#         d3 = self.conv3(torch.cat([self.up3(x4), x3], dim=1))
#         d2 = self.conv2(torch.cat([self.up2(d3), x2], dim=1))
#         d1 = self.conv1(torch.cat([self.up1(d2), x1], dim=1))
#         return self.final(d1)
#
# # -------------------------------------------------------------
# # 🆕  Top‑level network with MSF + CA
# # -------------------------------------------------------------
#
# class ResUNetTextureMSCA(nn.Module):
#     """ResNet‑UNet with Texture branch, Multi‑Scale Fusion and Cross‑Attention."""
#
#     def __init__(self, in_channels: int = 3, base_channels: int = 64):
#         super().__init__()
#         # --- texture encoder branch (returns fused texture map at full res)
#         self.texture = UNetEncoderWithTexture(in_channels, base_channels)
#
#         # --- ResNet encoder
#         self.encoder = ResNet50Encoder()
#
#         # --- MSF will reduce each encoder map to base_channels then fuse
#         self.msf = MultiScaleFusion(
#             in_channels=[256, 512, 1024, 2048], out_channels=base_channels
#         )
#
#         # --- Cross‑attention to inject texture cues
#         self.cross_attn = CrossAttentionBlock(dim=base_channels, heads=4)
#
#         # --- Decoder (expects original ResNet dims)
#         self.decoder = UNetDecoder()
#
#         # Project fused CA feature back to 2048 to replace x4
#         self.to_x4 = nn.Conv2d(base_channels, 2048, 1)
#
#     def forward(self, x):
#         # texture map (B, base_channels, H/4, W/4)  after selector & SE
#         tex = self.texture(x)
#
#         # ResNet features
#         x1, x2, x3, x4 = self.encoder(x)
#
#         # Multi‑scale fusion of conv features to same scale as tex (1/4)
#         msf_conv = self.msf([x1, x2, x3, x4])  # (B, base_channels, H/4, W/4)
#
#         # Cross‑attention between conv & texture representations
#         ca_feat = self.cross_attn(msf_conv, tex, tex)
#
#         # Inject CA output into deepest encoder feature (x4)
#         ca_x4 = x4 + F.interpolate(self.to_x4(ca_feat), size=x4.shape[-2:], mode="bilinear", align_corners=False)
#
#         # Decode
#         out = self.decoder(x1, x2, x3, ca_x4)
#         return out
#
# # -------------------------------------------------------------
# # Auxiliary: Texture‑aware UNet‑like encoder (inherited from user code)
# # -------------------------------------------------------------
#
# class UNetEncoderWithTexture(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 3,
#         base_channels: int = 64,
#         dilations: list[int] | None = None,
#         glcm_features: int = 4,
#     ) -> None:
#         super().__init__()
#         if dilations is None:
#             dilations = [1, 2, 3, 4]
#         self.texture = TextureMartingaleModule(dilations=dilations)
#         tex_ch = in_channels * glcm_features
#
#         self.conv_branches = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(in_channels + tex_ch, base_channels, 3, padding=1),
#                     nn.BatchNorm2d(base_channels),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(base_channels, base_channels, 3, padding=1),
#                     nn.BatchNorm2d(base_channels),
#                     nn.ReLU(inplace=True),
#                 )
#                 for _ in dilations
#             ]
#         )
#         self.selector = DynamicScaleSelector(len(dilations), base_channels)
#         self.merge = nn.Sequential(nn.Conv2d(base_channels, base_channels, 1), SEFusion(base_channels))
#
#     def forward(self, x: torch.Tensor):
#         tex_pyramid = self.texture(x)
#         feats = []
#         for tex, conv in zip(tex_pyramid, self.conv_branches):
#             feats.append(conv(torch.cat([x, tex], dim=1)))
#         out = self.merge(self.selector(feats))
#         return out
#
# # -------------------------------------------------------------
# # Quick sanity test (if run as script)
# # -------------------------------------------------------------
#
# if __name__ == "__main__":
#     model = ResUNetTextureMSCA(in_channels=3)
#     inp = torch.randn(2, 3, 256, 256)
#     with torch.no_grad():
#         y = model(inp)
#     print("Output shape:", y.shape)  # (2, 1, 256, 256)
