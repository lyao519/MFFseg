import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# =========================================================
# Utility
# =========================================================
def _safe_pretrained_resnet18():
    try:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        return resnet18(weights=None)


# =========================================================
# Texture Martingale Module
# Keep only shallow stages to avoid over-perturbing deep semantics
# =========================================================
class TextureMartingaleModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dilation: int = 1,
        theta: float = 1.0,
        include_features: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dilation = dilation
        self.theta = theta
        self.include_features = include_features or [
            "contrast", "energy", "entropy", "homogeneity"
        ]
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        d = self.dilation
        k = 3 + 2 * (d - 1)
        pad = d

        glcm_feats = []
        for ch in range(c):
            channel = x[:, ch:ch + 1, :, :]
            unfolded = F.unfold(channel, kernel_size=k, dilation=d, padding=pad)
            kk = unfolded.shape[1]
            patches = unfolded.view(b, kk, h, w)

            feats = self.compute_glcm_features(patches)

            channel_feat = []
            for feat_name, feat_value in feats.items():
                if feat_name in self.include_features:
                    log_feat = torch.clamp(feat_value, min=1e-5).log()
                    martingale = torch.clamp(
                        (self.theta * log_feat - 0.5 * self.theta ** 2).exp(),
                        min=1e-4,
                        max=1e4,
                    )
                    martingale = torch.where(
                        torch.isfinite(martingale),
                        martingale,
                        torch.full_like(martingale, 1.0),
                    )
                    channel_feat.append(martingale)

            glcm_feats.append(torch.stack(channel_feat, dim=1))

        out = torch.cat(glcm_feats, dim=1)
        return out

    def compute_glcm_features(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = patches.mean(dim=1, keepdim=True)
        std = patches.std(dim=1, keepdim=True).clamp(min=1e-3)
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


# =========================================================
# CNN Encoder with shallow TMM only
# =========================================================
class CascadedMartingaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = _safe_pretrained_resnet18()

        # ResNet18 stages
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)   # [B,64,H/2,W/2]
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)              # [B,64,H/4,W/4]
        self.layer2 = backbone.layer2                                               # [B,128,H/8,W/8]
        self.layer3 = backbone.layer3                                               # [B,256,H/16,W/16]

        # Only keep TMM on shallow stages
        self.tm1 = TextureMartingaleModule(in_channels=3, dilation=1)
        self.tm2 = TextureMartingaleModule(in_channels=64, dilation=1)

        self.reduce_tm1 = nn.Conv2d(3 * 4, 16, kernel_size=1)
        self.reduce_tm2 = nn.Conv2d(64 * 4, 32, kernel_size=1)

        # Channel alignment with transformer branch
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + 16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 + 32, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # stage1
        x1 = self.layer0(x)  # [B,64,H/2,W/2]
        m1 = self.tm1(x)
        m1 = self.reduce_tm1(m1)
        m1 = F.interpolate(m1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x1f = self.fuse1(torch.cat([x1, m1], dim=1))  # [B,32,H/2,W/2]

        # stage2
        x2 = self.layer1(x1)  # [B,64,H/4,W/4]
        m2 = self.tm2(x1)
        m2 = self.reduce_tm2(m2)
        m2 = F.interpolate(m2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x2f = self.fuse2(torch.cat([x2, m2], dim=1))  # [B,128,H/4,W/4]

        # stage3
        x3 = self.layer2(x2)  # [B,128,H/8,W/8]
        x3f = self.proj3(x3)  # [B,256,H/8,W/8]

        # stage4
        x4 = self.layer3(x3)  # [B,256,H/16,W/16]
        x4f = self.proj4(x4)  # [B,512,H/16,W/16]

        return x1f, x2f, x3f, x4f


# =========================================================
# Lightweight Transformer Branch
# =========================================================
class LightTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B,N,C]

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        out = x_flat.transpose(1, 2).view(b, c, h, w)
        return out


class UNetTransformerBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        depths: Tuple[int, int, int] = (2, 2, 2),
        num_heads: Tuple[int, int, int] = (2, 4, 8),
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )  # [B,32,H/2,W/2]

        self.layer1 = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [B,32,H/2,W/2]

        self.down2 = nn.MaxPool2d(2)
        self.trans2 = nn.Sequential(*[
            LightTransformerBlock(32, num_heads[0]) for _ in range(depths[0])
        ])
        self.out_proj_2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.down3 = nn.MaxPool2d(2)
        self.trans3 = nn.Sequential(*[
            LightTransformerBlock(128, num_heads[1]) for _ in range(depths[1])
        ])
        self.out_proj_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.down4 = nn.MaxPool2d(2)
        self.trans4 = nn.Sequential(*[
            LightTransformerBlock(256, num_heads[2]) for _ in range(depths[2])
        ])
        self.out_proj_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t1 = self.stem(x)
        t1 = self.layer1(t1)       # [B,32,H/2,W/2]

        t2 = self.down2(t1)
        t2 = self.trans2(t2)
        t2 = self.out_proj_2(t2)   # [B,128,H/4,W/4]

        t3 = self.down3(t2)
        t3 = self.trans3(t3)
        t3 = self.out_proj_3(t3)   # [B,256,H/8,W/8]

        t4 = self.down4(t3)
        t4 = self.trans4(t4)
        t4 = self.out_proj_4(t4)   # [B,512,H/16,W/16]

        return t1, t2, t3, t4


# =========================================================
# Real cross-attention, with optional spatial downsampling
# to avoid OOM on high-resolution features
# =========================================================
class CrossAttention2D(nn.Module):
    def __init__(self, channels: int, heads: int = 4, downsample: int = 1):
        super().__init__()
        self.channels = channels
        self.downsample = downsample
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, query_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        q_in = query_feat
        kv_in = kv_feat

        if self.downsample > 1:
            q_in = F.avg_pool2d(q_in, kernel_size=self.downsample, stride=self.downsample)
            kv_in = F.avg_pool2d(kv_in, kernel_size=self.downsample, stride=self.downsample)

        b, c, h, w = q_in.shape

        q = q_in.flatten(2).transpose(1, 2)   # [B,N,C]
        kv = kv_in.flatten(2).transpose(1, 2) # [B,N,C]

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn)
        out = out + q
        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.out_proj(out)

        if self.downsample > 1:
            out = F.interpolate(out, size=query_feat.shape[2:], mode="bilinear", align_corners=False)

        return out


class ChannelGatedFusion(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        z = torch.cat([feat_a, feat_b], dim=1)
        w = self.gate(self.pool(z))
        z = z * w
        out = self.fuse(z)
        return out


class BidirectionalCrossAttentionFusion(nn.Module):
    def __init__(self, channels: int, heads: int = 4, downsample: int = 1):
        super().__init__()
        self.cnn_to_trans = CrossAttention2D(channels, heads=heads, downsample=downsample)
        self.trans_to_cnn = CrossAttention2D(channels, heads=heads, downsample=downsample)
        self.gated_fusion = ChannelGatedFusion(channels)

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cnn_upd = self.cnn_to_trans(cnn_feat, trans_feat) + cnn_feat
        trans_upd = self.trans_to_cnn(trans_feat, cnn_feat) + trans_feat
        fused = self.gated_fusion(cnn_upd, trans_upd)
        return cnn_upd, trans_upd, fused


# =========================================================
# CAB: true fusion, not only gate cnn
# =========================================================
class CABModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(8, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels * 2, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        z = torch.cat([cnn_feat, trans_feat], dim=1)
        g = self.avg_pool(z)
        g = self.fc2(self.relu(self.fc1(g)))
        g = self.sigmoid(g)

        g_cnn, g_trans = torch.chunk(g, 2, dim=1)
        cnn_refined = cnn_feat * g_cnn
        trans_refined = trans_feat * g_trans

        out = self.fuse(torch.cat([cnn_refined, trans_refined], dim=1))
        return out


# =========================================================
# Decoder with deep supervision
# =========================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DeepSupervisionDecoder(nn.Module):
    def __init__(self, channels: Tuple[int, int, int, int] = (32, 128, 256, 512), out_channels: int = 1):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.center = nn.Sequential(
            ConvBNReLU(c4, 512),
            ConvBNReLU(512, 512),
        )

        self.dec3 = DecoderBlock(512, c3, 256)
        self.dec2 = DecoderBlock(256, c2, 128)
        self.dec1 = DecoderBlock(128, c1, 64)

        self.final_conv = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        self.aux3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.aux2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.aux1 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.center(f4)           # H/16
        d3 = self.dec3(x, f3)         # H/8
        d2 = self.dec2(d3, f2)        # H/4
        d1 = self.dec1(d2, f1)        # H/2

        main = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        main = self.final_conv(main)  # H

        aux3 = self.aux3(d3)
        aux2 = self.aux2(d2)
        aux1 = self.aux1(d1)

        return {
            "main": main,
            "aux3": aux3,
            "aux2": aux2,
            "aux1": aux1,
        }


# =========================================================
# Full model
# Key anti-OOM design:
# - fuse1: lightweight CAB only
# - fuse2: bidirectional cross-attention with downsample=2
# - fuse3: bidirectional cross-attention at lower resolution
# =========================================================
class MedSegNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CascadedMartingaleEncoder()
        self.trans_branch = UNetTransformerBranch()

        # High-resolution stage: no global attention
        self.fuse1 = CABModule(channels=32)

        # Medium stage: attention with spatial downsampling
        self.fuse2 = BidirectionalCrossAttentionFusion(
            channels=128, heads=4, downsample=2
        )

        # Lower stage: normal cross-attention
        self.fuse3 = BidirectionalCrossAttentionFusion(
            channels=256, heads=4, downsample=1
        )

        # Deepest stage: light fusion
        self.deep_fuse = CABModule(channels=512)

        self.decoder = DeepSupervisionDecoder(
            channels=(32, 128, 256, 512),
            out_channels=1,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1f, x2f, x3f, x4f = self.encoder(x)
        t1, t2, t3, t4 = self.trans_branch(x)

        f1 = self.fuse1(x1f, t1)
        _, _, f2 = self.fuse2(x2f, t2)
        _, _, f3 = self.fuse3(x3f, t3)
        f4 = self.deep_fuse(x4f, t4)

        outputs = self.decoder(f1, f2, f3, f4)
        return outputs


# =========================================================
# Losses
# =========================================================
def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, per_image: bool = True) -> torch.Tensor:
    if per_image:
        losses = []
        for logit, label in zip(logits, labels):
            losses.append(lovasz_hinge_flat(logit.view(-1), label.view(-1)))
        return torch.stack(losses).mean()
    return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


class SegmentationLoss(nn.Module):
    """
    total = main + 0.4*aux3 + 0.2*aux2 + 0.1*aux1
    each branch loss = 0.4*BCEWithLogits + 0.4*Dice + 0.2*Lovasz
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def single_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

        bce = self.bce(logits, target)
        dice = soft_dice_loss_from_logits(logits, target)
        lovasz = lovasz_hinge(logits.squeeze(1), target.squeeze(1))

        return 0.4 * bce + 0.4 * dice + 0.2 * lovasz

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        loss_main = self.single_loss(outputs["main"], target)
        loss_aux3 = self.single_loss(outputs["aux3"], target)
        loss_aux2 = self.single_loss(outputs["aux2"], target)
        loss_aux1 = self.single_loss(outputs["aux1"], target)

        total = loss_main + 0.4 * loss_aux3 + 0.2 * loss_aux2 + 0.1 * loss_aux1
        return total


# =========================================================
# Metrics
# =========================================================
@torch.no_grad()
def binary_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    target = target.float()

    dims = (1, 2, 3)
    tp = (pred * target).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    fn = ((1 - pred) * target).sum(dims)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    jaccard = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)

    return {
        "dice": dice.mean().item(),
        "jaccard": jaccard.mean().item(),
        "precision": precision.mean().item(),
    }


# =========================================================
# TTA inference
# =========================================================
@torch.no_grad()
def tta_predict_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    logits_list = []

    out = model(x)["main"]
    logits_list.append(out)

    x_h = torch.flip(x, dims=[3])
    out_h = model(x_h)["main"]
    out_h = torch.flip(out_h, dims=[3])
    logits_list.append(out_h)

    x_v = torch.flip(x, dims=[2])
    out_v = model(x_v)["main"]
    out_v = torch.flip(out_v, dims=[2])
    logits_list.append(out_v)

    x_hv = torch.flip(x, dims=[2, 3])
    out_hv = model(x_hv)["main"]
    out_hv = torch.flip(out_hv, dims=[2, 3])
    logits_list.append(out_hv)

    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return logits


# =========================================================
# Optional helper functions
# =========================================================
def training_step(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()

    use_amp = scaler is not None and images.is_cuda

    if use_amp:
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        metrics = binary_metrics_from_logits(outputs["main"], masks)

    return {
        "loss": float(loss.item()),
        "dice": metrics["dice"],
        "jaccard": metrics["jaccard"],
        "precision": metrics["precision"],
    }


@torch.no_grad()
def validation_step(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.Module,
    use_tta: bool = False,
) -> Dict[str, float]:
    model.eval()

    if use_tta:
        logits = tta_predict_logits(model, images)
        outputs = {"main": logits, "aux3": logits, "aux2": logits, "aux1": logits}
    else:
        outputs = model(images)

    loss = criterion(outputs, masks)
    metrics = binary_metrics_from_logits(outputs["main"], masks)

    return {
        "loss": float(loss.item()),
        "dice": metrics["dice"],
        "jaccard": metrics["jaccard"],
        "precision": metrics["precision"],
    }


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MedSegNetV2().to(device)
    criterion = SegmentationLoss()

    x = torch.randn(2, 3, 256, 256).to(device)
    y = (torch.rand(2, 1, 256, 256) > 0.5).float().to(device)

    outputs = model(x)
    print("main :", outputs["main"].shape)
    print("aux3 :", outputs["aux3"].shape)
    print("aux2 :", outputs["aux2"].shape)
    print("aux1 :", outputs["aux1"].shape)

    loss = criterion(outputs, y)
    print("loss :", float(loss.item()))

    metrics = binary_metrics_from_logits(outputs["main"], y)
    print("dice :", metrics["dice"])
    print("jac  :", metrics["jaccard"])
    print("pre  :", metrics["precision"])

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        info = training_step(model, x, y, criterion, optimizer, scaler=scaler)
        print("train info:", info)

    logits_tta = tta_predict_logits(model, x)
    print("tta logits:", logits_tta.shape)
