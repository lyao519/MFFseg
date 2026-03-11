# File: model/martingale_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MartingaleFilter(nn.Module):
    def __init__(self, channels):
        super(MartingaleFilter, self).__init__()
        self.theta = nn.Parameter(torch.ones(1))  # 可学习参数 θ
        self.eps_model = nn.Conv2d(channels, 1, kernel_size=1)  # 估计扰动 ε(i,j)

    def forward(self, x):
        log_x = torch.log(x + 1e-6)
        eps = self.eps_model(x)
        log_xt = log_x + eps
        m = torch.exp(self.theta * log_xt - 0.5 * self.theta ** 2)
        return m


class MartingaleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MartingaleEncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.tmb = MartingaleFilter(out_channels)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        t = self.tmb(x)
        x = torch.cat([x, t], dim=1)
        x = self.fusion(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MartingaleUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_c=64):
        super(MartingaleUNet, self).__init__()

        self.enc1 = MartingaleEncoderBlock(in_channels, base_c)
        self.enc2 = MartingaleEncoderBlock(base_c, base_c * 2)
        self.enc3 = MartingaleEncoderBlock(base_c * 2, base_c * 4)
        self.enc4 = MartingaleEncoderBlock(base_c * 4, base_c * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_c * 8, base_c * 16, 3, padding=1),
            nn.BatchNorm2d(base_c * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c * 16, base_c * 16, 3, padding=1),
            nn.BatchNorm2d(base_c * 16),
            nn.ReLU(inplace=True)
        )

        self.dec4 = DecoderBlock(base_c * 16 + base_c * 8, base_c * 8)
        self.dec3 = DecoderBlock(base_c * 8 + base_c * 4, base_c * 4)
        self.dec2 = DecoderBlock(base_c * 4 + base_c * 2, base_c * 2)
        self.dec1 = DecoderBlock(base_c * 2 + base_c, base_c)

        self.final = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        s1, x1 = self.enc1(x)
        s2, x2 = self.enc2(x1)
        s3, x3 = self.enc3(x2)
        s4, x4 = self.enc4(x3)

        b = self.bottleneck(x4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.final(d1)
