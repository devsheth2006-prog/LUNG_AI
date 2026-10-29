"""2-D and 3-D U-Nets (plain PyTorch, no external weights).

MONAI's `monai.networks.nets.UNet`/`BasicUNet` or nnU-Net can be swapped in through
`build_segmentation_model` — the training script only relies on the (N, C, ...) -> (N, K, ...)
contract. We ship our own implementation so the project runs without MONAI installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(dim: int, cin: int, cout: int, dropout: float = 0.0) -> nn.Sequential:
    Conv = nn.Conv2d if dim == 2 else nn.Conv3d
    Norm = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d  # robust to small batches
    layers = [Conv(cin, cout, 3, padding=1, bias=False), Norm(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
              Conv(cout, cout, 3, padding=1, bias=False), Norm(cout, affine=True), nn.LeakyReLU(0.01, inplace=True)]
    if dropout:
        layers.insert(3, nn.Dropout(dropout))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Classic U-Net with configurable dimensionality and depth."""

    def __init__(self, dim: int = 2, in_channels: int = 1, out_channels: int = 2,
                 base: int = 16, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        Pool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        Up = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        cin = in_channels
        for c in chs[:-1]:
            self.down.append(_conv_block(dim, cin, c))
            cin = c
        self.pool = Pool(2)
        self.bottleneck = _conv_block(dim, chs[-2], chs[-1], dropout)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c_hi, c_lo in zip(reversed(chs[1:]), reversed(chs[:-1])):
            self.up.append(Up(c_hi, c_lo, 2, stride=2))
            self.dec.append(_conv_block(dim, c_lo * 2, c_lo))
        self.head = Conv(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:  # odd sizes
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = dec(torch.cat([x, skip], 1))
        return self.head(x)


def build_segmentation_model(name: str, in_channels: int = 1, out_channels: int = 2, base: int = 16) -> nn.Module:
    name = name.lower()
    if name == "unet2d":
        return UNet(2, in_channels, out_channels, base)
    if name == "unet3d":
        return UNet(3, in_channels, out_channels, base, depth=3)
    if name == "monai_unet":
        from monai.networks.nets import BasicUNet  # optional dependency

        return BasicUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unknown segmentation model {name}")


class DiceCELoss(nn.Module):
    """Soft Dice + cross-entropy — standard for class-imbalanced segmentation (tiny nodules)."""

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.dw, self.cw, self.eps = dice_weight, ce_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = logits.softmax(1)[:, 1]
        tgt = (target == 1).float()
        dims = tuple(range(1, probs.ndim))
        inter = (probs * tgt).sum(dims)
        dice = (2 * inter + self.eps) / (probs.sum(dims) + tgt.sum(dims) + self.eps)
        return self.cw * ce + self.dw * (1 - dice.mean())
