"""Classification backbones, focal loss, temperature scaling and the multimodal fusion network.

Transfer learning strategy (2.5-D lesion patches, ImageNet-pretrained ResNet18/DenseNet121):
  * Why: public labelled lung-nodule data has ~10^3 lesions; training a deep CNN from scratch
    overfits. Early ImageNet filters (edges, blobs, textures) transfer well to CT patches.
  * Phase 1 (`freeze_backbone_epochs`): all convolutional layers frozen, only the new
    classification head is trained, so the random head does not destroy pretrained features.
  * Phase 2: the last stage (ResNet `layer4`, DenseNet `denseblock4`+`norm5`) and the head
    are unfrozen and fine-tuned with a 10x smaller LR for the backbone. Earlier stages stay
    frozen (generic low-level features; fewer trainable parameters = less overfitting).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) with optional class weights (alpha)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, 1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


class CNN3D(nn.Module):
    """Small 3-D CNN for lesion cubes (full profile)."""

    def __init__(self, in_channels: int = 1, n_classes: int = 2, base: int = 16, emb_dim: int = 128):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.BatchNorm3d(co), nn.ReLU(True),
                                 nn.Conv3d(co, co, 3, padding=1, bias=False), nn.BatchNorm3d(co), nn.ReLU(True),
                                 nn.MaxPool3d(2))

        self.features = nn.Sequential(blk(in_channels, base), blk(base, base * 2), blk(base * 2, base * 4),
                                      nn.Conv3d(base * 4, emb_dim, 3, padding=1), nn.ReLU(inplace=False))
        self.gradcam_layer = self.features[-1]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(emb_dim, n_classes))
        self.emb_dim = emb_dim

    def embed(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.fc(self.embed(x))


class Backbone2D(nn.Module):
    """ImageNet-pretrained torchvision backbone with a new head; exposes `embed()` and the
    Grad-CAM target layer."""

    def __init__(self, arch: str = "resnet18", n_classes: int = 2, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        self.arch = arch
        if arch == "resnet18":
            w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=w)
            self.emb_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.gradcam_layer = net.layer4
            self.finetune_modules = [net.layer4]
        elif arch == "densenet121":
            w = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.densenet121(weights=w)
            self.emb_dim = net.classifier.in_features
            net.classifier = nn.Identity()
            self.gradcam_layer = net.features.denseblock4
            self.finetune_modules = [net.features.denseblock4, net.features.norm5]
        elif arch == "efficientnet_b0":
            w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.efficientnet_b0(weights=w)
            self.emb_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.gradcam_layer = net.features[-1]
            self.finetune_modules = [net.features[-2], net.features[-1]]
        else:
            raise ValueError(f"Unknown architecture {arch}")
        self.net = net
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.emb_dim, n_classes))
        # ImageNet normalisation applied inside the model so callers pass [0,1] windowed CT.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def set_stage(self, stage: int) -> None:
        """stage 1: head only; stage 2: head + last backbone stage."""
        for p in self.net.parameters():
            p.requires_grad = False
        if stage >= 2:
            for m in self.finetune_modules:
                for p in m.parameters():
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def build_classifier(arch: str, n_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if arch == "cnn3d":
        return CNN3D(n_classes=n_classes)
    return Backbone2D(arch, n_classes, pretrained)


class TemperatureScaler(nn.Module):
    """Post-hoc calibration (Guo et al., 2017): fit a single temperature on the validation set."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> "TemperatureScaler":
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self


class MultimodalNet(nn.Module):
    """Late-fusion network: image embedding + clinical embedding -> task-specific heads.

    Heads are created only for tasks that have labels in the training data
    (see ctcancer.risk.labels). Missing clinical values are handled with an explicit
    missing-indicator per variable rather than silent imputation.
    """

    def __init__(self, img_dim: int, clin_dim: int, heads: dict[str, int], hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.clin_proj = nn.Sequential(nn.Linear(clin_dim, hidden // 2), nn.ReLU(True), nn.Dropout(dropout)) \
            if clin_dim > 0 else None
        fused = hidden + (hidden // 2 if clin_dim > 0 else 0)
        self.fusion = nn.Sequential(nn.Linear(fused, hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in heads.items()})

    def forward(self, img_emb: torch.Tensor, clin: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = [self.img_proj(img_emb)]
        if self.clin_proj is not None and clin is not None:
            z.append(self.clin_proj(clin))
        h = self.fusion(torch.cat(z, 1))
        return {k: head(h) for k, head in self.heads.items()}
