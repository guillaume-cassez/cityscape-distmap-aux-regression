"""
Model builder: constructs encoder-decoder segmentation models from config.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


def build_backbone(cfg: DictConfig) -> nn.Module:
    """Build backbone from config using timm."""
    import timm

    backbone_type = cfg.backbone.type

    timm_names = {
        "convnext_v2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
        "convnext_v2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
        "segformer_b5": "mit_b5",
        "internimage_t": "internimage_t",  # requires custom registration
        "swin_v2_base": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    }

    model_name = timm_names.get(backbone_type)
    if model_name is None:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    backbone = timm.create_model(
        model_name,
        pretrained=cfg.backbone.pretrained != "none",
        features_only=True,
        out_indices=list(cfg.backbone.out_indices),
        drop_path_rate=cfg.backbone.get("drop_path_rate", 0.0),
    )

    return backbone


def build_head(cfg: DictConfig, in_channels: list) -> nn.Module:
    """Build segmentation head from config."""
    head_type = cfg.head.type

    if head_type == "upernet":
        from src.models.heads.upernet import UPerNetHead
        return UPerNetHead(
            in_channels=in_channels,
            channels=cfg.head.channels,
            num_classes=cfg.head.num_classes,
            dropout=cfg.head.get("dropout", 0.1),
        )
    elif head_type == "segformer_head":
        from src.models.heads.segformer_head import SegFormerHead
        return SegFormerHead(
            in_channels=in_channels,
            channels=cfg.head.channels,
            num_classes=cfg.head.num_classes,
            dropout=cfg.head.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")


class SegmentationModel(nn.Module):
    """Generic encoder-decoder segmentation model."""

    def __init__(self, backbone: nn.Module, head: nn.Module, auxiliary=None, distmap=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.auxiliary = auxiliary
        self.distmap = distmap

    def _up(self, t, x):
        return nn.functional.interpolate(t, size=x.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        features = self.backbone(x)
        out = self._up(self.head(features), x)

        # DistMap variant (Paper 1): return a dict so the trainer can add the SDT
        # regression term. Kept separate from the legacy tuple path below so the
        # baseline / boundary variants are unaffected.
        if self.training and self.distmap is not None:
            result = {"main": out}
            if self.auxiliary is not None:
                result["aux_seg"] = self._up(self.auxiliary(features[-2]), x)
            result["distmap"] = torch.tanh(self._up(self.distmap(features[-2]), x))
            return result

        if self.training and self.auxiliary is not None:
            aux_out = self._up(self.auxiliary(features[-2]), x)  # from second-to-last stage
            return out, aux_out

        return out


def build_model(cfg: DictConfig) -> nn.Module:
    """Build full segmentation model from config."""
    backbone = build_backbone(cfg.model)

    # Get backbone output channels from a dummy forward pass
    dummy = torch.randn(1, 3, 256, 512)
    with torch.no_grad():
        feats = backbone(dummy)
    in_channels = [f.shape[1] for f in feats]

    head = build_head(cfg.model, in_channels)

    auxiliary = None
    if cfg.model.get("auxiliary", {}).get("enabled", False):
        aux_cfg = cfg.model.auxiliary
        auxiliary = nn.Sequential(
            nn.Conv2d(aux_cfg.in_channels, aux_cfg.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aux_cfg.channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(aux_cfg.channels, aux_cfg.num_classes, 1),
        )

    distmap = None
    dm_cfg = cfg.model.get("distmap", {})
    if dm_cfg and dm_cfg.get("enabled", False):
        # Auxiliary SDT-regression head (Paper 1 / DistMap). Same lightweight FCN as the
        # deep-supervision aux but outputs num_classes channels; tanh is applied in forward().
        distmap = nn.Sequential(
            nn.Conv2d(dm_cfg.in_channels, dm_cfg.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(dm_cfg.channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(dm_cfg.channels, dm_cfg.num_classes, 1),
        )

    model = SegmentationModel(backbone, head, auxiliary, distmap)
    return model
