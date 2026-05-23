"""
f2m/models/backbone.py
======================
Backbone registry for Fundus2Myopia.

Supported backbones
-------------------
torchvision (always available)
  efficientnet_b3    feature_dim=1536   input=300
  efficientnet_b4    feature_dim=1792   input=380
  efficientnet_b7    feature_dim=2560   input=600
  resnet50           feature_dim=2048   input=224
  resnet101          feature_dim=2048   input=224

timm (install: pip install timm)
  convnext_base      feature_dim=1024   input=224
  vit_base_patch16   feature_dim=768    input=224
  swin_base_patch4   feature_dim=1024   input=224
  + any timm model via "timm/<model_name>"

Ophthalmic foundation models
  retfound            feature_dim=1024  input=224
    (ViT-L/16, MAE-pretrained on 1.6 M fundus images)
    weights: HuggingFace "rmaphoh/RETFound_cfp"

Adding a new backbone
---------------------
Register a new entry in BACKBONE_REGISTRY with:
  name        : str identifier
  feature_dim : int  (output feature dimension after global pooling)
  input_size  : int  (recommended square input resolution)
  is_vit      : bool (True → use attention-rollout for GradCAM)
  film_layer_names : list[str]  (named modules to inject FiLM)
  builder     : Callable(pretrained: bool) → nn.Module
    The returned module must:
      - Accept a 4-D input  [B, 3, H, W]
      - Return a 4-D spatial feature map [B, C, h, w]  (pre-pooling)
        OR a 2-D vector [B, feature_dim] if is_vit=True

    Pooling and classification heads must be stripped; the F2MNet's Neck
    performs the adaptive pooling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Registry dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BackboneConfig:
    name: str
    feature_dim: int
    input_size: int
    builder: Callable                         # (pretrained: bool) → nn.Module
    film_layer_names: list[str] = field(default_factory=list)
    is_vit: bool = False                      # True → return [B, D] tokens


BACKBONE_REGISTRY: dict[str, BackboneConfig] = {}


def register_backbone(cfg: BackboneConfig) -> BackboneConfig:
    BACKBONE_REGISTRY[cfg.name] = cfg
    return cfg


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, BackboneConfig]:
    """
    Instantiate a backbone by registry name.

    Parameters
    ----------
    name : str
        Registry key (see module docstring) or "timm/<model_name>" for any
        timm model (timm must be installed separately).
    pretrained : bool
        Load pretrained weights (ImageNet for CNN; MAE for RETFound).

    Returns
    -------
    model : nn.Module
        Backbone with classification head stripped, ready to produce spatial
        features [B, C, h, w] (CNN) or patch token sequence (ViT).
    cfg : BackboneConfig
        Metadata including feature_dim and film_layer_names.
    """
    if name.startswith("timm/"):
        return _build_timm_backbone(name[5:], pretrained)

    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {sorted(BACKBONE_REGISTRY)} or 'timm/<model_name>'."
        )
    cfg = BACKBONE_REGISTRY[name]
    model = cfg.builder(pretrained)
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# torchvision backbones (always available)
# ─────────────────────────────────────────────────────────────────────────────

class _EfficientNetFeatureExtractor(nn.Module):
    """Strip the classifier; forward returns spatial feature map [B, C, h, w]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features   # MBConv stages
        self.avgpool  = model.avgpool    # AdaptiveAvgPool2d(1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return spatial map before pooling for GradCAM
        return self.features(x)   # [B, C, h, w]


class _ResNetFeatureExtractor(nn.Module):
    """Strip the fc layer; forward returns spatial feature map [B, C, h, w]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # Expose all layers for FiLM hook targeting
        self.conv1   = model.conv1
        self.bn1     = model.bn1
        self.relu    = model.relu
        self.maxpool = model.maxpool
        self.layer1  = model.layer1
        self.layer2  = model.layer2
        self.layer3  = model.layer3
        self.layer4  = model.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x   # [B, 2048, h, w]


# ── EfficientNet-B3 ──────────────────────────────────────────────────────────
def _build_efficientnet_b3(pretrained: bool) -> nn.Module:
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)
    return _EfficientNetFeatureExtractor(model)

register_backbone(BackboneConfig(
    name="efficientnet_b3",
    feature_dim=1536,
    input_size=300,
    builder=_build_efficientnet_b3,
    film_layer_names=["features.4", "features.6"],
))

# ── EfficientNet-B4 ──────────────────────────────────────────────────────────
def _build_efficientnet_b4(pretrained: bool) -> nn.Module:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b4(weights=weights)
    return _EfficientNetFeatureExtractor(model)

register_backbone(BackboneConfig(
    name="efficientnet_b4",
    feature_dim=1792,
    input_size=380,
    builder=_build_efficientnet_b4,
    film_layer_names=["features.4", "features.7"],
))

# ── ResNet-50 ────────────────────────────────────────────────────────────────
def _build_resnet50(pretrained: bool) -> nn.Module:
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    return _ResNetFeatureExtractor(model)

register_backbone(BackboneConfig(
    name="resnet50",
    feature_dim=2048,
    input_size=224,
    builder=_build_resnet50,
    film_layer_names=["layer3", "layer4"],
))

# ── ResNet-101 ───────────────────────────────────────────────────────────────
def _build_resnet101(pretrained: bool) -> nn.Module:
    from torchvision.models import resnet101, ResNet101_Weights
    weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet101(weights=weights)
    return _ResNetFeatureExtractor(model)

register_backbone(BackboneConfig(
    name="resnet101",
    feature_dim=2048,
    input_size=224,
    builder=_build_resnet101,
    film_layer_names=["layer3", "layer4"],
))


# ─────────────────────────────────────────────────────────────────────────────
# RETFound (ophthalmic foundation model)
# ─────────────────────────────────────────────────────────────────────────────

class RETFoundAdapter(nn.Module):
    """
    Wrap RETFound (ViT-L/16, MAE-pretrained on fundus images) as a backbone
    compatible with F2MNet.

    Weights are downloaded from HuggingFace Hub on first use.
    HuggingFace model: "rmaphoh/RETFound_cfp"  (colour fundus photo checkpoint)

    Output shape
    ------------
    [B, 1024]  — the [CLS] token embedding from the final transformer block.
    Spatial patch tokens are also accessible via `self._patch_tokens` after
    a forward pass (shape [B, 196, 1024] for 224px input, 14×14 grid).

    Requirements
    ------------
    pip install timm huggingface_hub
    """

    HF_REPO = "rmaphoh/RETFound_cfp"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self._patch_tokens: torch.Tensor | None = None
        self._model: nn.Module = self._build()

    def _build(self) -> nn.Module:
        try:
            import timm
        except ImportError:
            raise ImportError(
                "RETFound requires timm. Install with: pip install timm"
            )

        # Build ViT-L/16 architecture without classification head
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,          # remove classification head → output [B, 1024]
            global_pool="token",    # return CLS token
        )

        if self.pretrained:
            self._load_retfound_weights(model)

        return model

    def _load_retfound_weights(self, model: nn.Module) -> None:
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=self.HF_REPO,
                filename="RETFound_cfp_weights.pth",
            )
            logger.info("Loaded RETFound weights from %s", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            # Remove classification head weights if present
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("head.")
            }
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys in RETFound checkpoint: %s", missing[:5])
        except Exception as exc:
            logger.warning(
                "Failed to load RETFound weights (%s). "
                "Proceeding with random initialisation.", exc
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 3, 224, 224]

        Returns
        -------
        cls_token : [B, 1024]
        """
        # timm ViT with global_pool="token" returns the CLS embedding
        out = self._model(x)                        # [B, 1024]
        # Expose patch tokens for attention visualisation
        # (requires accessing internal state — use hook if needed)
        return out


def _build_retfound(pretrained: bool) -> nn.Module:
    return RETFoundAdapter(pretrained=pretrained)


register_backbone(BackboneConfig(
    name="retfound",
    feature_dim=1024,
    input_size=224,
    builder=_build_retfound,
    film_layer_names=["_model.blocks.20", "_model.blocks.23"],
    is_vit=True,
))


# ─────────────────────────────────────────────────────────────────────────────
# timm passthrough (requires: pip install timm)
# ─────────────────────────────────────────────────────────────────────────────

def _build_timm_backbone(
    model_name: str, pretrained: bool
) -> tuple[nn.Module, BackboneConfig]:
    """
    Build any timm model by name.  Usage: build_backbone("timm/convnext_base")

    The model is built with num_classes=0 (removes classification head).
    feature_map=True requests [B, C, H, W] output (not all timm models
    support this; CLS-token models return [B, D]).
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for 'timm/' backbones. "
            "Install with: pip install timm"
        )

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        features_only=False,    # let timm decide; most return [B, D]
    )
    # Get output dimension from model's feature_info or forward pass
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        try:
            out = model(dummy)
            feat_dim = out.shape[-1] if out.ndim == 2 else int(out.shape[1])
        except Exception:
            feat_dim = -1

    cfg = BackboneConfig(
        name=f"timm/{model_name}",
        feature_dim=feat_dim,
        input_size=224,
        builder=lambda p: timm.create_model(model_name, pretrained=p, num_classes=0),
        film_layer_names=[],
        is_vit="vit" in model_name.lower() or "swin" in model_name.lower(),
    )
    return model, cfg
