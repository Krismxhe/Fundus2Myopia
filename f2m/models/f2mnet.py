"""
f2m/models/f2mnet.py
====================
Fundus2Myopia Network — multi-task regression model.

Architecture
------------
Image [B,3,H,W]  +  Age [B]
       ↓                   ↓
   Backbone           AgeEncoder
  (CNN / ViT)         MLP(1→64)
       ↓                   ↓
  FiLM blocks  ←──── age_feat [B,64]
  (stage 3+4)
       ↓
  Spatial map [B,C,h,w]
       ↓
  Global avg pool → flatten [B,C]
       ↓
  Shared Neck (MLP: C→512→256, BN, GELU, Dropout)
       ↓
  shared_feat [B,256]
    ↙    ↓    ↘
 AL    SPH   CYL    (optional: OD area, PPA ratio)
 head  head  head
    ↘    ↓    ↙
       SE = SPH + CYL/2  (derived, no grad)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

from f2m.models.backbone import build_backbone, BackboneConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    """
    Encode demographic conditioning inputs — age (continuous) and gender
    (categorical) — into a single fixed-dimension vector consumed by FiLM blocks.

    Age
    ---
    z-scored continuous scalar.  Projected by a learned linear layer.

    Gender
    ------
    Integer category stored as int64.  Convention (matching governed datasets):
        0  =  unknown / not recorded   (maps to a zero vector — no information)
        1  =  male
        2  =  female

    ``padding_idx=0`` in the embedding table means the unknown category always
    produces a zero embedding, so it contributes nothing to the conditioning
    signal rather than pointing in some arbitrary direction.

    Architecture
    ------------
    age   → Linear(1, half_dim)  ─┐
                                   ├─ cat → GELU → Linear(dim, dim) → [B, dim]
    gender → Embedding(3, half_dim)─┘
    """

    def __init__(self, condition_dim: int = 64) -> None:
        super().__init__()
        if condition_dim % 2 != 0:
            raise ValueError("condition_dim must be even (split equally between age and gender).")
        half = condition_dim // 2
        self.age_proj   = nn.Linear(1, half)
        self.gender_emb = nn.Embedding(3, half, padding_idx=0)  # 0=unk, 1=M, 2=F
        self.fusion     = nn.Sequential(
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(
        self,
        age:    torch.Tensor,   # [B] or [B,1]  float, z-scored
        gender: torch.Tensor,   # [B]            int64, values ∈ {0, 1, 2}
    ) -> torch.Tensor:
        """Returns [B, condition_dim]."""
        if age.ndim == 1:
            age = age.unsqueeze(1)                              # [B, 1]
        age_feat    = self.age_proj(age.float())                # [B, half]
        gender_feat = self.gender_emb(gender.long())            # [B, half]
        combined    = torch.cat([age_feat, gender_feat], dim=-1)  # [B, dim]
        return self.fusion(combined)                            # [B, dim]


# Keep the old name as an alias so external code that imported AgeEncoder
# directly does not break immediately.
AgeEncoder = ConditionEncoder


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., 2018).

    Applies per-channel affine transformation conditioned on age_feat:
        out = γ(age_feat) ⊙ features + β(age_feat)

    Applied on spatial feature maps [B, C, h, w].
    """

    def __init__(self, feature_channels: int, condition_dim: int = 64) -> None:
        super().__init__()
        self.gamma = nn.Linear(condition_dim, feature_channels)
        self.beta  = nn.Linear(condition_dim, feature_channels)

    def forward(
        self,
        features: torch.Tensor,   # [B, C, h, w]
        condition: torch.Tensor,  # [B, condition_dim]
    ) -> torch.Tensor:
        gamma = self.gamma(condition)[:, :, None, None]   # [B, C, 1, 1]
        beta  = self.beta(condition)[:, :, None, None]
        return (1.0 + gamma) * features + beta            # residual-style


class SharedNeck(nn.Module):
    """
    Pool + project backbone features to a shared representation.

    Input : [B, in_features, h, w]  (spatial CNN map)
            [B, in_features]         (ViT CLS token — skip pool)
    Output: [B, out_features]
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 512,
        out_features: int = 256,
        dropout: float = 0.30,
        is_vit: bool = False,
    ) -> None:
        super().__init__()
        self.is_vit = is_vit
        if not is_vit:
            self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_vit and x.ndim == 4:
            x = self.pool(x).flatten(1)          # [B, C]
        return self.proj(x)                       # [B, out_features]


class RegressionHead(nn.Module):
    """
    Task-specific regression head: shared_feat → scalar prediction.

    Linear → GELU → Dropout → Linear(1)
    """

    def __init__(
        self,
        in_features: int = 256,
        hidden: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class F2MNet(nn.Module):
    """
    Fundus2Myopia Network.

    Parameters
    ----------
    backbone_name : str
        Registry key or "timm/<model>" (see f2m.models.backbone).
    task_names : list[str]
        Regression tasks. Default ["al", "sph", "cyl"].
        "sph" maps to "asph" column; "cyl" maps to "acyl" column.
    aux_task_names : list[str]
        Optional auxiliary regression tasks. Default [].
    neck_hidden : int
        Intermediate dimension of the shared neck. Default 512.
    neck_out : int
        Output dimension of the shared neck (input to task heads). Default 256.
    head_hidden : int
        Hidden dimension inside each task head. Default 128.
    condition_dim : int
        Dimension of the age conditioning embedding. Default 64.
    film_stages : list[int] | None
        Backbone stage indices where FiLM is applied.  If None, uses registry
        defaults from the backbone config.
    dropout_neck : float
        Dropout probability in the shared neck. Default 0.30.
    dropout_head : float
        Dropout probability in each task head. Default 0.10.
    pretrained : bool
        Load ImageNet / MAE pretrained weights for the backbone. Default True.
    freeze_stages : int
        Number of early backbone stages to freeze (0 = none). Default 0.
    """

    TASK_COLUMN_MAP = {
        "al":  "al",
        "sph": "asph",
        "cyl": "acyl",
        "se":  "se",
        "acd": "acd",
        "k1":  "k1",
    }

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        task_names: Sequence[str] = ("al", "sph", "cyl"),
        aux_task_names: Sequence[str] = (),
        neck_hidden: int = 512,
        neck_out: int = 256,
        head_hidden: int = 128,
        condition_dim: int = 64,
        film_stages: Sequence[int] | None = None,
        dropout_neck: float = 0.30,
        dropout_head: float = 0.10,
        pretrained: bool = True,
        freeze_stages: int = 0,
    ) -> None:
        super().__init__()
        self.task_names     = list(task_names)
        self.aux_task_names = list(aux_task_names)
        self.all_tasks      = self.task_names + self.aux_task_names

        # ── backbone ─────────────────────────────────────────────────────────
        self.backbone, self.backbone_cfg = build_backbone(
            backbone_name, pretrained=pretrained
        )
        feat_dim = self.backbone_cfg.feature_dim
        is_vit   = self.backbone_cfg.is_vit
        logger.info(
            "Backbone '%s': feature_dim=%d, input_size=%d, is_vit=%s",
            backbone_name, feat_dim, self.backbone_cfg.input_size, is_vit,
        )

        if freeze_stages > 0:
            self._freeze_backbone_stages(freeze_stages)

        # ── demographic conditioning (age + gender) ───────────────────────────
        self.condition_encoder = ConditionEncoder(condition_dim=condition_dim)

        # ── FiLM blocks ───────────────────────────────────────────────────────
        # For CNN backbones only (ViT uses token-level conditioning instead)
        self._film_hooks: list[torch.utils.hooks.RemovableHook] = []
        self._film_activations: dict[str, torch.Tensor] = {}
        self._current_age_feat: torch.Tensor | None = None

        film_layer_names = (
            self.backbone_cfg.film_layer_names if film_stages is None
            else self._resolve_film_names(film_stages)
        )

        self.film_blocks = nn.ModuleDict()
        if not is_vit and film_layer_names:
            self._register_film(film_layer_names, feat_dim, condition_dim)

        # ── shared neck ───────────────────────────────────────────────────────
        self.neck = SharedNeck(
            in_features=feat_dim,
            hidden=neck_hidden,
            out_features=neck_out,
            dropout=dropout_neck,
            is_vit=is_vit,
        )

        # ── task heads ────────────────────────────────────────────────────────
        self.task_heads = nn.ModuleDict({
            t: RegressionHead(neck_out, head_hidden, dropout_head)
            for t in self.all_tasks
        })

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        image:  torch.Tensor,                # [B, 3, H, W]
        age:    torch.Tensor,                # [B] or [B, 1]  float, z-scored
        gender: torch.Tensor | None = None,  # [B] int64  0=unk 1=M 2=F
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        image  : [B, 3, H, W]
        age    : [B] or [B, 1]  z-scored age (float)
        gender : [B]            int64 gender code.  If None, all samples are
                                treated as unknown (zeros — no gender signal).

        Returns
        -------
        {
          "al"          : [B, 1],
          "sph"         : [B, 1],
          "cyl"         : [B, 1],
          "se"          : [B, 1],            derived, no grad
          "shared_feat" : [B, neck_out],     for GradCAM / external hooks
          ...aux tasks...
        }
        """
        # 1. Demographic conditioning (age + gender → FiLM condition vector)
        B = image.shape[0]
        if gender is None:
            gender = torch.zeros(B, dtype=torch.long, device=image.device)
        cond_feat = self.condition_encoder(age, gender)   # [B, condition_dim]
        self._current_age_feat = cond_feat                # FiLM hooks read this

        # 2. Backbone (FiLM hooks fire during this call for CNN backbones)
        feat_map = self.backbone(image)                # [B, C, h, w] or [B, D]

        # 3. Neck
        shared = self.neck(feat_map)                   # [B, neck_out]

        # 4. Task heads
        out: dict[str, torch.Tensor] = {"shared_feat": shared}
        for task in self.all_tasks:
            out[task] = self.task_heads[task](shared)  # [B, 1]

        # 5. Derived SE (physics constraint; no gradient through this)
        if "sph" in out and "cyl" in out:
            with torch.no_grad():
                out["se"] = out["sph"] + out["cyl"] / 2.0

        return out

    # ── helpers ───────────────────────────────────────────────────────────────

    def _register_film(
        self,
        layer_names: list[str],
        feat_dim: int,
        condition_dim: int,
    ) -> None:
        """Register forward hooks to inject FiLM modulation after named layers."""
        named_modules = dict(self.backbone.named_modules())
        for name in layer_names:
            if name not in named_modules:
                logger.warning(
                    "FiLM target layer '%s' not found in backbone. Skipping.", name
                )
                continue
            # Determine channel count by running a tiny forward pass
            chan = self._infer_channels(named_modules[name], feat_dim)
            film = FiLMBlock(chan, condition_dim)
            self.film_blocks[name.replace(".", "_")] = film

            # Closure to capture name + film module
            def make_hook(film_block: FiLMBlock, _name: str):
                def hook(
                    module: nn.Module,
                    inp: tuple,
                    out: torch.Tensor,
                ) -> torch.Tensor:
                    if self._current_age_feat is not None and out.ndim == 4:
                        return film_block(out, self._current_age_feat)
                    return out
                return hook

            key = name.replace(".", "_")
            h = named_modules[name].register_forward_hook(
                make_hook(self.film_blocks[key], name)
            )
            self._film_hooks.append(h)
            logger.debug("Registered FiLM hook at backbone.%s (C=%d)", name, chan)

    def _infer_channels(self, layer: nn.Module, fallback: int) -> int:
        """
        Infer the output channel count of a backbone layer by running a
        tiny forward pass.
        """
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                # store output of this layer via a temporary hook
                _out = [None]
                h = layer.register_forward_hook(lambda m, i, o: _out.__setitem__(0, o))
                # Run a partial forward — use the whole backbone then detach
                # (expensive but only done once at init)
                self.backbone(dummy)
                h.remove()
                if _out[0] is not None and _out[0].ndim >= 2:
                    return _out[0].shape[1]
        except Exception as exc:
            logger.warning(
                "_infer_channels: dummy forward pass failed (%s). "
                "Falling back to backbone feature_dim=%d for FiLM block — "
                "if the target layer's actual channel count differs this "
                "will produce a shape mismatch on the first real forward pass.",
                exc, fallback,
            )
        return fallback

    def _resolve_film_names(self, stage_indices: Sequence[int]) -> list[str]:
        """Map numeric stage indices to named module keys (best effort)."""
        candidates = self.backbone_cfg.film_layer_names
        if not candidates:
            return []
        return [candidates[i] for i in stage_indices if i < len(candidates)]

    def _freeze_backbone_stages(self, n_stages: int) -> None:
        """Freeze the first `n_stages` stages of the backbone."""
        children = list(self.backbone.named_children())
        for i, (name, module) in enumerate(children[:n_stages]):
            for param in module.parameters():
                param.requires_grad = False
            logger.info("Froze backbone stage: %s", name)

    def param_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
    ) -> list[dict]:
        """
        Return parameter groups with differential learning rates.

        Backbone (pretrained) uses a smaller LR than the neck, heads, and FiLM.
        """
        backbone_params = list(self.backbone.parameters())
        other_params = [
            p for p in self.parameters()
            if not any(p is bp for bp in backbone_params)
        ]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params,    "lr": head_lr},
        ]

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters() if not trainable_only \
             else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in ps)
