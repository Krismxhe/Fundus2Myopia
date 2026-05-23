"""
f2m/analysis/gradcam.py
=======================
Task-specific GradCAM for multi-head F2MNet.

Design
------
With a shared backbone, the activation tensor is identical for all tasks.
Only the gradient signal differs per task head — this is the correct
interpretation: the model attends to the same retinal regions but with
different emphasis depending on which measurement it is predicting.

Algorithm (per task)
--------------------
1. Forward pass with retain_graph=True.
2. Zero all gradients.
3. Backward only through predictions[task].sum() — isolates this task.
4. Retrieve activations A [C, h, w] (from forward hook) and
   gradients  G [C, h, w] (from backward hook).
5. Weights α_c = GAP(G)  [C]
6. CAM = ReLU( Σ_c α_c * A_c )   [h, w]
7. Upsample to original image resolution (bilinear).
8. Min-max normalise → [0, 1].

For ViT backbones (RETFound), use attention rollout instead of gradient-weighted
activation maps (see `AttentionRollout` class at the bottom of this file).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM engine
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadGradCAM:
    """
    Compute per-task GradCAM activation maps for F2MNet.

    Parameters
    ----------
    model : F2MNet
        The model must be in eval mode with gradients enabled.
    target_layer_name : str | None
        Fully-qualified name of the convolutional layer to hook.
        If None, auto-detected from backbone config film_layer_names[-1].
        Examples:
            EfficientNet-B3 : "backbone.features.8.0.block"  (last MB block)
            ResNet-50/101   : "backbone.layer4"
            RETFound (ViT)  : None → falls back to AttentionRollout
    tasks : list[str]
        Task names to compute maps for.  Default: all task heads.
    """

    def __init__(
        self,
        model: "F2MNet",          # type: ignore  — avoid circular import
        target_layer_name: str | None = None,
        tasks: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.tasks = list(tasks or model.task_names)
        self._is_vit = model.backbone_cfg.is_vit

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._fwd_hook = None
        self._bwd_hook = None

        if not self._is_vit:
            layer_name = target_layer_name or self._auto_detect_layer()
            self._register_hooks(layer_name)
        else:
            logger.info(
                "ViT backbone detected; GradCAM falls back to AttentionRollout."
            )

    # ── public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        image:  torch.Tensor,                # [1, 3, H, W]  (single image)
        age:    torch.Tensor,                # [1] or [1, 1]
        gender: torch.Tensor | None = None,  # [1] int64  0=unk 1=M 2=F
        tasks:  Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute GradCAM maps for the requested tasks.

        Parameters
        ----------
        image  : [1, 3, H, W]
        age    : [1] z-scored age (float)
        gender : [1] int64 gender code — must match the value used at
                 inference time so FiLM conditioning is consistent.
                 If None, all-zero (unknown) is used, which produces maps
                 under different conditioning than a known-gender forward pass.

        Returns
        -------
        {task_name: cam_map}  float32 [H_orig, W_orig] in [0, 1].
        """
        if self._is_vit:
            return AttentionRollout(self.model)(image, age, gender, tasks or self.tasks)

        tasks = list(tasks or self.tasks)
        H, W = image.shape[-2], image.shape[-1]
        results: dict[str, np.ndarray] = {}

        self.model.eval()
        image = image.to(next(self.model.parameters()).device)
        age   = age.to(image.device)
        if gender is not None:
            gender = gender.to(image.device)

        for task in tasks:
            if task not in self.model.task_heads:
                logger.warning("Task '%s' not in model task_heads; skipped.", task)
                continue

            # zero grads before each task pass
            self.model.zero_grad()
            self._activations = None
            self._gradients   = None

            # forward (retain graph so we can backward multiple times)
            out = self.model(image, age, gender=gender)
            pred = out[task]   # [1, 1]

            # backward only through this task's output
            pred.sum().backward(retain_graph=True)

            if self._activations is None or self._gradients is None:
                logger.warning(
                    "Hook did not fire for task '%s'. Layer name may be wrong.",
                    task,
                )
                continue

            cam = self._compute_cam(self._activations, self._gradients, H, W)
            results[task] = cam

        self.model.zero_grad()
        return results

    def overlay(
        self,
        image_np: np.ndarray,     # [H, W, 3]  uint8
        cam:      np.ndarray,     # [H, W]     float32  [0, 1]
        alpha:    float = 0.5,
        colormap: int | None = None,
    ) -> np.ndarray:
        """
        Blend GradCAM heatmap onto the original image.

        Returns
        -------
        [H, W, 3] uint8 overlay.
        """
        import matplotlib.cm as mpl_cm
        cmap = mpl_cm.get_cmap("jet")
        heatmap = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
        blended = (alpha * heatmap + (1 - alpha) * image_np).clip(0, 255).astype(np.uint8)
        return blended

    def visualize_all_tasks(
        self,
        image:  torch.Tensor,                # [1, 3, H, W]
        age:    torch.Tensor,                # [1]
        gender: torch.Tensor | None = None,  # [1] int64
        save_path: Path | None = None,
        unnorm_fn=None,           # optional callable to un-normalise image tensor
    ):
        """
        Produce a matplotlib Figure with panels:
          [ Original | AL CAM | SPH CAM | CYL CAM ]

        Returns
        -------
        plt.Figure
        """
        import matplotlib.pyplot as plt

        maps = self.compute(image, age, gender=gender)

        # Prepare display image
        img_np = image[0].detach().cpu()
        if unnorm_fn is not None:
            img_np = unnorm_fn(img_np)
        img_np = (img_np.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        task_labels = {
            "al":  "Axial Length", "sph": "Sphere",
            "cyl": "Cylinder",     "se":  "SE",
        }

        n_panels = 1 + len(maps)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for ax, (task, cam) in zip(axes[1:], maps.items()):
            overlay = self.overlay(img_np, cam, alpha=0.45)
            ax.imshow(overlay)
            ax.set_title(task_labels.get(task, task))
            ax.axis("off")

        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── private ───────────────────────────────────────────────────────────────

    def _auto_detect_layer(self) -> str:
        """Return the last named FiLM target layer as the GradCAM target."""
        cfg = self.model.backbone_cfg
        if cfg.film_layer_names:
            name = f"backbone.{cfg.film_layer_names[-1]}"
            logger.info("Auto-detected GradCAM target layer: %s", name)
            return name
        # fallback
        layers = [n for n, m in self.model.named_modules()
                  if isinstance(m, torch.nn.Conv2d)]
        if layers:
            logger.info("Auto-detected GradCAM target layer (last conv): %s", layers[-1])
            return layers[-1]
        raise RuntimeError("Cannot auto-detect GradCAM target layer.")

    def _register_hooks(self, layer_name: str) -> None:
        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            raise ValueError(
                f"GradCAM target layer '{layer_name}' not found in model. "
                f"Available modules: {list(dict(self.model.named_modules()).keys())[:20]}"
            )

        def fwd_hook(module, inp, out):
            self._activations = out.detach().clone()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach().clone()

        self._fwd_hook = target.register_forward_hook(fwd_hook)
        self._bwd_hook = target.register_full_backward_hook(bwd_hook)
        logger.debug("GradCAM hooks registered on '%s'", layer_name)

    @staticmethod
    def _compute_cam(
        activations: torch.Tensor,   # [B, C, h, w]  (B=1)
        gradients:   torch.Tensor,   # [B, C, h, w]
        orig_H: int,
        orig_W: int,
    ) -> np.ndarray:
        """Compute and upsample GradCAM map."""
        # Global average pool over spatial dims → weights [C]
        alpha = gradients.mean(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
        cam = (alpha * activations).sum(dim=1, keepdim=True)  # [B, 1, h, w]
        cam = F.relu(cam)

        # Upsample to original resolution
        cam = F.interpolate(
            cam, size=(orig_H, orig_W), mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].cpu().numpy()   # [H, W]

        # Min-max normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks from the model."""
        if self._fwd_hook is not None:
            self._fwd_hook.remove()
        if self._bwd_hook is not None:
            self._bwd_hook.remove()

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Attention Rollout for ViT (RETFound)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionRollout:
    """
    Attention Rollout (Abnar & Zuidema, 2020) for ViT-based backbones.

    Aggregates self-attention weights across transformer layers to produce
    a spatial attention map [14×14] for 224-px inputs.
    The map is resized to the original image resolution.

    Usage
    -----
    rollout = AttentionRollout(model)
    maps = rollout(image, age, tasks=["al", "sph", "cyl"])
    # Returns the same map for all tasks (backbone is shared; no task gradient)
    """

    def __init__(self, model: "F2MNet") -> None:  # type: ignore
        self.model = model
        self._attention_weights: list[torch.Tensor] = []
        self._hooks: list = []
        self._register_attn_hooks()

    def _register_attn_hooks(self) -> None:
        for name, module in self.model.named_modules():
            # timm ViT attention modules
            if hasattr(module, "attn_drop") and hasattr(module, "proj"):
                def make_hook(mod=module):
                    def hook(m, inp, out):
                        # some timm versions expose attn via scale / softmax internals
                        # We hook into the module's softmax output indirectly
                        pass
                    return hook
                # Simplified: just store the attn attribute if exposed
                h = module.register_forward_hook(
                    lambda m, i, o: self._attention_weights.append(
                        getattr(m, "_attn", None)
                    ) if hasattr(m, "_attn") else None
                )
                self._hooks.append(h)

    def __call__(
        self,
        image:  torch.Tensor,
        age:    torch.Tensor,
        gender: torch.Tensor | None = None,  # [1] int64
        tasks:  Sequence[str] = (),
    ) -> dict[str, np.ndarray]:
        """Returns the same attention map for every task (shared backbone)."""
        H, W = image.shape[-2], image.shape[-1]
        self._attention_weights.clear()
        with torch.no_grad():
            self.model(image, age, gender=gender)

        attn_maps = [a for a in self._attention_weights if a is not None]
        if not attn_maps:
            # Fallback: return uniform map
            logger.warning("No attention weights captured; returning uniform map.")
            cam = np.ones((H, W), dtype=np.float32)
            return {t: cam for t in tasks}

        # Rollout: multiply attention matrices layer by layer
        rollout = attn_maps[0].mean(dim=1)   # mean over heads: [B, N, N]
        for a in attn_maps[1:]:
            rollout = rollout @ a.mean(dim=1)

        # CLS token attention to all patch tokens
        grid = int(rollout.shape[-1] ** 0.5)
        cls_attn = rollout[0, 0, 1:].reshape(grid, grid).cpu().numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

        # Upsample
        from PIL import Image as PILImage
        cam_img = PILImage.fromarray((cls_attn * 255).astype(np.uint8))
        cam_resized = np.array(cam_img.resize((W, H), PILImage.BILINEAR)).astype(np.float32) / 255.0
        return {t: cam_resized for t in tasks}

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
