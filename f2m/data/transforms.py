"""
f2m/data/transforms.py
======================
Fundus-specific image augmentation pipelines.

Key design decisions
--------------------
* Circular crop: fundus images are circular; mask the corners with the
  mean pixel value so the model never sees the black background as signal.
* Laterality flip is applied in F2MDataset BEFORE this transform pipeline
  so augmentation always operates on right-eye-equivalent images.
* Colour jitter is mild — fundus colour carries diagnostic meaning.
* No extreme geometric distortion (no perspective, no elastic) to preserve
  optic-disc geometry and vessel branching patterns.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ── constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── building blocks ───────────────────────────────────────────────────────────

class CircularCrop:
    """
    Mask the four corners of a square image with a neutral fill value so the
    model does not learn from the camera vignette / black corners typical of
    fundus photographs.

    The fundus disc fills roughly 85 % of the image diameter after centre-crop.
    We mask anything outside a circle of radius = min(H, W) / 2 * margin.

    Parameters
    ----------
    margin : float
        Fraction of the half-side that defines the valid radius. Default 0.97.
    fill_mode : str
        "imagenet_mean" — fill with per-channel ImageNet mean values.
        "image_mean"    — fill with the image's own mean (computed at call time).
    """

    def __init__(self, margin: float = 0.97, fill_mode: str = "imagenet_mean") -> None:
        self.margin = margin
        self.fill_mode = fill_mode

    def __call__(self, img: Image.Image) -> Image.Image:
        from PIL import ImageDraw
        W, H = img.size                      # PIL: (width, height)
        r = min(H, W) / 2.0 * self.margin

        # Determine fill colour
        if self.fill_mode == "imagenet_mean":
            fill_rgb = tuple(int(m * 255) for m in IMAGENET_MEAN)
        else:
            # sample mean from 4 corner pixels (fast, no full-array mean)
            corners = [
                img.getpixel((0,     0)),
                img.getpixel((W - 1, 0)),
                img.getpixel((0,     H - 1)),
                img.getpixel((W - 1, H - 1)),
            ]
            fill_rgb = tuple(
                int(sum(c[ch] for c in corners) / 4) for ch in range(3)
            )

        # Create a mask: white circle on black background
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        cx, cy = W / 2.0, H / 2.0
        # Bounding box of the circle
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=255)

        # Fill a solid-colour background image and composite using mask
        bg = Image.new("RGB", (W, H), fill_rgb)
        # Where mask=255 (inside circle), use img; where mask=0, use bg
        result = Image.composite(img, bg, mask)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(margin={self.margin})"


class GreenChannelCLAHE:
    """
    Enhance the green channel of the fundus image for vessel / RNFL contrast.

    Implementation: PIL global histogram equalisation on the green channel.
    This is PIL-native and avoids scikit-image / OpenMP conflicts that
    cause segfaults on macOS when mixing torchvision and skimage in the
    same process.  The visual benefit (improved vessel contrast) is the same
    for model training purposes.

    If `use_skimage=True` is passed AND scikit-image is available AND the
    runtime environment is Linux (no known OMP conflicts there), adaptive
    CLAHE via skimage.exposure.equalize_adapthist will be used instead.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8),
                 use_skimage: bool = False) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.use_skimage = use_skimage

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.use_skimage:
            try:
                from skimage import exposure
                arr = np.array(img)
                g = arr[:, :, 1].astype(np.float64) / 255.0
                g_eq = exposure.equalize_adapthist(
                    g, clip_limit=self.clip_limit, nbins=256
                )
                arr[:, :, 1] = (g_eq * 255).astype(np.uint8)
                return Image.frombytes(
                    "RGB", (arr.shape[1], arr.shape[0]), arr.tobytes()
                )
            except (ImportError, Exception):
                pass  # fall through to PIL implementation

        # PIL-native: split channels, equalise green, merge back
        # Uses no external C extensions — safe across Pillow 10+ and NumPy versions.
        from PIL import ImageOps
        r, g, b = img.split()
        g_eq = ImageOps.equalize(g)
        return Image.merge("RGB", (r, g_eq, b))


class SafeRotation:
    """
    Random rotation that fills newly introduced pixels with the image's
    border colour (extrapolation mode="border") rather than black.
    """

    def __init__(self, max_degrees: float = 30.0) -> None:
        self.max_degrees = max_degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = float(np.random.uniform(-self.max_degrees, self.max_degrees))
        # Use PIL's getpixel (no numpy) to get border fill colour.
        # np.array(img) on a real PIL image corrupts global PIL/PyTorch state
        # on macOS with Pillow 11+ / NumPy 1.21, causing SIGSEGV in T.ToTensor.
        fill = list(img.getpixel((0, 0)))
        return TF.rotate(img, angle, fill=fill)


# ── pipeline factories ────────────────────────────────────────────────────────

def build_train_transform(
    input_size: int = 300,
    circular_crop: bool = True,
    random_rotation: float = 30.0,
    random_flip_h: bool = False,
    color_jitter: dict | None = None,
    random_erasing: float = 0.1,
    mean: Sequence[float] = IMAGENET_MEAN,
    std:  Sequence[float] = IMAGENET_STD,
    use_clahe: bool = False,
) -> T.Compose:
    """
    Training augmentation pipeline.

    Order
    -----
    1. Resize (slightly larger than input_size to allow random crop)
    2. Random rotation
    3. Optional random horizontal flip  [default OFF — see note below]
    4. Colour jitter
    5. CentreCrop to input_size
    6. Circular corner masking
    7. Optional CLAHE
    8. ToTensor + Normalise
    9. Random Erasing (on tensor)

    Note on random_flip_h
    ---------------------
    ``F2MDataset`` normalises all images to right-eye orientation via a
    deterministic horizontal flip of left-eye images (``laterality_flip=True``).
    Enabling ``random_flip_h`` here would randomly undo that correction for
    ~50 % of left-eye samples, reintroducing the very orientation mismatch the
    dataset fix was designed to remove.  Set ``random_flip_h=True`` only when
    the dataset does NOT perform laterality correction.
    """
    cj = color_jitter or {
        "brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.02
    }
    scale_size = int(math.ceil(input_size * 1.15))

    ops: list = [
        T.Resize(scale_size, interpolation=T.InterpolationMode.BICUBIC),
    ]
    if random_rotation > 0:
        ops.append(SafeRotation(random_rotation))
    if random_flip_h:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops.append(T.ColorJitter(**cj))
    ops.append(T.CenterCrop(input_size))
    if circular_crop:
        ops.append(CircularCrop(margin=0.97, fill_mode="imagenet_mean"))
    if use_clahe:
        ops.append(GreenChannelCLAHE())
    ops += [
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ]
    if random_erasing > 0:
        ops.append(T.RandomErasing(p=random_erasing, scale=(0.02, 0.10)))
    return T.Compose(ops)


def build_val_transform(
    input_size: int = 300,
    circular_crop: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std:  Sequence[float] = IMAGENET_STD,
    use_clahe: bool = False,
) -> T.Compose:
    """
    Deterministic validation / test transform.

    Resize → CentreCrop → CircularCrop → CLAHE → ToTensor → Normalise.
    """
    ops: list = [
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
    ]
    if circular_crop:
        ops.append(CircularCrop(margin=0.97, fill_mode="imagenet_mean"))
    if use_clahe:
        ops.append(GreenChannelCLAHE())
    ops += [
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ]
    return T.Compose(ops)


def build_transform(
    split: str,
    input_size: int = 300,
    circular_crop: bool = True,
    random_rotation: float = 30.0,
    random_flip_h: bool = True,
    color_jitter: dict | None = None,
    random_erasing: float = 0.1,
    mean: Sequence[float] = IMAGENET_MEAN,
    std:  Sequence[float] = IMAGENET_STD,
    use_clahe: bool = False,
) -> T.Compose:
    """
    Convenience factory: returns train or val transform based on `split`.

    Parameters
    ----------
    split : "train" | "val" | "test" | "demo"
        "train" → augmented pipeline; all others → deterministic pipeline.
    """
    if split == "train":
        return build_train_transform(
            input_size=input_size,
            circular_crop=circular_crop,
            random_rotation=random_rotation,
            random_flip_h=random_flip_h,
            color_jitter=color_jitter,
            random_erasing=random_erasing,
            mean=mean,
            std=std,
            use_clahe=use_clahe,
        )
    return build_val_transform(
        input_size=input_size,
        circular_crop=circular_crop,
        mean=mean,
        std=std,
        use_clahe=use_clahe,
    )


def unnormalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std:  Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Inverse of Normalise for visualisation purposes.

    Parameters
    ----------
    tensor : [C, H, W] or [B, C, H, W], values in normalised space.

    Returns
    -------
    Tensor in [0, 1] range, same shape as input.
    """
    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    s = torch.tensor(std,  dtype=tensor.dtype, device=tensor.device)
    if tensor.ndim == 4:
        m, s = m[None, :, None, None], s[None, :, None, None]
    else:
        m, s = m[:, None, None], s[:, None, None]
    return (tensor * s + m).clamp(0.0, 1.0)
