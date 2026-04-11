"""
Shared image I/O helpers: loading, listing, JPEG compression.
"""

from __future__ import annotations

import io
import os
import glob
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Baselines that should be excluded from reference-image listings
_BASELINE_NAMES = {
    "black.jpg", "black.jpeg", "black.png",
    "white.jpg", "white.jpeg", "white.png",
}


def list_image_files(
    directory: str,
    max_files: Optional[int] = None,
    exclude_baselines: bool = True,
) -> List[str]:
    """Collect image paths from *directory*, sorted, optionally capped.

    Parameters
    ----------
    directory : folder to scan (non-recursive)
    max_files : if set, return at most this many paths
    exclude_baselines : drop pure black/white baseline filenames
    """
    paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))

    seen: set[str] = set()
    out: list[str] = []
    for p in sorted(paths):
        if p in seen:
            continue
        seen.add(p)
        if exclude_baselines and os.path.basename(p).lower() in _BASELINE_NAMES:
            continue
        out.append(p)

    if max_files:
        out = out[:max_files]
    return out


def load_image(
    path: str,
    target_shape: Optional[Tuple[int, int]] = None,
    as_float: bool = True,
) -> Optional[np.ndarray]:
    """Load an image as RGB, optionally resize and convert to float64.

    Parameters
    ----------
    path : filesystem path
    target_shape : (H, W) — resize with Lanczos if not None
    as_float : return float64 (range [0, 255]) instead of uint8

    Returns
    -------
    RGB ndarray or None on failure.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_shape and (rgb.shape[0], rgb.shape[1]) != target_shape:
        rgb = cv2.resize(
            rgb,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    if as_float:
        return rgb.astype(np.float64)
    return rgb


def image_shape(path: str) -> Optional[Tuple[int, int]]:
    """Read (H, W) without decoding the full pixel buffer."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return (img.shape[0], img.shape[1])


def jpeg_compress(image: np.ndarray, quality: int = 85) -> np.ndarray:
    """JPEG round-trip: encode then decode in memory.

    Accepts [0, 1] float or [0, 255] uint8/float. Returns the same
    dtype/range convention as the input.
    """
    is_float01 = image.max() <= 1.5 and image.dtype in (np.float32, np.float64)

    if is_float01:
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)

    mode = "RGB" if img_uint8.ndim == 3 else "L"
    pil_img = Image.fromarray(img_uint8, mode=mode)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = np.array(Image.open(buf))

    if is_float01:
        return result.astype(np.float32) / 255.0
    return result.astype(np.float32)
