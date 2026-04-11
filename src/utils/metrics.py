"""
Shared image quality metrics used across bypass, detection, and benchmarking.
"""

import numpy as np


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two images.

    Both inputs must use the same value range. Supports [0, 1] float
    and [0, 255] uint8/float conventions — the peak value is inferred
    from ``original.max()``.
    """
    a = original.astype(np.float64)
    b = modified.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    peak = 1.0 if a.max() <= 1.5 else 255.0
    return float(10.0 * np.log10(peak ** 2 / mse))


def compute_ssim(
    original: np.ndarray,
    modified: np.ndarray,
    block_size: int = 8,
) -> float:
    """Structural Similarity Index (simplified block-based, no scipy needed).

    Operates on luminance. Accepts RGB or grayscale, [0,1] or [0,255].
    """
    a = original.astype(np.float64)
    b = modified.astype(np.float64)
    if a.max() > 1.5:
        a /= 255.0
        b /= 255.0

    if a.ndim == 3:
        a = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        b = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]

    blk = block_size
    rows = (a.shape[0] // blk) * blk
    cols = (a.shape[1] // blk) * blk

    ba = a[:rows, :cols].reshape(rows // blk, blk, cols // blk, blk)
    ba = ba.transpose(0, 2, 1, 3).reshape(-1, blk, blk)
    bb = b[:rows, :cols].reshape(rows // blk, blk, cols // blk, blk)
    bb = bb.transpose(0, 2, 1, 3).reshape(-1, blk, blk)

    mu_a = ba.mean(axis=(1, 2))
    mu_b = bb.mean(axis=(1, 2))
    var_a = ba.var(axis=(1, 2))
    var_b = bb.var(axis=(1, 2))
    cov_ab = ((ba - mu_a[:, None, None]) * (bb - mu_b[:, None, None])).mean(
        axis=(1, 2)
    )

    c1 = (0.01) ** 2  # k1=0.01
    c2 = (0.03) ** 2  # k2=0.03

    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)

    return float(np.mean(num / den))
