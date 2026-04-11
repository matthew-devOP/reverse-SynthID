"""
Shared wavelet denoising used across extraction, analysis, and bypass modules.
"""

import numpy as np
import pywt


def wavelet_denoise(
    channel: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> np.ndarray:
    """Wavelet-based soft-threshold denoising for a single 2-D channel.

    Uses the universal threshold ``sigma * sqrt(2 * log(N))`` estimated
    from the finest-level horizontal detail coefficients.

    Parameters
    ----------
    channel : 2-D float array (any value range)
    wavelet : pywt wavelet name
    level : decomposition depth

    Returns
    -------
    Denoised array with the same shape as *channel*.
    """
    coeffs = pywt.wavedec2(channel, wavelet, level=level)

    detail = coeffs[-1][0]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(channel.size))

    new_coeffs = [coeffs[0]]
    for details in coeffs[1:]:
        new_coeffs.append(
            tuple(pywt.threshold(d, threshold, mode="soft") for d in details)
        )

    denoised = pywt.waverec2(new_coeffs, wavelet)
    return denoised[: channel.shape[0], : channel.shape[1]]


def extract_noise_residual(
    image: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> np.ndarray:
    """Return the noise residual (image - denoised) for an RGB or gray image.

    If the input is RGB (H, W, 3), denoising is applied per-channel.
    Input should be float in any range (typically [0, 1] or [0, 255]).
    """
    if image.ndim == 2:
        return image - wavelet_denoise(image, wavelet, level)

    noise = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        noise[:, :, c] = image[:, :, c] - wavelet_denoise(
            image[:, :, c], wavelet, level
        )
    return noise
