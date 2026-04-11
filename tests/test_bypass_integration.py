"""Integration tests for the bypass pipeline."""

import numpy as np
import pytest

from src.extraction.synthid_bypass import SynthIDBypass
from src.utils.metrics import compute_psnr, compute_ssim


@pytest.fixture
def synth_image():
    """A synthetic 128x128 RGB image with some structure."""
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 128)
    pattern = np.outer(np.sin(x), np.cos(x))
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    rgb = np.stack([pattern, pattern * 0.8, pattern * 0.6], axis=-1)
    return (rgb * 255).astype(np.uint8)


@pytest.fixture
def bypass():
    return SynthIDBypass()


class TestBypassSimple:
    def test_v1_jpeg_returns_valid_image(self, bypass, synth_image):
        result = bypass.bypass_simple(synth_image, jpeg_quality=50, verify=False)
        assert result.cleaned_image.shape == synth_image.shape
        assert result.cleaned_image.dtype == np.uint8

    def test_v1_psnr_above_30(self, bypass, synth_image):
        result = bypass.bypass_simple(synth_image, jpeg_quality=50, verify=False)
        assert result.psnr > 30


class TestBypassV2:
    def test_v2_moderate(self, bypass, synth_image):
        result = bypass.bypass_v2(
            synth_image, strength="moderate", iterations=1, verify=False
        )
        assert result.cleaned_image.shape == synth_image.shape
        assert result.psnr > 20

    def test_v2_aggressive(self, bypass, synth_image):
        result = bypass.bypass_v2(
            synth_image, strength="aggressive", iterations=1, verify=False
        )
        assert result.cleaned_image.shape == synth_image.shape


class TestQualityMetrics:
    def test_psnr_and_ssim_consistent(self, bypass, synth_image):
        result = bypass.bypass_simple(synth_image, jpeg_quality=85, verify=False)
        psnr = compute_psnr(synth_image, result.cleaned_image)
        ssim = compute_ssim(synth_image, result.cleaned_image)
        assert psnr > 30
        assert ssim > 0.9
