"""Tests for src.utils.denoise — wavelet denoising."""

import numpy as np
import pytest

from src.utils.denoise import wavelet_denoise, extract_noise_residual


class TestWaveletDenoise:
    def test_output_shape_matches_input(self):
        ch = np.random.rand(64, 64).astype(np.float32)
        out = wavelet_denoise(ch)
        assert out.shape == ch.shape

    def test_smooth_signal_nearly_unchanged(self):
        ch = np.ones((64, 64), dtype=np.float64) * 0.5
        out = wavelet_denoise(ch)
        assert np.allclose(out, ch, atol=1e-6)

    def test_noisy_signal_gets_smoother(self):
        clean = np.ones((64, 64), dtype=np.float64) * 0.5
        noisy = clean + np.random.normal(0, 0.1, clean.shape)
        denoised = wavelet_denoise(noisy)
        assert np.std(denoised - clean) < np.std(noisy - clean)

    def test_different_wavelets(self):
        ch = np.random.rand(64, 64)
        for wv in ("db4", "sym8", "coif3"):
            out = wavelet_denoise(ch, wavelet=wv)
            assert out.shape == ch.shape


class TestExtractNoiseResidual:
    def test_grayscale(self):
        img = np.random.rand(64, 64)
        noise = extract_noise_residual(img)
        assert noise.shape == img.shape
        assert np.mean(np.abs(noise)) < np.std(img)

    def test_rgb(self):
        img = np.random.rand(64, 64, 3)
        noise = extract_noise_residual(img)
        assert noise.shape == img.shape

    def test_noise_sums_with_denoised(self):
        img = np.random.rand(64, 64)
        noise = extract_noise_residual(img)
        denoised = img - noise
        reconstructed = denoised + noise
        assert np.allclose(reconstructed, img, atol=1e-10)
