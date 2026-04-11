"""Tests for src.utils.metrics — PSNR and SSIM."""

import numpy as np
import pytest

from src.utils.metrics import compute_psnr, compute_ssim


class TestPSNR:
    def test_identical_images_return_inf(self):
        img = np.random.rand(32, 32, 3)
        assert compute_psnr(img, img) == float("inf")

    def test_known_mse(self):
        a = np.zeros((16, 16))
        b = np.ones((16, 16)) * 0.1
        expected = 10 * np.log10(1.0 / 0.01)
        assert abs(compute_psnr(a, b) - expected) < 0.01

    def test_uint8_range(self):
        a = np.full((16, 16, 3), 128, dtype=np.uint8)
        b = np.full((16, 16, 3), 138, dtype=np.uint8)
        psnr = compute_psnr(a, b)
        assert 28 < psnr < 50

    def test_symmetry(self):
        a = np.random.rand(16, 16)
        b = a + np.random.normal(0, 0.05, a.shape)
        assert abs(compute_psnr(a, b) - compute_psnr(b, a)) < 1e-10

    def test_more_noise_lower_psnr(self):
        a = np.random.rand(32, 32, 3)
        b_small = a + np.random.normal(0, 0.01, a.shape)
        b_large = a + np.random.normal(0, 0.10, a.shape)
        assert compute_psnr(a, b_small) > compute_psnr(a, b_large)


class TestSSIM:
    def test_identical_images_return_one(self):
        img = np.random.rand(64, 64, 3)
        assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_random_images_low_ssim(self):
        a = np.random.rand(64, 64, 3)
        b = np.random.rand(64, 64, 3)
        assert compute_ssim(a, b) < 0.5

    def test_near_identical_high_ssim(self):
        a = np.random.rand(64, 64, 3)
        b = a + np.random.normal(0, 0.005, a.shape)
        assert compute_ssim(a, b) > 0.95

    def test_grayscale(self):
        a = np.random.rand(64, 64)
        b = a + np.random.normal(0, 0.01, a.shape)
        ssim = compute_ssim(a, b)
        assert 0.8 < ssim <= 1.0

    def test_uint8_range(self):
        a = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        b = np.clip(a.astype(float) + np.random.normal(0, 2, a.shape), 0, 255).astype(np.uint8)
        ssim = compute_ssim(a, b)
        assert 0.9 < ssim <= 1.0

    def test_symmetry(self):
        a = np.random.rand(64, 64, 3)
        b = a + np.random.normal(0, 0.02, a.shape)
        assert abs(compute_ssim(a, b) - compute_ssim(b, a)) < 1e-6
