"""Tests for src.utils.image_io — image loading, listing, JPEG round-trip."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.utils.image_io import (
    list_image_files,
    load_image,
    image_shape,
    jpeg_compress,
)


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Create a temp directory with a few PNG images."""
    for i in range(3):
        img = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
    # add a baseline that should be excluded
    cv2.imwrite(str(tmp_path / "black.png"), np.zeros((32, 48, 3), dtype=np.uint8))
    return str(tmp_path)


class TestListImageFiles:
    def test_finds_images(self, tmp_image_dir):
        files = list_image_files(tmp_image_dir)
        assert len(files) == 3  # black.png excluded

    def test_includes_baselines_when_allowed(self, tmp_image_dir):
        files = list_image_files(tmp_image_dir, exclude_baselines=False)
        assert len(files) == 4

    def test_max_files(self, tmp_image_dir):
        files = list_image_files(tmp_image_dir, max_files=2)
        assert len(files) == 2

    def test_empty_dir(self, tmp_path):
        assert list_image_files(str(tmp_path)) == []


class TestLoadImage:
    def test_loads_png(self, tmp_image_dir):
        files = list_image_files(tmp_image_dir)
        img = load_image(files[0])
        assert img is not None
        assert img.ndim == 3
        assert img.dtype == np.float64

    def test_resize(self, tmp_image_dir):
        files = list_image_files(tmp_image_dir)
        img = load_image(files[0], target_shape=(16, 24))
        assert img.shape == (16, 24, 3)

    def test_as_uint8(self, tmp_image_dir):
        files = list_image_files(tmp_image_dir)
        img = load_image(files[0], as_float=False)
        assert img.dtype == np.uint8

    def test_nonexistent_returns_none(self):
        assert load_image("/nonexistent/path.png") is None


class TestImageShape:
    def test_returns_shape(self, tmp_image_dir):
        files = list_image_files(tmp_image_dir)
        shape = image_shape(files[0])
        assert shape == (32, 48)


class TestJpegCompress:
    def test_float01_roundtrip(self):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = jpeg_compress(img, quality=95)
        assert result.shape == img.shape
        assert result.max() <= 1.0

    def test_uint8_roundtrip(self):
        img = np.random.randint(0, 255, (64, 64, 3)).astype(np.float32)
        result = jpeg_compress(img, quality=90)
        assert result.shape == img.shape

    def test_lower_quality_more_distortion(self):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        hq = jpeg_compress(img, quality=95)
        lq = jpeg_compress(img, quality=20)
        err_hq = np.mean((img - hq) ** 2)
        err_lq = np.mean((img - lq) ** 2)
        assert err_lq > err_hq
