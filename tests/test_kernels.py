"""
test_kernels.py — Tests for the kernels module.
"""

import numpy as np
from astroimg.kernels import (
    gaussian_kernel,
    log_kernel,
    matched_filter,
    estimate_background,
)


def test_gaussian_kernel_shape(fake_image):
    """Output must have the same shape as input."""
    result = gaussian_kernel(fake_image)
    assert result.shape == fake_image.shape


def test_log_kernel_shape(fake_image):
    """Output must have the same shape as input."""
    result = log_kernel(fake_image)
    assert result.shape == fake_image.shape


def test_matched_filter_shape(fake_image):
    """Output must have the same shape as input."""
    result = matched_filter(fake_image)
    assert result.shape == fake_image.shape


def test_gaussian_smooths(fake_image):
    """Gaussian should reduce noise (lower std than original)."""
    smoothed = gaussian_kernel(fake_image)
    original_std = np.std(fake_image[0:20, 0:20])
    smoothed_std = np.std(smoothed[0:20, 0:20])
    assert smoothed_std < original_std


def test_log_detects_peaks(fake_image):
    """LoG should produce strong response at star positions."""
    result = -log_kernel(fake_image)
    center_value = result[100, 100]
    bg_value = np.median(result[0:20, 0:20])
    assert center_value > bg_value


def test_estimate_background(fake_image):
    """Background should be close to 100 (the known value)."""
    bg, noise = estimate_background(fake_image)
    assert 90 < bg < 110
    assert 2 < noise < 10


def test_estimate_background_returns_tuple(fake_image):
    """estimate_background must return (background, noise)."""
    result = estimate_background(fake_image)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_gaussian_different_sigma(fake_image):
    """Different sigma values should produce different results."""
    r1 = gaussian_kernel(fake_image, sigma=1.0)
    r2 = gaussian_kernel(fake_image, sigma=5.0)
    assert not np.allclose(r1, r2)
