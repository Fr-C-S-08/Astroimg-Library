"""
kernels.py — Convolution kernels for astronomical image preprocessing.

Implements three convolution kernels from mathematical fundamentals
using numpy and scipy.ndimage:

- gaussian_kernel  : Smooths noise, preserves structure.
- log_kernel       : Laplacian of Gaussian — detects point sources.
- matched_filter   : Maximises SNR for point-like sources.
- estimate_background : Estimates sky background level and noise.
"""

import numpy as np
from scipy.ndimage import convolve


# ──────────────────────────────────────────────
# Kernel builders (internal)
# ──────────────────────────────────────────────

def _gaussian_2d(size, sigma):
    """Build a normalised 2D Gaussian kernel."""
    if size % 2 == 0:
        size += 1
    center = size // 2
    y, x = np.ogrid[-center : center + 1, -center : center + 1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _log_2d(size, sigma):
    """Build a Laplacian-of-Gaussian (LoG) kernel."""
    if size % 2 == 0:
        size += 1
    center = size // 2
    y, x = np.ogrid[-center : center + 1, -center : center + 1]
    r2 = x**2 + y**2
    sigma2 = sigma**2
    kernel = -(1.0 - r2 / (2.0 * sigma2)) * np.exp(-r2 / (2.0 * sigma2))
    kernel -= kernel.mean()
    return kernel


def _matched_filter_2d(size, sigma):
    """Build a matched filter kernel for point-source detection."""
    kernel = _gaussian_2d(size, sigma)
    kernel -= kernel.mean()
    return kernel


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def gaussian_kernel(data, sigma=2.0, size=None):
    """
    Convolve an image with a 2D Gaussian kernel (smoothing).

    Suppresses high-frequency detector noise while preserving
    extended structures like nebulae and galaxy arms.

    Parameters
    ----------
    data : np.ndarray
        2D input image.
    sigma : float
        Gaussian standard deviation in pixels. Default: 2.0.
    size : int or None
        Kernel side length. If None, defaults to 6*sigma + 1.

    Returns
    -------
    np.ndarray
        Smoothed image, same shape as input.
    """
    data = np.asarray(data, dtype=np.float64)
    if size is None:
        size = int(6 * sigma + 1)
    kernel = _gaussian_2d(size, sigma)
    return convolve(data, kernel, mode="reflect")


def log_kernel(data, sigma=2.0, size=None):
    """
    Apply a Laplacian-of-Gaussian (LoG) filter for source detection.

    Highlights regions of rapid intensity change. Point sources
    (stars) produce strong peaks in the output.

    Parameters
    ----------
    data : np.ndarray
        2D input image.
    sigma : float
        Gaussian scale in pixels. Default: 2.0.
    size : int or None
        Kernel side length. Defaults to 6*sigma + 1.

    Returns
    -------
    np.ndarray
        LoG-filtered image. Peaks correspond to candidate sources.
    """
    data = np.asarray(data, dtype=np.float64)
    if size is None:
        size = int(6 * sigma + 1)
    kernel = _log_2d(size, sigma)
    return convolve(data, kernel, mode="reflect")


def matched_filter(data, sigma=2.0, size=None):
    """
    Apply a matched filter optimised for Gaussian point sources.

    Maximises signal-to-noise ratio for detecting compact sources
    (stars) against a flat background. Extended sources (galaxies)
    produce weaker responses.

    Parameters
    ----------
    data : np.ndarray
        2D input image.
    sigma : float
        Expected PSF sigma in pixels. Default: 2.0.
    size : int or None
        Kernel side length. Defaults to 6*sigma + 1.

    Returns
    -------
    np.ndarray
        Filtered image. Bright compact peaks are candidate stars.
    """
    data = np.asarray(data, dtype=np.float64)
    if size is None:
        size = int(6 * sigma + 1)
    kernel = _matched_filter_2d(size, sigma)
    return convolve(data, kernel, mode="reflect")


def estimate_background(data):
    """
    Estimate the sky background level and noise of an image.

    Uses iterative sigma-clipping to ignore bright sources
    and measure only the background.

    Parameters
    ----------
    data : np.ndarray
        2D image.

    Returns
    -------
    background : float
        Estimated sky background level.
    noise : float
        Estimated 1-sigma noise level.
    """
    flat = data.flatten()
    for _ in range(5):
        median = np.median(flat)
        mad = np.median(np.abs(flat - median))
        sigma = mad / 0.6745
        flat = flat[np.abs(flat - median) < 3.0 * sigma]

    background = float(np.median(flat))
    noise = float(np.median(np.abs(flat - background)) / 0.6745)
    return background, noise
