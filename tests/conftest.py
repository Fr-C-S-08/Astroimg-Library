"""
conftest.py — Shared test fixtures for astroimg tests.

Creates synthetic astronomical images so tests run instantly
without needing to download real FITS images.
"""

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS


@pytest.fixture
def fake_image():
    """
    Create a 200x200 synthetic image with known stars.

    Background: 100 counts, noise: 5 sigma
    Stars at known positions with known brightnesses.
    """
    np.random.seed(42)
    size = 200
    background = 100.0
    noise = 5.0

    data = background + np.random.normal(0, noise, (size, size))

    stars = [
        {"x": 50, "y": 50, "flux": 5000},
        {"x": 100, "y": 100, "flux": 10000},
        {"x": 150, "y": 150, "flux": 3000},
        {"x": 70, "y": 130, "flux": 7000},
        {"x": 160, "y": 40, "flux": 1500},
    ]

    for star in stars:
        yy, xx = np.ogrid[:size, :size]
        r2 = (xx - star["x"]) ** 2 + (yy - star["y"]) ** 2
        psf = star["flux"] * np.exp(-r2 / (2 * 3.0 ** 2))
        data += psf

    return data


@pytest.fixture
def fake_wcs():
    """
    Create a simple WCS for a 200x200 image.
    Centered at RA=180, Dec=45.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [100, 100]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def fake_sources():
    """
    Create a source catalog matching the fake_image stars.
    """
    return pd.DataFrame({
        "x_pixel": [50, 100, 150, 70, 160],
        "y_pixel": [50, 100, 150, 130, 40],
        "ra": [180.05, 180.0, 179.95, 180.03, 179.94],
        "dec": [44.95, 45.0, 45.05, 45.03, 44.94],
        "peak_value": [5000, 10000, 3000, 7000, 1500],
        "snr": [50, 100, 30, 70, 15],
    })
