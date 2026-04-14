"""
test_photometry.py — Tests for the photometry module.
"""

import numpy as np
import pandas as pd
from astroimg.photometry import aperture_photometry, measure_source


def test_aperture_photometry_returns_dataframe(fake_image, fake_sources):
    """aperture_photometry must return a DataFrame."""
    result = aperture_photometry(fake_image, fake_sources)
    assert isinstance(result, pd.DataFrame)


def test_aperture_photometry_adds_columns(fake_image, fake_sources):
    """Output must have flux, mag, and sky columns."""
    result = aperture_photometry(fake_image, fake_sources)
    required = ["flux", "flux_err", "mag", "mag_err", "sky_mean", "sky_std"]
    for col in required:
        assert col in result.columns


def test_aperture_photometry_preserves_rows(fake_image, fake_sources):
    """Output must have the same number of rows as input."""
    result = aperture_photometry(fake_image, fake_sources)
    assert len(result) == len(fake_sources)


def test_brightest_has_highest_flux(fake_image, fake_sources):
    """Star with flux=10000 should have the highest measured flux."""
    result = aperture_photometry(fake_image, fake_sources)
    brightest_idx = result["flux"].idxmax()
    assert result.loc[brightest_idx, "x_pixel"] == 100
    assert result.loc[brightest_idx, "y_pixel"] == 100


def test_flux_is_positive(fake_image, fake_sources):
    """All known stars should have positive flux."""
    result = aperture_photometry(fake_image, fake_sources)
    assert all(result["flux"] > 0)


def test_magnitude_ordering(fake_image, fake_sources):
    """Brighter stars should have more negative magnitudes."""
    result = aperture_photometry(fake_image, fake_sources)
    valid = result.dropna(subset=["mag"])
    brightest = valid.loc[valid["flux"].idxmax()]
    faintest = valid.loc[valid["flux"].idxmin()]
    assert brightest["mag"] < faintest["mag"]


def test_sky_estimation(fake_image, fake_sources):
    """Sky should be close to 100 (known background)."""
    result = aperture_photometry(fake_image, fake_sources)
    valid = result.dropna(subset=["sky_mean"])
    assert all(valid["sky_mean"] > 80)
    assert all(valid["sky_mean"] < 120)


def test_measure_source_returns_dict(fake_image):
    """measure_source must return a dictionary."""
    result = measure_source(fake_image, 100, 100)
    assert isinstance(result, dict)
    assert "flux" in result
    assert "mag" in result


def test_empty_sources(fake_image):
    """Empty source catalog should return empty DataFrame."""
    empty = pd.DataFrame(columns=["x_pixel", "y_pixel", "ra", "dec", "peak_value", "snr"])
    result = aperture_photometry(fake_image, empty)
    assert len(result) == 0
