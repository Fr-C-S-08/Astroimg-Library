"""
test_detection.py — Tests for the detection module.
"""

import numpy as np
import pandas as pd
from astroimg.detection import (
    detect_sources,
    detect_sources_consensus,
    count_sources,
    filter_sources,
)


def test_detect_sources_returns_dataframe(fake_image, fake_wcs):
    """detect_sources must return a DataFrame."""
    sources = detect_sources(fake_image, fake_wcs)
    assert isinstance(sources, pd.DataFrame)


def test_detect_sources_columns(fake_image, fake_wcs):
    """Output must have the required columns."""
    sources = detect_sources(fake_image, fake_wcs)
    required = ["x_pixel", "y_pixel", "ra", "dec", "peak_value", "snr"]
    for col in required:
        assert col in sources.columns


def test_detect_sources_finds_stars(fake_image, fake_wcs):
    """Should detect at least some of the 5 known stars."""
    sources = detect_sources(fake_image, fake_wcs, threshold=3.0)
    assert len(sources) >= 3


def test_detect_sources_brightest(fake_image, fake_wcs):
    """Brightest star (flux=10000 at 100,100) should be detected."""
    sources = detect_sources(fake_image, fake_wcs, threshold=3.0)
    # Check that a source exists near (100, 100)
    dists = np.sqrt(
        (sources["x_pixel"] - 100) ** 2 +
        (sources["y_pixel"] - 100) ** 2
    )
    assert dists.min() < 5


def test_detect_sources_sorted(fake_image, fake_wcs):
    """Sources should be sorted by peak_value descending."""
    sources = detect_sources(fake_image, fake_wcs, threshold=3.0)
    if len(sources) > 1:
        values = sources["peak_value"].values
        assert values[0] >= values[1]


def test_count_sources(fake_sources):
    """count_sources should return the length."""
    assert count_sources(fake_sources) == 5


def test_filter_sources_by_peak(fake_sources):
    """Filtering by min_peak should reduce sources."""
    filtered = filter_sources(fake_sources, min_peak=5000)
    assert len(filtered) < len(fake_sources)
    assert all(filtered["peak_value"] >= 5000)


def test_filter_sources_empty(fake_sources):
    """Filtering with impossible range returns empty."""
    filtered = filter_sources(fake_sources, min_peak=999999)
    assert len(filtered) == 0


def test_detect_sources_high_threshold(fake_image, fake_wcs):
    """Very high threshold should find fewer sources."""
    low = detect_sources(fake_image, fake_wcs, threshold=3.0)
    high = detect_sources(fake_image, fake_wcs, threshold=10.0)
    assert len(high) <= len(low)
