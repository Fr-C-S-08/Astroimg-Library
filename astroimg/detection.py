"""
detection.py — Detect astronomical sources in convolved images.

Takes the output of a convolution kernel (log_kernel or matched_filter)
and returns a table of detected sources with pixel and sky coordinates.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, label
from astropy.wcs import WCS

from astroimg.kernels import estimate_background


def detect_sources(
    data,
    wcs,
    kernel="log",
    sigma=2.0,
    threshold=5.0,
    min_separation=5,
    border=20,
):
    """
    Detect point sources in a FITS image.

    Parameters
    ----------
    data : np.ndarray
        2D image array (raw, from download_fits or download_best).
    wcs : astropy.wcs.WCS
        WCS object for pixel-to-sky coordinate conversion.
    kernel : str
        Kernel to use: 'log', 'matched', 'gaussian'. Default: 'log'.
    sigma : float
        Gaussian sigma for the kernel in pixels. Default: 2.0.
    threshold : float
        Detection threshold in sigma above background. Default: 5.0.
    min_separation : int
        Minimum separation between sources in pixels. Default: 5.
    border : int
        Ignore sources within this many pixels of the image edge. Default: 20.

    Returns
    -------
    pd.DataFrame
        Table with columns: x_pixel, y_pixel, ra, dec, peak_value, snr.
        Sorted by peak_value descending.
    """
    from astroimg.kernels import gaussian_kernel, log_kernel, matched_filter

    # Step 1 — Apply kernel directly (LoG and matched are zero-sum,
    # they already ignore constant backgrounds)
    if kernel == "log":
        convolved = -log_kernel(data, sigma=sigma)
    elif kernel == "matched":
        convolved = matched_filter(data, sigma=sigma)
    elif kernel == "gaussian":
        convolved = gaussian_kernel(data, sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Use 'log', 'matched' or 'gaussian'.")

    # Step 2 — Estimate noise of the convolved image
    bg_conv, noise_conv = estimate_background(convolved)
    detection_threshold = bg_conv + threshold * noise_conv

    # Step 3 — Binary mask of pixels above threshold
    above_threshold = convolved > detection_threshold

    # Step 4 — Find local maxima
    size = min_separation * 2 + 1
    local_max = maximum_filter(convolved, size=size) == convolved

    # Step 5 — Combine
    peaks = above_threshold & local_max

    # Exclude border pixels
    peaks[:border, :] = False
    peaks[-border:, :] = False
    peaks[:, :border] = False
    peaks[:, -border:] = False

    # Step 6 — Get coordinates
    y_coords, x_coords = np.where(peaks)

    if len(x_coords) == 0:
        return pd.DataFrame(columns=["x_pixel", "y_pixel", "ra", "dec", "peak_value", "snr"])

    # Step 7 — Calculate SNR for each source
    peak_values = convolved[y_coords, x_coords]
    snr_values = (peak_values - bg_conv) / noise_conv

    # Step 8 — Convert pixel to sky coordinates
    sky_coords = wcs.pixel_to_world(x_coords, y_coords)
    ra_values = sky_coords.ra.deg
    dec_values = sky_coords.dec.deg

    # Step 9 — Build DataFrame
    sources = pd.DataFrame({
        "x_pixel": x_coords.astype(int),
        "y_pixel": y_coords.astype(int),
        "ra": ra_values,
        "dec": dec_values,
        "peak_value": peak_values,
        "snr": snr_values,
    })

    sources = sources.sort_values("peak_value", ascending=False).reset_index(drop=True)

    # Step 10 — Merge nearby detections (bright stars = 1 source, not 3)
    sources = _merge_nearby(sources, merge_radius=8)

    return sources


def count_sources(sources):
    """Return the number of detected sources."""
    return len(sources)


def filter_sources(sources, min_peak=None, max_peak=None, ra_range=None, dec_range=None):
    """
    Filter a source catalog by peak value or sky coordinates.

    Parameters
    ----------
    sources : pd.DataFrame
        Output from detect_sources().
    min_peak : float or None
        Minimum peak value to keep.
    max_peak : float or None
        Maximum peak value to keep.
    ra_range : tuple (min_ra, max_ra) or None
        Keep only sources within this RA range in degrees.
    dec_range : tuple (min_dec, max_dec) or None
        Keep only sources within this Dec range in degrees.

    Returns
    -------
    pd.DataFrame
        Filtered source catalog.
    """
    filtered = sources.copy()

    if min_peak is not None:
        filtered = filtered[filtered["peak_value"] >= min_peak]
    if max_peak is not None:
        filtered = filtered[filtered["peak_value"] <= max_peak]
    if ra_range is not None:
        filtered = filtered[
            (filtered["ra"] >= ra_range[0]) & (filtered["ra"] <= ra_range[1])
        ]
    if dec_range is not None:
        filtered = filtered[
            (filtered["dec"] >= dec_range[0]) & (filtered["dec"] <= dec_range[1])
        ]

    return filtered.reset_index(drop=True)


def _merge_nearby(sources, merge_radius=8):
    """
    Merge detections that are too close — keep only the brightest.

    If multiple sources fall within merge_radius pixels of each other,
    only the one with the highest peak_value survives.

    Parameters
    ----------
    sources : pd.DataFrame
        Source catalog with x_pixel, y_pixel, peak_value.
    merge_radius : int
        Maximum distance in pixels to consider two detections
        as the same source. Default: 8.

    Returns
    -------
    pd.DataFrame
        Merged catalog.
    """
    if len(sources) <= 1:
        return sources

    sources = sources.sort_values("peak_value", ascending=False).reset_index(drop=True)

    keep = [True] * len(sources)
    x = sources["x_pixel"].values
    y = sources["y_pixel"].values

    for i in range(len(sources)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(sources)):
            if not keep[j]:
                continue
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist <= merge_radius:
                keep[j] = False

    return sources[keep].reset_index(drop=True)


def detect_sources_consensus(
    data,
    wcs,
    threshold=5.0,
    min_separation=5,
    border=20,
    match_radius=3,
    min_snr=6.0,
):
    """
    Detect sources using strict consensus between LoG and Matched filter.

    A source is reported only if:
      1. BOTH kernels detect it within match_radius pixels
      2. The SNR from BOTH detectors is above min_snr

    Parameters
    ----------
    data : np.ndarray
        2D image array.
    wcs : astropy.wcs.WCS
        WCS object for coordinate conversion.
    threshold : float
        Detection threshold in sigma. Default: 5.0.
    min_separation : int
        Minimum separation between sources in pixels. Default: 5.
    border : int
        Ignore pixels within this many pixels of the edge. Default: 20.
    match_radius : int
        Max distance in pixels to consider two detections the same
        source. Tighter = fewer false matches. Default: 3.
    min_snr : float
        Minimum SNR required from BOTH detectors. Default: 6.0.

    Returns
    -------
    pd.DataFrame
        Table with columns: x_pixel, y_pixel, ra, dec, peak_value, snr.
    """
    log_sources = detect_sources(
        data, wcs, kernel="log",
        threshold=threshold, min_separation=min_separation, border=border,
    )
    matched_sources = detect_sources(
        data, wcs, kernel="matched",
        threshold=threshold, min_separation=min_separation, border=border,
    )

    consensus_rows = []

    for _, log_row in log_sources.iterrows():
        lx, ly = log_row["x_pixel"], log_row["y_pixel"]
        log_snr = log_row["snr"]

        if log_snr < min_snr:
            continue

        if len(matched_sources) > 0:
            dists = np.sqrt(
                (matched_sources["x_pixel"] - lx) ** 2 +
                (matched_sources["y_pixel"] - ly) ** 2
            )
            best_idx = dists.idxmin()
            best_dist = dists[best_idx]
            best_snr = matched_sources.loc[best_idx, "snr"]

            if best_dist <= match_radius and best_snr >= min_snr:
                consensus_rows.append(log_row.to_dict())

    if not consensus_rows:
        return pd.DataFrame(
            columns=["x_pixel", "y_pixel", "ra", "dec", "peak_value", "snr"]
        )

    consensus = pd.DataFrame(consensus_rows).reset_index(drop=True)
    consensus = _merge_nearby(consensus, merge_radius=8)
    consensus = consensus.sort_values("peak_value", ascending=False).reset_index(drop=True)
    return consensus
