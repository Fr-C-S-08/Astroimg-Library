"""
photometry.py — Aperture photometry for detected astronomical sources.

Measures the flux (quantity of light) of each detected source by summing pixel values
within a circular aperture and subtracting the local sky background
estimated from a surrounding ring.

"""

import numpy as np
import pandas as pd


def aperture_photometry(data, sources, r_aperture=8, r_inner=12, r_outer=20, ):
    """
    Perform aperture photometry on detected sources.

    For each source, sums pixel values in a circular aperture,
    estimates the sky from a surrounding ring, subtracts the
    sky contribution of light, and converts to instrumental magnitudes.

    Parameters
    ----------
    data : np.ndarray
        2D raw image array (NOT convolved — use the original).
    sources : pd.DataFrame
        Source catalog from detect_sources(). Must contain
        columns 'x_pixel' and 'y_pixel'.
    r_aperture : int
        Radius of the source aperture in pixels. Default: 8.
    r_inner : int
        Inner radius of the sky ring in pixels. Default: 12.
    r_outer : int
        Outer radius of the sky ring in pixels. Default: 20.

    Returns
    -------
    pd.DataFrame
        Copy of the input catalog with added columns:
        flux, flux_err, mag, mag_err, sky_mean, sky_std.
    """
    if len(sources) == 0:
        output = sources.copy()
        for col in ["flux", "flux_err", "mag", "mag_err", "sky_mean", "sky_std"]:
            output[col] = []
        return output

    results = []

    for _, row in sources.iterrows():
        cx = int(row["x_pixel"])
        cy = int(row["y_pixel"])
        measurement = _measure_single(data, cx, cy, r_aperture, r_inner, r_outer)
        results.append(measurement)

    phot = pd.DataFrame(results)

    output = sources.copy().reset_index(drop=True)
    for col in phot.columns:
        output[col] = phot[col].values

    return output


def measure_source(data, x, y, r_aperture=8, r_inner=12, r_outer=20):
    """
    Measure photometry for a single position.

    Useful for measuring a specific coordinate without needing
    a full source catalog.

    Parameters
    ----------
    data : np.ndarray
        2D raw image array. 
    x : int
        X pixel coordinate of the source.
    y : int
        Y pixel coordinate of the source.
    r_aperture : int
        Aperture radius in pixels. Default: 8.
    r_inner : int
        Inner sky annulus radius. Default: 12.
    r_outer : int
        Outer sky annulus radius. Default: 20.

    Returns
    -------
    dict
        Dictionary with keys: flux, flux_err, mag, mag_err,
        sky_mean, sky_std, n_aperture, n_sky.
    """
    return _measure_single(data, int(x), int(y), r_aperture, r_inner, r_outer)


def _measure_single(data, cx, cy, r_ap, r_in, r_out):
    """
    Measure flux (quantity of light) and magnitude for one source.

    Parameters
    ----------
    data : np.ndarray
        Full 2D image.
    cx, cy : int
        Source center coordinates (x=column, y=row).
    r_ap : int
        Aperture radius.
    r_in : int
        Inner annulus radius.
    r_out : int
        Outer annulus radius.

    Returns
    -------
    dict
        Measurement results.
    """
    h, w = data.shape
    empty = {
        "flux": np.nan, "flux_err": np.nan,
        "mag": np.nan, "mag_err": np.nan,
        "sky_mean": np.nan, "sky_std": np.nan,
        "n_aperture": 0, "n_sky": 0,
    }

    if cy - r_out < 0 or cy + r_out >= h or cx - r_out < 0 or cx + r_out >= w:
        if cy - r_ap < 0 or cy + r_ap >= h or cx - r_ap < 0 or cx + r_ap >= w:
            return empty

    y_lo = max(0, cy - r_out)
    y_hi = min(h, cy + r_out + 1)
    x_lo = max(0, cx - r_out)
    x_hi = min(w, cx + r_out + 1)

    cutout = data[y_lo:y_hi, x_lo:x_hi].astype(float)

    local_y = np.arange(y_lo, y_hi) - cy
    local_x = np.arange(x_lo, x_hi) - cx
    xx, yy = np.meshgrid(local_x, local_y)
    dist2 = xx ** 2 + yy ** 2

    ap_mask = dist2 <= r_ap ** 2

    ann_mask = (dist2 >= r_in ** 2) & (dist2 <= r_out ** 2)

    sky_pixels = cutout[ann_mask]
    n_sky = len(sky_pixels)

    if n_sky < 10:
        return empty

    sky_mean, sky_std = _sigma_clip_median(sky_pixels)

    if sky_std == 0:
        sky_std = np.std(sky_pixels)

    ap_pixels = cutout[ap_mask]
    n_ap = len(ap_pixels)
    sum_ap = np.sum(ap_pixels)

    flux = sum_ap - n_ap * sky_mean

    flux_err = np.sqrt(
        np.abs(flux) + n_ap * sky_std ** 2 * (1.0 + n_ap / n_sky)
    )

    if flux > 0:
        mag = -2.5 * np.log10(flux)
        mag_err = (2.5 / np.log(10)) * (flux_err / flux)
    else:
        mag = np.nan
        mag_err = np.nan

    return {
        "flux": round(flux, 2),
        "flux_err": round(flux_err, 2),
        "mag": round(mag, 4) if not np.isnan(mag) else np.nan,
        "mag_err": round(mag_err, 4) if not np.isnan(mag_err) else np.nan,
        "sky_mean": round(sky_mean, 2),
        "sky_std": round(sky_std, 2),
        "n_aperture": n_ap,
        "n_sky": n_sky,
    }


def _sigma_clip_median(values, sigma=3.0, iters=3):
    """
    Compute sigma clipped median and robust standard deviation.

    Iteratively rejects outliers (other stars in the ring)
    to get a clean estimate of the sky level.

    Parameters
    ----------
    values : np.ndarray
        Array of pixel values from the sky annulus.
    sigma : float
        Clipping threshold in sigma. Default: 3.0.
    iters : int
        Number of clipping iterations. Default: 3.

    Returns
    -------
    tuple (median, std)
        Clipped median and standard deviation.
    """
    clipped = values.copy()

    for _ in range(iters):
        med = np.median(clipped)
        mad = np.median(np.abs(clipped - med))
        std = 1.4826 * mad

        if std == 0:
            break

        mask = np.abs(clipped - med) < sigma * std
        if mask.sum() < 3:
            break
        clipped = clipped[mask]

    final_med = np.median(clipped)
    final_mad = np.median(np.abs(clipped - final_med))
    final_std = 1.4826 * final_mad

    return final_med, final_std
