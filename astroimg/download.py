"""
download.py — Download FITS images from SkyView.

The user only provides ra, dec, radius.
Internally tries all surveys and all pixel sizes from highest to lowest.
Returns the best combination automatically.
"""

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astroquery.skyview import SkyView


CANDIDATE_SURVEYS = ["DSS", "DSS2 Red", "DSS2 Blue"]

PIXEL_CANDIDATES = [2000, 1500, 1000, 900, 700, 500, 300]


def _try_download(position, survey, pixels):
    """Try to download from SkyView. Returns image_list or None."""
    try:
        image_list = SkyView.get_images(
            position=position,
            survey=[survey],
            pixels=str(pixels),
        )
        if image_list:
            return image_list
        return None
    except Exception:
        return None


def _extract_data(image_list, survey, pixels_used):
    """Extract numpy array, header, and WCS from SkyView result."""
    hdu = image_list[0][0]
    data = hdu.data.astype(np.float64)
    header = hdu.header
    header["SURV_USE"] = (survey, "Survey used")
    header["PIX_USE"] = (pixels_used, "Pixel size used")
    wcs = WCS(header)

    nan_mask = ~np.isfinite(data)
    if nan_mask.any():
        data[nan_mask] = np.nanmedian(data)

    return data, header, wcs


def _quality_score(data, pixels):
    """Score = image quality x resolution bonus."""
    effective_range = np.percentile(data, 99.5) - np.percentile(data, 0.5)
    median = np.median(data)
    lower_half = data[data <= median]
    noise = np.std(lower_half) if len(lower_half) > 0 else 1.0
    quality = effective_range / max(noise, 1.0)
    resolution_bonus = pixels / 300.0
    return quality * resolution_bonus


def download_fits(ra, dec, radius=0.5):
    """
    Download a FITS image centered on the given sky coordinates.

    Automatically finds the best survey and highest resolution.

    Parameters
    ----------
    ra : float
        Right ascension in degrees (J2000).
    dec : float
        Declination in degrees (J2000).
    radius : float
        Half-width of the image in degrees. Default: 0.5.

    Returns
    -------
    data : np.ndarray
        2D float array of pixel intensities.
    header : fits.Header
        FITS header with metadata.
    wcs : WCS
        World Coordinate System for pixel <-> sky conversion.
    """
    position = f"{ra} {dec}"

    best_score = -1
    best_result = None
    best_info = ""

    print(f"Buscando mejor imagen para ra={ra}, dec={dec}...")

    for survey in CANDIDATE_SURVEYS:
        # Para cada survey, probar de mayor a menor resolucion
        for px in PIXEL_CANDIDATES:
            result = _try_download(position, survey, px)
            if result is None:
                continue

            # Encontramos una imagen que funciona
            data, header, wcs = _extract_data(result, survey, px)
            score = _quality_score(data, px)

            print(f"  {survey} {px}px -> score={score:.0f}")

            if score > best_score:
                best_score = score
                best_result = (data, header, wcs)
                best_info = f"{survey} {px}px"

            # Ya encontramos la max resolucion para este survey,
            # no necesitamos probar resoluciones menores
            break

    if best_result is None:
        raise ValueError(f"No images found at ra={ra}, dec={dec}")

    print(f"=> Mejor: {best_info} (score={best_score:.0f})")
    return best_result


def list_surveys():
    """Return the list of candidate survey names."""
    return list(CANDIDATE_SURVEYS)


def radec_to_pixel(ra, dec, wcs):
    """Convert sky coordinates to pixel coordinates."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def pixel_to_radec(x, y, wcs):
    """Convert pixel coordinates to sky coordinates."""
    coord = wcs.pixel_to_world(x, y)
    return float(coord.ra.deg), float(coord.dec.deg)
