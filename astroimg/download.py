"""
download.py — Download FITS images from SkyView/DSS via astroquery.

Supports automatic survey selection and pixel size negotiation.
"""

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astroquery.skyview import SkyView


CANDIDATE_SURVEYS = [
    "DSS2 Red",
    "DSS2 Blue",
    "DSS",
    "DSS1 Red",
    "DSS1 Blue",
    "SDSSr",
    "SDSSg",
    "SDSSi",
    "2MASS-J",
    "2MASS-H",
    "2MASS-K",
]

PIXEL_CANDIDATES = [2000, 1500, 1000, 900, 700, 500, 400, 350, 300]


def _quality_score(data: np.ndarray) -> float:
    """
    Compute a quality score for a FITS image.
    Higher score = better image quality.
    """
    effective_range = np.percentile(data, 99.5) - np.percentile(data, 0.5)
    median = np.median(data)
    lower_half = data[data <= median]
    noise = np.std(lower_half) if len(lower_half) > 0 else 1.0
    return effective_range / max(noise, 1.0)


def _try_download(position: str, survey: str, pixels: int):
    """Try to download, return image_list or None."""
    try:
        image_list = SkyView.get_images(
            position=position,
            survey=[survey],
            pixels=str(pixels),
        )
        return image_list if image_list else None
    except Exception:
        return None


def _extract_data(
    image_list, survey: str, pixels_used: int
) -> tuple[np.ndarray, fits.Header, WCS]:
    """Extract data, header and WCS from a SkyView result."""
    hdu = image_list[0][0]
    data = hdu.data.astype(np.float64)
    header = hdu.header
    header["SURVEY_USED"] = survey
    header["PIXELS_USED"] = pixels_used
    wcs = WCS(header)

    nan_mask = ~np.isfinite(data)
    if nan_mask.any():
        data[nan_mask] = np.nanmedian(data)

    return data, header, wcs


def _find_best_pixels(position: str, survey: str, pixels: int) -> tuple:
    """Try requested pixels first, then fallbacks from high to low."""
    # Intentar el tamaño pedido primero
    result = _try_download(position, survey, pixels)
    if result:
        return result, pixels

    # Si falla, probar de mayor a menor
    for p in PIXEL_CANDIDATES:
        if p == pixels:
            continue
        result = _try_download(position, survey, p)
        if result:
            return result, p

    return None, 0


def download_fits(
    ra: float,
    dec: float,
    radius: float = 0.5,
    survey: str | None = None,
    pixels: int = 300,
    auto_best: bool = True,
) -> tuple[np.ndarray, fits.Header, WCS]:
    """
    Download a FITS image centered on the given sky coordinates.

    Parameters
    ----------
    ra : float
        Right ascension in degrees (J2000).
    dec : float
        Declination in degrees (J2000).
    radius : float
        Half-width of the image in degrees. Default: 0.5.
    survey : str or None
        Survey name. If None, tries all surveys and returns the
        best quality image automatically.
    pixels : int
        Requested image size in pixels. Default: 300.
        If not available, tries other sizes automatically.
    auto_best : bool
        If True and survey is None, compare all available surveys
        and return the highest quality. If False, return the first
        survey that works. Default: True.

    Returns
    -------
    data : np.ndarray
        2D float array of pixel intensities.
    header : fits.Header
        FITS header with metadata. Includes SURVEY_USED and
        PIXELS_USED keys.
    wcs : WCS
        World Coordinate System for pixel <-> sky conversion.

    Examples
    --------
    >>> data, header, wcs = download_fits(ra=83.82, dec=-5.39)
    >>> print(header["SURVEY_USED"], header["PIXELS_USED"])
    DSS2 Red 500
    """
    position_str = f"{ra} {dec}"

    if survey is not None:
        return _download_single(position_str, survey, pixels)

    if auto_best:
        return _download_best(position_str, pixels)
    else:
        return _download_first(position_str, pixels)


def _download_single(
    position: str, survey: str, pixels: int
) -> tuple[np.ndarray, fits.Header, WCS]:
    """Download from a specific survey, trying multiple pixel sizes."""
    result, pixels_used = _find_best_pixels(position, survey, pixels)

    if result is None:
        raise ValueError(
            f"No images returned for survey='{survey}' at {position}."
        )

    if pixels_used != pixels:
        print(f"pixels={pixels} not available, using pixels={pixels_used}")

    return _extract_data(result, survey, pixels_used)


def _download_best(
    position: str, pixels: int
) -> tuple[np.ndarray, fits.Header, WCS]:
    """Try all surveys, return the one with highest quality and resolution."""
    best_score = -1
    best_result = None
    best_survey = None
    best_pixels = 0
    tried = []

    for survey in CANDIDATE_SURVEYS:
        result, pixels_used = _find_best_pixels(position, survey, pixels)
        if result is None:
            continue

        data, header, wcs = _extract_data(result, survey, pixels_used)
        score = _quality_score(data)

        # Bonus por mayor resolución
        resolution_bonus = pixels_used / 300.0
        total_score = score * resolution_bonus

        tried.append(f"{survey}({pixels_used}px, score={total_score:.1f})")

        if total_score > best_score:
            best_score = total_score
            best_result = (data, header, wcs)
            best_survey = survey
            best_pixels = pixels_used

    if best_result is None:
        raise ValueError(
            f"No images found at {position} with any survey."
        )

    print(f"Best: {best_survey} at {best_pixels}px (score={best_score:.1f})")
    print(f"Tried: {', '.join(tried)}")
    return best_result


def _download_first(
    position: str, pixels: int
) -> tuple[np.ndarray, fits.Header, WCS]:
    """Try surveys in order, return the first that works."""
    for survey in CANDIDATE_SURVEYS:
        result, pixels_used = _find_best_pixels(position, survey, pixels)
        if result:
            print(f"Downloaded: {survey} at {pixels_used}px")
            return _extract_data(result, survey, pixels_used)

    raise ValueError(
        f"No images found at {position} with any survey."
    )


def list_surveys() -> list[str]:
    """Return the list of candidate survey names."""
    return list(CANDIDATE_SURVEYS)

def download_best(ra, dec, radius=0.5):
    """
    Download the best available FITS image for the given coordinates.

    Automatically tries all surveys and pixel sizes from highest
    to lowest, returning the best combination.

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
    surveys = ["DSS", "DSS2 Red", "DSS2 Blue"]
    pixels_to_try = [2000, 1500, 1000, 900, 700, 500, 300]
    position = f"{ra} {dec}"

    for survey in surveys:
        for px in pixels_to_try:
            result = _try_download(position, survey, px)
            if result is None:
                continue

            data, header, wcs = _extract_data(result, survey, px)
            print(f"  => {survey} {px}px")
            return data, header, wcs

    raise ValueError(f"No images found at ra={ra}, dec={dec}")

def radec_to_pixel(ra: float, dec: float, wcs: WCS) -> tuple[float, float]:
    """Convert sky coordinates to pixel coordinates."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


def pixel_to_radec(x: float, y: float, wcs: WCS) -> tuple[float, float]:
    """Convert pixel coordinates to sky coordinates."""
    coord = wcs.pixel_to_world(x, y)
    return float(coord.ra.deg), float(coord.dec.deg)