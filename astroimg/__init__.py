"""
astroimg — Astronomical image analysis library.

A Python library for downloading, processing, and analyzing
astronomical images from sky surveys like DSS.

Pipeline:
    download → kernels → detection → photometry → crossmatch → visualization
"""

__version__ = "0.1.0"

from astroimg.download import download_fits, download_best
from astroimg.kernels import (
    gaussian_kernel,
    log_kernel,
    matched_filter,
    estimate_background,
)
from astroimg.detection import (
    detect_sources,
    detect_sources_consensus,
    count_sources,
    filter_sources,
)
from astroimg.photometry import (
    aperture_photometry,
    measure_source,
)
from astroimg.crossmatch import (
    crossmatch_gaia,
    query_gaia_field,
    crossmatch_stats,
)
from astroimg.visualization import (
    highlight_star,
    highlight_multiple,
    plot_sources,
)
