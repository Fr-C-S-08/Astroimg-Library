"""
astroimg — Astronomical image analysis library.

Provides tools for downloading FITS images, detecting astronomical sources
via convolution kernels, measuring aperture photometry, and crossmatching
with the Gaia DR3 catalogue.

Basic usage
-----------
>>> from astroimg import ImageCatalog
>>> img = ImageCatalog.from_region(ra=266.4, dec=-29.0, radius=0.5)
>>> img.show()
>>> sources = img.detect_sources(kernel="log", threshold=3.0)
>>> sources.plot()
"""

__version__ = "0.1.0"
__author__ = "Paco"
__license__ = "MIT"

__all__ = [
    "ImageCatalog",
    "SourceCatalog",
    "__version__",
]
