# Astroimg

A Python library for astronomical image analysis. Download FITS images from sky surveys, detect sources, measure their brightness, and cross-match with the Gaia DR3 catalog.

## Installation

    pip install astroimg

Or install from source:

    git clone https://github.com/Fr-C-S-08/Astroimg-Library.git
    cd Astroimg-Library
    pip install -e .

## Quick Start

    from astroimg import download_best, detect_sources, aperture_photometry, crossmatch_gaia

    # Download a FITS image
    data, header, wcs = download_best(ra=29.23, dec=37.79, radius=0.12)

    # Detect sources
    sources = detect_sources(data, wcs, kernel="log", threshold=5.0)

    # Measure brightness
    phot = aperture_photometry(data, sources)

    # Cross-match with Gaia DR3
    result = crossmatch_gaia(phot)

## Pipeline

    download_best() → detect_sources() → aperture_photometry() → crossmatch_gaia()
         ↓                  ↓                     ↓                      ↓
     FITS image       source catalog        flux & magnitudes     Gaia match + temp

## Modules

| Module | Description |
|--------|-------------|
| download | Download FITS images from DSS/SkyView |
| kernels | Gaussian, LoG, and matched filter convolution |
| detection | Source detection with local maxima and consensus |
| photometry | Aperture photometry with sky subtraction |
| crossmatch | Cross-match with Gaia DR3 catalog |
| visualization | Highlight stars and plot sources |

## Features

**Download astronomical images**

    from astroimg import download_best
    data, header, wcs = download_best(ra=250.42, dec=36.46, radius=0.15)

**Detect and count sources**

    from astroimg import detect_sources, count_sources
    sources = detect_sources(data, wcs, kernel="log", threshold=5.0)
    print(f"Found {count_sources(sources)} sources")

**Aperture photometry**

    from astroimg import aperture_photometry
    phot = aperture_photometry(data, sources)
    print(phot[["x_pixel", "y_pixel", "flux", "mag"]].head())

**Cross-match with Gaia**

    from astroimg import crossmatch_gaia, crossmatch_stats
    result = crossmatch_gaia(phot)
    crossmatch_stats(result)

**Highlight specific stars**

    from astroimg import highlight_star
    highlight_star(data, wcs, name="HD 196885", sources=sources)

## Docker

    docker compose up

## Tests

    pip install -e ".[dev]"
    pytest tests/ -v

## Requirements

- Python >= 3.9
- numpy, scipy, matplotlib, astropy, astroquery, pandas

## License

MIT


