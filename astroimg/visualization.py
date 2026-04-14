"""
visualization.py — Visualization tools for astronomical images.

Provides functions to plot images with detected sources,
highlight specific stars by name or coordinates, and
generate scientific diagrams.

Typical usage:
    from astroimg.visualization import highlight_star, plot_sources
    highlight_star(data, wcs, name="Vega")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord
import astropy.units as u


def highlight_star(data, wcs, name=None, ra=None, dec=None, sources=None, radius=30, color="lime", figsize=(10, 10), show=True,):
    """
    Highlight a specific star on the image.

    You can identify the star either by name (queried from Simbad)
    or by RA/Dec coordinates. Optionally overlay all detected sources.

    Parameters
    ----------
    data : np.ndarray
        2D image array.
    wcs : astropy.wcs.WCS
        WCS object for coordinate conversion.
    name : str or None
        Star name to query from Simbad (e.g. "Vega", "HD 11885").
        If provided, ra/dec are ignored.
    ra : float or None
        Right ascension in degrees. Used if name is None.
    dec : float or None
        Declination in degrees. Used if name is None.
    sources : pd.DataFrame or None
        Source catalog from detect_sources(). If provided, all
        detected sources are shown in cyan.
    radius : int
        Radius of the highlight circle in pixels. Default: 30.
    color : str
        Color of the highlight circle. Default: "lime".
    figsize : tuple
        Figure size. Default: (10, 10).
    show : bool
        If True, call plt.show(). Set False to add more to the plot.

    Returns
    -------
    dict
        Information about the star:
        name, ra, dec, x_pixel, y_pixel, in_image, in_catalog.
    """
    if name is not None:
        star_ra, star_dec, resolved_name = _resolve_name(name)
    elif ra is not None and dec is not None:
        star_ra, star_dec = ra, dec
        resolved_name = f"RA={ra:.4f}, Dec={dec:.4f}"
    else:
        raise ValueError("Provide either 'name' or both 'ra' and 'dec'.")

    star_coord = SkyCoord(ra=star_ra, dec=star_dec, unit="deg")
    px, py = wcs.world_to_pixel(star_coord)
    px, py = float(px), float(py)

    h, w = data.shape
    in_image = 0 <= px < w and 0 <= py < h

    in_catalog = False
    catalog_match = None
    if sources is not None and len(sources) > 0 and in_image:
        dists = np.sqrt(
            (sources["x_pixel"] - px) ** 2 +
            (sources["y_pixel"] - py) ** 2
        )
        nearest_idx = dists.idxmin()
        nearest_dist = dists[nearest_idx]
        if nearest_dist <= radius:
            in_catalog = True
            catalog_match = sources.loc[nearest_idx]

    print(f"Star: {resolved_name}")
    print(f"  RA={star_ra:.4f}, Dec={star_dec:.4f}")
    print(f"  Pixel: x={px:.1f}, y={py:.1f}")
    print(f"  In image: {in_image}")
    if in_catalog:
        print(f"  Found in catalog (distance: {nearest_dist:.1f} px)")
        if "snr" in catalog_match.index:
            print(f"  SNR: {catalog_match['snr']:.1f}")
        if "mag" in catalog_match.index and not np.isnan(catalog_match["mag"]):
            print(f"  Mag: {catalog_match['mag']:.2f}")
    elif in_image:
        print(f"  NOT found in catalog")

    if not in_image:
        print(f"  WARNING: Star is outside the image boundaries!")
        return {
            "name": resolved_name, "ra": star_ra, "dec": star_dec,
            "x_pixel": px, "y_pixel": py,
            "in_image": False, "in_catalog": False,
        }

    norm = plt.matplotlib.colors.LogNorm(
        vmin=np.percentile(data, 5),
        vmax=np.percentile(data, 99.5),
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(data, cmap="inferno", origin="lower", norm=norm)

    if sources is not None and len(sources) > 0:
        ax.scatter(
            sources["x_pixel"], sources["y_pixel"],
            s=15, facecolors="none", edgecolors="cyan",
            linewidths=0.5, alpha=0.6,
        )

    circle = Circle(
        (px, py), radius=radius, fill=False,
        edgecolor=color, linewidth=2.5, linestyle="--",
    )
    ax.add_patch(circle)

    ax.annotate(
        resolved_name, (px, py), color=color, fontsize=12,
        fontweight="bold",
        xytext=(40, 40), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )

    ax.set_title(f"{resolved_name} — highlighted", fontsize=14)

    if show:
        plt.tight_layout()
        plt.show()

    return {
        "name": resolved_name,
        "ra": star_ra,
        "dec": star_dec,
        "x_pixel": px,
        "y_pixel": py,
        "in_image": in_image,
        "in_catalog": in_catalog,
    }


def highlight_multiple(data, wcs, stars, sources=None, radius=30, figsize=(12, 12),):
    """
    Highlight multiple stars on the same image.

    Parameters
    ----------
    data : np.ndarray
        2D image array.
    wcs : astropy.wcs.WCS
        WCS object.
    stars : list of str or list of dict
        Either a list of star names: ["HD 11885", "HD 11606"]
        or a list of dicts: [{"ra": 29.0, "dec": 37.5, "label": "my star"}]
    sources : pd.DataFrame or None
        Source catalog to overlay.
    radius : int
        Circle radius in pixels. Default: 30.
    figsize : tuple
        Figure size. Default: (12, 12).

    Returns
    -------
    list of dict
        Info for each star.
    """
    colors = ["lime", "red", "yellow", "magenta", "orange", "white", "cyan"]

    norm = plt.matplotlib.colors.LogNorm(
        vmin=np.percentile(data, 5),
        vmax=np.percentile(data, 99.5),
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(data, cmap="inferno", origin="lower", norm=norm)

    if sources is not None and len(sources) > 0:
        ax.scatter(
            sources["x_pixel"], sources["y_pixel"],
            s=15, facecolors="none", edgecolors="cyan",
            linewidths=0.5, alpha=0.4,
        )

    results = []
    h, w = data.shape

    for i, star in enumerate(stars):
        color = colors[i % len(colors)]

        if isinstance(star, str):
            try:
                star_ra, star_dec, label = _resolve_name(star)
            except ValueError as e:
                print(f"  Skipping {star}: {e}")
                continue
        elif isinstance(star, dict):
            star_ra = star["ra"]
            star_dec = star["dec"]
            label = star.get("label", f"RA={star_ra:.2f}")
        else:
            continue

        star_coord = SkyCoord(ra=star_ra, dec=star_dec, unit="deg")
        px, py = wcs.world_to_pixel(star_coord)
        px, py = float(px), float(py)

        in_image = 0 <= px < w and 0 <= py < h
        if not in_image:
            print(f"  {label}: outside image")
            continue

        circle = Circle(
            (px, py), radius=radius, fill=False,
            edgecolor=color, linewidth=2.5, linestyle="--",
        )
        ax.add_patch(circle)

        ax.annotate(
            label, (px, py), color=color, fontsize=11,
            fontweight="bold",
            xytext=(35, 35 + i * 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

        print(f"  {label}: pixel ({px:.0f}, {py:.0f}) — {color}")
        results.append({
            "name": label, "ra": star_ra, "dec": star_dec,
            "x_pixel": px, "y_pixel": py, "in_image": True,
        })

    ax.set_title(f"{len(results)} stars highlighted", fontsize=14)
    plt.tight_layout()
    plt.show()

    return results


def plot_sources(data, sources, title="Detected sources", figsize=(10, 10), color="cyan",):
    """
    Quick plot of an image with detected sources overlaid.

    Parameters
    ----------
    data : np.ndarray
        2D image array.
    sources : pd.DataFrame
        Source catalog with x_pixel, y_pixel columns.
    title : str
        Plot title. Default: "Detected sources".
    figsize : tuple
        Figure size. Default: (10, 10).
    color : str
        Source marker color. Default: "cyan".
    """
    norm = plt.matplotlib.colors.LogNorm(
        vmin=np.percentile(data, 5),
        vmax=np.percentile(data, 99.5),
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(data, cmap="inferno", origin="lower", norm=norm)

    if len(sources) > 0:
        ax.scatter(
            sources["x_pixel"], sources["y_pixel"],
            s=20, facecolors="none", edgecolors=color, linewidths=0.7,
        )

    ax.set_title(f"{title} ({len(sources)} sources)", fontsize=14)
    plt.tight_layout()
    plt.show()

def _resolve_name(name):
    """
    Query Simbad for a star name and return RA, Dec in degrees.

    Parameters
    ----------
    name : str
        Star name (e.g. "Vega", "HD 11885", "M31").

    Returns
    -------
    tuple (ra, dec, resolved_name)
        Coordinates in degrees and the resolved name.
    """
    from astroquery.simbad import Simbad

    result = Simbad.query_object(name)

    if result is None:
        raise ValueError(f"Star '{name}' not found in Simbad.")

    ra = float(result["ra"][0])
    dec = float(result["dec"][0])

    return ra, dec, name
