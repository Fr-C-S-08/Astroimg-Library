"""
crossmatch.py — Crossmatch detected sources with the Gaia DR3 catalog.

Queries the Gaia archive for all cataloged sources in the same
sky region as our detections, matches them by angular separation,
and returns a combined table with our photometry plus Gaia's
calibrated magnitudes, color index, temperature, and parallax.

Typical usage:
    from astroimg.crossmatch import crossmatch_gaia
    result = crossmatch_gaia(phot)
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def crossmatch_gaia(sources, radius_arcsec=5.0):
    """
    Crossmatch detected sources with the Gaia DR3 catalog.

    Steps:
      1. Calculate the sky region covered by our sources
      2. Query Gaia DR3 for all catalog sources in that region
      3. For each of our sources, find the nearest Gaia match
      4. Keep only matches closer than radius_arcsec (error)

    Parameters
    ----------
    sources : pd.DataFrame
        Source catalog from detect_sources() or aperture_photometry().
    radius_arcsec : float
        Maximum angular separation to accept a match.
        Default: 5.0 arcsec (typical for ground-based surveys).

    Returns
    -------
    pd.DataFrame
        Copy of the input catalog with added Gaia columns:
        gaia_source_id, gaia_ra, gaia_dec, gaia_parallax,
        gaia_parallax_error, gaia_pmra, gaia_pmdec,
        gaia_gmag, gaia_bp_mag, gaia_rp_mag, gaia_bp_rp,
        gaia_teff, gaia_ruwe, gaia_sep_arcsec.
        Unmatched sources have NaN in Gaia columns.
    """
    from astroquery.gaia import Gaia

    if len(sources) == 0:
        return _empty_output(sources)

    if "ra" not in sources.columns or "dec" not in sources.columns:
        raise ValueError("sources must contain 'ra' and 'dec' columns.")

    ra_min, ra_max = sources["ra"].min(), sources["ra"].max()
    dec_min, dec_max = sources["dec"].min(), sources["dec"].max()
    ra_center = (ra_min + ra_max) / 2
    dec_center = (dec_min + dec_max) / 2

    our_coords = SkyCoord(
        ra=sources["ra"].values,
        dec=sources["dec"].values,
        unit="deg",
    )
    center = SkyCoord(ra=ra_center, dec=dec_center, unit="deg")
    field_radius = center.separation(our_coords).max().deg + 0.01

    print(f"Querying Gaia DR3...")
    print(f"  Center: RA={ra_center:.4f}, Dec={dec_center:.4f}")
    print(f"  Radius: {field_radius:.4f} deg")

    query = f"""
    SELECT
        source_id, ra, dec,
        parallax, parallax_error,
        pmra, pmdec,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        bp_rp,
        teff_gspphot,
        ruwe
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center}, {dec_center}, {field_radius})
    )
    """

    try:
        Gaia.ROW_LIMIT = -1
        job = Gaia.launch_job_async(query)
        gaia_table = job.get_results()
    except Exception as e:
        print(f"  Error querying Gaia: {e}")
        return _empty_output(sources)

    gaia_df = _table_to_dataframe(gaia_table)
    print(f"  Gaia sources in field: {len(gaia_df)}")

    if len(gaia_df) == 0:
        print("  No Gaia sources found in this region.")
        return _empty_output(sources)

    gaia_coords = SkyCoord(
        ra=gaia_df["ra"].values,
        dec=gaia_df["dec"].values,
        unit="deg",
    )

    # For each of our sources, find the closest Gaia source (validation)
    idx, sep2d, _ = our_coords.match_to_catalog_sky(gaia_coords)
    sep_arcsec = sep2d.arcsec

    output = sources.copy().reset_index(drop=True)

    gaia_cols = _gaia_column_names()
    for col in gaia_cols:
        output[col] = np.nan

    good = sep_arcsec <= radius_arcsec

    if good.sum() > 0:
        matched_gaia = gaia_df.iloc[idx[good]].reset_index(drop=True)

        output.loc[good, "gaia_source_id"]      = matched_gaia["source_id"].values
        output.loc[good, "gaia_ra"]              = matched_gaia["ra"].values
        output.loc[good, "gaia_dec"]             = matched_gaia["dec"].values
        output.loc[good, "gaia_parallax"]        = matched_gaia["parallax"].values
        output.loc[good, "gaia_parallax_error"]  = matched_gaia["parallax_error"].values
        output.loc[good, "gaia_pmra"]            = matched_gaia["pmra"].values
        output.loc[good, "gaia_pmdec"]           = matched_gaia["pmdec"].values
        output.loc[good, "gaia_gmag"]            = matched_gaia["phot_g_mean_mag"].values
        output.loc[good, "gaia_bp_mag"]          = matched_gaia["phot_bp_mean_mag"].values
        output.loc[good, "gaia_rp_mag"]          = matched_gaia["phot_rp_mean_mag"].values
        output.loc[good, "gaia_bp_rp"]           = matched_gaia["bp_rp"].values
        output.loc[good, "gaia_teff"]            = matched_gaia["teff_gspphot"].values
        output.loc[good, "gaia_ruwe"]            = matched_gaia["ruwe"].values
        output.loc[good, "gaia_sep_arcsec"]      = sep_arcsec[good]

    n_matched = good.sum()
    pct = 100 * n_matched / len(sources)
    print(f"  Matched: {n_matched}/{len(sources)} sources ({pct:.1f}%)")

    return output


def query_gaia_field(ra, dec, radius_deg=0.15):
    """
    Query Gaia DR3 for all sources in a field.

    Useful for exploring the Gaia catalog independently
    of our detections.

    Parameters
    ----------
    ra : float
        Right ascension of field center in degrees.
    dec : float
        Declination of field center in degrees.
    radius_deg : float
        Search radius in degrees. Default: 0.15.

    Returns
    -------
    pd.DataFrame
        Gaia sources with standard columns.
    """
    from astroquery.gaia import Gaia

    print(f"Querying Gaia DR3 at RA={ra:.4f}, Dec={dec:.4f}, r={radius_deg:.3f} deg...")

    query = f"""
    SELECT
        source_id, ra, dec,
        parallax, parallax_error,
        pmra, pmdec,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        bp_rp,
        teff_gspphot,
        ruwe
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    """

    try:
        Gaia.ROW_LIMIT = -1
        job = Gaia.launch_job_async(query)
        gaia_table = job.get_results()
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

    gaia_df = _table_to_dataframe(gaia_table)
    print(f"  Found {len(gaia_df)} Gaia sources")
    return gaia_df


def crossmatch_stats(result):
    """
    Print summary statistics of a crossmatch result.

    Parameters
    ----------
    result : pd.DataFrame
        Output from crossmatch_gaia().
    """
    total = len(result)
    matched = result["gaia_source_id"].notna().sum()
    unmatched = total - matched

    print(f"Total sources:     {total}")
    print(f"Matched with Gaia: {matched} ({100*matched/total:.1f}%)")
    print(f"Unmatched:         {unmatched} ({100*unmatched/total:.1f}%)")

    if matched > 0:
        gm = result.loc[result["gaia_gmag"].notna(), "gaia_gmag"]
        print(f"\nGaia G magnitude range: {gm.min():.2f} to {gm.max():.2f}")

        sep = result.loc[result["gaia_sep_arcsec"].notna(), "gaia_sep_arcsec"]
        print(f"Separation: median={sep.median():.2f}\", max={sep.max():.2f}\"")

        has_teff = result["gaia_teff"].notna().sum()
        print(f"Sources with temperature: {has_teff}")

        has_parallax = result["gaia_parallax"].notna().sum()
        print(f"Sources with parallax:    {has_parallax}")

        has_color = result["gaia_bp_rp"].notna().sum()
        print(f"Sources with BP-RP color: {has_color}")

def _gaia_column_names():
    """Return the list of Gaia output column names."""
    return [
        "gaia_source_id", "gaia_ra", "gaia_dec",
        "gaia_parallax", "gaia_parallax_error",
        "gaia_pmra", "gaia_pmdec",
        "gaia_gmag", "gaia_bp_mag", "gaia_rp_mag",
        "gaia_bp_rp", "gaia_teff", "gaia_ruwe",
        "gaia_sep_arcsec",
    ]


def _empty_output(sources):
    """Return the input catalog with empty Gaia columns."""
    output = sources.copy()
    for col in _gaia_column_names():
        output[col] = np.nan
    return output


def _table_to_dataframe(table):
    """
    Convert an astropy Table to a pandas DataFrame,
    handling masked columns (replacing masked values with NaN).
    Integer columns are converted to float so NaN can be used.
    """
    df = pd.DataFrame()

    for col_name in table.colnames:
        col = table[col_name]
        try:
            if hasattr(col, "filled"):
                if col.dtype.kind == "i":
                    df[col_name] = col.astype(float).filled(np.nan).tolist()
                else:
                    df[col_name] = col.filled(np.nan).tolist()
            else:
                df[col_name] = col.tolist()
        except (TypeError, ValueError):
            values = []
            for val in col:
                if val is np.ma.masked:
                    values.append(np.nan)
                else:
                    values.append(val)
            df[col_name] = values

    return df

