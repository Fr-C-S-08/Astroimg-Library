"""
demo.py — Demonstrates the astroimg library pipeline.

Run with: docker compose up
"""

from astroimg import (
    download_best,
    detect_sources,
    count_sources,
    aperture_photometry,
    estimate_background,
)


def main():
    print("=" * 60)
    print("ASTROIMG — Astronomical Image Analysis Demo")
    print("=" * 60)

    # Step 1: Download
    print("\n[1] Downloading FITS image (NGC 752)...")
    data, header, wcs = download_best(ra=29.23, dec=37.79, radius=0.12)
    print(f"    Image size: {data.shape}")
    bg, noise = estimate_background(data)
    print(f"    Background: {bg:.1f}, Noise: {noise:.1f}")

    # Step 2: Detection
    print("\n[2] Detecting sources...")
    sources = detect_sources(data, wcs, kernel="log", threshold=5.0)
    print(f"    Found {count_sources(sources)} sources")
    print(f"    Brightest SNR: {sources['snr'].max():.1f}")

    # Step 3: Photometry
    print("\n[3] Performing aperture photometry...")
    phot = aperture_photometry(data, sources)
    valid = phot[phot["mag"].notna()]
    print(f"    Sources with valid magnitude: {len(valid)}")
    print(f"    Magnitude range: {valid['mag'].min():.2f} to {valid['mag'].max():.2f}")

    # Step 4: Top 5 brightest
    print("\n[4] Top 5 brightest sources:")
    top5 = valid.nsmallest(5, "mag")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"    {i}. pixel=({row['x_pixel']}, {row['y_pixel']}) "
              f"mag={row['mag']:.2f} flux={row['flux']:.0f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
