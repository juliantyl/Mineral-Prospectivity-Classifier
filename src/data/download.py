"""Download bbox-clipped geophysical grids from Geoscience Australia via WCS.

Why WCS (Web Coverage Service) and not just downloading the national grids?
The national magnetic/radiometric/gravity grids are continent-sized (gigabytes each).
WCS lets us send a GetCoverage request with a *spatial subset* and get back a GeoTIFF
clipped to exactly our study bbox — a few MB. That's the difference between a 2-minute
setup and an afternoon of wrangling.

Run:  python -m src.data.download            # download all coverages in config
      python -m src.data.download rad_k      # just one (handy for testing the mechanism)
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import requests

from src import config as C


def _describe_axis_labels(coverage_id: str) -> tuple[str, str]:
    """Ask the server (DescribeCoverage) what the two spatial axes are called.

    WCS 2.0 subsetting addresses axes *by name* (e.g. Long/Lat, or E/N, or x/y) and
    different servers label them differently. Rather than guess, we read the labels
    from the coverage description so the GetCoverage request can't be silently wrong.
    """
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "DescribeCoverage",
        "coverageId": coverage_id,
    }
    r = requests.get(C.WCS_GEOPHYS, params=params, timeout=120)
    r.raise_for_status()
    # axisLabels lives on the gml:Envelope, e.g. axisLabels="Lat Long"
    root = ET.fromstring(r.content)
    for el in root.iter():
        labels = el.attrib.get("axisLabels")
        if labels:
            parts = labels.split()
            if len(parts) >= 2:
                return parts[0], parts[1]
    raise RuntimeError(f"Could not find axisLabels in DescribeCoverage for {coverage_id}")


def download_coverage(short_name: str, *, overwrite: bool = False) -> "Path":
    """Download one coverage clipped to config.BBOX_GEO as a GeoTIFF into data/raw."""
    from pathlib import Path  # local import keeps module import light

    coverage_id, desc = C.COVERAGES[short_name]
    out = C.RAW / f"{short_name}.tif"
    if out.exists() and not overwrite:
        print(f"[skip] {short_name}: already have {out.name}")
        return out

    min_lon, min_lat, max_lon, max_lat = C.BBOX_GEO
    ax0, ax1 = _describe_axis_labels(coverage_id)

    # Map whichever axis is lat vs long onto our bbox bounds.
    def _subset(axis: str) -> str:
        name = axis.lower()
        if "lat" in name or name in ("y", "n", "north"):
            return f"{axis}({min_lat},{max_lat})"
        return f"{axis}({min_lon},{max_lon})"

    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": coverage_id,
        "format": "image/tiff",
        # requests will repeat the 'subset' key for each axis
        "subset": [_subset(ax0), _subset(ax1)],
    }
    print(f"[get ] {short_name}: {desc}  (axes {ax0}/{ax1})")
    r = requests.get(C.WCS_GEOPHYS, params=params, timeout=600)
    r.raise_for_status()

    ctype = r.headers.get("Content-Type", "")
    if "tiff" not in ctype:
        # On error WCS returns an XML ExceptionReport with a 200 — surface it.
        raise RuntimeError(f"{short_name}: expected GeoTIFF, got {ctype!r}:\n{r.text[:800]}")

    out.write_bytes(r.content)
    print(f"[ok  ] {short_name}: wrote {out} ({len(r.content)/1e6:.1f} MB)")
    return out


def main(argv: list[str]) -> None:
    names = argv if argv else list(C.COVERAGES)
    for name in names:
        if name not in C.COVERAGES:
            print(f"[err ] unknown coverage {name!r}; known: {list(C.COVERAGES)}")
            continue
        download_coverage(name)


if __name__ == "__main__":
    main(sys.argv[1:])
