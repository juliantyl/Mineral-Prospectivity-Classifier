"""Download known gold occurrences (positive labels) from GA's OZMIN WFS.

These points are our y=1. Remember the Phase-3 caveat: their *absence* somewhere does
NOT mean "barren" — it means "nobody has drilled/reported there." So this file only
defines the positives; how we treat everywhere-else is a separate modelling decision.

Definition chosen for v1: a positive is any mineral occurrence whose `commodity` field
mentions gold (Au) — including minor showings and prospects ("all gold occurrences").

Run:  python -m src.data.deposits
"""
from __future__ import annotations

import geopandas as gpd
import requests

from src import config as C

OUT = C.RAW / "gold_occurrences.gpkg"


def fetch_occurrences_in_bbox() -> gpd.GeoDataFrame:
    """Pull every mineral occurrence inside the study bbox as a GeoDataFrame."""
    min_lon, min_lat, max_lon, max_lat = C.BBOX_GEO
    # EPSG:4283 is a lat/lon (northing/easting) axis-order CRS, so the bbox KVP is
    # min_lat,min_lon,max_lat,max_lon + the CRS URI. (GeoJSON output is still lon,lat.)
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon},urn:ogc:def:crs:EPSG::4283"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": C.DEPOSITS_TYPENAME,
        "outputFormat": "application/json",
        "bbox": bbox,
    }
    r = requests.get(C.WFS_EARTHRESOURCE, params=params, timeout=300)
    r.raise_for_status()
    if "json" not in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"expected GeoJSON, got:\n{r.text[:800]}")
    gdf = gpd.read_file(r.text, driver="GeoJSON")
    if gdf.crs is None:
        gdf = gdf.set_crs(C.CRS_GEO)
    return gdf


def filter_gold(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep occurrences whose commodity string mentions gold."""
    commodity = gdf.get("commodity", "").fillna("")
    is_gold = commodity.str.contains("gold", case=False, na=False)
    return gdf.loc[is_gold].copy()


def main() -> None:
    gdf = fetch_occurrences_in_bbox()
    print(f"occurrences in bbox (all commodities): {len(gdf)}")

    # sanity-check that points actually fall inside our bbox (catches axis-order bugs)
    min_lon, min_lat, max_lon, max_lat = C.BBOX_GEO
    inside = gdf.geometry.x.between(min_lon, max_lon) & gdf.geometry.y.between(min_lat, max_lat)
    print(f"  of which inside bbox extent: {int(inside.sum())} (expect ~all)")

    gold = filter_gold(gdf)
    print(f"gold occurrences (y=1 positives): {len(gold)}")
    if len(gold):
        print("  example names:", list(gold["name"].head(5)))

    gold.to_file(OUT, driver="GPKG")
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
