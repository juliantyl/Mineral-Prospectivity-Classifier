"""Phase 2, step 1 — put every grid on ONE common analysis grid.

Right now the 7 rasters disagree on three things that make them impossible to stack into a
feature table: they're in lon/lat degrees (not metres), they have different pixel sizes
(93 m mag, 111 m rad, 464 m gravity), and slightly different extents/origins. A model needs
one row per location with all features lined up, so we resample everything onto a single
grid defined in MGA Zone 51 (metres) at config.TARGET_RES_M.

The resampling METHOD is chosen per layer by direction (this is the teaching point):
  * coarsening  (native finer than target, e.g. magnetics 93 m -> 250 m): use AVERAGE.
    Several source pixels fall in each output cell; averaging anti-aliases instead of
    discarding information.
  * interpolating (native coarser than target, e.g. gravity 464 m -> 250 m): use BILINEAR.
    There is no finer detail to average; we smoothly interpolate. NB this *invents* values
    gravity never measured — downstream we must not treat that smoothness as real structure.

Output: one multiband GeoTIFF data/interim/features_aligned.tif, bands named by layer,
float32 with NaN nodata throughout (uniform masking downstream).

Run:  python -m src.features.align
"""
from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from src import config as C

OUT = C.INTERIM / "features_aligned.tif"


def build_target_grid():
    """Define the common grid: bounds (in MGA51 metres) snapped to TARGET_RES_M."""
    res = C.TARGET_RES_M
    # Reproject the geographic study bbox into projected metres.
    w, s, e, n = transform_bounds(C.CRS_GEO, C.CRS_PROJ, *C.BBOX_GEO, densify_pts=21)
    # Snap outward to a tidy multiple of the resolution so cell edges are stable.
    w = math.floor(w / res) * res
    s = math.floor(s / res) * res
    e = math.ceil(e / res) * res
    n = math.ceil(n / res) * res
    width = int(round((e - w) / res))
    height = int(round((n - s) / res))
    transform = from_origin(w, n, res, res)  # top-left origin, y decreasing
    return transform, width, height


def _resampling_for(native_res_m: float) -> Resampling:
    """Average when coarsening, bilinear when interpolating (see module docstring)."""
    return Resampling.average if native_res_m < C.TARGET_RES_M else Resampling.bilinear


def align_all() -> None:
    transform, width, height = build_target_grid()
    res = C.TARGET_RES_M
    print(f"target grid: {width} x {height} cells @ {res} m  (CRS {C.CRS_PROJ})")
    print(f"             ~{width*res/1000:.0f} x {height*res/1000:.0f} km\n")

    names = list(C.COVERAGES)
    stack = np.full((len(names), height, width), np.nan, dtype="float32")

    for i, name in enumerate(names):
        with rasterio.open(C.RAW / f"{name}.tif") as src:
            native_res_m = src.res[0] * 111320  # deg -> m (approx, just to pick method)
            method = _resampling_for(native_res_m)
            src_arr = src.read(1).astype("float32")
            reproject(
                source=src_arr,
                destination=stack[i],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=C.CRS_PROJ,
                dst_nodata=np.nan,
                resampling=method,
            )
        valid = np.isfinite(stack[i]).mean() * 100
        print(f"  {name:10s} via {method.name:9s} (native ~{native_res_m:3.0f} m)  valid {valid:5.1f}%")

    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
        "width": width, "height": height, "count": len(names),
        "crs": C.CRS_PROJ, "transform": transform,
        "compress": "deflate", "predictor": 2,
    }
    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(stack)
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)
    print(f"\n[ok] wrote {OUT}  ({len(names)} bands)")


if __name__ == "__main__":
    align_all()
