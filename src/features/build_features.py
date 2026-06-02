"""Phase 2, step 2 — derive geologically-meaningful features from the aligned stack.

Raw band values are a weak signal. We add two families of derived features that encode
what actually controls orogenic gold:

  RATIOS (radiometric): K/Th, U/Th, K/U. Ratios cancel common factors (regolith cover,
    survey gain) and isolate *relative* enrichment. K/Th in particular is a classic
    hydrothermal-alteration indicator.

  FOCAL STATS (neighbourhood): local mean and standard deviation over a small window.
    - focal mean  = the regional background level around a cell.
    - focal std   = local "roughness"/texture. High magnetic texture flags structural
                    complexity (faults, folds, contacts) — exactly where gold concentrates.

Window is deliberately small (FOCAL_WIN cells) so it stays far below the Phase-3 spatial-CV
block size; a large window would smear information across train/test blocks and inflate scores.

Output: data/interim/features_enriched.tif (all features, named bands, float32 / NaN nodata).

Run:  python -m src.features.build_features
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from src import config as C

SRC = C.INTERIM / "features_aligned.tif"
OUT = C.INTERIM / "features_enriched.tif"

FOCAL_WIN = 5          # cells; 5 * 250 m = 1.25 km window
RATIO_EPS = 1e-3       # guards divide-by-zero in ratios
FOCAL_BANDS = ["mag_1vd", "mag_as", "grav_cba", "rad_k"]  # where texture is informative


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + RATIO_EPS)


def _focal_stats(a: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware focal mean & std over a square window.

    uniform_filter averages over the window. To ignore NaNs we average the NaN-filled
    array and divide by the local fraction of valid cells: that yields the mean over
    *valid* cells only. Same trick gives E[x^2] for the variance.
    """
    valid = np.isfinite(a).astype("float32")
    a0 = np.where(np.isfinite(a), a, 0.0).astype("float32")
    cnt = uniform_filter(valid, size=size, mode="nearest")
    mean = uniform_filter(a0, size=size, mode="nearest") / np.where(cnt > 0, cnt, np.nan)
    meansq = uniform_filter(a0 * a0, size=size, mode="nearest") / np.where(cnt > 0, cnt, np.nan)
    std = np.sqrt(np.clip(meansq - mean * mean, 0, None))
    # restore NaN where the cell itself had no data
    nanmask = ~np.isfinite(a)
    mean[nanmask] = np.nan
    std[nanmask] = np.nan
    return mean.astype("float32"), std.astype("float32")


def build() -> None:
    with rasterio.open(SRC) as ds:
        names = list(ds.descriptions)
        bands = {n: ds.read(i + 1) for i, n in enumerate(names)}  # float32, NaN nodata
        profile = ds.profile

    feats: dict[str, np.ndarray] = dict(bands)  # start with the 7 raw bands

    # --- radiometric ratios ---
    feats["k_th"] = _safe_ratio(bands["rad_k"], bands["rad_th"])
    feats["u_th"] = _safe_ratio(bands["rad_u"], bands["rad_th"])
    feats["k_u"] = _safe_ratio(bands["rad_k"], bands["rad_u"])

    # --- focal mean / std on informative bands ---
    for b in FOCAL_BANDS:
        fmean, fstd = _focal_stats(bands[b], FOCAL_WIN)
        feats[f"{b}_fmean"] = fmean
        feats[f"{b}_fstd"] = fstd

    out_names = list(feats)
    profile.update(count=len(out_names))
    with rasterio.open(OUT, "w", **profile) as dst:
        for i, n in enumerate(out_names, start=1):
            dst.write(feats[n], i)
            dst.set_band_description(i, n)

    print(f"[ok] wrote {OUT}")
    print(f"     {len(out_names)} feature bands:")
    for n in out_names:
        print(f"       - {n}")


if __name__ == "__main__":
    build()
