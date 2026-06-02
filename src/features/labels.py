"""Phase 2, step 3 — build the labelled training table (X, y, coordinates).

This is the positive-unlabelled (PU) step. We have positive points (gold) but no confirmed
negatives, so we MANUFACTURE pseudo-negatives by buffered random background sampling:

  1. Positives: rasterise gold points onto the grid; mark the containing cell + a 1-cell
     halo (orebodies have extent and point locations are imprecise).
  2. Exclusion buffer: dilate the positives by BUFFER_CELLS and forbid negatives there, so
     we never label ground right next to a known deposit as "barren".
  3. Negatives: randomly draw n_neg = RATIO * n_pos cells from the eligible pool
     (valid features, outside the buffer). RATIO is the only knob on class balance.

Why the ratio is artificial (and what it costs): the real prevalence of gold is far below
1/(1+RATIO). Ranking metrics are unaffected, but predicted probabilities come out inflated,
which Phase 4 calibration must correct back toward the true base rate.

We keep each cell's projected (x, y) centroid — Phase 3 spatial CV needs coordinates.

Output: data/processed/training_table.csv
Run:    python -m src.features.labels
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from scipy.ndimage import binary_dilation

from src import config as C

FEATURES = C.INTERIM / "features_enriched.tif"
DEPOSITS = C.RAW / "gold_occurrences.gpkg"
OUT = C.PROCESSED / "training_table.csv"

RATIO = 20             # negatives per positive
POS_HALO_CELLS = 1     # cells around each deposit also counted positive
BUFFER_CELLS = 4       # 4 * 250 m = 1 km exclusion buffer around positives


def _disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element of the given radius (in cells)."""
    r = radius
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def build() -> None:
    rng = np.random.default_rng(C.SEED)

    with rasterio.open(FEATURES) as ds:
        names = list(ds.descriptions)
        stack = ds.read()                      # (bands, H, W), float32, NaN nodata
        transform = ds.transform
        H, W = ds.height, ds.width

    # a cell is usable only if EVERY feature is finite there
    valid = np.all(np.isfinite(stack), axis=0)

    # --- positives: rasterise gold points to grid cells ---
    gold = gpd.read_file(DEPOSITS).to_crs(C.CRS_PROJ)
    pos_seed = np.zeros((H, W), dtype=bool)
    for geom in gold.geometry:
        col, row = ~transform * (geom.x, geom.y)   # world -> (col,row)
        r, c = int(row), int(col)
        if 0 <= r < H and 0 <= c < W:
            pos_seed[r, c] = True
    n_points_on_grid = int(pos_seed.sum())

    # halo + must have valid features
    pos_mask = binary_dilation(pos_seed, structure=_disk(POS_HALO_CELLS)) & valid

    # --- exclusion buffer: no negatives within BUFFER_CELLS of any positive ---
    forbidden = binary_dilation(pos_seed, structure=_disk(BUFFER_CELLS))
    eligible_neg = valid & ~forbidden

    pos_rc = np.argwhere(pos_mask)
    n_pos = len(pos_rc)
    n_neg = min(RATIO * n_pos, int(eligible_neg.sum()))

    neg_idx_all = np.argwhere(eligible_neg)
    pick = rng.choice(len(neg_idx_all), size=n_neg, replace=False)
    neg_rc = neg_idx_all[pick]

    print(f"gold points landing on grid : {n_points_on_grid}")
    print(f"positive cells (incl. halo) : {n_pos}")
    print(f"eligible negative cells     : {int(eligible_neg.sum())}")
    print(f"negatives sampled (1:{RATIO})    : {n_neg}")

    # --- assemble the table ---
    rows = np.concatenate([pos_rc[:, 0], neg_rc[:, 0]])
    cols = np.concatenate([pos_rc[:, 1], neg_rc[:, 1]])
    y = np.concatenate([np.ones(n_pos, "int8"), np.zeros(n_neg, "int8")])

    xs, ys = xy(transform, rows, cols)            # projected centroids (metres)
    data = {"x": np.asarray(xs), "y_coord": np.asarray(ys), "label": y}
    for i, name in enumerate(names):
        data[name] = stack[i, rows, cols]
    df = pd.DataFrame(data)

    # shuffle so positives aren't all stacked at the top
    df = df.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print(f"\n[ok] wrote {OUT}")
    print(f"     {len(df)} rows  |  {df['label'].mean()*100:.2f}% positive")
    print(f"     columns: x, y_coord, label, + {len(names)} features")


if __name__ == "__main__":
    build()
