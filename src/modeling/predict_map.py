"""Phase 5 — the prospectivity map: score EVERY grid cell.

Training/CV worked on a sample of cells. The deliverable is different: a continuous surface
over the whole region, so an explorer can see where to look. We:
  1. fit a sigmoid calibrator on spatially-honest out-of-fold predictions of the training rows,
  2. train the final GBM on ALL labelled rows,
  3. predict the calibrated probability for every valid cell in the 18-band feature stack,
  4. write a GeoTIFF (prospectivity.tif) + a map with the known deposits overlaid.

The map shows *relative* prospectivity (calibrated to the training prevalence, per Phase 4).
Known deposits are drawn on top as a visual sanity check — the hot zones should sit on them
without having been told exactly where they are at prediction time.

Run:  python -m src.modeling.predict_map
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src import config as C
from src.modeling.repeated_cv import load_xy
from src.modeling.spatial_cv import BLOCK_SIZE_M, N_FOLDS, assign_blocks, new_model

FEATURES = C.INTERIM / "features_enriched.tif"
DEPOSITS = C.RAW / "gold_occurrences.gpkg"
OUT_TIF = C.OUTPUTS / "prospectivity.tif"
OUT_FIG = C.FIGURES / "11_prospectivity_map.png"


def main() -> None:
    df, Xtr, ytr, feat = load_xy()
    xs, ys = df["x"].to_numpy(), df["y_coord"].to_numpy()

    # 1. sigmoid calibrator from spatially-honest out-of-fold predictions
    blocks = assign_blocks(xs, ys, BLOCK_SIZE_M)
    oof = np.zeros(len(ytr))
    for tr, te in GroupKFold(N_FOLDS).split(Xtr, ytr, groups=blocks):
        m = new_model().fit(Xtr[tr], ytr[tr])
        oof[te] = m.predict_proba(Xtr[te])[:, 1]
    sigmoid = LogisticRegression().fit(oof.reshape(-1, 1), ytr)

    # 2. final model on ALL labelled rows
    final = new_model().fit(Xtr, ytr)

    # 3. score every valid cell in the grid
    with rasterio.open(FEATURES) as ds:
        stack = ds.read()                       # (18, H, W)
        transform, crs = ds.transform, ds.crs
        H, W = ds.height, ds.width
        bounds = ds.bounds

    flat = stack.reshape(len(feat), -1).T        # (H*W, 18)
    valid = np.all(np.isfinite(flat), axis=1)
    raw = final.predict_proba(flat[valid])[:, 1]
    cal = sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1]

    prospectivity = np.full(H * W, np.nan, dtype="float32")
    prospectivity[valid] = cal
    prospectivity = prospectivity.reshape(H, W)
    print(f"scored {valid.sum():,} valid cells of {H*W:,}")
    print(f"prospectivity range: {np.nanmin(prospectivity):.3f} .. {np.nanmax(prospectivity):.3f}")

    # 4a. write GeoTIFF
    with rasterio.open(OUT_TIF, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(prospectivity, 1)
        dst.set_band_description(1, "calibrated_prospectivity")
    print(f"[ok] wrote {OUT_TIF}")

    # 4b. map with deposits overlaid
    gold = gpd.read_file(DEPOSITS).to_crs(crs)
    ext = [bounds.left / 1000, bounds.right / 1000, bounds.bottom / 1000, bounds.top / 1000]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(prospectivity, extent=ext, origin="upper", cmap="inferno", vmin=0,
                   vmax=np.nanpercentile(prospectivity, 99.5))
    ax.scatter(gold.geometry.x / 1000, gold.geometry.y / 1000, s=7, facecolor="none",
               edgecolor="cyan", linewidth=0.4, label=f"known gold (n={len(gold)})")
    ax.set_title("Gold prospectivity — Eastern Goldfields (calibrated GBM)\n"
                 "hot = more prospective; cyan = known deposits")
    ax.set_xlabel("easting (km, MGA51)"); ax.set_ylabel("northing (km, MGA51)")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="calibrated prospectivity", fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=130)
    print(f"[ok] saved {OUT_FIG}")


if __name__ == "__main__":
    main()
