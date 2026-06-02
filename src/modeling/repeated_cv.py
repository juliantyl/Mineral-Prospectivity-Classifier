"""Phase 3b — REPEATED spatial block CV (de-luck the single-partition estimate).

Plain spatial CV (spatial_cv.py) carves the region into blocks and groups them into folds
ONE fixed way. The resulting mean rides on that one arrangement (e.g. our fold 1 happened to
get the barren ground). Repeated spatial CV runs the whole 5-fold cycle R times, each time:
  * jittering the block-grid ORIGIN (so block boundaries shift), and
  * randomly assigning blocks to folds,
then pools all R*N_FOLDS scores. No single lucky/unlucky partition can dominate, and the
spread now reflects "across many ways of carving the region", not just five slices of one.

Run:  python -m src.modeling.repeated_cv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src.modeling.spatial_cv import BLOCK_SIZE_M, N_FOLDS, NON_FEATURES, new_model

TABLE = C.PROCESSED / "training_table.csv"
N_REPEATS = 10


def blocks_with_jitter(x, y, size, rng) -> np.ndarray:
    """Block ids after shifting the grid origin by a random offset in [0, size)."""
    ox, oy = rng.uniform(0, size), rng.uniform(0, size)
    bx = np.floor((x + ox) / size).astype(int)
    by = np.floor((y + oy) / size).astype(int)
    return bx * 100_000 + by


def random_block_folds(blocks, n_folds, rng):
    """Yield (train_idx, test_idx) with whole blocks randomly assigned to folds."""
    uniq = np.unique(blocks)
    rng.shuffle(uniq)
    fold_of_block = {b: i % n_folds for i, b in enumerate(uniq)}
    fold = np.array([fold_of_block[b] for b in blocks])
    for k in range(n_folds):
        test = np.where(fold == k)[0]
        train = np.where(fold != k)[0]
        yield train, test


def main() -> None:
    df = pd.read_csv(TABLE)
    feat = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feat].to_numpy("float32")
    y = df["label"].to_numpy("int8")
    xs, ys = df["x"].to_numpy(), df["y_coord"].to_numpy()
    rng = np.random.default_rng(C.SEED)

    repeat_means, all_ap, all_roc = [], [], []
    print(f"running {N_REPEATS} repeats x {N_FOLDS} folds = {N_REPEATS*N_FOLDS} model fits...\n")
    for r in range(N_REPEATS):
        blocks = blocks_with_jitter(xs, ys, BLOCK_SIZE_M, rng)
        ap_fold = []
        for tr, te in random_block_folds(blocks, N_FOLDS, rng):
            m = new_model()
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            ap = average_precision_score(y[te], p)
            ap_fold.append(ap)
            all_ap.append(ap)
            all_roc.append(roc_auc_score(y[te], p))
        rmean = float(np.mean(ap_fold))
        repeat_means.append(rmean)
        print(f"  repeat {r:2d}: mean AP {rmean:.3f}   (folds {np.round(ap_fold,3)})")

    repeat_means = np.array(repeat_means)
    all_ap = np.array(all_ap)
    print("\n--- REPEATED spatial CV ---")
    print(f"AP per-repeat mean  : {repeat_means.mean():.3f}")
    print(f"  std ACROSS repeats : ±{repeat_means.std():.3f}   <- how much the ESTIMATE wobbles by partition")
    print(f"  std ACROSS 50 folds: ±{all_ap.std():.3f}   <- total fold-to-fold variability")
    print(f"ROC-AUC (pooled)    : {np.mean(all_roc):.3f} ± {np.std(all_roc):.3f}")
    print(f"\nfor reference, single-partition spatial CV was: AP 0.155 ± 0.067")

    np.savez(C.MODELS / "repeated_cv_scores.npz",
             repeat_means=repeat_means, all_ap=all_ap, all_roc=np.array(all_roc))


if __name__ == "__main__":
    main()
