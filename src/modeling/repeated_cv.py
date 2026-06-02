"""Phase 3b / shared engine — REPEATED spatial block CV.

Plain spatial CV (spatial_cv.py) carves the region into blocks and groups them into folds
ONE fixed way, so the mean rides on that arrangement. Repeated spatial CV runs the whole
k-fold cycle R times, each time jittering the block-grid ORIGIN and randomly assigning blocks
to folds, then pools all R*N_FOLDS scores. No single lucky/unlucky partition dominates.

`repeated_spatial_cv(make_model, ...)` is the reusable engine: pass any estimator factory
(a zero-arg callable returning a fresh, unfitted sklearn-style model) and it is evaluated
under identical folds. This is how Phase 4 compares logistic regression vs the GBM fairly.

Run:  python -m src.modeling.repeated_cv      # evaluates the default GBM
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
        yield np.where(fold != k)[0], np.where(fold == k)[0]


def repeated_spatial_cv(make_model, X, y, xs, ys, *, n_repeats=N_REPEATS, seed=C.SEED, verbose=True):
    """Evaluate any model factory under repeated spatial block CV. Returns a dict of arrays."""
    rng = np.random.default_rng(seed)
    repeat_means, all_ap, all_roc = [], [], []
    for r in range(n_repeats):
        blocks = blocks_with_jitter(xs, ys, BLOCK_SIZE_M, rng)
        ap_fold = []
        for tr, te in random_block_folds(blocks, N_FOLDS, rng):
            model = make_model()
            model.fit(X[tr], y[tr])               # any preprocessing in the model is fit on TRAIN only
            p = model.predict_proba(X[te])[:, 1]
            ap_fold.append(average_precision_score(y[te], p))
            all_ap.append(ap_fold[-1])
            all_roc.append(roc_auc_score(y[te], p))
        repeat_means.append(float(np.mean(ap_fold)))
        if verbose:
            print(f"  repeat {r:2d}: mean AP {repeat_means[-1]:.3f}   (folds {np.round(ap_fold,3)})")
    return {
        "repeat_means": np.array(repeat_means),
        "all_ap": np.array(all_ap),
        "all_roc": np.array(all_roc),
    }


def load_xy():
    df = pd.read_csv(TABLE)
    feat = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feat].to_numpy("float32")
    y = df["label"].to_numpy("int8")
    return df, X, y, feat


def main() -> None:
    df, X, y, feat = load_xy()
    xs, ys = df["x"].to_numpy(), df["y_coord"].to_numpy()
    print(f"running {N_REPEATS} repeats x {N_FOLDS} folds = {N_REPEATS*N_FOLDS} model fits...\n")
    res = repeated_spatial_cv(new_model, X, y, xs, ys)
    rm, ap = res["repeat_means"], res["all_ap"]
    print("\n--- REPEATED spatial CV (GBM) ---")
    print(f"AP per-repeat mean  : {rm.mean():.3f}")
    print(f"  std ACROSS repeats : ±{rm.std():.3f}")
    print(f"  std ACROSS folds   : ±{ap.std():.3f}")
    print(f"ROC-AUC (pooled)    : {res['all_roc'].mean():.3f} ± {res['all_roc'].std():.3f}")
    np.savez(C.MODELS / "repeated_cv_scores.npz",
             repeat_means=rm, all_ap=ap, all_roc=res["all_roc"])


if __name__ == "__main__":
    main()
