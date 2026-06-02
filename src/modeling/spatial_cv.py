"""Phase 3 — the leakage demonstration: random CV vs spatial block CV.

Same data, same model, same metrics. The ONLY thing we change is how folds are formed:

  * random K-fold (StratifiedKFold): shuffle all rows into folds. Because deposits cluster
    and geophysics varies smoothly, a held-out cell almost always has a neighbour in the
    training set -> the model "looks up" the answer instead of generalising. Inflated scores.

  * spatial block CV (GroupKFold over square blocks): whole BLOCKS of ground go to train or
    test together, so test cells are physically separated from training cells. Honest scores.

The gap between the two is the headline result of the whole project.

Run:  python -m src.modeling.spatial_cv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src import config as C

TABLE = C.PROCESSED / "training_table.csv"
BLOCK_SIZE_M = 20_000   # 20 km spatial blocks
N_FOLDS = 5
NON_FEATURES = ("x", "y_coord", "label")


def assign_blocks(x: np.ndarray, y: np.ndarray, size: float) -> np.ndarray:
    """Map each (x, y) to an integer block id by snapping to a `size`-metre grid."""
    bx = np.floor(x / size).astype(int)
    by = np.floor(y / size).astype(int)
    # combine the two block coords into one id
    return bx * 100_000 + by


def new_model() -> LGBMClassifier:
    """One fixed model, reused across both CV schemes (so only the CV differs)."""
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=C.SEED, verbose=-1,
    )


def run_cv(X, y, splits) -> dict:
    """Train/score over the given (train_idx, test_idx) splits; return per-fold metrics."""
    ap, roc = [], []
    for tr, te in splits:
        model = new_model()
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        ap.append(average_precision_score(y[te], p))
        roc.append(roc_auc_score(y[te], p))
    return {"ap": np.array(ap), "roc": np.array(roc)}


def main() -> None:
    df = pd.read_csv(TABLE)
    feat_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feat_cols].to_numpy("float32")
    y = df["label"].to_numpy("int8")
    base_rate = y.mean()

    blocks = assign_blocks(df["x"].to_numpy(), df["y_coord"].to_numpy(), BLOCK_SIZE_M)
    n_blocks = len(np.unique(blocks))

    print(f"rows {len(df)} | positives {int(y.sum())} ({base_rate*100:.2f}%) | "
          f"features {len(feat_cols)} | spatial blocks {n_blocks} @ {BLOCK_SIZE_M/1000:.0f} km\n")

    # --- random K-fold (the leaky baseline) ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED)
    rand = run_cv(X, y, skf.split(X, y))

    # --- spatial block CV (the honest estimate) ---
    gkf = GroupKFold(n_splits=N_FOLDS)
    spat = run_cv(X, y, gkf.split(X, y, groups=blocks))

    def line(label, m):
        return (f"{label:18s}  AP {m['ap'].mean():.3f} ± {m['ap'].std():.3f}    "
                f"ROC-AUC {m['roc'].mean():.3f} ± {m['roc'].std():.3f}")

    print(f"{'(no-skill AP':18s}  AP {base_rate:.3f}            ROC-AUC 0.500)\n")
    print(line("RANDOM K-fold", rand), "   <-- looks great, but LEAKS")
    print(line("SPATIAL block CV", spat), "   <-- honest")

    gap_ap = rand["ap"].mean() - spat["ap"].mean()
    gap_roc = rand["roc"].mean() - spat["roc"].mean()
    print(f"\nLEAKAGE GAP:  AP {gap_ap:+.3f}   ROC-AUC {gap_roc:+.3f}")
    print("The spatial numbers are what we'd actually get drilling new ground.")

    # save for the figure / later phases
    np.savez(C.MODELS / "cv_scores.npz",
             rand_ap=rand["ap"], rand_roc=rand["roc"],
             spat_ap=spat["ap"], spat_roc=spat["roc"], base_rate=base_rate)


if __name__ == "__main__":
    main()
