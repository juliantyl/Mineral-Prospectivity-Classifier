"""Phase 4, step 1 — baseline model: logistic regression vs the gradient booster.

Always establish a simple floor before trusting a complex model. Logistic regression is that
floor: linear, interpretable, fast. The question it answers is "how much does the GBM's
non-linearity actually buy us?" If the GBM barely beats the line, the complexity isn't earning
its keep.

Two things logistic regression needs that trees don't:
  * FEATURE SCALING. Coefficients and the L2 penalty depend on feature scale; a feature in
    ppm vs one in % would be penalised unequally. StandardScaler fixes this. Crucially it
    lives INSIDE the model pipeline, so it's fit on each fold's TRAIN data only -- fitting the
    scaler on all data first would leak test statistics into training.
  * class_weight='balanced' to stop the 1:20 imbalance from swamping the minority class.

Both models are scored with the SAME repeated spatial CV engine, so the comparison is fair.

Run:  python -m src.modeling.baseline
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.modeling.repeated_cv import load_xy, repeated_spatial_cv
from src.modeling.spatial_cv import new_model


def make_logreg():
    """Scaler + logistic regression as one pipeline (scaler fit on train fold only)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=C.SEED),
    )


def _summary(name, res):
    rm, ap, roc = res["repeat_means"], res["all_ap"], res["all_roc"]
    print(f"{name:22s} AP {rm.mean():.3f} ± {rm.std():.3f} (across repeats)   "
          f"fold-spread ±{ap.std():.3f}   ROC-AUC {roc.mean():.3f}")
    return rm.mean()


def main() -> None:
    df, X, y, feat = load_xy()
    xs, ys = df["x"].to_numpy(), df["y_coord"].to_numpy()
    base_rate = y.mean()
    print(f"{len(df)} rows | {len(feat)} features | base rate {base_rate:.3f}\n")

    print("Logistic regression (baseline)...")
    lr = repeated_spatial_cv(make_logreg, X, y, xs, ys, verbose=False)
    print("Gradient boosting (LightGBM)...")
    gb = repeated_spatial_cv(new_model, X, y, xs, ys, verbose=False)

    print("\n--- repeated spatial CV comparison ---")
    print(f"{'no-skill':22s} AP {base_rate:.3f}")
    lr_ap = _summary("logistic regression", lr)
    gb_ap = _summary("gradient boosting", gb)
    lift = (gb_ap - lr_ap) / lr_ap * 100
    print(f"\nGBM lifts AP by {lift:+.0f}% over the linear baseline "
          f"(both ~{gb_ap/base_rate:.1f}x / {lr_ap/base_rate:.1f}x no-skill).")

    np.savez(C.MODELS / "baseline_scores.npz",
             lr_means=lr["repeat_means"], gb_means=gb["repeat_means"],
             lr_ap=lr["all_ap"], gb_ap=gb["all_ap"], base_rate=base_rate)


if __name__ == "__main__":
    main()
