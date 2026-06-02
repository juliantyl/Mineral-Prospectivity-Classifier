"""Phase 4, step 2 — calibration: make the probabilities mean something.

Two separate problems live here; keep them distinct:

  (A) CALIBRATION SHAPE. A model can rank well yet still output probabilities that don't match
      reality (says "0.6" for a group of cells where only 30% are positive). We fix the shape
      by mapping raw score -> observed frequency. Two maps:
        * sigmoid / Platt (logistic on the score): STRICTLY monotone -> ranking (AP) unchanged,
          but assumes an S-shaped distortion.
        * isotonic: flexible monotone step function -> usually the best-calibrated, but its flat
          steps + boundary clipping create ties that can slightly move AP.

  (B) BASE-RATE / PRIOR SHIFT. We trained on an artificial 1:20 mix (4.76% positive). The true
      field prevalence of gold is far lower. Calibration below makes probabilities consistent
      with the *training* prevalence, not the field. Converting to absolute field probabilities
      needs a known true base rate, which the PU setup does not give us. So we treat outputs as
      calibrated-within-sample prospectivity (great for ranking + relative confidence), and are
      explicit that they are NOT absolute drill-success probabilities.

Honesty about leakage: the calibrator must be fit on data the model never trained on, AND
spatially separated. We use a three-way split of BLOCKS: model-train / calibration / test,
all disjoint, rotated over outer spatial folds to pool honest out-of-fold predictions.

Run:  python -m src.modeling.calibrate
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold

from src import config as C
from src.modeling.repeated_cv import load_xy
from src.modeling.spatial_cv import BLOCK_SIZE_M, N_FOLDS, assign_blocks, new_model


def expected_calibration_error(y, p, n_bins=10) -> float:
    """Mean gap between predicted confidence and observed frequency, weighted by bin size."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e, n = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def main() -> None:
    df, X, y, feat = load_xy()
    xs, ys = df["x"].to_numpy(), df["y_coord"].to_numpy()
    blocks = assign_blocks(xs, ys, BLOCK_SIZE_M)

    raw = np.full(len(y), np.nan)
    iso_p = np.full(len(y), np.nan)
    sig_p = np.full(len(y), np.nan)
    fold_ap = {"raw": [], "sigmoid (Platt)": [], "isotonic": []}

    outer = GroupKFold(N_FOLDS)
    for tr_idx, te_idx in outer.split(X, y, groups=blocks):
        # inner split of TRAINING blocks: model-train vs calibration (disjoint blocks)
        tr_blocks = blocks[tr_idx]
        model_local, calib_local = next(GroupKFold(4).split(X[tr_idx], y[tr_idx], groups=tr_blocks))
        m_idx, c_idx = tr_idx[model_local], tr_idx[calib_local]

        model = new_model()
        model.fit(X[m_idx], y[m_idx])
        p_cal = model.predict_proba(X[c_idx])[:, 1]

        # fit BOTH calibrators on the calibration blocks (model never saw them)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y[c_idx])
        sig = LogisticRegression().fit(p_cal.reshape(-1, 1), y[c_idx])  # Platt / sigmoid

        p_te = model.predict_proba(X[te_idx])[:, 1]
        p_te_iso = iso.transform(p_te)
        p_te_sig = sig.predict_proba(p_te.reshape(-1, 1))[:, 1]
        raw[te_idx], iso_p[te_idx], sig_p[te_idx] = p_te, p_te_iso, p_te_sig

        # per-fold AP: the honest way to ask "did calibration change ranking?"
        # (each fold uses ONE calibrator, so a monotone map can't reshuffle within it)
        fold_ap["raw"].append(average_precision_score(y[te_idx], p_te))
        fold_ap["sigmoid (Platt)"].append(average_precision_score(y[te_idx], p_te_sig))
        fold_ap["isotonic"].append(average_precision_score(y[te_idx], p_te_iso))

    base = y.mean()
    print(f"training base rate (positive fraction): {base:.4f}\n")
    print(f"{'':16s}{'Brier':>10s}{'ECE':>10s}{'AP(per-fold)':>14s}")
    for name, p in [("raw", raw), ("sigmoid (Platt)", sig_p), ("isotonic", iso_p)]:
        ap = np.mean(fold_ap[name])
        print(f"{name:16s}{brier_score_loss(y,p):>10.4f}{expected_calibration_error(y,p):>10.4f}{ap:>14.4f}")
    print("\nBrier/ECE are pooled; AP is mean per-fold (each fold = one calibrator).")
    print("Sigmoid AP == raw (strictly monotone preserves ranking); isotonic dips a touch via ties.")

    # reliability diagram
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    for name, p, col in [("raw GBM", raw, "tomato"),
                         ("sigmoid-calibrated", sig_p, "steelblue"),
                         ("isotonic-calibrated", iso_p, "seagreen")]:
        fp, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, "o-", color=col, label=name)
    ax.axhline(base, ls="--", color="gray", lw=1, label=f"training base rate ({base:.3f})")
    ax.set_xlabel("mean predicted probability (per bin)")
    ax.set_ylabel("observed fraction positive (per bin)")
    ax.set_title("Reliability diagram (spatially honest out-of-fold)\nzoomed to populated range")
    # binned probabilities top out near the base-rate-driven range, so zoom there
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "08_calibration.png", dpi=120)
    print(f"\n[ok] saved {C.FIGURES / '08_calibration.png'}")


if __name__ == "__main__":
    main()
