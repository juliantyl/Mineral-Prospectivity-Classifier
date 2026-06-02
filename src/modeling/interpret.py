"""Phase 4, step 3 — interpretation: does the model agree with known geology?

A prospectivity model that scores well but leans on nonsense features is a liability. So we
open it up two ways:

  PERMUTATION IMPORTANCE (which features the model NEEDS). For each feature we shuffle its
  values on the held-out spatial test fold and measure how much AP drops. Big drop = the model
  genuinely relies on it to generalise. We compute it on held-out blocks (not training data),
  so it reflects real predictive value, not memorisation. Averaged over the spatial folds.

  PARTIAL DEPENDENCE (the DIRECTION of each effect). How does predicted prospectivity move as a
  feature increases? Geology gives us priors: hydrothermal K (and K/Th) alteration should push
  prospectivity UP; magnetic/gravity texture (structural complexity) should push it UP. If the
  curves disagree with that, something is wrong.

We also print the logistic regression's standardised coefficients as an interpretable
cross-check (sign = direction, magnitude = strength on a common scale).

Run:  python -m src.modeling.interpret
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import GroupKFold

from src import config as C
from src.modeling.baseline import make_logreg
from src.modeling.repeated_cv import load_xy
from src.modeling.spatial_cv import BLOCK_SIZE_M, N_FOLDS, assign_blocks, new_model


def main() -> None:
    df, X, y, feat = load_xy()
    feat = np.array(feat)
    xs, ys = df["x"].to_numpy(), df["y_coord"].to_numpy()
    blocks = assign_blocks(xs, ys, BLOCK_SIZE_M)
    rng_seed = C.SEED

    # --- permutation importance on held-out spatial folds (GBM) ---
    imp = np.zeros(len(feat))
    for tr, te in GroupKFold(N_FOLDS).split(X, y, groups=blocks):
        model = new_model().fit(X[tr], y[tr])
        r = permutation_importance(model, X[te], y[te], scoring="average_precision",
                                   n_repeats=5, random_state=rng_seed)
        imp += r.importances_mean
    imp /= N_FOLDS
    order = np.argsort(imp)[::-1]

    print("Permutation importance (mean AP drop when shuffled), GBM:")
    for i in order:
        print(f"  {feat[i]:16s} {imp[i]:+.4f}")

    # --- logistic standardised coefficients (interpretable direction check) ---
    lr = make_logreg().fit(X, y)
    coef = lr.named_steps["logisticregression"].coef_[0]  # on standardised features
    print("\nLogistic standardised coefficients (sign = direction):")
    for i in np.argsort(np.abs(coef))[::-1]:
        print(f"  {feat[i]:16s} {coef[i]:+.3f}")

    # --- figure: importance + coefficients ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].barh(feat[order][::-1], imp[order][::-1], color="seagreen")
    ax[0].set_title("GBM permutation importance\n(AP drop on held-out blocks)")
    ax[0].set_xlabel("mean AP decrease")
    co = np.argsort(coef)
    cols = ["tomato" if c < 0 else "steelblue" for c in coef[co]]
    ax[1].barh(feat[co], coef[co], color=cols)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_title("Logistic standardised coefficients\n(blue=raises, red=lowers prospectivity)")
    ax[1].set_xlabel("coefficient")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "09_importance.png", dpi=120)
    print(f"\n[ok] saved {C.FIGURES / '09_importance.png'}")

    # --- partial dependence for the top GBM features ---
    top = list(order[:6])
    model_full = new_model().fit(X, y)
    fig2, ax2 = plt.subplots(2, 3, figsize=(15, 8))
    PartialDependenceDisplay.from_estimator(
        model_full, X, features=top, feature_names=list(feat),
        ax=ax2.ravel()[: len(top)], kind="average",
    )
    fig2.suptitle("Partial dependence — how predicted prospectivity moves with each feature", y=1.02)
    fig2.tight_layout()
    fig2.savefig(C.FIGURES / "10_partial_dependence.png", dpi=120, bbox_inches="tight")
    print(f"[ok] saved {C.FIGURES / '10_partial_dependence.png'}")


if __name__ == "__main__":
    main()
