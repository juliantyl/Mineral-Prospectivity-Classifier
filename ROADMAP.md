# Mineral Prospectivity Model — Roadmap

**Goal:** Rank cells across a region of the Yilgarn Craton (WA) by likelihood of orogenic
gold mineralisation, using public geophysics (magnetics, radiometrics, gravity) and known
deposit locations as labels.

**Framing that matters:** the model's job is to *rank drill targets*, not to "classify".
Positives (deposits) are vanishingly rare, every label cost a drill hole, and nearby cells
are spatially correlated. Those three facts drive almost every design decision below.

---

## Phase 0 — Setup ✅
- venv + geo/ML stack (rasterio, geopandas, xgboost, lightgbm, scikit-learn)
- project skeleton, config with region bbox + CRS

## Phase 1 — Data acquisition ✅
- GA national grids via WCS (bbox-clipped GeoTIFFs in `data/raw/`):
  magnetics (1VD, analytic signal), radiometrics (K/Th/U), gravity (Bouguer, 1VD).
  Download: `python -m src.data.download`
- Gold occurrences via GA OZMIN WFS: 1182 positives in bbox (`data/raw/gold_occurrences.gpkg`).
  Definition = any occurrence whose commodity mentions gold. Download: `python -m src.data.deposits`
- MINEDEX (WA state) deferred to a later iteration to cross-check label completeness.
- Verified alignment in `outputs/figures/01_data_overview.png` (deposits track magnetic structures).
- **Teaching note:** grids arrive in GDA94 lon/lat (EPSG:4283). Phase 2 reprojects to MGA Zone 51
  (metres) so "distance" and "neighbourhood" mean something physical. Gravity is far coarser than
  magnetics/radiometrics — resampling to a common grid is a real decision, not a formality.

## Phase 2 — Feature engineering (rasters + points)
- Resample all grids to a common resolution & grid (align rasters)
- Per-cell features: raw bands + derivatives (magnetic gradients, analytic signal),
  neighbourhood stats (focal mean/std), distance-to-feature (e.g. interpreted faults if available)
- Fuse point geochem into the grid (interpolate / nearest sample)
- Build the labelled table: every grid cell is a row; `y=1` if a deposit falls in it.
  **Teaching note:** how we choose negatives matters enormously (see Phase 3).

## Phase 3 — The hard parts (this is the project)
- **Class imbalance:** positives may be <0.1%. Accuracy is meaningless. We'll use
  precision-recall, average precision, and ranking metrics (e.g. success-rate / capture curves).
- **Negative selection:** "unlabelled ≠ negative." We'll treat this as positive-unlabelled-ish:
  sample background cells, keep a buffer around known deposits, and be explicit about the assumption.
- **Spatial cross-validation:** the trap. Random splits leak because train/test cells are
  neighbours. We'll use spatial *block* CV (and discuss buffered leave-one-out). We will
  *show* the gap between random-CV AUC (inflated) and spatial-CV AUC (honest).

## Phase 4 — Modeling
- Baseline: logistic regression (interpretable, sanity check)
- Gradient boosting: XGBoost / LightGBM with spatial CV for hyperparameters
- **Calibration:** raw GBM scores aren't probabilities; calibrate so a "0.8" means something
- Feature importance + partial dependence → does the model agree with known geology?

## Phase 5 — Validation, uncertainty & communication
- Prospectivity map (calibrated probability raster) + capture-efficiency curve
- Uncertainty: where is the model extrapolating vs interpolating? (feature-space distance)
- Honest write-up: what would I tell someone about to spend drilling money on this?

---

## Metrics we trust (and why)
- **Average Precision / PR-AUC** — robust under extreme imbalance.
- **Success-rate curve** ("if we drill the top X% of ranked cells, what fraction of known
  deposits do we capture?") — the metric an exploration geologist actually cares about.
- **Spatial-CV scores only.** Any metric from a random split is treated as a leakage diagnostic,
  not a result.
