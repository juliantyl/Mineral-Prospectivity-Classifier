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

## Phase 2 — Feature engineering (rasters + points) ✅
- **Step 1 align** (`src/features/align.py`): all 7 grids resampled onto one 776x676 grid
  @ 250 m in MGA Zone 51. Method by direction: AVERAGE when coarsening (mag/rad ~100 m),
  BILINEAR when interpolating (gravity ~464 m). -> `data/interim/features_aligned.tif`.
- **Step 2 features** (`src/features/build_features.py`): 18 bands = 7 raw + 3 radiometric
  ratios (K/Th, U/Th, K/U) + focal mean/std (1.25 km window) on mag/grav/rad.
  -> `data/interim/features_enriched.tif`.
- **Step 3 labels** (`src/features/labels.py`): PU labelling. 1182 gold points -> 508 grid
  cells (250 m collapses dense camps) -> 2291 positives with 1-cell halo. Negatives =
  buffered (1 km) random background at 1:20. -> `data/processed/training_table.csv`
  (48,111 rows, 4.76% pos, 0 NaNs, cols: x, y_coord, label, 18 features).
- **Teaching notes:** ratio is artificial -> probabilities inflated -> Phase 4 calibration
  must correct to true base rate (ranking unaffected). Grid resolution sets effective
  positive count. QC figures 02 (aligned stack) + 03 (sample distribution).

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
