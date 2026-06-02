# Mineral Prospectivity Model — Yilgarn Craton (WA orogenic gold)

A classifier that ranks cells across a region by likelihood of gold mineralisation,
from public geophysics (magnetics, radiometrics, gravity) + known deposit locations.

This is a learning/portfolio project: the point is to do the *hard* parts of geoscience ML
properly — extreme class imbalance, spatial cross-validation, raster+point feature fusion,
calibrated gradient boosting, and honest uncertainty communication. See [ROADMAP.md](ROADMAP.md).

## Setup
```powershell
# Python 3.13, venv already created at .venv
.\.venv\Scripts\Activate.ps1
```

## Layout
```
data/raw         downloaded GA grids + deposit points (gitignored)
data/interim     aligned/clipped rasters
data/processed   labelled feature table (X, y)
src/config.py    paths, study bbox, CRS
src/data         acquisition
src/features     feature engineering
src/modeling     spatial CV, training, evaluation
outputs          figures + saved models
notebooks        exploration
```
