"""Central configuration: paths, study region, coordinate reference systems.

Keeping these in one place means every script agrees on *where* the study area is and
*which CRS* we compute distances in. Geoscience ML breaks in subtle ways when one step
uses lat/lon degrees and another assumes metres — so we pin both explicitly.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"

for _p in (RAW, INTERIM, PROCESSED, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Study region — Eastern Goldfields, Yilgarn Craton (WA orogenic gold)
# Bounding box in geographic coords (GDA94, lon/lat). Centred near Kalgoorlie.
# Start modest; widening the box later is a one-line change.
# ----------------------------------------------------------------------------
# (min_lon, min_lat, max_lon, max_lat)
BBOX_GEO = (120.5, -31.5, 122.5, -30.0)

# ----------------------------------------------------------------------------
# Coordinate reference systems
#   GEO  : GDA94 geographic (lon/lat degrees) — how most GA grids are distributed
#   PROJ : GDA94 / MGA Zone 51 (metres) — Kalgoorlie sits in Zone 51.
#          We do ALL distance/neighbourhood/spatial-CV work in this projected CRS.
# ----------------------------------------------------------------------------
CRS_GEO = "EPSG:4283"    # GDA94 lon/lat
CRS_PROJ = "EPSG:28351"  # GDA94 / MGA Zone 51 (metres)

# Target analysis resolution, in metres (projected CRS). 250 m is a sane start for
# regional prospectivity — fine enough to be useful, coarse enough to stay tractable.
TARGET_RES_M = 250

# Random seed for any sampling/splitting we control.
SEED = 42

# ----------------------------------------------------------------------------
# Geoscience Australia National Geophysical Grids — WCS endpoint + coverages.
# Service supports WCS 2.0.1 and returns clipped GeoTIFFs, so we only pull our bbox.
# GetCapabilities: https://services.ga.gov.au/gis/geophysical-grids/ows?SERVICE=WCS&REQUEST=GetCapabilities
# Each entry: short_name -> (WCS coverageId, human description).
# ----------------------------------------------------------------------------
WCS_GEOPHYS = "https://services.ga.gov.au/gis/geophysical-grids/ows"

# GA OZMIN mineral occurrences (positive labels). WFS returns GeoJSON in EPSG:4283.
WFS_EARTHRESOURCE = "https://services.ga.gov.au/gis/earthresource/wfs"
DEPOSITS_TYPENAME = "erl:MineralOccurrenceView"

COVERAGES = {
    "mag_1vd":  ("geophys__magmap_v7_2019_VRTP_1VD",          "Magnetic RTP first vertical derivative"),
    "mag_as":   ("geophys__magmap_v7_2019_VRTP_AS",           "Magnetic RTP analytic signal"),
    "rad_k":    ("geophys__radmap_v4_2019_filtered_pctk",     "Radiometric potassium (%)"),
    "rad_th":   ("geophys__radmap_v4_2019_filtered_ppmth",    "Radiometric thorium (ppm)"),
    "rad_u":    ("geophys__radmap_v4_2019_filtered_ppmu",     "Radiometric uranium (ppm)"),
    "grav_cba": ("geophys__2019_A4_CBA",                      "Complete Bouguer gravity anomaly"),
    "grav_1vd": ("geophys__Gravmap2019-grid-grv_cscba_1vd",   "Gravity CSCBA first vertical derivative"),
}
