"""Central paths and settings.

Every module resolves file locations through here so nothing hardcodes a path
relative to its own working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # exactly as downloaded; never edited by hand
INTERIM_DIR = DATA_DIR / "interim"   # partially cleaned, reproducible from raw
PROCESSED_DIR = DATA_DIR / "processed"  # analysis-ready tables

DB_PATH = PROCESSED_DIR / "crac.duckdb"

SITES_PATH = PROJECT_ROOT / "crac" / "ingest" / "sites.yaml"

MODELS_DIR = PROJECT_ROOT / "data" / "models"  # trained weights + metrics reports
