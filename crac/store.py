"""The only module that talks to DuckDB.

Keeping all SQL here means you can swap the storage engine later by rewriting one
file, and it stops connection handling from leaking into the pipeline.

Tables:
    observations   -- environmental time series (Phase 1)
    classifications -- one row per classified image (Phase 2)
    events         -- detected bleaching / recovery episodes (Phase 3)
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    site_id      TEXT NOT NULL,
    date         DATE NOT NULL,
    sst          DOUBLE,
    sst_anomaly  DOUBLE,
    dhw          DOUBLE,
    alert_level  INTEGER,
    source       TEXT,
    PRIMARY KEY (site_id, date)
);

CREATE TABLE IF NOT EXISTS classifications (
    image_id     TEXT PRIMARY KEY,
    site_id      TEXT,
    captured_on  DATE,
    label        TEXT NOT NULL,
    confidence   DOUBLE,
    model_version TEXT NOT NULL,
    image_path   TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    site_id      TEXT NOT NULL,
    kind         TEXT NOT NULL,   -- 'bleaching' | 'recovery' | 'degradation'
    start_date   DATE NOT NULL,
    end_date     DATE,
    peak_dhw     DOUBLE,
    notes        TEXT
);
"""


def connect():
    """Open the DuckDB file, creating parent dirs and applying SCHEMA.

    TODO: duckdb.connect(str(config.DB_PATH)), then execute SCHEMA.
    """
    raise NotImplementedError


def upsert_observations(df):
    """Insert or replace rows keyed on (site_id, date). TODO."""
    raise NotImplementedError


def observations_for(site_id, start=None, end=None):
    """Return a DataFrame of observations, date-ordered. TODO."""
    raise NotImplementedError
