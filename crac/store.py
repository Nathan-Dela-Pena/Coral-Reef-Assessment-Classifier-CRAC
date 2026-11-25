"""The only module that talks to DuckDB.

Keeping all SQL here means you can swap the storage engine later by rewriting one
file, and it stops connection handling from leaking into the pipeline.

Tables:
    observations    -- environmental time series (Phase 1)
    classifications -- one row per classified image (Phase 2)
    events          -- detected bleaching / recovery episodes (Phase 3)
"""

from __future__ import annotations

import duckdb
import pandas as pd

from crac import config

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

OBSERVATION_COLUMNS = [
    "site_id", "date", "sst", "sst_anomaly", "dhw", "alert_level", "source",
]


def connect(db_path=None):
    """Open the DuckDB file, creating parent dirs and applying SCHEMA."""
    path = config.DB_PATH if db_path is None else db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


def upsert_observations(df: pd.DataFrame, con=None) -> int:
    """Insert or replace rows keyed on (site_id, date). Returns rows written."""
    if df is None or df.empty:
        return 0
    own = con is None
    con = con or connect()
    try:
        frame = df[OBSERVATION_COLUMNS].copy()
        con.register("incoming", frame)
        # DuckDB has no UPSERT on a plain PK here, so delete-then-insert the keys.
        con.execute(
            "DELETE FROM observations o USING incoming i "
            "WHERE o.site_id = i.site_id AND o.date = i.date"
        )
        columns = ", ".join(OBSERVATION_COLUMNS)
        con.execute(f"INSERT INTO observations SELECT {columns} FROM incoming")
        con.unregister("incoming")
        return len(frame)
    finally:
        if own:
            con.close()


def observations_for(site_id, start=None, end=None, con=None) -> pd.DataFrame:
    """Return a DataFrame of observations, date-ordered."""
    own = con is None
    con = con or connect()
    try:
        sql = "SELECT * FROM observations WHERE site_id = ?"
        params = [site_id]
        if start is not None:
            sql += " AND date >= ?"
            params.append(str(start))
        if end is not None:
            sql += " AND date <= ?"
            params.append(str(end))
        return con.execute(sql + " ORDER BY date", params).fetchdf()
    finally:
        if own:
            con.close()


def latest_observations(con=None) -> pd.DataFrame:
    """One row per site: the most recent observation. Drives the site list."""
    own = con is None
    con = con or connect()
    try:
        return con.execute(
            "SELECT o.* FROM observations o JOIN ("
            "  SELECT site_id, max(date) AS date FROM observations GROUP BY site_id"
            ") m ON o.site_id = m.site_id AND o.date = m.date ORDER BY o.site_id"
        ).fetchdf()
    finally:
        if own:
            con.close()


def peak_dhw_by_site(con=None) -> pd.DataFrame:
    """Each site's worst DHW on record, and the day it happened.

    The cards need this because the latest reading is seasonal: a site that hit
    Alert Level 2 last summer reads 0.0 in midwinter, which looks like no history
    of thermal stress at all.
    """
    own = con is None
    con = con or connect()
    try:
        return con.execute(
            "SELECT o.site_id, o.dhw AS peak_dhw, o.date AS peak_date "
            "FROM observations o JOIN ("
            "  SELECT site_id, max(dhw) AS dhw FROM observations GROUP BY site_id"
            ") m ON o.site_id = m.site_id AND o.dhw = m.dhw "
            "QUALIFY row_number() OVER (PARTITION BY o.site_id ORDER BY o.date) = 1"
        ).fetchdf()
    finally:
        if own:
            con.close()


def insert_classification(row: dict, con=None) -> None:
    """Insert or replace one classified image."""
    own = con is None
    con = con or connect()
    try:
        con.execute("DELETE FROM classifications WHERE image_id = ?", [row["image_id"]])
        con.execute(
            "INSERT INTO classifications VALUES (?,?,?,?,?,?,?)",
            [row["image_id"], row.get("site_id"), row.get("captured_on"), row["label"],
             row.get("confidence"), row["model_version"], row.get("image_path")],
        )
    finally:
        if own:
            con.close()


def classifications_for(site_id=None, con=None) -> pd.DataFrame:
    own = con is None
    con = con or connect()
    try:
        if site_id is None:
            return con.execute("SELECT * FROM classifications ORDER BY captured_on").fetchdf()
        return con.execute(
            "SELECT * FROM classifications WHERE site_id = ? ORDER BY captured_on", [site_id]
        ).fetchdf()
    finally:
        if own:
            con.close()


def replace_events(site_id, rows, con=None) -> int:
    """Swap in a freshly detected event set for one site (detection is deterministic)."""
    own = con is None
    con = con or connect()
    try:
        con.execute("DELETE FROM events WHERE site_id = ?", [site_id])
        for r in rows:
            con.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                [r["event_id"], r["site_id"], r["kind"], r["start_date"], r.get("end_date"),
                 r.get("peak_dhw"), r.get("notes")],
            )
        return len(rows)
    finally:
        if own:
            con.close()


def events_for(site_id, con=None) -> pd.DataFrame:
    own = con is None
    con = con or connect()
    try:
        return con.execute(
            "SELECT * FROM events WHERE site_id = ? ORDER BY start_date", [site_id]
        ).fetchdf()
    finally:
        if own:
            con.close()
