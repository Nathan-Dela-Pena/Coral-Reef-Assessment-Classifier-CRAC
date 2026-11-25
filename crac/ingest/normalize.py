"""Phase 1: turn raw source files into one uniform observations table.

The output schema is the contract the rest of the project depends on. Everything
downstream (tracking, API, dashboard) reads this shape and nothing else:

    site_id     TEXT     matches an id in sites.yaml
    date        DATE     one row per site per day
    sst         DOUBLE   sea surface temperature, degrees C
    sst_anomaly DOUBLE   degrees C above the local monthly mean
    dhw         DOUBLE   degree heating weeks
    alert_level INTEGER  NOAA bleaching alert, 0-4
    source      TEXT     which fetcher produced the row

Missing values stay NULL. Do not forward-fill here -- gaps are real information
and the tracking layer needs to know about them.
"""

from __future__ import annotations

import pandas as pd

SCHEMA_COLUMNS = [
    "site_id", "date", "sst", "sst_anomaly", "dhw", "alert_level", "source",
]

# ERDDAP column name -> our schema name.
CRW_RENAMES = {
    "time": "date",
    "CRW_SST": "sst",
    "CRW_SSTANOMALY": "sst_anomaly",
    "CRW_DHW": "dhw",
    "CRW_BAA": "alert_level",
}

SST_MIN, SST_MAX = -2.0, 40.0


def normalize_crw(raw_path, site_id=None, source=None) -> pd.DataFrame:
    """Parse one raw CRW file into a DataFrame matching the schema above."""
    raw = pd.read_csv(raw_path)

    # ERDDAP emits a units row directly under the header; it is not data.
    if len(raw) and str(raw.iloc[0].get("time", "")).strip() == "UTC":
        raw = raw.iloc[1:]

    if site_id is None:
        site_id = str(raw_path).rsplit("/", 1)[-1].split("_")[0]
    if source is None:
        source = "crw_synthetic" if "synthetic" in str(raw_path) else "crw"

    df = raw.rename(columns=CRW_RENAMES)
    missing = {"date", "sst"} - set(df.columns)
    if missing:
        raise ValueError(f"{raw_path}: raw file is missing {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date"])

    for col in ("sst", "sst_anomaly", "dhw"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["alert_level"] = pd.to_numeric(df.get("alert_level"), errors="coerce").astype("Int64")

    df["site_id"] = site_id
    df["source"] = source

    # ERDDAP can return several grid cells for one query; collapse to one row per day.
    df = (df[SCHEMA_COLUMNS]
          .groupby(["site_id", "date", "source"], as_index=False)
          .agg({"sst": "mean", "sst_anomaly": "mean", "dhw": "max", "alert_level": "max"}))

    df = df[SCHEMA_COLUMNS].sort_values("date").reset_index(drop=True)
    assert list(df.columns) == SCHEMA_COLUMNS, "normalize_crw broke the schema contract"
    return df


def validate(df: pd.DataFrame, known_site_ids=None) -> pd.DataFrame:
    """Raise if the frame violates the contract.

    A pipeline that silently ingests garbage is worse than one that crashes, so
    this is called before every write.
    """
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"observations frame is missing columns: {missing}")

    if df.empty:
        raise ValueError("observations frame is empty")

    dupes = df.duplicated(subset=["site_id", "date"])
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicate (site_id, date) rows")

    if known_site_ids is not None:
        unknown = sorted(set(df["site_id"]) - set(known_site_ids))
        if unknown:
            raise ValueError(f"site_id values not in sites.yaml: {unknown}")

    sst = df["sst"].dropna()
    if not sst.empty and (sst.min() < SST_MIN or sst.max() > SST_MAX):
        raise ValueError(f"sst outside plausible range [{SST_MIN}, {SST_MAX}]: "
                         f"min={sst.min():.2f} max={sst.max():.2f}")

    dhw = df["dhw"].dropna()
    if not dhw.empty and dhw.min() < 0:
        raise ValueError(f"negative dhw: min={dhw.min():.2f}")

    alert = df["alert_level"].dropna()
    if not alert.empty and (alert.min() < 0 or alert.max() > 4):
        raise ValueError("alert_level outside 0-4")

    return df
