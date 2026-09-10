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


def normalize_crw(raw_path):
    """Parse one raw CRW file into a DataFrame matching the schema above.

    TODO: parse, rename columns to the schema, coerce units, drop rows with no
    date, and assert the frame has exactly the schema columns before returning.
    """
    raise NotImplementedError


def validate(df):
    """Raise if the frame violates the contract.

    Check: required columns present, no duplicate (site_id, date), site_id values
    all exist in sites.yaml, sst within a physically plausible range, dhw >= 0.

    Call this before every write. A pipeline that silently ingests garbage is
    worse than one that crashes.
    """
    raise NotImplementedError
