"""Start testing here, not with the network calls.

tests/fixtures/crw_florida_keys_2024-07.csv is one real CRW response, saved
verbatim. Asserting that normalize_crw() turns it into exactly the observations
schema catches most of what actually breaks as the pipeline changes.

No test here hits NOAA. Fixtures only -- fast, offline, deterministic.
"""

from pathlib import Path

import pandas as pd
import pytest

from crac.ingest import normalize
from crac.ingest.normalize import SCHEMA_COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "crw_florida_keys_2024-07.csv"
KNOWN_SITES = ["florida_keys", "moorea"]


def good_frame():
    return pd.DataFrame({
        "site_id": ["florida_keys", "florida_keys"],
        "date": pd.to_datetime(["2024-07-01", "2024-07-02"]).date,
        "sst": [29.99, 30.2],
        "sst_anomaly": [0.93, 1.11],
        "dhw": [1.53, 1.53],
        "alert_level": [1, 1],
        "source": ["crw", "crw"],
    })


def test_normalize_crw_matches_schema():
    df = normalize.normalize_crw(FIXTURE, site_id="florida_keys")

    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df) > 0
    assert set(df["site_id"]) == {"florida_keys"}
    assert set(df["source"]) == {"crw"}

    # The ERDDAP units row must not survive as data.
    assert df["sst"].notna().all()
    assert df["sst"].between(20, 35).all()
    assert (df["dhw"] >= 0).all()
    assert df["alert_level"].between(0, 4).all()

    # One row per day, ordered.
    assert not df.duplicated(subset=["site_id", "date"]).any()
    assert list(df["date"]) == sorted(df["date"])


def test_normalize_crw_infers_source_for_synthetic_files(tmp_path):
    target = tmp_path / "florida_keys_synthetic_2024.csv"
    target.write_text(FIXTURE.read_text())
    df = normalize.normalize_crw(target, site_id="florida_keys")
    assert set(df["source"]) == {"crw_synthetic"}


def test_validate_accepts_a_good_frame():
    assert normalize.validate(good_frame(), known_site_ids=KNOWN_SITES) is not None


def test_validate_rejects_duplicate_site_date():
    df = pd.concat([good_frame(), good_frame().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        normalize.validate(df, known_site_ids=KNOWN_SITES)


def test_validate_rejects_unknown_site_id():
    df = good_frame()
    df.loc[0, "site_id"] = "atlantis"
    with pytest.raises(ValueError, match=r"not in sites\.yaml"):
        normalize.validate(df, known_site_ids=KNOWN_SITES)


def test_validate_rejects_implausible_sst():
    df = good_frame()
    df.loc[0, "sst"] = 310.0          # someone forgot to convert from Kelvin
    with pytest.raises(ValueError, match="plausible range"):
        normalize.validate(df, known_site_ids=KNOWN_SITES)


def test_validate_rejects_negative_dhw():
    df = good_frame()
    df.loc[1, "dhw"] = -0.5
    with pytest.raises(ValueError, match="negative dhw"):
        normalize.validate(df, known_site_ids=KNOWN_SITES)


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        normalize.validate(good_frame().drop(columns=["dhw"]), known_site_ids=KNOWN_SITES)


def test_validate_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        normalize.validate(good_frame().iloc[0:0], known_site_ids=KNOWN_SITES)
