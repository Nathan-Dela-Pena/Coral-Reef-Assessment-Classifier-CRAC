"""Storage round-trips, against a throwaway DuckDB file."""

import pandas as pd
import pytest

from crac import store


@pytest.fixture
def con(tmp_path):
    c = store.connect(tmp_path / "test.duckdb")
    yield c
    c.close()


def frame(dates, site="moorea", dhw=1.0):
    return pd.DataFrame({
        "site_id": site,
        "date": pd.to_datetime(dates).date,
        "sst": 29.0,
        "sst_anomaly": 0.5,
        "dhw": dhw,
        "alert_level": 1,
        "source": "test",
    })


def test_upsert_then_read_back(con):
    assert store.upsert_observations(frame(["2024-01-01", "2024-01-02"]), con=con) == 2
    df = store.observations_for("moorea", con=con)
    assert len(df) == 2
    assert list(df["date"].astype(str)) == ["2024-01-01", "2024-01-02"]


def test_upsert_replaces_rather_than_duplicating(con):
    store.upsert_observations(frame(["2024-01-01"], dhw=1.0), con=con)
    store.upsert_observations(frame(["2024-01-01"], dhw=7.5), con=con)
    df = store.observations_for("moorea", con=con)
    assert len(df) == 1
    assert df["dhw"].iloc[0] == 7.5


def test_observations_respect_date_bounds(con):
    store.upsert_observations(frame(["2024-01-01", "2024-02-01", "2024-03-01"]), con=con)
    df = store.observations_for("moorea", start="2024-01-15", end="2024-02-15", con=con)
    assert len(df) == 1


def test_empty_upsert_is_a_no_op(con):
    assert store.upsert_observations(pd.DataFrame(), con=con) == 0


def test_latest_observations_is_one_row_per_site(con):
    store.upsert_observations(frame(["2024-01-01", "2024-06-01"], site="moorea"), con=con)
    store.upsert_observations(frame(["2024-03-01"], site="palau"), con=con)
    latest = store.latest_observations(con=con)
    assert len(latest) == 2
    assert str(latest.set_index("site_id").loc["moorea", "date"])[:10] == "2024-06-01"


def test_classifications_round_trip(con):
    store.insert_classification({
        "image_id": "abc", "site_id": "moorea", "captured_on": "2024-02-02",
        "label": "bleached", "confidence": 0.81, "model_version": "test", "image_path": "x.jpg",
    }, con=con)
    df = store.classifications_for("moorea", con=con)
    assert len(df) == 1 and df["label"].iloc[0] == "bleached"


def test_replace_events_swaps_the_whole_set(con):
    rows = [{"event_id": "e1", "site_id": "moorea", "kind": "bleaching",
             "start_date": "2024-01-01", "end_date": "2024-01-10",
             "peak_dhw": 6.0, "notes": "first"}]
    store.replace_events("moorea", rows, con=con)
    store.replace_events("moorea", [{**rows[0], "event_id": "e2", "notes": "second"}], con=con)
    df = store.events_for("moorea", con=con)
    assert len(df) == 1 and df["event_id"].iloc[0] == "e2"
