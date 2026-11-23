"""Phase 3 rules, tested on hand-built series.

Every assertion here corresponds to a definition written down in
crac/tracking/trends.py. If a threshold changes, this file should fail.
"""

import pandas as pd
import pytest

from crac.tracking import trends


def series(dhws, start="2024-01-01", site="florida_keys"):
    dates = pd.date_range(start, periods=len(dhws), freq="D").date
    return pd.DataFrame({
        "site_id": site,
        "date": dates,
        "sst": [30.0] * len(dhws),
        "sst_anomaly": [1.0] * len(dhws),
        "dhw": dhws,
        "alert_level": [3 if d >= 4 else 0 for d in dhws],
        "source": "test",
    })


def test_no_event_below_threshold():
    assert trends.detect_thermal_events(series([0, 1, 2, 3.9, 2, 0])) == []


def test_single_event_spans_the_crossing():
    events = trends.detect_thermal_events(series([0, 4.0, 6.0, 5.0, 1.0, 0]))
    assert len(events) == 1
    assert events[0]["start_date"] == "2024-01-02"
    assert events[0]["end_date"] == "2024-01-04"
    assert events[0]["peak_dhw"] == 6.0
    assert events[0]["kind"] == "bleaching"


def test_short_dip_does_not_split_one_event():
    # Five days under the line is shorter than min_gap_days, so it stays one episode.
    dhws = [5.0] * 3 + [2.0] * 5 + [5.0] * 3
    events = trends.detect_thermal_events(series(dhws))
    assert len(events) == 1
    assert events[0]["end_date"] == "2024-01-11"


def test_long_gap_splits_into_two_events():
    dhws = [5.0] * 3 + [0.0] * 40 + [5.0] * 3
    events = trends.detect_thermal_events(series(dhws))
    assert len(events) == 2


def test_alert_level_2_is_called_out_in_the_notes():
    events = trends.detect_thermal_events(series([9.0, 9.5]))
    assert "mortality likely" in events[0]["notes"]


def test_empty_input_is_not_an_error():
    assert trends.detect_thermal_events(pd.DataFrame()) == []


def classifications(rows):
    return pd.DataFrame(
        [{"image_id": f"i{i}", "site_id": site, "captured_on": date, "label": label,
          "confidence": 0.9, "model_version": "test", "image_path": ""}
         for i, (site, date, label) in enumerate(rows)]
    )


def test_health_trajectory_reports_fraction_and_count():
    traj = trends.health_trajectory(classifications([
        ("moorea", "2024-01-10", "healthy"),
        ("moorea", "2024-02-10", "healthy"),
        ("moorea", "2024-03-10", "bleached"),
        ("moorea", "2024-05-10", "dead"),
    ]))
    first = traj.iloc[0]
    assert first["n_images"] == 3
    assert first["healthy_fraction"] == pytest.approx(2 / 3)
    assert traj.iloc[1]["healthy_fraction"] == 0.0


def test_health_trajectory_handles_no_classifications():
    assert trends.health_trajectory(pd.DataFrame()).empty


def test_correlate_reports_sample_size_when_too_small():
    result = trends.correlate(series([1, 2, 3]), trends.health_trajectory(pd.DataFrame()))
    assert result["n"] == 0
    assert result["pearson_r"] is None
