"""Phase 3: turn per-day observations and per-image labels into a timeline.

Keep this simple and explainable. Rolling windows and threshold crossings, not
ARIMA -- you need to be able to justify every detected event to a reader.

NOAA's own alert scale is the natural rule set:
    DHW >= 4   -> significant bleaching likely  (Alert Level 1)
    DHW >= 8   -> mortality likely              (Alert Level 2)

Definitions, settled:
  - an EVENT starts on the first day DHW crosses the threshold, and ends on the
    last day at or above it. A dip below the threshold shorter than
    `min_gap_days` does not end the event -- DHW is itself a 12-week accumulator,
    so brief wobbles around the line are the same episode, not two. 14 days is
    the default: two survey-scale weeks, long enough to exclude noise and short
    enough not to merge two separate summers.
  - RECOVERY is a rise in the healthy fraction of classifications at a site,
    sustained across at least two surveys after an event ends.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from crac.model.dataset import UNHEALTHY

ALERT_1_DHW = 4.0    # significant bleaching likely
ALERT_2_DHW = 8.0    # mortality likely


def _as_date(value):
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    return pd.to_datetime(value).date()


def detect_thermal_events(observations: pd.DataFrame, threshold=ALERT_1_DHW,
                          min_gap_days=14) -> list:
    """Find bleaching episodes in one site's observation series.

    Returns rows shaped for the events table.
    """
    if observations is None or observations.empty:
        return []

    df = observations.dropna(subset=["dhw"]).sort_values("date")
    if df.empty:
        return []

    site_id = str(df["site_id"].iloc[0])
    over = df[df["dhw"] >= threshold]
    if over.empty:
        return []

    # Group consecutive over-threshold days, bridging gaps shorter than min_gap_days.
    dates = [_as_date(d) for d in over["date"]]
    spans, current = [], [dates[0], dates[0]]
    for day in dates[1:]:
        if (day - current[1]).days <= min_gap_days:
            current[1] = day
        else:
            spans.append(tuple(current))
            current = [day, day]
    spans.append(tuple(current))

    events = []
    for start, end in spans:
        window = df[[start <= _as_date(d) <= end for d in df["date"]]]
        peak = float(window["dhw"].max())
        severe = peak >= ALERT_2_DHW
        headline = ("Alert Level 2 -- mortality likely" if severe
                    else "Alert Level 1 -- significant bleaching likely")
        events.append({
            "event_id": f"{site_id}:{start.isoformat()}",
            "site_id": site_id,
            "kind": "bleaching",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "peak_dhw": peak,
            "notes": (f"{headline}; peak DHW {peak:.1f} "
                      f"over {(end - start).days + 1} days"),
        })
    return events


def health_trajectory(classifications: pd.DataFrame, freq="Q") -> pd.DataFrame:
    """Aggregate image labels per site into a class-fraction time series.

    Resampled to quarters -- surveys are sparse and irregular, so daily is noise.
    `n_images` is carried through on purpose: a 100% healthy bucket built from
    two images must not be plotted like one built from two hundred.
    """
    columns = ["site_id", "period", "n_images", "healthy_fraction"]
    if classifications is None or classifications.empty:
        return pd.DataFrame(columns=columns)

    df = classifications.dropna(subset=["captured_on"]).copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["captured_on"] = pd.to_datetime(df["captured_on"])
    df["period"] = df["captured_on"].dt.to_period(freq)
    df["is_healthy"] = ~df["label"].isin(UNHEALTHY)

    rows = []
    for (site_id, period), group in df.groupby(["site_id", "period"], observed=True):
        row = {
            "site_id": site_id,
            "period": str(period),
            "period_start": period.start_time.date().isoformat(),
            "n_images": len(group),
            "healthy_fraction": float(group["is_healthy"].mean()),
        }
        fractions = group["label"].value_counts(normalize=True)
        for label, frac in fractions.items():
            row[f"frac_{label}"] = float(frac)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["site_id", "period"]).reset_index(drop=True)


def detect_recovery(trajectory: pd.DataFrame, events, min_surveys=2,
                    min_rise=0.10) -> list:
    """Flag sustained rises in the healthy fraction after a thermal event ends.

    Requires the rise to hold across at least `min_surveys` consecutive buckets --
    a single good quarter is a survey artefact, not a recovery.
    """
    if trajectory is None or trajectory.empty or not events:
        return []

    out = []
    for event in events:
        end = _as_date(event["end_date"] or event["start_date"])
        site = trajectory[trajectory["site_id"] == event["site_id"]].copy()
        if site.empty:
            continue
        site["start"] = site["period_start"].map(_as_date)

        before = site[site["start"] <= end]
        after = site[site["start"] > end].head(min_surveys + 2)
        if before.empty or len(after) < min_surveys:
            continue

        baseline = float(before["healthy_fraction"].iloc[-1])
        run = after[after["healthy_fraction"] >= baseline + min_rise]
        if len(run) < min_surveys:
            continue

        out.append({
            "event_id": f"{event['site_id']}:recovery:{run['period_start'].iloc[0]}",
            "site_id": event["site_id"],
            "kind": "recovery",
            "start_date": run["period_start"].iloc[0],
            "end_date": run["period_start"].iloc[min(len(run), min_surveys) - 1],
            "peak_dhw": None,
            "notes": (f"healthy fraction rose from {baseline:.0%} to "
                      f"{float(run['healthy_fraction'].iloc[min(len(run), min_surveys) - 1]):.0%} "
                      f"across {len(run)} surveys after {event['end_date']}"),
        })
    return out


def correlate(observations: pd.DataFrame, trajectory: pd.DataFrame, max_lag_quarters=4):
    """The project's central question: does thermal stress at a site precede a
    drop in the healthy fraction there?

    Lags the environmental series against the trajectory and reports the
    association with its sample size. With a handful of sites this is
    descriptive, not inferential -- the write-up must say so.
    """
    empty = {"n": 0, "best_lag_quarters": None, "pearson_r": None,
             "note": "not enough paired observations to say anything"}
    if observations is None or observations.empty or trajectory is None or trajectory.empty:
        return empty

    obs = observations.dropna(subset=["dhw"]).copy()
    obs["date"] = pd.to_datetime(obs["date"])
    quarterly = (obs.set_index("date")["dhw"].resample("QE").max()
                 .rename("max_dhw").to_frame())
    quarterly["period"] = quarterly.index.to_period("Q").astype(str)

    traj = trajectory[["period", "healthy_fraction", "n_images"]].copy()
    merged_base = quarterly.merge(traj, on="period", how="inner")
    if len(merged_base) < 4:
        return {**empty, "n": len(merged_base)}

    best = None
    for lag in range(max_lag_quarters + 1):
        frame = quarterly.copy()
        frame["lagged"] = frame["max_dhw"].shift(lag)
        merged = frame.dropna(subset=["lagged"]).merge(traj, on="period", how="inner")
        if len(merged) < 4 or merged["lagged"].nunique() < 2:
            continue
        r = float(np.corrcoef(merged["lagged"], merged["healthy_fraction"])[0, 1])
        if np.isnan(r):
            continue
        if best is None or abs(r) > abs(best["pearson_r"]):
            best = {"n": len(merged), "best_lag_quarters": lag, "pearson_r": r}

    if best is None:
        return empty
    direction = "lower" if best["pearson_r"] < 0 else "higher"
    best["note"] = (
        f"with a {best['best_lag_quarters']}-quarter lag, higher peak DHW is associated with a "
        f"{direction} healthy fraction (r={best['pearson_r']:.2f}, n={best['n']} quarters). "
        "Descriptive only: a handful of sites cannot support a causal claim."
    )
    return best


def refresh_events(site_id, con=None, threshold=ALERT_1_DHW, min_gap_days=14):
    """Recompute and store the event set for one site. Detection is deterministic,
    so a full replace is correct and keeps the table from accumulating stale rows.
    """
    from crac import store

    own = con is None
    con = con or store.connect()
    try:
        obs = store.observations_for(site_id, con=con)
        events = detect_thermal_events(obs, threshold=threshold, min_gap_days=min_gap_days)
        traj = health_trajectory(store.classifications_for(site_id, con=con))
        events += detect_recovery(traj, events)
        store.replace_events(site_id, events, con=con)
        return events
    finally:
        if own:
            con.close()
