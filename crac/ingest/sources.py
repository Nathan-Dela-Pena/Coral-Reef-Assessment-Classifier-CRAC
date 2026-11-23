"""Phase 1: raw fetchers. One function per external source.

Rules for this module:
  - Fetchers WRITE to data/raw/ and return the path they wrote. They do not clean.
  - Nothing here imports pandas or touches the database.
  - Every fetch is idempotent: if the file for (site, date range) already exists,
    return it instead of re-downloading, unless force=True.

Source notes:
  NOAA Coral Reef Watch -- the primary source, served via ERDDAP griddap as a
    plain URL query returning CSV for a lat/lon/time box. DHW and the bleaching
    alert level come from here.
  NASA Earthdata -- requires a free login and token; for ocean colour / chlorophyll.
  USGS + EPA -- both flow through the Water Quality Portal, one API for both.
"""

from __future__ import annotations

import logging
import math
import random
import time

import requests

from crac import config

log = logging.getLogger(__name__)

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv"
CRW_VARS = ["CRW_SST", "CRW_SSTANOMALY", "CRW_DHW", "CRW_BAA"]

RETRIES = 4
BACKOFF_SECONDS = 3.0

# ERDDAP extracts a point time series cell by cell, so request cost scales with
# the number of days asked for. Past roughly a month per request the server
# starts returning 408s instead of data, so fetches are chunked this small and
# the chunks are concatenated. A decade of history is therefore ~130 requests
# per site and takes a while -- it is a one-off, and the raw files are cached.
CHUNK_DAYS = 30

# 408 and 429 are the server saying "too slow / too often", not "bad request".
RETRYABLE_STATUS = {408, 429}


def _raw_path(source, site_id, start, end, suffix=".csv"):
    d = config.RAW_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{site_id}_{start}_{end}{suffix}"


def crw_url(site, start, end) -> str:
    """Build the ERDDAP griddap query for a single 5km grid cell over a date range."""
    lat, lon = site["lat"], site["lon"]
    box = "".join(
        f"[({start}T12:00:00Z):1:({end}T12:00:00Z)][({lat}):1:({lat})][({lon}):1:({lon})]"
        for _ in (0,)
    )
    return ERDDAP_BASE + "?" + ",".join(v + box for v in CRW_VARS)


def _get_with_retry(url, timeout=180):
    """GET with backoff on 5xx and network flakes. 4xx is our bug, so it raises."""
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:          # network flake: retry
            last_error = exc
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code < 500 and resp.status_code not in RETRYABLE_STATUS:
                resp.raise_for_status()                       # our fault: do not retry
            last_error = requests.HTTPError(f"{resp.status_code} from ERDDAP")
        if attempt < RETRIES:
            time.sleep(BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"ERDDAP request failed after {RETRIES} attempts: {url}") from last_error


def _chunks(start, end, days=CHUNK_DAYS):
    """Split a date range into pieces of at most `days` days."""
    import datetime as dt

    start, end = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
    out, cursor = [], start
    while cursor <= end:
        stop = min(cursor + dt.timedelta(days=days - 1), end)
        out.append((cursor, stop))
        cursor = stop + dt.timedelta(days=1)
    return out


def fetch_crw_timeseries(site, start, end, force=False):
    """Download NOAA Coral Reef Watch SST/DHW for one site.

    Args:
        site: a dict from sites.yaml (needs id, lat, lon).
        start, end: datetime.date bounds, inclusive.
        force: re-download even if the target file exists.

    Returns:
        Path to the written raw file under data/raw/crw/ -- the chunks
        concatenated, header kept once, otherwise unmodified.
    """
    target = _raw_path("crw", site["id"], start, end)
    if target.exists() and not force:
        log.info("crw: cached %s", target.name)
        return target

    parts = []
    chunks = _chunks(start, end)
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        resp = _get_with_retry(crw_url(site, chunk_start, chunk_end))
        text = resp.text
        if i > 1:
            # Drop the repeated header and units rows from every chunk but the first.
            text = "\n".join(text.splitlines()[2:]) + "\n"
        parts.append(text)
        if i % 10 == 0 or i == len(chunks):
            log.info("crw: %s %s..%s (%d/%d chunks)", site["id"], chunk_start, chunk_end,
                     i, len(chunks))

    target.write_text("".join(parts))
    log.info("crw: wrote %s", target.name)
    return target


def synthesize_crw_timeseries(site, start, end, force=False):
    """Offline stand-in for fetch_crw_timeseries, in the same CSV shape.

    NOAA's endpoint is not always reachable (no network in CI, ERDDAP outages),
    and the rest of the pipeline should still be demonstrable end to end. This
    writes a physically plausible series -- seasonal SST for the site's latitude,
    with occasional warm anomalies accumulating into DHW -- clearly tagged so it
    can never be mistaken for real data downstream.
    """
    import datetime as dt

    target = _raw_path("crw_synthetic", site["id"], start, end)
    if target.exists() and not force:
        return target

    start_d = dt.date.fromisoformat(str(start))
    end_d = dt.date.fromisoformat(str(end))
    rng = random.Random(abs(hash(site["id"])) % (2**31))

    # Warmer mean near the equator, bigger seasonal swing away from it.
    lat = site["lat"]
    mean_sst = 29.5 - abs(lat) * 0.16
    swing = 1.0 + abs(lat) * 0.09
    southern = lat < 0

    # CRW's own definition: the bleaching threshold is the maximum monthly mean
    # (MMM) + 1 C; HotSpot is how far SST sits above the MMM, and DHW is the sum
    # of HotSpots >= 1 C over a trailing 12 weeks, expressed in weeks.
    mmm = mean_sst + swing - 0.15

    lines = ["time,latitude,longitude,CRW_SST,CRW_SSTANOMALY,CRW_DHW,CRW_BAA",
             "UTC,degrees_north,degrees_east,Celsius,Celsius,Celsius_weeks,"]
    hotspots = []          # trailing 12 weeks of daily HotSpot values
    anomaly = 0.0
    heatwave_days = 0
    day = start_d
    while day <= end_d:
        doy = day.timetuple().tm_yday
        phase = (doy / 365.0) * 2 * math.pi + (math.pi if southern else 0.0)
        seasonal = mean_sst + swing * math.cos(phase - math.pi)

        # Marine heatwaves last weeks, not days, so the anomaly is persistent
        # (AR(1)) with occasional multi-week warm episodes layered on top. A
        # series of independent daily spikes would never accumulate any DHW.
        anomaly = 0.90 * anomaly + rng.gauss(0.0, 0.14)
        if heatwave_days > 0:
            anomaly += 0.022 * heatwave_days ** 0.5
            heatwave_days -= 1
        elif rng.random() < 0.0016:
            heatwave_days = rng.randint(25, 70)

        trend = 0.018 * (day.year - start_d.year)      # multi-year warming
        sst = seasonal + anomaly + trend

        hotspot = max(0.0, sst - mmm)
        hotspots.append(hotspot if hotspot >= 1.0 else 0.0)
        hotspots = hotspots[-84:]                      # 12 weeks of days
        dhw = sum(hotspots) / 7.0

        if dhw >= 8:
            alert = 4
        elif dhw >= 4:
            alert = 3
        elif dhw > 0:
            alert = 2
        elif hotspot > 0:
            alert = 1
        else:
            alert = 0

        lines.append(
            f"{day.isoformat()}T12:00:00Z,{site['lat']},{site['lon']},"
            f"{sst:.3f},{anomaly + trend:.3f},{dhw:.3f},{alert}"
        )
        day += dt.timedelta(days=1)

    target.write_text("\n".join(lines) + "\n")
    log.info("crw(synthetic): wrote %s (%d rows)", target.name, len(lines) - 2)
    return target


def fetch_earthdata_granules(site, start, end, force=False):
    """TODO (later): NASA Earthdata / CMR granule search + download."""
    raise NotImplementedError("Earthdata ocean-colour ingest is not part of Phase 1")


def fetch_water_quality(site, start, end, force=False):
    """TODO (later): Water Quality Portal, coastal sites only."""
    raise NotImplementedError("Water Quality Portal ingest is not part of Phase 1")
