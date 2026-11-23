"""Phase 1 orchestration: fetch -> normalize -> validate -> store.

This is the only thing scripts/refresh_data.py should need to call.
Keep it thin; all real work lives in sources.py, normalize.py, store.py.
"""

from __future__ import annotations

import datetime as dt
import logging

import yaml

from crac import config, store
from crac.ingest import normalize, sources

log = logging.getLogger(__name__)

DEFAULT_START = dt.date(2015, 1, 1)


def load_sites(path=None):
    """Read sites.yaml and return the list of site dicts."""
    path = path or config.SITES_PATH
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    sites = doc.get("sites") or []
    for s in sites:
        for key in ("id", "name", "lat", "lon"):
            if key not in s:
                raise ValueError(f"site {s!r} in {path} is missing '{key}'")
    return sites


def refresh(site_ids=None, start=None, end=None, force=False, offline=False):
    """Run the full ingest for the given sites (default: all).

    One dead source must not abort the run: failures are logged, the loop
    continues, and the summary reports them so the caller can exit non-zero.
    """
    sites = load_sites()
    known = {s["id"] for s in sites}
    if site_ids:
        unknown = sorted(set(site_ids) - known)
        if unknown:
            raise ValueError(f"unknown site ids: {unknown} (known: {sorted(known)})")
        sites = [s for s in sites if s["id"] in set(site_ids)]

    start = start or DEFAULT_START
    end = end or dt.date.today() - dt.timedelta(days=1)
    fetch = sources.synthesize_crw_timeseries if offline else sources.fetch_crw_timeseries

    written, failures = {}, {}
    con = store.connect()
    try:
        for site in sites:
            try:
                raw_path = fetch(site, start, end, force=force)
                df = normalize.normalize_crw(raw_path, site_id=site["id"])
                normalize.validate(df, known_site_ids=known)
                rows = store.upsert_observations(df, con=con)
                written[site["id"]] = rows
                log.info("%s: %d rows %s..%s", site["id"], rows,
                         df["date"].min(), df["date"].max())
            except Exception as exc:  # one dead source must not abort the whole run
                failures[site["id"]] = str(exc)
                log.error("%s: FAILED -- %s", site["id"], exc)
    finally:
        con.close()

    log.info("refresh complete: %d sites ok, %d failed", len(written), len(failures))
    return {"written": written, "failures": failures}
