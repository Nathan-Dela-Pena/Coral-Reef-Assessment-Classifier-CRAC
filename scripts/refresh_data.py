"""CLI entry point for Phase 1.

    python scripts/refresh_data.py --start 2015-01-01 --end 2025-01-01
    python scripts/refresh_data.py --site florida_keys --force
    python scripts/refresh_data.py --offline          # no network; synthetic series

Scripts in this directory are thin argparse wrappers. Logic lives in crac/.
"""

import argparse
import datetime as dt
import logging
import sys

from crac.ingest import pipeline


def main(argv=None):
    p = argparse.ArgumentParser(description="Refresh CRAC environmental observations.")
    p.add_argument("--site", action="append", dest="sites", help="repeatable site id")
    p.add_argument("--start", type=dt.date.fromisoformat)
    p.add_argument("--end", type=dt.date.fromisoformat)
    p.add_argument("--force", action="store_true", help="re-download cached raw files")
    p.add_argument("--offline", action="store_true",
                   help="generate a synthetic series instead of calling NOAA")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = pipeline.refresh(site_ids=args.sites, start=args.start, end=args.end,
                              force=args.force, offline=args.offline)

    for site_id, rows in result["written"].items():
        print(f"  ok      {site_id}: {rows} rows")
    for site_id, err in result["failures"].items():
        print(f"  FAILED  {site_id}: {err}", file=sys.stderr)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
