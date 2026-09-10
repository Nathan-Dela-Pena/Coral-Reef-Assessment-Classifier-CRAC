"""CLI entry point for Phase 1.

    python scripts/refresh_data.py --start 2015-01-01 --end 2025-01-01
    python scripts/refresh_data.py --site florida_keys --force

Scripts in this directory are thin argparse wrappers. Logic lives in crac/.
"""

# TODO: argparse (--site repeatable, --start, --end, --force), then call
# crac.ingest.pipeline.refresh(). Exit non-zero if any site failed.
