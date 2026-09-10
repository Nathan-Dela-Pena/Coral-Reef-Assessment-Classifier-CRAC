"""Phase 1 orchestration: fetch -> normalize -> validate -> store.

This is the only thing scripts/refresh_data.py should need to call.
Keep it thin; all real work lives in sources.py, normalize.py, store.py.
"""


def load_sites():
    """Read sites.yaml and return the list of site dicts. TODO."""
    raise NotImplementedError


def refresh(site_ids=None, start=None, end=None, force=False):
    """Run the full ingest for the given sites (default: all).

    For each site: fetch -> normalize -> validate -> upsert.

    Log per site what was fetched and how many rows landed. On failure, log and
    continue to the next site -- one dead source should not abort the run --
    then report a summary of failures at the end and exit non-zero if any.

    TODO.
    """
    raise NotImplementedError
