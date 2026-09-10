"""Phase 1: raw fetchers. One function per external source.

Rules for this module:
  - Fetchers WRITE to data/raw/ and return the path they wrote. They do not clean.
  - Nothing here imports pandas or touches the database.
  - Every fetch is idempotent: if the file for (site, date range) already exists,
    return it instead of re-downloading, unless force=True.

Source notes:
  NOAA Coral Reef Watch -- the primary source. Serve via ERDDAP, which exposes a
    plain URL query returning CSV or NetCDF for a lat/lon/time box. This is the
    one to build first; DHW and the bleaching alert level come from here.
  NASA Earthdata -- requires a free login and token; use for ocean color / chlorophyll.
  USGS + EPA -- both flow through the Water Quality Portal, one API for both.
    Only useful for coastal sites (Florida Keys), not open-ocean reefs.
"""


def fetch_crw_timeseries(site, start, end, force=False):
    """Download NOAA Coral Reef Watch SST/DHW for one site.

    Args:
        site: a dict from sites.yaml (needs id, lat, lon).
        start, end: datetime.date bounds, inclusive.
        force: re-download even if the target file exists.

    Returns:
        Path to the written raw file under data/raw/crw/.

    TODO: build the ERDDAP griddap URL, request it, write the response
    unmodified. Retry on 5xx with a backoff; let 4xx raise.
    """
    raise NotImplementedError


def fetch_earthdata_granules(site, start, end, force=False):
    """TODO (later): NASA Earthdata / CMR granule search + download."""
    raise NotImplementedError


def fetch_water_quality(site, start, end, force=False):
    """TODO (later): Water Quality Portal, coastal sites only."""
    raise NotImplementedError
