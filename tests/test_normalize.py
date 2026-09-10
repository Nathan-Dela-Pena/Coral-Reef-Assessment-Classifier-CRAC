"""Start testing here, not with the network calls.

Save one small real CRW response into tests/fixtures/ and assert that
normalize_crw() turns it into exactly the observations schema. That single test
catches most of what will actually break as you iterate.

Do not write tests that hit NOAA. Fixtures only -- fast, offline, deterministic.
"""

import pytest


@pytest.mark.skip(reason="TODO: add a real CRW fixture and implement normalize_crw")
def test_normalize_crw_matches_schema():
    ...


@pytest.mark.skip(reason="TODO")
def test_validate_rejects_duplicate_site_date():
    ...


@pytest.mark.skip(reason="TODO")
def test_validate_rejects_unknown_site_id():
    ...
