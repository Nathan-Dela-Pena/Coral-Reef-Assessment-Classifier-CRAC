"""The /classify rate limiter.

crac.api.main imports fastapi, which is the `api` extra and is deliberately not
installed in CI (the suite is offline and does not need it), so skip rather than
fail when it is absent.
"""

import pytest

pytest.importorskip("fastapi")

from crac.api.main import SlidingWindowLimiter


def test_allows_up_to_the_rate_then_refuses():
    limiter = SlidingWindowLimiter(rate=3, window=60)
    assert [limiter.check("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.check("a") > 0


def test_callers_are_counted_separately():
    limiter = SlidingWindowLimiter(rate=1, window=60)
    assert limiter.check("a") == 0.0
    assert limiter.check("b") == 0.0, "one caller's budget must not spend another's"
    assert limiter.check("a") > 0


def test_window_slides(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("crac.api.main.time.monotonic", lambda: now[0])

    limiter = SlidingWindowLimiter(rate=2, window=10)
    assert limiter.check("a") == 0.0
    assert limiter.check("a") == 0.0
    assert limiter.check("a") == pytest.approx(10, abs=0.01)

    # Still inside the window: refused, with the wait shrinking as it ages.
    now[0] += 6
    assert limiter.check("a") == pytest.approx(4, abs=0.01)

    # Past it: the two old hits expire and the budget is whole again.
    now[0] += 5
    assert limiter.check("a") == 0.0


def test_idle_callers_are_evicted(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("crac.api.main.time.monotonic", lambda: now[0])

    limiter = SlidingWindowLimiter(rate=1, window=10)
    for i in range(1100):
        limiter.check(f"ip-{i}")
    now[0] += 11
    limiter.check("trigger-sweep")
    assert len(limiter._hits) < 1100, "stale per-caller deques must not accumulate"
