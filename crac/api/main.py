"""Phase 4: read-only JSON API over the processed tables.

Endpoints (build in this order):
    GET /sites                      -> list with lat/lon and latest alert_level
    GET /sites/{id}/observations    -> time series for the timeline chart
    GET /sites/{id}/health          -> class fractions over time
    GET /sites/{id}/events          -> detected bleaching / recovery episodes
    GET /footprint/context?tonnes=  -> framing numbers for the footprint panel

The API never computes. It reads what the pipeline already wrote. If a response
needs a calculation, that calculation belongs in crac/tracking/ and its result
belongs in the database.

Footprint note: present this as context, not causation. You cannot attribute a
given reef's decline to one person's emissions, and claiming so is the easiest
thing for a reviewer to attack. Show the user's share of the global total and
the levers that matter.
"""

# TODO: app = FastAPI(); wire the routes above against crac.store.
