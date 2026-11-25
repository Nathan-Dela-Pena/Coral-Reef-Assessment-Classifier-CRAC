"""Phase 4: read-only JSON API over the processed tables.

Endpoints:
    GET  /sites                      -> list with lat/lon and latest alert_level
    GET  /sites/{id}/observations    -> time series for the timeline chart
    GET  /sites/{id}/health          -> class fractions over time
    GET  /sites/{id}/events          -> detected bleaching / recovery episodes
    GET  /footprint/context?tonnes=  -> framing numbers for the footprint panel
    GET  /model                      -> which model version is serving, and its metrics
    POST /classify                   -> classify one uploaded reef image

The read endpoints never compute. They read what the pipeline already wrote; if a
response needs a calculation, that calculation belongs in crac/tracking/ and its
result belongs in the database. /classify is the one exception, and it has to be:
inference on a user's upload cannot be precomputed.

Footprint note: this is context, not causation. You cannot attribute a given
reef's decline to one person's emissions, and claiming so is the easiest thing
for a reviewer to attack. Show the user's share of the global total and the
levers that actually matter.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from crac import config, store
from crac.ingest import pipeline

log = logging.getLogger(__name__)

app = FastAPI(
    title="CRAC",
    description="Coral Reef Assessment Classifier -- reef health classification and tracking.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
)

DASHBOARD_DIR = config.PROJECT_ROOT / "dashboard" / "src"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# Rate limit for /classify. The read endpoints are cheap table reads and are left
# alone; /classify runs a CNN forward pass, which is the only thing here expensive
# enough for a crawler to turn into a hosting bill. Override per deployment with
# CRAC_CLASSIFY_RATE / CRAC_CLASSIFY_WINDOW.
CLASSIFY_RATE = int(os.environ.get("CRAC_CLASSIFY_RATE", "20"))
CLASSIFY_WINDOW = float(os.environ.get("CRAC_CLASSIFY_WINDOW", "60"))


class SlidingWindowLimiter:
    """Per-client sliding window, in process memory.

    Deliberately dependency-free: slowapi would pull in limits+redis to solve a
    problem one deque per caller already solves at this scale.

    The limit is per worker process, not global. Two uvicorn workers, or two
    Cloud Run instances, each allow the full rate -- so this is a brake on
    casual abuse and runaway cost, not an authorization boundary. A real one
    needs shared state (Redis) or an edge limiter in front of the service.
    """

    def __init__(self, rate: int, window: float):
        self.rate = rate
        self.window = window
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """Return 0.0 if the call is allowed, else seconds until it would be."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.rate:
                return hits[0] + self.window - now
            hits.append(now)

            # Drop idle callers so a long-running process does not accumulate a
            # deque per IP it has ever seen.
            if len(self._hits) > 1024:
                for k in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
                    del self._hits[k]
            return 0.0


_classify_limiter = SlidingWindowLimiter(CLASSIFY_RATE, CLASSIFY_WINDOW)


def rate_limit_classify(request: Request):
    """429 when a caller exceeds the /classify budget.

    Behind a proxy (Cloud Run, HF Spaces) request.client.host is the proxy, so
    prefer the first hop in X-Forwarded-For. That header is caller-controlled and
    trivially spoofed -- acceptable for cost control, not for anything stronger.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    client = forwarded.split(",")[0].strip() or (request.client.host if request.client else "-")
    retry_after = _classify_limiter.check(client)
    if retry_after > 0:
        raise HTTPException(
            429,
            f"rate limit: {CLASSIFY_RATE} classifications per "
            f"{int(CLASSIFY_WINDOW)}s. Try again in {retry_after:.0f}s.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.5)))},
        )

_predictor = None
_predictor_error = None


def get_predictor():
    """Load the model once, lazily. A missing model must not break the read API."""
    global _predictor, _predictor_error
    if _predictor is None and _predictor_error is None:
        try:
            from crac.model import predict
            _predictor = predict.load()
            log.info("loaded model %s", _predictor.version)
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
            # A missing or unreadable model must not take down the read API.
            _predictor_error = str(exc)
            log.warning("no model available: %s", exc)
    return _predictor


def _sites_by_id():
    return {s["id"]: s for s in pipeline.load_sites()}


def _require_site(site_id):
    site = _sites_by_id().get(site_id)
    if site is None:
        raise HTTPException(404, f"unknown site '{site_id}'")
    return site


def _records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/health")
def service_health():
    """Liveness, plus whether a classifier is actually loaded."""
    predictor = get_predictor()
    return {"status": "ok", "model_loaded": predictor is not None,
            "model_version": predictor.version if predictor else None,
            "model_error": _predictor_error}


@app.get("/sites")
def list_sites():
    """Every tracked site with its most recent reading and its worst on record."""
    latest, peaks = {}, {}
    try:
        with store.connect() as con:
            for row in _records(store.latest_observations(con=con)):
                latest[row["site_id"]] = row
            for row in _records(store.peak_dhw_by_site(con=con)):
                peaks[row["site_id"]] = row
    except (OSError, RuntimeError, ValueError) as exc:
        # The site list is still useful without readings; say so rather than 500.
        log.warning("observations unavailable: %s", exc)

    out = []
    for site in pipeline.load_sites():
        obs = latest.get(site["id"], {})
        peak = peaks.get(site["id"], {})
        out.append({
            **site,
            "latest_date": obs.get("date"),
            "sst": obs.get("sst"),
            "sst_anomaly": obs.get("sst_anomaly"),
            "dhw": obs.get("dhw"),
            "alert_level": obs.get("alert_level"),
            "source": obs.get("source"),
            "peak_dhw": peak.get("peak_dhw"),
            "peak_date": peak.get("peak_date"),
        })
    return out


@app.get("/sites/{site_id}/observations")
def site_observations(site_id: str, start: str | None = None, end: str | None = None,
                      every: int = Query(1, ge=1, le=90,
                                         description="return every Nth day (thinning for charts)")):
    _require_site(site_id)
    with store.connect() as con:
        df = store.observations_for(site_id, start=start, end=end, con=con)
    if every > 1:
        df = df.iloc[::every]
    return {"site_id": site_id, "n": len(df), "observations": _records(df)}


@app.get("/sites/{site_id}/health")
def site_health(site_id: str, freq: str = "Q"):
    """Class fractions over time, from the stored per-image classifications."""
    from crac.tracking.trends import health_trajectory

    _require_site(site_id)
    with store.connect() as con:
        classifications = store.classifications_for(site_id, con=con)
    traj = health_trajectory(classifications, freq=freq)
    return {"site_id": site_id, "n_surveys": len(traj), "trajectory": _records(traj)}


@app.get("/sites/{site_id}/events")
def site_events(site_id: str):
    _require_site(site_id)
    with store.connect() as con:
        df = store.events_for(site_id, con=con)
    return {"site_id": site_id, "n": len(df), "events": _records(df)}


@app.get("/sites/{site_id}/correlation")
def site_correlation(site_id: str):
    """Does thermal stress at this site precede a drop in its healthy fraction?"""
    from crac.tracking.trends import correlate, health_trajectory

    _require_site(site_id)
    with store.connect() as con:
        obs = store.observations_for(site_id, con=con)
        traj = health_trajectory(store.classifications_for(site_id, con=con))
    return {"site_id": site_id, **correlate(obs, traj)}


@app.get("/model")
def model_info():
    """The serving model version and its honest metrics, straight from metrics.json."""
    from crac.model import predict

    version = None
    predictor = get_predictor()
    if predictor:
        version = predictor.version
    else:
        version = predict.latest_version()

    if version is None:
        return {"loaded": False, "version": None, "error": _predictor_error,
                "available_versions": predict.available_versions(), "metrics": None}

    metrics_path = Path(config.MODELS_DIR) / version / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return {
        "loaded": predictor is not None,
        "version": version,
        "labels": predictor.labels if predictor else None,
        "available_versions": predict.available_versions(),
        "metrics": metrics,
    }


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    _: None = Depends(rate_limit_classify),
):
    """Classify one uploaded reef image. Returns the label, confidence and the
    full probability vector -- the UI shows all four so a 40/35 split can never
    be read as a confident call.
    """
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(
            503,
            "No trained model is loaded. Train one with notebooks/train.ipynb and "
            f"place the exported folder in {config.MODELS_DIR}.",
        )

    blob = await file.read()
    if not blob:
        raise HTTPException(400, "empty upload")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    from PIL import Image, UnidentifiedImageError

    from crac.model.predict import predict_pil

    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "could not read that file as an image") from exc

    result = predict_pil(predictor, image)
    result["filename"] = file.filename
    return result


SAMPLE_DIR = config.RAW_DIR / "images"
SAMPLES_PER_LABEL = 1


@app.get("/samples")
def list_samples():
    """A couple of local images the dashboard can offer as one-click examples.

    Only exposes files under data/raw/images, by index -- never an arbitrary path.
    """
    from crac.model.dataset import IMAGE_SUFFIXES, LABELS

    out = []
    for label in LABELS:
        folder = SAMPLE_DIR / label
        if not folder.is_dir():
            continue
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        for i in range(min(SAMPLES_PER_LABEL, len(files))):
            out.append({"label": label, "index": i, "url": f"/samples/{label}/{i}"})
    return {"synthetic": (SAMPLE_DIR / "SYNTHETIC_DEMO_DATA").exists(), "samples": out}


@app.get("/samples/{label}/{index}")
def get_sample(label: str, index: int):
    from crac.model.dataset import IMAGE_SUFFIXES, LABELS

    if label not in LABELS:
        raise HTTPException(404, f"unknown label '{label}'")
    folder = SAMPLE_DIR / label
    if not folder.is_dir():
        raise HTTPException(404, "no sample images on disk")
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not 0 <= index < len(files):
        raise HTTPException(404, "no such sample")
    return FileResponse(files[index])


# Rough global figures, stated as rough on purpose. Sources in the README.
LEVERS = [
    {"lever": "Flying less",
     "note": "One long-haul return flight is roughly 1-2 t CO2."},
    {"lever": "Driving less / electrifying",
     "note": "A petrol car emits roughly 2 t CO2 per 10,000 km."},
    {"lever": "Grid electricity",
     "note": "Switching supply often beats every household change combined."},
    {"lever": "Diet",
     "note": "Shifting away from beef and lamb is the largest single food lever."},
    {"lever": "Voting and policy",
     "note": "Grid and transport policy move far more carbon than any household can."},
]

GLOBAL_ANNUAL_EMISSIONS_GT = 37.4      # Gt CO2, fossil + industry, recent year
WORLD_POPULATION_BN = 8.1
PARIS_ALIGNED_TONNES_PER_PERSON = 2.0  # rough per-capita budget for 1.5 C


@app.get("/footprint/context")
def footprint_context(tonnes: float = Query(..., ge=0, le=1000,
                                            description="your annual CO2 in tonnes")):
    """Framing numbers for the footprint panel. Context, not attribution."""
    global_tonnes = GLOBAL_ANNUAL_EMISSIONS_GT * 1e9
    share = tonnes / global_tonnes
    return {
        "tonnes": tonnes,
        "share_of_global_annual": share,
        "share_as_one_in": round(1 / share) if share else None,
        "vs_paris_aligned_budget": tonnes / PARIS_ALIGNED_TONNES_PER_PERSON,
        "paris_aligned_tonnes": PARIS_ALIGNED_TONNES_PER_PERSON,
        "global_annual_tonnes": global_tonnes,
        "levers": LEVERS,
        "caveat": (
            "No individual's emissions can be attributed to any particular reef's decline. "
            "This panel shows your share of the global total and where the leverage actually is; "
            "it is context for the thermal-stress data above, not a causal claim."
        ),
    }


# The dashboard is mounted last, at the root, so it does not shadow the API
# routes above -- Starlette matches routes in registration order. Mounting at "/"
# is what lets index.html reference `styles.css` and `app.js` as plain relative
# paths, which keeps the page openable from the filesystem too.
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


@app.exception_handler(FileNotFoundError)
def _missing_file(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})
