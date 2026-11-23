# CRAC — Coral Reef Assessment Classifier

Classifies the health of a coral reef from a photograph, and tracks thermal stress at
reef sites over time.

A ResNet-18 convolutional network sorts a reef image into **healthy**, **bleached**,
**dead**, or **algae_covered**. Alongside it, a pipeline pulls NOAA Coral Reef Watch
sea-surface temperature and Degree Heating Weeks for each tracked site, detects
bleaching episodes, and serves the whole thing through a local web dashboard.

```
photo ──▶ ResNet-18 ──▶ label + confidence ──┐
                                             ├──▶ DuckDB ──▶ FastAPI ──▶ dashboard
NOAA CRW ──▶ normalize ──▶ validate ─────────┘        │
                                                      └──▶ event detection (DHW ≥ 4)
```

---

## Quick start

```bash
pip install -e ".[model,api,dev]"

# 1. Environmental data. --offline generates a synthetic series instead of
#    calling NOAA, so the rest of the pipeline is demonstrable immediately.
python scripts/refresh_data.py --offline --start 2015-01-01 --end 2025-11-01

# 2. A model. Either train one in Colab (see below), or generate demo images
#    and train locally in about a minute:
python scripts/make_demo_images.py --per-class 90
python scripts/build_manifest.py data/raw/images -o data/processed/manifest.csv
python -c "from crac.model.train import train; train('data/processed/manifest.csv', epochs=6, out_version='resnet18-demo')"

# 3. Label the images and detect events
python scripts/classify_images.py data/raw/images

# 4. Serve
python scripts/serve.py        # http://127.0.0.1:8000
```

Tests: `pytest` (35 tests, all offline — no test touches the network).

---

## Training the model (Google Colab)

`notebooks/train.ipynb` is written to run on Colab's free T4. Open it there, set
**Runtime → Change runtime type → T4 GPU**, and run top to bottom. It clones this repo,
installs the package, fetches a dataset, trains, evaluates, and exports the model.

Three data sources, picked with one variable in the notebook:

| `DATA_SOURCE` | what it does |
|---|---|
| `"demo"` | generates synthetic images in the runtime. Proves the pipeline end to end. |
| `"drive"` | reads a labelled folder from your Google Drive |
| `"kaggle"` | downloads a Kaggle dataset (upload `kaggle.json` when prompted) |

Whatever the source, the images must be in one folder per label:

```
images/
  healthy/  bleached/  dead/  algae_covered/
```

Name files `site=<site_id>_<YYYY-MM-DD>_0001.jpg` where you can. Provenance in the
filename is what lets the split stay honest — see below.

The notebook writes a zip. Unzip it into `data/models/` in your checkout and restart
the API; the dashboard picks up the newest version automatically.

### What the training does

1. **Baseline first.** A majority-class baseline is scored before anything else.
   "The CNN got 82%" is meaningless without the number it had to beat.
2. **Grouped split.** `split_by_site()` assigns whole sites (or surveys/dives, when the
   manifest records them) to train/val/test. Frames from one dive are near-duplicates;
   a random split leaks them across the boundary and inflates test accuracy badly.
3. **Two-stage transfer learning.** Freeze the ImageNet backbone and train only the new
   4-way head, then unfreeze everything at a 10× lower learning rate.
4. **Weighted loss.** Class imbalance is guaranteed — far more healthy frames than dead
   ones — so the loss is inverse-frequency weighted and results are judged on per-class
   recall, not accuracy.

Every run writes `data/models/<version>/` containing `weights.pt`, `labels.json`,
`metrics.json`, `hyperparameters.json`, `confusion_matrix.png`, and the three split
manifests, so any two runs can be compared after the fact.

### Reading the metrics

The failure mode to watch for is high overall accuracy driven by the majority class
while recall on `bleached` — the class the whole project exists to detect — sits near
zero. The dashboard's model panel shows per-class precision/recall/F1 and the confusion
matrix for exactly this reason.

**A model trained on `make_demo_images.py` output is flagged as synthetic** in
`metrics.json` and carries a warning in the dashboard. Those images are procedural
textures with the right colour statistics; a CNN separates them perfectly and the score
means nothing about coral.

---

## The data pipeline

`python scripts/refresh_data.py [--site ID] [--start DATE] [--end DATE] [--force] [--offline]`

Fetches NOAA Coral Reef Watch (CoralTemp 5 km) via ERDDAP, one request per 30-day
chunk — ERDDAP extracts a point series cell by cell and returns 408s on longer spans.
Raw responses are cached under `data/raw/`, so a re-run is free. Expect a full decade
for five sites to take a while; it is a one-off.

Normalization produces one contract the rest of the project depends on:

| column | meaning |
|---|---|
| `site_id` | matches an id in `crac/ingest/sites.yaml` |
| `date` | one row per site per day |
| `sst` | sea-surface temperature, °C |
| `sst_anomaly` | °C above the local monthly mean |
| `dhw` | degree heating weeks |
| `alert_level` | NOAA bleaching alert, 0–4 |
| `source` | which fetcher produced the row (`crw` or `crw_synthetic`) |

`validate()` runs before every write and raises on duplicate keys, unknown sites,
implausible SST, or negative DHW. Missing values stay NULL — gaps are real information
and the tracking layer needs to see them.

### Event detection

NOAA's own alert scale is the rule set:

- **DHW ≥ 4** — significant bleaching likely (Alert Level 1)
- **DHW ≥ 8** — mortality likely (Alert Level 2)

An event starts the first day DHW crosses 4 and ends on the last day at or above it.
A dip below the line shorter than **14 days** does not end the event: DHW is itself a
12-week accumulator, so brief wobbles around the threshold are one episode, not two.
Fourteen days is two survey-scale weeks — long enough to exclude noise, short enough not
to merge two separate summers.

A **recovery** is a rise in the healthy fraction of classifications at a site, sustained
across at least two surveys after an event ends. One good quarter is a survey artefact.

---

## The dashboard

`python scripts/serve.py` serves both the API and a single-page dashboard at
`http://127.0.0.1:8000` — no build step, no CDN, no framework.

- **Classify** — drop a reef photo and get the label, the confidence, and the full
  probability vector. All four classes are always shown, so a 40/35 split can't be read
  as a confident call.
- **Tracked sites** — latest SST/DHW per site plus the worst DHW on record, and a
  dual-axis timeline with detected bleaching episodes shaded.
- **Model** — per-class metrics and the confusion matrix, straight from `metrics.json`.
- **Footprint** — your annual emissions against the global total.

### API

| endpoint | returns |
|---|---|
| `GET /sites` | sites with latest readings and peak DHW |
| `GET /sites/{id}/observations?every=N` | the time series, thinnable for charts |
| `GET /sites/{id}/health` | class fractions over time |
| `GET /sites/{id}/events` | detected bleaching / recovery episodes |
| `GET /sites/{id}/correlation` | lagged DHW vs healthy fraction |
| `GET /model` | serving version and its metrics |
| `POST /classify` | classify one uploaded image |
| `GET /footprint/context?tonnes=` | framing numbers |

Interactive docs at `/docs`. The read endpoints never compute — they return what the
pipeline already wrote. `/classify` is the one exception, because inference on a user's
upload cannot be precomputed.

---

## Honest limits

- **The classifier sees one frame.** It cannot distinguish recent bleaching from a
  naturally pale species, or a dead colony from one in shadow. It is a screening pass
  over a survey, not a verdict on a reef.
- **Metrics are only as good as the labels.** Train on a real labelled source before
  reporting anything: [CoralNet](https://coralnet.ucsd.edu/), the Moorea LTER MLC set,
  or NOAA NCRMP imagery.
- **The correlation is descriptive.** With a handful of sites, lagged DHW against the
  healthy fraction is a description, not an inference. The API says so in its own
  response.
- **The footprint panel is context, not attribution.** No individual's emissions can be
  tied to a particular reef's decline, and claiming otherwise is the easiest thing for a
  reviewer to attack. The panel shows your share of the global total and where the
  leverage actually is.
- **Synthetic data is labelled as such** everywhere it appears — in the `source` column,
  in `metrics.json`, and in the dashboard.

## Layout

```
crac/
  ingest/      sources.py (fetch) · normalize.py (schema) · pipeline.py (orchestrate)
  model/       dataset.py (taxonomy, grouped split) · train.py · predict.py · evaluate.py
  tracking/    trends.py (events, trajectories, correlation)
  api/         main.py (FastAPI)
  store.py     the only module that talks to DuckDB
  config.py    every path in the project
dashboard/src/ index.html · styles.css · app.js
notebooks/     train.ipynb (Colab)
scripts/       refresh_data · build_manifest · make_demo_images · classify_images · serve
tests/         offline, fixture-based
```

## Sources

NOAA Coral Reef Watch CoralTemp 5 km, served via ERDDAP. Global emissions figures in the
footprint panel are approximate (≈37.4 Gt CO₂/yr fossil + industry; ≈2 t/person for a
1.5 °C-aligned budget) and are used only for order-of-magnitude framing.

## License

MIT — see [LICENSE](LICENSE).
