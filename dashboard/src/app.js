/* CRAC dashboard.
 *
 * Talks to the FastAPI service that serves this page. No framework, no build
 * step, no CDN -- the whole app has to run from a laptop with the API on
 * localhost and nothing else available.
 */
"use strict";

const API = location.origin;

const LABEL_TEXT = {
  healthy:       "Living coral, no bleaching visible",
  bleached:      "Bleaching — the colony has expelled its algae and is under stress",
  dead:          "Dead coral or rubble",
  algae_covered: "Algae-dominated — typical of a reef that has not recovered",
};

const ALERT_TEXT = {
  0: ["No stress",          "#8fe3f7"],
  1: ["Watch",              "#8fe3f7"],
  2: ["Warning",            "#ffd27f"],
  3: ["Alert 1 — bleaching likely", "#ffb547"],
  4: ["Alert 2 — mortality likely", "#ff6b6b"],
};

const $ = (id) => document.getElementById(id);
const fmtPct = (x) => `${(x * 100).toFixed(1)}%`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function getJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON error */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ── Model banner + model card ──────────────────────────────── */

async function loadModel() {
  const banner = $("model-banner");
  const card = $("model-card");
  let info;
  try {
    info = await getJSON("/model");
  } catch (err) {
    banner.textContent = `API unreachable — is the server running? (${err.message})`;
    banner.dataset.state = "warn";
    card.innerHTML = `<p class="muted">Could not reach the API.</p>`;
    return;
  }

  if (!info.version) {
    banner.textContent = "No trained model loaded — train one with notebooks/train.ipynb";
    banner.dataset.state = "warn";
    card.innerHTML = `<p class="muted">No model in <code>data/models/</code> yet. Run the
      training notebook, then unzip the exported folder there and restart the API.</p>`;
    return;
  }

  banner.textContent = `model ${info.version} · ${info.loaded ? "loaded" : "on disk, not loaded"}`;
  renderModelCard(card, info);
}

function renderModelCard(el, info) {
  const m = info.metrics;
  if (!m) {
    el.innerHTML = `<p class="muted">Model <code>${esc(info.version)}</code> has no
      <code>metrics.json</code> — it was not evaluated on a held-out split.</p>`;
    return;
  }

  const baseline = m.baselines?.majority_class;
  const rows = Object.entries(m.per_class).map(([label, v]) => `
    <tr>
      <td>${esc(label)}</td>
      <td>${v.precision.toFixed(3)}</td>
      <td>${v.recall.toFixed(3)}</td>
      <td>${v["f1-score"].toFixed(3)}</td>
      <td>${v.support}</td>
    </tr>`).join("");

  // Row/column counts, so the matrix is readable without a separate legend.
  const cm = m.confusion_matrix;
  const matrix = `
    <div class="table-wrap"><table class="metrics">
      <thead><tr><th>true \\ predicted</th>${cm.labels.map((l) => `<th>${esc(l)}</th>`).join("")}</tr></thead>
      <tbody>${cm.counts.map((row, i) =>
        `<tr><td>${esc(cm.labels[i])}</td>${row.map((n, j) =>
          `<td${i === j ? ' style="color:var(--healthy)"' : ""}>${n}</td>`).join("")}</tr>`).join("")}
      </tbody>
    </table></div>`;

  const synthetic = m.synthetic_demo_data
    ? `<p class="caveat" style="margin:0 0 1.2rem">
        <strong>This model was trained on synthetic demo images.</strong> The scores below
        describe procedurally generated textures, not coral. Retrain on a real labelled
        source (CoralNet, Moorea LTER, NOAA NCRMP) before quoting any of these numbers.
       </p>`
    : "";

  el.innerHTML = `
    ${synthetic}
    <div class="metric-row">
      <div class="metric"><b>${fmtPct(m.accuracy)}</b><span>accuracy (test split)</span></div>
      <div class="metric"><b>${m.macro_f1.toFixed(3)}</b><span>macro F1</span></div>
      ${baseline ? `<div class="metric"><b>${fmtPct(baseline.accuracy)}</b>
        <span>majority-class baseline</span></div>` : ""}
      <div class="metric"><b>${m.n_test_images}</b><span>held-out images</span></div>
    </div>
    <div class="table-wrap"><table class="metrics">
      <thead><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>n</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <h3 style="margin-top:1.6rem;font-size:1rem">Confusion matrix</h3>
    ${matrix}
    <p class="muted" style="margin-top:1rem;font-size:.84rem">
      Model <code>${esc(info.version)}</code>${m.test_sites?.length
        ? ` · test sites: ${m.test_sites.map(esc).join(", ")}` : ""}.
      The split is grouped by site, so no survey appears in both training and test.
    </p>`;
}

/* ── Classifier ─────────────────────────────────────────────── */

function showError(message) {
  $("result-idle").hidden = true;
  $("result-body").hidden = true;
  const box = $("result-error");
  box.hidden = false;
  box.textContent = message;
}

function showResult(result) {
  $("result-idle").hidden = true;
  $("result-error").hidden = true;
  $("result-body").hidden = false;

  const verdict = $("verdict");
  verdict.dataset.health = result.healthy ? "healthy" : "unhealthy";
  $("verdict-label").textContent = result.label.replace("_", " ");
  $("verdict-sub").textContent =
    `${fmtPct(result.confidence)} confidence · ${LABEL_TEXT[result.label] ?? ""}`;

  const entries = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  $("bars").innerHTML = entries.map(([label, p], i) => `
    <li class="${i === 0 ? "is-top" : ""}">
      <span class="name">${esc(label.replace("_", " "))}</span>
      <span class="track"><span class="fill fill--${esc(label)}" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="pct">${(p * 100).toFixed(1)}%</span>
    </li>`).join("");

  const spread = entries[0][1] - entries[1][1];
  $("result-meta").innerHTML =
    `model ${esc(result.model_version)}${result.filename ? ` · ${esc(result.filename)}` : ""}` +
    (spread < 0.2
      ? `<br><span style="color:var(--warn)">top two classes are within
         ${fmtPct(spread)} — treat this as undecided</span>`
      : "");
}

async function classifyBlob(blob, filename) {
  const zone = $("dropzone");
  zone.classList.add("is-busy");
  $("result-idle").hidden = true;
  $("result-error").hidden = true;
  $("result-body").hidden = true;
  $("result").insertAdjacentHTML("beforeend", '<div class="spinner" id="spin"></div>');

  const form = new FormData();
  form.append("file", blob, filename || "upload.jpg");
  try {
    const res = await fetch(API + "/classify", { method: "POST", body: form });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || res.statusText);
    showResult(payload);
  } catch (err) {
    showError(err.message);
  } finally {
    $("spin")?.remove();
    zone.classList.remove("is-busy");
  }
}

function previewFile(file) {
  const img = $("preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("dropzone-prompt").style.display = "none";
}

function wireClassifier() {
  const zone = $("dropzone");
  const input = $("file-input");

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) { previewFile(file); classifyBlob(file, file.name); }
  });

  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, () => zone.classList.remove("is-over")));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) return showError("That file is not an image.");
    previewFile(file);
    classifyBlob(file, file.name);
  });
}

async function loadSamples() {
  let data;
  try {
    data = await getJSON("/samples");
  } catch { return; }
  if (!data.samples?.length) return;

  $("samples").hidden = false;
  $("samples-row").innerHTML = data.samples.map((s) =>
    `<button type="button" data-label="${esc(s.label)}" data-url="${esc(s.url)}">${
      esc(s.label.replace("_", " "))}</button>`).join("");

  const runSample = async (btn) => {
    const res = await fetch(API + btn.dataset.url);
    const blob = await res.blob();
    const img = $("preview");
    img.src = URL.createObjectURL(blob);
    img.hidden = false;
    $("dropzone-prompt").style.display = "none";
    classifyBlob(blob, btn.dataset.url.split("/").slice(-2).join("-") + ".jpg");
  };

  $("samples-row").querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => runSample(btn)));

  // #sample=bleached runs that example on load, so a demo is linkable.
  const wanted = new URLSearchParams(location.hash.slice(1)).get("sample");
  const match = wanted && $("samples-row").querySelector(`[data-label="${CSS.escape(wanted)}"]`);
  if (match) runSample(match);

  if (data.synthetic) {
    $("samples").insertAdjacentHTML("beforeend",
      '<span class="samples__note">samples are synthetic demo images</span>');
  }
}

/* ── Sites ──────────────────────────────────────────────────── */

let SITES = [];

async function loadSites() {
  const grid = $("site-grid");
  try {
    SITES = await getJSON("/sites");
  } catch (err) {
    grid.innerHTML = `<p class="muted">Could not load sites: ${esc(err.message)}</p>`;
    return;
  }
  if (!SITES.length) { grid.innerHTML = `<p class="muted">No sites configured.</p>`; return; }

  grid.innerHTML = SITES.map((s) => {
    const level = s.alert_level ?? null;
    const [text, colour] = ALERT_TEXT[level] ?? ["No data", "#a9cfe2"];
    const hasData = s.latest_date != null;
    return `
      <button class="site-card" type="button" aria-pressed="false"
              data-site="${esc(s.id)}" style="--alert:${colour}">
        <h3>${esc(s.name)}</h3>
        <span class="region">${esc(s.region ?? "")}</span>
        <div class="readings">
          <span>SST<b>${hasData && s.sst != null ? s.sst.toFixed(1) + "°" : "—"}</b></span>
          <span>DHW now<b>${hasData && s.dhw != null ? s.dhw.toFixed(1) : "—"}</b></span>
          <span title="worst degree heating weeks on record at this site">
            Peak DHW<b style="${(s.peak_dhw ?? 0) >= 8 ? "color:var(--alarm)"
              : (s.peak_dhw ?? 0) >= 4 ? "color:var(--warn)" : ""}">${
              s.peak_dhw != null ? s.peak_dhw.toFixed(1) : "—"}</b></span>
        </div>
        <span class="alert-tag">${esc(text)}</span>
        ${hasData ? `<span class="region" style="margin-top:.4rem">as of ${esc(String(s.latest_date).slice(0, 10))}${
          s.peak_date ? ` · peak ${esc(String(s.peak_date).slice(0, 7))}` : ""}</span>`
                  : `<span class="region" style="margin-top:.4rem">no observations ingested</span>`}
      </button>`;
  }).join("");

  grid.querySelectorAll(".site-card").forEach((btn) =>
    btn.addEventListener("click", () => selectSite(btn.dataset.site)));

  // #site=moorea opens that site directly, so a particular reef is linkable.
  const deepLink = new URLSearchParams(location.hash.slice(1)).get("site");
  if (deepLink && SITES.some((s) => s.id === deepLink)) selectSite(deepLink);
}

async function selectSite(siteId) {
  const site = SITES.find((s) => s.id === siteId);
  if (!site) return;

  document.querySelectorAll(".site-card").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.site === siteId)));
  history.replaceState(null, "", `#site=${encodeURIComponent(siteId)}`);

  const detail = $("site-detail");
  detail.hidden = false;
  $("detail-name").textContent = site.name;
  $("chart").innerHTML = '<div class="spinner"></div>';
  $("events").innerHTML = "";

  try {
    // Thin to roughly weekly: a decade of daily points is far more than the
    // chart can resolve, and the shape is identical.
    const [obs, events] = await Promise.all([
      getJSON(`/sites/${encodeURIComponent(siteId)}/observations?every=7`),
      getJSON(`/sites/${encodeURIComponent(siteId)}/events`),
    ]);
    if (!obs.observations.length) {
      $("chart").innerHTML = `<p class="muted">No observations stored for this site yet.
        Run <code>python scripts/refresh_data.py</code> to ingest NOAA data.</p>`;
      return;
    }
    $("chart").innerHTML = buildChart(obs.observations, events.events);
    renderEvents(events.events);
  } catch (err) {
    $("chart").innerHTML = `<p class="muted">${esc(err.message)}</p>`;
  }
}

function renderEvents(events) {
  const box = $("events");
  if (!events.length) {
    box.innerHTML = `<p class="muted" style="font-size:.86rem">No thermal event crossed
      DHW&nbsp;4 in the stored record for this site.</p>`;
    return;
  }
  box.innerHTML = events.map((e) => {
    const severe = e.kind === "bleaching" && (e.peak_dhw ?? 0) >= 8;
    const cls = e.kind === "recovery" ? "event--recovery" : severe ? "event--severe" : "";
    const end = e.end_date ? String(e.end_date).slice(0, 10) : "ongoing";
    return `<div class="event ${cls}">
      <time>${esc(String(e.start_date).slice(0, 10))} → ${esc(end)}</time>
      <span>${esc(e.notes ?? e.kind)}</span>
    </div>`;
  }).join("");
}

/* ── Chart: hand-rolled SVG, dual axis (SST left, DHW right) ── */

function buildChart(obs, events) {
  const W = 920, H = 320;
  const pad = { t: 16, r: 46, b: 28, l: 44 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;

  const times = obs.map((o) => Date.parse(o.date));
  const t0 = Math.min(...times), t1 = Math.max(...times);
  const x = (t) => pad.l + ((t - t0) / Math.max(t1 - t0, 1)) * iw;

  const ssts = obs.map((o) => o.sst).filter((v) => v != null);
  const sMin = Math.floor(Math.min(...ssts) - 0.5), sMax = Math.ceil(Math.max(...ssts) + 0.5);
  const ySST = (v) => pad.t + ih - ((v - sMin) / Math.max(sMax - sMin, 1)) * ih;

  const dhws = obs.map((o) => o.dhw ?? 0);
  const dMax = Math.max(8, Math.ceil(Math.max(...dhws)));
  const yDHW = (v) => pad.t + ih - (v / dMax) * ih;

  const path = (accessor, yScale) => {
    let d = "", pen = false;
    obs.forEach((o, i) => {
      const v = accessor(o);
      if (v == null || Number.isNaN(v)) { pen = false; return; }   // gaps stay gaps
      d += `${pen ? "L" : "M"}${x(times[i]).toFixed(1)},${yScale(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const years = [];
  for (let y = new Date(t0).getUTCFullYear(); y <= new Date(t1).getUTCFullYear(); y++) {
    const t = Date.UTC(y, 0, 1);
    if (t >= t0 && t <= t1) years.push([y, t]);
  }
  const step = Math.ceil(years.length / 8) || 1;

  const bands = (events || []).filter((e) => e.kind === "bleaching").map((e) => {
    const a = x(Date.parse(e.start_date));
    const b = x(Date.parse(e.end_date || e.start_date));
    return `<rect x="${a.toFixed(1)}" y="${pad.t}" width="${Math.max(b - a, 2).toFixed(1)}"
      height="${ih}" fill="rgba(255,107,107,.18)"/>`;
  }).join("");

  const gridY = [];
  for (let v = sMin; v <= sMax; v += Math.max(1, Math.round((sMax - sMin) / 5))) {
    gridY.push(`<line x1="${pad.l}" x2="${W - pad.r}" y1="${ySST(v)}" y2="${ySST(v)}"
        stroke="rgba(255,255,255,.08)"/>
      <text x="${pad.l - 7}" y="${ySST(v) + 4}" fill="#a9cfe2" font-size="11"
        text-anchor="end">${v}°</text>`);
  }

  const threshold = (v, colour, label) => v <= dMax ? `
    <line x1="${pad.l}" x2="${W - pad.r}" y1="${yDHW(v)}" y2="${yDHW(v)}"
      stroke="${colour}" stroke-width="1" stroke-dasharray="4 4" opacity=".65"/>
    <text x="${W - pad.r + 5}" y="${yDHW(v) + 4}" fill="${colour}" font-size="10">${label}</text>` : "";

  return `<svg viewBox="0 0 ${W} ${H}" role="img"
    aria-label="Sea-surface temperature and degree heating weeks over time">
    ${bands}
    ${gridY.join("")}
    ${threshold(4, "#ffb547", "DHW 4")}
    ${threshold(8, "#ff6b6b", "DHW 8")}
    <path d="${path((o) => o.dhw, yDHW)}" fill="none" stroke="#ffb547" stroke-width="1.6" opacity=".9"/>
    <path d="${path((o) => o.sst, ySST)}" fill="none" stroke="#7fe6ff" stroke-width="1.4" opacity=".95"/>
    ${years.filter((_, i) => i % step === 0).map(([y, t]) =>
      `<text x="${x(t).toFixed(1)}" y="${H - 8}" fill="#a9cfe2" font-size="11"
        text-anchor="middle">${y}</text>`).join("")}
  </svg>`;
}

/* ── Footprint ──────────────────────────────────────────────── */

function wireFootprint() {
  $("footprint-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const tonnes = parseFloat($("tonnes").value);
    const out = $("footprint-out");
    if (!Number.isFinite(tonnes) || tonnes < 0) {
      out.innerHTML = `<p class="muted">Enter a number of tonnes.</p>`;
      return;
    }
    out.innerHTML = '<div class="spinner"></div>';
    try {
      const d = await getJSON(`/footprint/context?tonnes=${tonnes}`);
      const ratio = d.vs_paris_aligned_budget;
      out.innerHTML = `
        <div class="metric-row">
          <div class="metric"><b>1 in ${d.share_as_one_in.toLocaleString()}</b>
            <span>your share of global annual CO₂</span></div>
          <div class="metric"><b>${ratio.toFixed(1)}×</b>
            <span>a ${d.paris_aligned_tonnes} t/person 1.5 °C budget</span></div>
        </div>
        <ul class="levers">
          ${d.levers.map((l) => `<li><b>${esc(l.lever)}</b> — ${esc(l.note)}</li>`).join("")}
        </ul>
        <p class="caveat" style="margin-top:1.2rem">${esc(d.caveat)}</p>`;
    } catch (err) {
      out.innerHTML = `<p class="muted">${esc(err.message)}</p>`;
    }
  });
}

/* ── Boot ───────────────────────────────────────────────────── */

$("detail-close").addEventListener("click", () => {
  $("site-detail").hidden = true;
  history.replaceState(null, "", location.pathname);
  document.querySelectorAll(".site-card").forEach((b) => b.setAttribute("aria-pressed", "false"));
});

wireClassifier();
wireFootprint();
loadModel();
loadSites();
loadSamples();
