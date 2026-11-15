"use strict";
// Smoke & Signal front end: renders /api/meta and drives live forecasts via /api/predict.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid);
  return el;
};
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${Number(v).toFixed(d)}%`);
const MODEL_NAMES = { Ridge: "Ridge", RandomForest: "Random Forest", HistGB: "HistGB",
  Persistence: "Persistence", SeasonalMean: "Seasonal mean" };
const CAT_VAR = { "Good": "--c-good", "Satisfactory": "--c-satisfactory", "Moderate": "--c-moderate",
  "Poor": "--c-poor", "Very Poor": "--c-very-poor", "Severe": "--c-severe" };
const catColor = (c) => `var(${CAT_VAR[c] || "--rule"})`;
const PRESETS = [
  ["Smog peak · 1 Nov 2019", "2019-11-01"],
  ["Diwali · 27 Oct 2019", "2019-10-27"],
  ["Great Smog · 5 Nov 2016", "2016-11-05"],
  ["Winter, no fires · 15 Jan 2019", "2019-01-15"],
  ["Monsoon · 15 Aug 2019", "2019-08-15"],
  ["Lockdown · 1 Apr 2020", "2020-04-01"],
];
const WEATHER = ["temp", "humidity", "wind_speed", "wind_dir", "precip", "blh_mean", "blh_min"];

let META = null;
let observed = null;   // {actual, split, date} of the loaded historical day
let edited = false;

// ------------------------------------------------------------------ boot
init();
async function init() {
  buildWeek(new Date("2019-11-01"));
  $("#presets").append(...PRESETS.map(([label, date]) =>
    h("button", { type: "button", class: "chip", "data-date": date, onclick: () => loadDay(date) }, label)));
  $("#load-day").addEventListener("click", () => loadDay($("#date").value));
  $("#date").addEventListener("change", () => buildWeek(new Date($("#date").value)));
  $("#inputs").addEventListener("input", (e) => {
    if (e.target.id === "date") return;
    edited = true; updateCompass(); schedule();
  });
  setupLightbox();
  try {
    META = await (await fetch("/api/meta")).json();
    renderMeta(META);
  } catch (e) { console.error(e); }
  loadDay("2019-11-01");
}

// ------------------------------------------------------------------ inputs
function buildWeek(t) {
  if (Number.isNaN(t.getTime())) return;
  const week = $("#week");
  const keep = { pm25: $$("[data-pm]").map((i) => i.value), fires: $$("[data-fire]").map((i) => i.value) };
  week.replaceChildren(h("span"));
  for (let i = 6; i >= 0; i--) {
    const d = new Date(t); d.setDate(d.getDate() - i);
    week.append(h("span", { class: "h" + (i === 0 ? " today" : "") },
      i === 0 ? "today" : d.toLocaleDateString("en-GB", { day: "numeric", month: "short" })));
  }
  week.append(h("span", { class: "rl" }, "PM2.5"));
  for (let i = 0; i < 7; i++) week.append(h("input", { type: "number", step: "any", min: 0, max: 1000,
    "data-pm": i, "aria-label": `PM2.5 day ${i - 6}`, value: keep.pm25[i] ?? "", required: i === 6 }));
  week.append(h("span", { class: "rl" }, "Fires"));
  for (let i = 0; i < 7; i++) week.append(h("input", { type: "number", step: 1, min: 0,
    "data-fire": i, "aria-label": `fire detections day ${i - 6}`, value: keep.fires[i] ?? "" }));
}

async function loadDay(date) {
  if (!date) return;
  $$(".chip").forEach((c) => c.setAttribute("aria-pressed", c.dataset.date === date));
  const r = await fetch(`/api/example?date=${date}`);
  const j = await r.json();
  if (!r.ok) { $("#form-error").textContent = j.error; return; }
  const inp = j.inputs;
  $("#date").value = inp.date;
  buildWeek(new Date(inp.date));
  $$("[data-pm]").forEach((el, i) => (el.value = inp.pm25[i] ?? ""));
  $$("[data-fire]").forEach((el, i) => (el.value = inp.fires[i] ?? 0));
  $$("[data-k]").forEach((el) => (el.value = inp[el.dataset.k] ?? ""));
  $$("[data-w]").forEach((el) => (el.value = inp.weather[el.dataset.w] ?? ""));
  observed = { actual: j.actual, split: j.split, date: inp.date };
  edited = false;
  $("#day-note").textContent = `Observed data for ${niceDate(inp.date)} · this day is in the ${j.split} period` +
    (j.split === "train" || j.split === "validation" ? " (the models trained on this day, so expect a closer fit)." : j.split === "test" ? " (never seen in training)." : ".");
  updateCompass();
  run();
}

function collect() {
  const num = (v) => (v === "" || v == null ? null : Number(v));
  return {
    date: $("#date").value,
    pm25: $$("[data-pm]").map((i) => num(i.value)),
    fires: $$("[data-fire]").map((i) => num(i.value) ?? 0),
    ...Object.fromEntries($$("[data-k]").map((i) => [i.dataset.k, num(i.value)])),
    weather: Object.fromEntries($$("[data-w]").map((i) => [i.dataset.w, num(i.value)])),
  };
}

function updateCompass() {
  const d = Number($('[data-w="wind_dir"]').value);
  if (Number.isNaN(d) || $('[data-w="wind_dir"]').value === "") { $("#compass").textContent = ""; return; }
  const pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  const name = pts[Math.round((d % 360) / 22.5) % 16];
  $("#compass").textContent = `${name}${d >= 270 && d <= 360 ? " · from Punjab" : ""}`;
}

let timer;
function schedule() { clearTimeout(timer); timer = setTimeout(run, 350); }

async function run() {
  const err = $("#form-error");
  const body = collect();
  if (!body.date || body.pm25[6] == null) { err.textContent = "Today's PM2.5 and a date are required."; return; }
  if (WEATHER.some((k) => body.weather[k] == null)) { err.textContent = "Fill in all weather fields."; return; }
  err.textContent = "";
  $(".outputs").classList.add("loading");
  try {
    const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) { err.textContent = j.error; return; }
    renderForecast(j);
  } catch (e) { err.textContent = "Could not reach the server."; }
  finally { $(".outputs").classList.remove("loading"); }
}

// ------------------------------------------------------------------ forecast output
function niceDate(s) {
  return new Date(s + "T00:00").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
}

function renderForecast(res) {
  const best = res.selected_model;
  const cards = ["1", "2", "3"].map((k) => {
    const hz = res.horizons[k];
    const act = observed && observed.date === $("#date").value ? observed.actual[k] : null;
    const err = act != null ? hz.main - act : null;
    return h("article", { class: "hz", style: `--cat:${catColor(hz.category)}` },
      h("div", { class: "when" }, `${k} day${k === "1" ? "" : "s"} ahead`),
      h("div", { class: "date" }, niceDate(hz.target_date)),
      h("div", { class: "big" }, fmt(hz.main, 0), h("small", {}, "µg/m³")),
      h("span", { class: "cat" }, hz.category),
      act != null ? h("div", { class: "obs" },
        "actually measured ", h("b", {}, fmt(act, 0)),
        ` · off by ${err > 0 ? "+" : ""}${fmt(err, 0)}${edited ? " · inputs edited" : ""}`) : null,
      hz.is_diwali_window ? h("div", { class: "obs" }, "Diwali window (fireworks flag on)") : null);
  });
  $("#horizons").replaceChildren(...cards);

  const h1 = res.horizons["1"];
  const ab = $("#alert-banner");
  ab.hidden = false;
  ab.className = "alert-banner " + (h1.severe_alert ? "on" : "off");
  ab.replaceChildren(h("span", { class: "icon" }, h1.severe_alert ? "ALERT" : "no alert"),
    h("span", {}, h1.severe_alert
      ? `Severe smog likely on ${niceDate(h1.target_date)}: the forecast (${fmt(h1.main, 0)}) is above the alert trigger of ${res.alert_threshold}.`
      : `No Severe-smog alert for tomorrow: the forecast (${fmt(h1.main, 0)}) is below the alert trigger of ${res.alert_threshold}.`,
      h("small", { class: "muted" }, "The trigger sits below 250 because the model tends to under-forecast peaks; it was set on 2018 data.")));

  const d = h1.main - h1.main_no_fires;
  $("#fire-delta").innerHTML = `Tomorrow's forecast is <b>${fmt(h1.main, 0)}</b> with fire data and <b>${fmt(h1.main_no_fires, 0)}</b> without ` +
    `(a difference of ${d >= 0 ? "+" : ""}${fmt(d, 0)} µg/m³), from the ${MODEL_NAMES[best]} model.`;
  drawWhatIf(res.fire_what_if, h1.main_no_fires);
  drawModelTable(res, best);
}

function drawModelTable(res, best) {
  const keys = Object.keys(res.horizons["1"].predictions);
  const order = ["Ridge|no_fires", "Ridge|with_fires", "RandomForest|no_fires", "RandomForest|with_fires",
    "HistGB|no_fires", "HistGB|with_fires", "Persistence|baseline", "SeasonalMean|baseline"].filter((k) => keys.includes(k));
  const head = h("tr", {}, h("th", {}, "Model"), h("th", {}, "Features"),
    ...["1", "2", "3"].map((k) => h("th", { class: "n" }, `${k} day${k === "1" ? "" : "s"}`)));
  const rows = order.map((key) => {
    const [m, v] = key.split("|");
    return h("tr", { class: m === best && v === "with_fires" ? "best" : "" },
      h("td", {}, MODEL_NAMES[m], m === best && v !== "baseline" ? h("span", { class: "tag" }, "selected") : null),
      h("td", {}, v === "baseline" ? h("span", { class: "muted" }, "baseline")
        : h("span", { class: "tag" + (v === "with_fires" ? " fire" : "") }, v === "with_fires" ? "with fires" : "no fires")),
      ...["1", "2", "3"].map((k) => h("td", { class: "n" }, fmt(res.horizons[k].predictions[key], 0))));
  });
  $("#model-table").replaceChildren(h("thead", {}, head), h("tbody", {}, rows));
}

function drawWhatIf(rows, noFires) {
  const W = 640, H = 230, P = { l: 44, r: 16, t: 14, b: 40 };
  const ys = rows.map((r) => r.prediction).concat([noFires]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = Math.max(8, (hi - lo) * 0.25); lo = Math.max(0, lo - pad); hi += pad;
  const xmax = rows[rows.length - 1].multiplier;
  const x = (m) => P.l + (m / xmax) * (W - P.l - P.r);
  const y = (v) => H - P.b - ((v - lo) / (hi - lo)) * (H - P.t - P.b);
  const ns = "http://www.w3.org/2000/svg";
  const s = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; return e; };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Forecast versus scaled fire activity" });
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = lo + (i / ticks) * (hi - lo);
    svg.append(s("line", { x1: P.l, x2: W - P.r, y1: y(v), y2: y(v), stroke: "var(--rule)", "stroke-width": 1 }),
      s("text", { x: P.l - 8, y: y(v) + 4, "text-anchor": "end" }, Math.round(v)));
  }
  rows.forEach((r) => svg.append(s("text", { x: x(r.multiplier), y: H - P.b + 16, "text-anchor": "middle" }, `×${r.multiplier}`)));
  svg.append(s("text", { x: (P.l + W - P.r) / 2, y: H - 4, "text-anchor": "middle", class: "lbl" },
    "fire activity, relative to what you entered"));
  svg.append(s("line", { x1: x(1), x2: x(1), y1: P.t, y2: H - P.b, stroke: "var(--muted)", "stroke-dasharray": "2 3" }));
  svg.append(s("line", { x1: P.l, x2: W - P.r, y1: y(noFires), y2: y(noFires), stroke: "var(--blue)", "stroke-width": 2, "stroke-dasharray": "6 4" }));
  svg.append(s("text", { x: W - P.r, y: y(noFires) - 6, "text-anchor": "end", fill: "var(--blue)", style: "fill:var(--blue)" }, `model without fires: ${Math.round(noFires)}`));
  svg.append(s("polyline", { points: rows.map((r) => `${x(r.multiplier)},${y(r.prediction)}`).join(" "),
    fill: "none", stroke: "var(--accent)", "stroke-width": 2.5, "stroke-linejoin": "round" }));
  rows.forEach((r) => {
    const c = s("circle", { cx: x(r.multiplier), cy: y(r.prediction), r: r.multiplier === 1 ? 6 : 4.5,
      fill: "var(--accent)", stroke: "var(--card)", "stroke-width": 2 });
    c.append(s("title", {}, `×${r.multiplier} (${Math.round(r.fires_today)} fires today) → ${r.prediction.toFixed(0)} µg/m³`));
    svg.append(c);
  });
  const one = rows.find((r) => r.multiplier === 1);
  svg.append(s("text", { x: x(1) + 8, y: y(one.prediction) - 10, class: "lbl" }, `as entered: ${Math.round(one.prediction)}`));
  $("#whatif").replaceChildren(svg);
}

// ------------------------------------------------------------------ static sections
function renderMeta(m) {
  const g = m.goals;
  const rob = m.robustness;
  const gains = rob.map((r) => r["gain %"]);
  const h1 = rob.filter((r) => r.horizon === 1), h3 = rob.filter((r) => r.horizon === 3);
  const rf = (rows) => rows.find((r) => r.model === m.selected_model);
  const edge = (r) => 100 * (1 - r["season MAE no fires"] / r["persistence season MAE"]);
  const cal = g.G3.results.with_fires.severe_val_calibrated;
  const lo = Math.min(...gains), hi = Math.max(...gains);
  const best = MODEL_NAMES[m.selected_model];
  $("#hero-stats").replaceChildren(
    stat("≈ 0%", "Fire data does not improve the forecast",
      `Adding fire counts changed burning-season forecast error by between ${Math.abs(lo).toFixed(1)}% worse and ` +
      `${hi.toFixed(1)}% better, across 2017–2019 and 1–3 days ahead. Every 95% confidence range includes zero, ` +
      `so no version showed a real effect.`),
    stat(`${edge(rf(h1)).toFixed(0)}% → ${edge(rf(h3)).toFixed(0)}%`, "Smaller errors than the naive forecast",
      `During the burning season, the ${best} model's forecasts miss by ${edge(rf(h1)).toFixed(0)}% less than ` +
      `"tomorrow = today" one day ahead, and ${edge(rf(h3)).toFixed(0)}% less three days ahead. ` +
      `Its edge grows the further ahead it looks.`),
    stat(`${(100 * cal.recall).toFixed(0)}%`, "Severe smog days flagged a day early",
      `In 2019, an alert fired the day before ${(100 * cal.recall).toFixed(0)}% of the days PM2.5 went above 250. ` +
      `${(100 * cal.precision).toFixed(0)}% of alerts were correct. The alert trigger (${cal.forecast_threshold}) ` +
      `was set on 2018 data.`),
    stat(`${(m.cleaning.fires.in_bbox / 1000).toFixed(0)}k`, "Satellite fire detections analysed",
      `NASA fire detections over Punjab and Haryana, matched day by day against ` +
      `${m.data_range.days.toLocaleString()} days of Delhi air-quality and weather records (2015–2020).`));

  renderGoals(m);
  renderTestTable(m);
  renderRobustTable(m);
  const figs = (grp) => m.figures.filter((f) => (grp === "eda" ? !f.file.startsWith("R") : f.file.startsWith("R")));
  $("#gallery-results").replaceChildren(...figs("results").map(figCard));
  $("#gallery-eda").replaceChildren(...figs("eda").map(figCard));

  const c = m.cleaning;
  $("#clean-stats").replaceChildren(
    cs(c.aqi.rows.toLocaleString(), `Delhi days, ${c.aqi.start} → ${c.aqi.end}`),
    cs(`${(100 * c.aqi.pm25_missing_frac).toFixed(1)}%`, "days with PM2.5 missing: dropped, never filled in"),
    cs(c.fires.raw_rows.toLocaleString(), `raw fire pixels → ${c.fires.in_bbox.toLocaleString()} after filtering`),
    cs(c.fires.ist_date_differs_from_utc, "fires re-dated from satellite (UTC) time to Indian time"));

  const fe = (names) => h("dd", {}, ...names.map((n) => h("code", {}, n)));
  const groups = {
    Persistence: m.features.base.filter((f) => /^(pm|no2|co)_?/.test(f)),
    Weather: m.features.base.filter((f) => /^(wind_|nw_wind|temp|humidity|precip|blh)/.test(f)),
    Calendar: m.features.base.filter((f) => /^(month|dow|is_)/.test(f)),
  };
  $("#features").replaceChildren(
    ...Object.entries(groups).flatMap(([k, v]) => [h("dt", {}, k), fe(v)]),
    h("dt", { class: "fire" }, "Fires (tested)"), fe(m.features.fires));

  const cvRows = Object.entries(m.cv).map(([k, v]) => {
    const [model, variant] = k.split("/");
    const params = Object.entries(v.best_params).filter(([p]) => !["n_estimators", "l2_regularization", "min_samples_leaf"].includes(p) || model !== "HistGB")
      .map(([p, val]) => `${p}=${val}`).join(", ");
    return h("tr", { class: model === m.selected_model ? "best" : "" },
      h("td", {}, MODEL_NAMES[model], h("span", { class: "tag" + (variant === "with_fires" ? " fire" : "") }, variant.replace("_", " "))),
      h("td", { style: "white-space:normal;font-size:12px" }, h("code", {}, params)),
      h("td", { class: "n" }, fmt(v.cv_mae, 1)));
  });
  $("#cv-table").replaceChildren(h("thead", {}, h("tr", {}, h("th", {}, "Model"), h("th", {}, "Best parameters"), h("th", { class: "n" }, "CV MAE"))),
    h("tbody", {}, cvRows));
  $("#selection-note").textContent = `Selection rule: ${m.selection_rule}. Validation MAE (2018): ` +
    Object.entries(m.family_val_mae).map(([k, v]) => `${MODEL_NAMES[k]} ${v.toFixed(2)}`).join(" · ") +
    `, so ${MODEL_NAMES[m.selected_model]} was selected.`;
}

const stat = (v, title, text) => h("div", { class: "stat" }, h("div", { class: "v" }, v),
  h("div", {}, h("h3", { class: "t" }, title), h("p", { class: "l" }, text)));
const cs = (v, l) => h("div", {}, h("b", {}, String(v)), h("span", {}, l));

function renderGoals(m) {
  const g = m.goals, best = MODEL_NAMES[m.selected_model];
  const g2 = g.G2.selected_model, g3 = g.G3.results, rob = m.robustness;
  const goal = (id, verdict, cls, title, text, kv) => h("div", { class: "goal" },
    h("span", { class: `verdict ${cls}` }, verdict), h("div", { class: "gid" }, id), h("h4", {}, title),
    h("p", {}, text), h("div", { class: "kv" }, kv));
  $("#goals").replaceChildren(
    goal("G1", "met", "met", "Beat the naive forecast",
      `${best} misses by less than "tomorrow = today", both across all of 2019 and in the burning season.`,
      `average error ${fmt(g.G1.best_with_fires_overall)} vs ${fmt(g.G1.persistence_overall)} µg/m³ · burning season ${fmt(g.G1.best_with_fires_fire_season)} vs ${fmt(g.G1.persistence_fire_season)}`),
    goal("G2", "not met", "not", "Fire data cuts error by 10% or more",
      `In 2019 alone, fire data cut error by ${(100 * g2.rel_reduction).toFixed(1)}%, short of the goal and only borderline-significant. Checked across three years, the benefit disappears.`,
      `three-year change: ${Math.abs(Math.min(...rob.map((r) => r["gain %"]))).toFixed(1)}% worse to ${Math.max(...rob.map((r) => r["gain %"])).toFixed(1)}% better`),
    goal("G3", "met after tuning", "partial", "Warn of 70%+ of Severe days",
      `Alerting only when the forecast tops 250 caught ${(100 * g3.with_fires.severe.recall).toFixed(0)}% of Severe days, because the model under-shoots peaks. Lowering the trigger to ${g3.with_fires.severe_val_calibrated.forecast_threshold} (chosen on 2018) caught ${(100 * g3.with_fires.severe_val_calibrated.recall).toFixed(0)}%.`,
      `${(100 * g3.with_fires.severe_val_calibrated.precision).toFixed(0)}% of alerts correct · naive forecast: catches ${(100 * g3.Persistence.severe.recall).toFixed(0)}%, ${(100 * g3.Persistence.severe.precision).toFixed(0)}% correct`));
}

function renderTestTable(m) {
  const rows = {};
  for (const r of m.test_metrics) {
    const k = `${r.model}|${r.variant}`;
    (rows[k] ||= { model: r.model, variant: r.variant })[r.scope] = r;
  }
  const head = h("tr", {}, ...["Model", "Features"].map((t) => h("th", {}, t)),
    ...["Avg error, all year", "RMSE, all year", "R², all year", "Avg error, burning season", "R², burning season"].map((t) => h("th", { class: "n" }, t)));
  const body = Object.values(rows).map((r) => h("tr", { class: r.model === m.selected_model && r.variant === "with_fires" ? "best" : "" },
    h("td", {}, MODEL_NAMES[r.model]),
    h("td", {}, r.variant === "baseline" ? h("span", { class: "muted" }, "baseline") :
      h("span", { class: "tag" + (r.variant === "with_fires" ? " fire" : "") }, r.variant.replace("_", " "))),
    h("td", { class: "n" }, fmt(r.overall.MAE)), h("td", { class: "n" }, fmt(r.overall.RMSE)),
    h("td", { class: "n" }, fmt(r.overall.R2, 2)), h("td", { class: "n" }, fmt(r.fire_season.MAE)),
    h("td", { class: "n" }, fmt(r.fire_season.R2, 2))));
  $("#test-table").replaceChildren(h("thead", {}, head), h("tbody", {}, body));
}

function renderRobustTable(m) {
  const head = h("tr", {}, h("th", {}, "Horizon"), h("th", {}, "Model"),
    ...["Avg error without fires", "Avg error with fires", "Fire benefit", "95% range", "Benefit 2017", "Benefit 2018", "Benefit 2019"].map((t) => h("th", { class: "n" }, t)));
  const sign = (v) => h("td", { class: "n " + (v > 0 ? "pos" : "neg") }, pct(v));
  const body = m.robustness.map((r) => h("tr", {},
    h("td", {}, `${r.horizon} day${r.horizon > 1 ? "s" : ""} ahead`), h("td", {}, MODEL_NAMES[r.model]),
    h("td", { class: "n" }, fmt(r["season MAE no fires"])), h("td", { class: "n" }, fmt(r["season MAE with fires"])),
    sign(r["gain %"]), h("td", { class: "n" }, `[${fmt(r["iid CI low"])}, ${fmt(r["iid CI high"])}]`),
    sign(r["gain % 2017"]), sign(r["gain % 2018"]), sign(r["gain % 2019"])));
  $("#robust-table").replaceChildren(h("thead", {}, head), h("tbody", {}, body));
}

function figCard(f) {
  return h("button", { class: "fig", type: "button", onclick: () => openFig(f) },
    h("img", { src: `/figures/${f.file}`, alt: f.title, loading: "lazy" }),
    h("div", {}, h("b", {}, f.title), h("span", {}, f.caption)));
}

function setupLightbox() {
  const d = $("#lightbox");
  $(".close", d).addEventListener("click", () => d.close());
  d.addEventListener("click", (e) => { if (e.target === d) d.close(); });
}
function openFig(f) {
  const d = $("#lightbox");
  $("img", d).src = `/figures/${f.file}`; $("img", d).alt = f.title;
  $("p", d).textContent = f.caption;
  d.showModal();
}
