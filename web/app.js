"use strict";
// ClinicTune UI: plain JS, no build step. All server text is inserted with textContent (never innerHTML).

const $ = (sel) => document.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) n.append(k);
  return n;
};
const pct = (v) => (v === null || v === undefined ? "-" : `${Math.round(v * 100)}%`);
const LABEL = { refer: "Refer", low_risk: "Low risk", uncertain: "Uncertain" };
const ACTION = {
  refer: "refer for confirmatory testing",
  low_risk: "routine care",
  uncertain: "escalate to a senior clinician or retake",
};

function storedKey() {
  try { return localStorage.getItem("clinictune.key") || ""; } catch { return ""; }
}
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  const key = storedKey();
  if (key) headers["X-API-Key"] = key;
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  const type = res.headers.get("content-type") || "";
  return type.includes("json") ? res.json() : res.text();
}

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
    document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== `tab-${b.dataset.tab}`));
    if (b.dataset.tab === "monitor") loadMonitoring();
  })
);

// ------------------------------------------------------------------ API key
$("#key-btn").addEventListener("click", () => { $("#key-input").value = storedKey(); $("#key-dialog").showModal(); });
$("#key-dialog").addEventListener("close", () => {
  if ($("#key-dialog").returnValue === "save") {
    try { localStorage.setItem("clinictune.key", $("#key-input").value); } catch {}
    init();
  }
});

// ------------------------------------------------------------------ screening
let currentFile = null;
let lastRequestId = null;
const drop = $("#drop");
function setFile(f) {
  currentFile = f;
  $("#screen-btn").disabled = !f;
  const img = $("#preview");
  if (f && f.type.startsWith("image/")) {
    img.src = URL.createObjectURL(f); img.hidden = false; $("#drop-text").hidden = true;
  } else {
    img.hidden = true; $("#drop-text").hidden = false;
    if (f) $("#drop-text").textContent = f.name;
  }
}
$("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

$("#screen-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!currentFile) return;
  const btn = $("#screen-btn");
  btn.disabled = true; btn.textContent = "Screening...";
  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("task", document.querySelector("input[name=task]:checked").value);
  fd.append("lang", $("#lang").value);
  fd.append("heatmap", $("#heatmap").checked);
  if ($("#site-select").value) fd.append("site_id", $("#site-select").value);
  try {
    renderResult(await api("/v1/screen", { method: "POST", body: fd }));
  } catch (err) {
    $("#result-empty").hidden = false; $("#result").hidden = true;
    $("#result-empty").textContent = `Could not screen this image: ${err.message}`;
  } finally {
    btn.disabled = false; btn.textContent = "Screen image";
  }
});

function renderResult(r) {
  lastRequestId = r.request_id;
  $("#result-empty").hidden = true; $("#result").hidden = false;
  const d = $("#decision");
  d.className = `decision ${r.decision}`;
  d.replaceChildren(el("span", { textContent: LABEL[r.decision] }),
    el("small", { textContent: r.reason === "quality_gate" ? "retake the image" : ACTION[r.decision] }));
  $("#note").textContent = r.note.text;
  $("#note").lang = r.note.lang;
  const warns = [...r.warnings];
  if (!r.note.translation_reviewed) warns.push("This translation is a draft and has not been reviewed by a native-speaking clinician.");
  $("#warnings").replaceChildren(...warns.map((w) => el("li", { textContent: w })));

  $("#gate").replaceChildren(...r.quality_gate.checks.map((c) =>
    el("li", { className: c.passed ? "" : "fail", title: c.passed ? "" : c.message },
      el("span", { textContent: `${c.passed ? "✓" : "✗"} ${c.name.replace(/_/g, " ")}` }),
      el("span", { textContent: c.passed ? "" : c.message.split(".")[0] }))));

  const kv = [];
  if (r.probability !== null && r.probability !== undefined) {
    kv.push(["Calibrated score", r.probability.toFixed(3)], ["Referral threshold", r.threshold.toFixed(3)],
      ["Conformal set", r.prediction_set.length ? r.prediction_set.map((s) => LABEL[s]).join(" + ") : "(empty)"],
      ["Unfamiliar image?", r.ood.flagged ? "Yes - unlike training data" : "No"]);
  }
  kv.push(["Calibration", r.calibration], ["Reason", r.reason.replace(/_/g, " ")]);
  $("#triage").replaceChildren(...kv.flatMap(([k, v]) => [el("dt", { textContent: k }), el("dd", { textContent: v })]));

  if (r.heatmap_png) { $("#heat").src = r.heatmap_png; $("#heat-fig").hidden = false; } else { $("#heat-fig").hidden = true; }
  $("#outcome-msg").textContent = "";
  $("#meta").textContent = `request ${r.request_id} · model ${r.model.bundle_id} (${r.model.backbone}) · ${r.latency_ms} ms · ${r.disclaimer}`;
}

document.querySelectorAll("[data-outcome]").forEach((b) => b.addEventListener("click", async () => {
  if (!lastRequestId) return;
  try {
    await api(`/v1/screenings/${lastRequestId}/outcome`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outcome: Number(b.dataset.outcome) }),
    });
    $("#outcome-msg").textContent = "Saved.";
  } catch (err) { $("#outcome-msg").textContent = err.message; }
}));

// ------------------------------------------------------------------ sites
let sites = [];
async function loadSites() {
  sites = await api("/v1/sites");
  const opts = (withNone) => [
    ...(withNone ? [el("option", { value: "", textContent: "Not onboarded (training-site settings)" })] : []),
    ...sites.map((s) => el("option", { value: s.site_id, textContent: `${s.name} (${s.site_id})` })),
  ];
  const keep = $("#site-select").value;
  $("#site-select").replaceChildren(...opts(true)); $("#site-select").value = keep;
  $("#onboard-site").replaceChildren(...opts(false));
  $("#mon-site").replaceChildren(...opts(false));
  const list = $("#sites-list");
  if (!sites.length) { list.className = "empty"; list.textContent = "No sites yet."; return; }
  list.className = "";
  list.replaceChildren(...sites.map((s) => {
    const cals = s.calibrations.map((c) => {
      const row = el("div", { className: "cal" },
        el("span", { textContent: `${c.task.toUpperCase()} v${c.version}` }),
        el("span", { className: `pill ${c.verdict}`, textContent: (c.verdict || "").replace(/_/g, " ") }));
      if (c.active) row.append(el("span", { className: "pill active", textContent: "active" }));
      const view = el("button", { className: "link", type: "button", textContent: "report card" });
      view.onclick = () => showReport(s.site_id, c.task, c.version);
      row.append(view);
      if (!c.active) {
        const act = el("button", { className: "link", type: "button", textContent: "activate" });
        act.onclick = async () => { await api(`/v1/sites/${s.site_id}/calibrations/${c.task}/${c.version}/activate`, { method: "POST" }); loadSites(); };
        row.append(act);
      }
      return row;
    });
    return el("div", { className: "site" }, el("strong", { textContent: s.name }),
      el("span", { className: "hint", textContent: `  ${s.site_id}` }),
      ...(cals.length ? cals : [el("div", { className: "hint", textContent: "Not onboarded yet." })]));
  }));
}

async function showReport(siteId, task, version) {
  const html = await api(`/v1/sites/${siteId}/report.html?task=${task}&version=${version}`);
  $("#report-frame").srcdoc = html;
  $("#report-wrap").hidden = false;
  $("#report-wrap").scrollIntoView({ behavior: "smooth" });
}
$("#print-report").addEventListener("click", () => $("#report-frame").contentWindow.print());

$("#site-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/v1/sites", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ site_id: $("#site-id").value.trim(), name: $("#site-name").value.trim() }) });
    e.target.reset(); await loadSites();
  } catch (err) { alert(err.message); }
});

$("#onboard-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("task", $("#onboard-task").value);
  fd.append("pack", $("#onboard-file").files[0]);
  const siteId = $("#onboard-site").value;
  $("#job").hidden = false; $("#job-msg").textContent = "Uploading..."; $("#job-bar").style.width = "2%";
  try {
    const j = await api(`/v1/sites/${siteId}/onboard`, { method: "POST", body: fd });
    pollJob(j.job_id, siteId, $("#onboard-task").value);
  } catch (err) { $("#job-msg").textContent = `Failed: ${err.message}`; }
});

async function pollJob(id, siteId, task) {
  const j = await api(`/v1/jobs/${id}`);
  $("#job-bar").style.width = `${Math.round(j.progress * 100)}%`;
  $("#job-msg").textContent = `${j.status}: ${j.message || ""}`;
  if (j.status === "done") { await loadSites(); showReport(siteId, task, j.result.version); return; }
  if (j.status === "failed") return;
  setTimeout(() => pollJob(id, siteId, task), 1000);
}

// ------------------------------------------------------------------ monitoring
let monSeq = 0;
async function loadMonitoring() {
  const site = $("#mon-site").value;
  if (!site) { $("#mon").replaceChildren(el("div", { className: "empty", textContent: "Create and use a site first." })); return; }
  const seq = ++monSeq;
  try {
    const m = await api(`/v1/sites/${site}/monitoring?task=${$("#mon-task").value}&days=${$("#mon-days").value}`);
    if (seq !== monSeq) return; // a newer request superseded this one
    const tiles = [["Screenings", m.n]];
    if (m.n) {
      tiles.push(["Refer", pct(m.decision_rates.refer)], ["Low risk", pct(m.decision_rates.low_risk)],
        ["Uncertain", pct(m.decision_rates.uncertain)], ["Quality-gate failures", pct(m.gate_fail_rate)],
        ["Unfamiliar images", pct(m.ood_rate)], ["Median latency", `${Math.round(m.latency_ms_p50)} ms`],
        ["Confirmed outcomes", m.confirmed_outcomes]);
      if (m.observed_sensitivity !== undefined) tiles.push(["Observed sensitivity", pct(m.observed_sensitivity)]);
      if (m.observed_specificity !== undefined) tiles.push(["Observed specificity", pct(m.observed_specificity)]);
      if (m.missed_positives_low_risk) tiles.push(["Positives called low risk", m.missed_positives_low_risk]);
    }
    $("#mon").replaceChildren(...tiles.map(([k, v]) => el("div", { className: "tile" }, el("b", { textContent: v }), el("span", { textContent: k }))));
    $("#mon-note").textContent = m.note || "";
  } catch (err) { $("#mon").replaceChildren(el("div", { className: "empty", textContent: err.message })); }
}
$("#mon-btn").addEventListener("click", loadMonitoring);
["#mon-site", "#mon-task"].forEach((s) => $(s).addEventListener("change", loadMonitoring));

// ------------------------------------------------------------------ about
function renderAbout(info) {
  const a = $("#about");
  const items = [
    el("h2", { textContent: "Intended use" }), el("p", { textContent: info.disclaimer }),
    el("p", { textContent: "Outputs: Refer, Low risk, or Uncertain (escalate or retake). A quality gate runs first; an out-of-distribution check and a class-conditional conformal layer turn ambiguous cases into Uncertain rather than guessing." }),
    el("h2", { textContent: "Model" }),
    el("p", { textContent: `Bundle ${info.bundle_id} · backbone ${info.backbone} · ClinicTune ${info.version}${info.demo ? " · DEMO (synthetic)" : ""}` }),
  ];
  for (const [task, t] of Object.entries(info.tasks)) {
    const m = t.internal_metrics || {};
    items.push(el("p", { textContent: `${task.toUpperCase()} - ${t.modality}: internal held-out sensitivity ${pct(m.sensitivity)}, specificity ${pct(m.specificity)}, ROC-AUC ${m.roc_auc ? m.roc_auc.toFixed(2) : "-"} (training sites only; performance at a new site must be measured by onboarding).` }));
  }
  items.push(el("h2", { textContent: "Training data" }), el("p", { textContent: (info.datasets || []).join(", ") || "-" }));
  a.replaceChildren(...items);
}

// ------------------------------------------------------------------ init
async function init() {
  try {
    const info = await api("/v1/info");
    $("#demo-banner").hidden = !info.demo;
    renderAbout(info);
    await loadSites();
  } catch (err) {
    $("#result-empty").textContent = `Service not ready: ${err.message}. ${err.message.startsWith("401") ? "Set the API key (Key button)." : ""}`;
  }
}
init();
