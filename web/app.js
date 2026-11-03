"use strict";
// ClinicTune web UI. Plain JS, no build step, works offline once loaded.
// Every server-provided string is inserted with textContent - never innerHTML.

/* ================================================================ helpers */
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  setAttrs(n, attrs);
  append(n, kids);
  return n;
}
function sv(tag, attrs = {}, ...kids) {
  const n = document.createElementNS(SVGNS, tag);
  setAttrs(n, attrs);
  append(n, kids);
  return n;
}
function setAttrs(n, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "text") n.textContent = v;
    else if (k === "class") n.setAttribute("class", v);
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : String(v));
  }
}
function append(n, kids) {
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}
function icon(id, cls) {
  const s = sv("svg", { class: cls, "aria-hidden": "true" });
  const u = document.createElementNS(SVGNS, "use");
  u.setAttribute("href", "#" + id);
  s.append(u);
  return s;
}
const pct = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function pref(key, value) {
  try {
    if (value === undefined) return localStorage.getItem("clinictune." + key);
    localStorage.setItem("clinictune." + key, value);
  } catch { /* storage unavailable: preferences are a convenience only */ }
  return null;
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  const key = pref("key");
  if (key) headers["X-API-Key"] = key;
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* not JSON */ }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return (res.headers.get("content-type") || "").includes("json") ? res.json() : res.text();
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3200);
}

/* ================================================================ vocabulary */
const DECISION = {
  refer: { label: "Refer", icon: "i-refer" },
  low_risk: { label: "Low risk", icon: "i-low" },
  uncertain: { label: "Uncertain", icon: "i-uncertain" },
};
const TASKS = {
  tb: { short: "Chest X-ray", condition: "TB", positive: "TB", negative: "No TB" },
  dr: { short: "Fundus photo", condition: "referable DR", positive: "Referable", negative: "Not referable" },
};
const NEXT_STEP = {
  tb: {
    refer: ["Send for confirmatory testing", "Sputum NAAT (for example Xpert MTB/RIF) and clinical review, as the national programme advises."],
    low_risk: ["No TB referral needed from this image", "Continue routine care. A cough of two weeks or more, fever or weight loss still needs assessment."],
  },
  dr: {
    refer: ["Refer to an eye specialist", "Signs of moderate or worse diabetic retinopathy. Arrange an ophthalmology review."],
    low_risk: ["Rescreen at the usual interval", "No referable diabetic retinopathy seen. Usually rescreen in a year, sooner if vision changes."],
  },
};
const REASON = {
  conformal_refer: "The calibrated score sits clearly on the referral side for this clinic.",
  ambiguous_set_above_threshold: "Both outcomes are still possible, and the score is above this clinic's referral line, so the aid errs towards referral.",
  ambiguous_set: "Both outcomes are still possible at 90% confidence, so the aid will not call this low risk.",
  conformal_low_risk: "Only low risk is consistent with this image at 90% confidence, and the score is below the referral line.",
  conflict_threshold_vs_conformal: "The score and the confidence check disagree, so a clinician should decide.",
  empty_set: "The image fits neither outcome well. It may be unusual.",
  out_of_distribution: "This image looks unlike the images the model was calibrated on. A different machine, view or body part?",
  quality_gate: "The image did not pass the quality check.",
};
const SHORT_REASON = {
  conformal_refer: "Clear referral",
  ambiguous_set_above_threshold: "Leans referral",
  ambiguous_set: "Not certain enough",
  conformal_low_risk: "Clear low risk",
  conflict_threshold_vs_conformal: "Checks disagree",
  empty_set: "Unusual image",
  out_of_distribution: "Unfamiliar image",
  quality_gate: "Retake image",
};

/* ================================================================ app state */
const state = { info: null, sites: [], evidence: undefined };
const currentTask = () => $("input[name=task]:checked").value;
const currentSite = () => $("#site-select").value;
const siteName = (id) => (state.sites.find((s) => s.site_id === id) || {}).name || id;

/* ================================================================ tabs */
const TAB_NAMES = ["screen", "queue", "sites", "monitor", "validation", "about"];
function showTab(name, { focus = false, push = true } = {}) {
  if (!TAB_NAMES.includes(name)) name = "screen";
  for (const b of $$(".tabs [role=tab]")) {
    const on = b.dataset.tab === name;
    b.setAttribute("aria-selected", on);
    b.tabIndex = on ? 0 : -1;
    if (on && focus) b.focus();
  }
  for (const p of $$(".tab")) p.hidden = p.id !== `tab-${name}`;
  if (push) history.replaceState(null, "", `#${name}`);
  if (name === "monitor") loadMonitoring();
  if (name === "validation") loadValidation();
  if (name === "sites") loadSites();
}
$(".tabs").addEventListener("click", (e) => {
  const b = e.target.closest("[role=tab]");
  if (b) showTab(b.dataset.tab);
});
$(".tabs").addEventListener("keydown", (e) => {
  const tabs = $$(".tabs [role=tab]");
  const i = tabs.indexOf(document.activeElement);
  if (i < 0) return;
  const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
  if (next === undefined) return;
  e.preventDefault();
  showTab(tabs[(next + tabs.length) % tabs.length].dataset.tab, { focus: true });
});

/* ================================================================ masthead preferences */
for (const [sel, key] of [["#lang", "lang"]]) {
  const v = pref(key);
  if (v) $(sel).value = v;
  $(sel).addEventListener("change", () => pref(key, $(sel).value));
}
const savedTask = pref("task");
if (savedTask && $(`input[name=task][value=${savedTask}]`)) $(`input[name=task][value=${savedTask}]`).checked = true;
$$("input[name=task]").forEach((r) => r.addEventListener("change", () => pref("task", currentTask())));
$("#site-select").addEventListener("change", () => pref("site", currentSite()));

$("#key-btn").addEventListener("click", () => {
  $("#key-input").value = pref("key") || "";
  $("#key-dialog").showModal();
});
$("#key-dialog").addEventListener("close", () => {
  if ($("#key-dialog").returnValue === "save") {
    pref("key", $("#key-input").value);
    init();
  }
});

/* ================================================================ screen: lightbox */
const lightbox = $("#lightbox");
const film = { file: null, url: null, result: null };

function setFilm(file, { result = null } = {}) {
  if (film.url && film.owned) URL.revokeObjectURL(film.url); // queue thumbnails own their URLs; never revoke those
  film.file = file;
  film.result = result;
  const viewable = file && file.type.startsWith("image/") && !/\.dcm$/i.test(file.name);
  if (result?.__url) {
    film.url = result.__url;
    film.owned = false;
  } else {
    film.url = viewable ? URL.createObjectURL(file) : null;
    film.owned = Boolean(film.url);
  }
  lightbox.classList.add("has-image");
  $("#film").hidden = false;
  const img = $("#film-img");
  const heat = $("#film-heat");
  heat.hidden = true;
  heat.classList.remove("developed");
  $("#regions").replaceChildren();
  img.src = film.url || (result && result.heatmap_png) || "";
  img.alt = file ? `Image being screened: ${file.name}` : "Image being screened";
  img.onload = () => {
    const kb = file ? `${Math.max(1, Math.round(file.size / 1024))} KB` : "";
    $("#film-meta").textContent = `${file ? file.name : ""}  ${img.naturalWidth} × ${img.naturalHeight} px  ${kb}`.trim();
  };
  if (!film.url && !result) $("#film-meta").textContent = `${file.name}: DICOM preview appears after screening`;
  $("#screen-btn").disabled = !file;
  $("#change-btn").hidden = false;
  if (!result) resetSlip();
}

function chooseFile() { $("#file").click(); }
$("#file").addEventListener("change", (e) => e.target.files[0] && setFilm(e.target.files[0]));
$("#change-btn").addEventListener("click", chooseFile);
["dragenter", "dragover"].forEach((ev) =>
  lightbox.addEventListener(ev, (e) => { e.preventDefault(); $("#drop").classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  lightbox.addEventListener(ev, (e) => { e.preventDefault(); $("#drop").classList.remove("over"); }));
lightbox.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) setFilm(f);
});
document.addEventListener("paste", (e) => {
  if ($("#tab-screen").hidden) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) setFilm(new File([item.getAsFile()], "pasted-image.png", { type: item.type }));
});
$("#heatmap").addEventListener("change", () => lightbox.classList.toggle("heat-off", !$("#heatmap").checked));
$("#screen-btn").addEventListener("click", () => screenCurrent());

async function screenFile(file, { task, site, lang }) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("task", task);
  fd.append("lang", lang);
  fd.append("heatmap", "true");
  if (site) fd.append("site_id", site);
  return api("/v1/screen", { method: "POST", body: fd });
}

async function screenCurrent() {
  if (!film.file) return;
  const btn = $("#screen-btn");
  btn.disabled = true;
  btn.textContent = "Screening…";
  lightbox.classList.add("scanning");
  $("#slip").setAttribute("aria-busy", "true");
  try {
    const r = await screenFile(film.file, { task: currentTask(), site: currentSite(), lang: $("#lang").value });
    r.__site = currentSite();
    film.result = r;
    showResult(r, { animate: true });
  } catch (err) {
    renderSlipError(err);
  } finally {
    lightbox.classList.remove("scanning");
    $("#slip").setAttribute("aria-busy", "false");
    btn.disabled = false;
    btn.textContent = "Screen again";
  }
}

function showResult(r, { animate }) {
  const heat = $("#film-heat");
  const img = $("#film-img");
  if (!img.getAttribute("src") && r.heatmap_png) img.src = r.heatmap_png;
  if (r.heatmap_png) {
    heat.src = r.heatmap_png;
    heat.hidden = false;
    heat.classList.remove("developed");
    void heat.offsetWidth; // restart the develop animation
    heat.classList.add("developed");
  }
  const regions = $("#regions");
  regions.replaceChildren(
    ...(r.regions || []).map((g) =>
      el("div", {
        class: "region",
        style: `left:${g.x * 100}%;top:${g.y * 100}%;width:${g.w * 100}%;height:${g.h * 100}%;opacity:${0.45 + 0.55 * g.weight}`,
      })),
  );
  renderSlip(r, { animate });
}

/* ================================================================ screen: the slip */
function resetSlip() {
  const slip = $("#slip");
  slip.className = "slip";
  $("#slip-empty").hidden = false;
  $("#slip-body").hidden = true;
  $("#slip-body").replaceChildren();
}

function renderSlipError(err) {
  const slip = $("#slip");
  slip.className = "slip";
  $("#slip-empty").hidden = true;
  const body = $("#slip-body");
  body.hidden = false;
  const msg = err.status === 401
    ? "This server needs an API key. Use the key button at the top right."
    : err.status === 413 ? "The file is larger than this server accepts."
    : err.status === 422 ? `The image could not be read: ${err.message}`
    : `The screening service did not answer: ${err.message}`;
  body.replaceChildren(
    el("p", { class: "slip-empty-title", text: "Not screened" }),
    el("p", { class: "muted", text: msg, style: "margin-top:8px" }),
  );
}

function renderSlip(r, { animate = true } = {}) {
  const slip = $("#slip");
  const task = r.task;
  const decision = r.decision;
  slip.className = "slip";
  $("#slip-empty").hidden = true;
  const body = $("#slip-body");
  body.hidden = false;

  const when = new Date();
  const site = r.__site !== undefined ? r.__site : currentSite();
  const head = el("header", { class: "slip-head" },
    el("p", { class: "slip-title", text: "Screening slip" }),
    el("p", { class: "slip-meta", text: `${site ? siteName(site) : "Clinic not onboarded"}, ${when.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })} ${when.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}` }),
  );

  const stamp = el("div", { class: `stamp ${decision}`, role: "img", "aria-label": `Decision: ${DECISION[decision].label}` },
    icon(DECISION[decision].icon), el("span", { text: DECISION[decision].label }));
  let next;
  if (decision === "uncertain") {
    const failed = (r.quality_gate?.checks || []).filter((c) => !c.passed);
    next = r.reason === "quality_gate"
      ? ["Retake the image", failed.length ? failed[0].message : REASON.quality_gate]
      : ["Ask a senior clinician to read this image", REASON[r.reason] || ""];
  } else {
    next = NEXT_STEP[task][decision];
  }
  const verdict = el("div", { class: "verdict" }, stamp,
    el("p", { class: "next-step" }, next[0], el("small", { text: next[1] })));

  const note = el("p", { class: "note", lang: r.note.lang, text: r.note.text });
  const draft = r.note.translation_reviewed ? null
    : el("p", { class: "draft", text: "Draft translation. A native-speaking clinician has not reviewed it yet." });
  const warnings = (r.warnings || []).length
    ? el("ul", { class: "warnings" }, r.warnings.map((w) => el("li", {}, icon("i-alert"), el("span", { text: w }))))
    : null;

  const sections = [why(r), qualitySection(r), similarSection(r)].filter(Boolean);

  const printFilm = r.heatmap_png
    ? el("img", { class: "print-only print-film", src: r.heatmap_png, alt: "Screened image with heatmap" }) : null;

  const outcomeMsg = el("span", { class: "muted" });
  const recordBtn = (y, label) => el("button", {
    class: "button quiet", type: "button", text: label,
    onclick: async () => {
      try {
        await api(`/v1/screenings/${r.request_id}/outcome`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ outcome: y }),
        });
        outcomeMsg.textContent = `Saved as confirmed ${y ? "positive" : "negative"}.`;
        toast("Confirmed result saved. It now counts in monitoring.");
      } catch (err) { outcomeMsg.textContent = err.message; }
    },
  });
  const foot = el("footer", { class: "slip-foot" },
    el("button", { class: "button quiet", type: "button", onclick: () => window.print() }, icon("i-print"), "Print slip"),
  );
  const outcome = r.request_id ? el("div", { class: "outcome-row" },
    el("span", { text: "Confirmatory test came back:" }), recordBtn(1, "Positive"), recordBtn(0, "Negative"), outcomeMsg) : null;
  const fine = el("p", { class: "fine-print", text:
    `${r.disclaimer} Reference ${r.request_id || "not logged"}. Model ${r.model.bundle_id} with ${r.model.backbone}; settings from ${r.calibration === "training-sites" ? "the training hospitals" : r.calibration.replace(/^site:/, "clinic ").replace(/:v/, ", version ")}${r.adaptation === "adapt" ? " with a local adaptation layer" : ""}. ${r.latency_ms} ms.` });

  body.replaceChildren(...[head, printFilm, verdict, note, draft, warnings, ...sections, foot, outcome, fine].filter(Boolean));

  const tint = () => slip.classList.add(decision);
  if (animate && !reduceMotion()) {
    requestAnimationFrame(() => stamp.classList.add("pressed"));
    setTimeout(tint, 160);
  } else {
    tint();
  }
}

function why(r) {
  if (r.probability === null || r.probability === undefined) return null;
  const p = Math.min(Math.max(r.probability, 0), 1);
  const thr = Math.min(Math.max(r.threshold, 0), 1);
  const meter = el("div", { class: "meter", style: `--p:${p * 100}%;--thr:${thr * 100}%`, role: "img",
    "aria-label": `Score ${num(p)} against a referral line of ${num(thr)}` },
  el("div", { class: "meter-track" }), el("div", { class: "meter-thr" }), el("div", { class: "meter-dot" }));
  const set = r.prediction_set || [];
  const setText = set.length === 2 ? "both outcomes" : set.length === 0 ? "neither outcome" : `only ${DECISION[set[0]].label.toLowerCase()}`;
  const lines = [
    el("p", { class: "why-text" }, "Score ", el("b", { text: num(p) }), ` against this clinic's referral line of ${num(thr)}. The 90% confidence check allows `, el("b", { text: setText }), "."),
    el("p", { class: "why-text", text: REASON[r.reason] || "" }),
  ];
  if (r.ood && r.ood.flagged) lines.push(el("p", { class: "why-text", text: "This image is unfamiliar to the model, so it is not trusted to give a firm answer." }));
  if ((r.regions || []).length) lines.push(el("p", { class: "why-text", text: "The heatmap shows which parts of the image moved the score: red towards referral, blue away from it. The yellow boxes mark the strongest areas. It explains the score; it does not outline disease." }));
  return el("section", { class: "slip-section" }, el("h3", { text: "Why" }), meter,
    el("div", { class: "meter-legend" }, el("span", { text: "0, low risk" }), el("span", { text: "1, refer" })), ...lines);
}

function qualitySection(r) {
  const checks = r.quality_gate?.checks || [];
  if (!checks.length) return null;
  const failed = checks.filter((c) => !c.passed);
  const list = (cs) => el("ul", { class: "checks" }, cs.map((c) =>
    el("li", { class: c.passed ? "" : "fail" }, el("span", { text: c.passed ? labelCheck(c.name) : `${labelCheck(c.name)}: ${c.message}` }))));
  if (failed.length) {
    return el("section", { class: "slip-section" }, el("h3", { text: "Image quality" }), list(failed),
      checks.length > failed.length ? el("p", { class: "checks-summary muted", text: `${checks.length - failed.length} other checks passed.` }) : null);
  }
  return el("section", { class: "slip-section" },
    el("details", { class: "checks-detail" },
      el("summary", { class: "checks-summary", text: `All ${checks.length} image quality checks passed` }), list(checks)));
}
function labelCheck(name) {
  const map = {
    resolution: "Resolution", contrast: "Contrast", exposure: "Exposure", greyscale: "Greyscale X-ray",
    aspect_ratio: "Frontal framing", sharpness: "Sharpness", lungs_visible: "Both lung fields visible",
    field_of_view: "Retina in view", colour: "Retinal colour", illumination: "Even illumination",
    learned_gradability: "Gradable, per the learned quality model",
  };
  return map[name] || name.replace(/_/g, " ");
}

function similarSection(r) {
  const cases = r.similar_cases || [];
  if (!cases.length) return null;
  const t = TASKS[r.task];
  const pos = cases.filter((c) => c.label === 1).length;
  return el("section", { class: "slip-section" },
    el("h3", { text: "Most similar reference cases" }),
    el("p", { class: "similar-sum", text: `${pos} of the ${cases.length} closest confirmed cases from the training hospitals were ${t.positive === "TB" ? "TB" : "referable"}.` }),
    el("div", { class: "similar-grid" }, cases.map((c) => el("figure", { class: "case" },
      el("img", { src: `/v1/${c.thumb}`, alt: `${c.label ? t.positive : t.negative} reference case from ${c.dataset}`, loading: "lazy" }),
      el("figcaption", {},
        el("span", { class: `tag ${c.label ? "pos" : "neg"}`, text: c.label ? t.positive : t.negative }),
        el("span", { text: c.dataset }))))),
  );
}

/* ================================================================ queue */
const queue = { items: [], running: 0, seq: 0 };
const PRIORITY = { refer: 0, uncertain: 1, low_risk: 2 };

$("#queue-files").addEventListener("change", (e) => {
  addToQueue([...e.target.files]);
  e.target.value = "";
});
$("#tab-queue").addEventListener("dragover", (e) => e.preventDefault());
$("#tab-queue").addEventListener("drop", (e) => {
  e.preventDefault();
  addToQueue([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/") || /\.dcm$/i.test(f.name)));
});
$("#queue-clear").addEventListener("click", () => {
  queue.items.filter((i) => i.status !== "pending").forEach((i) => i.url && URL.revokeObjectURL(i.url));
  queue.items = queue.items.filter((i) => i.status === "pending");
  renderQueue();
});
$("#queue-export").addEventListener("click", exportQueue);

function addToQueue(files) {
  const ctx = { task: currentTask(), site: currentSite(), lang: $("#lang").value };
  for (const f of files) {
    const viewable = f.type.startsWith("image/") && !/\.dcm$/i.test(f.name);
    queue.items.push({ id: ++queue.seq, file: f, url: viewable ? URL.createObjectURL(f) : null, status: "pending", ctx, fresh: true });
  }
  renderQueue();
  pump();
}
function pump() {
  while (queue.running < 2) {
    const next = queue.items.find((i) => i.status === "pending" && !i.started);
    if (!next) break;
    next.started = true;
    queue.running++;
    screenFile(next.file, next.ctx)
      .then((r) => { r.__site = next.ctx.site; r.__url = next.url; next.result = r; next.status = "done"; })
      .catch((err) => { next.status = "error"; next.error = err.message; })
      .finally(() => { queue.running--; next.fresh = true; renderQueue(); pump(); });
  }
}
function sortedQueue() {
  const rank = (i) => (i.status === "done" ? PRIORITY[i.result.decision] : i.status === "error" ? 3 : 4);
  return [...queue.items].sort((a, b) => rank(a) - rank(b)
    || ((b.result?.probability ?? -1) - (a.result?.probability ?? -1)) || a.id - b.id);
}
function renderQueue() {
  const items = sortedQueue();
  const has = items.length > 0;
  $("#queue-table").hidden = !has;
  $("#queue-empty").hidden = has;
  $("#queue-export").disabled = !items.some((i) => i.status === "done");
  $("#queue-clear").disabled = !items.some((i) => i.status !== "pending");
  const done = items.filter((i) => i.status === "done");
  const count = (d) => done.filter((i) => i.result.decision === d).length;
  const tally = $("#queue-tally");
  tally.hidden = !has;
  const pending = items.length - done.length - items.filter((i) => i.status === "error").length;
  tally.replaceChildren(...[
    ...["refer", "uncertain", "low_risk"].map((d) => el("span", {}, el("b", { text: count(d) }), el("span", { class: `pill ${d}` }, icon(DECISION[d].icon), DECISION[d].label))),
    pending ? el("span", { class: "muted" }, `${pending} still screening`) : null,
  ].filter(Boolean));
  const referCount = count("refer");
  const badge = $("#queue-count");
  badge.hidden = !referCount;
  badge.textContent = referCount;
  badge.setAttribute("aria-label", `${referCount} to refer`);
  $("#queue-body").replaceChildren(...items.map(queueRow));
}
function queueRow(item) {
  const r = item.result;
  let result;
  if (item.status === "pending") result = el("span", { class: "pill pending", text: item.started ? "Screening" : "Waiting" });
  else if (item.status === "error") result = el("span", { class: "pill not_ready", text: "Failed" });
  else result = el("span", { class: `pill ${r.decision}` }, icon(DECISION[r.decision].icon), DECISION[r.decision].label);
  const tr = el("tr", { tabindex: item.status === "done" ? "0" : null, class: item.fresh ? "entering" : "" },
    el("td", {}, el("div", { class: "thumb-cell" },
      item.url || r?.heatmap_png ? el("img", { src: item.url || r.heatmap_png, alt: "" }) : el("img", { alt: "" }),
      el("span", { text: item.file.name, title: item.file.name }))),
    el("td", {}, result),
    el("td", { class: "num", text: r && r.probability !== null && r.probability !== undefined ? num(r.probability) : "–" }),
    el("td", { text: item.status === "error" ? item.error : r ? SHORT_REASON[r.reason] || r.reason : "" }),
    el("td", { class: "num", text: r ? `${Math.round(r.latency_ms)} ms` : "" }),
  );
  item.fresh = false;
  if (item.status === "done") {
    const open = () => openQueued(item);
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  }
  return tr;
}
function openQueued(item) {
  showTab("screen");
  const radio = $(`input[name=task][value=${item.result.task}]`);
  if (radio) radio.checked = true;
  setFilm(item.file, { result: item.result });
  showResult(item.result, { animate: true });
  $("#slip").scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "nearest" });
}
function exportQueue() {
  const rows = [["file", "decision", "reason", "score", "referral_line", "clinic", "calibration", "reference"]];
  for (const i of sortedQueue().filter((x) => x.status === "done")) {
    const r = i.result;
    rows.push([i.file.name, r.decision, r.reason, r.probability ?? "", r.threshold ?? "", i.ctx.site || "", r.calibration, r.request_id || ""]);
  }
  const csv = rows.map((row) => row.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\r\n");
  const a = el("a", { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: `clinictune-queue-${new Date().toISOString().slice(0, 10)}.csv` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

/* ================================================================ clinics */
async function loadSites() {
  try {
    state.sites = await api("/v1/sites");
  } catch { return; }
  const opts = (withNone) => [
    ...(withNone ? [el("option", { value: "", text: "Not onboarded" })] : []),
    ...state.sites.map((s) => el("option", { value: s.site_id, text: s.name })),
  ];
  const keep = $("#site-select").value || pref("site") || "";
  $("#site-select").replaceChildren(...opts(true));
  if (state.sites.some((s) => s.site_id === keep)) $("#site-select").value = keep;
  for (const sel of ["#onboard-site", "#mon-site"]) {
    const k = $(sel).value;
    $(sel).replaceChildren(...opts(false));
    if (k) $(sel).value = k;
  }
  const list = $("#sites-list");
  if (!state.sites.length) {
    list.replaceChildren(el("p", { class: "muted", text: "No clinics registered yet. Register one, then onboard it with a pack of locally confirmed images." }));
    return;
  }
  list.replaceChildren(...state.sites.map((s) => el("div", { class: "site-row" },
    el("header", {}, el("strong", { text: s.name }), el("span", { class: "site-id", text: s.site_id })),
    s.calibrations.length
      ? el("div", { class: "cal-list" }, s.calibrations.map((c) => calRow(s, c)))
      : el("p", { class: "muted small", text: "Not onboarded yet. Screening here uses the training hospitals' settings." }),
  )));
}
function calRow(site, c) {
  const verdictText = { ready: "Ready", use_with_caution: "Use with caution", not_ready: "Not ready" }[c.verdict] || c.verdict;
  return el("div", { class: "cal" },
    el("span", { class: "cal-name", text: `${TASKS[c.task].short}, v${c.version}` }),
    el("span", { class: `pill ${c.verdict}`, text: verdictText }),
    c.active ? el("span", { class: "pill active", text: "In use" }) : null,
    el("button", { class: "link", type: "button", text: "Report card", onclick: () => showReport(site.site_id, c.task, c.version) }),
    c.active ? null : el("button", { class: "link", type: "button", text: "Use this version", onclick: async () => {
      await api(`/v1/sites/${site.site_id}/calibrations/${c.task}/${c.version}/activate`, { method: "POST" });
      toast(`${site.name} now uses version ${c.version} for ${TASKS[c.task].short.toLowerCase()}s.`);
      loadSites();
    } }),
  );
}
async function showReport(siteId, task, version) {
  try {
    const html = await api(`/v1/sites/${encodeURIComponent(siteId)}/report.html?task=${task}&version=${version}`);
    $("#report-frame").srcdoc = html;
    $("#report-title").textContent = `Report card: ${siteName(siteId)}, ${TASKS[task].short.toLowerCase()}, version ${version}`;
    $("#report-wrap").hidden = false;
    $("#report-wrap").scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth" });
  } catch (err) { toast(err.message); }
}
$("#print-report").addEventListener("click", () => $("#report-frame").contentWindow.print());

$("#site-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#site-msg");
  msg.className = "form-msg";
  try {
    const s = await api("/v1/sites", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ site_id: $("#site-id").value.trim(), name: $("#site-name").value.trim() }) });
    e.target.reset();
    msg.textContent = `${s.name} registered. Onboard it next.`;
    await loadSites();
    $("#onboard-site").value = s.site_id;
  } catch (err) {
    msg.className = "form-msg error";
    msg.textContent = err.status === 409 ? "A clinic with that ID already exists." : err.message;
  }
});

$("#onboard-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("#onboard-file").files[0];
  const siteId = $("#onboard-site").value;
  const task = $("#onboard-task").value;
  const msg = $("#onboard-msg");
  msg.className = "form-msg";
  if (!siteId) { msg.textContent = "Register a clinic first."; return; }
  setSteps(-1);
  $("#onboard-steps").hidden = false;
  msg.textContent = "Uploading the pack…";
  const fd = new FormData();
  fd.append("task", task);
  fd.append("pack", file);
  try {
    const j = await api(`/v1/sites/${encodeURIComponent(siteId)}/onboard`, { method: "POST", body: fd });
    msg.textContent = `Pack accepted: ${j.images} images.`;
    pollJob(j.job_id, siteId, task);
  } catch (err) {
    msg.className = "form-msg error";
    msg.textContent = err.message;
    setSteps(-1);
  }
});
function setSteps(active, failed = false) {
  $$("#onboard-steps li").forEach((li, i) => {
    li.className = i < active ? "done" : i === active ? (failed ? "failed" : "active") : "";
  });
}
function stepFor(job) {
  if (job.status === "done") return 5;
  if (job.progress < 0.3) return 0;
  if (job.progress < 0.9) return 1;
  return (job.message || "").includes("recalibrat") ? 3 : 2;
}
async function pollJob(id, siteId, task) {
  let job;
  try { job = await api(`/v1/jobs/${id}`); } catch (err) { $("#onboard-msg").textContent = err.message; return; }
  const msg = $("#onboard-msg");
  if (job.status === "failed") {
    setSteps(stepFor(job), true);
    msg.className = "form-msg error";
    msg.textContent = `Onboarding stopped: ${job.message}`;
    return;
  }
  setSteps(stepFor(job));
  msg.textContent = job.status === "done" ? "Done. The new settings are in use." : job.message || "Waiting to start";
  if (job.status === "done") {
    await loadSites();
    toast(`${siteName(siteId)} onboarded: ${({ ready: "ready", use_with_caution: "use with caution", not_ready: "not ready" })[job.result.verdict]}.`);
    showReport(siteId, task, job.result.version);
    return;
  }
  setTimeout(() => pollJob(id, siteId, task), 900);
}

/* ================================================================ tooltips for charts */
const tip = el("div", { class: "tip", role: "tooltip" });
document.body.append(tip);
function bindTip(node, title, rows) {
  node.classList.add("chart-mark");
  node.setAttribute("tabindex", "0");
  node.setAttribute("aria-label", `${title}: ${rows.map((r) => `${r.label} ${r.value}`).join(", ")}`);
  const show = (x, y) => {
    tip.replaceChildren(el("div", { class: "tip-title", text: title }),
      ...rows.map((r) => el("div", { class: "tip-row" }, el("i", { style: `background:${r.color || "transparent"}` }), el("span", { text: r.label }), el("b", { text: r.value }))));
    tip.classList.add("show");
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    tip.style.left = `${Math.min(Math.max(8, x + 14), window.innerWidth - w - 8)}px`;
    tip.style.top = `${Math.max(8, y - h - 12)}px`;
  };
  node.addEventListener("pointermove", (e) => show(e.clientX, e.clientY));
  node.addEventListener("focus", () => { const b = node.getBoundingClientRect(); show(b.right, b.top); });
  node.addEventListener("pointerleave", () => tip.classList.remove("show"));
  node.addEventListener("blur", () => tip.classList.remove("show"));
}
function legend(items) {
  return el("div", { class: "legend" }, items.map((i) => el("span", {},
    i.line ? el("i", { style: `height:3px;width:16px;background:${i.color}` }) : el("i", { style: `background:${i.color}` }),
    i.label)));
}
function tableView(head, rows, caption = "Show the numbers") {
  return el("details", { class: "table-view" }, el("summary", { text: caption }),
    el("table", { class: "data-table" },
      el("thead", {}, el("tr", {}, head.map((h) => el("th", { scope: "col", text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c) => (c instanceof Node ? el("td", {}, c) : el("td", { text: c }))))))));
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function niceMax(v) {
  if (v <= 5) return 5;
  const p = 10 ** Math.floor(Math.log10(v));
  return Math.ceil(v / p) * p;
}

/* ================================================================ monitoring */
let monSeq = 0;
["#mon-site", "#mon-task", "#mon-days"].forEach((s) => $(s).addEventListener("change", loadMonitoring));
async function loadMonitoring() {
  const site = $("#mon-site").value;
  if (!site) {
    $("#mon-alerts").replaceChildren();
    $("#mon-stats").replaceChildren(el("p", { class: "muted", text: "Register and onboard a clinic to see its monitoring." }));
    $("#mon-chart-fig").hidden = true;
    return;
  }
  const seq = ++monSeq;
  let m;
  try {
    m = await api(`/v1/sites/${encodeURIComponent(site)}/monitoring?task=${$("#mon-task").value}&days=${$("#mon-days").value}`);
  } catch (err) { $("#mon-stats").replaceChildren(el("p", { class: "muted", text: err.message })); return; }
  if (seq !== monSeq) return;
  const alerts = m.alerts || [];
  const alertNodes = alerts.map((a) => el("div", { class: `alert ${a.level}` }, icon("i-alert"), el("p", { text: a.message })));
  if (!alerts.length && m.n) alertNodes.push(el("div", { class: "alert ok" }, icon("i-low"), el("p", { text: "No alerts. Results at this clinic look consistent with its onboarding." })));
  $("#mon-alerts").replaceChildren(...alertNodes);
  if (!m.n) {
    $("#mon-stats").replaceChildren(el("p", { class: "muted", text: "No screenings at this clinic in this period." }));
    $("#mon-chart-fig").hidden = true;
    return;
  }
  const stats = [
    [m.n, "screenings"],
    [pct(m.decision_rates.refer), "referred"],
    [pct(m.decision_rates.uncertain), "uncertain"],
    [pct(m.decision_rates.low_risk), "low risk"],
    [pct(m.gate_fail_rate), "failed the quality check"],
    [pct(m.ood_rate), "unfamiliar images"],
    [m.latency_ms_p50 ? `${Math.round(m.latency_ms_p50)} ms` : "–", "median time per image"],
    [m.confirmed_outcomes, "confirmed test results entered"],
  ];
  if (m.observed_sensitivity !== undefined) stats.push([pct(m.observed_sensitivity), "of confirmed positives were referred"]);
  if (m.observed_specificity !== undefined) stats.push([pct(m.observed_specificity), "of confirmed negatives were low risk"]);
  $("#mon-stats").replaceChildren(...stats.map(([v, l]) => el("div", { class: "stat" }, el("b", { text: v }), el("span", { text: l }))));
  $("#mon-note").textContent = m.note || "";
  $("#mon-chart").replaceChildren(dailyChart(m.daily || [], Number($("#mon-days").value)));
  $("#mon-chart-fig").hidden = false;
}

function dailyChart(daily, days) {
  const end = new Date();
  const dates = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(end);
    d.setDate(d.getDate() - i);
    dates.push(d.toISOString().slice(0, 10));
  }
  const byDate = Object.fromEntries(daily.map((d) => [d.date, d]));
  const W = 760, H = 240, L = 36, R = 8, T = 10, B = 26;
  const band = (W - L - R) / dates.length;
  const bw = Math.min(24, Math.max(3, band - 4));
  const max = niceMax(Math.max(1, ...daily.map((d) => d.n)));
  const y = (v) => T + (H - T - B) * (1 - v / max);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Screening results per day, stacked by decision" });
  const series = [["low_risk", "--fill-low"], ["uncertain", "--fill-unc"], ["refer", "--fill-refer"]];
  for (let t = 0; t <= 4; t++) {
    const v = (max / 4) * t;
    svg.append(sv("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: cssVar("--rule"), "stroke-width": 1 }),
      sv("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", text: Math.round(v) }));
  }
  dates.forEach((date, i) => {
    const d = byDate[date];
    const x = L + band * i + (band - bw) / 2;
    if (i % Math.ceil(dates.length / 8) === 0 || i === dates.length - 1) {
      svg.append(sv("text", { x: x + bw / 2, y: H - 8, "text-anchor": "middle", text: new Date(`${date}T12:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short" }) }));
    }
    if (!d) return;
    const g = sv("g", {});
    let acc = 0;
    const top = series.filter(([k]) => d[k]).at(-1)?.[0];
    for (const [k, color] of series) {
      const v = d[k] || 0;
      if (!v) continue;
      const y0 = y(acc), y1 = y(acc + v);
      const h = Math.max(1, y0 - y1 - (k === top ? 0 : 2));
      if (k === top) {
        const r = Math.min(4, h / 2, bw / 2);
        g.append(sv("path", { d: `M${x},${y0} V${y1 + r} Q${x},${y1} ${x + r},${y1} H${x + bw - r} Q${x + bw},${y1} ${x + bw},${y1 + r} V${y0} Z`, fill: cssVar(color) }));
      } else {
        g.append(sv("rect", { x, y: y1 + 2, width: bw, height: h, fill: cssVar(color) }));
      }
      acc += v;
    }
    g.append(sv("rect", { x: x - 2, y: T, width: bw + 4, height: H - T - B, fill: "transparent" }));
    bindTip(g, new Date(`${date}T12:00:00`).toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" }), [
      { label: "Refer", value: d.refer || 0, color: cssVar("--fill-refer") },
      { label: "Uncertain", value: d.uncertain || 0, color: cssVar("--fill-unc") },
      { label: "Low risk", value: d.low_risk || 0, color: cssVar("--fill-low") },
    ]);
    svg.append(g);
  });
  svg.append(sv("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: cssVar("--rule-strong"), "stroke-width": 1 }));
  return el("div", {}, svg,
    legend([{ label: "Refer", color: cssVar("--fill-refer") }, { label: "Uncertain", color: cssVar("--fill-unc") }, { label: "Low risk", color: cssVar("--fill-low") }]),
    tableView(["Date", "Screened", "Refer", "Uncertain", "Low risk", "Unfamiliar", "Failed quality"],
      daily.map((d) => [d.date, d.n, d.refer || 0, d.uncertain || 0, d.low_risk || 0, d.ood, d.gate_fail])));
}

/* ================================================================ validation (evidence) */
async function loadValidation() {
  if (state.evidence !== undefined) return;
  const root = $("#validation");
  try {
    state.evidence = await api("/v1/evidence");
  } catch (err) {
    state.evidence = null;
    root.replaceChildren(el("p", { class: "muted", text: err.status === 404
      ? "This model bundle ships without validation results. Run the leave-one-site-out experiments and `clinictune evidence` to add them."
      : err.message }));
    return;
  }
  renderValidation(root, state.evidence);
}

const STRATEGY = {
  source_calibration: "Training hospitals' settings",
  site_recalibration: "Recalibrated with local labels",
  site_adapt: "Local adaptation layer",
  site_only_head: "Model trained only on local labels",
};
const SITE_LABEL = {
  shenzhen: "Shenzhen, China", montgomery: "Montgomery County, USA", tbx11k: "TBX11K hospitals, China",
  ddr: "DDR, 147 hospitals in China", aptos2019: "Aravind Eye Hospital, India (APTOS)", idrid: "IDRiD, Nanded, India",
};
const siteLabel = (s) => SITE_LABEL[s] || s;

function deployedRun(ev, task) {
  const runs = ev.loso?.[task] || [];
  const deployed = state.info?.tasks?.[task]?.backbone;
  return runs.find((r) => r.backbone === deployed) || runs[runs.length - 1];
}
function rowsFor(run, strategy, k) {
  return (run?.summary || []).filter((r) => r.strategy === strategy && (k === undefined || r.k_labels === k));
}

function renderValidation(root, ev) {
  const tasks = Object.keys(ev.loso || {});
  const findings = computeFindings(ev, tasks);
  const sections = [];
  if (findings.length) sections.push(el("div", { class: "findings" }, findings.map((f) => el("div", { class: "finding" }, el("b", { text: f.value }), el("p", { text: f.text })))));

  const body = el("div", {});
  const switcher = el("div", { class: "task-switch", role: "group", "aria-label": "Condition" },
    tasks.map((t, i) => el("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false", text: TASKS[t]?.short || t,
      onclick: (e) => {
        $$(".task-switch button", root).forEach((b) => b.setAttribute("aria-pressed", "false"));
        e.currentTarget.setAttribute("aria-pressed", "true");
        body.replaceChildren(...taskEvidence(ev, t));
      } })));
  sections.push(switcher, body);
  root.replaceChildren(...sections);
  if (tasks.length) body.replaceChildren(...taskEvidence(ev, tasks[0]));
}

function computeFindings(ev, tasks) {
  const out = [];
  let sites = 0, images = 0, broke = 0, tested = 0, fixed = 0, bestGain = null;
  for (const t of tasks) {
    const ds = ev.datasets?.[t] || {};
    sites += Object.keys(ds).length;
    images += Object.values(ds).reduce((a, d) => a + (d.images || 0), 0);
    const run = deployedRun(ev, t);
    const before = rowsFor(run, "source_calibration", 0);
    const kMax = Math.max(0, ...rowsFor(run, "site_recalibration").map((r) => r.k_labels));
    const after = rowsFor(run, "site_recalibration", kMax);
    for (const b of before) {
      tested++;
      if (Math.min(b.coverage_neg ?? 1, b.coverage_pos ?? 1) < 0.85) {
        broke++;
        const a = after.find((x) => x.held_out_site === b.held_out_site);
        if (a && Math.min(a.coverage_neg ?? 0, a.coverage_pos ?? 0) >= 0.85) fixed++;
      }
    }
    for (const a of rowsFor(run, "site_adapt")) {
      const b = before.find((x) => x.held_out_site === a.held_out_site);
      if (b && a.roc_auc !== undefined && b.roc_auc !== undefined) {
        const g = a.roc_auc - b.roc_auc;
        if (bestGain === null || g > bestGain.g) bestGain = { g, site: a.held_out_site, k: a.k_labels, from: b.roc_auc, to: a.roc_auc };
      }
    }
  }
  if (images) out.push({ value: images.toLocaleString(), text: `public images from ${sites} hospitals or hospital networks, in China, India and the USA.` });
  if (tested) out.push({ value: `${broke} of ${tested}`, text: "unseen hospitals broke the 90% confidence guarantee when the training hospitals' settings were used unchanged." });
  if (broke) out.push({ value: `${fixed} of ${broke}`, text: "of those were back near the 90% target after recalibrating with local labels." });
  if (bestGain && bestGain.g > 0.005) out.push({ value: `+${bestGain.g.toFixed(2)}`, text: `ROC-AUC from the local adaptation layer at ${siteLabel(bestGain.site)} (${bestGain.from.toFixed(2)} to ${bestGain.to.toFixed(2)} with ${bestGain.k} labels). Recalibration alone cannot change ranking.` });
  return out;
}

function taskEvidence(ev, task) {
  const run = deployedRun(ev, task);
  const out = [];
  const ds = ev.datasets?.[task];
  if (ds) {
    out.push(el("section", { class: "v-section" },
      el("h2", { text: "Hospitals and images" }),
      el("p", { text: "Each source is treated as a separate hospital. When one is being tested, the model is trained on the others only." }),
      el("table", { class: "data-table" },
        el("thead", {}, el("tr", {}, ["Hospital", "Images", TASKS[task].positive, TASKS[task].negative].map((h) => el("th", { text: h })))),
        el("tbody", {}, Object.entries(ds).map(([s, d]) => el("tr", {}, [siteLabel(s), d.images.toLocaleString(), d.positive.toLocaleString(), d.negative.toLocaleString()].map((v) => el("td", { text: v }))))))));
  }
  if (run) {
    out.push(el("section", { class: "v-section" },
      el("h2", { text: "Ranking at a hospital the model never saw" }),
      el("p", { text: `ROC-AUC on the held-out half of each hospital, as more local labels are used (${run.backbone}, mean of ${run.repeats} repeats). Recalibration keeps the training model's ranking; the local adaptation layer and a model trained only on local labels can change it.` }),
      legend([
        { label: "Training model, recalibrated", color: cssVar("--series-1"), line: true },
        { label: "Local adaptation layer", color: cssVar("--series-2"), line: true },
        { label: "Trained only on local labels", color: cssVar("--series-3"), line: true },
      ]),
      el("div", { class: "smalls", style: "margin-top:12px" }, [...new Set(run.summary.map((r) => r.held_out_site))].map((s) => aucChart(run, s))),
    ));
    out.push(coverageSection(run, task));
    out.push(detailTable(run, task));
  }
  const bb = backboneSection(ev, task);
  if (bb) out.push(bb);
  const sc = shortcutSection(ev, task);
  if (sc) out.push(sc);
  const sg = subgroupSection(ev, task);
  if (sg) out.push(sg);
  return out;
}

function aucChart(run, site) {
  const rows = run.summary.filter((r) => r.held_out_site === site);
  const ks = [0, ...run.ks].filter((k, i, a) => a.indexOf(k) === i).sort((a, b) => a - b);
  const src = rows.find((r) => r.strategy === "source_calibration");
  const series = [
    { key: "recal", color: cssVar("--series-1"), label: "Recalibrated", pts: [src, ...rows.filter((r) => r.strategy === "site_recalibration")] },
    { key: "adapt", color: cssVar("--series-2"), label: "Local adaptation", pts: [src, ...rows.filter((r) => r.strategy === "site_adapt")] },
    { key: "local", color: cssVar("--series-3"), label: "Local labels only", pts: rows.filter((r) => r.strategy === "site_only_head") },
  ].map((s) => ({ ...s, pts: s.pts.filter((p) => p && p.roc_auc !== undefined && p.roc_auc !== null).sort((a, b) => a.k_labels - b.k_labels) }));
  const vals = series.flatMap((s) => s.pts.map((p) => p.roc_auc));
  const lo = Math.min(0.5, Math.floor((Math.min(...vals) - 0.02) * 20) / 20);
  const W = 300, H = 190, L = 34, R = 10, T = 10, B = 28;
  const x = (k) => L + (W - L - R) * (ks.indexOf(k) / Math.max(1, ks.length - 1));
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / (1 - lo));
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `ROC-AUC by number of local labels at ${siteLabel(site)}` });
  for (let v = lo; v <= 1.0001; v += (1 - lo) / 4) {
    svg.append(sv("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: cssVar("--rule"), "stroke-width": 1 }),
      sv("text", { x: L - 5, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(2) }));
  }
  ks.forEach((k) => svg.append(sv("text", { x: x(k), y: H - 10, "text-anchor": "middle", text: k })));
  svg.append(sv("text", { x: W - R, y: H - 10 + 12, "text-anchor": "end", text: "" }));
  for (const s of series) {
    if (s.pts.length > 1) {
      svg.append(sv("polyline", { points: s.pts.map((p) => `${x(p.k_labels)},${y(p.roc_auc)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
  }
  for (const s of series) {
    for (const p of s.pts) {
      const g = sv("g", {});
      g.append(sv("circle", { cx: x(p.k_labels), cy: y(p.roc_auc), r: 12, fill: "transparent" }),
        sv("circle", { cx: x(p.k_labels), cy: y(p.roc_auc), r: 4.5, fill: s.color, stroke: cssVar("--paper"), "stroke-width": 2 }));
      bindTip(g, `${siteLabel(site)}, ${p.k_labels} local labels`, [{ label: s.label, value: `${p.roc_auc.toFixed(3)} ± ${(p.roc_auc_sd || 0).toFixed(3)}`, color: s.color }]);
      svg.append(g);
    }
  }
  return el("figure", {}, el("figcaption", {}, siteLabel(site), el("small", { text: "ROC-AUC by local labels used" })), svg);
}

function coverageSection(run, task) {
  const sites = [...new Set(run.summary.map((r) => r.held_out_site))];
  const kMax = Math.max(0, ...run.summary.filter((r) => r.strategy === "site_recalibration").map((r) => r.k_labels));
  const rows = [];
  for (const s of sites) {
    const b = run.summary.find((r) => r.held_out_site === s && r.strategy === "source_calibration");
    const a = run.summary.find((r) => r.held_out_site === s && r.strategy === "site_recalibration" && r.k_labels === kMax);
    for (const [cls, key] of [[TASKS[task].positive, "coverage_pos"], [TASKS[task].negative, "coverage_neg"]]) {
      rows.push({ site: s, cls, before: b?.[key], after: a?.[key] });
    }
  }
  const W = 640, rowH = 26, L = 230, R = 16, T = 22, H = T + rows.length * rowH + 24;
  const x = (v) => L + (W - L - R) * v;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Conformal coverage per class before and after recalibration" });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(sv("line", { x1: x(v), x2: x(v), y1: T - 6, y2: H - 22, stroke: cssVar("--rule"), "stroke-width": 1 }),
      sv("text", { x: x(v), y: H - 6, "text-anchor": "middle", text: `${v * 100}%` }));
  }
  svg.append(sv("line", { x1: x(0.9), x2: x(0.9), y1: T - 10, y2: H - 22, stroke: cssVar("--ink-2"), "stroke-width": 1.5, "stroke-dasharray": "0" }),
    sv("text", { x: x(0.9), y: T - 12, "text-anchor": "middle", text: "90% target" }));
  rows.forEach((r, i) => {
    const cy = T + i * rowH + rowH / 2;
    svg.append(sv("text", { x: L - 10, y: cy + 4, "text-anchor": "end", text: `${siteLabel(r.site).split(",")[0]}, ${r.cls.toLowerCase()}` }));
    if (r.before !== undefined && r.after !== undefined) {
      svg.append(sv("line", { x1: x(r.before), x2: x(r.after), y1: cy, y2: cy, stroke: cssVar("--rule-strong"), "stroke-width": 2 }));
    }
    const g = sv("g", {});
    if (r.before !== undefined) g.append(sv("circle", { cx: x(r.before), cy, r: 5, fill: cssVar("--paper"), stroke: cssVar("--ink-3"), "stroke-width": 2 }));
    if (r.after !== undefined) g.append(sv("circle", { cx: x(r.after), cy, r: 5, fill: cssVar("--series-1"), stroke: cssVar("--paper"), "stroke-width": 2 }));
    g.append(sv("rect", { x: L, y: cy - rowH / 2, width: W - L - R, height: rowH, fill: "transparent" }));
    bindTip(g, `${siteLabel(r.site)}, ${r.cls}`, [
      { label: "Training settings", value: pct(r.before), color: cssVar("--ink-3") },
      { label: `Recalibrated, ${kMax} labels`, value: pct(r.after), color: cssVar("--series-1") },
    ]);
    svg.append(g);
  });
  return el("section", { class: "v-section" },
    el("h2", { text: "Does the 90% confidence guarantee hold?" }),
    el("p", { text: `For each hospital and each class, the share of held-out images whose true answer was inside the aid's confidence set. Hollow dots use the training hospitals' settings unchanged; filled dots are after recalibrating with ${kMax} local labels. Below the line, the guarantee failed. This is reported, not hidden.` }),
    legend([{ label: "Training hospitals' settings", color: cssVar("--ink-3") }, { label: `Recalibrated with ${kMax} local labels`, color: cssVar("--series-1") }]),
    el("figure", { class: "chart", style: "margin-top:10px" }, svg),
    tableView(["Hospital", "Class", "Before", "After"], rows.map((r) => [siteLabel(r.site), r.cls, pct(r.before), pct(r.after)])),
  );
}

function detailTable(run, task) {
  const order = ["source_calibration", "site_recalibration", "site_adapt", "site_only_head"];
  const rows = [...run.summary].sort((a, b) => a.held_out_site.localeCompare(b.held_out_site)
    || order.indexOf(a.strategy) - order.indexOf(b.strategy) || a.k_labels - b.k_labels);
  const cov = (v) => {
    const td = el("span", { text: pct(v) });
    if (v !== undefined && v !== null && v < 0.85) td.style.color = "var(--refer)";
    return td;
  };
  const sd = (r, k) => (r[k] === undefined || r[k] === null ? "–" : `${num(r[k])} ± ${num(r[k + "_sd"] || 0)}`);
  return el("section", { class: "v-section" },
    el("h2", { text: "Every number" }),
    el("p", { text: `Sensitivity and specificity at each strategy's referral line, with ROC-AUC, PR-AUC, calibration error and per-class coverage (target ${pct(1 - (run.alpha || 0.1))}). Mean ± SD over ${run.repeats} repeats. Coverage below 85% is shown in red.` }),
    el("div", { class: "register-wrap" }, el("table", { class: "data-table" },
      el("thead", {}, el("tr", {}, ["Held-out hospital", "Strategy", "Local labels", "Sensitivity", "Specificity", "ROC-AUC", "PR-AUC", "ECE", "Coverage, positives", "Coverage, negatives"].map((h) => el("th", { text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {},
        el("td", { text: siteLabel(r.held_out_site) }), el("td", { text: STRATEGY[r.strategy] || r.strategy }), el("td", { text: r.k_labels }),
        el("td", { text: sd(r, "sensitivity") }), el("td", { text: sd(r, "specificity") }), el("td", { text: sd(r, "roc_auc") }),
        el("td", { text: sd(r, "pr_auc") }), el("td", { text: r.ece === undefined ? "–" : num(r.ece) }),
        el("td", {}, cov(r.coverage_pos)), el("td", {}, cov(r.coverage_neg))))))),
  );
}

function backboneSection(ev, task) {
  const runs = ev.loso?.[task] || [];
  if (runs.length < 2) return null;
  const bench = Object.fromEntries((ev.backbones || []).map((b) => [b.backbone, b]));
  const deployed = state.info?.tasks?.[task]?.backbone;
  const rows = runs.map((r) => {
    const src = r.summary.filter((x) => x.strategy === "source_calibration");
    const adapt = r.summary.filter((x) => x.strategy === "site_adapt");
    const mean = (a) => (a.length ? a.reduce((s, x) => s + x.roc_auc, 0) / a.length : null);
    return [r.backbone + (r.backbone === deployed ? " (in use)" : ""), bench[r.backbone] ? `${Math.round(bench[r.backbone].ms_per_image_median)} ms` : "–",
      num(mean(src), 3), adapt.length ? num(mean(adapt.filter((x) => x.k_labels === Math.max(...adapt.map((y) => y.k_labels)))), 3) : "–"];
  });
  return el("section", { class: "v-section" },
    el("h2", { text: "Choosing a model that runs on a clinic laptop" }),
    el("p", { text: "Each candidate image model was put through the same leave-one-hospital-out test and timed on a laptop CPU (Intel Core Ultra 5, no GPU). Mean ROC-AUC across held-out hospitals." }),
    el("table", { class: "data-table" },
      el("thead", {}, el("tr", {}, ["Image model", "Time per image", "ROC-AUC, unseen hospital", "With local adaptation"].map((h) => el("th", { text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((v) => el("td", { text: v })))))));
}

function shortcutSection(ev, task) {
  const rows = ev.shortcut?.[task];
  if (!rows || !rows.length) return null;
  const conds = [["original", "Whole image", "--series-1"], ["border_only", "Border only, centre blanked", "--series-2"], ["lungs_removed", "Lungs blanked out", "--series-3"]];
  const sites = [...new Set(rows.map((r) => r.site))];
  const W = 640, rowH = 30, L = 210, R = 16, T = 24, H = T + sites.length * rowH + 24;
  const x = (v) => L + (W - L - R) * ((v - 0.3) / 0.7);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Shortcut test ROC-AUC by image condition" });
  for (const v of [0.3, 0.5, 0.7, 0.9, 1.0]) {
    svg.append(sv("line", { x1: x(v), x2: x(v), y1: T - 6, y2: H - 22, stroke: cssVar("--rule"), "stroke-width": 1 }),
      sv("text", { x: x(v), y: H - 6, "text-anchor": "middle", text: v.toFixed(1) }));
  }
  svg.append(sv("line", { x1: x(0.5), x2: x(0.5), y1: T - 10, y2: H - 22, stroke: cssVar("--ink-2"), "stroke-width": 1.5 }),
    sv("text", { x: x(0.5), y: T - 12, "text-anchor": "middle", text: "chance" }));
  sites.forEach((s, i) => {
    const cy = T + i * rowH + rowH / 2;
    svg.append(sv("text", { x: L - 10, y: cy + 4, "text-anchor": "end", text: siteLabel(s).split(",")[0] }));
    for (const [c, label, color] of conds) {
      const r = rows.find((q) => q.site === s && q.condition === c);
      if (!r) continue;
      const g = sv("g", {});
      g.append(sv("circle", { cx: x(Math.max(0.3, r.roc_auc)), cy, r: 12, fill: "transparent" }),
        sv("circle", { cx: x(Math.max(0.3, r.roc_auc)), cy, r: 5, fill: cssVar(color), stroke: cssVar("--paper"), "stroke-width": 2 }));
      bindTip(g, siteLabel(s), [{ label, value: r.roc_auc.toFixed(3), color: cssVar(color) }]);
      svg.append(g);
    }
  });
  const present = conds.filter(([c]) => rows.some((r) => r.condition === c));
  return el("section", { class: "v-section" },
    el("h2", { text: "Is it reading the lungs, or the hospital's markings?" }),
    el("p", { text: "Models can learn shortcuts such as hospital text, markers or collimation. Here the trained model scores images with the centre blanked out (only the border left) and, for Montgomery, with both lung fields blanked using the manual lung masks. A score near chance means the shortcut is not available to it." }),
    legend(present.map(([, label, color]) => ({ label, color: cssVar(color) }))),
    el("figure", { class: "chart", style: "margin-top:10px" }, svg),
    tableView(["Hospital", "Condition", "ROC-AUC", "Images"], rows.map((r) => [siteLabel(r.site), r.condition.replace(/_/g, " "), num(r.roc_auc, 3), r.n])),
  );
}

function subgroupSection(ev, task) {
  const rows = ev.subgroups?.[task];
  if (!rows || !rows.length) return null;
  const varLabel = { sex: "Sex", age: "Age" };
  const grp = (v, g) => (v === "sex" ? { M: "Men", F: "Women" }[g] || g : g);
  return el("section", { class: "v-section" },
    el("h2", { text: "Does it work equally for women, men and older patients?" }),
    el("p", { text: "Coverage is guaranteed per class, not per subgroup. After recalibration, the held-out hospitals with recorded sex and age were broken down by group. Small groups give noisy numbers; a gap here is a reason to calibrate per group once enough local labels exist." }),
    el("div", { class: "register-wrap" }, el("table", { class: "data-table" },
      el("thead", {}, el("tr", {}, ["Hospital", "Group", "Images per test", "Sensitivity", "Specificity", "Coverage, positives", "Coverage, negatives"].map((h) => el("th", { text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {}, [siteLabel(r.held_out_site), `${varLabel[r.variable] || r.variable}: ${grp(r.variable, r.group)}`, r.n_per_repeat,
        pct(r.sensitivity), pct(r.specificity), pct(r.coverage_pos), pct(r.coverage_neg)].map((v) => el("td", { text: v }))))))));
}

/* ================================================================ about */
function renderAbout(info) {
  const tasks = Object.entries(info.tasks).map(([k, t]) => {
    const m = t.internal_metrics || {};
    const g = t.gradability?.test;
    return el("li", {},
      el("strong", { text: `${TASKS[k]?.short || k}: ` }),
      `${t.finding}, read by ${t.backbone}. Trained on ${(t.training?.sites || []).map(siteLabel).join("; ")}. On held-out patients from those hospitals: sensitivity ${pct(m.sensitivity)}, specificity ${pct(m.specificity)}, ROC-AUC ${num(m.roc_auc)}.`,
      g ? ` A learned quality model, trained on expert gradability labels, catches ${pct(g.sensitivity)} of ungradable photos while rejecting ${pct(1 - g.specificity)} of good ones.` : "");
  });
  $("#about").replaceChildren(
    el("h1", { text: "About ClinicTune" }),
    el("p", { class: "lede", text: info.disclaimer }),
    el("h2", { text: "What happens to each image" }),
    el("ol", {}, [
      "Quality check. Rules catch blur, exposure, wrong modality and missing lung fields; for fundus photos a learned quality model also judges gradability. A failed check means retake, not a guess.",
      "A frozen foundation model turns the image into a description of every small patch of it. It never changes; only small, auditable layers on top of it are fitted.",
      "Familiarity check. Images unlike anything the clinic was calibrated on are escalated.",
      "A calibrated score and a 90% confidence check per class, fitted for this clinic during onboarding.",
      "The decision. Low risk is only given when the score and the confidence check agree, because missing disease is the costlier error.",
      "An exact heatmap of what moved the score, the most similar confirmed reference cases, and a plain-language note in English, Hindi or Kannada.",
    ].map((t) => el("li", { text: t }))),
    el("h2", { text: "Models in this bundle" }),
    el("ul", {}, tasks),
    el("h2", { text: "Limits you should know" }),
    el("ul", {}, [
      "It has not been prospectively or clinically validated, and it is not a medical device.",
      "Performance at a new clinic is unknown until that clinic is onboarded. The report card shows what was measured there, with confidence intervals.",
      "With 50 to 100 local labels, intervals are wide. Read the interval, not the point estimate.",
      "The Hindi and Kannada notes are drafts that a native-speaking clinician has not reviewed.",
      "The confidence guarantee is per class, not per subgroup. The Validation page shows how it held by sex and age.",
    ].map((t) => el("li", { text: t }))),
    el("p", { class: "muted small", style: "margin-top:24px", text: `ClinicTune ${info.version}, model bundle ${info.bundle_id}. Data sources and licences: ${(info.datasets || []).join(", ")}.` }),
  );
}

/* ================================================================ boot */
async function init() {
  try {
    state.info = await api("/v1/info");
    $("#demo-banner").hidden = !state.info.demo;
    renderAbout(state.info);
    await loadSites();
    const sp = pref("site");
    if (sp && state.sites.some((s) => s.site_id === sp)) $("#site-select").value = sp;
  } catch (err) {
    const msg = err.status === 401 ? "This server needs an API key. Use the key button at the top right." : `The screening service is not ready: ${err.message}`;
    $("#slip-empty").replaceChildren(el("p", { class: "slip-empty-title", text: "Not connected" }), el("p", { text: msg }));
  }
}
showTab(location.hash.slice(1) || "screen", { push: false });
window.addEventListener("hashchange", () => showTab(location.hash.slice(1), { push: false }));
init();

if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("sw.js").catch(() => { /* offline support is best-effort */ });
}
