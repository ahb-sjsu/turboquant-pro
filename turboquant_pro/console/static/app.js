"use strict";
// TurboQuant Console client. Reads only the telemetry contract (docs/DESIGN_console.md).
// Every dynamic value is written with textContent: nothing from the server becomes markup.

const TOKEN = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const H = { "X-TQP-Token": TOKEN };
const S = {
  snap: null, traces: [], sel: 0, filter: "", paused: false, lastMsg: 0,
  qps: [], p95: [], inspected: null, replay: null, readscope: null, maxed: null,
};
const $ = (id) => document.getElementById(id);

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v;
    else if (k === "title") e.title = v;
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) {
    if (c == null) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}
const fmt = (v, d = 2) => (v == null ? "—" : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(d));
const short = (s, n = 12) => (s ? String(s).slice(0, n) : "—");

async function api(path) {
  const r = await fetch(path, { headers: H, cache: "no-store" });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

// ------------------------------------------------------------------ stream
async function stream() {
  for (;;) {
    try {
      const r = await fetch("/api/stream", { headers: H, cache: "no-store" });
      if (r.status === 401) { setConn("red", "bad token: open the URL tqp console printed"); return; }
      const rd = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, i); buf = buf.slice(i + 2);
          let ev = "message", data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) ev = line.slice(7);
            else if (line.startsWith("data: ")) data += line.slice(6);
          }
          if (data) onEvent(ev, JSON.parse(data));
        }
      }
    } catch (e) { /* fall through to reconnect */ }
    setConn("red", "disconnected — retrying");
    await new Promise((res) => setTimeout(res, 2000));
  }
}

function onEvent(ev, d) {
  S.lastMsg = Date.now();
  if (ev === "trace") {
    if (S.traces.some((x) => x.id === d.id)) return;
    S.traces.push(d);
    if (S.traces.length > 500) S.traces.splice(0, S.traces.length - 500);
  } else if (ev === "snapshot") {
    S.snap = d;
    const r = byName(d.readings);
    S.qps.push(r["search.qps"]?.value ?? null);
    S.p95.push(r["search.latency_ms.p95"]?.value ?? null);
    for (const a of [S.qps, S.p95]) if (a.length > 120) a.shift();
  }
  if (!S.paused) render();
}

function setConn(cls, text) { const c = $("conn"); c.className = "badge " + cls; c.textContent = text; }
setInterval(() => {  // freshness (UX-005, NFR-004): age of the last message, with text
  if (!S.lastMsg) return;
  const age = (Date.now() - S.lastMsg) / 1000;
  if (age <= 2.5) setConn("cyan", "live");
  else if (age <= 10) setConn("amber", `stale ${age.toFixed(0)} s`);
  else setConn("red", `disconnected ${age.toFixed(0)} s`);
}, 500);

const byName = (rs) => Object.fromEntries((rs || []).map((r) => [r.name, r]));

// ------------------------------------------------------------------ render
function render() {
  const d = S.snap;
  if (!d) return;
  const r = byName(d.readings);
  const w = d.workload;
  $("mode").textContent = "mode: " + w.mode;
  $("mode").className = "badge " + (w.mode === "approximate" ? "amber" : "green");
  const last = S.traces[S.traces.length - 1];
  $("path").textContent = "scan: " + (last?.scan_path || "—") + (d.index.kernel ? " · AVX2" : "");
  $("path").className = "badge cyan";

  const kp = $("kpis"); kp.replaceChildren();
  const show = [
    ["QPS", "search.qps", 1], ["p50", "search.latency_ms.p50"], ["p95", "search.latency_ms.p95"],
    ["p99", "search.latency_ms.p99"], ["rerank agree", "search.rerank_agreement"],
    ["compression", "index.compression_ratio", 1], ["rows", "index.rows", 0],
    ["CPU", "process.cpu_percent", 0], ["RSS", "process.rss_mb", 0],
  ];
  for (const [label, name, dg] of show) {
    const x = r[name];
    if (!x) continue;
    const un = x.value == null;
    kp.append(el("div", { class: "kpi" + (un ? " unavail" : ""), title: un ? x.unavailable_reason : `${x.description} · ${x.source}` + (x.reference ? ` · judged against: ${x.reference}` : "") },
      el("span", { class: "k" }, label),
      el("span", {}, el("span", { class: "v" }, fmt(x.value, dg ?? 2)), el("span", { class: "u" }, x.unit === "fraction" ? "" : x.unit), el("span", { class: "kind" }, x.kind))));
  }
  if (w.errors) kp.append(el("div", { class: "kpi" }, el("span", { class: "k status-bad" }, "workload errors"), el("span", { class: "v status-bad", title: w.last_error || "" }, String(w.errors))));

  $("qpsNow").textContent = fmt(r["search.qps"]?.value, 1);
  $("p95Now").textContent = fmt(r["search.latency_ms.p95"]?.value, 3);
  spark($("sparkQps"), S.qps, "#2fd3e8");
  spark($("sparkLat"), S.p95, "#b58cff");

  const pl = $("pipeline"); pl.replaceChildren();
  const stages = [["encode", "search.stage_ms.encode"], ["scan", "search.stage_ms.scan"], ["rerank", "search.stage_ms.rerank"]];
  const vals = stages.map(([, n]) => r[n]?.value);
  const mx = Math.max(...vals.filter((v) => v != null), 1e-9);
  stages.forEach(([s, n], i) => {
    const v = vals[i];
    const bar = el("div", { class: "bar", title: r[n]?.unavailable_reason || "" });
    bar.style.width = v == null ? "0" : `${Math.max(2, (100 * v) / mx)}%`;
    pl.append(el("div", { class: "stage" }, el("span", {}, s), el("div", {}, bar), el("span", { class: "r" }, v == null ? "—" : fmt(v, 3))));
    if (i < stages.length - 1) pl.append(el("div", { class: "arrow" }, "↓"));
  });
  pl.append(el("div", { class: "note" }, `results → top-${w.k}${w.rerank ? `, from ${w.k * w.rerank} candidates reranked exactly` : ", approximate scores (no rerank)"}`));

  const ix = $("index"); ix.replaceChildren();
  const ie = d.index;
  ix.append(kv([["kind", ie.kind], ["rows", ie.rows], ["dim (coded)", ie.dim], ["metric", ie.metric],
    ["stored bytes/row", ie.stored_bytes_per_row], ["kernel", ie.kernel ? "AVX2 compiled" : "numpy"],
    ["workload", `${w.rows} queries @ ${w.target_qps} qps target, k=${w.k}`],
    ["tracer", `rate ${d.tracer.rate}, ${d.tracer.captured}/${d.tracer.capacity} kept`]]));

  renderStream();
}

function kv(rows) {
  return el("dl", { class: "kv" }, rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v == null ? "—" : String(v))]));
}

function spark(c, data, color) {
  const g = c.getContext("2d");
  const W = c.width, Hh = c.height;
  g.clearRect(0, 0, W, Hh);
  const v = data.filter((x) => x != null);
  if (v.length < 2) {
    g.fillStyle = "#5b6875"; g.font = "12px sans-serif";
    g.fillText("collecting…", 8, Hh / 2 + 4);
    return;
  }
  const mx = Math.max(...v) * 1.1 || 1;
  g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
  data.forEach((x, i) => {
    if (x == null) return;
    const px = (i / 119) * W, py = Hh - (x / mx) * (Hh - 4) - 2;
    i ? g.lineTo(px, py) : g.moveTo(px, py);
  });
  g.stroke();
}

function visibleTraces() {
  const f = S.filter.trim();
  const t = S.traces.slice().reverse();
  return f ? t.filter((x) => x.id.includes(f) || String(x.params.workload_row) === f) : t;
}

function stageMs(t, name) { const s = t.stages.find((x) => x.name === name); return s ? s.ms : null; }

function renderStream() {
  const tb = $("stream").tBodies[0]; tb.replaceChildren();
  const rows = visibleTraces().slice(0, 80);
  S.sel = Math.min(S.sel, Math.max(rows.length - 1, 0));
  rows.forEach((t, i) => {
    const tr = el("tr", { class: i === S.sel ? "sel" : "" },
      el("td", {}, t.started_utc.slice(11, 23)), el("td", {}, t.id),
      el("td", {}, t.params.workload_row ?? "—"), el("td", {}, t.scan_path || "—"),
      el("td", { class: "r" }, fmt(t.total_ms, 3)), el("td", { class: "r" }, fmt(stageMs(t, "encode"), 3)),
      el("td", { class: "r" }, fmt(stageMs(t, "scan"), 3)), el("td", { class: "r" }, fmt(stageMs(t, "rerank"), 3)),
      el("td", { class: "r" }, fmt(t.results?.rerank_agreement, 2)));
    tr.addEventListener("click", () => { S.sel = i; inspect(t); });
    tb.append(tr);
  });
}

// --------------------------------------------------------------- readscope
async function loadReadscope() {
  try { S.readscope = await api("/api/readscope"); } catch (e) { S.readscope = { error: e.message }; }
  const box = $("readscope"); box.replaceChildren();
  const rs = S.readscope;
  if (rs.error) { box.append(el("p", { class: "status-bad" }, rs.error)); return; }
  if (rs.observer) {
    const o = rs.observer.reference;
    $("obs").hidden = false;
    $("obs").textContent = "observer: " + (o.observer || "—") + " · " + short(o.sha256, 8);
    box.append(el("h3", {}, "Observer"), kv([["name", o.observer], ["target", o.target], ["sha256", o.sha256],
      ["consumers", (o.consumers || []).map((c) => c.metric || c.name || JSON.stringify(c)).join(", ")],
      ["primary", JSON.stringify(o.primary_consumer)]]));
  } else {
    box.append(el("p", { class: "muted" }, "No observer contract loaded (start with --observer X.tqo). Results are not tied to a declared reader."));
  }
  if (rs.certificate) {
    const c = rs.certificate, cc = c.certificate || {};
    const ok = c.passed, vac = cc.vacuous;
    box.append(el("h3", {}, "Certificate"), kv([
      ["status", ok ? "PASSED" : "NOT PASSED"], ["tau floor", cc.tau_floor], ["spearman floor", cc.spearman_floor],
      ["vacuous", vac ? "yes (rerank required)" : "no"], ["validity", rs.validity?.status || "UNCHECKED"],
      ["issued", c.created_utc]]));
    box.lastChild.querySelector("dd").className = ok ? "status-ok" : "status-bad";
  }
  if (rs.provenance?.length) {
    box.append(el("h3", {}, "Provenance"), el("ol", { class: "chain" },
      rs.provenance.map((p) => el("li", {}, `${p.step}: ${p.sha256 ? short(p.sha256, 16) : JSON.stringify(Object.fromEntries(Object.entries(p).filter(([k]) => k !== "step")))}`))));
  }
}

// --------------------------------------------------------------- inspector
function inspect(t) {
  if (!t) return;
  S.inspected = t; S.replay = null;
  drawInspector();
  $("overlay").hidden = false;
}

function drawInspector() {
  const t = S.inspected;
  const b = $("ovBody"); b.replaceChildren();
  b.append(kv([["trace", t.id], ["component", t.component], ["started", t.started_utc],
    ["query", `row ${t.params.workload_row ?? "—"} · sha256 ${short(t.input.sha256, 16)} · dim ${t.input.dim}`],
    ["params", JSON.stringify(t.params)], ["scan path", t.scan_path], ["total", fmt(t.total_ms, 3) + " ms"],
    ["observer", t.observer ? `${t.observer.observer} (${short(t.observer.sha256, 8)})` : "none declared"],
    ["sampled", t.sampled ? "yes (one of a sample)" : "no (every call traced)"]]));
  b.append(el("h3", {}, "Stages"), el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "stage"), el("th", { class: "r" }, "ms"), el("th", { class: "r" }, "candidates"), el("th", { class: "r" }, "rows"))),
    el("tbody", {}, t.stages.map((s) => el("tr", {}, el("td", {}, s.name), el("td", { class: "r" }, fmt(s.ms, 4)), el("td", { class: "r" }, s.candidates ?? "—"), el("td", { class: "r" }, s.rows ?? "—"))))));
  const res = t.results;
  if (res) {
    const fin = res.final;
    const head = fin ? ["final rank", "id", "exact score", "approx rank", "movement"] : ["rank", "id", "approx score"];
    const rows = fin ? fin.map((x, i) => el("tr", {}, el("td", {}, i), el("td", {}, x.id), el("td", { class: "r" }, fmt(x.exact_score, 4)),
      el("td", { class: "r" }, x.approx_rank ?? "not in approx top"), el("td", { class: "r " + (x.rank_movement > 0 ? "up" : x.rank_movement < 0 ? "down" : "") },
        x.rank_movement == null ? "new" : x.rank_movement > 0 ? `▲${x.rank_movement}` : x.rank_movement < 0 ? `▼${-x.rank_movement}` : "=")))
      : res.approximate.map((x, i) => el("tr", {}, el("td", {}, i), el("td", {}, x.id), el("td", { class: "r" }, fmt(x.score, 4))));
    b.append(el("h3", {}, fin ? `Results: approximate → exact rerank (agreement ${fmt(res.rerank_agreement, 2)})` : "Results: approximate scores (no rerank)"),
      el("table", {}, el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))), el("tbody", {}, rows)));
  }
  if (S.replay) {
    const r = S.replay;
    b.append(el("h3", {}, "Replay"));
    if (r.error) b.append(el("p", { class: "status-bad" }, r.error));
    else b.append(kv([["replayed as", r.after], ["same query (sha256)", r.same_input_sha256 ? "yes" : "NO"],
      ["pinned", `query ${r.pinned.query ? "yes" : "no"} · index ${r.pinned.index ? "yes" : "no"}`],
      ["result", r.diff.same_ids_in_order ? "identical ids in identical order" : `overlap ${r.diff.overlap}/${r.diff.k}, moved ${Object.keys(r.diff.moved).length}`],
      ["latency ms", `${fmt(r.diff.latency_ms.before, 3)} → ${fmt(r.diff.latency_ms.after, 3)}`],
      ["nondeterminism", r.nondeterminism.length ? r.nondeterminism.join("; ") : "none detected"]]));
  }
}

async function doReplay() {
  const t = S.inspected || visibleTraces()[S.sel];
  if (!t) return;
  S.inspected = t;
  try { S.replay = await api("/api/replay/" + t.id); } catch (e) { S.replay = { error: e.message }; }
  drawInspector(); $("overlay").hidden = false;
}

async function doExport() {
  let doc;
  try { doc = await api("/api/export"); } catch (e) { alert("export failed: " + e.message); return; }
  doc.view = { inspected: S.inspected?.id || null, filter: S.filter, paused: S.paused };
  const blob = new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `tqp-console-${new Date().toISOString().replace(/[:.]/g, "-")}.json` });
  document.body.append(a); a.click(); a.remove();
}

// ---------------------------------------------------------------- keyboard
const panels = () => [...document.querySelectorAll(".panel")];
document.addEventListener("keydown", (e) => {
  if (e.target === $("filter")) {
    if (e.key === "Escape" || e.key === "Enter") { $("filter").blur(); if (e.key === "Escape") { S.filter = ""; $("filter").value = ""; $("filter").hidden = true; } renderStream(); }
    return;
  }
  const ov = !$("overlay").hidden, hp = !$("help").hidden;
  if (e.key === "Escape") { $("overlay").hidden = true; $("help").hidden = true; if (S.maxed) { S.maxed.classList.remove("max"); S.maxed = null; } return; }
  if (e.key === "?") { $("help").hidden = !$("help").hidden; return; }
  if (hp) return;
  if (e.key === "p") { S.paused = !S.paused; $("pause").hidden = !S.paused; if (!S.paused) render(); return; }
  if (e.key === "r") { doReplay(); return; }
  if (e.key === "e") { doExport(); return; }
  if (ov) return;
  if (e.key === "/") { e.preventDefault(); $("filter").hidden = false; $("filter").focus(); return; }
  if (/^[1-6]$/.test(e.key)) { $("p" + e.key).focus(); return; }
  if (e.key === "m") {
    const p = document.activeElement.closest?.(".panel");
    if (S.maxed) { S.maxed.classList.remove("max"); S.maxed = null; } else if (p) { p.classList.add("max"); S.maxed = p; }
    return;
  }
  if (e.key === "ArrowDown") { S.sel++; renderStream(); e.preventDefault(); return; }
  if (e.key === "ArrowUp") { S.sel = Math.max(0, S.sel - 1); renderStream(); e.preventDefault(); return; }
  if (e.key === "Enter") { inspect(visibleTraces()[S.sel]); return; }
});
$("filter").addEventListener("input", (e) => { S.filter = e.target.value; S.sel = 0; renderStream(); });

// -------------------------------------------------------------------- boot
(async () => {
  if (!TOKEN) { setConn("red", "no token: open the full URL tqp console printed"); return; }
  try {
    const v = await api("/api/version");
    document.title = `TurboQuant Console · ${v.tool_version}`;
    S.traces = await api("/api/traces?n=200");
  } catch (e) { setConn("red", "cannot reach the console API: " + e.message); return; }
  loadReadscope();
  stream();
})();
