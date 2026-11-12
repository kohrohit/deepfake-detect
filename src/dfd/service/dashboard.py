"""The operator page. One file, no CDN, no build step.

The banner at the top is not decoration. This service will happily print a
verdict for every file it is given, and the measured truth today is that its
one detector with weights scores 0.289 AUC on an unseen corpus. A dashboard
that shows the verdict and hides that is the exact artefact this project has
spent its whole history refusing to produce, so the report card is rendered
above the results, from /api/evidence, and says so when it is missing.
"""
from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dfd — deepfake detection service</title>
<style>
  :root {
    --bg: #f7f7f5; --fg: #1a1a18; --muted: #6b6b66; --line: #dedede;
    --card: #ffffff; --warn-bg: #fff4e5; --warn-line: #e0a458;
    --bad-bg: #fdecea; --bad-line: #d9534f; --ok: #2e7d5b;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #16161a; --fg: #ececea; --muted: #9a9a94; --line: #2e2e34;
      --card: #1e1e24; --warn-bg: #3a2c15; --warn-line: #b8862f;
      --bad-bg: #3a1d1b; --bad-line: #c25650; --ok: #55b98b;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 24px 20px 64px; }
  h1 { font-size: 20px; margin: 0 0 2px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); margin: 0 0 20px; }
  .card { background: var(--card); border: 1px solid var(--line);
          border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
  .banner { background: var(--warn-bg); border-color: var(--warn-line); }
  .banner.bad { background: var(--bad-bg); border-color: var(--bad-line); }
  .banner h2 { font-size: 14px; margin: 0 0 6px; text-transform: uppercase;
               letter-spacing: 0.06em; }
  .row { display: flex; gap: 24px; flex-wrap: wrap; }
  .stat { min-width: 108px; }
  .stat .n { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat .k { color: var(--muted); font-size: 12px; text-transform: uppercase;
             letter-spacing: 0.05em; }
  table { width: 100%; border-collapse: collapse; }
  .scroll { overflow-x: auto; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
           white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; font-size: 12px;
       text-transform: uppercase; letter-spacing: 0.05em; }
  td.file { white-space: normal; word-break: break-all; }
  /* The evidence notes are the most important prose on the page and the
     longest cells in it. Left to the nowrap rule above they push the card
     into a horizontal scroller and the sentence that matters is the half
     that is off-screen. */
  td.note { white-space: normal; min-width: 22em; }
  #evidence table { table-layout: auto; }
  tr.click { cursor: pointer; }
  tr.click:hover td { background: var(--bg); }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
          border: 1px solid var(--line); font-size: 12px; }
  .pill.fake { border-color: var(--bad-line); color: var(--bad-line); }
  .pill.real { border-color: var(--ok); color: var(--ok); }
  .pill.failed { border-color: var(--bad-line); color: var(--bad-line); }
  .pill.running, .pill.queued { border-color: var(--warn-line); color: var(--warn-line); }
  .muted { color: var(--muted); }
  pre { background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
        padding: 12px; overflow-x: auto; font-size: 12px; max-height: 420px; }
  button, .drop { font: inherit; }
  .drop { border: 1.5px dashed var(--line); border-radius: 10px; padding: 22px;
          text-align: center; color: var(--muted); cursor: pointer; }
  .drop.over { border-color: var(--ok); color: var(--fg); }
  button { background: var(--fg); color: var(--bg); border: 0; border-radius: 7px;
           padding: 7px 14px; cursor: pointer; }
  code { font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>dfd — deepfake detection service</h1>
  <p class="sub" id="sub">connecting…</p>

  <div class="card banner" id="evidence"><h2>Detector evidence</h2>
    <div id="evidence-body">loading…</div></div>

  <div class="card"><div class="row" id="stats"></div></div>

  <div class="card">
    <div class="drop" id="drop">Drop a file here, or click to choose one.
      The watched inbox works too — anything copied into it is picked up.</div>
    <input type="file" id="file" hidden multiple>
    <p class="muted" id="upload-note"></p>
  </div>

  <div class="card">
    <div class="row" style="justify-content: space-between; align-items: baseline">
      <h2 style="font-size:14px;margin:0 0 10px">Recent submissions</h2>
      <span class="muted" id="refreshed"></span>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>received</th><th>file</th><th>source</th><th>status</th>
        <th>verdict</th><th>llr</th></tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
  </div>

  <div class="card" id="detail" hidden>
    <div class="row" style="justify-content: space-between; align-items: baseline">
      <h2 style="font-size:14px;margin:0 0 10px" id="detail-title">Audit record</h2>
      <button id="close">close</button>
    </div>
    <pre id="detail-body"></pre>
  </div>
</div>
<script>
const $ = (id) => document.getElementById(id);
const fmt = (t) => t ? new Date(t).toLocaleString() : "—";

async function j(url, opts) {
  const r = await fetch(url, opts);
  return { ok: r.ok, status: r.status, body: await r.json().catch(() => ({})) };
}

async function loadEvidence() {
  const { body } = await j("/api/evidence");
  const el = $("evidence"), out = $("evidence-body");
  if (!body.available) {
    el.classList.add("bad");
    out.innerHTML = "<b>No measured detector performance on this deployment.</b> "
      + (body.note || "") + " Treat every verdict below as unvalidated.";
    return;
  }
  const ds = body.detectors || {};
  const names = Object.keys(ds);
  let worst = 1;
  const cells = names.map((n) => {
    const d = ds[n];
    if (typeof d.auc === "number") worst = Math.min(worst, d.auc);
    const auc = (typeof d.auc === "number") ? d.auc.toFixed(3) : "not measured";
    return `<tr><td><code>${n}</code></td><td>${auc}</td>`
      + `<td>${d.corpus || "—"}</td>`
      + `<td class="muted note">${d.note || ""}</td></tr>`;
  }).join("");
  if (worst < 0.55) el.classList.add("bad");
  out.innerHTML =
    (worst < 0.55
      ? "<b>The detectors on this deployment do not work.</b> The best measured "
        + "AUC below is at or under chance on an unseen corpus, so a verdict "
        + "here is not evidence of anything. The service is running so the "
        + "pipeline can be exercised, not so its answers can be believed.<br><br>"
      : "")
    + `<div class="scroll"><table><thead><tr><th>detector</th><th>AUC</th>`
    + `<th>measured on</th><th>note</th></tr></thead><tbody>${cells}</tbody></table></div>`
    + (body.generated_at ? `<p class="muted">measured ${fmt(body.generated_at)}</p>` : "");
}

async function loadStats() {
  const [h, s] = await Promise.all([j("/health"), j("/api/stats")]);
  $("sub").textContent = h.ok
    ? `up ${Math.round(h.body.uptime_s)}s · ${h.body.submissions} submissions seen`
    : "service unreachable";
  const st = s.body.status || {}, vd = s.body.verdict || {};
  const box = (k, n) => `<div class="stat"><div class="n">${n || 0}</div>`
    + `<div class="k">${k}</div></div>`;
  $("stats").innerHTML = box("queued", st.queued) + box("running", st.running)
    + box("done", st.done) + box("failed", st.failed)
    + box("fake", vd.fake) + box("real", vd.real)
    + box("insufficient", vd.insufficient_evidence);
}

async function loadRows() {
  const { body } = await j("/api/submissions?limit=50");
  const rows = (body.submissions || []).map((r) => {
    const v = r.verdict || "—";
    const vc = v.includes("fake") ? "fake" : v.includes("real") ? "real" : "";
    return `<tr class="click" data-id="${r.id}">`
      + `<td class="muted">${fmt(r.received_at)}</td>`
      + `<td class="file">${r.filename}</td>`
      + `<td class="muted">${r.source}</td>`
      + `<td><span class="pill ${r.status}">${r.status}</span></td>`
      + `<td>${v === "—" ? "—" : `<span class="pill ${vc}">${v}</span>`}</td>`
      + `<td>${r.llr_total === null || r.llr_total === undefined
              ? "—" : r.llr_total.toFixed(2)}</td></tr>`;
  }).join("");
  $("rows").innerHTML = rows || `<tr><td colspan="6" class="muted">`
    + `nothing submitted yet</td></tr>`;
  $("refreshed").textContent = "refreshed " + new Date().toLocaleTimeString();
  document.querySelectorAll("tr.click").forEach((tr) =>
    tr.onclick = () => showDetail(tr.dataset.id));
}

async function showDetail(id) {
  const { body } = await j("/api/submissions/" + id);
  $("detail").hidden = false;
  $("detail-title").textContent = body.filename + " — " + (body.verdict || body.status);
  $("detail-body").textContent = JSON.stringify(body.record ?? body, null, 2);
  $("detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
$("close").onclick = () => { $("detail").hidden = true; };

async function upload(files) {
  const note = $("upload-note");
  for (const f of files) {
    note.textContent = `uploading ${f.name}…`;
    const r = await fetch("/api/scan?filename=" + encodeURIComponent(f.name),
      { method: "POST", body: f });
    const b = await r.json().catch(() => ({}));
    note.textContent = r.ok ? `queued ${f.name} as ${b.id}`
                            : `refused ${f.name}: ${b.error || r.status}`;
  }
  refresh();
}
$("drop").onclick = () => $("file").click();
$("file").onchange = (e) => upload(e.target.files);
["dragenter", "dragover"].forEach((k) => $("drop").addEventListener(k, (e) => {
  e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave", "drop"].forEach((k) => $("drop").addEventListener(k, (e) => {
  e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", (e) => upload(e.dataTransfer.files));

function refresh() { loadStats(); loadRows(); }
loadEvidence(); refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""
