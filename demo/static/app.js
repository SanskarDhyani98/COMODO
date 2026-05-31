// COMODO Live Demo — frontend logic
const CHANNEL_COLORS = [
  "#7c83ff", "#5cd2c6", "#facc15",
  "#fb7185", "#4ade80", "#f97316",
];
const CHANNEL_LABELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"];

Chart.defaults.color = "#9aa6d6";
Chart.defaults.borderColor = "#2c376b";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, Inter, Segoe UI, sans-serif";

// ─── tab nav ───────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    document.getElementById("tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "results" && !resultsLoaded) loadResults();
    if (t.dataset.tab === "improvements" && !improvementsBootstrapped) {
      bootstrapImprovements();
    }
  });
});

// ─── state ────────────────────────────────────────────────
let selectedActivity = "walking";
let activitiesLoaded = false;
let resultsLoaded = false;
let imuChart, embedChart, simChart, resultsChart;
let sweepChart, batchedChart, salChart;
let improvementsBootstrapped = false;

// ─── boot ─────────────────────────────────────────────────
async function boot() {
  await loadActivities();
  await refreshHealth();
  await previewIMU();   // load default activity preview
}
boot();

// ─── health / warmup ──────────────────────────────────────
async function refreshHealth() {
  const r = await fetch("/api/health").then((x) => x.json());
  document.getElementById("statusDevice").textContent = r.device;
  document.getElementById("statusState").textContent = r.model_loaded ? "loaded" : "cold";
  document.getElementById("statusLoad").textContent = r.load_seconds
    ? r.load_seconds + " s"
    : "—";
}

document.getElementById("warmupBtn").addEventListener("click", async () => {
  const btn = document.getElementById("warmupBtn");
  btn.disabled = true;
  btn.textContent = "Loading Mantis-8M...";
  try {
    const info = await fetch("/api/model-info").then((x) => x.json());
    btn.textContent = "Model ready ✓";
    renderModelInfo(info);
    await refreshHealth();
  } catch (e) {
    btn.textContent = "Load failed";
    console.error(e);
  } finally {
    setTimeout(() => (btn.disabled = false), 400);
  }
});

function renderModelInfo(info) {
  const fmt = (n) => n.toLocaleString();
  document.getElementById("modelInfo").innerHTML = `
<b>IMU backbone:</b>           ${info.imu_backbone}
<b>Projection head:</b>        ${info.projection_head}
<b>Total parameters:</b>       ${fmt(info.total_parameters)}
<b>Trainable parameters:</b>   ${fmt(info.trainable_parameters)}
<b>Device:</b>                 ${info.device}
<b>Input shape:</b>            [${info.input_shape.join(", ")}]   (batch, channels, time)
<b>Output shape:</b>           [${info.output_shape.join(", ")}]   (batch, embedding)
<b>Cold-start load time:</b>   ${info.load_seconds} s
<b>Cross-modal queue size:</b> ${info.queue_size}`;
}

// ─── activities ───────────────────────────────────────────
async function loadActivities() {
  const r = await fetch("/api/activities").then((x) => x.json());
  const grid = document.getElementById("activityGrid");
  const sel = document.getElementById("lossActivity");
  grid.innerHTML = "";
  sel.innerHTML = "";
  r.activities.forEach((a, i) => {
    const div = document.createElement("div");
    div.className = "activity" + (a.id === selectedActivity ? " selected" : "");
    div.innerHTML = `<div class="activity-title">${a.label}</div>
                     <div class="activity-desc">${a.description}</div>`;
    div.addEventListener("click", () => {
      selectedActivity = a.id;
      document.querySelectorAll(".activity").forEach((x) => x.classList.remove("selected"));
      div.classList.add("selected");
      previewIMU();
    });
    grid.appendChild(div);

    const opt = document.createElement("option");
    opt.value = a.id;
    opt.textContent = a.label;
    sel.appendChild(opt);
  });
  activitiesLoaded = true;
}

// ─── IMU plot ────────────────────────────────────────────
async function previewIMU() {
  const seed = +document.getElementById("seedInput").value || 42;
  const r = await fetch(`/api/imu/${selectedActivity}?seed=${seed}`).then((x) => x.json());
  drawIMU(r);
}

function drawIMU(r) {
  const labels = r.samples[0].map((_, i) => (i * r.sample_step / 100).toFixed(2));
  const datasets = r.samples.map((s, i) => ({
    label: r.channels[i],
    data: s,
    borderColor: CHANNEL_COLORS[i],
    backgroundColor: CHANNEL_COLORS[i] + "33",
    borderWidth: 1.4,
    pointRadius: 0,
    tension: 0.25,
  }));
  if (imuChart) imuChart.destroy();
  imuChart = new Chart(document.getElementById("imuChart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      animation: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: { title: { display: true, text: "time (s)" }, ticks: { maxTicksLimit: 11 } },
        y: { title: { display: true, text: "sensor value" } },
      },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 14, padding: 10 } } },
    },
  });
}

// ─── inference ────────────────────────────────────────────
document.getElementById("runBtn").addEventListener("click", runInference);
document.getElementById("seedInput").addEventListener("change", previewIMU);

async function runInference() {
  const seed = +document.getElementById("seedInput").value || 42;
  const tta = document.getElementById("ttaToggle").checked;
  const kproto = document.getElementById("kprotoToggle").checked;
  const status = document.getElementById("runStatus");
  status.textContent = "running…";
  status.className = "status-pill busy";
  try {
    const r = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        activity: selectedActivity,
        seed,
        tta,
        multi_prototype: kproto,
      }),
    }).then((x) => x.json());
    drawEmbedding(r.embedding);
    drawSimilarities(r.similarities, selectedActivity);
    renderPrediction(r);
    const flags = [
      r.tta ? `TTA×${r.tta_n}` : "TTA off",
      r.multi_prototype ? `K=${r.k_prototypes}` : "K=1",
    ].join(", ");
    document.getElementById("latencyBox").innerHTML =
      `<b>Forward pass:</b>  ${r.latency_ms} ms
<b>Embedding dim:</b> ${r.embedding_dim}
<b>‖z‖₂:</b>           ${r.embedding_norm.toFixed(4)}
<b>Config:</b>         ${flags}
<b>Confidence:</b>     ${(r.predicted_confidence * 100).toFixed(1)}%`;
    status.textContent = `done in ${r.latency_ms} ms`;
    status.className = "status-pill done";
    await refreshHealth();
  } catch (e) {
    status.textContent = "error";
    status.className = "status-pill err";
    console.error(e);
  }
}

function drawEmbedding(emb) {
  const labels = emb.map((_, i) => i);
  if (embedChart) embedChart.destroy();
  embedChart = new Chart(document.getElementById("embedChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: emb,
        backgroundColor: emb.map((v) => (v >= 0 ? "#7c83ff" : "#fb7185")),
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        x: { title: { display: true, text: "embedding dim (0 … 127)" }, ticks: { maxTicksLimit: 16 } },
        y: { title: { display: true, text: "value" } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function drawSimilarities(sims, truth) {
  const labels = Object.keys(sims);
  const data = labels.map((k) => sims[k]);
  if (simChart) simChart.destroy();
  simChart = new Chart(document.getElementById("simChart"), {
    type: "bar",
    data: {
      labels: labels.map((k) => k.replace("_", " ")),
      datasets: [{
        label: "cosine similarity",
        data,
        backgroundColor: labels.map((k) => (k === truth ? "#5cd2c6" : "#7c83ff")),
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      animation: false,
      scales: { x: { min: -1, max: 1, title: { display: true, text: "cos(z_query, z_prototype)" } } },
      plugins: { legend: { display: false } },
    },
  });
}

function renderPrediction(r) {
  const box = document.getElementById("predBox");
  const cls = r.correct ? "correct" : "wrong";
  const icon = r.correct ? "✓" : "✗";
  box.className = "pred-box " + cls;
  // Top-3 calibrated probabilities for the audience.
  const top3 = Object.entries(r.confidence)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(
      ([k, p]) =>
        `<div style="display:flex;justify-content:space-between;gap:12px;font-size:13px;color:var(--muted)">` +
        `<span>${k.replace("_", " ")}</span>` +
        `<span style="color:var(--text)"><b>${(p * 100).toFixed(1)}%</b></span></div>`
    )
    .join("");
  box.innerHTML =
    `<div style="font-size:13px;color:var(--muted);margin-bottom:6px">predicted</div>` +
    `<div><b>${r.predicted_label}</b> ${icon}` +
    `<span style="font-size:14px;color:var(--muted);font-weight:400"> · ${(r.predicted_confidence * 100).toFixed(1)}% confident</span></div>` +
    `<div style="font-size:13px;color:var(--muted);margin-top:10px">true label: <b>${r.activity}</b></div>` +
    `<div style="margin-top:12px;padding-top:10px;border-top:1px dashed rgba(255,255,255,0.1)">${top3}</div>`;
}

// ─── loss ─────────────────────────────────────────────────
document.getElementById("lossBtn").addEventListener("click", async () => {
  const activity = document.getElementById("lossActivity").value || "walking";
  const seed = +document.getElementById("lossSeed").value || 42;
  document.getElementById("lossBox").textContent = "running…";
  try {
    const r = await fetch("/api/loss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity, seed }),
    }).then((x) => x.json());
    document.getElementById("lossBox").innerHTML =
      `<b>Activity:</b>        ${r.activity}
<b>COMODOLoss:</b>      ${r.loss.toFixed(6)}
<b>teacher_temp:</b>    ${r.teacher_temp}
<b>student_temp:</b>    ${r.student_temp}
<b>queue_size:</b>      ${r.queue_size}
<b>latency:</b>         ${r.latency_ms} ms`;
  } catch (e) {
    document.getElementById("lossBox").textContent = "error: " + e;
  }
});

// ─── results ──────────────────────────────────────────────
async function loadResults() {
  try {
    const r = await fetch("/api/results").then((x) => x.json());
    drawResults(r.summary);
    fillResultsTable(r.summary);
    resultsLoaded = true;
  } catch (e) {
    console.error(e);
  }
}

function drawResults(summary) {
  // X axis: dataset+IMU pair; Y: acc@1; series: method (IMU2CLIP&L2 vs COMODO)
  const pairs = [...new Set(summary.map((r) => `${r.dataset}\n${r.imu}`))].sort();
  const methods = ["IMU2CLIP&L2", "COMODO"];
  const colors = { "IMU2CLIP&L2": "#fb7185", "COMODO": "#5cd2c6" };
  const datasets = methods.map((m) => ({
    label: m,
    backgroundColor: colors[m],
    borderRadius: 6,
    data: pairs.map((p) => {
      const [dataset, imu] = p.split("\n");
      const row = summary.find((s) => s.dataset === dataset && s.imu === imu && s.method === m);
      return row ? row.best_acc1 : null;
    }),
  }));
  if (resultsChart) resultsChart.destroy();
  resultsChart = new Chart(document.getElementById("resultsChart"), {
    type: "bar",
    data: { labels: pairs.map((p) => p.replace("\n", " · ")), datasets },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100, title: { display: true, text: "best top-1 accuracy (%)" } },
        x: { ticks: { autoSkip: false, maxRotation: 30, minRotation: 0 } },
      },
      plugins: { legend: { position: "top" } },
    },
  });
}

// ─── improvements tab ─────────────────────────────────────
async function bootstrapImprovements() {
  improvementsBootstrapped = true;
  // Pre-fill the saliency activity dropdown from the same activities list.
  const sel = document.getElementById("salActivity");
  if (sel && sel.options.length === 0) {
    document.querySelectorAll("#activityGrid .activity").forEach((el) => {
      const id = el.querySelector(".activity-title").textContent.toLowerCase().replace(/\s+/g, "_");
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = el.querySelector(".activity-title").textContent;
      sel.appendChild(opt);
    });
    // Use canonical ids from /api/activities — the human label split above
    // is too brittle. Re-fetch and overwrite to be safe.
    const acts = await fetch("/api/activities").then((x) => x.json());
    sel.innerHTML = "";
    acts.activities.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a.id;
      opt.textContent = a.label;
      sel.appendChild(opt);
    });
  }
  // Show the cached batched-vs-sequential numbers immediately (needs model warmup).
  try {
    const r = await fetch("/api/batched-bench").then((x) => x.json());
    drawBatchedBench(r);
  } catch (e) {
    document.getElementById("batchedKv").textContent =
      "Warm up the model first (top-right of page).";
  }
}

document.getElementById("sweepBtn").addEventListener("click", runSweep);
document.getElementById("salBtn").addEventListener("click", runSaliency);
document.getElementById("thrBtn").addEventListener("click", runThroughput);

async function runSweep() {
  const samples = +document.getElementById("sweepSamples").value || 3;
  const btn = document.getElementById("sweepBtn");
  const status = document.getElementById("sweepStatus");
  btn.disabled = true;
  status.textContent = "running on CPU…";
  status.className = "status-pill busy";
  try {
    const r = await fetch("/api/noise-sweep", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ samples_per_activity: samples }),
    }).then((x) => x.json());
    drawSweep(r);
    document.getElementById("sweepSamplesLabel").textContent =
      r.samples_per_point + ` (${samples}/activity × 6 activities)`;
    status.textContent = `done in ${r.elapsed_s}s`;
    status.className = "status-pill done";
  } catch (e) {
    status.textContent = "error";
    status.className = "status-pill err";
    console.error(e);
  } finally {
    btn.disabled = false;
  }
}

function drawSweep(r) {
  const colors = {
    "baseline": "#fb7185",
    "+ K-prototypes": "#facc15",
    "+ TTA": "#7c83ff",
    "+ K-proto + TTA": "#5cd2c6",
  };
  const styles = {
    "baseline": { dash: [6, 4], width: 2 },
    "+ K-prototypes": { dash: [], width: 2 },
    "+ TTA": { dash: [], width: 2 },
    "+ K-proto + TTA": { dash: [], width: 3 },
  };
  const datasets = r.configs.map((c) => ({
    label: c,
    data: r.accuracy_by_config[c],
    borderColor: colors[c],
    backgroundColor: colors[c] + "22",
    borderWidth: styles[c].width,
    borderDash: styles[c].dash,
    pointRadius: 4,
    pointHoverRadius: 6,
    tension: 0.25,
  }));
  if (sweepChart) sweepChart.destroy();
  sweepChart = new Chart(document.getElementById("sweepChart"), {
    type: "line",
    data: { labels: r.noise_levels.map((x) => "σ=" + x.toFixed(2)), datasets },
    options: {
      responsive: true,
      animation: { duration: 400 },
      scales: {
        x: { title: { display: true, text: "added Gaussian noise σ" } },
        y: {
          title: { display: true, text: "top-1 accuracy (%)" },
          min: 0, max: 100,
        },
      },
      plugins: { legend: { position: "top" } },
    },
  });
}

function drawBatchedBench(r) {
  if (batchedChart) batchedChart.destroy();
  batchedChart = new Chart(document.getElementById("batchedChart"), {
    type: "bar",
    data: {
      labels: [
        `Sequential\n(${r.n_forward_passes_sequential} forward passes)`,
        `Batched\n(1 forward pass)`,
      ],
      datasets: [{
        data: [r.sequential_ms, r.batched_ms],
        backgroundColor: ["#fb7185", "#5cd2c6"],
        borderRadius: 8,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      animation: false,
      scales: { x: { title: { display: true, text: "ms (lower is better)" } } },
      plugins: { legend: { display: false } },
    },
  });
  document.getElementById("batchedKv").innerHTML =
    `<b>Sequential:</b> ${r.sequential_ms} ms (${r.n_forward_passes_sequential} forward passes)
<b>Batched:</b>    ${r.batched_ms} ms (1 forward pass)
<b>Speedup:</b>    ${r.speedup}×`;
}

let thrChart;
async function runThroughput() {
  const btn = document.getElementById("thrBtn");
  const status = document.getElementById("thrStatus");
  btn.disabled = true;
  status.textContent = "benchmarking…";
  status.className = "status-pill busy";
  try {
    const r = await fetch("/api/throughput-bench", { method: "GET" }).then((x) => x.json());
    drawThroughput(r);
    status.textContent = "done";
    status.className = "status-pill done";
  } catch (e) {
    status.textContent = "error";
    status.className = "status-pill err";
    console.error(e);
  } finally {
    btn.disabled = false;
  }
}

function drawThroughput(r) {
  const labels = r.results.map((x) => "bs=" + x.batch_size);
  if (thrChart) thrChart.destroy();
  thrChart = new Chart(document.getElementById("thrChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "queries / second",
          data: r.results.map((x) => x.queries_per_sec),
          backgroundColor: "#5cd2c6",
          borderRadius: 6,
          yAxisID: "y",
        },
        {
          label: "latency (ms)",
          data: r.results.map((x) => x.latency_ms),
          backgroundColor: "#facc15",
          borderRadius: 6,
          yAxisID: "y1",
          type: "line",
          borderColor: "#facc15",
          tension: 0.25,
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 400 },
      scales: {
        y:  { position: "left",  title: { display: true, text: "queries/sec" }, beginAtZero: true },
        y1: { position: "right", title: { display: true, text: "latency (ms)" }, grid: { drawOnChartArea: false }, beginAtZero: true },
      },
      plugins: { legend: { position: "top" } },
    },
  });
  const fmt = (n) => n.toLocaleString();
  const rows = r.results.map(
    (x) => `<b>bs=${String(x.batch_size).padStart(2," ")}</b>  ${String(x.latency_ms).padStart(7," ")} ms   ${String(x.queries_per_sec).padStart(7," ")} q/s   (${x.speedup_vs_bs1}× vs bs=1)`
  );
  document.getElementById("thrKv").innerHTML = rows.join("\n");
}

async function runSaliency() {
  const activity = document.getElementById("salActivity").value || "walking";
  const seed = +document.getElementById("salSeed").value || 42;
  try {
    const r = await fetch("/api/saliency", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activity, seed }),
    }).then((x) => x.json());
    drawSaliency(r);
  } catch (e) {
    console.error(e);
  }
}

function drawSaliency(r) {
  if (salChart) salChart.destroy();
  const colors = ["#7c83ff", "#5cd2c6", "#facc15", "#fb7185", "#4ade80", "#f97316"];
  salChart = new Chart(document.getElementById("salChart"), {
    type: "bar",
    data: {
      labels: r.channels,
      datasets: [{
        label: `importance for "${r.activity}"`,
        data: r.importance,
        backgroundColor: colors,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      animation: { duration: 400 },
      scales: {
        y: { title: { display: true, text: "relative importance" }, beginAtZero: true },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function fillResultsTable(summary) {
  const tbody = document.querySelector("#resultsTable tbody");
  tbody.innerHTML = "";
  // sort: COMODO first within each (dataset, imu)
  summary.sort((a, b) => {
    if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset);
    if (a.imu !== b.imu) return a.imu.localeCompare(b.imu);
    return a.method === "COMODO" ? -1 : 1;
  });
  for (const r of summary) {
    const tr = document.createElement("tr");
    const isCOMODO = r.method === "COMODO";
    tr.innerHTML = `
      <td class="${isCOMODO ? "method-comodo" : ""}">${r.method}</td>
      <td>${r.imu}</td>
      <td>${r.dataset}</td>
      <td>${r.best_acc1.toFixed(2)}%</td>
      <td>${r.best_acc3.toFixed(2)}%</td>
      <td>${r.best_acc5.toFixed(2)}%</td>
      <td>${r.n_runs}</td>`;
    tbody.appendChild(tr);
  }
}
