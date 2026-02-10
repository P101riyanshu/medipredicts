const API_BASE = "http://localhost:8000";
const HISTORY_KEY = "cdss_case_history";
const fallbackSymptoms = [
  "fever", "cough", "headache", "fatigue", "sore_throat", "shortness_of_breath",
  "nausea", "vomiting", "diarrhea", "chest_pain", "dizziness", "body_ache", "runny_nose", "loss_of_taste"
];

const byId = (id) => document.getElementById(id);

async function fetchJSON(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = body?.detail || body?.error || `Request failed: ${path}`;
    throw new Error(message);
  }
  return body;
}

function toTitle(text) {
  return text.replaceAll("_", " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

function buildSymptomChip(symptom) {
  const label = document.createElement("label");
  label.className = "symptom-item";
  label.innerHTML = `<input type="checkbox" value="${symptom}" class="symptom-check" /> ${toTitle(symptom)}`;
  return label;
}

function renderPredictions(predictions = []) {
  return predictions.map((item) => {
    const percentage = (item.confidence * 100).toFixed(1);
    return `
      <div class="prediction-card">
        <div class="prediction-header"><strong>${item.disease}</strong><span class="prediction-score">${percentage}%</span></div>
        <div class="progress"><div class="progress-bar" style="width:${percentage}%"></div></div>
      </div>`;
  }).join("");
}

function renderImportantSymptoms(symptoms = []) {
  if (!symptoms.length) return "";
  const tags = symptoms.map((s) => `<span class="tag">${toTitle(s)}</span>`).join("");
  return `<div class="mt-3"><strong>Important Symptoms</strong><div class="important-tags">${tags}</div></div>`;
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(entry) {
  const history = [entry, ...readHistory()].slice(0, 8);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function renderHistoryTable() {
  const table = byId("history-table");
  if (!table) return;
  const tbody = table.querySelector("tbody");
  const history = readHistory();
  if (!history.length) {
    tbody.innerHTML = "<tr><td colspan='5' class='text-secondary'>No prediction history yet.</td></tr>";
    return;
  }
  tbody.innerHTML = history.map((h) => `
    <tr>
      <td>${h.time}</td>
      <td>${h.age} / ${h.gender}</td>
      <td>${h.topDisease}</td>
      <td>${h.topConfidence}%</td>
      <td>${h.symptoms}</td>
    </tr>`).join("");
}

async function initPredictPage() {
  if (!byId("predict-btn")) return;

  const symptomContainer = byId("symptoms-container");
  const searchInput = byId("symptom-search");
  const status = byId("status");
  const results = byId("results");
  const predictButton = byId("predict-btn");
  const clearButton = byId("clear-btn");

  let symptoms = fallbackSymptoms;
  try {
    const data = await fetchJSON("/metadata");
    symptoms = data.symptoms || fallbackSymptoms;
  } catch {
    symptoms = fallbackSymptoms;
  }

  const renderSymptoms = (items) => {
    symptomContainer.innerHTML = "";
    items.forEach((symptom) => symptomContainer.appendChild(buildSymptomChip(symptom)));
  };

  renderSymptoms(symptoms);
  renderHistoryTable();

  searchInput.addEventListener("input", (event) => {
    const q = event.target.value.toLowerCase().trim();
    renderSymptoms(!q ? symptoms : symptoms.filter((s) => s.includes(q)));
  });

  clearButton.addEventListener("click", () => {
    byId("age").value = 28;
    byId("gender").value = "female";
    byId("duration").value = 2;
    byId("smoking").checked = false;
    byId("alcohol").checked = false;
    searchInput.value = "";
    renderSymptoms(symptoms);
    status.textContent = "Form cleared.";
    status.classList.remove("text-danger");
    results.innerHTML = "";
  });

  byId("clear-history-btn")?.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistoryTable();
  });

  predictButton.addEventListener("click", async () => {
    status.classList.remove("text-danger");

    const payload = {
      age: Number(byId("age").value),
      gender: byId("gender").value,
      symptoms: [...document.querySelectorAll(".symptom-check:checked")].map((el) => el.value),
      duration_days: Number(byId("duration").value),
      lifestyle: { smoking: byId("smoking").checked, alcohol: byId("alcohol").checked },
    };

    try {
      if (!payload.symptoms.length) throw new Error("Please select at least one symptom.");
      status.textContent = "Running model inference...";
      results.innerHTML = "";
      predictButton.disabled = true;

      const data = await fetchJSON("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      results.innerHTML = `${renderPredictions(data.predictions)}${renderImportantSymptoms(data.important_symptoms)}`;
      status.textContent = `Completed using ${data.model_used || "best model"}.`;

      const top = data.predictions?.[0];
      if (top) {
        saveHistory({
          time: new Date().toLocaleString(),
          age: payload.age,
          gender: payload.gender,
          topDisease: top.disease,
          topConfidence: (top.confidence * 100).toFixed(1),
          symptoms: payload.symptoms.slice(0, 4).map(toTitle).join(", "),
        });
        renderHistoryTable();
      }
    } catch (error) {
      status.textContent = error.message;
      status.classList.add("text-danger");
    } finally {
      predictButton.disabled = false;
    }
  });
}

async function initMetricsPage() {
  if (!byId("metrics-table")) return;

  const tbody = document.querySelector("#metrics-table tbody");
  const kpiRow = byId("kpi-row");
  try {
    const data = await fetchJSON("/metrics");
    const entries = Object.entries(data.results || {});

    tbody.innerHTML = entries.map(([name, metric]) => {
      const isBest = name === data.best_model;
      return `
      <tr>
        <td>${name}</td>
        <td>${metric.accuracy}</td>
        <td>${metric.precision}</td>
        <td>${metric.recall}</td>
        <td>${metric.f1_score}</td>
        <td>${isBest ? '<span class="badge text-bg-success">Best</span>' : '<span class="badge text-bg-secondary">Compared</span>'}</td>
      </tr>`;
    }).join("");

    const best = entries.find(([name]) => name === data.best_model)?.[1] || {};
    kpiRow.innerHTML = `
      <div class="col-md-4"><div class="metric-card"><div class="text-secondary">Best Model</div><h5>${data.best_model}</h5></div></div>
      <div class="col-md-4"><div class="metric-card"><div class="text-secondary">Best F1 Score</div><h5>${best.f1_score ?? "-"}</h5></div></div>
      <div class="col-md-4"><div class="metric-card"><div class="text-secondary">Total Models</div><h5>${entries.length}</h5></div></div>`;
  } catch (error) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-danger">${error.message}</td></tr>`;
  }
}

async function initSymptomsPage() {
  if (!byId("importance-list")) return;

  let rows = [];
  const render = (query = "") => {
    const q = query.toLowerCase();
    const filtered = rows.filter((item) => item.symptom.toLowerCase().includes(q));
    byId("importance-list").innerHTML = filtered
      .map((item, idx) => `
        <div class="list-group-item">
          <div class="d-flex justify-content-between"><span>${idx + 1}. ${toTitle(item.symptom)}</span><strong>${(item.importance * 100).toFixed(2)}%</strong></div>
          <div class="progress mt-2"><div class="progress-bar" style="width:${(item.importance * 100).toFixed(2)}%"></div></div>
        </div>`)
      .join("") || "<div class='list-group-item text-secondary'>No matching symptoms.</div>";
  };

  try {
    const data = await fetchJSON("/importance?limit=20");
    rows = data.top_symptoms || [];
    render();
    byId("importance-search")?.addEventListener("input", (e) => render(e.target.value));
  } catch (error) {
    byId("importance-list").innerHTML = `<div class="list-group-item text-danger">${error.message}</div>`;
  }
}

async function initApiPlaygroundPage() {
  if (!byId("send-api-btn")) return;

  const payloadInput = byId("api-payload");
  const responseOutput = byId("api-response");
  const loadSampleBtn = byId("load-sample-btn");
  const downloadResponseBtn = byId("download-response-btn");

  const fillSample = async () => {
    try {
      const sample = await fetchJSON("/sample-payload");
      payloadInput.value = JSON.stringify(sample, null, 2);
    } catch {
      payloadInput.value = JSON.stringify({
        age: 28,
        gender: "female",
        symptoms: ["fever", "cough", "headache"],
        duration_days: 2,
        lifestyle: { smoking: false, alcohol: false }
      }, null, 2);
    }
  };

  await fillSample();
  loadSampleBtn.addEventListener("click", fillSample);

  byId("send-api-btn").addEventListener("click", async () => {
    try {
      const parsed = JSON.parse(payloadInput.value);
      const body = await fetchJSON("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      responseOutput.textContent = JSON.stringify(body, null, 2);
    } catch (error) {
      responseOutput.textContent = `Error: ${error.message}`;
    }
  });

  downloadResponseBtn.addEventListener("click", () => {
    const content = responseOutput.textContent || "{}";
    const blob = new Blob([content], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "cdss-response.json";
    link.click();
    URL.revokeObjectURL(link.href);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initPredictPage();
  initMetricsPage();
  initSymptomsPage();
  initApiPlaygroundPage();
});
