const API_BASE = "http://localhost:8000";
const fallbackSymptoms = [
  "fever", "cough", "headache", "fatigue", "sore_throat", "shortness_of_breath",
  "nausea", "vomiting", "diarrhea", "chest_pain", "dizziness", "body_ache",
  "runny_nose", "loss_of_taste"
];

function byId(id) {
  return document.getElementById(id);
}

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`Request failed: ${path}`);
  return res.json();
}

async function fetchSymptoms() {
  try {
    const data = await fetchJSON("/metadata");
    return data.symptoms || fallbackSymptoms;
  } catch {
    return fallbackSymptoms;
  }
}

function buildSymptomChip(symptom) {
  const label = document.createElement("label");
  label.className = "symptom-item";
  label.innerHTML = `<input type="checkbox" value="${symptom}" class="symptom-check" /> ${symptom.replaceAll("_", " ")}`;
  return label;
}

function renderPredictions(predictions) {
  return predictions.map((item) => {
    const percentage = (item.confidence * 100).toFixed(1);
    return `<div class="prediction-card"><div class="prediction-header"><strong>${item.disease}</strong><span class="prediction-score">${percentage}%</span></div><div class="progress"><div class="progress-bar" style="width:${percentage}%"></div></div></div>`;
  }).join("");
}

function renderImportantSymptoms(important) {
  if (!important?.length) return "";
  const tags = important.map((s) => `<span class="tag">${s.replaceAll("_", " ")}</span>`).join("");
  return `<div class="mt-3"><strong>Important symptoms</strong><div class="important-tags">${tags}</div></div>`;
}

async function initPredictPage() {
  if (!byId("predict-btn")) return;
  const container = byId("symptoms-container");
  const search = byId("symptom-search");
  const status = byId("status");
  const results = byId("results");
  let allSymptoms = await fetchSymptoms();

  const renderSymptoms = (symptoms) => {
    container.innerHTML = "";
    symptoms.forEach((symptom) => container.appendChild(buildSymptomChip(symptom)));
  };

  renderSymptoms(allSymptoms);
  search.addEventListener("input", (e) => {
    const query = e.target.value.trim().toLowerCase();
    const filtered = !query ? allSymptoms : allSymptoms.filter((s) => s.includes(query));
    renderSymptoms(filtered);
  });

  byId("predict-btn").addEventListener("click", async () => {
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
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed");
      status.textContent = "Prediction complete.";
      results.innerHTML = `${renderPredictions(data.predictions)}${renderImportantSymptoms(data.important_symptoms)}`;
    } catch (error) {
      status.textContent = error.message;
      status.classList.add("text-danger");
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
    tbody.innerHTML = entries.map(([name, m]) => `<tr><td>${name}</td><td>${m.accuracy}</td><td>${m.precision}</td><td>${m.recall}</td><td>${m.f1_score}</td></tr>`).join("");
    kpiRow.innerHTML = `<div class="col-md-4"><div class="kpi"><div class="text-secondary">Best Model</div><h4>${data.best_model}</h4></div></div><div class="col-md-4"><div class="kpi"><div class="text-secondary">Models Trained</div><h4>${entries.length}</h4></div></div>`;
  } catch {
    tbody.innerHTML = "<tr><td colspan='5'>Unable to load metrics. Start backend API first.</td></tr>";
  }
}

async function initSymptomsPage() {
  if (!byId("importance-list")) return;
  try {
    const data = await fetchJSON("/importance?limit=14");
    byId("importance-list").innerHTML = data.top_symptoms.map((item, idx) => `<div class='list-group-item d-flex justify-content-between'><span>${idx + 1}. ${item.symptom.replaceAll("_", " ")}</span><strong>${(item.importance * 100).toFixed(2)}%</strong></div>`).join("");
  } catch {
    byId("importance-list").innerHTML = "<div class='list-group-item'>Unable to load symptom importance.</div>";
  }
}

function initPlaygroundPage() {
  if (!byId("send-api-btn")) return;
  const payloadInput = byId("api-payload");
  const responseOutput = byId("api-response");
  payloadInput.value = JSON.stringify({
    age: 28,
    gender: "female",
    symptoms: ["fever", "cough", "headache"],
    duration_days: 2,
    lifestyle: { smoking: false, alcohol: false }
  }, null, 2);

  byId("send-api-btn").addEventListener("click", async () => {
    try {
      const payload = JSON.parse(payloadInput.value);
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const body = await res.json();
      responseOutput.textContent = JSON.stringify(body, null, 2);
    } catch (err) {
      responseOutput.textContent = `Error: ${err.message}`;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initPredictPage();
  initMetricsPage();
  initSymptomsPage();
  initPlaygroundPage();
});
