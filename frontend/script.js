const API_BASE = "http://localhost:8000";
const fallbackSymptoms = [
  "fever", "cough", "headache", "fatigue", "sore_throat", "shortness_of_breath",
  "nausea", "vomiting", "diarrhea", "chest_pain", "dizziness", "body_ache",
  "runny_nose", "loss_of_taste"
];

let allSymptoms = [];
let selected = new Set();

function setStatus(message, error = false) {
  const status = document.getElementById("status");
  status.textContent = message;
  status.className = `status-text ${error ? "text-danger" : ""}`;
}

async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`Failed to fetch ${path}`);
  return res.json();
}

function symptomChip(symptom) {
  const checked = selected.has(symptom) ? "checked" : "";
  return `<label class="symptom-chip"><input type="checkbox" class="symptom-check" value="${symptom}" ${checked}/> ${symptom.replaceAll("_", " ")}</label>`;
}

function renderSymptoms(list) {
  const el = document.getElementById("symptoms-container");
  el.innerHTML = list.map(symptomChip).join("");

  document.querySelectorAll(".symptom-check").forEach((node) => {
    node.addEventListener("change", () => {
      if (node.checked) selected.add(node.value);
      else selected.delete(node.value);
    });
  });
}

function filterSymptoms(query) {
  const lower = query.trim().toLowerCase();
  const filtered = !lower ? allSymptoms : allSymptoms.filter((s) => s.includes(lower));
  renderSymptoms(filtered);
}

function validateInput(payload) {
  if (!payload.symptoms.length) throw new Error("Select at least one symptom.");
  if (payload.age < 1 || payload.age > 100) throw new Error("Age must be 1-100.");
  if (payload.duration_days < 1 || payload.duration_days > 60) throw new Error("Duration must be 1-60 days.");
}

function predictionCard(p) {
  const percent = (p.confidence * 100).toFixed(1);
  return `
    <div class="prediction-card">
      <div class="d-flex justify-content-between mb-1">
        <strong>${p.disease}</strong>
        <span class="score">${percent}%</span>
      </div>
      <div class="progress" role="progressbar" aria-label="${p.disease} confidence">
        <div class="progress-bar" style="width:${percent}%"></div>
      </div>
    </div>
  `;
}

function renderResults(data) {
  const results = document.getElementById("results");
  const predictionHtml = (data.predictions || []).map(predictionCard).join("");
  const important = (data.important_symptoms || [])
    .map((symptom) => `<span class="tag">${symptom.replaceAll("_", " ")}</span>`)
    .join("");

  results.innerHTML = `
    ${predictionHtml}
    <div class="small mt-3 text-secondary">Input Summary: Age ${data.input_summary.age}, ${data.input_summary.gender}, Duration ${data.input_summary.duration_days} day(s).</div>
    <div class="mt-2"><strong>Important symptoms</strong></div>
    <div class="tag-wrap mt-1">${important}</div>
  `;
}

function renderModelInfo(info) {
  const container = document.getElementById("model-info");
  const metricRows = Object.entries(info.metrics || {})
    .map(([name, m]) => `<tr><td>${name}</td><td>${(m.accuracy * 100).toFixed(1)}%</td><td>${(m.f1_score * 100).toFixed(1)}%</td></tr>`)
    .join("");

  const topSymptoms = (info.top_global_symptoms || []).slice(0, 6)
    .map((x) => `<span class="tag">${x.symptom.replaceAll("_", " ")}</span>`)
    .join("");

  container.innerHTML = `
    <div class="small mb-2"><strong>Selected Best Model:</strong> ${info.best_model}</div>
    <div class="table-responsive">
      <table class="table table-sm align-middle mb-2">
        <thead><tr><th>Model</th><th>Accuracy</th><th>F1</th></tr></thead>
        <tbody>${metricRows}</tbody>
      </table>
    </div>
    <div class="small"><strong>Top Global Symptoms:</strong></div>
    <div class="tag-wrap mt-1">${topSymptoms}</div>
  `;
}

async function onPredict() {
  const button = document.getElementById("predict-btn");
  const payload = {
    age: Number(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    symptoms: [...selected],
    duration_days: Number(document.getElementById("duration").value),
    lifestyle: {
      smoking: document.getElementById("smoking").checked,
      alcohol: document.getElementById("alcohol").checked,
    },
  };

  try {
    validateInput(payload);
    setStatus("Running prediction...");
    button.disabled = true;

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");

    renderResults(data);
    setStatus("Prediction completed successfully.");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.disabled = false;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  try {
    const metadata = await fetchJson("/metadata");
    allSymptoms = metadata.symptoms || fallbackSymptoms;
  } catch {
    allSymptoms = fallbackSymptoms;
  }

  renderSymptoms(allSymptoms);

  try {
    const info = await fetchJson("/model-info");
    renderModelInfo(info);
  } catch {
    document.getElementById("model-info").textContent = "Model info unavailable until backend/model.pkl is generated.";
  }

  document.getElementById("predict-btn").addEventListener("click", onPredict);
  document.getElementById("symptom-search").addEventListener("input", (e) => filterSymptoms(e.target.value));
});
