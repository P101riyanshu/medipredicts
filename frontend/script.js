const API_BASE = "http://localhost:8000";
const fallbackSymptoms = [
  "fever", "cough", "headache", "fatigue", "sore_throat", "shortness_of_breath",
  "nausea", "vomiting", "diarrhea", "chest_pain", "dizziness", "body_ache",
  "runny_nose", "loss_of_taste"
];

let allSymptoms = [];

function setStatus(message, isError = false) {
  const status = document.getElementById("status");
  status.textContent = message;
  status.className = `status-text ${isError ? "text-danger" : ""}`;
}

async function fetchSymptoms() {
  try {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    return data.symptoms || fallbackSymptoms;
  } catch {
    return fallbackSymptoms;
  }
}

function buildSymptomChip(symptom) {
  const label = document.createElement("label");
  label.className = "symptom-item";
  label.dataset.symptom = symptom;
  label.innerHTML = `<input type="checkbox" value="${symptom}" class="symptom-check" /> ${symptom.replaceAll("_", " ")}`;
  return label;
}

function renderSymptoms(symptoms) {
  const container = document.getElementById("symptoms-container");
  container.innerHTML = "";
  symptoms.forEach((symptom) => container.appendChild(buildSymptomChip(symptom)));
}

function filterSymptoms(keyword) {
  const query = keyword.trim().toLowerCase();
  const filtered = !query
    ? allSymptoms
    : allSymptoms.filter((s) => s.toLowerCase().includes(query));
  renderSymptoms(filtered);
}

function selectedSymptoms() {
  return [...document.querySelectorAll(".symptom-check:checked")].map((el) => el.value);
}

function renderPredictions(predictions) {
  return predictions.map((item) => {
    const percentage = (item.confidence * 100).toFixed(1);
    return `
      <div class="prediction-card">
        <div class="prediction-header">
          <strong>${item.disease}</strong>
          <span class="prediction-score">${percentage}%</span>
        </div>
        <div class="progress" role="progressbar" aria-label="${item.disease} confidence">
          <div class="progress-bar" style="width:${percentage}%"></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderImportantSymptoms(important) {
  if (!important?.length) return "";
  const tags = important.map((s) => `<span class="tag">${s.replaceAll("_", " ")}</span>`).join("");
  return `
    <div class="mt-3">
      <div><strong>Important symptoms</strong></div>
      <div class="important-tags">${tags}</div>
    </div>
  `;
}

function validatePayload(payload) {
  if (!payload.age || payload.age < 1 || payload.age > 100) {
    throw new Error("Age must be between 1 and 100.");
  }
  if (!payload.duration_days || payload.duration_days < 1 || payload.duration_days > 60) {
    throw new Error("Duration must be between 1 and 60 days.");
  }
  if (!payload.symptoms.length) {
    throw new Error("Please select at least one symptom.");
  }
}

async function submitPrediction() {
  const button = document.getElementById("predict-btn");
  const results = document.getElementById("results");

  const payload = {
    age: Number(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    symptoms: selectedSymptoms(),
    duration_days: Number(document.getElementById("duration").value),
    lifestyle: {
      smoking: document.getElementById("smoking").checked,
      alcohol: document.getElementById("alcohol").checked,
    },
  };

  try {
    validatePayload(payload);
    button.disabled = true;
    button.textContent = "Predicting...";
    setStatus("Running model inference...");
    results.innerHTML = "";

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed.");
    }

    const data = await res.json();
    results.innerHTML = `
      ${renderPredictions(data.predictions || [])}
      ${renderImportantSymptoms(data.important_symptoms || [])}
    `;
    setStatus("Prediction complete.");
  } catch (error) {
    setStatus(error.message || "Unexpected error", true);
  } finally {
    button.disabled = false;
    button.textContent = "Predict Top 3 Diseases";
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  allSymptoms = await fetchSymptoms();
  renderSymptoms(allSymptoms);

  document.getElementById("predict-btn").addEventListener("click", submitPrediction);
  document.getElementById("symptom-search").addEventListener("input", (e) => {
    filterSymptoms(e.target.value);
  });
});
