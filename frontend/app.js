const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const results = document.getElementById("results");
const errorEl = document.getElementById("error");
const modeBadge = document.getElementById("mode-badge");

let selectedFile = null;

// Show demo/live mode badge.
fetch("/api/health")
  .then((r) => r.json())
  .then((d) => {
    if (d.demo_mode) {
      modeBadge.textContent = "DEMO MODE — no trained weights loaded";
      modeBadge.classList.remove("hidden");
    }
  })
  .catch(() => {});

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.classList.remove("hidden");
  analyzeBtn.disabled = false;
  results.classList.add("hidden");
  errorEl.classList.add("hidden");
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  errorEl.classList.add("hidden");

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    render(await res.json());
  } catch (err) {
    errorEl.textContent = "⚠️ " + err.message;
    errorEl.classList.remove("hidden");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze scan";
  }
});

function render(data) {
  document.getElementById("prediction").textContent = data.prediction;
  document.getElementById("confidence").textContent =
    (data.confidence * 100).toFixed(1) + "%";

  const bars = document.getElementById("bars");
  bars.innerHTML = "";
  data.probabilities.forEach((p) => {
    const pct = (p.probability * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label"><span>${p.label}</span><span>${pct}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>`;
    bars.appendChild(row);
  });

  const heatBlock = document.getElementById("heatmap-block");
  if (data.heatmap) {
    document.getElementById("heatmap").src = data.heatmap;
    heatBlock.classList.remove("hidden");
  } else {
    heatBlock.classList.add("hidden");
  }

  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
