const dropzone   = document.getElementById("dropzone");
const fileInput  = document.getElementById("file-input");
const previewWrap= document.getElementById("preview-wrap");
const preview    = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const results    = document.getElementById("results");
const emptyState = document.getElementById("empty-state");
const errorEl    = document.getElementById("error");
const modeBadge  = document.getElementById("mode-badge");

let selectedFile = null;

// Demo / live badge.
fetch("/api/health")
  .then((r) => r.json())
  .then((d) => {
    if (d.demo_mode) {
      modeBadge.textContent = "DEMO MODE";
      modeBadge.classList.remove("hidden");
    }
  })
  .catch(() => {});

// Browse + drag/drop.
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  e.stopPropagation();
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  dropzone.classList.add("has-file");
  analyzeBtn.disabled = false;
  results.classList.add("hidden");
  emptyState.classList.remove("hidden");
  errorEl.classList.add("hidden");
}

analyzeBtn.addEventListener("click", async (e) => {
  e.stopPropagation();
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.classList.add("loading");
  analyzeBtn.querySelector("span").textContent = "▶ ANALYZING…";
  previewWrap.classList.add("scanning");
  errorEl.classList.add("hidden");

  const form = new FormData();
  form.append("file", selectedFile);

  const started = Date.now();
  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    // Let the scan animation breathe for a beat.
    const wait = Math.max(0, 900 - (Date.now() - started));
    setTimeout(() => render(data), wait);
  } catch (err) {
    errorEl.textContent = "⚠ " + err.message;
    errorEl.classList.remove("hidden");
    resetButton();
  }
});

function resetButton() {
  analyzeBtn.disabled = false;
  analyzeBtn.classList.remove("loading");
  analyzeBtn.querySelector("span").textContent = "▶ RUN ANALYSIS";
  previewWrap.classList.remove("scanning");
}

function render(data) {
  emptyState.classList.add("hidden");

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
      <div class="bar-track"><div class="bar-fill"></div></div>`;
    bars.appendChild(row);
    // Animate fill on next frame.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        row.querySelector(".bar-fill").style.width = pct + "%";
      })
    );
  });

  const heatBlock = document.getElementById("heatmap-block");
  if (data.heatmap) {
    document.getElementById("heatmap").src = data.heatmap;
    heatBlock.classList.remove("hidden");
  } else {
    heatBlock.classList.add("hidden");
  }

  results.classList.remove("hidden");
  resetButton();
  results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
