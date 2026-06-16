const dropzone   = document.getElementById("dropzone");
const fileInput  = document.getElementById("file-input");
const previewWrap= document.getElementById("preview-wrap");
const preview    = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const results    = document.getElementById("results");
const emptyState = document.getElementById("empty-state");
const errorEl    = document.getElementById("error");
const modeBadge  = document.getElementById("mode-badge");
const validation = document.getElementById("validation");

let selectedFile = null;

const MAX_BYTES = 10 * 1024 * 1024;
const VALID_HINT = "PNG / JPG · max 10 MB · a single 2D axial brain slice";

function setValidation(msg, invalid) {
  validation.innerHTML = msg;
  validation.classList.toggle("invalid", !!invalid);
}

// Theme toggle: Lab (default) <-> Clinical, persisted in localStorage.
const themeToggle = document.getElementById("theme-toggle");
function applyTheme(mode) {
  const clinical = mode === "clinical";
  document.body.classList.toggle("clinical", clinical);
  if (themeToggle) themeToggle.textContent = clinical ? "◑ LAB VIEW" : "◐ CLINICAL VIEW";
}
applyTheme(localStorage.getItem("alz-theme") || "lab");
if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const next = document.body.classList.contains("clinical") ? "lab" : "clinical";
    localStorage.setItem("alz-theme", next);
    applyTheme(next);
  });
}

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
  if (!file) return;
  // Client-side validation -> the always-visible slot under the button.
  if (!file.type.startsWith("image/")) {
    setValidation("✗ UNSUPPORTED FILE — need a PNG or JPG image (not " +
      (file.type || file.name.split(".").pop() || "?") + ")", true);
    return;
  }
  if (file.size > MAX_BYTES) {
    setValidation("✗ TOO LARGE — " + (file.size / 1048576).toFixed(1) +
      " MB exceeds the 10 MB limit", true);
    return;
  }
  setValidation(VALID_HINT, false);
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  dropzone.classList.add("has-file");
  analyzeBtn.disabled = false;
  results.classList.add("hidden");
  emptyState.classList.remove("hidden");
  errorEl.classList.add("hidden");
}

// ---- Sample images: probe /static/samples/<key>.jpg, reveal only what exists.
document.querySelectorAll(".sample-btn").forEach((btn) => {
  const url = "/static/samples/" + btn.dataset.key + ".jpg";
  fetch(url, { method: "HEAD" })
    .then((r) => {
      if (r.ok) {
        btn.classList.remove("hidden");
        document.getElementById("samples").classList.remove("hidden");
      }
    })
    .catch(() => {});
  btn.addEventListener("click", async () => {
    try {
      const blob = await (await fetch(url)).blob();
      handleFile(new File([blob], btn.dataset.key + ".jpg", { type: blob.type }));
    } catch {
      setValidation("✗ Could not load that sample image", true);
    }
  });
});

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

// Clinical severity scale: stage label -> {level, status word}.
const SEVERITY = {
  "Non Demented":       { level: 0, status: "No dementia indicated" },
  "Very mild Dementia": { level: 1, status: "Very mild — early stage" },
  "Mild Dementia":      { level: 2, status: "Mild stage" },
  "Moderate Dementia":  { level: 3, status: "Moderate stage" },
};

function render(data) {
  emptyState.classList.add("hidden");

  document.getElementById("prediction").textContent = data.prediction;
  document.getElementById("confidence").textContent =
    (data.confidence * 100).toFixed(1) + "%";

  const sev = SEVERITY[data.prediction] || { level: 0, status: "" };
  const verdict = document.getElementById("verdict");
  verdict.className = "verdict sev-" + sev.level;
  document.getElementById("sev-tag").textContent = sev.status;

  const bars = document.getElementById("bars");
  bars.innerHTML = "";
  data.probabilities.forEach((p, i) => {
    const pct = (p.probability * 100).toFixed(1);
    const lvl = (SEVERITY[p.label] || { level: 0 }).level;
    const row = document.createElement("div");
    row.className = "bar-row sev-" + lvl + (i === 0 ? " is-top" : "");
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
