# AlzDetect — Journal Manuscript

`main.tex` is a journal-ready draft (IEEEtran) built from your capstone, written
to match what the code **actually does** and to survive peer review.

## How to compile
1. Go to [overleaf.com](https://overleaf.com) → **New Project → Upload Project** →
   upload this `paper/` folder (or just `main.tex`).
2. Menu → Compiler: **pdfLaTeX**, Main document: **main.tex** → Recompile.
3. IEEEtran is built into Overleaf, so it compiles out of the box.

## Before you submit — the TODO checklist
Search the `.tex` for `TODO`. Each one is a blocker. Grouped by priority:

### ✅ Done (numbers now sourced from your real notebooks)
- Numbers reconciled to your **actual runs**: baseline **56%** (macro-F1 0.25),
  fuzzy **92%** (macro-F1 0.93); moderate-dementia recall 0% → 92%. The abstract's
  56→92% pair was the correct one; the 63→86% and 94.2→96.1% figures were dropped.
- Overclaims removed (no attention gates / learned membership functions).
- A **Threats to Validity** section now honestly states the confounds.

### 🔴 Still must address for a strong paper
- [ ] **The confound is the #1 issue.** Your baseline and fuzzy runs differ in
  dataset size, test distribution, AND architecture/loss — not just fuzzy vs.
  no-fuzzy. The current draft discloses this honestly, but the *strong* version
  needs the controlled experiment (`experiments/run_experiments.py`): identical
  architecture + data, only the balancing varies, vs. class weights / SMOTE /
  focal loss, under k-fold CV. This fills Table II and turns the paper from
  "suggestive" into "convincing".
- [ ] **Leakage check.** Split by *patient*, not by image, or keep the disclosure.

### 🟡 Strongly recommended
- [ ] 5-fold cross-validation, report mean ± std (not a single split).
- [ ] Confusion matrices + per-class precision/recall/F1 figures.
- [ ] Exact dataset citation + per-class sample counts.
- [ ] Architecture figure (`arch.png`) and training-curve figures.
- [ ] Fill the 10 references with full author/venue/year/DOI.

### 🟢 Polish
- [ ] Author order, emails, corresponding author, supervisor.
- [ ] Membership-function sensitivity analysis (nice-to-have, impresses reviewers).

## Suggested target venues (avoid predatory ones!)
- **IEEE Access** — broad, rigorous, fast; needs the baseline + CV work above.
- **SN Computer Science** (Springer) — good fit for applied ML.
- A **Scopus-indexed IEEE/Springer conference** is the safer first target if you
  want a faster, lower-risk acceptance, then extend to a journal.
- ⚠️ Ignore journals/conferences that email you offering "fast publication for a
  fee" — many are predatory and will hurt your record. Check the venue is in
  Scopus/Web of Science and not on Beall's list.

## How I can help next
- Generate the exact experiment code (k-fold + SMOTE/focal/class-weight
  baselines) for your team to run on the dataset.
- Format the references properly once you give me the source details.
- Tighten any section or rewrite for a specific journal's template.
