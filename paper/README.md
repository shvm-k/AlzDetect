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

### 🔴 Must fix (or you get rejected)
- [ ] **One set of numbers.** Your three sources disagree:
  abstract 56→92%, results table 63→86%, GitHub README 94.2→96.1%.
  Do **one** clean run and make every number match it.
- [ ] **Baseline comparison table** (Table II): fuzzy vs. no-balancing, class
  weights, random oversampling, SMOTE, focal loss — on identical folds.
  This is the experiment that justifies the whole paper.
- [ ] **Leakage check.** Split by *patient*, not by image, or explicitly disclose
  the limitation. The Kaggle AD set is known for train/test leakage.
- [ ] **Removed overclaims** — already done in this draft. Do **not** add back
  "attention gates", "learned/dynamic membership functions", or "image-quality
  confidence scoring": the code does none of these.

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
