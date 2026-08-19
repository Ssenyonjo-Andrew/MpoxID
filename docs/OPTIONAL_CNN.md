# Optional 1D CNN — comparison only (NOT deployed by default)

A lightweight 1D CNN over one-hot DNA (or k-mer tokens) can be trained for research
comparison. It is **not** the primary deployed model in this project.

## Why XGBoost stays the default

| Criterion | XGBoost (deployed) | 1D CNN (optional) |
|-----------|--------------------|-------------------|
| CPU latency / genome | Typically ≪ 1 s including features | Often slower on CPU for ~200 kb inputs |
| Artifact size | Usually &lt; 10 MB with extractor | Can exceed tens of MB (weights) |
| Dependencies | scikit-learn + xgboost | TensorFlow or PyTorch (+GPU temptation) |
| Offline lab ops | Simple joblib copy | Heavier runtime |
| Tabular k-mer/codon features | Excellent fit | Overkill unless raw-seq signal dominates |

**Promotion rule:** only replace XGBoost if the CNN shows clearly higher **macro-F1**
(especially Ia/Ib recall) on a held-out real-genome set **and** stays under the
100 MB / sub-second CPU budget.

## Suggested comparison protocol

1. Freeze the same train/test splits used for classical models.
2. Train a small Conv1D (e.g. 2–3 layers, &lt; 500k parameters) on one-hot windows or
   on the same k-mer vectors (as an MLP/CNN hybrid).
3. Report accuracy, macro-F1, per-class recall, size on disk, and ms/genome on CPU.
4. Document results in `reports/evaluation_report.md` under “Optional CNN”.

TensorFlow is listed as commented-out in `requirements.txt` to keep the default
install lightweight for district labs.
