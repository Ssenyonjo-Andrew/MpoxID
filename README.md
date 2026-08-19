# Mpox Clade Classifier

Offline, CPU-only tool that predicts **Monkeypox virus clade** (Ia / Ib / IIa / IIb) from FASTA genomes and reports **sequence quality** (Ns, ambiguity, stop codons, frameshift indicators). Built for national and district virus labs with limited internet and no GPU.

---

## What you get

| Output | Source |
|--------|--------|
| Predicted clade + confidence | Classical ML (XGBoost by default) |
| Good / Fair / Poor quality flag | Transparent rules (not learned) |
| N%, non-ACGTN count, stops, frameshifts, GC%, length, N50 | Deterministic sequence stats |
| Top discriminative k-mers | Training centroids + per-sequence k=4 |

**Hard constraints met:** no GPU, offline after training, model artifacts typically ≪ 100 MB, Streamlit UI (upload → table → CSV), inference ≪ 1 s/genome on a laptop.

---

## Folder structure

```
mega_mpox/
├── app/streamlit_app.py          # Lab technician UI (zero CLI after launch)
├── config/default_config.yaml    # Tunable quality thresholds + training options
├── data/
│   ├── raw/                      # Input FASTA genomes
│   ├── processed/                # Feature matrices after training
│   └── metadata/metadata.csv     # accession → clade labels
├── docs/                         # Data sourcing, deployment, optional CNN
├── examples/                     # Small sample FASTA files
├── models/                       # joblib bundles (deploy_bundle.joblib)
├── reports/                      # Evaluation markdown + figures
├── scripts/                      # Demo data, train, CLI predict
├── src/mpox_clf/                 # Python package
│   ├── preprocessing/            # FASTA I/O + metadata join
│   ├── features/                 # k-mers, codon usage, quality
│   ├── training/                 # LR / RF / XGBoost + evaluation
│   ├── inference/                # predict() entry point
│   └── utils/
└── tests/
```

---

## Quick start (lab laptop with internet — first install)

1. Install **Python 3.10** (3.9–3.11 OK).
2. Open a terminal in this folder and run:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Generate demo data, train models, open the app:

```bash
python scripts/generate_demo_data.py
python scripts/train_all.py
streamlit run app/streamlit_app.py
```

4. In the browser: upload a `.fasta` file → read the results table → **Download results as CSV**.

> **Important:** Demo sequences are synthetic. For real public-health use, follow [docs/DATA_SOURCING.md](docs/DATA_SOURCING.md) and retrain.

---

## Production training (real genomes)

1. Download MPXV FASTA from NCBI Virus (taxid 10244) into `data/raw/`.
2. Assign clades with Nextclade (offline dataset) or trusted metadata.
3. Create `data/metadata/metadata.csv`:

```csv
accession,clade,source,date
NC_003310,Ia,ncbi,2002-01-01
...
```

4. Run `python scripts/train_all.py`.
5. Copy the entire `models/` folder to lab machines (USB/email — usually a few MB).

---

## Using the Python API

```python
from mpox_clf.inference import predict

df = predict("path/to/genome.fasta")       # or a folder of FASTA files
print(df[["sequence_id", "predicted_clade", "confidence", "quality_flag"]])
```

---

## Class imbalance (Ia / Ib / IIa / IIb)

Public repositories are skewed toward **Clade IIb**. This project uses:

- **Stratified train/test and stratified k-fold CV**
- **Balanced class / sample weights** (LR, RF, XGBoost)
- **Macro-F1 + per-class recall** for model selection (not accuracy alone)

Optional random oversampling is available in `src/mpox_clf/training/imbalance.py` for folds where a clade has very few genomes — see comments there for tradeoffs.

---

## Offline / no-admin install

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for USB virtualenv packaging, hardware minimums, model updates, and troubleshooting for non-technical users.

---

## Tests

```bash
python -m pytest tests/ -q
```

---

## Global Mpox Surveillance & Geo-Intelligence Hub

An interactive epidemiological dashboard powered by **Our World in Data (OWID)** (`owid-monkeypox-data.csv`) is integrated directly into the suite:

| Capability | Features |
|---|---|
| **Interactive Spatial Maps** | Choropleth density, proportional bubble scatter maps, 3D orthographic globe, and animated timeline playback (2022–Present). |
| **Rich Area Hover Cards** | Instant country popover with confirmed cases, fatalities, Case Fatality Rate (CFR %), 7-day smoothed incidence, cases/million, and epidemic risk tier. |
| **Clade Linkage & Epidemiology** | Mapping of Clade I (Ia / Ib) Central/East African epicenters vs Clade IIa West Africa and Clade IIb multi-country distribution. |
| **Epidemic Curves & Trajectory** | Dual-axis country drilldowns with daily cases and 7-day moving averages, plus multi-country overlay comparisons. |
| **Offline Resilience** | Automatically caches `data/raw/owid-monkeypox-data.csv` for 100% offline air-gapped lab operations, with one-click live refresh. |

### Launching the Dashboard

```bash
# Launch unified suite (toggle between Genomic Classifier and Geo Surveillance in sidebar):
streamlit run app/streamlit_app.py

# Or launch standalone Geo Surveillance hub directly:
streamlit run app/geo_dashboard.py
```

---

## License / disclaimer

Research / public-health surveillance support tool. Clade calls on **Poor**-flagged assemblies should be confirmed with established methods (e.g. Nextclade) before operational decisions. Not a medical device.

