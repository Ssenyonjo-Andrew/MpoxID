# Sourcing and labeling Mpox training data

Real accession-linked genomes are **not** shipped in this repository (size + redistribution
limits). Training and demo CI use `scripts/generate_demo_data.py` (synthetic). Replace with
the workflow below before any lab deployment.

## 1. Download genomes (NCBI Virus)

1. Open [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/) and search **Monkeypox virus** (taxid **10244**).
2. Filter to complete or near-complete genomes if available.
3. Download nucleotide FASTA into `data/raw/` (single multi-FASTA or many files).
4. Keep a download manifest (accessions) for audit.

Alternative CLI (when internet is available on a prep machine):

```bash
# Example using NCBI datasets CLI (install separately on the prep machine)
datasets download virus genome taxon 10244 --filename mpxv.zip
```

## 2. Assign clade labels (Nextclade)

Nextclade maintains MPXV clade definitions aligned with WHO nomenclature (Ia, Ib, IIa, IIb).

**Recommended for air-gapped labs:**

1. On an internet-connected machine, download the Nextclade MPXV dataset once.
2. Copy the dataset + Nextclade CLI binary to USB.
3. Run offline:

```bash
nextclade run --input-dataset mpxv_dataset --output-csv nextclade_out.csv data/raw/*.fasta
```

4. Map Nextclade `clade` column → our labels (`Ia`, `Ib`, `IIa`, `IIb`).

GISAID may also provide clade annotations where your institute has access; respect GISAID
sharing terms and do **not** commit restricted sequences to public git remotes.

## 3. Build `metadata.csv`

Create `data/metadata/metadata.csv`:

```csv
accession,clade,source,date,country
NC_003310,Ia,ncbi+nextclade,2002-01-01,USA
MK783028,IIa,ncbi+nextclade,2018-01-01,
ON563414,IIb,ncbi+nextclade,2022-05-01,
```

Rules enforced by `prepare_training_data.py`:

- Required columns: `accession`, `clade`
- Clade must be one of: `Ia`, `Ib`, `IIa`, `IIb` (aliases like `1a` / `2b` are normalized)
- Duplicate accessions → first row kept
- FASTA headers are matched to accession (exact, version-stripped, or substring)

## 4. Class imbalance — what to expect

| Clade | Typical public-data abundance | Risk if ignored |
|-------|-------------------------------|-----------------|
| IIb   | Very high (2022+ outbreak)    | Dominates accuracy |
| IIa   | Moderate                      | OK |
| Ia    | Low                           | Poor recall |
| Ib    | Low / emerging interest       | Poor recall |

Mitigations implemented in training:

1. **Stratified splits / CV** — every fold contains all clades when counts allow.
2. **Class / sample weights** — minority clades contribute more to the loss (default).
3. **Optional oversampling** — `random_oversample()` inside a training fold only; use when a clade has ≪ 30 genomes.

**Tradeoffs:** weighting preserves real genomes and avoids duplicate-row leakage; oversampling can help tiny classes but must stay inside CV folds. Never judge models by accuracy alone — use **macro-F1** and **Ia/Ib recall**.

## 5. Retrain

```bash
python scripts/train_all.py
```

Copy updated `models/deploy_bundle.joblib` (+ `training_meta.json`, `model_comparison.json`) to lab PCs.
