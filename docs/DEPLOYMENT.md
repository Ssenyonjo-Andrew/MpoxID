# Deployment for UVRI / district reference labs

## Minimum hardware

| Item | Minimum | Comfortable |
|------|---------|-------------|
| CPU | Dual-core x86_64, ~2 GHz | Quad-core |
| RAM | 4 GB | 8 GB |
| Disk | 2 GB free | 5 GB free |
| GPU | **Not required** | — |
| OS | Windows 10/11 or Ubuntu 20.04+ | — |
| Network | None at inference time | Optional for first-time package copy |

Single-genome inference target: **well under 1 second** on the minimum laptop after features are computed (12–200 kb sequences). Batch of ~100 genomes: **seconds**, not minutes, for classical models.

## Install options when the lab PC has no admin and no internet

### Option A — Pre-built virtual environment on USB (recommended)

On a prep machine **with** internet and the **same OS + Python minor version** as the lab PC:

```bash
python -m venv mpox_venv
mpox_venv\Scripts\activate          # Windows
pip install -r requirements.txt
python scripts/generate_demo_data.py   # or train on real data
python scripts/train_all.py
```

Copy to USB:

- `mpox_venv/` (the whole virtualenv)
- project folder `mega_mpox/` including `models/`

On the lab PC:

1. Copy folder to `C:\Tools\mega_mpox` (or home directory — no admin needed).
2. Double-click a helper script, or open cmd:

```bat
C:\Tools\mega_mpox\mpox_venv\Scripts\activate.bat
cd C:\Tools\mega_mpox
streamlit run app\streamlit_app.py
```

If Windows blocks `python.exe` from USB, copy to local disk first.

### Option B — Wheelhouse (offline pip)

On the prep machine:

```bash
mkdir wheelhouse
pip download -r requirements.txt -d wheelhouse
```

On the lab PC (Python already installed):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --no-index --find-links=wheelhouse -r requirements.txt
```

### Option C — Packaged executable (advanced)

Use PyInstaller/Briefcase on the prep machine to build a folder-based app that embeds Python + Streamlit. Keep `models/` beside the executable. This needs a one-time packaging expert; Option A is simpler for most institutes.

## Updating the model when new labeled genomes arrive

1. Add new FASTA files under `data/raw/`.
2. Append rows to `data/metadata/metadata.csv` (`accession,clade,...`).
3. On the prep/training PC: `python scripts/train_all.py`.
4. Check `reports/evaluation_report.md` — confirm **Ia/Ib recall** did not collapse.
5. Copy only these files to lab PCs (USB/email):
   - `models/deploy_bundle.joblib`
   - `models/training_meta.json`
   - `models/model_comparison.json`
6. Restart Streamlit (or it will pick up the new bundle after cache clear / app restart).

No internet is required on the lab PC for updates — only a file copy.

## Troubleshooting (non-technical)

| Symptom | What to try |
|---------|-------------|
| Browser does not open | In the terminal window, hold Ctrl and click the `localhost` link, or open Chrome to `http://localhost:8501` |
| “Model not found” | Ensure `models/deploy_bundle.joblib` exists; copy `models/` from USB |
| “No sequences found” | Confirm the file ends in `.fasta` / `.fa` and opens in Notepad showing `>` headers |
| Quality always Poor | Short contigs or many Ns — re-assemble or check the sample; thresholds live in `config/default_config.yaml` |
| Antivirus quarantines Python | Ask IT to whitelist the project folder; or run from `C:\Tools\...` not Desktop sync folders |
| Very slow first run | Normal while Windows Defender scans new files; later runs are faster |
| Wrong clade vs Nextclade | Retrain with more local genomes; treat Poor-quality calls as provisional |

## Security / governance notes

- The app makes **no outbound network calls** during prediction.
- Do not place identifiable patient names in FASTA headers if local policy forbids it; use lab accession IDs.
- Keep a written log of model version (`training_meta.json` → `trained_at`, `deploy_model`) with each surveillance report.
