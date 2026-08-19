"""
Config-driven Retraining Trigger Check Script.

Checks if new FASTA sequences in data/raw/ or active learning corrections in data/corrections/
exceed the configured threshold since the last model build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpox_clf.utils.config import load_config, project_root


def check_retrain_needed(threshold_new_genomes: int = 10) -> bool:
    root = project_root()
    meta_path = root / "models" / "training_meta.json"
    raw_dir = root / "data" / "raw"
    corrections_path = root / "data" / "corrections" / "active_learning_log.csv"

    if not meta_path.exists():
        print("[retrain_check] No existing model meta found. Retraining REQUIRED.")
        return True

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    last_count = meta.get("n_sequences", 0)

    # Count current FASTA files
    fasta_files = list(raw_dir.glob("*.fasta")) + list(raw_dir.glob("*.fa"))
    current_raw_count = len(fasta_files)

    # Count correction overrides
    correction_count = 0
    if corrections_path.exists():
        correction_count = sum(1 for line in corrections_path.read_text(encoding="utf-8").splitlines() if line.strip()) - 1
        correction_count = max(0, correction_count)

    new_items = max(0, current_raw_count - last_count) + correction_count

    print(f"[retrain_check] Last trained sequence count: {last_count}")
    print(f"[retrain_check] Current raw genomes: {current_raw_count}, Lab corrections logged: {correction_count}")

    if new_items >= threshold_new_genomes:
        print(f"[retrain_check] Retraining RECOMMENDED ({new_items} new items >= threshold {threshold_new_genomes}).")
        return True
    else:
        print(f"[retrain_check] Model is up to date ({new_items} new items < threshold {threshold_new_genomes}).")
        return False


if __name__ == "__main__":
    retrain = check_retrain_needed()
    sys.exit(0 if not retrain else 1)
