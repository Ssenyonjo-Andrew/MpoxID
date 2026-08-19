"""
Train all classical models from data/raw + data/metadata/metadata.csv.

Usage (from project root):
    python scripts/generate_demo_data.py   # first time / CI only
    python scripts/train_all.py

For production: point config paths at real FASTA + metadata (docs/DATA_SOURCING.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpox_clf.training import train_from_fasta_and_metadata  # noqa: E402
from mpox_clf.utils.config import load_config, project_root  # noqa: E402


def main() -> None:
    cfg = load_config()
    root = project_root()
    fasta = root / cfg["paths"]["raw_fasta"]
    real_fa = fasta / "real_mpox_genomes.fasta"
    demo = fasta / "demo_genomes.fasta"
    if real_fa.exists():
        fasta_path = real_fa
    elif demo.exists():
        fasta_path = demo
    else:
        fasta_path = fasta
    metadata = root / cfg["paths"]["metadata"]

    if not metadata.exists():
        raise SystemExit(
            f"Missing {metadata}. Run scripts/generate_demo_data.py for a demo set, "
            "or create metadata.csv (accession,clade) for real genomes."
        )
    if not Path(fasta_path).exists():
        raise SystemExit(f"Missing FASTA input: {fasta_path}")

    print(f"Training from FASTA={fasta_path}  metadata={metadata}")
    if fasta_path == demo:
        cfg.setdefault("quality", {})
        cfg["quality"]["min_length_fair"] = 8_000
        cfg["quality"]["min_length_poor"] = 3_000
        print("[train_all] Demo mode: lowered min_length quality thresholds.")
    result = train_from_fasta_and_metadata(fasta_path, metadata, config=cfg)
    print("Deploy model:", result["deploy_model"])
    for name, m in result["results"].items():
        print(f"  {name}: accuracy={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
