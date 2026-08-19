"""
Quick CLI inference (optional — technicians should use Streamlit).

    python scripts/predict_cli.py examples/sample_iib.fasta
    python scripts/predict_cli.py data/raw/ --out results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpox_clf.inference import predict  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Mpox clade prediction (offline)")
    p.add_argument("fasta", help="FASTA file or directory")
    p.add_argument("--out", default=None, help="Optional CSV output path")
    p.add_argument(
        "--bundle",
        default=None,
        help="Path to deploy_bundle.joblib (default: models/deploy_bundle.joblib)",
    )
    args = p.parse_args()
    df = predict(args.fasta, bundle_path=args.bundle)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out} ({len(df)} rows)")
    else:
        cols = [
            c
            for c in [
                "sequence_id",
                "predicted_clade",
                "confidence",
                "quality_flag",
                "n_pct",
                "length",
            ]
            if c in df.columns
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
