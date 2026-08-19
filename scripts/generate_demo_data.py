"""
Generate synthetic labeled MPXV-like sequences for end-to-end pipeline testing.

REAL DATA REQUIRED FOR PRODUCTION
---------------------------------
These sequences are NOT real Monkeypox genomes. They embed clade-specific
k-mer biases so classical models can learn a separable signal for CI/demo.
For deployment at UVRI / district labs, replace with real NCBI genomes +
Nextclade clade labels (see docs/DATA_SOURCING.md).

Usage:
    python scripts/generate_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpox_clf.preprocessing.fasta_io import SequenceRecord, write_fasta  # noqa: E402

# Motifs enriched per clade (toy signal — replace with real data for production)
CLADE_MOTIFS = {
    "Ia": ["ACGTAC", "TTGACA", "GGCCTA", "AAATTT"],
    "Ib": ["GCTAGC", "CCTAGG", "TATATA", "CGCGCG"],
    "IIa": ["AGTCAG", "TCAGTC", "GATTAC", "CATCAT"],
    "IIb": ["ATGGCC", "GCCATG", "TTCGAA", "AAGGTT"],
}

# Intentionally imbalanced to mirror real public data skew toward IIb
CLADE_COUNTS = {"Ia": 18, "Ib": 14, "IIa": 20, "IIb": 48}


def _random_seq(rng: np.random.Generator, length: int, gc: float = 0.33) -> str:
    """MPXV-like AT-rich background."""
    # P(G)=P(C)=gc/2, P(A)=P(T)=(1-gc)/2
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G T
    bases = rng.choice(list("ACGT"), size=length, p=probs)
    return "".join(bases.tolist())


def _implant_motifs(seq: str, motifs: list, rng: np.random.Generator, n_insert: int = 80) -> str:
    arr = list(seq)
    L = len(arr)
    for _ in range(n_insert):
        motif = motifs[int(rng.integers(0, len(motifs)))]
        pos = int(rng.integers(0, max(1, L - len(motif))))
        arr[pos : pos + len(motif)] = list(motif)
    return "".join(arr)


def _add_quality_noise(seq: str, rng: np.random.Generator, level: str) -> str:
    """Optionally inject Ns / ambiguity for QC demos."""
    arr = list(seq)
    L = len(arr)
    if level == "fair":
        for _ in range(max(1, L // 80)):
            arr[int(rng.integers(0, L))] = "N"
    elif level == "poor":
        for _ in range(max(1, L // 15)):
            arr[int(rng.integers(0, L))] = "N"
        if L > 10:
            arr[5] = "R"
            arr[6] = "Y"
    return "".join(arr)


def main() -> None:
    rng = np.random.default_rng(42)
    # Short enough for fast CI, long enough for k-mers/codons (not full 197kb)
    length = 12_000

    records = []
    meta_rows = []
    out_fa = ROOT / "data" / "raw" / "demo_genomes.fasta"
    out_meta = ROOT / "data" / "metadata" / "metadata.csv"
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    for clade, n in CLADE_COUNTS.items():
        for i in range(n):
            acc = f"DEMO_{clade}_{i+1:03d}"
            seq = _random_seq(rng, length)
            seq = _implant_motifs(seq, CLADE_MOTIFS[clade], rng, n_insert=120)
            # A few noisy examples for quality-flag demos
            if i == 0 and clade == "IIb":
                seq = _add_quality_noise(seq, rng, "poor")
            elif i == 1 and clade == "Ia":
                seq = _add_quality_noise(seq, rng, "fair")
            records.append(
                SequenceRecord(id=acc, sequence=seq, description=f"clade={clade} synthetic")
            )
            meta_rows.append({"accession": acc, "clade": clade, "source": "synthetic_demo"})

    write_fasta(records, out_fa)
    pd.DataFrame(meta_rows).to_csv(out_meta, index=False)

    # Also write a couple of standalone example FASTAs for the UI
    examples = ROOT / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    write_fasta(records[:2], examples / "sample_good.fasta")
    write_fasta([records[-1]], examples / "sample_iib.fasta")

    print(f"Wrote {len(records)} synthetic genomes -> {out_fa}")
    print(f"Wrote metadata -> {out_meta}")
    print(
        "WARNING: Replace with real NCBI + Nextclade labels before clinical/public-health use."
    )


if __name__ == "__main__":
    main()
