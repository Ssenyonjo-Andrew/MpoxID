"""
Codon-usage features from the best available reading frame.

We do not ship gene annotations (offline constraint). Instead we choose the
frame (0/1/2) with the fewest unambiguous stop codons and compute raw codon
frequencies over unambiguous ACGT codons in that frame. Relative synonymous
codon usage (RSCU) is also provided for the 59 sense synonymous codons.

These features capture clade-associated codon bias without requiring BLAST
or online CDS tables.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .quality import GENETIC_CODE, STOP_CODONS, _best_frame_stop_count

# All 64 DNA codons in lexical order for stable columns
ALL_CODONS: List[str] = [
    a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
]

# Synonymous groups for RSCU (amino acid → list of codons), excluding Met/Trp/stops
_AA_TO_CODONS: Dict[str, List[str]] = defaultdict(list)
for codon, aa in GENETIC_CODE.items():
    if aa != "*":
        _AA_TO_CODONS[aa].append(codon)


def _codon_counts_in_frame(sequence: str, frame: int) -> Dict[str, int]:
    counts = {c: 0 for c in ALL_CODONS}
    s = sequence.upper()[frame:]
    for i in range(0, len(s) - 2, 3):
        codon = s[i : i + 3]
        if codon in counts:
            counts[codon] += 1
    return counts


def compute_codon_features(sequence: str) -> Dict[str, float]:
    """
    Return dict of:
      - codon_<XYZ>: frequency among unambiguous sense+stop codons in best frame
      - rscu_<XYZ>: RSCU for synonymous sense codons (0 if unused AA)
    """
    info = _best_frame_stop_count(sequence)
    frame = int(info["best_frame"])
    counts = _codon_counts_in_frame(sequence, frame)
    total = sum(counts.values()) or 1
    feats: Dict[str, float] = {
        f"codon_{c}": counts[c] / total for c in ALL_CODONS
    }

    # RSCU: for each AA with degeneracy d, RSCU_i = obs_i / (mean obs for that AA)
    for aa, codons in _AA_TO_CODONS.items():
        obs = [counts[c] for c in codons]
        s = sum(obs)
        d = len(codons)
        if s == 0:
            for c in codons:
                feats[f"rscu_{c}"] = 0.0
        else:
            expected = s / d
            for c in codons:
                feats[f"rscu_{c}"] = (counts[c] / expected) if expected else 0.0

    feats["codon_best_frame"] = float(frame)
    return feats


def discriminative_codon_hint(clade_centroid: Dict[str, float], top_n: int = 5) -> List[Tuple[str, float]]:
    """Helper for reports: highest-weight codon frequencies from a clade mean vector."""
    items = [(k, v) for k, v in clade_centroid.items() if k.startswith("codon_")]
    items.sort(key=lambda x: -x[1])
    return items[:top_n]
