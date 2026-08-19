"""
Canonical k-mer frequency features.

Design
------
For DNA, k-mer X and its reverse complement RC(X) are biologically equivalent
on an unstranded or randomly-oriented assembly. We keep only the
lexicographically smaller of (X, RC(X)) — "canonical" k-mers — which roughly
halves the feature space and removes strand artifacts.

Dimensionality (canonical, ACGT only):
  k=2 → 10 features
  k=3 → 32 features
  k=4 → 136 features
Total ≈ 178 k-mer features — tiny, fast, USB-friendly.

Ambiguous bases (N, R, Y, ...) cause that window to be skipped so we do not
invent fake counts.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(kmer: str) -> str:
    return kmer.translate(_COMPLEMENT)[::-1]


def canonical_kmer(kmer: str) -> str:
    rc = reverse_complement(kmer)
    return kmer if kmer <= rc else rc


def build_canonical_vocab(k: int) -> List[str]:
    """All canonical ACGT k-mers of length k, sorted for stable feature order."""
    seen = set()
    for tup in product("ACGT", repeat=k):
        seen.add(canonical_kmer("".join(tup)))
    return sorted(seen)


def count_canonical_kmers(sequence: str, k: int, vocab: Sequence[str]) -> Dict[str, float]:
    """
    Return frequency (count / n_valid_windows) for each vocab k-mer.

    Frequencies (not raw counts) make features comparable across sequence lengths.
    """
    seq = sequence.upper()
    counts = {km: 0 for km in vocab}
    valid = 0
    n = len(seq)
    if n < k:
        return {f"kmer{k}_{km}": 0.0 for km in vocab}

    for i in range(n - k + 1):
        window = seq[i : i + k]
        if any(b not in "ACGT" for b in window):
            continue
        counts[canonical_kmer(window)] += 1
        valid += 1

    denom = float(valid) if valid else 1.0
    return {f"kmer{k}_{km}": counts[km] / denom for km in vocab}


def top_kmers_for_sequence(
    sequence: str,
    k: int,
    vocab: Sequence[str],
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """Return the top_n most frequent canonical k-mers in this sequence."""
    freqs = count_canonical_kmers(sequence, k, vocab)
    items = [(name.replace(f"kmer{k}_", ""), val) for name, val in freqs.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:top_n]
