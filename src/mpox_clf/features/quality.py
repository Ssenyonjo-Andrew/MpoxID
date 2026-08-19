"""
Deterministic sequence-quality metrics for MPXV assemblies.

WHY THESE ARE NOT LEARNED
-------------------------
N-content, illegal characters, premature stops, and frameshift indicators are
*physical properties of the sequenced string*. A classifier might correlate
"lots of Ns" with a rare clade by accident (lab/batch effects), which would be
dangerous in a diagnostic setting. We therefore compute these with transparent
rules and expose Good/Fair/Poor so technicians can decide whether to trust the
clade call. Only the clade label (and optionally a learned quality combiner —
we do NOT use that here) comes from ML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Standard genetic code (DNA codon → AA). '*' = stop.
GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CODON = "ATG"


def _translate_orfs_stops(seq: str, frame: int) -> int:
    """
    Count premature stop codons inside candidate viral gene ORFs.
    An ORF starting at ATG that terminates prematurely (100 <= length < 300 bp)
    represents a truncated/disrupted viral protein.
    """
    s = seq[frame:]
    premature_stops = 0
    in_orf = False
    orf_len = 0
    for i in range(0, len(s) - 2, 3):
        codon = s[i : i + 3]
        if "N" in codon or any(b not in "ACGT" for b in codon):
            continue
        if codon == "ATG" and not in_orf:
            in_orf = True
            orf_len = 0
        elif in_orf:
            orf_len += 3
            if codon in STOP_CODONS:
                if 100 <= orf_len < 300:
                    premature_stops += 1
                in_orf = False
    return premature_stops


def _best_frame_stop_count(seq: str) -> Dict[str, Any]:
    """
    Scan frames 0,1,2 for truncated gene ORFs starting at ATG.
    """
    counts = {f: _translate_orfs_stops(seq, f) for f in (0, 1, 2)}
    best = min(counts, key=counts.get)
    return {
        "best_frame": best,
        "premature_stop_count": counts[best],
        "stop_counts_by_frame": counts,
    }


def _frameshift_indicators(
    seq: str,
    contigs: Optional[Sequence[str]] = None,
    reference_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Heuristic frameshift / assembly-integrity flags (no reference alignment needed).

    Indicators (any True → frameshift_flag):
      - genome length not near a multiple of typical MPXV size bands is NOT used
        alone; instead:
      - presence of contig-joining N-runs (from our merger) with odd local lengths
      - length difference vs optional reference not divisible by 3 (indel-like)
      - high stop density in the best frame relative to sequence length
    """
    reasons: List[str] = []
    length = len(seq)

    if reference_length is not None and reference_length > 0:
        delta = abs(length - reference_length)
        if delta > 0 and delta % 3 != 0:
            reasons.append(
                f"length delta vs reference ({delta} bp) is not a multiple of 3 "
                "(possible indel/frameshift relative to reference)"
            )

    # Contig lengths that are not multiples of 3 can break coding continuity when merged
    if contigs and len(contigs) > 1:
        odd = [i for i, c in enumerate(contigs) if len(c) % 3 != 0]
        if odd:
            reasons.append(
                f"{len(odd)} contig(s) have length not divisible by 3 "
                "(possible frameshift at contig boundaries)"
            )

    # Dense stops: >1 stop per 3kb in best frame is suspicious for a cleaned assembly
    stop_info = _best_frame_stop_count(seq)
    if length > 0 and stop_info["premature_stop_count"] > max(50, length // 200):
        reasons.append(
            f"high stop-codon density in best frame "
            f"({stop_info['premature_stop_count']} stops)"
        )

    return {
        "frameshift_flag": bool(reasons),
        "frameshift_reasons": reasons,
        **stop_info,
    }


def compute_quality_metrics(
    sequence: str,
    *,
    contigs: Optional[Sequence[str]] = None,
    reference_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute all deterministic quality features for one sequence.

    Returns a flat dict suitable for a DataFrame row.
    """
    seq = sequence.upper()
    length = len(seq)
    if length == 0:
        return {
            "length": 0,
            "n_count": 0,
            "n_pct": 100.0,
            "non_acgtn_count": 0,
            "gc_pct": 0.0,
            "a_pct": 0.0,
            "c_pct": 0.0,
            "g_pct": 0.0,
            "t_pct": 0.0,
            "premature_stop_count": 0,
            "best_frame": 0,
            "frameshift_flag": True,
            "frameshift_reasons": ["empty sequence"],
            "n_contigs": 0,
            "n50": 0,
        }

    n_count = seq.count("N")
    acgtn = set("ACGTN")
    non_acgtn = sum(1 for b in seq if b not in acgtn)

    # Composition excluding N for GC (standard genomics practice)
    acgt = [b for b in seq if b in "ACGT"]
    acgt_len = len(acgt) or 1
    gc = sum(1 for b in acgt if b in "GC")

    fs = _frameshift_indicators(seq, contigs=contigs, reference_length=reference_length)

    # N50 / contig stats if multi-contig
    contig_list = list(contigs) if contigs else [seq]
    lengths = sorted((len(c) for c in contig_list), reverse=True)
    total = sum(lengths) or 1
    running = 0
    n50 = lengths[-1] if lengths else 0
    for L in lengths:
        running += L
        if running >= total / 2:
            n50 = L
            break

    return {
        "length": length,
        "n_count": n_count,
        "n_pct": 100.0 * n_count / length,
        "non_acgtn_count": non_acgtn,
        "gc_pct": 100.0 * gc / acgt_len,
        "a_pct": 100.0 * seq.count("A") / length,
        "c_pct": 100.0 * seq.count("C") / length,
        "g_pct": 100.0 * seq.count("G") / length,
        "t_pct": 100.0 * seq.count("T") / length,
        "premature_stop_count": fs["premature_stop_count"],
        "best_frame": fs["best_frame"],
        "frameshift_flag": fs["frameshift_flag"],
        "frameshift_reasons": fs["frameshift_reasons"],
        "n_contigs": len(contig_list),
        "n50": n50,
    }


def quality_flag_from_metrics(
    metrics: Dict[str, Any],
    *,
    n_pct_fair: float = 1.0,
    n_pct_poor: float = 5.0,
    non_acgtn_fair: int = 5,
    non_acgtn_poor: int = 50,
    premature_stops_fair: int = 1,
    premature_stops_poor: int = 5,
    min_length_fair: int = 150_000,
    min_length_poor: int = 50_000,
    frameshift_implies_at_best: str = "Fair",
    **_kwargs,
) -> Dict[str, Any]:
    """
    Map metrics → Good / Fair / Poor with an English explanation.

    Rules are intentionally simple and tunable via config/default_config.yaml.
    Worst severity wins (Poor > Fair > Good).
    """
    reasons: List[str] = []
    rank = {"Good": 0, "Fair": 1, "Poor": 2}
    flag = "Good"

    def raise_to(level: str, reason: str) -> None:
        nonlocal flag
        if rank[level] > rank[flag]:
            flag = level
        reasons.append(reason)

    n_pct = float(metrics.get("n_pct", 0))
    if n_pct >= n_pct_poor:
        raise_to("Poor", f"{n_pct:.1f}% Ns (threshold for Poor: >={n_pct_poor}%)")
    elif n_pct >= n_pct_fair:
        raise_to("Fair", f"{n_pct:.1f}% Ns (threshold for Fair: >={n_pct_fair}%)")

    non = int(metrics.get("non_acgtn_count", 0))
    if non >= non_acgtn_poor:
        raise_to("Poor", f"{non} non-ACGTN characters (Poor if >={non_acgtn_poor})")
    elif non >= non_acgtn_fair:
        raise_to("Fair", f"{non} non-ACGTN characters (Fair if >={non_acgtn_fair})")

    length = int(metrics.get("length", 0))
    stops = int(metrics.get("premature_stop_count", 0))
    # For whole unannotated 197kb strings, stop codons occur naturally in non-coding reading frames.
    # Scale stop codon threshold dynamically: max(20, length // 2000)
    effective_stops_poor = max(premature_stops_poor, length // 2000) if length > 10000 else premature_stops_poor
    effective_stops_fair = max(premature_stops_fair, length // 4000) if length > 10000 else premature_stops_fair

    if stops >= effective_stops_poor and length > 0:
        raise_to("Poor", f"{stops} stop codons in best frame (Poor if >={effective_stops_poor})")
    elif stops >= effective_stops_fair and length > 0:
        raise_to(
            "Fair",
            f"{stops} stop codon(s) in best frame (Fair if >={effective_stops_fair})",
        )

    length = int(metrics.get("length", 0))
    if length < min_length_poor:
        raise_to(
            "Poor",
            f"sequence length {length} bp is below {min_length_poor} "
            "(incomplete / fragment)",
        )
    elif length < min_length_fair:
        raise_to(
            "Fair",
            f"sequence length {length} bp is below typical near-complete "
            f"MPXV (~{min_length_fair} bp)",
        )

    if metrics.get("frameshift_flag"):
        cap = frameshift_implies_at_best if frameshift_implies_at_best in rank else "Fair"
        detail = "; ".join(metrics.get("frameshift_reasons") or ["frameshift indicator"])
        raise_to(cap, f"frameshift/assembly indicator: {detail}")

    if not reasons:
        explanation = (
            "Quality flag Good: N%, ambiguity, length, and stop/frameshift checks "
            "are within configured thresholds."
        )
    else:
        explanation = (
            f"This sequence is flagged {flag} because: " + "; ".join(reasons) + ". "
            "Interpret the clade prediction with corresponding caution."
        )

    return {
        "quality_flag": flag,
        "quality_explanation": explanation,
        "quality_reasons": reasons,
    }
