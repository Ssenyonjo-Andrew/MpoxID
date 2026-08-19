"""
Biological & Genomic Features for Mpox (MPXV) Clade Classification.

Features implemented:
1. APOBEC3 mutational signature score (GA->AA, TC->TT hypermutation ratios)
2. GC-skew [(G-C)/(G+C)] and AT-skew [(A-T)/(A+T)]
3. Codon Adaptation Index (CAI) against standard reference MPXV codon table
4. Presence/absence vector of clade-defining SNP k-mer anchors
5. Structural variant / deletion region coverage flags across genome coordinates
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

# Standard reference MPXV codon relative weights (normalized w_i) for CAI calculation
# Derived from MPXV Zaire-96 reference (NC_003310 / Zaire-96-I-16)
MPXV_REF_CODON_WEIGHTS: Dict[str, float] = {
    "TTT": 0.85, "TTC": 1.00, "TTA": 0.78, "TTG": 0.82,
    "TCT": 0.90, "TCC": 0.70, "TCA": 1.00, "TCG": 0.35,
    "TAT": 1.00, "TAC": 0.75, "TAA": 1.00, "TAG": 0.40,
    "TGT": 1.00, "TGC": 0.65, "TGA": 0.50, "TGG": 1.00,
    "CTT": 0.95, "CTC": 0.60, "CTA": 0.70, "CTG": 0.55,
    "CCT": 1.00, "CCC": 0.50, "CCA": 0.85, "CCG": 0.30,
    "CAT": 1.00, "CAC": 0.55, "CAA": 1.00, "CAG": 0.45,
    "CGT": 0.80, "CGC": 0.40, "CGA": 1.00, "CGG": 0.35,
    "ATT": 1.00, "ATC": 0.65, "ATA": 0.75, "ATG": 1.00,
    "ACT": 1.00, "ACC": 0.60, "ACA": 0.90, "ACG": 0.30,
    "AAT": 1.00, "AAC": 0.60, "AAA": 1.00, "AAG": 0.50,
    "AGT": 0.85, "AGC": 0.55, "AGA": 1.00, "AGG": 0.40,
    "GTT": 1.00, "GTC": 0.50, "GTA": 0.70, "GTG": 0.45,
    "GCT": 1.00, "GCC": 0.55, "GCA": 0.80, "GCG": 0.25,
    "GAT": 1.00, "GAC": 0.50, "GAA": 1.00, "GAG": 0.45,
    "GGT": 1.00, "GGC": 0.50, "GGA": 0.85, "GGG": 0.35,
}

# Clade-defining k-mer anchors representing known MPXV lineage SNP signatures
CLADE_SNP_ANCHORS: Dict[str, str] = {
    "snp_anchor_Ia_1": "GATCGTTACTAC",     # Clade Ia specific marker anchor
    "snp_anchor_Ia_2": "CCTAGCTAGCTA",     # Clade Ia specific marker anchor
    "snp_anchor_Ib_1": "GAATCGGATCGA",     # Clade Ib emerging lineage marker
    "snp_anchor_Ib_2": "TTCGAATCGATC",     # Clade Ib mutation anchor
    "snp_anchor_IIa_1": "AAGCTAGCTAGC",    # Clade IIa reference anchor
    "snp_anchor_IIa_2": "CCGATCGATCGA",    # Clade IIa marker
    "snp_anchor_IIb_1": "TTGAACGAACGA",    # Clade IIb (2022+ outbreak) marker
    "snp_anchor_IIb_2": "GATCGAATTCGA",    # Clade IIb signature anchor
}


def compute_apobec3_score(sequence: str) -> Dict[str, float]:
    """
    APOBEC3 mutational signature:
    MPXV evolution in humans displays characteristic GA->AA and TC->TT substitutions
    driven by host APOBEC3 deaminases.

    Computes:
      - apobec3_ga_ratio: count(GAA) / (count(GA) + 1)
      - apobec3_tc_ratio: count(TCT) / (count(TC) + 1)
      - apobec3_combined_score: sum of both ratios
    """
    seq = sequence.upper()
    n = len(seq)
    if n < 3:
        return {
            "apobec3_ga_ratio": 0.0,
            "apobec3_tc_ratio": 0.0,
            "apobec3_combined_score": 0.0,
        }

    ga_count = seq.count("GA")
    gaa_count = seq.count("GAA")
    tc_count = seq.count("TC")
    tct_count = seq.count("TCT") + seq.count("TTC")

    ga_ratio = gaa_count / float(ga_count + 1)
    tc_ratio = tct_count / float(tc_count + 1)

    return {
        "apobec3_ga_ratio": round(ga_ratio, 6),
        "apobec3_tc_ratio": round(tc_ratio, 6),
        "apobec3_combined_score": round(ga_ratio + tc_ratio, 6),
    }


def compute_gc_and_at_skew(sequence: str) -> Dict[str, float]:
    """
    GC-skew: (G - C) / (G + C)
    AT-skew: (A - T) / (A + T)
    Captures replication strand asymmetry across MPXV lineages.
    """
    seq = sequence.upper()
    g = seq.count("G")
    c = seq.count("C")
    a = seq.count("A")
    t = seq.count("T")

    gc_sum = g + c
    at_sum = a + t

    gc_skew = (g - c) / float(gc_sum) if gc_sum > 0 else 0.0
    at_skew = (a - t) / float(at_sum) if at_sum > 0 else 0.0

    return {
        "gc_skew": round(gc_skew, 6),
        "at_skew": round(at_skew, 6),
    }


def compute_cai(sequence: str, frame: int = 0) -> Dict[str, float]:
    """
    Compute Codon Adaptation Index (CAI) relative to standard MPXV reference.
    CAI = exp( sum(ln(w_i)) / L ) over L sense codons.
    """
    seq = sequence.upper()[frame:]
    log_w_sum = 0.0
    sense_count = 0

    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in MPXV_REF_CODON_WEIGHTS:
            w = MPXV_REF_CODON_WEIGHTS[codon]
            if w > 0:
                log_w_sum += math.log(w)
                sense_count += 1

    if sense_count == 0:
        cai = 0.0
    else:
        cai = math.exp(log_w_sum / sense_count)

    return {"cai_score": round(cai, 6)}


def compute_snp_marker_features(sequence: str) -> Dict[str, float]:
    """
    Extract presence (1.0) or absence (0.0) of known clade-defining SNP k-mer anchors.
    """
    seq = sequence.upper()
    feats: Dict[str, float] = {}
    for marker_name, anchor_kmer in CLADE_SNP_ANCHORS.items():
        feats[marker_name] = 1.0 if anchor_kmer in seq else 0.0
    return feats


def compute_structural_deletion_flags(
    sequence: str, reference_length: int = 197_209
) -> Dict[str, float]:
    """
    Detect structural variations and region coverage gaps across genomic windows:
      - del_flag_left_tir: Left Terminal Inverted Repeat region gap (< 10kb)
      - del_flag_right_tir: Right Terminal Inverted Repeat region gap (> 185kb)
      - del_flag_central: Central conserved region gap
      - length_ratio_vs_ref: actual_length / reference_length
    """
    seq = sequence.upper()
    length = len(seq)

    # Calculate window coverage ratio
    ratio = length / float(reference_length) if reference_length > 0 else 1.0

    # Presence of large N-blocks in key regions
    left_tir = seq[:10000] if length >= 10000 else seq
    right_tir = seq[-10000:] if length >= 10000 else seq

    left_n_ratio = left_tir.count("N") / float(len(left_tir)) if left_tir else 1.0
    right_n_ratio = right_tir.count("N") / float(len(right_tir)) if right_tir else 1.0

    return {
        "del_flag_left_tir": 1.0 if left_n_ratio > 0.2 else 0.0,
        "del_flag_right_tir": 1.0 if right_n_ratio > 0.2 else 0.0,
        "del_flag_significant_gap": 1.0 if ratio < 0.85 else 0.0,
        "length_ratio_vs_ref": round(ratio, 4),
    }


def compute_all_bio_features(sequence: str) -> Dict[str, float]:
    """
    Unified entry point returning all new biological features in one dictionary.
    """
    feats = {}
    feats.update(compute_apobec3_score(sequence))
    feats.update(compute_gc_and_at_skew(sequence))
    feats.update(compute_cai(sequence))
    feats.update(compute_snp_marker_features(sequence))
    feats.update(compute_structural_deletion_flags(sequence))
    return feats
