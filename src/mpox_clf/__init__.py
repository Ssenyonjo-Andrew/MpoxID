"""
Mpox (MPXV) clade classifier for low-resource laboratories.

Primary task: predict clade Ia / Ib / IIa / IIb from FASTA genomes (CPU, offline).
Also reports deterministic sequence-quality metrics (Ns, ambiguity, stops, frameshifts).

Package layout
--------------
preprocessing  — FASTA I/O and training-data assembly
features       — k-mers, composition, codon usage, quality stats
training       — classical ML (LR, RF, XGBoost) + evaluation
inference      — single predict() entry point for batch/single FASTA
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
