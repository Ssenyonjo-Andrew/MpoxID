"""Unit tests — run with: python -m pytest tests/ -q"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpox_clf.features.extractor import FeatureExtractor
from mpox_clf.features.kmer import build_canonical_vocab, canonical_kmer
from mpox_clf.features.quality import compute_quality_metrics, quality_flag_from_metrics
from mpox_clf.preprocessing.fasta_io import load_fasta, normalize_sequence, write_fasta
from mpox_clf.preprocessing.fasta_io import SequenceRecord
from mpox_clf.utils.config import DEFAULT_CFG


def test_normalize_lowercase_and_u():
    assert normalize_sequence("acgu\nacg") == "ACGTACG"


def test_canonical_kmer_rc_collapse():
    assert canonical_kmer("AA") == canonical_kmer("TT")
    vocab = build_canonical_vocab(2)
    assert len(vocab) == 10


def test_quality_n_and_flag(tmp_path):
    seq = "ACGT" * 50 + "N" * 20  # ~9% N
    m = compute_quality_metrics(seq)
    assert m["n_count"] == 20
    flag = quality_flag_from_metrics(
        m, n_pct_fair=1.0, n_pct_poor=5.0, min_length_fair=10, min_length_poor=5
    )
    assert flag["quality_flag"] == "Poor"


def test_feature_extractor_shape():
    seqs = ["ACGTACGT" * 200, "TGCATGCA" * 200]
    recs = [SequenceRecord(id=f"s{i}", sequence=s) for i, s in enumerate(seqs)]
    ext = FeatureExtractor(
        kmer_sizes=[2, 3],
        include_codon_usage=True,
        quality_thresholds={
            "min_length_fair": 100,
            "min_length_poor": 50,
            "n_pct_fair": 1.0,
            "n_pct_poor": 5.0,
        },
    )
    df = ext.fit_transform_records(recs)
    X = ext.model_matrix(df)
    assert X.shape[0] == 2
    assert X.shape[1] == len(ext.feature_names_)
    assert "quality_flag" in df.columns


def test_default_training_models_are_supported():
    assert DEFAULT_CFG["training"]["models"] == [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]


def test_load_fasta_merge_contigs(tmp_path):
    fa = tmp_path / "multi.fasta"
    fa.write_text(">c1\nACGTACGT\n>c2\nTTGCTTGC\n", encoding="utf-8")
    recs = load_fasta(fa, merge_contigs=True)
    assert len(recs) == 1
    assert recs[0].n_contigs == 2
    assert "N" * 50 in recs[0].sequence


def test_duplicate_ids_disambiguated(tmp_path):
    fa = tmp_path / "dup.fasta"
    fa.write_text(">same\nAAAA\n>same\nCCCC\n", encoding="utf-8")
    recs = load_fasta(fa, merge_contigs=False)
    ids = [r.id for r in recs]
    assert len(ids) == len(set(ids))


def test_bio_features_calculation():
    from mpox_clf.features.bio_features import compute_all_bio_features
    seq = "GAAATCCT" * 100
    bio = compute_all_bio_features(seq)
    assert "apobec3_combined_score" in bio
    assert "gc_skew" in bio
    assert "cai_score" in bio
    assert "snp_anchor_Ia_1" in bio
    assert "del_flag_left_tir" in bio


def test_ensemble_and_ood():
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from mpox_clf.training.ensemble import MpoxEnsemble
    from mpox_clf.training.ood import NoveltyDetector

    X = np.random.randn(10, 5)
    y = np.array(["Ia"] * 5 + ["Ib"] * 5)
    m1 = DummyClassifier(strategy="most_frequent").fit(X, y)
    m2 = DummyClassifier(strategy="most_frequent").fit(X, y)
    
    ens = MpoxEnsemble({"m1": m1, "m2": m2}, ["Ia", "Ib"])
    preds = ens.predict_with_consensus(X)
    assert len(preds) == 10
    assert "consensus_ratio" in preds[0]

    ood = NoveltyDetector().fit(X)
    anom = ood.predict_anomaly(X)
    assert len(anom) == 10
    assert "is_ood" in anom[0]


def test_audit_and_active_learning(tmp_path):
    from mpox_clf.utils.audit import log_audit_event
    from mpox_clf.inference.active_learning import log_user_correction
    from mpox_clf.utils.pdf_report import generate_printable_html_report

    audit_file = tmp_path / "audit.jsonl"
    log_audit_event("seq1", "ACGT", "Ia", 0.99, "Good", log_path=audit_file)
    assert audit_file.exists()

    corr_file = tmp_path / "corr.csv"
    log_user_correction("seq1", "ACGT", "Ia", "Ib", 0.99, log_path=corr_file)
    assert corr_file.exists()

    html = generate_printable_html_report({"sequence_id": "seq1", "predicted_clade": "Ia"})
    assert "Mpox Diagnostic Report" in html

