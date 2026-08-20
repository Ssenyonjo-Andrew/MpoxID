"""Configuration helpers — YAML defaults with optional overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

PathLike = Union[str, Path]

DEFAULT_CFG = {
    "project": {"name": "mega_mpox", "version": "2.0.0", "random_seed": 42},
    "paths": {
        "raw_fasta": "data/raw",
        "metadata": "data/metadata/metadata.csv",
        "models_dir": "models",
        "processed_features": "data/processed/features.parquet",
        "reports_dir": "reports",
    },
    "quality": {
        "min_length_fair": 150000,
        "min_length_poor": 50000,
        "n_pct_fair": 1.0,
        "n_pct_poor": 5.0,
        "max_non_acgtn": 20,
        "max_premature_stops": 3,
        "frameshift_window": 3,
    },
    "features": {"kmer_sizes": [2, 3, 4], "use_canonical_kmers": True, "include_codon_usage": True},
    "training": {
        "test_size": 0.2,
        "chronological_holdout_fraction": 0.2,
        "n_cv_folds": 5,
        "deploy_model": "auto",
        "models": ["logistic_regression", "random_forest", "xgboost"],
        "random_seed": 42,
        "tie_break_priority_clades": ["Ia", "Ib"],
        "min_priority_recall_floor": 0.2,
        "quality_filter_for_training": False,
        "deep_learning": {
            "backend": "auto",
            "max_epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.001,
            "patience": 4,
            "max_sequence_length": 4096,
            "representation": "chunked_onehot",
            "chunk_strategy": "evenly_spaced",
            "n_chunks": 32,
            "chunk_length": 128,
            "hidden_dim": 128,
            "dropout": 0.3,
        },
    },
    "inference": {
        "inference_mode": "online",
        "online_provider": "inprocess",
        "online_url": "http://127.0.0.1:8765/predict",
        "request_timeout_seconds": 30,
        "confidence_decimals": 4,
        "top_kmers_per_clade": 10,
    },
}


def project_root() -> Path:
    """Return repository root (two levels above this file: utils/ → mpox_clf/ → src/ → root)."""
    return Path(__file__).resolve().parents[3]


def load_config(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """
    Load YAML config with fallback dictionary if PyYAML is unavailable.
    """
    cfg_path = Path(path) if path else project_root() / "config" / "default_config.yaml"
    if HAS_YAML and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    return DEFAULT_CFG
