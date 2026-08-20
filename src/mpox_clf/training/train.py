"""
Train classical clade classifiers: Logistic Regression, Random Forest, XGBoost.

Primary deployment artifact: XGBoost (+ FeatureExtractor + LabelEncoder metadata)
saved under models/ via joblib. Total size is typically well under 10 MB.

Optional 1D CNN is intentionally NOT in the default path — see docs/OPTIONAL_CNN.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

from ..features.extractor import FeatureExtractor
from ..preprocessing.fasta_io import SequenceRecord
from ..preprocessing.prepare_training_data import build_labeled_dataset
from ..utils.config import load_config, project_root
import hashlib
from .ensemble import MpoxEnsemble
from .evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_model_comparison,
    write_evaluation_report,
)
from .imbalance import balanced_class_weight_dict, imbalance_report, sample_weights
from .ood import NoveltyDetector

PathLike = Union[str, Path]


def _make_model(name: str, n_classes: int, seed: int = 42):
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        if HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=seed,
            )
        else:
            return HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                learning_rate=0.08,
                random_state=seed,
            )
    raise ValueError(f"Unknown model: {name}")


def _min_class_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def _cv_predict(name: str, X, y_int, y_text, le, n_classes, n_folds, seed):
    """Manual stratified CV so sample_weight works for all estimators."""
    n_splits = min(n_folds, _min_class_count(y_text))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty(len(y_text), dtype=object)
    for train_idx, test_idx in skf.split(X, y_text):
        model = _make_model(name, n_classes, seed)
        sw = sample_weights(y_text[train_idx])
        if name == "xgboost" and HAS_XGBOOST:
            model.fit(X[train_idx], y_int[train_idx], sample_weight=sw)
            preds[test_idx] = le.inverse_transform(
                model.predict(X[test_idx]).astype(int)
            )
        elif name == "logistic_regression":
            model.fit(X[train_idx], y_text[train_idx])
            preds[test_idx] = model.predict(X[test_idx])
        else:
            model.fit(X[train_idx], y_text[train_idx], sample_weight=sw)
            preds[test_idx] = model.predict(X[test_idx])
    return preds


def _fit_full(name: str, model, X_train, y_train, y_text_train, sw_train):
    if name == "xgboost" and HAS_XGBOOST:
        model.fit(X_train, y_train, sample_weight=sw_train)
    elif name == "logistic_regression":
        model.fit(X_train, y_text_train)
    else:
        model.fit(X_train, y_text_train, sample_weight=sw_train)
    return model


def _predict(name: str, model, X_test, le):
    if name == "xgboost" and HAS_XGBOOST:
        return le.inverse_transform(model.predict(X_test).astype(int))
    return model.predict(X_test)


def train_from_dataframe(
    labeled_df: pd.DataFrame,
    *,
    config: Optional[Dict] = None,
    models_dir: Optional[PathLike] = None,
    reports_dir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Train/compare models from a DataFrame with columns: accession, clade, sequence.
    """
    cfg = config or load_config()
    root = project_root()
    models_dir = Path(models_dir or root / cfg["paths"]["models_dir"])
    reports_dir = Path(reports_dir or root / cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("project", {}).get("random_seed", 42))
    labels_order = list(cfg.get("clades", {}).get("labels", ["Ia", "Ib", "IIa", "IIb"]))
    quality_kwargs = dict(cfg.get("quality", {}))

    records = [
        SequenceRecord(
            id=str(r.accession), sequence=str(r.sequence), description=str(r.clade)
        )
        for r in labeled_df.itertuples()
    ]
    y_text = labeled_df["clade"].astype(str).to_numpy()
    print(imbalance_report(y_text))

    feat_cfg = cfg.get("features", {})
    extractor = FeatureExtractor(
        kmer_sizes=feat_cfg.get("kmer_sizes", [2, 3, 4]),
        use_canonical_kmers=feat_cfg.get("use_canonical_kmers", True),
        include_codon_usage=feat_cfg.get("include_codon_usage", True),
        quality_thresholds=quality_kwargs,
    )
    print("[train] Extracting features...")
    feat_df = extractor.fit_transform_records(records)
    X = extractor.model_matrix(feat_df)

    le = LabelEncoder()
    present = [c for c in labels_order if c in set(y_text)]
    extras = sorted(set(y_text) - set(present))
    le.fit(present + extras)
    y = le.transform(y_text)
    n_classes = len(le.classes_)

    test_size = float(cfg.get("training", {}).get("test_size", 0.2))
    n_folds = int(cfg.get("training", {}).get("n_cv_folds", 5))

    X_train, X_test, y_train, y_test, y_text_train, y_text_test = train_test_split(
        X, y, y_text, test_size=test_size, random_state=seed, stratify=y_text
    )

    wanted = cfg.get("training", {}).get(
        "models", ["logistic_regression", "random_forest", "xgboost"]
    )
    results: Dict[str, Dict] = {}
    fitted: Dict[str, Any] = {}
    sw_train = sample_weights(y_text_train)

    for name in wanted:
        print(f"[train] Fitting {name}...")
        model = _make_model(name, n_classes, seed)
        _fit_full(name, model, X_train, y_train, y_text_train, sw_train)
        y_pred = _predict(name, model, X_test, le)

        cv_pred = _cv_predict(name, X, y, y_text, le, n_classes, n_folds, seed)
        holdout = evaluate_predictions(y_text_test, y_pred, labels=list(le.classes_))
        cv_eval = evaluate_predictions(y_text, cv_pred, labels=list(le.classes_))
        summary = {
            **cv_eval,
            "holdout_accuracy": holdout["accuracy"],
            "holdout_macro_f1": holdout["macro_f1"],
        }
        results[name] = summary
        fitted[name] = model

        plot_confusion_matrix(
            cv_eval["confusion_matrix"],
            cv_eval["labels"],
            reports_dir / "figures" / f"cm_{name}.png",
            title=f"{name} (stratified CV)",
        )
        joblib.dump(model, models_dir / f"model_{name}.joblib")

    plot_model_comparison(results, reports_dir / "figures" / "model_comparison.png")

    deploy_name = cfg.get("training", {}).get("deploy_model", "xgboost")
    if deploy_name not in fitted:
        deploy_name = max(results, key=lambda n: results[n]["macro_f1"])
        print(f"[train] Using best macro-F1 model: {deploy_name}")

    # Build Ensemble Classifier & Out-of-Distribution Novelty Detector
    print("[train] Fitting ensemble classifier & OOD novelty detector...")
    ensemble = MpoxEnsemble(fitted, list(le.classes_))
    ood_detector = NoveltyDetector()
    ood_detector.fit(X)

    # Compute schema hash and clade feature means for explainability
    schema_str = ",".join(extractor.feature_names_)
    schema_hash = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:12]

    feat_df = feat_df.copy()
    feat_df["clade"] = y_text

    clade_means = {}
    for clade in le.classes_:
        sub_X = X[y_text == clade]
        if len(sub_X) > 0:
            mean_vec = sub_X.mean(axis=0)
            clade_means[clade] = {
                fname: float(mean_vec[j]) for j, fname in enumerate(extractor.feature_names_)
            }

    joblib.dump(extractor, models_dir / "feature_extractor.joblib")
    joblib.dump(le, models_dir / "label_encoder.joblib")

    bundle = {
        "model_name": deploy_name,
        "model": fitted[deploy_name],
        "ensemble": ensemble,
        "ood_detector": ood_detector,
        "extractor": extractor,
        "label_encoder": le,
        "feature_names": extractor.feature_names_,
        "clades": list(le.classes_),
        "schema_hash": schema_hash,
        "n_sequences": int(len(labeled_df)),
        "clade_means": clade_means,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "version": cfg.get("project", {}).get("version", "1.0.0"),
    }
    joblib.dump(bundle, models_dir / "deploy_bundle.joblib")

    kmer_cols = [c for c in extractor.feature_names_ if c.startswith("kmer4_")]
    top_kmers = {}
    for clade in le.classes_:
        sub = feat_df.loc[feat_df["clade"] == clade, kmer_cols]
        if len(sub) == 0:
            continue
        means = sub.mean().sort_values(ascending=False).head(10)
        top_kmers[clade] = [
            {"kmer": idx.replace("kmer4_", ""), "mean_freq": float(val)}
            for idx, val in means.items()
        ]

    meta = {
        "deploy_model": deploy_name,
        "ensemble_models": list(fitted.keys()),
        "n_sequences": int(len(labeled_df)),
        "n_features": int(X.shape[1]),
        "schema_hash": schema_hash,
        "classes": list(le.classes_),
        "class_counts": {c: int((y_text == c).sum()) for c in le.classes_},
        "class_weights": balanced_class_weight_dict(y_text),
        "metrics": {
            n: {"accuracy": results[n]["accuracy"], "macro_f1": results[n]["macro_f1"]}
            for n in results
        },
        "top_kmers_per_clade": top_kmers,
        "trained_at": bundle["trained_at"],
    }
    accuracy_artifact = {
        "dataset": {
            "n_sequences": int(len(labeled_df)),
            "class_counts": meta["class_counts"],
            "classes": list(le.classes_),
            "feature_count": int(X.shape[1]),
            "trained_at": bundle["trained_at"],
        },
        "selection_metric": "stratified_cross_validation_macro_f1",
        "deployed_model": deploy_name,
        "models": {
            name: {
                "cross_validation_accuracy": float(results[name]["accuracy"]),
                "cross_validation_macro_f1": float(results[name]["macro_f1"]),
                "holdout_accuracy": float(results[name]["holdout_accuracy"]),
                "holdout_macro_f1": float(results[name]["holdout_macro_f1"]),
                "per_class": results[name]["per_class"].to_dict(orient="records"),
            }
            for name in results
        },
    }
    (models_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (models_dir / "model_comparison.json").write_text(
        json.dumps(meta["metrics"], indent=2), encoding="utf-8"
    )
    (models_dir / "accuracy_artifact.json").write_text(
        json.dumps(accuracy_artifact, indent=2), encoding="utf-8"
    )

    write_evaluation_report(
        results,
        reports_dir / "evaluation_report.md",
        deploy_model=deploy_name,
        imbalance_text=imbalance_report(y_text),
        extra_notes=(
            "Quality metrics (N%, stops, frameshifts) are deterministic, not learned. "
            "Class weighting + stratified CV address Ia/Ib/IIa/IIb imbalance."
        ),
    )

    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    out_feats = feat_df.drop(
        columns=[
            c
            for c in feat_df.columns
            if c
            in (
                "quality_explanation",
                "quality_reasons",
                "frameshift_reasons",
                "top_kmers_k4",
            )
        ],
        errors="ignore",
    )
    try:
        out_feats.to_parquet(proc / "features.parquet", index=False)
    except Exception:
        out_feats.to_csv(proc / "features.csv", index=False)

    print(f"[train] Done. Deploy bundle -> {models_dir / 'deploy_bundle.joblib'}")
    return {"results": results, "deploy_model": deploy_name, "meta": meta}


def train_from_fasta_and_metadata(
    fasta_path: PathLike,
    metadata_path: PathLike,
    **kwargs,
) -> Dict[str, Any]:
    labeled = build_labeled_dataset(fasta_path, metadata_path)
    if labeled.empty:
        raise RuntimeError(
            "No labeled sequences — check FASTA IDs vs metadata.accession. "
            "See docs/DATA_SOURCING.md"
        )
    return train_from_dataframe(labeled, **kwargs)
