from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpox_clf.training.model_selection import choose_best_model
from mpox_clf.utils.config import load_config


def main() -> None:
    cfg = load_config()
    comparison_path = ROOT / "models" / "model_comparison.json"
    if not comparison_path.exists():
        raise SystemExit(f"Missing comparison file: {comparison_path}")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    chrono = {k: v["chronological_holdout"] for k, v in comparison.items() if "chronological_holdout" in v}
    winner, annotated = choose_best_model(
        chrono,
        priority_clades=list(cfg.get("training", {}).get("tie_break_priority_clades", ["Ia", "Ib"])),
        min_priority_recall_floor=float(cfg.get("training", {}).get("min_priority_recall_floor", 0.2)),
    )
    print(json.dumps({"winner": winner, "annotated": annotated}, indent=2))


if __name__ == "__main__":
    main()
