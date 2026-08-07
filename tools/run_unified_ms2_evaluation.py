"""CLI for the unified manifest-driven MS2 evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ms2_eval.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true", help="Run manifest/hash/GT audit without predictions")
    # Comparing routes only means something when every one of them was measured the
    # same way, so the overrides reach which route is being read and nowhere else.
    # Manifest, depth bounds, alignment and aggregation stay wherever the config
    # put them; change the ruler by editing the config, for every route at once.
    parser.add_argument("--route", help="Override model.route")
    parser.add_argument("--prediction-dir", type=Path, help="Override model.prediction_dir")
    parser.add_argument("--output-dir", type=Path, help="Override output_dir")
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("PyYAML is required") from error
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.audit_only: config["audit_only"] = True
    if args.route: config["model"]["route"] = args.route
    if args.prediction_dir: config["model"]["prediction_dir"] = str(args.prediction_dir)
    if args.output_dir: config["output_dir"] = str(args.output_dir)
    print(json.dumps(run(config), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__": main()
