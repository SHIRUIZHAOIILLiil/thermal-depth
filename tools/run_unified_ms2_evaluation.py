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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true", help="Run manifest/hash/GT audit without predictions")
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("PyYAML is required") from error
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.audit_only: config["audit_only"] = True
    print(json.dumps(run(config), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__": main()
