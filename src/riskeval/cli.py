from __future__ import annotations

import argparse
import json

from .config import load_config
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RiskEval framework reproduction")
    p.add_argument("--config", required=True, help="Path to config TOML")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    out = run(cfg)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
