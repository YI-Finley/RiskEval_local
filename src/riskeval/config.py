from __future__ import annotations

import os
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.8  uses another package
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    solver_model: str
    parser_model: str
    judge_model: str
    supports_vision: bool = False
    temperature: float = 0.0
    max_tokens: int = 1200


@dataclass
class APIConfig:
    api_key_env: str
    base_url: str
    api_version: str = "2024-12-01-preview"
    request_timeout_sec: int = 300
    max_retries: int = 3


@dataclass
class SweepConfig:
    penalties: list[float]


@dataclass
class RunConfig:
    data_path: Path
    out_dir: Path
    prompt_strategy: int
    max_examples: int | None
    random_seed: int


@dataclass
class Config:
    api: APIConfig
    models: ModelConfig
    sweep: SweepConfig
    run: RunConfig


def _expand(path_str: str) -> Path:
    return Path(os.path.expandvars(path_str)).expanduser().resolve()


def load_config(path: str | Path) -> Config:
    p = Path(path)
    data = tomllib.loads(p.read_text(encoding="utf-8"))

    api = APIConfig(
        api_key_env=data["api"]["api_key_env"],
        base_url=data["api"]["base_url"],
        api_version=data["api"].get("api_version", "2024-12-01-preview"),
        request_timeout_sec=int(data["api"].get("request_timeout_sec", 300)),
        max_retries=int(data["api"].get("max_retries", 3)),
    )

    models = ModelConfig(
        solver_model=data["models"]["solver_model"],
        parser_model=data["models"].get("parser_model", data["models"]["solver_model"]),
        judge_model=data["models"]["judge_model"],
        supports_vision=bool(data["models"].get("supports_vision", False)),
        temperature=float(data["models"].get("temperature", 0.0)),
        max_tokens=int(data["models"].get("max_tokens", 1200)),
    )

    sweep = SweepConfig(
        penalties=[float(x) for x in data["sweep"]["penalties"]],
    )

    run = RunConfig(
        data_path=_expand(data["run"]["data_path"]),
        out_dir=_expand(data["run"]["out_dir"]),
        prompt_strategy=int(data["run"].get("prompt_strategy", 1)),
        max_examples=(int(data["run"]["max_examples"]) if data["run"].get("max_examples") else None),
        random_seed=int(data["run"].get("random_seed", 42)),
    )

    return Config(api=api, models=models, sweep=sweep, run=run)


def resolve_api_key(api_key_env: str) -> str:
    key = os.getenv(api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {api_key_env} first."
        )
    return key
