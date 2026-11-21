# src/config.py
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE pairs from local .env without overriding OS env vars."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_path(path_value: str | Path) -> Path:
    p = Path(path_value).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def fix_socks_proxy() -> None:
    for key, val in list(os.environ.items()):
        if "proxy" in key.lower() and val.startswith("socks://"):
            os.environ[key] = val.replace("socks://", "socks5://", 1)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str | Path = "configs/phase_a1.yaml") -> dict[str, Any]:
    """Load YAML config, resolve paths from PROJECT_ROOT/.env, and set runtime defaults."""
    fix_socks_proxy()
    _load_dotenv()

    cfg_file = resolve_path(config_path)
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = resolve_path(os.environ.get("EEG_RAW_DATA_DIR") or cfg["paths"]["raw_data_dir"])
    processed_dir = resolve_path(
        os.environ.get("EEG_PROCESSED_DATA_DIR") or cfg["paths"]["processed_data_dir"]
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    is_gpu = torch.cuda.is_available()
    cfg["runtime"] = {
        "project_root": PROJECT_ROOT,
        "raw_data_dir": raw_dir,
        "processed_data_dir": processed_dir,
        "device": torch.device("cuda" if is_gpu else "cpu"),
        "is_gpu": is_gpu,
        "epochs": cfg["training"]["epochs_gpu"] if is_gpu else cfg["training"]["epochs_cpu"],
        "batch_size": cfg["training"]["batch_size_gpu"] if is_gpu else cfg["training"]["batch_size_cpu"],
        "num_workers": cfg["training"]["num_workers_gpu"] if is_gpu else cfg["training"]["num_workers_cpu"],
    }

    set_global_seed(int(cfg["project"]["seed"]))
    return cfg