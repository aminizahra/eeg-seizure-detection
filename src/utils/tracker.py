# src/utils/tracker.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, (list, tuple)):
            text = str(v)
            items[new_key] = text if len(text) <= 240 else f"len={len(v)}"
        elif isinstance(v, Path):
            items[new_key] = str(v)
        else:
            items[new_key] = v
    return items


class ExperimentTracker:
    """Unified tracker for MLflow (local SQLite) and Weights & Biases (cloud GPU)."""

    def __init__(self, cfg: dict[str, Any], run_name: str, tags: dict[str, str] | None = None):
        self._t_cfg = cfg.get("tracker", {})
        self.backend = str(self._t_cfg.get("backend", "none")).lower()
        self.run_name = run_name
        self.tags = tags or {}
        self._mlflow = None
        self._wandb_run = None

    def start(self) -> "ExperimentTracker":
        if self.backend == "mlflow":
            import mlflow

            db_path = (PROJECT_ROOT / self._t_cfg.get("mlflow_db", "mlflow.db")).resolve()
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment(self._t_cfg.get("experiment_name", "CHB-MIT-Phase-A1"))
            mlflow.start_run(run_name=self.run_name, tags=self.tags)
            self._mlflow = mlflow

        elif self.backend == "wandb":
            import wandb

            self._wandb_run = wandb.init(
                project=self._t_cfg.get("wandb_project", "eeg-seizure-chbmit"),
                name=self.run_name,
                mode=self._t_cfg.get("wandb_mode", "online"),
                tags=list(self.tags.values()) if self.tags else None,
                reinit=True,
            )

        elif self.backend != "none":
            raise ValueError(f"Unsupported tracker backend: {self.backend}")

        return self

    def log_params(self, params: dict[str, Any]) -> None:
        flat = _flatten_dict(params)
        if self.backend == "mlflow" and self._mlflow is not None:
            items = list(flat.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))
        elif self.backend == "wandb" and self._wandb_run is not None:
            self._wandb_run.config.update(flat, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        if not clean:
            return
        if self.backend == "mlflow" and self._mlflow is not None:
            self._mlflow.log_metrics(clean, step=step)
        elif self.backend == "wandb" and self._wandb_run is not None:
            if step is not None:
                clean["step"] = step
            self._wandb_run.log(clean, step=step)

    def log_artifact(self, filepath: str | Path) -> None:
        path = Path(filepath)
        if not path.exists():
            return
        if self.backend == "mlflow" and self._mlflow is not None:
            self._mlflow.log_artifact(str(path))
        elif self.backend == "wandb" and self._wandb_run is not None:
            import wandb
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                self._wandb_run.log({path.stem: wandb.Image(str(path))})
            else:
                self._wandb_run.save(str(path))

    def finish(self) -> None:
        if self.backend == "mlflow" and self._mlflow is not None:
            self._mlflow.end_run()
            self._mlflow = None
        elif self.backend == "wandb" and self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def __enter__(self) -> "ExperimentTracker":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()