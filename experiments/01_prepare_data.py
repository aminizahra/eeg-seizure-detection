# experiments/01_prepare_data.py
from __future__ import annotations

import argparse
import pandas as pd

from src.config import load_config
from src.data.manifest import build_manifests
from src.data.preprocess import build_hdf5_windows
from src.data.splits import build_splits
from src.utils.tracker import ExperimentTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Build CHB-MIT manifests, splits, and HDF5 windows.")
    parser.add_argument("--config", type=str, default="configs/phase_a1.yaml")
    parser.add_argument("--force", action="store_true", help="Re-run EDF preprocessing even if eeg_windows.h5 exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["runtime"]["processed_data_dir"]
    cases_tag = ",".join(cfg["dataset"]["selected_cases"]) if cfg["dataset"]["selected_cases"] else "all_23"

    with ExperimentTracker(cfg, run_name=f"01_prepare_data_{cases_tag}", tags={"stage": "prepare_data"}) as tr:
        tr.log_params({
            "dataset": cfg["dataset"],
            "preprocessing": cfg["preprocessing"],
            "splits": cfg["splits"],
        })

        files_df, seizures_df = build_manifests(cfg)
        splits_df = build_splits(files_df, cfg)

        hdf5_path = out_dir / "eeg_windows.h5"
        meta_path = out_dir / "windows_metadata.csv"

        if hdf5_path.exists() and meta_path.exists() and not args.force:
            print(f"[SKIP] Found existing {hdf5_path.name} and {meta_path.name}. Use --force to rebuild.")
            meta_df = pd.read_csv(meta_path)
        else:
            hdf5_path, meta_df = build_hdf5_windows(files_df, seizures_df, cfg)

        metrics = {
            "n_subjects": files_df["subject_id"].nunique(),
            "n_valid_files": int(((files_df["read_ok"]) & (files_df["has_std_18"])).sum()),
            "n_seizures": len(seizures_df),
            "n_folds": splits_df["fold"].nunique(),
            "n_windows_total": len(meta_df),
            "n_windows_eval_grid": int(meta_df["is_eval_grid"].sum()),
            "n_windows_positive": int((meta_df["label"] == 1).sum()),
        }
        tr.log_metrics(metrics)
        tr.log_artifact(out_dir / "files_manifest.csv")
        tr.log_artifact(out_dir / "seizures_manifest.csv")
        tr.log_artifact(out_dir / "splits.csv")

        print("=== Stage 1 Complete ===")
        for k, v in metrics.items():
            print(f"  {k:22s}: {v}")


if __name__ == "__main__":
    main()