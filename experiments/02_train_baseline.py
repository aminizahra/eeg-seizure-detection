# experiments/02_train_baseline.py
from __future__ import annotations

import argparse
import pandas as pd

from src.config import load_config
from src.features.bandpower import extract_or_load_xgb_features
from src.models.baseline_xgb import train_xgboost_cv
from src.utils.tracker import ExperimentTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Extract Welch/LineLength features and train XGBoost baseline.")
    parser.add_argument("--config", type=str, default="configs/phase_a1.yaml")
    parser.add_argument("--force-features", action="store_true", help="Recompute xgb_features.npy even if cached.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["runtime"]["processed_data_dir"]
    cases_tag = ",".join(cfg["dataset"]["selected_cases"]) if cfg["dataset"]["selected_cases"] else "all_23"

    meta_df = pd.read_csv(out_dir / "windows_metadata.csv")
    splits_df = pd.read_csv(out_dir / "splits.csv")

    with ExperimentTracker(
        cfg,
        run_name=f"02_baseline_xgb_{cases_tag}",
        tags={"stage": "baseline", "model": "xgboost"},
    ) as tr:
        X, feature_names = extract_or_load_xgb_features(
            cfg=cfg,
            n_expected_windows=len(meta_df),
            force=args.force_features,
        )
        _, summary, imp_df = train_xgboost_cv(
            X=X,
            feature_names=feature_names,
            meta_df=meta_df,
            splits_df=splits_df,
            cfg=cfg,
            tracker=tr,
        )

        print("\n=== Stage 2 Complete (XGBoost Baseline) ===")
        for k, v in summary.items():
            print(f"  {k:22s}: {v:.4f}")
        print("\nTop 5 Features:")
        print(imp_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()