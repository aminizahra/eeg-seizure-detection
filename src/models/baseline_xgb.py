# src/models/baseline_xgb.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.data.splits import get_fold_window_indices
from src.utils.tracker import ExperimentTracker


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")


def _save_feature_importance_plot(imp_df: pd.DataFrame, save_path: Path, top_k: int = 20) -> None:
    top = imp_df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_importance"])
    ax.set_xlabel("Mean Gain Importance across Folds")
    ax.set_title(f"XGBoost Baseline — Top {top_k} EEG Features")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_xgboost_cv(
    X: np.ndarray,
    feature_names: list[str],
    meta_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    cfg: dict[str, Any],
    tracker: ExperimentTracker | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    out_dir: Path = cfg["runtime"]["processed_data_dir"]
    seed = int(cfg["project"]["seed"])
    neg_pos_ratio = int(cfg["training"]["train_neg_pos_ratio_xgb"])
    use_gpu = bool(cfg["runtime"]["is_gpu"])

    y_all = meta_df["label"].values.astype(np.int32)
    folds = sorted(splits_df["fold"].unique())

    xgb_hparams = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "n_jobs": -1,
    }
    if tracker is not None:
        tracker.log_params({
            "model": "xgboost",
            "n_features": len(feature_names),
            "train_neg_pos_ratio": neg_pos_ratio,
            **xgb_hparams,
        })

    oof_frames: list[pd.DataFrame] = []
    fold_val_auprcs: list[float] = []
    fold_test_auprcs: list[float] = []
    fold_test_aurocs: list[float] = []
    importances = np.zeros((len(folds), len(feature_names)), dtype=np.float64)

    for f_pos, fold_idx in enumerate(folds):
        idx = get_fold_window_indices(
            windows_meta_df=meta_df,
            splits_df=splits_df,
            fold_idx=int(fold_idx),
            max_neg_pos_ratio=neg_pos_ratio,
            seed=seed,
        )
        X_tr, y_tr = X[idx["train"]], y_all[idx["train"]]
        X_va, y_va = X[idx["val"]], y_all[idx["val"]]
        X_te, y_te = X[idx["test"]], y_all[idx["test"]]

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        scale_pos_weight = float(n_neg / max(n_pos, 1))

        model = XGBClassifier(
            **xgb_hparams,
            scale_pos_weight=scale_pos_weight,
            random_state=seed + int(fold_idx),
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        val_probs = model.predict_proba(X_va)[:, 1]
        test_probs = model.predict_proba(X_te)[:, 1]

        val_auprc = _safe_auprc(y_va, val_probs)
        val_auroc = _safe_auroc(y_va, val_probs)
        test_auprc = _safe_auprc(y_te, test_probs)
        test_auroc = _safe_auroc(y_te, test_probs)

        fold_val_auprcs.append(val_auprc)
        fold_test_auprcs.append(test_auprc)
        fold_test_aurocs.append(test_auroc)
        importances[f_pos] = model.feature_importances_

        print(
            f"[Fold {fold_idx}] best_iter={model.best_iteration:3d} | "
            f"val_AUPRC={val_auprc:.4f} | test_AUPRC={test_auprc:.4f} | test_AUROC={test_auroc:.4f}"
        )

        if tracker is not None:
            tracker.log_metrics(
                {
                    "fold_val_auprc": val_auprc,
                    "fold_val_auroc": val_auroc,
                    "fold_test_auprc": test_auprc,
                    "fold_test_auroc": test_auroc,
                    "fold_best_iteration": float(model.best_iteration),
                },
                step=int(fold_idx),
            )

        fold_oof = meta_df.iloc[idx["test"]][
            ["h5_idx", "subject_id", "case_id", "filename", "rel_path", "window_start_sec", "window_end_sec", "label"]
        ].copy()
        fold_oof["fold"] = int(fold_idx)
        fold_oof["prob"] = test_probs.astype(np.float32)
        oof_frames.append(fold_oof)

    oof_df = pd.concat(oof_frames, ignore_index=True).sort_values("h5_idx").reset_index(drop=True)
    oof_path = out_dir / "oof_preds_xgb.csv"
    oof_df.to_csv(oof_path, index=False)

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": importances.mean(axis=0),
        "std_importance": importances.std(axis=0),
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    imp_csv = out_dir / "xgb_feature_importance.csv"
    imp_png = out_dir / "xgb_feature_importance.png"
    imp_df.to_csv(imp_csv, index=False)
    _save_feature_importance_plot(imp_df, imp_png, top_k=20)

    summary = {
        "oof_auprc": _safe_auprc(oof_df["label"].values, oof_df["prob"].values),
        "oof_auroc": _safe_auroc(oof_df["label"].values, oof_df["prob"].values),
        "mean_fold_val_auprc": float(np.nanmean(fold_val_auprcs)),
        "mean_fold_test_auprc": float(np.nanmean(fold_test_auprcs)),
        "std_fold_test_auprc": float(np.nanstd(fold_test_auprcs)),
        "mean_fold_test_auroc": float(np.nanmean(fold_test_aurocs)),
    }

    if tracker is not None:
        tracker.log_metrics(summary)
        tracker.log_artifact(imp_csv)
        tracker.log_artifact(imp_png)
        tracker.log_artifact(out_dir / "xgb_feature_names.json")

    return oof_df, summary, imp_df