# src/data/splits.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold


def verify_zero_leakage(splits_df: pd.DataFrame, multi_subject: bool) -> None:
    """Hard assertion guaranteeing zero file overlap (and zero patient overlap in multi-subject mode)."""
    for fold, grp in splits_df.groupby("fold"):
        roles = {r: set(sub["rel_path"]) for r, sub in grp.groupby("role")}
        assert roles["train"].isdisjoint(roles["val"]), f"Fold {fold}: file leak train<->val!"
        assert roles["train"].isdisjoint(roles["test"]), f"Fold {fold}: file leak train<->test!"
        assert roles["val"].isdisjoint(roles["test"]), f"Fold {fold}: file leak val<->test!"

        if multi_subject:
            s_roles = {r: set(sub["subject_id"]) for r, sub in grp.groupby("role")}
            assert s_roles["train"].isdisjoint(s_roles["val"]), f"Fold {fold}: patient leak train<->val!"
            assert s_roles["train"].isdisjoint(s_roles["test"]), f"Fold {fold}: patient leak train<->test!"
            assert s_roles["val"].isdisjoint(s_roles["test"]), f"Fold {fold}: patient leak val<->test!"


def build_splits(files_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir: Path = cfg["runtime"]["processed_data_dir"]
    seed = int(cfg["project"]["seed"])
    split_cfg = cfg["splits"]

    valid_df = files_df[(files_df["read_ok"]) & (files_df["has_std_18"])].reset_index(drop=True)
    unique_subjects = sorted(valid_df["subject_id"].unique())
    multi_subject = len(unique_subjects) > 1
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []

    if not multi_subject:
        # Single-subject dev mode: file-level stratified split ensuring seizures in train/val/test
        n_folds = int(split_cfg.get("n_folds_single", 3))
        y_file = valid_df["has_seizure"].astype(int).values
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(valid_df, y_file)):
            tv_df = valid_df.iloc[trainval_idx]
            tv_y = tv_df["has_seizure"].astype(int).values
            # Stratified val split (~val_ratio) preserving at least 1 seizure file in val
            n_val_splits = max(2, int(round(1.0 / float(split_cfg["val_ratio"]))))
            n_pos_tv = int(tv_y.sum())
            n_val_splits = min(n_val_splits, max(2, n_pos_tv))
            inner_skf = StratifiedKFold(n_splits=n_val_splits, shuffle=True, random_state=seed + fold_idx)
            tr_sub, va_sub = next(inner_skf.split(tv_df, tv_y))

            role_map = {}
            for i in trainval_idx[tr_sub]:
                role_map[i] = "train"
            for i in trainval_idx[va_sub]:
                role_map[i] = "val"
            for i in test_idx:
                role_map[i] = "test"

            for idx, role in role_map.items():
                r = valid_df.iloc[idx]
                rows.append({
                    "fold": fold_idx,
                    "role": role,
                    "subject_id": r["subject_id"],
                    "case_id": r["case_id"],
                    "filename": r["filename"],
                    "rel_path": r["rel_path"],
                    "has_seizure": bool(r["has_seizure"]),
                })
    else:
        # Multi-patient mode: strict patient-level GroupKFold or LOSO
        strategy = str(split_cfg.get("patient_strategy", "group_kfold")).lower()
        groups = valid_df["subject_id"].values
        if strategy == "loso":
            splitter = LeaveOneGroupOut()
        else:
            n_folds = min(int(split_cfg.get("n_folds_multi", 5)), len(unique_subjects))
            splitter = GroupKFold(n_splits=n_folds)

        for fold_idx, (trainval_idx, test_idx) in enumerate(splitter.split(valid_df, groups=groups)):
            tv_subjects = sorted(valid_df.iloc[trainval_idx]["subject_id"].unique())
            n_val_subj = max(1, int(round(len(tv_subjects) * float(split_cfg["val_ratio"]))))
            val_subjects = set(rng.choice(tv_subjects, size=n_val_subj, replace=False).tolist())

            for idx in trainval_idx:
                r = valid_df.iloc[idx]
                role = "val" if r["subject_id"] in val_subjects else "train"
                rows.append({
                    "fold": fold_idx,
                    "role": role,
                    "subject_id": r["subject_id"],
                    "case_id": r["case_id"],
                    "filename": r["filename"],
                    "rel_path": r["rel_path"],
                    "has_seizure": bool(r["has_seizure"]),
                })
            for idx in test_idx:
                r = valid_df.iloc[idx]
                rows.append({
                    "fold": fold_idx,
                    "role": "test",
                    "subject_id": r["subject_id"],
                    "case_id": r["case_id"],
                    "filename": r["filename"],
                    "rel_path": r["rel_path"],
                    "has_seizure": bool(r["has_seizure"]),
                })

    splits_df = pd.DataFrame(rows)
    verify_zero_leakage(splits_df, multi_subject=multi_subject)
    splits_df.to_csv(out_dir / "splits.csv", index=False)
    return splits_df


def get_fold_window_indices(
    windows_meta_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    fold_idx: int,
    max_neg_pos_ratio: int | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Return h5_idx arrays for train / val / test in a fold:
      - train: uses both eval-grid and dense ictal windows, optionally subsampling negatives
      - val / test: strictly non-overlapping `is_eval_grid == True` windows only
    """
    fold_splits = splits_df[splits_df["fold"] == fold_idx]
    role_files = {
        role: set(fold_splits.loc[fold_splits["role"] == role, "rel_path"])
        for role in ("train", "val", "test")
    }

    tr_mask = windows_meta_df["rel_path"].isin(role_files["train"])
    va_mask = windows_meta_df["rel_path"].isin(role_files["val"]) & (windows_meta_df["is_eval_grid"])
    te_mask = windows_meta_df["rel_path"].isin(role_files["test"]) & (windows_meta_df["is_eval_grid"])

    tr_df = windows_meta_df.loc[tr_mask]
    if max_neg_pos_ratio is not None:
        pos_idx = tr_df.loc[tr_df["label"] == 1, "h5_idx"].values
        neg_idx = tr_df.loc[tr_df["label"] == 0, "h5_idx"].values
        max_neg = min(len(neg_idx), len(pos_idx) * int(max_neg_pos_ratio))
        rng = np.random.default_rng(seed + fold_idx)
        sampled_neg = rng.choice(neg_idx, size=max_neg, replace=False) if max_neg > 0 else np.array([], dtype=np.int64)
        train_idx = np.sort(np.concatenate([pos_idx, sampled_neg])).astype(np.int64)
    else:
        train_idx = tr_df["h5_idx"].values.astype(np.int64)

    return {
        "train": train_idx,
        "val": windows_meta_df.loc[va_mask, "h5_idx"].values.astype(np.int64),
        "test": windows_meta_df.loc[te_mask, "h5_idx"].values.astype(np.int64),
    }