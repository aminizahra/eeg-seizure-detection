# src/data/preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.manifest import resolve_channel_mapping


def _compute_seizure_overlap(win_start: float, win_end: float, intervals: list[tuple[float, float]]) -> float:
    win_len = win_end - win_start
    if win_len <= 0 or not intervals:
        return 0.0
    overlap = 0.0
    for s_start, s_end in intervals:
        ov = max(0.0, min(win_end, s_end) - max(win_start, s_start))
        overlap += ov
    return overlap / win_len


def _preprocess_single_edf(
    edf_path: Path,
    standard_channels: list[str],
    fs: int,
    notch_freqs: list[float],
    bandpass_low: float,
    bandpass_high: float,
    clip_zscore: float,
) -> np.ndarray:
    """Load, harmonize 18 channels, filter, and apply per-channel robust Z-normalization."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    ch_map = resolve_channel_mapping(raw.ch_names, standard_channels)
    if ch_map is None:
        raise ValueError(f"Missing standard 18 channels in {edf_path.name}")

    ordered_orig = [ch_map[ch] for ch in standard_channels]
    raw.pick(ordered_orig)
    raw.reorder_channels(ordered_orig)

    if abs(float(raw.info["sfreq"]) - fs) > 1e-3:
        raw.resample(fs, verbose="ERROR")

    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs, verbose="ERROR")
    raw.filter(l_freq=bandpass_low, h_freq=bandpass_high, verbose="ERROR")

    data = raw.get_data().astype(np.float32)  # (18, N_samples)
    medians = np.median(data, axis=1, keepdims=True)
    mads = np.median(np.abs(data - medians), axis=1, keepdims=True)
    scales = np.maximum(1.4826 * mads, 1e-6)
    data = np.clip((data - medians) / scales, -clip_zscore, clip_zscore)
    return data


def build_hdf5_windows(
    files_df: pd.DataFrame,
    seizures_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    """Stream-process all valid EDFs into eeg_windows.h5 and windows_metadata.csv."""
    raw_dir: Path = cfg["runtime"]["raw_data_dir"]
    out_dir: Path = cfg["runtime"]["processed_data_dir"]
    ds_cfg = cfg["dataset"]
    pp_cfg = cfg["preprocessing"]

    fs = int(ds_cfg["fs"])
    win_sec = float(ds_cfg["window_sec"])
    eval_step_sec = float(ds_cfg["eval_step_sec"])
    ictal_step_sec = float(ds_cfg["train_ictal_step_sec"])
    ov_thresh = float(ds_cfg["seizure_overlap_thresh"])
    std_chs: list[str] = ds_cfg["standard_18_channels"]
    storage_dtype = np.float16 if pp_cfg["storage_dtype"] == "float16" else np.float32

    win_samples = int(round(win_sec * fs))
    eval_step_samples = int(round(eval_step_sec * fs))
    ictal_step_samples = int(round(ictal_step_sec * fs))

    valid_files = files_df[(files_df["read_ok"]) & (files_df["has_std_18"])].reset_index(drop=True)
    seiz_lookup: dict[str, list[tuple[float, float]]] = {}
    for rel_path, grp in seizures_df.groupby("rel_path"):
        seiz_lookup[str(rel_path)] = list(zip(grp["start_sec"].astype(float), grp["end_sec"].astype(float)))

    hdf5_path = out_dir / "eeg_windows.h5"
    meta_rows: list[dict[str, Any]] = []
    h5_cursor = 0

    with h5py.File(hdf5_path, "w") as h5f:
        dset = h5f.create_dataset(
            "windows",
            shape=(0, len(std_chs), win_samples),
            maxshape=(None, len(std_chs), win_samples),
            chunks=(64, len(std_chs), win_samples),
            dtype=storage_dtype,
        )

        for row in tqdm(valid_files.itertuples(index=False), total=len(valid_files), desc="Preprocessing EDFs"):
            edf_path = raw_dir / row.rel_path
            sig = _preprocess_single_edf(
                edf_path=edf_path,
                standard_channels=std_chs,
                fs=fs,
                notch_freqs=pp_cfg["notch_freqs"],
                bandpass_low=float(pp_cfg["bandpass_low"]),
                bandpass_high=float(pp_cfg["bandpass_high"]),
                clip_zscore=float(pp_cfg["clip_zscore"]),
            )

            n_total = sig.shape[1]
            intervals = seiz_lookup.get(row.rel_path, [])

            # 1. Non-overlapping evaluation grid
            eval_starts = list(range(0, n_total - win_samples + 1, eval_step_samples))
            seen_starts = set(eval_starts)
            window_specs: list[tuple[int, bool, int]] = []

            for s in eval_starts:
                t0, t1 = s / fs, (s + win_samples) / fs
                lbl = int(_compute_seizure_overlap(t0, t1, intervals) >= ov_thresh)
                window_specs.append((s, True, lbl))

            # 2. Dense 1-second ictal grid around seizures (used strictly in training!)
            for s_start, s_end in intervals:
                dense_lo = max(0, int((s_start - win_sec) * fs))
                dense_hi = min(n_total - win_samples, int(s_end * fs))
                for s in range(dense_lo, dense_hi + 1, ictal_step_samples):
                    if s in seen_starts:
                        continue
                    t0, t1 = s / fs, (s + win_samples) / fs
                    if _compute_seizure_overlap(t0, t1, intervals) >= ov_thresh:
                        seen_starts.add(s)
                        window_specs.append((s, False, 1))

            window_specs.sort(key=lambda item: item[0])
            if not window_specs:
                continue

            batch = np.stack([sig[:, s : s + win_samples] for s, _, _ in window_specs], axis=0).astype(storage_dtype)
            n_new = batch.shape[0]
            dset.resize(h5_cursor + n_new, axis=0)
            dset[h5_cursor : h5_cursor + n_new] = batch

            for offset, (s, is_eval, lbl) in enumerate(window_specs):
                meta_rows.append({
                    "h5_idx": h5_cursor + offset,
                    "subject_id": row.subject_id,
                    "case_id": row.case_id,
                    "filename": row.filename,
                    "rel_path": row.rel_path,
                    "window_start_sec": round(s / fs, 3),
                    "window_end_sec": round((s + win_samples) / fs, 3),
                    "label": lbl,
                    "is_eval_grid": is_eval,
                })
            h5_cursor += n_new

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "windows_metadata.csv", index=False)
    return hdf5_path, meta_df