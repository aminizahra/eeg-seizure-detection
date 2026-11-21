# src/features/bandpower.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from tqdm import tqdm

EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def build_feature_names(channels: list[str]) -> list[str]:
    names: list[str] = []
    for ch in channels:
        for band in EEG_BANDS:
            names.append(f"{ch}_logpow_{band}")
        for band in EEG_BANDS:
            names.append(f"{ch}_relpow_{band}")
        names.append(f"{ch}_line_length")
        names.append(f"{ch}_rms")
    return names


def extract_bandpower_batch(windows: np.ndarray, fs: int) -> np.ndarray:
    """
    Vectorized feature extraction for a batch of windows.
    Input : windows of shape (B, C, T)
    Output: features of shape (B, C * 12) as float32
    """
    x = windows.astype(np.float32, copy=False)
    nperseg = min(fs, x.shape[-1])
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, axis=-1)  # (B, C, F)

    total_mask = (freqs >= 0.5) & (freqs <= 40.0)
    total_pow = trapezoid(psd[..., total_mask], freqs[total_mask], axis=-1) + 1e-12  # (B, C)

    log_band_pows: list[np.ndarray] = []
    rel_band_pows: list[np.ndarray] = []
    for f_lo, f_hi in EEG_BANDS.values():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        bp = trapezoid(psd[..., mask], freqs[mask], axis=-1)  # (B, C)
        log_band_pows.append(np.log10(bp + 1e-12))
        rel_band_pows.append(bp / total_pow)

    line_length = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)  # (B, C)
    rms = np.sqrt(np.mean(x ** 2, axis=-1))                      # (B, C)

    # Stack per-channel features -> (B, C, 12) -> flatten to (B, C * 12)
    per_ch = np.stack([*log_band_pows, *rel_band_pows, line_length, rms], axis=-1)
    return per_ch.reshape(x.shape[0], -1).astype(np.float32)


def extract_or_load_xgb_features(
    cfg: dict[str, Any],
    n_expected_windows: int,
    force: bool = False,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    out_dir: Path = cfg["runtime"]["processed_data_dir"]
    fs = int(cfg["dataset"]["fs"])
    channels: list[str] = cfg["dataset"]["standard_18_channels"]

    feat_path = out_dir / "xgb_features.npy"
    names_path = out_dir / "xgb_feature_names.json"
    hdf5_path = out_dir / "eeg_windows.h5"

    if not force and feat_path.exists() and names_path.exists():
        X = np.load(feat_path)
        feature_names = json.loads(names_path.read_text(encoding="utf-8"))
        if X.shape[0] == n_expected_windows and X.shape[1] == len(feature_names):
            print(f"[SKIP] Loaded cached {feat_path.name} with shape {X.shape}.")
            return X, feature_names
        print("[REBUILD] Cached feature shape mismatch; recomputing...")

    feature_names = build_feature_names(channels)
    with h5py.File(hdf5_path, "r") as h5f:
        dset = h5f["windows"]
        n_total = dset.shape[0]
        X = np.empty((n_total, len(feature_names)), dtype=np.float32)

        for start in tqdm(range(0, n_total, chunk_size), desc="Extracting Welch + LineLength"):
            end = min(start + chunk_size, n_total)
            X[start:end] = extract_bandpower_batch(dset[start:end], fs=fs)

    np.save(feat_path, X)
    names_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    return X, feature_names