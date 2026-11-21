# src/data/manifest.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mne
import pandas as pd

# CHB-MIT clinical fact: chb21 is the same patient as chb01 (~1.5 years later)
CASE_TO_SUBJECT_OVERRIDE = {"chb21": "chb01"}


def discover_case_dirs(raw_data_dir: Path, selected_cases: list[str] | None = None) -> list[Path]:
    case_dirs = sorted(
        p for p in raw_data_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"chb\d{2}", p.name)
    )
    if selected_cases is not None:
        selected_set = set(selected_cases)
        case_dirs = [p for p in case_dirs if p.name in selected_set]
    if not case_dirs:
        raise FileNotFoundError(
            f"No matching CHB-MIT case folders found in {raw_data_dir} (selected={selected_cases})."
        )
    return case_dirs


def parse_summary_file(summary_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse a CHB-MIT *-summary.txt file into {edf_filename: [(start_sec, end_sec), ...]}."""
    seizures_by_file: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    pending_start: int | None = None

    for raw_line in summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_file = re.match(r"File Name:\s*(\S+\.edf)", line, flags=re.IGNORECASE)
        if m_file:
            current_file = m_file.group(1)
            seizures_by_file.setdefault(current_file, [])
            pending_start = None
            continue

        if current_file is None:
            continue

        m_start = re.match(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s*(\d+)\s*seconds?", line, flags=re.IGNORECASE)
        if m_start:
            pending_start = int(m_start.group(1))
            continue

        m_end = re.match(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s*(\d+)\s*seconds?", line, flags=re.IGNORECASE)
        if m_end and pending_start is not None:
            seizures_by_file[current_file].append((pending_start, int(m_end.group(1))))
            pending_start = None

    return seizures_by_file


def resolve_channel_mapping(raw_ch_names: list[str], standard_channels: list[str]) -> dict[str, str] | None:
    """
    Map each standard 18-channel bipolar label to the exact channel name inside an EDF.
    Handles MNE's duplicate suffixing (e.g., 'T8-P8-0' when EDF has two 'T8-P8' channels).
    Returns None if any required channel is missing.
    """
    upper_to_orig = {ch.upper(): ch for ch in raw_ch_names}
    mapping: dict[str, str] = {}
    for target in standard_channels:
        t_up = target.upper()
        if t_up in upper_to_orig:
            mapping[target] = upper_to_orig[t_up]
        elif f"{t_up}-0" in upper_to_orig:
            mapping[target] = upper_to_orig[f"{t_up}-0"]
        else:
            return None
    return mapping


def build_manifests(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir: Path = cfg["runtime"]["raw_data_dir"]
    out_dir: Path = cfg["runtime"]["processed_data_dir"]
    selected_cases = cfg["dataset"]["selected_cases"]
    standard_channels: list[str] = cfg["dataset"]["standard_18_channels"]

    case_dirs = discover_case_dirs(raw_dir, selected_cases)
    file_rows: list[dict[str, Any]] = []
    seizure_rows: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        case_id = case_dir.name
        subject_id = CASE_TO_SUBJECT_OVERRIDE.get(case_id, case_id)
        summary_files = list(case_dir.glob("*-summary.txt"))
        seizure_map = parse_summary_file(summary_files[0]) if summary_files else {}

        for edf_path in sorted(case_dir.glob("*.edf")):
            rel_path = f"{case_id}/{edf_path.name}"
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
                sfreq = float(raw.info["sfreq"])
                duration_sec = float(raw.n_times / sfreq)
                n_channels = len(raw.ch_names)
                ch_map = resolve_channel_mapping(raw.ch_names, standard_channels)
                has_std_18 = ch_map is not None
                read_ok = True
            except Exception:
                sfreq, duration_sec, n_channels, has_std_18, read_ok = 0.0, 0.0, 0, False, False

            seizures = seizure_map.get(edf_path.name, [])
            file_rows.append({
                "subject_id": subject_id,
                "case_id": case_id,
                "filename": edf_path.name,
                "rel_path": rel_path,
                "sfreq": sfreq,
                "duration_sec": round(duration_sec, 3),
                "n_channels": n_channels,
                "has_std_18": has_std_18,
                "read_ok": read_ok,
                "n_seizures": len(seizures),
                "has_seizure": len(seizures) > 0,
            })

            for s_idx, (start_s, end_s) in enumerate(seizures):
                seizure_rows.append({
                    "subject_id": subject_id,
                    "case_id": case_id,
                    "filename": edf_path.name,
                    "rel_path": rel_path,
                    "seizure_idx": s_idx,
                    "start_sec": float(start_s),
                    "end_sec": float(end_s),
                    "duration_sec": float(end_s - start_s),
                })

    files_df = pd.DataFrame(file_rows)
    seizures_df = pd.DataFrame(
        seizure_rows,
        columns=["subject_id", "case_id", "filename", "rel_path", "seizure_idx", "start_sec", "end_sec", "duration_sec"],
    )

    files_df.to_csv(out_dir / "files_manifest.csv", index=False)
    seizures_df.to_csv(out_dir / "seizures_manifest.csv", index=False)
    return files_df, seizures_df