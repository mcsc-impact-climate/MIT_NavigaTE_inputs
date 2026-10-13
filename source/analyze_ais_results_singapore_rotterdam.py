"""
Date: 260228
Author: danikae
Purpose: Analyze AIS-derived Singapore-Rotterdam route activity and estimate
full-time-equivalent vessels on the Suez route.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


AIS_START = pd.Timestamp("2023-01-28")
AIS_END = pd.Timestamp("2024-12-14")
SUEZ_DISTANCE_NM = 8440.0

REQUIRED_COLUMNS = [
    "ship_class",
    "size_class",
    "voyages_from_suez",
    "mean_speed",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _observation_days(start: pd.Timestamp = AIS_START, end: pd.Timestamp = AIS_END) -> int:
    return int((end - start).days + 1)


def _parse_numeric_size(size_label: str) -> float:
    match = re.search(r"([\d,]+(?:\.\d+)?)", str(size_label))
    if match is None:
        return np.inf
    return float(match.group(1).replace(",", ""))


def _candidate_input_files(root: Path) -> list[Path]:
    from_dir = sorted((root / "data" / "NavigaTE_data" / "xlsx").glob("*.xlsx"))
    from_dir = [path for path in from_dir if not path.name.startswith("~$")]
    if from_dir:
        return from_dir

    single_file = root / "data" / "NavigaTE_data.xlsx"
    if single_file.exists():
        return [single_file]

    raise FileNotFoundError(
        "No input file found. Expected either data/NavigaTE_data/xlsx/*.xlsx "
        "or data/NavigaTE_data.xlsx"
    )


def load_classification_data(root: Path | None = None) -> pd.DataFrame:
    base = root or _repo_root()
    files = _candidate_input_files(base)

    all_frames: list[pd.DataFrame] = []
    for workbook in files:
        try:
            frame = pd.read_excel(workbook, sheet_name="classification")
        except ValueError:
            continue

        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {workbook}")

        subset = frame[REQUIRED_COLUMNS].copy()
        subset["source_file"] = workbook.name
        all_frames.append(subset)

    if not all_frames:
        raise ValueError("No classification sheet found in the provided input workbooks.")

    data = pd.concat(all_frames, ignore_index=True)
    data["ship_class"] = data["ship_class"].astype(str).str.strip()
    data["size_class"] = data["size_class"].astype(str).str.strip()

    for numeric_column in ["voyages_from_suez", "mean_speed"]:
        data[numeric_column] = (
            data[numeric_column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"": np.nan, "nan": np.nan})
            .astype(float)
        )

    data = data.dropna(subset=["ship_class", "size_class", "voyages_from_suez", "mean_speed"])
    data = data[(data["voyages_from_suez"] > 0) & (data["mean_speed"] > 0)].copy()

    if data.empty:
        raise ValueError("No valid rows left after filtering for positive voyages and speed.")

    return data


def estimate_effective_vessels(
    data: pd.DataFrame,
    suez_distance_nm: float = SUEZ_DISTANCE_NM,
    observation_days: int | None = None,
) -> pd.DataFrame:
    window_days = observation_days or _observation_days()
    result = data.copy()

    result["one_way_days_via_suez"] = suez_distance_nm / (result["mean_speed"] * 24.0)
    result["effective_vessels"] = result["voyages_from_suez"] * result["one_way_days_via_suez"] / window_days

    return result


def summarize_by_size_class(estimates: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        estimates.groupby(["ship_class", "size_class"], as_index=False)
        .agg(
            voyages_from_suez=("voyages_from_suez", "sum"),
            effective_vessels=("effective_vessels", "sum"),
            mean_speed_knots=("mean_speed", "mean"),
        )
        .sort_values(["ship_class", "size_class"], kind="stable")
    )

    totals = (
        grouped.groupby("ship_class", as_index=False)["effective_vessels"]
        .sum()
        .rename(columns={"effective_vessels": "effective_vessels_ship_class_total"})
    )
    grouped = grouped.merge(totals, on="ship_class", how="left")
    grouped["fraction_within_ship_class"] = (
        grouped["effective_vessels"] / grouped["effective_vessels_ship_class_total"]
    )

    grouped["size_sort_key"] = grouped["size_class"].apply(_parse_numeric_size)
    grouped = grouped.sort_values(["ship_class", "size_sort_key", "size_class"]).drop(columns=["size_sort_key"])

    return grouped


def summarize_by_ship_class(size_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        size_summary.groupby("ship_class", as_index=False)
        .agg(
            voyages_from_suez=("voyages_from_suez", "sum"),
            effective_vessels=("effective_vessels", "sum"),
        )
        .sort_values("effective_vessels", ascending=False)
    )


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _repo_root()
    raw = load_classification_data(root=root)
    estimates = estimate_effective_vessels(raw)
    by_size = summarize_by_size_class(estimates)
    by_class = summarize_by_ship_class(by_size)

    output_dir = root / "processed_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_size_path = output_dir / "ais_suez_effective_vessels_by_size_class.csv"
    by_class_path = output_dir / "ais_suez_effective_vessels_by_ship_class.csv"
    by_size.to_csv(by_size_path, index=False)
    by_class.to_csv(by_class_path, index=False)

    print(f"AIS window: {AIS_START.date()} to {AIS_END.date()} ({_observation_days()} days)")
    print(f"Assumed one-way Suez route distance: {SUEZ_DISTANCE_NM:,.0f} nm")
    print(f"Saved size-class results to: {by_size_path}")
    print(f"Saved ship-class results to: {by_class_path}\n")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("Effective vessels by ship class:")
    print(by_class.to_string(index=False))
    print("\nSize-class fractional composition within each ship class:")
    print(
        by_size[
            [
                "ship_class",
                "size_class",
                "voyages_from_suez",
                "mean_speed_knots",
                "effective_vessels",
                "fraction_within_ship_class",
            ]
        ].to_string(index=False)
    )

    return by_size, by_class


if __name__ == "__main__":
    run_analysis()
