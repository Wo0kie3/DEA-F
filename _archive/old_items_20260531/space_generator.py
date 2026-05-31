from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


# =========================================================
# 1. DATASET
# =========================================================

def build_airports_dataset() -> pd.DataFrame:
    alternative_names = [
        "WAW", "KRK", "KAT", "WRO", "POZ", "LCJ",
        "GDN", "SZZ", "BZG", "RZE", "IEG"
    ]

    inputs = np.array([
        [10.5, 36, 129.4, 7.0],
        [3.1, 19, 31.6, 7.9],
        [3.6, 32, 57.4, 10.5],
        [1.5, 12, 18.0, 3.0],
        [1.5, 10, 24.0, 4.0],
        [0.6, 12, 24.0, 3.9],
        [1.0, 15, 42.9, 2.5],
        [0.7, 10, 25.7, 1.9],
        [0.3, 6, 3.4, 1.2],
        [0.6, 6, 11.3, 2.7],
        [0.1, 10, 63.4, 3.0],
    ])

    outputs = np.array([
        [9.5, 129.7],
        [2.9, 31.3],
        [2.4, 21.1],
        [1.5, 18.8],
        [1.3, 16.2],
        [0.3, 4.2],
        [2.0, 23.6],
        [0.3, 4.2],
        [0.3, 6.2],
        [0.3, 3.5],
        [0.005, 0.61],
    ])

    input_cols = [f"i{k+1}" for k in range(inputs.shape[1])]
    output_cols = [f"o{k+1}" for k in range(outputs.shape[1])]

    df = pd.DataFrame(
        np.hstack([inputs, outputs]),
        columns=input_cols + output_cols
    )
    df.insert(0, "name", alternative_names)
    return df


# =========================================================
# 2. CONFIG
# =========================================================

ModeType = Literal["absolute", "percent"]
VarType = Literal["input", "output"]


@dataclass
class ColumnGridSpec:
    """
    Specification for a single variable that may be modified.

    mode="absolute":
        values are interpreted as deltas, e.g. [-0.2, 0.0, 0.2]

    mode="percent":
        values are interpreted as percentage changes, e.g. [-20, -10, 0, 10]
        meaning -20%, -10%, 0%, +10%
    """
    column: str
    var_type: VarType
    mode: ModeType
    values: list[float]
    clip_min: float = 0.0


@dataclass
class SearchSpaceConfig:
    dataset: pd.DataFrame
    target_name: str
    variable_specs: list[ColumnGridSpec]
    keep_original_columns: bool = True
    add_metadata_columns: bool = True
    sort_columns: bool = False


# =========================================================
# 3. VALIDATION
# =========================================================

def validate_dataset(df: pd.DataFrame) -> None:
    required = {"name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df["name"].duplicated().any():
        duplicated = df.loc[df["name"].duplicated(), "name"].tolist()
        raise ValueError(f"Duplicate alternative names found: {duplicated}")


def split_input_output_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    input_cols = [c for c in df.columns if c.startswith("i")]
    output_cols = [c for c in df.columns if c.startswith("o")]
    return input_cols, output_cols


def validate_config(config: SearchSpaceConfig) -> None:
    validate_dataset(config.dataset)

    if config.target_name not in set(config.dataset["name"]):
        raise ValueError(f"Target '{config.target_name}' not found in dataset.")

    input_cols, output_cols = split_input_output_columns(config.dataset)
    all_allowed = {"name", *input_cols, *output_cols}

    for spec in config.variable_specs:
        if spec.column not in all_allowed:
            raise ValueError(f"Column '{spec.column}' not found in dataset.")

        if spec.var_type == "input" and spec.column not in input_cols:
            raise ValueError(f"Column '{spec.column}' is not an input column.")

        if spec.var_type == "output" and spec.column not in output_cols:
            raise ValueError(f"Column '{spec.column}' is not an output column.")

        if spec.mode not in {"absolute", "percent"}:
            raise ValueError(f"Unsupported mode '{spec.mode}' for column '{spec.column}'.")

        if len(spec.values) == 0:
            raise ValueError(f"No grid values provided for column '{spec.column}'.")


# =========================================================
# 4. HELPERS
# =========================================================

ROUND_DECIMALS = 2

def frange(start: float, stop: float, step: float, decimals: int = ROUND_DECIMALS) -> list[float]:
    if step == 0:
        raise ValueError("step cannot be 0")

    values = []
    x = start
    eps = abs(step) / 1_000_000

    if step > 0:
        while x <= stop + eps:
            values.append(round(x, decimals))
            x += step
    else:
        while x >= stop - eps:
            values.append(round(x, decimals))
            x += step

    return values


def apply_change(base_value: float, mode: ModeType, change_value: float, clip_min: float = 0.0) -> float:
    if mode == "absolute":
        new_value = base_value + change_value
    elif mode == "percent":
        new_value = base_value * (1.0 + change_value / 100.0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    new_value = max(clip_min, new_value)
    return round(new_value, ROUND_DECIMALS)


def get_target_row(df: pd.DataFrame, target_name: str) -> pd.Series:
    row = df.loc[df["name"] == target_name]
    if row.empty:
        raise ValueError(f"Target '{target_name}' not found.")
    return row.iloc[0].copy()


# =========================================================
# 5. CORE GENERATOR
# =========================================================

def generate_search_space(config: SearchSpaceConfig) -> pd.DataFrame:
    validate_config(config)

    df = config.dataset.copy()
    target_row = get_target_row(df, config.target_name)

    specs = config.variable_specs
    grid_lists = [spec.values for spec in specs]

    combinations = list(product(*grid_lists))
    results = []

    for combo_idx, combo in enumerate(combinations, start=1):
        candidate = target_row.copy()

        change_map = {}
        for spec, change_value in zip(specs, combo):
            base_value = float(target_row[spec.column])
            new_value = apply_change(
                base_value=base_value,
                mode=spec.mode,
                change_value=change_value,
                clip_min=spec.clip_min,
            )
            candidate[spec.column] = new_value
            change_map[f"{spec.column}_change"] = change_value

        row_dict = {}

        if config.add_metadata_columns:
            row_dict["candidate_id"] = combo_idx
            row_dict["base_name"] = config.target_name

        row_dict["name"] = f"{config.target_name}_cand_{combo_idx}"

        if config.keep_original_columns:
            for col in df.columns:
                if col == "name":
                    continue
                row_dict[col] = candidate[col]

        if config.add_metadata_columns:
            for spec in specs:
                base_val = float(target_row[spec.column])
                new_val = float(candidate[spec.column])
                row_dict[f"{spec.column}_base"] = base_val
                row_dict[f"{spec.column}_new"] = new_val
                row_dict[f"{spec.column}_delta_abs"] = round(new_val - base_val, ROUND_DECIMALS)

                if abs(base_val) > 1e-12:
                    row_dict[f"{spec.column}_delta_pct"] = round(
                    100.0 * (new_val - base_val) / base_val,
                    ROUND_DECIMALS
                )
                else:
                    row_dict[f"{spec.column}_delta_pct"] = np.nan

                row_dict[f"{spec.column}_grid_value"] = change_map[f"{spec.column}_change"]
                row_dict[f"{spec.column}_grid_mode"] = spec.mode

        results.append(row_dict)

    result_df = pd.DataFrame(results)

    if config.sort_columns:
        result_df = result_df.reindex(sorted(result_df.columns), axis=1)

    return result_df


# =========================================================
# 6. OPTIONAL: GRID BUILDERS BASED ON DATASET MAX
# =========================================================

def build_percent_grid_from_max(
    df: pd.DataFrame,
    column: str,
    step_pct_of_max: float,
    min_pct_of_max: float,
    max_pct_of_max: float,
) -> list[float]:
    """
    Helper only for convenience if you want grid ranges based on dataset max.
    Returns ABSOLUTE deltas, not percent deltas.

    Example:
        if max in i1 is 10.5 and you want from -20% of max to +20% of max
        with step 5% of max:
            -> [-2.1, -1.575, -1.05, -0.525, 0, 0.525, 1.05, 1.575, 2.1]
    """
    col_max = float(df[column].max())
    start = col_max * (min_pct_of_max / 100.0)
    stop = col_max * (max_pct_of_max / 100.0)
    step = col_max * (step_pct_of_max / 100.0)
    return frange(start, stop, step)


def build_percent_change_grid(start_pct: float, stop_pct: float, step_pct: float) -> list[float]:
    """
    Returns percentage changes directly, e.g. [-20, -10, 0, 10, 20]
    """
    return frange(start_pct, stop_pct, step_pct)


# =========================================================
# 7. CSV EXPORT
# =========================================================

def export_search_space_to_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# =========================================================
# 8. EXAMPLE USAGE
# =========================================================

def main() -> None:
    # Build dataset
    df = build_airports_dataset()

    # Example:
    # We want to modify RZE
    # - decrease i1 by 0.0 to 0.5 in step 0.1
    # - decrease i4 by 0.0 to 1.0 in step 0.2
    # - increase o2 by 0.0 to 2.0 in step 0.5
    #
    # Interpretation:
    # inputs:
    #   negative absolute delta means improvement (lower input)
    # outputs:
    #   positive absolute delta means improvement (higher output)

    config = SearchSpaceConfig(
        dataset=df,
        target_name="RZE",
        variable_specs=[
            ColumnGridSpec(
                column="i1",
                var_type="input",
                mode="absolute",
                values=frange(-1.5, 0.3, 0.1),   # lower i1
                clip_min=0.0
            ),
            ColumnGridSpec(
                column="i4",
                var_type="input",
                mode="absolute",
                values=frange(-2.0, 0.4, 0.1),   # lower i4
                clip_min=0.0
            ),
            ColumnGridSpec(
                column="o2",
                var_type="output",
                mode="absolute",
                values=frange(-1.0, 5.0, 0.2),    # increase o2
                clip_min=0.0
            ),
        ],
        keep_original_columns=True,
        add_metadata_columns=True,
        sort_columns=False
    )

    search_df = generate_search_space(config)

    print("Generated candidates:", len(search_df))
    print(search_df.head())

    output_file = export_search_space_to_csv(search_df, "output/dea_search_space_RZE.csv")
    print(f"CSV saved to: {output_file}")


if __name__ == "__main__":
    main()