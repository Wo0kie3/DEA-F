import pandas as pd
from pathlib import Path
from itertools import product


def _get_feature_columns(df: pd.DataFrame):
    inputs = sorted(
        [c for c in df.columns if c.startswith("i")],
        key=lambda x: int(x[1:])
    )
    outputs = sorted(
        [c for c in df.columns if c.startswith("o")],
        key=lambda x: int(x[1:])
    )
    return inputs + outputs


def _get_efficiency_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.endswith("_efficiency")]


def annotate_boundary_neighbors(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Annotate all evaluated grid points with boundary-neighborhood metadata.

    Directional neighborhood follows the "worse-than-current" side:
    - inputs:  0 or +1 grid step
    - outputs: 0 or -1 grid step

    For i1, i2, o1 we check combinations such as:
    (+1, 0, 0), (0, +1, 0), (0, 0, -1),
    (+1, +1, 0), (+1, 0, -1), (0, +1, -1), (+1, +1, -1)
    """

    if feature_cols is None:
        feature_cols = _get_feature_columns(df)

    if not feature_cols:
        raise ValueError("feature_cols is empty.")

    if "candidate_efficient" not in df.columns:
        raise ValueError("Column 'candidate_efficient' not found.")

    eff = df["candidate_efficient"].astype(str).str.lower()
    df_true = df[eff == "true"].copy().reset_index(drop=False)
    df_false = df[eff == "false"].copy().reset_index(drop=False)

    if df_true.empty:
        raise ValueError("No candidate_efficient == true points found.")

    if df_false.empty:
        raise ValueError("No candidate_efficient == false points found.")

    rounding_digits = 10

    level_maps = {}
    for col in feature_cols:
        unique_values = sorted({round(float(v), rounding_digits) for v in df[col].tolist()})
        if not unique_values:
            raise ValueError(f"No grid levels found for column '{col}'.")
        level_maps[col] = {value: idx for idx, value in enumerate(unique_values)}

    def _grid_index_key(row: pd.Series) -> tuple[int, ...]:
        return tuple(
            level_maps[col][round(float(row[col]), rounding_digits)]
            for col in feature_cols
        )

    false_index_keys = {
        _grid_index_key(row)
        for _, row in df_false.iterrows()
    }

    offsets_per_dim = []
    for col in feature_cols:
        if col.startswith("i"):
            offsets_per_dim.append((0, 1))
        elif col.startswith("o"):
            offsets_per_dim.append((0, -1))
        else:
            raise ValueError(f"Column '{col}' is neither input nor output.")

    out = df.copy()
    out["boundary_has_false_neighbor"] = False
    out["boundary_false_neighbor_count"] = 0

    for row_idx, row in df.iterrows():
        if str(row["candidate_efficient"]).lower() != "true":
            continue

        base_index_key = _grid_index_key(row)
        false_neighbor_count = 0

        for offset_combo in product(*offsets_per_dim):
            if all(offset == 0 for offset in offset_combo):
                continue

            neighbor_index_key = tuple(
                base_idx + offset
                for base_idx, offset in zip(base_index_key, offset_combo)
            )

            if neighbor_index_key in false_index_keys:
                false_neighbor_count += 1

        if false_neighbor_count > 0:
            out.at[row_idx, "boundary_has_false_neighbor"] = True
            out.at[row_idx, "boundary_false_neighbor_count"] = false_neighbor_count

    return out


def select_boundary_true_points(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    annotated = annotate_boundary_neighbors(
        df=df,
        feature_cols=feature_cols,
    )

    selected_true = (
        annotated[
            (annotated["candidate_efficient"].astype(str).str.lower() == "true")
            & (annotated["boundary_has_false_neighbor"])
        ]
        .copy()
        .reset_index(drop=True)
    )

    return selected_true


def select_best_candidate_by_efficiency_sum(df: pd.DataFrame) -> pd.Series:
    eff_cols = _get_efficiency_columns(df)

    if not eff_cols:
        raise ValueError("No efficiency columns found.")

    result = df.copy()
    result["efficiency_sum"] = result[eff_cols].sum(axis=1)
    result = result.sort_values("efficiency_sum", ascending=False).reset_index(drop=True)

    return result.iloc[0].copy()


def process_dea_results(
    input_csv: str,
    boundary_output_csv: str,
    best_output_csv: str,
    feature_cols: list[str] | None = None,
):
    df = pd.read_csv(input_csv)

    boundary_true_df = select_boundary_true_points(
        df=df,
        feature_cols=feature_cols,
    )

    best_point = select_best_candidate_by_efficiency_sum(boundary_true_df)

    Path(boundary_output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(best_output_csv).parent.mkdir(parents=True, exist_ok=True)

    boundary_true_df.to_csv(boundary_output_csv, index=False)
    pd.DataFrame([best_point]).to_csv(best_output_csv, index=False)

    print(f"Saved boundary true points: {boundary_output_csv}")
    print(f"Boundary true count: {len(boundary_true_df)}")
    print(f"Saved best point: {best_output_csv}")
    print(f"Best point name: {best_point.get('name', '<no name>')}")
    print(f"Best efficiency_sum: {best_point['efficiency_sum']:.10f}")
