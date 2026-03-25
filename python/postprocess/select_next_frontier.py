from pathlib import Path
import numpy as np
import pandas as pd


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


def _pairwise_euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def select_boundary_true_points(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    k_nearest_true_per_false: int = 3,
) -> pd.DataFrame:
    """
    Keep efficient=True points that lie closest to inefficient=False points.

    New logic:
    - for each false point, keep k nearest true points
    - then deduplicate them

    This keeps a thicker boundary and preserves a fuller grid.
    """

    if feature_cols is None:
        feature_cols = _get_feature_columns(df)

    if "candidate_efficient" not in df.columns:
        raise ValueError("Column 'candidate_efficient' not found.")

    eff = df["candidate_efficient"].astype(str).str.lower()
    df_true = df[eff == "true"].copy().reset_index(drop=False)
    df_false = df[eff == "false"].copy().reset_index(drop=False)

    if df_true.empty:
        raise ValueError("No candidate_efficient == true points found.")

    if df_false.empty:
        raise ValueError("No candidate_efficient == false points found.")

    x_true = df_true[feature_cols].to_numpy(dtype=float)
    x_false = df_false[feature_cols].to_numpy(dtype=float)

    dist_matrix = _pairwise_euclidean_distance(x_false, x_true)

    k = min(k_nearest_true_per_false, len(df_true))
    nearest_true_pos = np.argsort(dist_matrix, axis=1)[:, :k]

    selected_parts = []

    for false_idx in range(len(df_false)):
        true_positions = nearest_true_pos[false_idx]
        distances = dist_matrix[false_idx, true_positions]

        part = df_true.iloc[true_positions].copy()
        part["source_false_idx"] = false_idx
        part["min_distance_to_false"] = distances
        selected_parts.append(part)

    selected_true = pd.concat(selected_parts, ignore_index=True)

    selected_true = (
        selected_true
        .sort_values("min_distance_to_false", ascending=True)
        .drop_duplicates(subset=["index"])
        .drop(columns=["index", "source_false_idx"], errors="ignore")
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
    k_nearest_true_per_false: int = 3,
):
    df = pd.read_csv(input_csv)

    boundary_true_df = select_boundary_true_points(
        df=df,
        feature_cols=feature_cols,
        k_nearest_true_per_false=k_nearest_true_per_false,
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