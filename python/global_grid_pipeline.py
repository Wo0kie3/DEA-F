import argparse
import csv
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from java_runner import (
    generate_frontiers_with_java,
    evaluate_candidates_with_java,
)
from postprocess.select_next_frontier import select_boundary_true_points


# =========================================================
# ARGPARSE
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--columns", required=True)

    parser.add_argument("--frontiers-output", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--java-entry", required=True)
    parser.add_argument(
        "--frontier-main-class",
        default="org.example.DeaFrontierLayersExporter",
    )
    parser.add_argument(
        "--evaluator-main-class",
        default="org.example.CsvFrontierCandidateEvaluator",
    )
    parser.add_argument("--maven-executable", default="mvn")

    parser.add_argument("--start-front", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    # Global grid params
    parser.add_argument("--pct-above", type=float, default=30.0)
    parser.add_argument("--step-pct", type=float, default=5.0)
    parser.add_argument("--min-points-per-dim", type=int, default=10)

    # Postprocess
    parser.add_argument("--boundary-k", type=int, default=5)
    parser.add_argument("--points-per-front", type=int, default=100)

    # Plot axes
    parser.add_argument("--plot-x", required=True)
    parser.add_argument("--plot-y", required=True)
    parser.add_argument("--plot-z", required=True)

    return parser.parse_args()


# =========================================================
# PATH / UTILS
# =========================================================

def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def path_for_java(path_str: str, java_entry: str) -> str:
    abs_target = Path(path_str).resolve()
    abs_java = Path(java_entry).resolve()
    return os.path.relpath(abs_target, start=abs_java)


def get_frontier_layer_for_name(df_frontiers: pd.DataFrame, name: str) -> int:
    match = df_frontiers[df_frontiers["name"] == name]
    if match.empty:
        raise ValueError(f"Target '{name}' not found in frontiers file.")
    return int(match.iloc[0]["frontier_layer"])


def get_max_frontier_layer(df_frontiers: pd.DataFrame) -> int:
    return int(df_frontiers["frontier_layer"].max())


def get_io_columns(df: pd.DataFrame):
    inputs = sorted(
        [c for c in df.columns if c.startswith("i") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    outputs = sorted(
        [c for c in df.columns if c.startswith("o") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    return inputs, outputs


def get_efficiency_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.endswith("_efficiency")]


def _make_grid_values(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"Step must be positive, got {step}")

    lo = min(start, end)
    hi = max(start, end)

    values = np.arange(lo, hi + step, step, dtype=float)

    if values.size == 0:
        values = np.array([lo, hi], dtype=float)

    values = np.unique(values.astype(float))
    values.sort()
    return values


def _ensure_min_points(start: float, end: float, values: np.ndarray, min_points: int) -> np.ndarray:
    if min_points is None or min_points <= 1:
        return values

    lo = min(start, end)
    hi = max(start, end)

    if np.isclose(lo, hi):
        return np.array([lo], dtype=float)

    if len(values) >= min_points:
        return values

    return np.linspace(lo, hi, num=min_points, dtype=float)


# =========================================================
# GLOBAL GRID GENERATION
# =========================================================

def generate_global_samples(
    df: pd.DataFrame,
    target_row: pd.Series,
    columns_to_modify: list[str],
    pct_above: float = 30.0,
    step_pct: float = 5.0,
    min_points_per_dim: int = 10,
) -> pd.DataFrame:
    """
    Build ONE global grid from the original start point to the full-data extrema.

    DEA-consistent direction:
    - inputs: minimize -> from target towards GLOBAL MIN
    - outputs: maximize -> from target towards GLOBAL MAX
    """
    inputs, outputs = get_io_columns(df)
    io_cols = inputs + outputs

    missing = [c for c in columns_to_modify if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    grid = {}

    for col in columns_to_modify:
        t = float(target_row[col])
        col_values = df[col].astype(float)

        global_min = float(col_values.min())
        global_max = float(col_values.max())

        if col.startswith("i"):
            # Inputy minimalizujemy
            start = t
            end = global_min * (1.0 - pct_above / 100.0)

        elif col.startswith("o"):
            # Outputy maksymalizujemy
            start = t
            end = global_max * (1.0 + pct_above / 100.0)

        else:
            raise ValueError(f"Column '{col}' is neither input nor output.")

        span = abs(end - start)
        step = max(span * (step_pct / 100.0), 1e-9)

        values = _make_grid_values(start, end, step)
        values = _ensure_min_points(start, end, values, min_points_per_dim)

        if values.size == 0:
            raise ValueError(
                f"Empty global grid for column '{col}'. "
                f"target={t}, global_min={global_min}, global_max={global_max}, "
                f"start={start}, end={end}, span={span}, step={step}"
            )

        grid[col] = values

        print(
            f"[GLOBAL GRID] {col}: "
            f"target={t:.6f}, "
            f"global_min={global_min:.6f}, "
            f"global_max={global_max:.6f}, "
            f"start={start:.6f}, "
            f"end={end:.6f}, "
            f"span={span:.6f}, "
            f"step={step:.6f}, "
            f"points={len(values)}, "
            f"grid_lo={values.min():.6f}, "
            f"grid_hi={values.max():.6f}"
        )

    combos = list(product(*grid.values()))
    if not combos:
        raise ValueError("No candidate combinations generated for global grid.")

    base_name = str(target_row["name"])
    rows = []

    for i, combo in enumerate(combos):
        row = {"name": f"{base_name}_global_cand_{i}"}

        # wszystkie IO startowo jak w target
        for c in io_cols:
            row[c] = target_row[c]

        # tylko wybrane kolumny podmieniane gridem
        for c, v in zip(columns_to_modify, combo):
            row[c] = v

        rows.append(row)

    result_df = pd.DataFrame(rows)
    print(f"[GLOBAL GRID] TOTAL CANDIDATES: {len(result_df)}")
    return result_df


# =========================================================
# RANKING / SELECTION
# =========================================================

def rank_and_limit_boundary_points(df_boundary: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df_boundary.empty:
        return df_boundary.copy()

    eff_cols = get_efficiency_columns(df_boundary)
    if not eff_cols:
        raise ValueError("No efficiency columns found in boundary dataframe.")

    out = df_boundary.copy()
    out["efficiency_sum"] = out[eff_cols].sum(axis=1)

    out = out.sort_values(
        by=["efficiency_sum", "candidate_efficiency"],
        ascending=[False, False]
    ).reset_index(drop=True)

    out["selected_rank"] = range(1, len(out) + 1)
    out = out.head(limit).copy().reset_index(drop=True)
    return out


# =========================================================
# PLOTS
# =========================================================

def save_all_results_plot(results_csv: str, output_html: str, x: str, y: str, z: str, title: str):
    df = pd.read_csv(results_csv)

    if "candidate_efficient" in df.columns:
        df["candidate_efficient"] = df["candidate_efficient"].astype(str).str.lower()

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="candidate_efficient",
        hover_data=["name", "candidate_efficiency", "candidate_efficient"],
        title=title,
    )

    ensure_parent_dir(output_html)
    fig.write_html(output_html)
    print(f"Saved plot: {output_html}")


def save_selected_points_plot(selected_csv: str, output_html: str, x: str, y: str, z: str, title: str):
    df = pd.read_csv(selected_csv)

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="reference_frontier",
        hover_data=["name", "candidate_efficiency", "efficiency_sum", "selected_rank"],
        title=title,
    )

    ensure_parent_dir(output_html)
    fig.write_html(output_html)
    print(f"Saved plot: {output_html}")


# =========================================================
# PATH EXPORT
# =========================================================

def save_paths_cartesian_product(
    layer_point_dfs: list[pd.DataFrame],
    output_csv: str,
    start_row: pd.Series,
    io_cols: list[str],
):
    """
    Build all possible paths:
    start x layer1 x layer2 x ... x layerN
    """
    ensure_parent_dir(output_csv)

    if not layer_point_dfs:
        raise ValueError("No layer point dataframes provided for path export.")

    for idx, df in enumerate(layer_point_dfs, start=1):
        if df.empty:
            raise ValueError(f"Layer dataframe #{idx} is empty. Cannot build paths.")

    base_cols = [
        "path_id",
        "start_name",
        "start_frontier_layer",
    ] + [f"start_{c}" for c in io_cols]

    dynamic_cols = []
    for step_idx, _df_layer in enumerate(layer_point_dfs, start=1):
        dynamic_cols.extend([
            f"step_{step_idx:02d}_iteration",
            f"step_{step_idx:02d}_frontier",
            f"step_{step_idx:02d}_selected_rank",
            f"step_{step_idx:02d}_point_name",
            f"step_{step_idx:02d}_candidate_efficiency",
            f"step_{step_idx:02d}_candidate_efficient",
            f"step_{step_idx:02d}_efficiency_sum",
        ])
        dynamic_cols.extend([f"step_{step_idx:02d}_{c}" for c in io_cols])

    fieldnames = base_cols + dynamic_cols

    index_ranges = [range(len(df)) for df in layer_point_dfs]
    total_paths = 1
    for df in layer_point_dfs:
        total_paths *= len(df)

    print("=" * 80)
    print("Generating full Cartesian product of paths...")
    print(f"Number of layers: {len(layer_point_dfs)}")
    print(f"Total paths to write: {total_paths}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for path_idx, index_combo in enumerate(product(*index_ranges), start=1):
            row_out = {
                "path_id": f"path_{path_idx:012d}",
                "start_name": start_row["name"],
                "start_frontier_layer": int(start_row["frontier_layer"]),
            }

            for c in io_cols:
                row_out[f"start_{c}"] = start_row[c]

            for step_idx, (df_layer, row_idx) in enumerate(zip(layer_point_dfs, index_combo), start=1):
                r = df_layer.iloc[row_idx]

                row_out[f"step_{step_idx:02d}_iteration"] = int(r["iteration"])
                row_out[f"step_{step_idx:02d}_frontier"] = int(r["reference_frontier"])
                row_out[f"step_{step_idx:02d}_selected_rank"] = int(r["selected_rank"])
                row_out[f"step_{step_idx:02d}_point_name"] = r["name"]
                row_out[f"step_{step_idx:02d}_candidate_efficiency"] = r.get("candidate_efficiency")
                row_out[f"step_{step_idx:02d}_candidate_efficient"] = r.get("candidate_efficient")
                row_out[f"step_{step_idx:02d}_efficiency_sum"] = r.get("efficiency_sum")

                for c in io_cols:
                    row_out[f"step_{step_idx:02d}_{c}"] = r[c]

            writer.writerow(row_out)

            if path_idx % 100000 == 0:
                print(f"Written paths: {path_idx}/{total_paths}")

    print(f"Saved all paths CSV: {output_csv}")


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()
    columns = [c.strip() for c in args.columns.split(",")]

    ensure_parent_dir(args.frontiers_output)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. Generate frontiers
    # -----------------------------------------------------
    print("Step 1: generating frontier layers with Java...")
    input_java = path_for_java(args.input, args.java_entry)
    frontiers_output_java = path_for_java(args.frontiers_output, args.java_entry)

    generate_frontiers_with_java(
        input_csv=input_java,
        output_csv=frontiers_output_java,
        java_entry=args.java_entry,
        main_class=args.frontier_main_class,
        maven_executable=args.maven_executable,
    )

    df_frontiers = pd.read_csv(args.frontiers_output)
    inputs, outputs = get_io_columns(df_frontiers)
    io_cols = inputs + outputs

    start_point_row = df_frontiers[df_frontiers["name"] == args.target]
    if start_point_row.empty:
        raise ValueError(f"Start point '{args.target}' not found in frontiers file.")
    start_point_row = start_point_row.iloc[0].copy()

    initial_target_frontier = get_frontier_layer_for_name(df_frontiers, args.target)
    max_front = get_max_frontier_layer(df_frontiers)

    if args.start_front is None:
        current_front = initial_target_frontier - 1
    else:
        current_front = args.start_front

    if current_front < 1:
        raise ValueError(
            f"Computed start frontier is {current_front}. "
            f"Expected frontier >= 1. Check initial target frontier."
        )

    print(f"Initial target: {args.target}")
    print(f"Initial target frontier_layer: {initial_target_frontier}")
    print(f"Starting from frontier: {current_front}")
    print(f"Max frontier in dataset: {max_front}")

    # -----------------------------------------------------
    # 2. Generate ONE global grid
    # -----------------------------------------------------
    global_dir = Path(args.output_dir) / "global_grid"
    global_dir.mkdir(parents=True, exist_ok=True)

    global_samples_csv = global_dir / "samples_global.csv"
    global_grid_plot_csv = global_dir / "samples_global_for_plot.csv"
    global_grid_selected_plot = global_dir / "samples_global_plot.html"

    print("=" * 80)
    print("Step 2: generating ONE global grid...")

    global_samples_df = generate_global_samples(
        df=df_frontiers,
        target_row=start_point_row,
        columns_to_modify=columns,
        pct_above=args.pct_above,
        step_pct=args.step_pct,
        min_points_per_dim=args.min_points_per_dim,
    )

    global_samples_df.to_csv(global_samples_csv, index=False)
    print(f"Saved global samples: {global_samples_csv}")

    # mały trick, żeby dało się to narysować przez ten sam helper
    plot_df = global_samples_df.copy()
    plot_df["candidate_efficient"] = "all"
    plot_df["candidate_efficiency"] = np.nan
    plot_df.to_csv(global_grid_plot_csv, index=False)

    save_all_results_plot(
        results_csv=str(global_grid_plot_csv),
        output_html=str(global_grid_selected_plot),
        x=args.plot_x,
        y=args.plot_y,
        z=args.plot_z,
        title="Global sampling grid",
    )

    # -----------------------------------------------------
    # 3. Evaluate same grid against each frontier
    # -----------------------------------------------------
    all_boundary_points = []
    all_selected_points = []
    selected_layers_for_paths = []

    step_idx = 1

    while current_front >= 1:
        if args.max_steps is not None and step_idx > args.max_steps:
            print(f"Stopping because max_steps={args.max_steps}")
            break

        print("=" * 80)
        print(f"GLOBAL GRID ITERATION {step_idx}")
        print(f"Reference frontier: {current_front}")
        print("Evaluating SAME global grid against this frontier...")

        iter_dir = Path(args.output_dir) / f"iter_{step_idx:02d}_front_{current_front}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        results_output = iter_dir / "results.csv"
        all_results_plot_output = iter_dir / "all_results_plot.html"
        boundary_output = iter_dir / "boundary_true.csv"
        selected_output = iter_dir / "selected_points.csv"

        evaluate_candidates_with_java(
            frontiers_csv=frontiers_output_java,
            candidates_csv=path_for_java(str(global_samples_csv), args.java_entry),
            results_csv=path_for_java(str(results_output), args.java_entry),
            target_front=current_front,
            java_entry=args.java_entry,
            main_class=args.evaluator_main_class,
            maven_executable=args.maven_executable,
        )

        save_all_results_plot(
            results_csv=str(results_output),
            output_html=str(all_results_plot_output),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
            title=f"All evaluated grid points vs frontier {current_front}",
        )

        df_results = pd.read_csv(results_output)

        print("Selecting boundary solution set...")
        df_boundary = select_boundary_true_points(
            df=df_results,
            feature_cols=columns,
            k_nearest_true_per_false=args.boundary_k,
        )

        df_boundary["iteration"] = step_idx
        df_boundary["reference_frontier"] = current_front

        df_selected = rank_and_limit_boundary_points(
            df_boundary=df_boundary,
            limit=args.points_per_front,
        )

        df_boundary.to_csv(boundary_output, index=False)
        df_selected.to_csv(selected_output, index=False)

        print(f"Saved boundary true points: {boundary_output}")
        print(f"Boundary true count: {len(df_boundary)}")
        print(f"Saved selected points: {selected_output}")
        print(f"Selected points count: {len(df_selected)}")

        all_boundary_points.append(df_boundary)
        all_selected_points.append(df_selected)
        selected_layers_for_paths.append(df_selected)

        current_front -= 1
        step_idx += 1

    # -----------------------------------------------------
    # 4. Aggregate outputs
    # -----------------------------------------------------
    print("=" * 80)
    print("Saving aggregated outputs...")

    boundary_all_df = (
        pd.concat(all_boundary_points, ignore_index=True)
        if all_boundary_points else pd.DataFrame()
    )
    selected_all_df = (
        pd.concat(all_selected_points, ignore_index=True)
        if all_selected_points else pd.DataFrame()
    )

    boundary_all_path = Path(args.output_dir) / "boundary_true_all_layers.csv"
    selected_all_path = Path(args.output_dir) / "selected_points_all_layers.csv"
    selected_all_plot = Path(args.output_dir) / "selected_points_all_layers.html"
    paths_all_path = Path(args.output_dir) / "all_paths_cartesian.csv"

    boundary_all_df.to_csv(boundary_all_path, index=False)
    selected_all_df.to_csv(selected_all_path, index=False)

    print(f"Aggregated boundary points: {boundary_all_path}")
    print(f"Aggregated selected points: {selected_all_path}")

    if not selected_all_df.empty:
        save_selected_points_plot(
            selected_csv=str(selected_all_path),
            output_html=str(selected_all_plot),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
            title="Selected points across all frontiers",
        )

    if selected_layers_for_paths:
        save_paths_cartesian_product(
            layer_point_dfs=selected_layers_for_paths,
            output_csv=str(paths_all_path),
            start_row=start_point_row,
            io_cols=io_cols,
        )

    print("DONE")
    print(f"Global samples:         {global_samples_csv}")
    print(f"Boundary all layers:    {boundary_all_path}")
    print(f"Selected all layers:    {selected_all_path}")
    print(f"Selected all plot:      {selected_all_plot}")
    print(f"All paths CSV:          {paths_all_path}")


if __name__ == "__main__":
    main()