import argparse
import csv
import os
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from java_runner import (
    generate_frontiers_with_java,
    evaluate_candidates_with_java,
)
from sampling.generator import generate_frontier_samples
from postprocess.select_next_frontier import (
    annotate_boundary_neighbors,
    select_boundary_true_points,
)
from plotting.iterative_plots import (
    save_boundary_plot,
    save_all_results_plot,
    save_boundary_flag_plot,
    save_selected_points_plot,
)


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

    # Zgodnie z nową ideą: nie pogarszamy parametrów
    parser.add_argument("--pct-below", type=float, default=0.0)
    parser.add_argument("--pct-above", type=float, default=30.0)
    parser.add_argument("--step-pct", type=float, default=2.0)

    parser.add_argument("--boundary-k", type=int, default=5)
    parser.add_argument("--points-per-front", type=int, default=100)

    parser.add_argument("--plot-x", required=True)
    parser.add_argument("--plot-y", required=True)
    parser.add_argument("--plot-z", required=True)

    return parser.parse_args()


def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def create_run_output_dir(base_output_dir: str, method_name: str) -> Path:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / method_name / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def save_paths_cartesian_product(
    layer_point_dfs: list[pd.DataFrame],
    output_csv: str,
    start_row: pd.Series,
    io_cols: list[str],
):
    """
    Tworzy wszystkie możliwe ścieżki:
    start x layer1 x layer2 x ... x layerN

    Zapis strumieniowy do CSV, żeby nie trzymać wszystkiego w RAM.
    """
    ensure_parent_dir(output_csv)

    if not layer_point_dfs:
        raise ValueError("No layer point dataframes provided for path export.")

    # Jeśli jakaś warstwa jest pusta, nie da się zbudować ścieżek
    for idx, df in enumerate(layer_point_dfs, start=1):
        if df.empty:
            raise ValueError(f"Layer dataframe #{idx} is empty. Cannot build paths.")

    base_cols = [
        "path_id",
        "start_name",
        "start_frontier_layer",
    ] + [f"start_{c}" for c in io_cols]

    dynamic_cols = []
    for step_idx, df_layer in enumerate(layer_point_dfs, start=1):
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


def main():
    args = parse_args()
    columns = [c.strip() for c in args.columns.split(",")]

    run_dir = create_run_output_dir(args.output_dir, "frontier")
    frontiers_output_path = run_dir / Path(args.frontiers_output).name

    print("Step 1: generating frontier layers with Java...")
    print(f"Run output directory: {run_dir}")
    input_java = path_for_java(args.input, args.java_entry)
    frontiers_output_java = path_for_java(str(frontiers_output_path), args.java_entry)

    generate_frontiers_with_java(
        input_csv=input_java,
        output_csv=frontiers_output_java,
        java_entry=args.java_entry,
        main_class=args.frontier_main_class,
        maven_executable=args.maven_executable,
    )

    df_frontiers = pd.read_csv(frontiers_output_path)
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
    print(f"Starting layered search from frontier: {current_front}")
    print(f"Max frontier in dataset: {max_front}")
    print("IMPORTANT: each frontier is sampled relative to the ORIGINAL start point.")

    all_selected_points = []
    all_boundary_points = []
    selected_layers_for_paths = []

    step_idx = 1


    while current_front >= 1:
        if args.max_steps is not None and step_idx > args.max_steps:
            print(f"Stopping because max_steps={args.max_steps}")
            break

        print("=" * 70)
        print(f"LAYER ITERATION {step_idx}")
        print(f"Reference frontier: {current_front}")
        print(f"Sampling ALWAYS from start point: {start_point_row['name']}")

        iter_dir = run_dir / f"iter_{step_idx:02d}_front_{current_front}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        samples_output = iter_dir / "samples.csv"
        results_output = iter_dir / "results.csv"
        results_boundary_flag_output = iter_dir / "results_with_boundary_flag.csv"
        boundary_output = iter_dir / "boundary_true.csv"
        selected_output = iter_dir / "selected_points.csv"
        all_results_plot_output = iter_dir / "all_results_plot.html"
        boundary_flag_plot_output = iter_dir / "boundary_flag_plot.html"

        print("Generating candidate samples...")
        sampled_df = generate_frontier_samples(
            df=df_frontiers,
            target_row=start_point_row,
            columns_to_modify=columns,
            target_front=current_front,
            pct_below=args.pct_below,
            pct_above=args.pct_above,
            step_pct=args.step_pct,
        )

        if sampled_df.empty:
            raise ValueError(
                f"Sampling produced 0 rows for iteration={step_idx}, frontier={current_front}, "
                f"target={args.target}"
            )

        sampled_df.to_csv(samples_output, index=False)
        print(f"Saved samples: {samples_output}")
        print(f"Generated candidates: {len(sampled_df)}")

        print("Evaluating candidates with Java...")
        evaluate_candidates_with_java(
            frontiers_csv=frontiers_output_java,
            candidates_csv=path_for_java(str(samples_output), args.java_entry),
            results_csv=path_for_java(str(results_output), args.java_entry),
            target_front=current_front,
            java_entry=args.java_entry,
            main_class=args.evaluator_main_class,
            maven_executable=args.maven_executable,
        )

        print("Saving full-grid plot before boundary reduction...")
        save_all_results_plot(
            results_csv=str(results_output),
            output_html=str(all_results_plot_output),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

        print("Selecting boundary solution set...")
        df_results = pd.read_csv(results_output)
        df_results_annotated = annotate_boundary_neighbors(
            df=df_results,
            feature_cols=columns,
        )
        df_results_annotated.to_csv(results_boundary_flag_output, index=False)

        save_boundary_flag_plot(
            results_csv=str(results_boundary_flag_output),
            output_html=str(boundary_flag_plot_output),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

        df_boundary = select_boundary_true_points(
            df=df_results_annotated,
            feature_cols=columns,
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
        print(f"Saved annotated results: {results_boundary_flag_output}")
        print(f"Saved boundary flag plot: {boundary_flag_plot_output}")
        print(f"Saved selected points: {selected_output}")
        print(f"Selected points count: {len(df_selected)}")

        all_boundary_points.append(df_boundary)
        all_selected_points.append(df_selected)
        selected_layers_for_paths.append(df_selected)


        current_front -= 1
        step_idx += 1

    print("=" * 70)
    print("Saving aggregated outputs...")

    boundary_all_df = (
        pd.concat(all_boundary_points, ignore_index=True)
        if all_boundary_points else pd.DataFrame()
    )
    selected_all_df = (
        pd.concat(all_selected_points, ignore_index=True)
        if all_selected_points else pd.DataFrame()
    )

    boundary_all_path = run_dir / "boundary_true_all_layers.csv"
    selected_all_path = run_dir / "selected_points_all_layers.csv"
    paths_all_path = run_dir / "all_paths_cartesian.csv"

    boundary_all_df.to_csv(boundary_all_path, index=False)
    selected_all_df.to_csv(selected_all_path, index=False)

    print(f"Aggregated boundary points: {boundary_all_path}")
    print(f"Aggregated selected points: {selected_all_path}")

    if selected_layers_for_paths:
        save_paths_cartesian_product(
            layer_point_dfs=selected_layers_for_paths,
            output_csv=str(paths_all_path),
            start_row=start_point_row,
            io_cols=io_cols,
        )

    print("=" * 70)
    print("Generating plots...")

    boundary_plot_path = run_dir / "boundary_true_all_layers.html"
    selected_plot_path = run_dir / "selected_points_all_layers.html"

    if not boundary_all_df.empty:
        save_boundary_plot(
            boundary_csv=str(boundary_all_path),
            output_html=str(boundary_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

    if not selected_all_df.empty:
        save_selected_points_plot(
            selected_csv=str(selected_all_path),
            output_html=str(selected_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

    print("DONE")
    print(f"Boundary all layers: {boundary_all_path}")
    print(f"Selected all layers: {selected_all_path}")
    print(f"All paths CSV:       {paths_all_path}")
    print(f"Boundary plot:       {boundary_plot_path}")
    print(f"Selected plot:       {selected_plot_path}")


if __name__ == "__main__":
    main()
