import argparse
import os
from pathlib import Path
import pandas as pd

from java_runner import (
    generate_frontiers_with_java,
    evaluate_candidates_with_java,
)
from sampling.generator import generate_frontier_samples
from postprocess.select_next_frontier import process_dea_results
from plotting.iterative_plots import (
    save_boundary_plot,
    save_best_points_plot,
    save_combined_iterative_plot,
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

    parser.add_argument("--pct-below", type=float, default=20.0)
    parser.add_argument("--pct-above", type=float, default=15.0)
    parser.add_argument("--step-pct", type=float, default=5.0)

    parser.add_argument("--plot-x", required=True)
    parser.add_argument("--plot-y", required=True)
    parser.add_argument("--plot-z", required=True)

    parser.add_argument("--boundary-k", type=int, default=3)
    return parser.parse_args()


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


def load_best_point_name(best_point_csv: str) -> str:
    df = pd.read_csv(best_point_csv)
    if df.empty:
        raise ValueError(f"Best point file is empty: {best_point_csv}")
    return str(df.iloc[0]["name"])


def main():
    args = parse_args()
    columns = [c.strip() for c in args.columns.split(",")]

    ensure_parent_dir(args.frontiers_output)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================
    # STEP 1: generate frontier layers once
    # =========================================================
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

    # start frontier = frontier of initial target minus 1
    initial_target_frontier = get_frontier_layer_for_name(df_frontiers, args.target)
    current_target_name = args.target

    if args.start_front is None:
        current_front = initial_target_frontier - 1
    else:
        current_front = args.start_front

    max_front = get_max_frontier_layer(df_frontiers)

    if current_front < 1:
        raise ValueError(
            f"Computed start frontier is {current_front}. "
            f"Expected frontier >= 1. Check initial target frontier."
        )

    print(f"Initial target: {args.target}")
    print(f"Initial target frontier_layer: {initial_target_frontier}")
    print(f"Starting iterative search from frontier: {current_front}")
    print(f"Max frontier in dataset: {max_front}")

    all_best_points = []
    all_boundary_points = []

    step_idx = 1
    prev_best_row = None

    while current_front >= 1:

        if current_front < 1:
            print("Reached best frontier. Stopping.")
            break

        if args.max_steps is not None and step_idx > args.max_steps:
            print(f"Stopping because max_steps={args.max_steps}")
            break

        print("=" * 70)
        print(f"ITERATION {step_idx}")
        print(f"Current target: {current_target_name}")
        print(f"Reference frontier: {current_front}")

        iter_dir = Path(args.output_dir) / f"iter_{step_idx:02d}_front_{current_front}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        samples_output = iter_dir / "samples.csv"
        results_output = iter_dir / "results.csv"
        boundary_output = iter_dir / "boundary_true.csv"
        best_point_output = iter_dir / "best_point.csv"

        # =========================================================
        # STEP 2: Python -> generate samples
        # =========================================================
        print("Generating candidate samples...")
        print("Generating candidate samples...")

        if step_idx == 1:
            sampled_df = generate_frontier_samples(
                df=df_frontiers,
                target_name=current_target_name,
                columns_to_modify=columns,
                target_front=current_front,
                pct_below=args.pct_below,
                pct_above=args.pct_above,
                step_pct=args.step_pct,
            )
        else:
            if prev_best_row is None:
                raise ValueError("prev_best_row is None before iteration > 1")

            sampled_df = generate_frontier_samples(
                df=df_frontiers,
                target_row=prev_best_row,
                columns_to_modify=columns,
                target_front=current_front,
                pct_below=args.pct_below,
                pct_above=args.pct_above,
                step_pct=args.step_pct,
            )

        if sampled_df.empty:
            raise ValueError(
                f"Sampling produced 0 rows for iteration={step_idx}, frontier={current_front}, "
                f"target={current_target_name}"
            )

        sampled_df.to_csv(samples_output, index=False)
        print(f"Saved samples: {samples_output}")
        print(f"Generated candidates: {len(sampled_df)}")

        # =========================================================
        # STEP 3: Java -> evaluate candidates
        # =========================================================
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

        # =========================================================
        # STEP 4: Python -> select boundary and best point
        # =========================================================
        print("Selecting boundary solution set and best next point...")
        process_dea_results(
            input_csv=str(results_output),
            boundary_output_csv=str(boundary_output),
            best_output_csv=str(best_point_output),
            feature_cols=columns,
            k_nearest_true_per_false=args.boundary_k,
        )

        df_best = pd.read_csv(best_point_output)
        df_best["iteration"] = step_idx
        df_best["reference_frontier"] = current_front
        all_best_points.append(df_best)

        prev_best_row = df_best.iloc[0].copy()

        df_boundary = pd.read_csv(boundary_output)
        df_boundary["iteration"] = step_idx
        df_boundary["reference_frontier"] = current_front
        all_boundary_points.append(df_boundary)

        # next iteration target = selected best point
        next_target_name = load_best_point_name(str(best_point_output))

        print(f"Selected next target: {next_target_name}")

        current_target_name = next_target_name
        current_front -= 1
        step_idx += 1

    # =========================================================
    # FINAL AGGREGATION
    # =========================================================
    print("=" * 70)
    print("Saving aggregated outputs...")

    if all_best_points:
        best_points_all_df = pd.concat(all_best_points, ignore_index=True)
    else:
        best_points_all_df = pd.DataFrame()

    if all_boundary_points:
        boundary_all_df = pd.concat(all_boundary_points, ignore_index=True)
    else:
        boundary_all_df = pd.DataFrame()

    best_points_all_path = Path(args.output_dir) / "best_points_all_iterations.csv"
    boundary_all_path = Path(args.output_dir) / "boundary_true_all_iterations.csv"

    best_points_all_df.to_csv(best_points_all_path, index=False)
    boundary_all_df.to_csv(boundary_all_path, index=False)

    print("DONE")
    print(f"Aggregated best points: {best_points_all_path}")
    print(f"Aggregated boundary points: {boundary_all_path}")

    start_point_row = df_frontiers[df_frontiers["name"] == args.target]
    if start_point_row.empty:
        raise ValueError(f"Start point '{args.target}' not found in frontiers file.")
    start_point_row = start_point_row.iloc[0]

    # =========================================================
    # FINAL PLOTS
    # =========================================================
    print("=" * 70)
    print("Generating plots...")

    boundary_plot_path = Path(args.output_dir) / "boundary_true_all_iterations.html"
    best_plot_path = Path(args.output_dir) / "best_points_all_iterations.html"
    combined_plot_path = Path(args.output_dir) / "iterative_path_combined.html"

    if not boundary_all_df.empty:
        save_boundary_plot(
            boundary_csv=str(boundary_all_path),
            output_html=str(boundary_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

    if not best_points_all_df.empty:
        save_best_points_plot(
            best_csv=str(best_points_all_path),
            output_html=str(best_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

    if not boundary_all_df.empty and not best_points_all_df.empty:
        save_combined_iterative_plot(
            boundary_csv=str(boundary_all_path),
            best_csv=str(best_points_all_path),
            output_html=str(combined_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
            start_point={
                "name": start_point_row["name"],
                args.plot_x: float(start_point_row[args.plot_x]),
                args.plot_y: float(start_point_row[args.plot_y]),
                args.plot_z: float(start_point_row[args.plot_z]),
            },
        )

    print("DONE")
    print(f"Aggregated best points: {best_points_all_path}")
    print(f"Aggregated boundary points: {boundary_all_path}")
    print(f"Boundary plot: {boundary_plot_path}")
    print(f"Best points plot: {best_plot_path}")
    print(f"Combined plot: {combined_plot_path}")


if __name__ == "__main__":
    main()