import argparse
import os
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from java_runner import (
    generate_frontiers_with_java,
    evaluate_candidates_with_java,
)
from postprocess.select_next_frontier import process_dea_results
from plotting.pairwise_plots import (
    save_pairwise_boundary_plot,
    save_pairwise_best_points_plot,
    save_pairwise_tree_plot,
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

    parser.add_argument("--pct-below", type=float, default=20.0)
    parser.add_argument("--pct-above", type=float, default=15.0)
    parser.add_argument("--step-pct", type=float, default=5.0)

    parser.add_argument("--boundary-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--plot-x", required=True)
    parser.add_argument("--plot-y", required=True)
    parser.add_argument("--plot-z", required=True)

    return parser.parse_args()


def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def path_for_java(path_str: str, java_entry: str) -> str:
    abs_target = Path(path_str).resolve()
    abs_java = Path(java_entry).resolve()
    return os.path.relpath(abs_target, start=abs_java)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))


def get_frontier_layer_for_name(df_frontiers: pd.DataFrame, name: str) -> int:
    match = df_frontiers[df_frontiers["name"] == name]
    if match.empty:
        raise ValueError(f"Target '{name}' not found in frontiers file.")
    return int(match.iloc[0]["frontier_layer"])


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


def build_axis_values(current_value, reference_value, step_pct, pct_below, pct_above):
    current_value = float(current_value)
    reference_value = float(reference_value)

    lo = min(current_value, reference_value)
    hi = max(current_value, reference_value)

    span = hi - lo
    scale = max(span, abs(current_value), abs(reference_value), 1.0)

    start = lo - scale * (pct_below / 100.0)
    end = hi + scale * (pct_above / 100.0)
    step = scale * (step_pct / 100.0)

    if step <= 0:
        raise ValueError(f"Computed non-positive step: {step}")

    values = np.arange(start, end + step, step)

    # dopnij dokładnie current/reference do siatki
    values = np.unique(np.concatenate([values, [current_value, reference_value]])).astype(float)
    values.sort()

    return values


def generate_pairwise_samples(
    current_point_row: pd.Series,
    reference_row: pd.Series,
    columns_to_modify: list[str],
    io_cols: list[str],
    pct_below: float,
    pct_above: float,
    step_pct: float,
) -> pd.DataFrame:
    grid = {}

    for col in columns_to_modify:
        if col not in current_point_row.index:
            raise ValueError(f"Column '{col}' missing in current_point_row")
        if col not in reference_row.index:
            raise ValueError(f"Column '{col}' missing in reference_row")

        current_val = float(current_point_row[col])
        reference_val = float(reference_row[col])

        values = build_axis_values(
            current_value=current_val,
            reference_value=reference_val,
            step_pct=step_pct,
            pct_below=pct_below,
            pct_above=pct_above,
        )

        grid[col] = values

        print(
            f"[GRID] {col}: current={current_val:.6f}, reference={reference_val:.6f}, "
            f"points={len(values)}, min={values.min():.6f}, max={values.max():.6f}"
        )

    combos = list(product(*grid.values()))
    if not combos:
        raise ValueError("No candidate combinations generated for pairwise sampling.")

    current_name = str(current_point_row["name"])
    reference_name = str(reference_row["name"])

    results = []
    for i, combo in enumerate(combos):
        row = {
            "name": f"{current_name}__to__{reference_name}_cand_{i}"
        }

        for c in io_cols:
            row[c] = current_point_row[c]

        for c, v in zip(columns_to_modify, combo):
            row[c] = v

        results.append(row)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError("Pairwise sampling produced an empty dataframe.")

    return result_df


def export_single_reference_frontier(
    reference_row: pd.Series,
    frontier_columns: list[str],
    output_csv: str,
):
    row = {col: reference_row[col] for col in frontier_columns if col in reference_row.index}
    df = pd.DataFrame([row])
    ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)


def main():
    args = parse_args()
    columns = [c.strip() for c in args.columns.split(",")]

    ensure_parent_dir(args.frontiers_output)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================
    # STEP 1: generate frontier layers once
    # =========================================================
    print("Generating frontier layers with Java...")
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
    frontier_columns = df_frontiers.columns.tolist()
    inputs, outputs = get_io_columns(df_frontiers)
    io_cols = inputs + outputs

    start_match = df_frontiers[df_frontiers["name"] == args.target]
    if start_match.empty:
        raise ValueError(f"Start target '{args.target}' not found in frontiers file.")

    start_row = start_match.iloc[0].copy()
    start_frontier = int(start_row["frontier_layer"])

    print(f"Start target: {args.target}")
    print(f"Start frontier_layer: {start_frontier}")
    print("This pipeline assumes lower frontier number = better frontier.")

    # aktywne węzły drzewa ścieżek
    active_nodes = [
        {
            "node_id": "node_root",
            "parent_node_id": None,
            "path_id": args.target,
            "point_row": start_row,
            "point_name": str(start_row["name"]),
            "source_reference_name": None,
            "source_frontier": start_frontier,
        }
    ]

    all_best_points = []
    all_boundary_points = []
    all_edges = []
    skipped_branches = []

    current_front = start_frontier - 1
    step_idx = 1
    branch_counter = 1

    while current_front >= 1:
        if args.max_steps is not None and step_idx > args.max_steps:
            print(f"Stopping because max_steps={args.max_steps}")
            break

        reference_units = (
            df_frontiers[df_frontiers["frontier_layer"] == current_front]
            .copy()
            .reset_index(drop=True)
        )

        if reference_units.empty:
            print(f"No units found in frontier {current_front}. Stopping.")
            break

        print("=" * 80)
        print(f"PAIRWISE ITERATION {step_idx}")
        print(f"REFERENCE FRONTIER: {current_front}")
        print(f"REFERENCE UNITS: {', '.join(reference_units['name'].astype(str).tolist())}")
        print(f"ACTIVE NODES TO EXPAND: {len(active_nodes)}")

        next_active_nodes = []

        for active_node in active_nodes:
            current_point_row = active_node["point_row"]
            current_point_name = str(current_point_row["name"])

            print("-" * 80)
            print(f"Expanding node: {active_node['node_id']}")
            print(f"Current point: {current_point_name}")

            for _, reference_row in reference_units.iterrows():
                reference_name = str(reference_row["name"])

                branch_id = f"branch_{branch_counter:06d}"
                branch_counter += 1

                branch_dir = (
                    Path(args.output_dir)
                    / f"iter_{step_idx:02d}_front_{current_front}"
                    / f"{sanitize_name(current_point_name)}__to__{sanitize_name(reference_name)}"
                )
                branch_dir.mkdir(parents=True, exist_ok=True)

                reference_frontier_csv = branch_dir / "reference_frontier.csv"
                samples_csv = branch_dir / "samples.csv"
                results_csv = branch_dir / "results.csv"
                boundary_csv = branch_dir / "boundary_true.csv"
                best_point_csv = branch_dir / "best_point.csv"

                print(f"Running branch: {current_point_name} -> {reference_name}")

                try:
                    # 1) zapisz jednoelementowy frontier referencyjny
                    export_single_reference_frontier(
                        reference_row=reference_row,
                        frontier_columns=frontier_columns,
                        output_csv=str(reference_frontier_csv),
                    )

                    # 2) sampling dla pary
                    sampled_df = generate_pairwise_samples(
                        current_point_row=current_point_row,
                        reference_row=reference_row,
                        columns_to_modify=columns,
                        io_cols=io_cols,
                        pct_below=args.pct_below,
                        pct_above=args.pct_above,
                        step_pct=args.step_pct,
                    )

                    if sampled_df.empty:
                        raise ValueError("Sampling produced 0 rows.")

                    sampled_df.to_csv(samples_csv, index=False)
                    print(f"Saved samples: {samples_csv}")
                    print(f"Generated candidates: {len(sampled_df)}")

                    # 3) evaluator Java
                    evaluate_candidates_with_java(
                        frontiers_csv=path_for_java(str(reference_frontier_csv), args.java_entry),
                        candidates_csv=path_for_java(str(samples_csv), args.java_entry),
                        results_csv=path_for_java(str(results_csv), args.java_entry),
                        target_front=current_front,
                        java_entry=args.java_entry,
                        main_class=args.evaluator_main_class,
                        maven_executable=args.maven_executable,
                    )

                    # 4) boundary + best point
                    process_dea_results(
                        input_csv=str(results_csv),
                        boundary_output_csv=str(boundary_csv),
                        best_output_csv=str(best_point_csv),
                        feature_cols=columns,
                        k_nearest_true_per_false=args.boundary_k,
                    )

                    df_best = pd.read_csv(best_point_csv)
                    if df_best.empty:
                        raise ValueError("best_point.csv is empty.")

                    df_boundary = pd.read_csv(boundary_csv)
                    if df_boundary.empty:
                        raise ValueError("boundary_true.csv is empty.")

                    best_row = df_best.iloc[0].copy()
                    best_name = str(best_row["name"])

                    node_id = f"node_{branch_id}"
                    path_id = f"{active_node['path_id']} -> {reference_name}"

                    # metadata do best point
                    df_best["iteration"] = step_idx
                    df_best["reference_frontier"] = current_front
                    df_best["from_point_name"] = current_point_name
                    df_best["to_reference_name"] = reference_name
                    df_best["node_id"] = node_id
                    df_best["parent_node_id"] = active_node["node_id"]
                    df_best["path_id"] = path_id
                    all_best_points.append(df_best)

                    # metadata do boundary
                    df_boundary["iteration"] = step_idx
                    df_boundary["reference_frontier"] = current_front
                    df_boundary["from_point_name"] = current_point_name
                    df_boundary["to_reference_name"] = reference_name
                    df_boundary["node_id"] = node_id
                    df_boundary["parent_node_id"] = active_node["node_id"]
                    df_boundary["path_id"] = path_id
                    all_boundary_points.append(df_boundary)

                    all_edges.append(
                        {
                            "branch_id": branch_id,
                            "iteration": step_idx,
                            "reference_frontier": current_front,
                            "parent_node_id": active_node["node_id"],
                            "node_id": node_id,
                            "from_point_name": current_point_name,
                            "to_reference_name": reference_name,
                            "selected_best_point_name": best_name,
                            "path_id": path_id,
                            "samples_csv": str(samples_csv),
                            "results_csv": str(results_csv),
                            "boundary_csv": str(boundary_csv),
                            "best_point_csv": str(best_point_csv),
                        }
                    )

                    next_active_nodes.append(
                        {
                            "node_id": node_id,
                            "parent_node_id": active_node["node_id"],
                            "path_id": path_id,
                            "point_row": best_row,
                            "point_name": best_name,
                            "source_reference_name": reference_name,
                            "source_frontier": current_front,
                        }
                    )

                    print(f"Selected best point: {best_name}")

                except Exception as e:
                    msg = str(e)
                    print(f"[SKIP] Branch failed: {current_point_name} -> {reference_name} | {msg}")
                    skipped_branches.append(
                        {
                            "iteration": step_idx,
                            "reference_frontier": current_front,
                            "from_point_name": current_point_name,
                            "to_reference_name": reference_name,
                            "error": msg,
                            "branch_dir": str(branch_dir),
                        }
                    )

        if not next_active_nodes:
            print("No successful branches in this iteration. Stopping.")
            break

        active_nodes = next_active_nodes
        current_front -= 1
        step_idx += 1

    # =========================================================
    # FINAL AGGREGATION
    # =========================================================
    print("=" * 80)
    print("Saving aggregated outputs...")

    best_points_all_df = (
        pd.concat(all_best_points, ignore_index=True)
        if all_best_points else pd.DataFrame()
    )
    boundary_all_df = (
        pd.concat(all_boundary_points, ignore_index=True)
        if all_boundary_points else pd.DataFrame()
    )
    edges_df = pd.DataFrame(all_edges)
    skipped_df = pd.DataFrame(skipped_branches)

    best_points_all_path = Path(args.output_dir) / "pairwise_best_points_all.csv"
    boundary_all_path = Path(args.output_dir) / "pairwise_boundary_true_all.csv"
    edges_path = Path(args.output_dir) / "pairwise_path_edges.csv"
    skipped_path = Path(args.output_dir) / "pairwise_skipped_branches.csv"

    best_points_all_df.to_csv(best_points_all_path, index=False)
    boundary_all_df.to_csv(boundary_all_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    print("=" * 80)
    print("Generating plots...")

    boundary_plot_path = Path(args.output_dir) / "pairwise_boundary_all.html"
    best_plot_path = Path(args.output_dir) / "pairwise_best_points_all.html"
    tree_plot_path = Path(args.output_dir) / "pairwise_tree.html"

    if not boundary_all_df.empty:
        save_pairwise_boundary_plot(
            boundary_csv=str(boundary_all_path),
            output_html=str(boundary_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

    if not best_points_all_df.empty:
        save_pairwise_best_points_plot(
            best_csv=str(best_points_all_path),
            output_html=str(best_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
        )

        save_pairwise_tree_plot(
            best_csv=str(best_points_all_path),
            output_html=str(tree_plot_path),
            x=args.plot_x,
            y=args.plot_y,
            z=args.plot_z,
            start_point={
                "name": start_row["name"],
                args.plot_x: float(start_row[args.plot_x]),
                args.plot_y: float(start_row[args.plot_y]),
                args.plot_z: float(start_row[args.plot_z]),
            },
        )

    print("DONE")
    print(f"Best points:      {best_points_all_path}")
    print(f"Boundary points:  {boundary_all_path}")
    print(f"Path edges:       {edges_path}")
    print(f"Skipped branches: {skipped_path}")
    print(f"Boundary plot:    {boundary_plot_path}")
    print(f"Best plot:        {best_plot_path}")
    print(f"Tree plot:        {tree_plot_path}")


if __name__ == "__main__":
    main()