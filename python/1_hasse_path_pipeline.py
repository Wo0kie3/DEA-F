import argparse
from pathlib import Path

import pandas as pd

from java_runner import (
    export_candidate_robust_metrics_with_java,
    generate_preference_relations_with_java,
)
from path_metrics import write_path_metrics
from path_pipeline_common import (
    add_effort_columns,
    build_component_graph,
    build_relation_matrix,
    build_worse_to_better_graph,
    component_paths_to_frame,
    component_requirement_mask,
    create_run_output_dir,
    ensure_parent_dir,
    enumerate_state_paths,
    enumerate_paths_from_start,
    expand_component_paths_to_dmu_paths,
    generate_attainable_fictive_candidates,
    get_io_columns,
    limit_paths,
    normalization_ranges_from_frame,
    parse_columns_arg,
    paths_to_frame,
    path_for_java,
    state_paths_to_frame,
    strongly_connected_components,
    transitive_reduce_dag,
    write_stage_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--java-entry", required=True)
    parser.add_argument("--mode", choices=["real", "fictive", "mixed"], default="real")
    parser.add_argument("--columns", default=None)
    parser.add_argument("--pct-above", type=float, default=30.0)
    parser.add_argument("--step-pct", type=float, default=10.0)
    parser.add_argument("--step-abs", type=float, default=None)
    parser.add_argument("--min-points-per-dim", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--points-per-stage", type=int, default=None)
    parser.add_argument(
        "--preference-main-class",
        default="org.example.CsvPreferenceRelationsPreview",
    )
    parser.add_argument(
        "--candidate-metrics-main-class",
        default="org.example.CsvCandidateRobustMetricsExporter",
    )
    parser.add_argument("--maven-executable", default="mvn")
    parser.add_argument("--max-paths", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = create_run_output_dir(args.output_dir, "hasse_path")

    df_input = pd.read_csv(args.input)
    if "name" not in df_input.columns:
        raise ValueError("Input CSV must contain a 'name' column.")

    dmu_order = df_input["name"].astype(str).tolist()
    if args.target not in dmu_order:
        raise ValueError(f"Target DMU '{args.target}' not found in input CSV.")
    inputs, outputs = get_io_columns(df_input)
    io_cols = inputs + outputs
    target_row = df_input[df_input["name"].astype(str) == args.target].iloc[0].copy()
    normalization_ranges = normalization_ranges_from_frame(
        df_input,
        io_cols,
        fallback_row=target_row,
    )

    relations_csv = run_dir / "preference_relations_all.csv"
    generate_preference_relations_with_java(
        input_csv=path_for_java(args.input, args.java_entry),
        output_csv=path_for_java(str(relations_csv), args.java_entry),
        java_entry=args.java_entry,
        main_class=args.preference_main_class,
        maven_executable=args.maven_executable,
    )

    df_relations = pd.read_csv(relations_csv)
    if "necessary_preferred" not in df_relations.columns:
        raise ValueError("Relations CSV must contain 'necessary_preferred' column.")

    necessary_matrix = build_relation_matrix(
        df_relations=df_relations,
        dmu_order=dmu_order,
        value_col="necessary_preferred",
    )
    necessary_matrix.to_csv(run_dir / "necessary_matrix.csv", index=True)

    graph = build_worse_to_better_graph(necessary_matrix)
    components = strongly_connected_components(graph)
    component_id_by_dmu, component_members, component_graph = build_component_graph(
        components=components,
        graph=graph,
        dmu_order=dmu_order,
    )
    reduced_graph = transitive_reduce_dag(component_graph)

    components_rows = []
    for component_id, members in component_members.items():
        components_rows.append(
            {
                "component_id": component_id,
                "members_count": len(members),
                "members": "|".join(members),
            }
        )
    pd.DataFrame(components_rows).to_csv(run_dir / "necessary_components.csv", index=False)

    edge_rows = []
    for source, targets in sorted(reduced_graph.items()):
        for target in sorted(targets):
            edge_rows.append({"source_component": source, "target_component": target})
    pd.DataFrame(edge_rows).to_csv(run_dir / "necessary_cover_edges.csv", index=False)

    start_component = component_id_by_dmu[args.target]
    component_paths = enumerate_paths_from_start(start_component, reduced_graph)
    component_paths = limit_paths(component_paths, args.max_paths)
    component_paths_to_frame(component_paths, component_members).to_csv(
        run_dir / "component_paths.csv",
        index=False,
    )

    dmu_paths = expand_component_paths_to_dmu_paths(
        component_paths=component_paths,
        component_members=component_members,
        start_dmu=args.target,
    )
    dmu_paths = limit_paths(dmu_paths, args.max_paths)
    ensure_parent_dir(str(run_dir / "paths.csv"))
    real_paths_frame = paths_to_frame(dmu_paths, state_data=df_input)
    real_paths_frame.to_csv(run_dir / "real_paths.csv", index=False)
    if args.mode == "real":
        real_paths_frame.to_csv(run_dir / "paths.csv", index=False)
        write_path_metrics(
            real_paths_frame,
            run_dir / "path_metrics.csv",
            method_name="hasse_path",
            io_columns=io_cols,
            normalization_ranges=normalization_ranges,
        )
        return

    columns_to_modify = parse_columns_arg(args.columns, io_cols)

    candidates = generate_attainable_fictive_candidates(
        df=df_input,
        target_row=target_row,
        columns_to_modify=columns_to_modify,
        pct_above=args.pct_above,
        step_pct=args.step_pct,
        step_abs=args.step_abs,
        min_points_per_dim=args.min_points_per_dim,
        max_candidates=args.max_candidates,
        name_prefix=args.target,
    )
    candidates_csv = run_dir / "fictive_candidates.csv"
    candidates.to_csv(candidates_csv, index=False)

    candidate_metrics_csv = run_dir / "fictive_candidate_metrics.csv"
    export_candidate_robust_metrics_with_java(
        reference_csv=path_for_java(args.input, args.java_entry),
        candidates_csv=path_for_java(str(candidates_csv), args.java_entry),
        output_csv=path_for_java(str(candidate_metrics_csv), args.java_entry),
        java_entry=args.java_entry,
        main_class=args.candidate_metrics_main_class,
        maven_executable=args.maven_executable,
    )
    fictive_metrics = pd.read_csv(candidate_metrics_csv)
    fictive_metrics = add_effort_columns(
        fictive_metrics,
        target_row,
        io_cols,
        normalization_ranges=normalization_ranges,
    )
    fictive_metrics.to_csv(run_dir / "fictive_hasse_metrics.csv", index=False)

    real_states = df_input[["name", *io_cols]].copy()
    real_states["state_type"] = "real"
    real_states = add_effort_columns(
        real_states,
        target_row,
        io_cols,
        normalization_ranges=normalization_ranges,
    )
    start_state = real_states[real_states["name"] == args.target].iloc[0].copy()

    all_stage_candidates = []
    all_state_paths = []
    transition_log = []
    for component_path in component_paths:
        stage_candidates = []
        for stage_idx, component_id in enumerate(component_path[1:], start=1):
            frames = []
            members = component_members[component_id]

            if args.mode == "mixed":
                real_stage = real_states[real_states["name"].isin(members)].copy()
                real_stage["milestone_component_id"] = component_id
                real_stage["milestone_members"] = "|".join(members)
                real_stage["milestone_gap"] = 0.0
                frames.append(real_stage)

            mask = component_requirement_mask(fictive_metrics, members)
            fictive_stage = fictive_metrics[mask].copy()
            if not fictive_stage.empty:
                fictive_stage["milestone_component_id"] = component_id
                fictive_stage["milestone_members"] = "|".join(members)
                fictive_stage["milestone_gap"] = 0.0
                frames.append(fictive_stage)

            if not frames:
                stage_candidates = []
                break

            stage = pd.concat(frames, ignore_index=True)
            stage.insert(0, "stage", stage_idx)
            all_stage_candidates.append(stage)
            stage_candidates.append(stage.drop(columns=["stage"]))

        if not stage_candidates:
            continue

        remaining = None if args.max_paths is None else max(args.max_paths - len(all_state_paths), 0)
        if remaining == 0:
            break

        all_state_paths.extend(
            enumerate_state_paths(
                start_row=start_state,
                stage_candidates=stage_candidates,
                inputs=inputs,
                outputs=outputs,
                max_paths=remaining,
                points_per_transition=args.points_per_stage,
                normalization_ranges=normalization_ranges,
                minimal_inputs=[col for col in inputs if col in columns_to_modify],
                minimal_outputs=[col for col in outputs if col in columns_to_modify],
                transition_log=transition_log,
            )
        )

    if all_stage_candidates:
        pd.concat(all_stage_candidates, ignore_index=True).to_csv(
            run_dir / "stage_candidates.csv",
            index=False,
        )
    else:
        write_stage_candidates([], str(run_dir / "stage_candidates.csv"))
    pd.DataFrame(transition_log).to_csv(
        run_dir / "transition_candidates.csv",
        index=False,
    )

    final_paths_frame = state_paths_to_frame(all_state_paths, io_cols)
    final_paths_frame.to_csv(run_dir / "paths.csv", index=False)
    write_path_metrics(
        final_paths_frame,
        run_dir / "path_metrics.csv",
        method_name="hasse_path",
        io_columns=io_cols,
        normalization_ranges=normalization_ranges,
    )


if __name__ == "__main__":
    main()
