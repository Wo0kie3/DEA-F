import argparse

import pandas as pd

from candidate_refinement import refine_numeric_goal_candidates
from java_runner import (
    export_candidate_robust_metrics_with_java,
    export_extreme_efficiencies_with_java,
    export_extreme_ranks_with_java,
)
from path_metrics import write_path_metrics
from path_pipeline_common import (
    add_effort_columns,
    add_real_state_type,
    candidate_grid_steps,
    create_run_output_dir,
    enumerate_state_paths,
    generate_attainable_fictive_candidates,
    get_io_columns,
    linear_milestones,
    normalization_ranges_from_frame,
    parse_columns_arg,
    path_for_java,
    resolve_first_present,
    state_paths_to_frame,
    write_stage_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-width", type=float, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--java-entry", required=True)
    parser.add_argument("--mode", choices=["real", "fictive", "mixed"], default="real")
    parser.add_argument("--width-kind", choices=["score", "rank"], default="score")
    parser.add_argument("--columns", default=None)
    parser.add_argument("--pct-above", type=float, default=30.0)
    parser.add_argument("--step-pct", type=float, default=10.0)
    parser.add_argument("--step-abs", type=float, default=None)
    parser.add_argument("--min-points-per-dim", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--points-per-stage", type=int, default=None)
    parser.add_argument(
        "--efficiencies-main-class",
        default="org.example.CsvExtremeEfficienciesExporter",
    )
    parser.add_argument(
        "--ranks-main-class",
        default="org.example.CsvExtremeRanksExporter",
    )
    parser.add_argument(
        "--candidate-metrics-main-class",
        default="org.example.CsvCandidateRobustMetricsExporter",
    )
    parser.add_argument("--maven-executable", default="mvn")
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--refine-fictive-candidates", action="store_true")
    parser.add_argument("--refine-iterations", type=int, default=8)
    parser.add_argument("--refine-max-seeds", type=int, default=20)
    parser.add_argument("--local-search-samples", type=int, default=0)
    parser.add_argument("--local-search-step-multiplier", type=float, default=1.0)
    parser.add_argument("--local-search-random-state", type=int, default=42)
    parser.add_argument(
        "--local-search-sampling",
        choices=["random", "stratified"],
        default="random",
    )
    return parser.parse_args()


def normalize_efficiency_metrics(df_eff: pd.DataFrame) -> pd.DataFrame:
    name_col = resolve_first_present(df_eff, ["name", "dmu_name", "dmu"])
    best_eff_col = resolve_first_present(
        df_eff,
        ["best_efficiency", "max_efficiency", "efficiency_max", "bestEfficiency"],
    )
    worst_eff_col = resolve_first_present(
        df_eff,
        ["worst_efficiency", "min_efficiency", "efficiency_min", "worstEfficiency"],
    )
    return df_eff.rename(
        columns={
            name_col: "name",
            best_eff_col: "best_efficiency",
            worst_eff_col: "worst_efficiency",
        }
    )[["name", "best_efficiency", "worst_efficiency"]].copy()


def normalize_rank_metrics(df_ranks: pd.DataFrame) -> pd.DataFrame:
    name_col = resolve_first_present(df_ranks, ["name", "dmu_name", "dmu"])
    best_rank_col = resolve_first_present(
        df_ranks,
        ["best_rank", "min_rank", "rank_best", "bestRank"],
    )
    worst_rank_col = resolve_first_present(
        df_ranks,
        ["worst_rank", "max_rank", "rank_worst", "worstRank"],
    )
    return df_ranks.rename(
        columns={
            name_col: "name",
            best_rank_col: "best_rank",
            worst_rank_col: "worst_rank",
        }
    )[["name", "best_rank", "worst_rank"]].copy()


def progress_ok(previous: pd.Series, current: pd.Series, width_kind: str) -> bool:
    if width_kind == "score":
        return float(current["best_efficiency"]) + 1e-9 >= float(previous["best_efficiency"])
    return int(current["best_rank"]) <= int(previous["best_rank"])

def main():
    args = parse_args()
    run_dir = create_run_output_dir(args.output_dir, "robustness_width_path")

    df_input = pd.read_csv(args.input)
    if args.target not in df_input["name"].astype(str).tolist():
        raise ValueError(f"Target DMU '{args.target}' not found in input CSV.")

    inputs, outputs = get_io_columns(df_input)
    io_cols = inputs + outputs
    target_row = df_input[df_input["name"].astype(str) == args.target].iloc[0].copy()
    normalization_ranges = normalization_ranges_from_frame(
        df_input,
        io_cols,
        fallback_row=target_row,
    )
    columns_to_modify = parse_columns_arg(args.columns, io_cols)
    grid_steps = candidate_grid_steps(
        df=df_input,
        target_row=target_row,
        columns_to_modify=columns_to_modify,
        pct_above=args.pct_above,
        step_pct=args.step_pct,
        step_abs=args.step_abs,
    )

    eff_csv = run_dir / "extreme_efficiencies.csv"
    export_extreme_efficiencies_with_java(
        input_csv=path_for_java(args.input, args.java_entry),
        output_csv=path_for_java(str(eff_csv), args.java_entry),
        java_entry=args.java_entry,
        main_class=args.efficiencies_main_class,
        maven_executable=args.maven_executable,
    )
    real_metrics = normalize_efficiency_metrics(pd.read_csv(eff_csv))

    ranks_csv = run_dir / "extreme_ranks.csv"
    export_extreme_ranks_with_java(
        input_csv=path_for_java(args.input, args.java_entry),
        output_csv=path_for_java(str(ranks_csv), args.java_entry),
        java_entry=args.java_entry,
        main_class=args.ranks_main_class,
        maven_executable=args.maven_executable,
    )
    real_metrics = real_metrics.merge(normalize_rank_metrics(pd.read_csv(ranks_csv)), on="name", how="left")
    real_metrics["score_width"] = real_metrics["best_efficiency"] - real_metrics["worst_efficiency"]
    real_metrics["rank_width"] = real_metrics["worst_rank"] - real_metrics["best_rank"]
    real_metrics = add_real_state_type(real_metrics, df_input)
    real_metrics = add_effort_columns(
        real_metrics,
        target_row,
        io_cols,
        normalization_ranges=normalization_ranges,
    )
    real_metrics.to_csv(run_dir / "width_metrics.csv", index=False)

    fictive_metrics = pd.DataFrame()
    if args.mode in {"fictive", "mixed"}:
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
        fictive_metrics.to_csv(run_dir / "fictive_width_metrics.csv", index=False)

    width_col = "score_width" if args.width_kind == "score" else "rank_width"
    start_state = real_metrics[real_metrics["name"] == args.target].iloc[0].copy()
    start_width = float(start_state[width_col])
    if args.target_width > start_width:
        raise ValueError("Target width must be <= current robustness width.")

    milestones = linear_milestones(
        start_value=start_width,
        target_value=args.target_width,
        stages=args.stages,
    )
    pd.DataFrame(
        {
            "stage": list(range(args.stages + 1)),
            "milestone_width": milestones,
            "width_kind": args.width_kind,
        }
    ).to_csv(run_dir / "width_milestones.csv", index=False)

    stage_candidates = []
    for stage_idx in range(1, args.stages + 1):
        current_milestone = milestones[stage_idx]
        frames = []

        if args.mode in {"real", "mixed"}:
            eligible_real = real_metrics[real_metrics[width_col] <= current_milestone].copy()
            if not eligible_real.empty:
                eligible_real["milestone_target"] = current_milestone
                eligible_real["milestone_gap"] = (eligible_real[width_col] - current_milestone).abs()
                min_gap = eligible_real["milestone_gap"].min()
                frames.append(eligible_real[eligible_real["milestone_gap"] == min_gap].copy())

        if args.mode in {"fictive", "mixed"} and not fictive_metrics.empty:
            eligible_fictive = fictive_metrics[fictive_metrics[width_col] <= current_milestone].copy()
            if not eligible_fictive.empty:
                eligible_fictive["milestone_target"] = current_milestone
                eligible_fictive["milestone_gap"] = (eligible_fictive[width_col] - current_milestone).abs()
                if args.refine_fictive_candidates:
                    eligible_fictive = refine_numeric_goal_candidates(
                        seed_metrics=eligible_fictive,
                        target_row=target_row,
                        io_cols=io_cols,
                        inputs=inputs,
                        outputs=outputs,
                        metric_col=width_col,
                        threshold=current_milestone,
                        direction="lower",
                        run_dir=run_dir,
                        reference_csv=args.input,
                        java_entry=args.java_entry,
                        main_class=args.candidate_metrics_main_class,
                        maven_executable=args.maven_executable,
                        name_prefix=f"stage_{stage_idx:02d}_{args.width_kind}_width",
                        iterations=args.refine_iterations,
                        max_seed_candidates=args.refine_max_seeds,
                        search_columns=columns_to_modify,
                        local_step_by_column=grid_steps,
                        local_random_samples=args.local_search_samples,
                        local_random_step_multiplier=args.local_search_step_multiplier,
                        local_random_state=args.local_search_random_state + stage_idx,
                        local_sampling_strategy=args.local_search_sampling,
                        prune_seed_front=False,
                        prune_front=False,
                    )
                    eligible_fictive["milestone_target"] = current_milestone
                    eligible_fictive["milestone_gap"] = (eligible_fictive[width_col] - current_milestone).abs()
                eligible_fictive = add_effort_columns(
                    eligible_fictive,
                    target_row,
                    io_cols,
                    normalization_ranges=normalization_ranges,
                )
                frames.append(eligible_fictive)

        if not frames:
            raise ValueError(f"No {args.mode} candidates found for width milestone stage {stage_idx}.")

        stage = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["name"], keep="last")
            .reset_index(drop=True)
        )
        stage_candidates.append(stage)

    write_stage_candidates(stage_candidates, str(run_dir / "stage_candidates.csv"))

    transition_log = []
    paths = enumerate_state_paths(
        start_row=start_state,
        stage_candidates=stage_candidates,
        inputs=inputs,
        outputs=outputs,
        max_paths=args.max_paths,
        points_per_transition=args.points_per_stage,
        normalization_ranges=normalization_ranges,
        minimal_inputs=[col for col in inputs if col in columns_to_modify],
        minimal_outputs=[col for col in outputs if col in columns_to_modify],
        transition_predicate=lambda previous, current: progress_ok(
            previous,
            current,
            args.width_kind,
        ),
        transition_log=transition_log,
    )
    pd.DataFrame(transition_log).to_csv(
        run_dir / "transition_candidates.csv",
        index=False,
    )
    final_paths_frame = state_paths_to_frame(paths, io_cols)
    final_paths_frame.to_csv(run_dir / "paths.csv", index=False)
    write_path_metrics(
        final_paths_frame,
        run_dir / "path_metrics.csv",
        method_name="robustness_width_path",
        io_columns=io_cols,
        normalization_ranges=normalization_ranges,
    )


if __name__ == "__main__":
    main()
