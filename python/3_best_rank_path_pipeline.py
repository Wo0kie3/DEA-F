import argparse

import pandas as pd

from java_runner import (
    export_candidate_robust_metrics_with_java,
    export_extreme_ranks_with_java,
)
from path_pipeline_common import (
    add_effort_columns,
    add_real_state_type,
    create_run_output_dir,
    enumerate_state_paths,
    generate_attainable_fictive_candidates,
    get_io_columns,
    parse_columns_arg,
    path_for_java,
    rank_milestones,
    resolve_first_present,
    state_paths_to_frame,
    write_stage_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-best-rank", type=int, required=True)
    parser.add_argument("--stages", type=int, required=True)
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
        "--ranks-main-class",
        default="org.example.CsvExtremeRanksExporter",
    )
    parser.add_argument(
        "--candidate-metrics-main-class",
        default="org.example.CsvCandidateRobustMetricsExporter",
    )
    parser.add_argument("--maven-executable", default="mvn")
    parser.add_argument("--max-paths", type=int, default=None)
    return parser.parse_args()


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
    metrics = df_ranks.rename(
        columns={
            name_col: "name",
            best_rank_col: "best_rank",
            worst_rank_col: "worst_rank",
        }
    ).copy()
    metrics["best_rank"] = metrics["best_rank"].astype(int)
    metrics["worst_rank"] = metrics["worst_rank"].astype(int)
    metrics["rank_width"] = metrics["worst_rank"] - metrics["best_rank"]
    return metrics


def limit_stage(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    frame = frame.sort_values(
        by=["milestone_gap", "effort_from_start", "name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    if limit is not None and limit >= 0:
        frame = frame.head(limit).copy()
    return frame


def main():
    args = parse_args()
    run_dir = create_run_output_dir(args.output_dir, "best_rank_path")

    df_input = pd.read_csv(args.input)
    if args.target not in df_input["name"].astype(str).tolist():
        raise ValueError(f"Target DMU '{args.target}' not found in input CSV.")

    inputs, outputs = get_io_columns(df_input)
    io_cols = inputs + outputs
    target_row = df_input[df_input["name"].astype(str) == args.target].iloc[0].copy()
    columns_to_modify = parse_columns_arg(args.columns, io_cols)

    ranks_csv = run_dir / "extreme_ranks.csv"
    export_extreme_ranks_with_java(
        input_csv=path_for_java(args.input, args.java_entry),
        output_csv=path_for_java(str(ranks_csv), args.java_entry),
        java_entry=args.java_entry,
        main_class=args.ranks_main_class,
        maven_executable=args.maven_executable,
    )

    real_metrics = normalize_rank_metrics(pd.read_csv(ranks_csv))
    real_metrics = add_real_state_type(real_metrics[["name", "best_rank", "worst_rank", "rank_width"]], df_input)
    real_metrics = add_effort_columns(real_metrics, target_row, io_cols)
    real_metrics.to_csv(run_dir / "rank_metrics.csv", index=False)

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
        fictive_metrics = add_effort_columns(fictive_metrics, target_row, io_cols)
        fictive_metrics.to_csv(run_dir / "fictive_rank_metrics.csv", index=False)

    start_best_rank = int(real_metrics.loc[real_metrics["name"] == args.target, "best_rank"].iloc[0])
    if args.target_best_rank > start_best_rank:
        raise ValueError(
            f"Target best rank must be <= current best rank ({start_best_rank})."
        )

    milestones = rank_milestones(
        start_rank=start_best_rank,
        target_rank=args.target_best_rank,
        stages=args.stages,
    )
    pd.DataFrame(
        {
            "stage": list(range(args.stages + 1)),
            "milestone_best_rank": milestones,
        }
    ).to_csv(run_dir / "rank_milestones.csv", index=False)

    stage_candidates = []
    for stage_idx in range(1, args.stages + 1):
        prev_milestone = milestones[stage_idx - 1]
        current_milestone = milestones[stage_idx]
        frames = []

        if args.mode in {"real", "mixed"}:
            eligible_real = real_metrics[real_metrics["best_rank"] <= prev_milestone].copy()
            if not eligible_real.empty:
                eligible_real["milestone_target"] = current_milestone
                eligible_real["milestone_gap"] = (
                    eligible_real["best_rank"] - current_milestone
                ).abs()
                min_gap = eligible_real["milestone_gap"].min()
                frames.append(eligible_real[eligible_real["milestone_gap"] == min_gap].copy())

        if args.mode in {"fictive", "mixed"} and not fictive_metrics.empty:
            eligible_fictive = fictive_metrics[fictive_metrics["best_rank"] <= current_milestone].copy()
            if not eligible_fictive.empty:
                eligible_fictive["milestone_target"] = current_milestone
                eligible_fictive["milestone_gap"] = (
                    eligible_fictive["best_rank"] - current_milestone
                ).abs()
                frames.append(eligible_fictive)

        if not frames:
            raise ValueError(f"No {args.mode} candidates found for rank milestone stage {stage_idx}.")

        stage = pd.concat(frames, ignore_index=True)
        stage_candidates.append(limit_stage(stage, args.points_per_stage))

    write_stage_candidates(stage_candidates, str(run_dir / "stage_candidates.csv"))

    start_state = real_metrics[real_metrics["name"] == args.target].iloc[0].copy()
    paths = enumerate_state_paths(
        start_row=start_state,
        stage_candidates=stage_candidates,
        inputs=inputs,
        outputs=outputs,
        max_paths=args.max_paths,
    )
    state_paths_to_frame(paths, io_cols).to_csv(run_dir / "paths.csv", index=False)


if __name__ == "__main__":
    main()
