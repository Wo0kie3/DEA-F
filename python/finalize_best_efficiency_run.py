import argparse
from pathlib import Path

import pandas as pd

from path_metrics import write_path_metrics
from path_pipeline_common import (
    add_effort_columns,
    enumerate_state_paths,
    get_io_columns,
    normalization_ranges_from_frame,
    parse_columns_arg,
    state_paths_to_frame,
    write_stage_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--columns", default=None)
    parser.add_argument("--points-per-stage", type=int, default=None)
    parser.add_argument("--max-paths", type=int, default=0)
    return parser.parse_args()


def read_stage_results(
    run_dir: Path,
    stage: int,
    threshold: float,
    target_row: pd.Series,
    io_columns: list[str],
    normalization_ranges: dict[str, float],
    global_metrics: pd.DataFrame,
) -> pd.DataFrame:
    prefix = f"stage_{stage:02d}_eff"
    frames = []
    for suffix in ["refined_final_metrics.csv", "local_search_metrics.csv"]:
        path = run_dir / f"{prefix}_{suffix}"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[
            pd.to_numeric(frame["best_efficiency"], errors="coerce") + 1e-9
            >= threshold
        ].copy()
        if not frame.empty:
            frames.append(frame)

    if not global_metrics.empty:
        global_eligible = global_metrics[
            pd.to_numeric(global_metrics["best_efficiency"], errors="coerce") + 1e-9
            >= threshold
        ].copy()
        if not global_eligible.empty:
            frames.append(global_eligible)

    if not frames:
        raise ValueError(f"No saved candidates satisfy stage {stage} threshold {threshold}.")

    stage_frame = (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates(subset=["name"], keep="last")
        .reset_index(drop=True)
    )
    stage_frame = add_effort_columns(
        stage_frame,
        target_row,
        io_columns,
        normalization_ranges=normalization_ranges,
    )
    stage_frame["milestone_target"] = threshold
    stage_frame["milestone_gap"] = (
        pd.to_numeric(stage_frame["best_efficiency"], errors="coerce") - threshold
    ).abs()
    return stage_frame


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    input_frame = pd.read_csv(args.input)
    inputs, outputs = get_io_columns(input_frame)
    io_columns = [*inputs, *outputs]
    columns_to_modify = parse_columns_arg(args.columns, io_columns)
    target_row = input_frame[
        input_frame["name"].astype(str) == str(args.target)
    ].iloc[0]
    normalization_ranges = normalization_ranges_from_frame(
        input_frame,
        io_columns,
        fallback_row=target_row,
    )

    real_metrics = pd.read_csv(run_dir / "efficiency_metrics.csv")
    start_state = real_metrics[
        real_metrics["name"].astype(str) == str(args.target)
    ].iloc[0]
    milestones = pd.read_csv(run_dir / "efficiency_milestones.csv").set_index("stage")
    global_path = run_dir / "global_search_metrics.csv"
    global_metrics = pd.read_csv(global_path) if global_path.exists() else pd.DataFrame()

    stage_candidates = []
    for stage in sorted(int(value) for value in milestones.index if int(value) > 0):
        threshold = float(milestones.loc[stage, "milestone_best_efficiency"])
        stage_candidates.append(
            read_stage_results(
                run_dir=run_dir,
                stage=stage,
                threshold=threshold,
                target_row=target_row,
                io_columns=io_columns,
                normalization_ranges=normalization_ranges,
                global_metrics=global_metrics,
            )
        )

    write_stage_candidates(stage_candidates, str(run_dir / "stage_candidates.csv"))
    transition_log = []
    max_paths = args.max_paths if args.max_paths > 0 else None
    paths = enumerate_state_paths(
        start_row=start_state,
        stage_candidates=stage_candidates,
        inputs=inputs,
        outputs=outputs,
        max_paths=max_paths,
        points_per_transition=args.points_per_stage,
        normalization_ranges=normalization_ranges,
        minimal_inputs=[col for col in inputs if col in columns_to_modify],
        minimal_outputs=[col for col in outputs if col in columns_to_modify],
        transition_log=transition_log,
    )
    pd.DataFrame(transition_log).to_csv(
        run_dir / "transition_candidates.csv",
        index=False,
    )
    paths_frame = state_paths_to_frame(paths, io_columns)
    paths_frame.to_csv(run_dir / "paths.csv", index=False)
    write_path_metrics(
        paths_frame,
        run_dir / "path_metrics.csv",
        method_name="best_efficiency_path",
        io_columns=io_columns,
        normalization_ranges=normalization_ranges,
    )
    print(f"Finalized stage pools: {[len(frame) for frame in stage_candidates]}")
    print(f"Finalized transition candidates: {len(transition_log)}")
    print(f"Finalized paths: {len(paths_frame)}")


if __name__ == "__main__":
    main()
