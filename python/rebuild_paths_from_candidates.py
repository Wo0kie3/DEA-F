import argparse
from pathlib import Path

import pandas as pd

from path_metrics import write_path_metrics
from path_pipeline_common import (
    enumerate_state_paths,
    get_io_columns,
    normalization_ranges_from_frame,
    parse_columns_arg,
    state_paths_to_frame,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--method", default="best_efficiency_path")
    parser.add_argument("--max-paths", type=int, default=0)
    parser.add_argument("--points-per-transition", type=int, default=None)
    parser.add_argument("--columns", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    input_frame = pd.read_csv(args.input)
    inputs, outputs = get_io_columns(input_frame)
    io_columns = inputs + outputs
    columns_to_modify = parse_columns_arg(args.columns, io_columns)

    metrics = pd.read_csv(run_dir / "efficiency_metrics.csv")
    start = metrics[metrics["name"].astype(str) == str(args.target)]
    if start.empty:
        raise ValueError(f"Target {args.target} not found in efficiency_metrics.csv.")
    normalization_ranges = normalization_ranges_from_frame(
        input_frame,
        io_columns,
        fallback_row=start.iloc[0],
    )

    candidates = pd.read_csv(run_dir / "stage_candidates.csv")
    stage_candidates = [
        group.reset_index(drop=True)
        for _, group in candidates.groupby("stage", sort=True)
    ]
    max_paths = args.max_paths if args.max_paths > 0 else None
    transition_log = []
    paths = enumerate_state_paths(
        start_row=start.iloc[0],
        stage_candidates=stage_candidates,
        inputs=inputs,
        outputs=outputs,
        max_paths=max_paths,
        points_per_transition=args.points_per_transition,
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
        method_name=args.method,
        io_columns=io_columns,
        normalization_ranges=normalization_ranges,
    )
    print(f"Rebuilt attainable paths: {len(paths_frame)}")
    print(f"Paths: {run_dir / 'paths.csv'}")
    print(f"Metrics: {run_dir / 'path_metrics.csv'}")


if __name__ == "__main__":
    main()
