import argparse
from pathlib import Path

import pandas as pd


SUMMARY_METRICS = [
    "path_length",
    "tc",
    "msc",
    "cdir",
    "dr",
    "bp",
    "wbp",
    "sbp",
    "swbp",
    "mcp",
    "md",
    "pyv",
    "pym",
    "apw",
    "fw",
    "ww",
    "pc",
    "opp",
    "rr",
    "final_effort_from_start",
    "total_effort_movement",
    "total_milestone_gap",
    "attainable_transition_violations",
    "best_efficiency_improvement",
    "worst_efficiency_improvement",
    "best_rank_improvement",
    "worst_rank_improvement",
    "score_width_reduction",
    "rank_width_reduction",
]


def collect_path_metrics(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_files = sorted(experiment_dir.rglob("path_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(
            f"No path_metrics.csv files found below: {experiment_dir}"
        )

    frames = []
    for metric_file in metric_files:
        frame = pd.read_csv(metric_file)
        frame.insert(0, "run_dir", str(metric_file.parent.relative_to(experiment_dir)))
        frames.append(frame)

    all_metrics = pd.concat(frames, ignore_index=True, sort=False)
    if all_metrics.empty:
        return all_metrics, pd.DataFrame(columns=["method", "path_count"])

    summary_rows = []
    for method, group in all_metrics.groupby("method", dropna=False):
        row = {
            "method": method,
            "path_count": len(group),
        }
        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            if not values.notna().any():
                continue
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        summary_rows.append(row)

    return all_metrics, pd.DataFrame(summary_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--all-output", default="all_path_metrics.csv")
    parser.add_argument("--summary-output", default="method_summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    all_metrics, method_summary = collect_path_metrics(experiment_dir)

    all_output = experiment_dir / args.all_output
    summary_output = experiment_dir / args.summary_output
    all_metrics.to_csv(all_output, index=False)
    method_summary.to_csv(summary_output, index=False)

    print(f"Collected path rows: {len(all_metrics)}")
    print(f"All metrics: {all_output}")
    print(f"Method summary: {summary_output}")


if __name__ == "__main__":
    main()
