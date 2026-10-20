"""Run a reproducible input-only comparison, sequentially, on any DEA CSV."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from path_pipeline_common import get_io_columns, normalization_ranges_from_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--columns", default="i1,i2")
    parser.add_argument("--target-best-rank", type=int, default=1)
    parser.add_argument("--target-best-efficiency", type=float, default=0.95)
    parser.add_argument("--target-width", type=float, default=0.15)
    parser.add_argument("--width-kind", choices=["score", "rank"], default="score")
    parser.add_argument("--step-pct", type=float, default=5)
    parser.add_argument("--points-per-transition", type=int, default=15)
    parser.add_argument("--max-paths", type=int, default=20000)
    parser.add_argument("--local-samples", type=int, default=36)
    parser.add_argument("--global-samples", type=int, default=400)
    parser.add_argument("--reuse-snapshot", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("*/run_*")):
        parser.error("Use a new output root; existing method runs must not be mixed with this experiment")
    input_path = args.input.resolve()
    data = pd.read_csv(input_path)
    inputs, outputs = get_io_columns(data)
    target = data.set_index("name").loc[args.target]
    columns = args.columns.split(",")
    if len(columns) not in (2, 3) or not set(columns).issubset(inputs):
        parser.error("--columns must name two or three inputs")
    factor_weights = ",".join(f"{column}=1" for column in columns)
    fixed = [c for c in inputs + outputs if c not in columns]
    params = {
        **vars(args), "input": str(input_path), "root": str(root),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "mode": "fictive", "dimensions": args.columns, "fixed_columns": ",".join(fixed),
        "stages_numeric_methods": 3, "stages_structural_methods": "determined by NEC DAG",
        "factor_weights": factor_weights, "pct_above": 5,
        "max_candidates": 500, "min_points_per_dim": 5,
        "refine_iterations": 4, "refine_max_seeds": 6,
        "local_search_step_multiplier": 1.0, "local_search_random_state": 42,
        "local_search_sampling": "stratified", "global_search_random_state": 142,
        "max_stage_effort": "none", "max_factor_changes": "none",
        "stage_weights": "equal", "width_state_weights": "equal",
        "normalization_ranges": json.dumps(normalization_ranges_from_frame(data, inputs + outputs, target)),
        "publication_selection": "lexicographic; min TC also shown",
        "candidate_pool_note": "common base grid; methods 3-5 refinement; method 4 also global sampling",
        "started_at": datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    }
    pd.DataFrame(params.items(), columns=["parameter", "value"]).to_csv(root / "experiment_params.csv", index=False)
    (root / "git_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True), encoding="utf-8")
    records = []

    def run(label, script, arguments):
        command = [sys.executable, str(repo / "python" / script), *map(str, arguments)]
        start = time.perf_counter()
        print(f"Starting {label}", flush=True)
        with (root / f"{label}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        records.append({"step": label, "returncode": result.returncode,
                        "elapsed_seconds": round(time.perf_counter() - start, 3),
                        "command_json": json.dumps(command)})
        pd.DataFrame(records).to_csv(root / "execution_status.csv", index=False)
        print(f"Finished {label}: exit={result.returncode}, seconds={records[-1]['elapsed_seconds']}", flush=True)
        return result.returncode

    if args.reuse_snapshot:
        saved = pd.read_csv(root / "dea_input_snapshot" / "decision_data.csv")
        pd.testing.assert_frame_equal(data, saved, check_dtype=False)
    elif run("snapshot", "generate_dea_input_snapshot.py", ["--input", input_path, "--output-dir", root / "dea_input_snapshot", "--java-entry", repo / "java"]):
        raise SystemExit("Snapshot failed; see snapshot.log")

    common = ["--input", input_path, "--target", args.target, "--output-dir", root,
              "--java-entry", repo / "java", "--mode", "fictive", "--columns", args.columns,
              "--pct-above", 5, "--step-pct", args.step_pct, "--min-points-per-dim", 5,
              "--max-candidates", 500, "--points-per-stage", args.points_per_transition,
              "--max-paths", args.max_paths, "--factor-weights", factor_weights]
    refinement = ["--stages", 3, "--refine-fictive-candidates", "--refine-iterations", 4,
                  "--refine-max-seeds", 6, "--local-search-samples", args.local_samples,
                  "--local-search-step-multiplier", 1, "--local-search-random-state", 42,
                  "--local-search-sampling", "stratified"]
    methods = [
        ("hasse_path", "1_hasse_path_pipeline.py", []),
        ("front_path", "2_necessary_layer_path_pipeline.py", []),
        ("best_rank_path", "3_best_rank_path_pipeline.py", refinement + ["--target-best-rank", args.target_best_rank]),
        ("best_efficiency_path", "4_best_efficiency_path_pipeline.py", refinement + ["--target-best-efficiency", args.target_best_efficiency, "--global-search-samples", args.global_samples, "--global-search-random-state", 142]),
        ("robustness_width_path", "5_robustness_width_path_pipeline.py", refinement + ["--target-width", args.target_width, "--width-kind", args.width_kind]),
    ]
    failures = [label for label, script, extra in methods if run(label, script, common + extra)]
    completion_code = run("metric_completion", "complete_experiment_metrics.py", ["--root", root, "--java-entry", repo / "java"]) if not failures else 1
    if completion_code:
        failures.append("metric_completion")
    run("collection", "collect_path_metrics.py", ["--experiment-dir", root])
    validation_code = run("validation", "validate_controlled_experiment.py", ["--root", root, "--fixed-columns", ",".join(fixed), "--max-paths", args.max_paths])
    saved_params = pd.read_csv(root / "experiment_params.csv", keep_default_na=False)
    params.update(dict(zip(saved_params.parameter, saved_params.value)))
    params.update(completed_at=datetime.now().astimezone().isoformat(),
                  status="requires_review" if failures or validation_code else "success")
    pd.DataFrame(params.items(), columns=["parameter", "value"]).to_csv(root / "experiment_params.csv", index=False)
    report_code = run("report", "generate_five_method_report.py", ["--root", root, "--input", input_path, "--target", args.target, "--output", root / "five_method_report.html"])
    if failures or report_code:
        raise SystemExit(f"Failed steps: {failures}; report exit: {report_code}")
    print(root / "five_method_report.html")


if __name__ == "__main__":
    main()
