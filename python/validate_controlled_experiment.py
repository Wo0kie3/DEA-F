import argparse
from pathlib import Path

import pandas as pd


METHODS = [
    "hasse_path",
    "front_path",
    "best_rank_path",
    "best_efficiency_path",
    "robustness_width_path",
]
ROBUST_FIELDS = [
    "best_efficiency",
    "worst_efficiency",
    "best_rank",
    "worst_rank",
    "score_width",
    "rank_width",
]
TOLERANCE = 1e-9


def latest_run(method_dir: Path) -> Path:
    runs = sorted(path for path in method_dir.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run below {method_dir}.")
    return runs[-1]


def validate_experiment(
    root: Path,
    fixed_columns: list[str],
    max_paths: int | None,
    max_stage_effort: float | None,
) -> pd.DataFrame:
    checks = []

    def record(method: str, check: str, passed: bool, details: str):
        checks.append(
            {"method": method, "check": check, "passed": bool(passed), "details": details}
        )

    for method in METHODS:
        try:
            run_dir = latest_run(root / method)
            paths = pd.read_csv(run_dir / "paths.csv")
            metrics = pd.read_csv(run_dir / "path_metrics.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError) as error:
            record(method, "complete_run", False, str(error))
            continue

        record(method, "nonempty_complete_paths", not paths.empty, f"path_count={len(paths)}")
        if max_paths is not None:
            record(
                method,
                "not_truncated_at_max_paths",
                len(paths) < max_paths,
                f"path_count={len(paths)}, max_paths={max_paths}",
            )

        missing_start = [
            field
            for field in ROBUST_FIELDS
            if f"stage_00_{field}" not in paths.columns
            or paths[f"stage_00_{field}"].isna().any()
        ]
        record(
            method,
            "complete_start_robust_metrics",
            not missing_start,
            "missing=" + ("|".join(missing_start) if missing_start else "none"),
        )

        fixed_violations = 0
        for column in fixed_columns:
            start_column = f"stage_00_{column}"
            if start_column not in paths.columns:
                fixed_violations += len(paths)
                continue
            for stage in range(1, int(paths["path_length"].max()) + 1):
                current_column = f"stage_{stage:02d}_{column}"
                if current_column not in paths.columns:
                    continue
                comparable = paths[[start_column, current_column]].dropna()
                fixed_violations += int(
                    ((comparable[current_column] - comparable[start_column]).abs() > TOLERANCE).sum()
                )
        record(
            method,
            "fixed_columns_unchanged",
            fixed_violations == 0,
            f"violations={fixed_violations}",
        )

        transition_reference_violations = 0
        tc_violations = 0
        stage_budget_violations = 0
        metrics_by_path = metrics.set_index("path_id")
        for _, path in paths.iterrows():
            efforts = []
            for stage in range(1, int(path["path_length"]) + 1):
                prefix = f"stage_{stage:02d}"
                previous_prefix = f"stage_{stage - 1:02d}"
                reference = path.get(f"{prefix}_transition_reference_name")
                previous_name = path.get(f"{previous_prefix}_name")
                if pd.isna(reference) or str(reference) != str(previous_name):
                    transition_reference_violations += 1
                effort = path.get(f"{prefix}_effort_from_previous")
                if pd.notna(effort):
                    efforts.append(float(effort))
                    if max_stage_effort is not None and float(effort) > max_stage_effort + TOLERANCE:
                        stage_budget_violations += 1
            path_id = str(path["path_id"])
            if path_id not in metrics_by_path.index or abs(
                sum(efforts) - float(metrics_by_path.loc[path_id, "tc"])
            ) > TOLERANCE:
                tc_violations += 1
        record(
            method,
            "actual_predecessor_references",
            transition_reference_violations == 0,
            f"violations={transition_reference_violations}",
        )
        record(method, "tc_equals_step_sum", tc_violations == 0, f"violations={tc_violations}")
        record(
            method,
            "aggregate_stage_budget",
            stage_budget_violations == 0,
            f"violations={stage_budget_violations}",
        )

        metric_checks = {
            "attainability": pd.to_numeric(
                metrics["attainable_transition_violations"], errors="coerce"
            ).fillna(1).eq(0).all(),
            "complete_md": "md" in metrics.columns and metrics["md"].notna().all(),
            "declared_cdir_scope": "cdir_scope" in metrics.columns
            and metrics["cdir_scope"].notna().all(),
            "declared_path_kind": "path_kind" in metrics.columns
            and metrics["path_kind"].notna().all(),
        }
        for check, passed in metric_checks.items():
            record(method, check, bool(passed), f"path_count={len(metrics)}")

        selection_file = run_dir / "path_selection.csv"
        pareto_file = run_dir / "path_pareto.csv"
        representatives_file = run_dir / "path_pareto_representatives.csv"
        selection_ok = False
        if selection_file.is_file():
            selection = pd.read_csv(selection_file)
            selection_ok = (selection["selection_type"] == "lexicographic").any()
        record(
            method,
            "publication_selection_outputs",
            selection_ok and pareto_file.is_file() and representatives_file.is_file(),
            f"run={run_dir.name}",
        )

    return pd.DataFrame(checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--fixed-columns", default="")
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--max-stage-effort", type=float, default=None)
    parser.add_argument("--output", default="validation_checks.csv")
    args = parser.parse_args()

    fixed_columns = [part.strip() for part in args.fixed_columns.split(",") if part.strip()]
    checks = validate_experiment(
        args.root.resolve(),
        fixed_columns,
        args.max_paths,
        args.max_stage_effort,
    )
    output = args.root.resolve() / args.output
    checks.to_csv(output, index=False)
    failed = checks[~checks["passed"]]
    print(f"Validation checks: {len(checks)}; failed: {len(failed)}")
    print(output)
    if not failed.empty:
        print(failed.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
