from pathlib import Path
from itertools import product
import math
import random

import pandas as pd

from java_runner import export_candidate_robust_metrics_with_java
from path_pipeline_common import add_effort_columns, path_for_java


def satisfies_numeric_goal(value, threshold: float, direction: str, tolerance: float = 1e-9) -> bool:
    if pd.isna(value):
        return False
    value = float(value)
    if direction == "higher":
        return value + tolerance >= float(threshold)
    if direction == "lower":
        return value <= float(threshold) + tolerance
    raise ValueError(f"Unsupported refinement direction: {direction}")


def _candidate_at_alpha(
    source_row: pd.Series,
    target_row: pd.Series,
    io_cols: list[str],
    alpha: float,
    name: str,
) -> dict:
    row = {"name": name, "state_type": "fictive"}
    for col in io_cols:
        start = float(target_row[col])
        end = float(source_row[col])
        row[col] = start + float(alpha) * (end - start)
    return row


def _stratified_unit_points(samples: int, dimension_count: int, rng) -> list[list[float]]:
    if samples <= 0 or dimension_count <= 0:
        return []
    if dimension_count not in {1, 2, 3}:
        return [
            [rng.random() for _ in range(dimension_count)]
            for _ in range(samples)
        ]

    side = math.ceil(samples ** (1.0 / dimension_count))
    cells = list(product(range(side), repeat=dimension_count))
    rng.shuffle(cells)
    return [
        [
            (cell[dim] + rng.random()) / side
            for dim in range(dimension_count)
        ]
        for cell in cells[:samples]
    ]


def _local_random_candidates(
    centers: pd.DataFrame,
    target_row: pd.Series,
    io_cols: list[str],
    inputs: list[str],
    outputs: list[str],
    search_columns: list[str],
    step_by_column: dict[str, float],
    samples_per_center: int,
    step_multiplier: float,
    random_state: int,
    name_prefix: str,
    sampling_strategy: str = "random",
) -> pd.DataFrame:
    if centers.empty or samples_per_center <= 0:
        return pd.DataFrame()

    rng = random.Random(random_state)
    rows = []
    search_columns = [col for col in search_columns if col in io_cols]
    for center_idx, (_, center) in enumerate(centers.iterrows(), start=1):
        unit_points = None
        if sampling_strategy == "stratified":
            unit_points = _stratified_unit_points(
                samples_per_center,
                len(search_columns),
                rng,
            )
        elif sampling_strategy != "random":
            raise ValueError(f"Unsupported local sampling strategy: {sampling_strategy}")

        for sample_idx in range(1, samples_per_center + 1):
            row = {
                "name": f"{name_prefix}_local_{center_idx:04d}_{sample_idx:03d}",
                "state_type": "fictive",
            }
            for col in io_cols:
                row[col] = float(center[col])

            for column_idx, col in enumerate(search_columns):
                center_value = float(center[col])
                start_value = float(target_row[col])
                radius = max(float(step_by_column.get(col, 0.0)) * float(step_multiplier), 1e-12)
                if sampling_strategy == "stratified":
                    if col in inputs:
                        low = max(0.0, center_value - radius)
                        high = min(start_value, center_value + radius)
                    elif col in outputs:
                        low = max(start_value, center_value - radius)
                        high = center_value + radius
                    else:
                        low = center_value - radius
                        high = center_value + radius
                    sampled = low + unit_points[sample_idx - 1][column_idx] * (high - low)
                else:
                    sampled = center_value + rng.uniform(-radius, radius)
                    if col in inputs:
                        sampled = min(start_value, max(0.0, sampled))
                    elif col in outputs:
                        sampled = max(start_value, sampled)
                row[col] = sampled
                row[f"local_search_step_{col}"] = radius

            row["local_search_center_name"] = center.get("name", "")
            row["local_search_center_index"] = center_idx
            row["local_search_sample_index"] = sample_idx
            row["local_search_step_multiplier"] = step_multiplier
            row["local_search_sampling_strategy"] = sampling_strategy
            rows.append(row)

    return pd.DataFrame(rows)


def global_stratified_candidates(
    reference: pd.DataFrame,
    target_row: pd.Series,
    io_cols: list[str],
    inputs: list[str],
    outputs: list[str],
    search_columns: list[str],
    pct_above: float,
    samples: int,
    random_state: int,
    name_prefix: str,
) -> pd.DataFrame:
    if samples <= 0:
        return pd.DataFrame()

    search_columns = [col for col in search_columns if col in io_cols]
    rng = random.Random(random_state)
    unit_points = _stratified_unit_points(samples, len(search_columns), rng)
    bounds = {}
    for col in search_columns:
        start = float(target_row[col])
        observed = pd.to_numeric(reference[col], errors="coerce").dropna()
        if col in inputs:
            bounds[col] = (
                max(0.0, float(observed.min()) * (1.0 - pct_above / 100.0)),
                start,
            )
        elif col in outputs:
            bounds[col] = (
                start,
                float(observed.max()) * (1.0 + pct_above / 100.0),
            )
        else:
            raise ValueError(f"Column {col} is not a DEA input or output.")

    rows = []
    for sample_idx, unit_point in enumerate(unit_points, start=1):
        row = {
            "name": f"{name_prefix}_global_{sample_idx:06d}",
            "state_type": "fictive",
            "global_search_sample_index": sample_idx,
            "global_search_sampling_strategy": "stratified",
        }
        for col in io_cols:
            row[col] = float(target_row[col])
        for column_idx, col in enumerate(search_columns):
            low, high = bounds[col]
            row[col] = low + unit_point[column_idx] * (high - low)
            row[f"global_search_low_{col}"] = low
            row[f"global_search_high_{col}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_candidates(
    candidates: pd.DataFrame,
    run_dir: Path,
    reference_csv: str,
    java_entry: str,
    main_class: str,
    maven_executable: str,
    output_stem: str,
    batch_size: int = 500,
) -> pd.DataFrame:
    candidates_csv = run_dir / f"{output_stem}_candidates.csv"
    metrics_csv = run_dir / f"{output_stem}_metrics.csv"
    candidates.to_csv(candidates_csv, index=False)

    if len(candidates) <= batch_size:
        export_candidate_robust_metrics_with_java(
            reference_csv=path_for_java(reference_csv, java_entry),
            candidates_csv=path_for_java(str(candidates_csv), java_entry),
            output_csv=path_for_java(str(metrics_csv), java_entry),
            java_entry=java_entry,
            main_class=main_class,
            maven_executable=maven_executable,
        )
        return pd.read_csv(metrics_csv)

    metric_batches = []
    batch_count = (len(candidates) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(candidates), batch_size), start=1):
        batch = candidates.iloc[start:start + batch_size].copy()
        batch_stem = f"{output_stem}_batch_{batch_index:03d}_of_{batch_count:03d}"
        batch_candidates_csv = run_dir / f"{batch_stem}_candidates.csv"
        batch_metrics_csv = run_dir / f"{batch_stem}_metrics.csv"
        batch.to_csv(batch_candidates_csv, index=False)
        print(
            f"Evaluating candidate batch {batch_index}/{batch_count} "
            f"({len(batch)} rows) for {output_stem}"
        )
        export_candidate_robust_metrics_with_java(
            reference_csv=path_for_java(reference_csv, java_entry),
            candidates_csv=path_for_java(str(batch_candidates_csv), java_entry),
            output_csv=path_for_java(str(batch_metrics_csv), java_entry),
            java_entry=java_entry,
            main_class=main_class,
            maven_executable=maven_executable,
        )
        metric_batches.append(pd.read_csv(batch_metrics_csv))

    metrics = pd.concat(metric_batches, ignore_index=True)
    metrics.to_csv(metrics_csv, index=False)
    return metrics


def prune_minimal_change_front(
    frame: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    rows_to_keep = []
    records = list(frame.to_dict("records"))
    for idx, candidate in enumerate(records):
        dominated = False
        for other_idx, other in enumerate(records):
            if idx == other_idx:
                continue

            no_more_input_reduction = all(
                float(other[col]) + tolerance >= float(candidate[col])
                for col in inputs
            )
            no_more_output_increase = all(
                float(other[col]) <= float(candidate[col]) + tolerance
                for col in outputs
            )
            strictly_less_change = any(
                float(other[col]) > float(candidate[col]) + tolerance
                for col in inputs
            ) or any(
                float(other[col]) + tolerance < float(candidate[col])
                for col in outputs
            )
            if no_more_input_reduction and no_more_output_increase and strictly_less_change:
                dominated = True
                break

        if not dominated:
            rows_to_keep.append(idx)

    return frame.iloc[rows_to_keep].copy().reset_index(drop=True)


def refine_numeric_goal_candidates(
    seed_metrics: pd.DataFrame,
    target_row: pd.Series,
    io_cols: list[str],
    inputs: list[str],
    outputs: list[str],
    metric_col: str,
    threshold: float,
    direction: str,
    run_dir: Path,
    reference_csv: str,
    java_entry: str,
    main_class: str,
    maven_executable: str,
    name_prefix: str,
    iterations: int = 12,
    max_seed_candidates: int | None = None,
    search_columns: list[str] | None = None,
    local_step_by_column: dict[str, float] | None = None,
    local_random_samples: int = 0,
    local_random_step_multiplier: float = 1.0,
    local_random_state: int = 42,
    local_sampling_strategy: str = "random",
    prune_seed_front: bool = True,
    prune_front: bool = True,
) -> pd.DataFrame:
    if seed_metrics.empty:
        return seed_metrics.copy()

    eligible = seed_metrics[
        seed_metrics[metric_col].apply(lambda value: satisfies_numeric_goal(value, threshold, direction))
    ].copy()
    if eligible.empty:
        return eligible

    seed_candidates_before_front = len(eligible)
    if prune_seed_front:
        active_columns = set(search_columns or io_cols)
        eligible = prune_minimal_change_front(
            eligible,
            inputs=[col for col in inputs if col in active_columns],
            outputs=[col for col in outputs if col in active_columns],
        )
    seed_candidates_after_front = len(eligible)

    sort_cols = [col for col in ["milestone_gap", "name"] if col in eligible.columns]
    if sort_cols:
        eligible = eligible.sort_values(sort_cols).copy()
    if max_seed_candidates is not None and max_seed_candidates > 0:
        eligible = eligible.head(max_seed_candidates).copy()

    eligible = eligible.reset_index(drop=True)
    seed_candidates_used = len(eligible)
    low = [0.0 for _ in range(len(eligible))]
    high = [1.0 for _ in range(len(eligible))]

    for iteration in range(iterations):
        rows = []
        names = []
        for idx, seed in eligible.iterrows():
            alpha = (low[idx] + high[idx]) / 2.0
            name = f"{name_prefix}_refine_{idx + 1:04d}_it_{iteration + 1:02d}"
            rows.append(_candidate_at_alpha(seed, target_row, io_cols, alpha, name))
            names.append(name)

        probe_candidates = pd.DataFrame(rows)
        probe_metrics = _evaluate_candidates(
            candidates=probe_candidates,
            run_dir=run_dir,
            reference_csv=reference_csv,
            java_entry=java_entry,
            main_class=main_class,
            maven_executable=maven_executable,
            output_stem=f"{name_prefix}_refine_iter_{iteration + 1:02d}",
        )
        metrics_by_name = {
            str(row["name"]): row
            for _, row in probe_metrics.iterrows()
        }

        for idx, name in enumerate(names):
            metric_value = metrics_by_name[name][metric_col]
            if satisfies_numeric_goal(metric_value, threshold, direction):
                high[idx] = (low[idx] + high[idx]) / 2.0
            else:
                low[idx] = (low[idx] + high[idx]) / 2.0

    final_rows = []
    for idx, seed in eligible.iterrows():
        name = f"{name_prefix}_refined_{idx + 1:04d}"
        final_rows.append(_candidate_at_alpha(seed, target_row, io_cols, high[idx], name))

    final_candidates = pd.DataFrame(final_rows)
    final_metrics = _evaluate_candidates(
        candidates=final_candidates,
        run_dir=run_dir,
        reference_csv=reference_csv,
        java_entry=java_entry,
        main_class=main_class,
        maven_executable=maven_executable,
        output_stem=f"{name_prefix}_refined_final",
    )
    final_metrics = add_effort_columns(final_metrics, target_row, io_cols)
    final_metrics["refinement_source_name"] = eligible["name"].tolist()
    final_metrics["refinement_alpha"] = high
    final_metrics["refinement_metric"] = metric_col
    final_metrics["refinement_threshold"] = threshold
    final_metrics["refinement_direction"] = direction
    final_metrics["refinement_iterations"] = iterations
    final_metrics["refinement_seed_candidates_before_front"] = seed_candidates_before_front
    final_metrics["refinement_seed_candidates_after_front"] = seed_candidates_after_front
    final_metrics["refinement_seed_candidates_used"] = seed_candidates_used

    if local_random_samples > 0:
        local_candidates = _local_random_candidates(
            centers=final_metrics,
            target_row=target_row,
            io_cols=io_cols,
            inputs=inputs,
            outputs=outputs,
            search_columns=search_columns or io_cols,
            step_by_column=local_step_by_column or {},
            samples_per_center=local_random_samples,
            step_multiplier=local_random_step_multiplier,
            random_state=local_random_state,
            name_prefix=name_prefix,
            sampling_strategy=local_sampling_strategy,
        )
        if not local_candidates.empty:
            metadata_columns = [
                "name",
                "local_search_center_name",
                "local_search_center_index",
                "local_search_sample_index",
                "local_search_step_multiplier",
                "local_search_sampling_strategy",
                *[f"local_search_step_{col}" for col in (search_columns or io_cols)],
            ]
            local_metadata = local_candidates[[col for col in metadata_columns if col in local_candidates.columns]].copy()
            local_metrics = _evaluate_candidates(
                candidates=local_candidates.drop(
                    columns=[col for col in [
                        "local_search_center_name",
                        "local_search_center_index",
                        "local_search_sample_index",
                        "local_search_step_multiplier",
                        "local_search_sampling_strategy",
                        *[f"local_search_step_{col}" for col in (search_columns or io_cols)],
                    ] if col in local_candidates.columns]
                ),
                run_dir=run_dir,
                reference_csv=reference_csv,
                java_entry=java_entry,
                main_class=main_class,
                maven_executable=maven_executable,
                output_stem=f"{name_prefix}_local_search",
            )
            local_metrics = local_metrics.merge(local_metadata, on="name", how="left")
            local_metrics = add_effort_columns(local_metrics, target_row, io_cols)
            local_metrics.to_csv(run_dir / f"{name_prefix}_local_search_metrics.csv", index=False)
            local_metrics = local_metrics[
                local_metrics[metric_col].apply(
                    lambda value: satisfies_numeric_goal(value, threshold, direction)
                )
            ].copy()
            if not local_metrics.empty:
                local_metrics["refinement_source_name"] = local_metrics["local_search_center_name"]
                local_metrics["refinement_alpha"] = None
                local_metrics["refinement_metric"] = metric_col
                local_metrics["refinement_threshold"] = threshold
                local_metrics["refinement_direction"] = direction
                local_metrics["refinement_iterations"] = iterations
                local_metrics["local_search_samples_per_center"] = local_random_samples
                final_metrics = pd.concat([final_metrics, local_metrics], ignore_index=True, sort=False)

    if prune_front:
        before = len(final_metrics)
        final_metrics = prune_minimal_change_front(final_metrics, inputs, outputs)
        final_metrics["refinement_front_pruned_from"] = before

    return final_metrics.reset_index(drop=True)
