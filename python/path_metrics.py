import argparse
import re
from pathlib import Path

import pandas as pd


STAGE_FIELD_PATTERN = re.compile(r"^stage_(\d+)_([A-Za-z]\w*)$")
INPUT_PATTERN = re.compile(r"i\d+$")
OUTPUT_PATTERN = re.compile(r"o\d+$")
EPS = 1e-12

ROBUST_METRICS = [
    "best_efficiency",
    "worst_efficiency",
    "best_rank",
    "worst_rank",
    "score_width",
    "rank_width",
]

PEER_SET_FIELDS = [
    "peer_refs",
    "peer_set",
    "peers",
    "reference_set",
    "reference_refs",
]

RELATION_SET_FIELDS = [
    "candidate_necessary_over_refs",
    "candidate_possible_over_refs",
    "reference_necessary_over_candidate_refs",
    "reference_possible_over_candidate_refs",
]

PROGRESS_FIELDS = [
    ("best_efficiency", 1.0),
    ("worst_efficiency", 1.0),
    ("best_rank", -1.0),
    ("worst_rank", -1.0),
]

SUMMARY_COLUMNS = [
    "method",
    "path_id",
    "path_length",
    "stage_count",
    "start_name",
    "final_name",
    "unique_state_count",
    "repeated_state_count",
    "real_state_count",
    "fictive_state_count",
    "mixed_state_path",
    "path_kind",
    "rr",
    "tc",
    "msc",
    "cdir",
    "cdir_endpoint",
    "cdir_scope",
    "dr",
    "bp",
    "wbp",
    "sbp",
    "swbp",
    "mcp",
    "md",
    "pyv",
    "pym",
    "pyv_best_efficiency",
    "pym_best_efficiency",
    "pyv_worst_efficiency",
    "pym_worst_efficiency",
    "pyv_best_rank",
    "pym_best_rank",
    "pyv_worst_rank",
    "pym_worst_rank",
    "apw",
    "fw",
    "ww",
    "apw_score",
    "fw_score",
    "ww_score",
    "apw_rank",
    "fw_rank",
    "ww_rank",
    "pc",
    "pc_pair_count",
    "pc_peer_refs",
    "pc_peer_refs_pair_count",
    "pc_peer_set",
    "pc_peer_set_pair_count",
    "pc_peers",
    "pc_peers_pair_count",
    "pc_reference_set",
    "pc_reference_set_pair_count",
    "pc_reference_refs",
    "pc_reference_refs_pair_count",
    "rc",
    "rc_pair_count",
    "opp",
    "final_effort_from_start",
    "max_effort_from_start",
    "total_effort_movement",
    "final_milestone_gap",
    "total_milestone_gap",
    "mean_milestone_gap",
    "max_milestone_gap",
    "total_input_reduction",
    "total_output_increase",
    "total_io_abs_change",
    "total_step_io_abs_change",
    "max_step_io_abs_change",
    "io_directness",
    "attainable_transition_violations",
    "best_efficiency_start",
    "best_efficiency_final",
    "best_efficiency_delta",
    "best_efficiency_improvement",
    "worst_efficiency_start",
    "worst_efficiency_final",
    "worst_efficiency_delta",
    "worst_efficiency_improvement",
    "best_rank_start",
    "best_rank_final",
    "best_rank_delta",
    "best_rank_improvement",
    "worst_rank_start",
    "worst_rank_final",
    "worst_rank_delta",
    "worst_rank_improvement",
    "score_width_start",
    "score_width_final",
    "score_width_delta",
    "score_width_reduction",
    "rank_width_start",
    "rank_width_final",
    "rank_width_delta",
    "rank_width_reduction",
]


def discover_stage_indices(columns) -> list[int]:
    stages = set()
    for column in columns:
        match = STAGE_FIELD_PATTERN.match(str(column))
        if match:
            stages.add(int(match.group(1)))
    return sorted(stages)


def discover_io_columns(columns) -> tuple[list[str], list[str]]:
    fields = set()
    for column in columns:
        match = STAGE_FIELD_PATTERN.match(str(column))
        if match:
            fields.add(match.group(2))

    inputs = sorted(
        [field for field in fields if INPUT_PATTERN.fullmatch(field)],
        key=lambda value: int(value[1:]),
    )
    outputs = sorted(
        [field for field in fields if OUTPUT_PATTERN.fullmatch(field)],
        key=lambda value: int(value[1:]),
    )
    return inputs, outputs


def _stage_column(stage_idx: int, field: str) -> str:
    return f"stage_{stage_idx:02d}_{field}"


def _value(row: pd.Series, stage_idx: int, field: str):
    column = _stage_column(stage_idx, field)
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def _numeric_stage_values(row: pd.Series, stage_indices: list[int], field: str) -> list[float]:
    values = []
    for stage_idx in stage_indices:
        value = _value(row, stage_idx, field)
        if value is not None:
            values.append(float(value))
    return values


def _complete_numeric_stage_values(
    row: pd.Series,
    stage_indices: list[int],
    field: str,
) -> list[float] | None:
    values = []
    for stage_idx in stage_indices:
        value = _value(row, stage_idx, field)
        if value is None:
            return None
        values.append(float(value))
    return values


def _named_stage_values(row: pd.Series, stage_indices: list[int]) -> list[str]:
    names = []
    for stage_idx in stage_indices:
        value = _value(row, stage_idx, "name")
        if value is not None:
            names.append(str(value))
    return names


def _state_types(row: pd.Series, stage_indices: list[int]) -> list[str]:
    types = []
    for stage_idx in stage_indices:
        value = _value(row, stage_idx, "state_type")
        if value is not None:
            types.append(str(value).strip().lower())
    return types


def _parse_ref_set(value) -> set[str] | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return set()
    return {part for part in text.split("|") if part}


def _safe_div(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= EPS:
        return None
    return numerator / denominator


def _variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _normalization_ranges(
    paths: pd.DataFrame,
    stage_indices: list[int],
    factors: list[str],
) -> dict[str, float]:
    ranges = {}
    for factor in factors:
        values = []
        for stage_idx in stage_indices:
            column = _stage_column(stage_idx, factor)
            if column not in paths.columns:
                continue
            numeric = pd.to_numeric(paths[column], errors="coerce").dropna()
            values.extend(numeric.astype(float).tolist())

        if not values:
            ranges[factor] = 1.0
            continue

        span = max(values) - min(values)
        if span <= EPS:
            span = max(max(abs(value) for value in values), 1.0)
        ranges[factor] = float(span)
    return ranges


def _transition_violations(
    row: pd.Series,
    stage_indices: list[int],
    inputs: list[str],
    outputs: list[str],
) -> int | None:
    if not inputs and not outputs:
        return None

    violations = 0
    comparable_pairs = 0
    for prev_stage, current_stage in zip(stage_indices, stage_indices[1:]):
        pair_comparable = False
        pair_violation = False

        for col in inputs:
            previous = _value(row, prev_stage, col)
            current = _value(row, current_stage, col)
            if previous is None or current is None:
                continue
            pair_comparable = True
            if float(current) > float(previous) + 1e-9:
                pair_violation = True

        for col in outputs:
            previous = _value(row, prev_stage, col)
            current = _value(row, current_stage, col)
            if previous is None or current is None:
                continue
            pair_comparable = True
            if float(current) + 1e-9 < float(previous):
                pair_violation = True

        if pair_comparable:
            comparable_pairs += 1
            if pair_violation:
                violations += 1

    if comparable_pairs == 0:
        return None
    return violations


def _stage_delta(
    row: pd.Series,
    previous_stage: int,
    current_stage: int,
    factor: str,
    inputs: list[str],
    ranges: dict[str, float],
) -> float | None:
    previous = _value(row, previous_stage, factor)
    current = _value(row, current_stage, factor)
    if previous is None or current is None:
        return None

    if factor in inputs:
        raw_change = float(previous) - float(current)
    else:
        raw_change = float(current) - float(previous)
    return raw_change / ranges[factor]


def _direct_delta(
    row: pd.Series,
    start_stage: int,
    final_stage: int,
    factor: str,
    inputs: list[str],
    ranges: dict[str, float],
) -> float | None:
    return _stage_delta(row, start_stage, final_stage, factor, inputs, ranges)


def _normalized_modification_metrics(
    row: pd.Series,
    stage_indices: list[int],
    inputs: list[str],
    outputs: list[str],
    ranges: dict[str, float],
    factor_weights: dict[str, float] | None = None,
    stage_weights: list[float] | None = None,
) -> tuple[dict[str, float | None], list[float] | None]:
    factors = [*inputs, *outputs]
    stage_count = max(len(stage_indices) - 1, 0)
    if stage_count < 1 or not factors:
        return {}, None

    modifications = {factor: [] for factor in factors}
    stage_efforts = []
    raw_factor_weights = {
        factor: float(factor_weights.get(factor, 1.0) if factor_weights else 1.0)
        for factor in factors
    }
    factor_weight_sum = sum(raw_factor_weights.values())
    if factor_weight_sum <= EPS or any(value < 0 for value in raw_factor_weights.values()):
        raise ValueError("Factor weights must be non-negative with a positive sum.")
    normalized_factor_weights = {
        factor: value / factor_weight_sum
        for factor, value in raw_factor_weights.items()
    }
    if stage_weights is None:
        normalized_stage_weights = [1.0 / stage_count for _ in range(stage_count)]
    else:
        if len(stage_weights) != stage_count:
            raise ValueError(f"Expected {stage_count} stage weights, got {len(stage_weights)}.")
        if any(value < 0 for value in stage_weights) or sum(stage_weights) <= EPS:
            raise ValueError("Stage weights must be non-negative with a positive sum.")
        total_stage_weight = sum(stage_weights)
        normalized_stage_weights = [value / total_stage_weight for value in stage_weights]

    for previous_stage, current_stage in zip(stage_indices, stage_indices[1:]):
        stage_values = []
        for factor in factors:
            delta = _stage_delta(row, previous_stage, current_stage, factor, inputs, ranges)
            if delta is None:
                return {}, None
            modifications[factor].append(delta)
            stage_values.append(delta)
        stage_efforts.append(
            sum(
                normalized_factor_weights[factor] * value
                for factor, value in zip(factors, stage_values)
            )
        )

    direct_values = [
        _direct_delta(row, stage_indices[0], stage_indices[-1], factor, inputs, ranges)
        for factor in factors
    ]
    if any(value is None for value in direct_values):
        cdir_endpoint = None
    else:
        cdir_endpoint = sum(
            normalized_factor_weights[factor] * float(value)
            for factor, value in zip(factors, direct_values)
        )

    tc = sum(stage_efforts)
    msc = max(stage_efforts)
    metrics = {
        "tc": tc,
        "msc": msc,
        "cdir_endpoint": cdir_endpoint,
    }

    active_factors = [
        factor
        for factor, values in modifications.items()
        if sum(values) > EPS
    ]
    if active_factors:
        balance_terms_by_factor = {}
        concentration_terms = []
        for factor in active_factors:
            values = modifications[factor]
            total_change = sum(values)
            mean_change = total_change / stage_count
            terms = [
                ((value - mean_change) / total_change) ** 2
                for value in values
            ]
            balance_terms_by_factor[factor] = terms
            concentration_terms.extend(value / total_change for value in values)

        factor_balance_sum = sum(
            sum(terms) for terms in balance_terms_by_factor.values()
        )
        active_count = len(active_factors)
        bp = factor_balance_sum / (active_count * stage_count)

        active_weight_sum = sum(normalized_factor_weights[factor] for factor in active_factors)
        active_factor_weights = {
            factor: normalized_factor_weights[factor] / active_weight_sum
            for factor in active_factors
        }
        wbp = (1.0 / stage_count) * sum(
            active_factor_weights[factor] * sum(terms)
            for factor, terms in balance_terms_by_factor.items()
        )
        sbp = (1.0 / active_count) * sum(
            sum(normalized_stage_weights[h] * terms[h] for h in range(stage_count))
            for terms in balance_terms_by_factor.values()
        )
        swbp = sum(
            active_factor_weights[factor]
            * sum(normalized_stage_weights[h] * terms[h] for h in range(stage_count))
            for factor, terms in balance_terms_by_factor.items()
        )

        metrics.update(
            {
                "bp": bp,
                "wbp": wbp,
                "sbp": sbp,
                "swbp": swbp,
                "mcp": max(concentration_terms),
            }
        )

    return metrics, stage_efforts


def _io_change_metrics(
    row: pd.Series,
    stage_indices: list[int],
    inputs: list[str],
    outputs: list[str],
) -> dict[str, float | int | None]:
    if len(stage_indices) < 2 or not (inputs or outputs):
        return {
            "total_input_reduction": None,
            "total_output_increase": None,
            "total_io_abs_change": None,
            "total_step_io_abs_change": None,
            "max_step_io_abs_change": None,
            "io_directness": None,
            "attainable_transition_violations": _transition_violations(row, stage_indices, inputs, outputs),
        }

    start_stage = stage_indices[0]
    final_stage = stage_indices[-1]
    total_input_reduction = 0.0
    total_output_increase = 0.0
    total_io_abs_change = 0.0

    for col in inputs:
        start = _value(row, start_stage, col)
        final = _value(row, final_stage, col)
        if start is None or final is None:
            continue
        delta = float(final) - float(start)
        total_input_reduction += -delta
        total_io_abs_change += abs(delta)

    for col in outputs:
        start = _value(row, start_stage, col)
        final = _value(row, final_stage, col)
        if start is None or final is None:
            continue
        delta = float(final) - float(start)
        total_output_increase += delta
        total_io_abs_change += abs(delta)

    step_changes = []
    for prev_stage, current_stage in zip(stage_indices, stage_indices[1:]):
        step_change = 0.0
        comparable = False
        for col in [*inputs, *outputs]:
            previous = _value(row, prev_stage, col)
            current = _value(row, current_stage, col)
            if previous is None or current is None:
                continue
            comparable = True
            step_change += abs(float(current) - float(previous))
        if comparable:
            step_changes.append(step_change)

    total_step_io_abs_change = sum(step_changes) if step_changes else None
    max_step_io_abs_change = max(step_changes) if step_changes else None
    io_directness = None
    if total_step_io_abs_change and total_step_io_abs_change > 0:
        io_directness = total_io_abs_change / total_step_io_abs_change

    return {
        "total_input_reduction": total_input_reduction,
        "total_output_increase": total_output_increase,
        "total_io_abs_change": total_io_abs_change,
        "total_step_io_abs_change": total_step_io_abs_change,
        "max_step_io_abs_change": max_step_io_abs_change,
        "io_directness": io_directness,
        "attainable_transition_violations": _transition_violations(row, stage_indices, inputs, outputs),
    }


def _add_series_summary(
    summary: dict,
    row: pd.Series,
    stage_indices: list[int],
    field: str,
):
    values = _complete_numeric_stage_values(row, stage_indices, field)
    if values is None or not values:
        return

    start = values[0]
    final = values[-1]
    summary[f"{field}_start"] = start
    summary[f"{field}_final"] = final
    summary[f"{field}_delta"] = final - start

    if field in {"best_efficiency", "worst_efficiency"}:
        summary[f"{field}_improvement"] = final - start
    elif field in {"best_rank", "worst_rank"}:
        summary[f"{field}_improvement"] = start - final
    elif field in {"score_width", "rank_width"}:
        summary[f"{field}_reduction"] = start - final


def _add_width_metrics(
    summary: dict,
    row: pd.Series,
    stage_indices: list[int],
    width_field: str | None = None,
    width_state_weights: list[float] | None = None,
):
    width_specs = [
        ("score_width", "score"),
        ("rank_width", "rank"),
    ]
    for field, suffix in width_specs:
        values = _complete_numeric_stage_values(row, stage_indices, field)
        if values is None:
            continue
        apw = sum(values) / len(values)
        fw = values[-1]
        if width_state_weights is None:
            ww = apw
        else:
            if len(width_state_weights) != len(values):
                raise ValueError(
                    f"Expected {len(values)} width-state weights, got {len(width_state_weights)}."
                )
            if any(weight < 0 for weight in width_state_weights) or sum(width_state_weights) <= EPS:
                raise ValueError("Width-state weights must be non-negative with a positive sum.")
            total_weight = sum(width_state_weights)
            ww = sum(
                value * weight / total_weight
                for value, weight in zip(values, width_state_weights)
            )
        summary[f"apw_{suffix}"] = apw
        summary[f"fw_{suffix}"] = fw
        summary[f"ww_{suffix}"] = ww
        if field == width_field:
            summary["apw"] = apw
            summary["fw"] = fw
            summary["ww"] = ww


def _add_progress_yield_metrics(
    summary: dict,
    row: pd.Series,
    stage_indices: list[int],
    stage_efforts: list[float] | None,
    progress_field: str | None = None,
):
    if stage_efforts is None or len(stage_efforts) != max(len(stage_indices) - 1, 0):
        return

    for field, sign in PROGRESS_FIELDS:
        values = _complete_numeric_stage_values(row, stage_indices, field)
        if values is None:
            continue

        yields = []
        valid = True
        for h, effort in enumerate(stage_efforts, start=1):
            if effort is None or abs(effort) <= EPS:
                valid = False
                break
            gain = sign * (values[h] - values[h - 1])
            yields.append(gain / effort)

        if not valid or not yields:
            continue

        pyv = _variance(yields)
        pym = min(yields)
        summary[f"pyv_{field}"] = pyv
        summary[f"pym_{field}"] = pym
        if field == progress_field:
            summary["pyv"] = pyv
            summary["pym"] = pym


def _add_set_continuity_metrics(
    summary: dict,
    row: pd.Series,
    stage_indices: list[int],
    fields: list[str],
    prefix: str,
):
    generic_set = False
    for field in fields:
        similarities = []
        for previous_stage, current_stage in zip(stage_indices, stage_indices[1:]):
            previous = _parse_ref_set(_value(row, previous_stage, field))
            current = _parse_ref_set(_value(row, current_stage, field))
            if previous is None or current is None:
                continue
            union = previous | current
            similarity = 1.0 if not union else len(previous & current) / len(union)
            similarities.append(similarity)

        if not similarities:
            continue

        continuity = 1.0 - (sum(similarities) / len(similarities))
        summary[f"{prefix}_{field}"] = continuity
        summary[f"{prefix}_{field}_pair_count"] = len(similarities)
        if not generic_set:
            summary[prefix] = continuity
            summary[f"{prefix}_pair_count"] = len(similarities)
            generic_set = True


def _add_operational_profile_metrics(
    summary: dict,
    row: pd.Series,
    stage_indices: list[int],
    inputs: list[str],
    outputs: list[str],
):
    if len(stage_indices) < 2 or not (inputs or outputs):
        return

    def shares(stage_idx: int, columns: list[str]) -> list[float] | None:
        if not columns:
            return []
        values = []
        for column in columns:
            value = _value(row, stage_idx, column)
            if value is None:
                return None
            values.append(float(value))
        total = sum(values)
        if abs(total) <= EPS:
            return None
        return [value / total for value in values]

    start_input_shares = shares(stage_indices[0], inputs)
    start_output_shares = shares(stage_indices[0], outputs)
    if start_input_shares is None or start_output_shares is None:
        return
    start_profile = [*start_input_shares, *start_output_shares]

    total_distance = 0.0
    for stage_idx in stage_indices[1:]:
        current_input_shares = shares(stage_idx, inputs)
        current_output_shares = shares(stage_idx, outputs)
        if current_input_shares is None or current_output_shares is None:
            return
        current_profile = [*current_input_shares, *current_output_shares]
        total_distance += sum(
            abs(current - start)
            for current, start in zip(current_profile, start_profile)
        )

    summary["opp"] = total_distance


def _realness_ratio(state_types: list[str], stage_indices: list[int]) -> float | None:
    stage_count = max(len(stage_indices) - 1, 0)
    if stage_count < 1 or len(state_types) != len(stage_indices):
        return None
    intermediate_and_final = state_types[1:]
    return intermediate_and_final.count("real") / stage_count


def summarize_path_row(
    row: pd.Series,
    stage_indices: list[int],
    inputs: list[str],
    outputs: list[str],
    normalization_ranges: dict[str, float],
    method_name: str | None = None,
    factor_weights: dict[str, float] | None = None,
    stage_weights: list[float] | None = None,
    width_state_weights: list[float] | None = None,
    progress_field: str | None = None,
    width_field: str | None = None,
    profile_inputs: list[str] | None = None,
    profile_outputs: list[str] | None = None,
) -> dict:
    names = _named_stage_values(row, stage_indices)
    state_types = _state_types(row, stage_indices)

    recommended_types = state_types[1:] if len(state_types) == len(stage_indices) else []
    if recommended_types and all(value == "real" for value in recommended_types):
        path_kind = "real"
    elif recommended_types and all(value == "fictive" for value in recommended_types):
        path_kind = "fictive"
    elif recommended_types:
        path_kind = "mixed"
    else:
        path_kind = None

    summary = {
        "method": method_name,
        "path_id": row.get("path_id"),
        "path_length": int(row["path_length"]) if "path_length" in row and pd.notna(row["path_length"]) else max(len(names) - 1, 0),
        "stage_count": len(names),
        "start_name": names[0] if names else None,
        "final_name": names[-1] if names else None,
        "unique_state_count": len(set(names)),
        "repeated_state_count": len(names) - len(set(names)),
        "real_state_count": state_types.count("real") if state_types else None,
        "fictive_state_count": state_types.count("fictive") if state_types else None,
        "mixed_state_path": path_kind == "mixed" if path_kind else None,
        "path_kind": path_kind,
        "rr": _realness_ratio(state_types, stage_indices),
    }

    normalized_metrics, stage_efforts = _normalized_modification_metrics(
        row=row,
        stage_indices=stage_indices,
        inputs=inputs,
        outputs=outputs,
        ranges=normalization_ranges,
        factor_weights=factor_weights,
        stage_weights=stage_weights,
    )
    summary.update(normalized_metrics)

    effort_values = _numeric_stage_values(row, stage_indices, "effort_from_start")
    if effort_values:
        summary["final_effort_from_start"] = effort_values[-1]
        summary["max_effort_from_start"] = max(effort_values)
        summary["total_effort_movement"] = sum(
            abs(current - previous)
            for previous, current in zip(effort_values, effort_values[1:])
        )

    gap_values = _complete_numeric_stage_values(row, stage_indices[1:], "milestone_gap")
    if gap_values is not None and gap_values:
        summary["final_milestone_gap"] = gap_values[-1]
        summary["total_milestone_gap"] = sum(gap_values)
        summary["mean_milestone_gap"] = sum(gap_values) / len(gap_values)
        summary["max_milestone_gap"] = max(gap_values)
        summary["md"] = summary["mean_milestone_gap"]

    summary.update(
        _io_change_metrics(
            row,
            stage_indices,
            inputs if profile_inputs is None else profile_inputs,
            outputs if profile_outputs is None else profile_outputs,
        )
    )
    _add_width_metrics(
        summary,
        row,
        stage_indices,
        width_field=width_field,
        width_state_weights=width_state_weights,
    )
    _add_progress_yield_metrics(
        summary,
        row,
        stage_indices,
        stage_efforts,
        progress_field=progress_field,
    )
    _add_set_continuity_metrics(summary, row, stage_indices, PEER_SET_FIELDS, "pc")
    _add_set_continuity_metrics(summary, row, stage_indices, RELATION_SET_FIELDS, "rc")
    _add_operational_profile_metrics(
        summary,
        row,
        stage_indices,
        inputs if profile_inputs is None else profile_inputs,
        outputs if profile_outputs is None else profile_outputs,
    )

    for metric in ROBUST_METRICS:
        _add_series_summary(summary, row, stage_indices, metric)

    return summary


def summarize_paths(
    paths: pd.DataFrame,
    method_name: str | None = None,
    io_columns: list[str] | None = None,
    normalization_ranges: dict[str, float] | None = None,
    adjustable_columns: list[str] | None = None,
    factor_weights: dict[str, float] | None = None,
    stage_weights: list[float] | None = None,
    width_state_weights: list[float] | None = None,
    progress_field: str | None = None,
    width_field: str | None = None,
    direct_target_cost: float | None = None,
    cdir_scope: str | None = None,
) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    stage_indices = discover_stage_indices(paths.columns)
    discovered_inputs, discovered_outputs = discover_io_columns(paths.columns)
    if io_columns is None:
        profile_inputs = discovered_inputs
        profile_outputs = discovered_outputs
    else:
        requested = set(io_columns)
        profile_inputs = [col for col in discovered_inputs if col in requested]
        profile_outputs = [col for col in discovered_outputs if col in requested]
    adjustable = set(adjustable_columns or [*profile_inputs, *profile_outputs])
    inputs = [col for col in profile_inputs if col in adjustable]
    outputs = [col for col in profile_outputs if col in adjustable]

    if progress_field is None:
        progress_field = "best_rank" if method_name == "best_rank_path" else "best_efficiency"
    if width_field is None:
        width_field = (
            "rank_width"
            if method_name == "best_rank_path"
            else "score_width"
        )

    ranges = normalization_ranges or _normalization_ranges(
        paths,
        stage_indices,
        [*inputs, *outputs],
    )
    for factor in [*inputs, *outputs]:
        if factor not in ranges or float(ranges[factor]) <= EPS:
            raise ValueError(f"Invalid normalization range for {factor}.")
    rows = [
        summarize_path_row(
            row=row,
            stage_indices=stage_indices,
            inputs=inputs,
            outputs=outputs,
            normalization_ranges=ranges,
            method_name=method_name,
            factor_weights=factor_weights,
            stage_weights=stage_weights,
            width_state_weights=width_state_weights,
            progress_field=progress_field,
            width_field=width_field,
            profile_inputs=profile_inputs,
            profile_outputs=profile_outputs,
        )
        for _, row in paths.iterrows()
    ]
    summary = pd.DataFrame(rows)

    endpoint_costs = pd.to_numeric(summary.get("cdir_endpoint"), errors="coerce")
    if direct_target_cost is None:
        finite_costs = endpoint_costs.dropna()
        cdir = float(finite_costs.min()) if not finite_costs.empty else None
        effective_scope = cdir_scope or "complete_path_endpoints"
    else:
        cdir = float(direct_target_cost)
        effective_scope = cdir_scope or "evaluated_final_stage_pool"
    summary["cdir"] = cdir
    summary["cdir_scope"] = effective_scope
    summary["dr"] = summary["tc"].apply(
        lambda tc: _safe_div(float(tc), cdir)
        if cdir is not None and pd.notna(tc)
        else None
    )

    ordered_columns = [col for col in SUMMARY_COLUMNS if col in summary.columns]
    extra_columns = [col for col in summary.columns if col not in ordered_columns]
    return summary[ordered_columns + extra_columns]


def write_path_metrics(
    paths: pd.DataFrame,
    output_csv: str | Path,
    method_name: str | None = None,
    io_columns: list[str] | None = None,
    normalization_ranges: dict[str, float] | None = None,
    adjustable_columns: list[str] | None = None,
    factor_weights: dict[str, float] | None = None,
    stage_weights: list[float] | None = None,
    width_state_weights: list[float] | None = None,
    progress_field: str | None = None,
    width_field: str | None = None,
    direct_target_cost: float | None = None,
    cdir_scope: str | None = None,
) -> pd.DataFrame:
    metrics = summarize_paths(
        paths,
        method_name=method_name,
        io_columns=io_columns,
        normalization_ranges=normalization_ranges,
        adjustable_columns=adjustable_columns,
        factor_weights=factor_weights,
        stage_weights=stage_weights,
        width_state_weights=width_state_weights,
        progress_field=progress_field,
        width_field=width_field,
        direct_target_cost=direct_target_cost,
        cdir_scope=cdir_scope,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--io-columns", default=None)
    parser.add_argument("--adjustable-columns", default=None)
    parser.add_argument("--factor-weights", default=None)
    parser.add_argument("--stage-weights", default=None)
    parser.add_argument("--width-state-weights", default=None)
    parser.add_argument("--progress-field", default=None)
    parser.add_argument("--width-field", default=None)
    parser.add_argument("--direct-target-cost", type=float, default=None)
    parser.add_argument("--cdir-scope", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    paths_csv = Path(args.paths)
    output_csv = Path(args.output) if args.output else paths_csv.with_name("path_metrics.csv")
    io_columns = None
    if args.io_columns:
        io_columns = [part.strip() for part in args.io_columns.split(",") if part.strip()]
    adjustable_columns = None
    if args.adjustable_columns:
        adjustable_columns = [
            part.strip() for part in args.adjustable_columns.split(",") if part.strip()
        ]

    try:
        paths = pd.read_csv(paths_csv)
    except pd.errors.EmptyDataError:
        paths = pd.DataFrame()
    factor_weights = None
    if args.factor_weights:
        allowed = adjustable_columns or io_columns
        if not allowed:
            discovered_inputs, discovered_outputs = discover_io_columns(paths.columns)
            allowed = [*discovered_inputs, *discovered_outputs]
        factor_weights = {}
        for item in args.factor_weights.split(","):
            name, raw = (part.strip() for part in item.split("=", 1))
            if name not in allowed:
                raise ValueError(f"Unknown factor weight: {name}.")
            factor_weights[name] = float(raw)
    stage_count = max(len(discover_stage_indices(paths.columns)) - 1, 0)
    stage_weights = (
        [float(part.strip()) for part in args.stage_weights.split(",") if part.strip()]
        if args.stage_weights
        else None
    )
    width_state_weights = (
        [float(part.strip()) for part in args.width_state_weights.split(",") if part.strip()]
        if args.width_state_weights
        else None
    )
    if stage_weights is not None and len(stage_weights) != stage_count:
        raise ValueError(f"Expected {stage_count} stage weights, got {len(stage_weights)}.")
    write_path_metrics(
        paths=paths,
        output_csv=output_csv,
        method_name=args.method_name,
        io_columns=io_columns,
        adjustable_columns=adjustable_columns,
        factor_weights=factor_weights,
        stage_weights=stage_weights,
        width_state_weights=width_state_weights,
        progress_field=args.progress_field,
        width_field=args.width_field,
        direct_target_cost=args.direct_target_cost,
        cdir_scope=args.cdir_scope,
    )


if __name__ == "__main__":
    main()
