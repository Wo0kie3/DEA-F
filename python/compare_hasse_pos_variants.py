"""Controlled pairwise-vs-joint POS comparison on the exact grid of a saved experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from complete_experiment_metrics import PEER_CONVENTION
from generate_five_method_robot_report import _latest_run
from hasse_pos_goals import VARIANTS, build_pos_routes, enumerate_pos_variant
from java_runner import export_possible_goals_with_java, export_peer_sets_with_java, export_candidate_robust_metrics_with_java
from path_metrics import RELATION_SET_FIELDS, _parse_ref_set
from path_pipeline_common import add_effort_columns, get_io_columns, normalization_ranges_from_frame, is_attainable_transition
from path_selection import write_publication_selection
from metric_reporting import metric_coverage
from pos_candidate_sampling import expand_grid


def evaluate_batches(states, output, callback, batch_size=500, witnesses=False):
    directory = output.parent / (output.stem + "_batches")
    directory.mkdir()
    results, evidence = [], []
    for start in range(0, len(states), batch_size):
        source = directory / f"input_{start:06d}.csv"
        destination = directory / f"result_{start:06d}.csv"
        batch = states.iloc[start:start+batch_size]
        batch.to_csv(source, index=False)
        callback(source, destination)
        result = pd.read_csv(destination, keep_default_na=False)
        if result.name.duplicated().any() or set(result.name) != set(batch.name):
            raise ValueError("Incomplete batch evaluation")
        results.append(result)
        if witnesses:
            evidence.append(pd.read_csv(str(destination)+".witnesses.csv", keep_default_na=False))
    pd.concat(results, ignore_index=True).to_csv(output, index=False)
    if witnesses:
        pd.concat(evidence, ignore_index=True).to_csv(str(output)+".witnesses.csv", index=False)


def weight_map(text):
    return {key: float(value) for key, value in (part.split("=") for part in text.split("|"))}


def audit_weights(reference, states, goals, top):
    rows = []
    indexed = states.set_index("name", drop=False)
    for _, result in goals.iterrows():
        z = indexed.loc[result["name"]]
        v, u = weight_map(result.joint_input_weights), weight_map(result.joint_output_weights)
        denominator = sum(v[c]*z[c] for c in v)
        numerator = sum(u[c]*z[c] for c in u)
        if not np.isclose(denominator, 1., atol=1e-8) or not np.isclose(numerator, result.joint_score, atol=1e-8):
            raise ValueError("Joint witness normalization/score mismatch")
        if min([*v.values(), *u.values()]) < -1e-9:
            raise ValueError("Negative witness weight")
        for _, target in reference.iterrows():
            score = sum(u[c]*target[c] for c in u) / sum(v[c]*target[c] for c in v)
            margin = numerator / denominator - score
            required = target["name"] in top
            if result.joint_top and required and margin < -1e-8:
                raise ValueError("Returned common weights do not satisfy all top comparisons")
            rows.append(dict(name=z["name"], target=target["name"], top_target=required,
                             joint_top=result.joint_top, candidate_score=numerator/denominator,
                             target_score=score, margin=margin))
    return pd.DataFrame(rows)


def run_comparison(source, root, java_entry, candidate_limit=None, grid_levels=None,
                   jitter_samples=2, jitter_fraction=.2, sampling_seed=20261019):
    if root.exists():
        raise ValueError("Use a new comparison directory; historical results are never overwritten")
    source_run = _latest_run(source / "hasse_path")
    reference = pd.read_csv(source / "dea_input_snapshot/decision_data.csv")
    params = pd.read_csv(source / "experiment_params.csv", keep_default_na=False)
    settings = dict(zip(params.parameter, params.value))
    if settings.get("mode") != "fictive":
        raise ValueError("This controlled comparison currently supports fictive runs only")
    target = settings["target"]
    columns = settings["dimensions"].split(",")
    inputs, outputs = get_io_columns(reference)
    io = inputs+outputs
    if len(columns) not in (2, 3) or not set(columns).issubset(inputs):
        raise ValueError("The comparison report supports two or three modified inputs")
    if settings.get("stage_weights", "equal") != "equal" or settings.get("width_state_weights", "equal") != "equal":
        raise ValueError("This comparison currently requires equal stage/state weights")
    if candidate_limit is not None and candidate_limit < 1:
        raise ValueError("candidate-limit must be positive")
    original = pd.read_csv(source_run / "fictive_candidates.csv")
    old_metrics = pd.read_csv(source_run / "fictive_candidate_metrics.csv")
    pd.testing.assert_frame_equal(original[["name", *io]], old_metrics[["name", *io]], check_dtype=False)
    if original.name.duplicated().any():
        raise ValueError("Duplicate candidate IDs")
    baseline = pd.read_csv(source / "dea_input_snapshot/reference_dmu_results.csv").set_index("name").loc[target].copy()
    baseline["name"], baseline["state_type"], baseline["effort_from_start"] = target, "real", 0.
    expected_ranges = normalization_ranges_from_frame(reference, io, baseline)
    ranges = json.loads(settings["normalization_ranges"])
    if any(not np.isclose(ranges[c], expected_ranges[c]) for c in io):
        raise ValueError("Source normalization ranges differ from input")
    if not original.apply(lambda row: is_attainable_transition(baseline, row, inputs, outputs), axis=1).all():
        raise ValueError("Source grid contains unattainable points")
    for column in set(io)-set(columns):
        if not np.allclose(original[column], baseline[column], atol=1e-12, rtol=0):
            raise ValueError("Source grid changes a fixed factor")
    if grid_levels is not None:
        original = expand_grid(original, columns, grid_levels, jitter_samples, jitter_fraction, sampling_seed)
    if candidate_limit is not None:
        original = original.head(candidate_limit).copy()
    root.mkdir(parents=True)
    shutil.copytree(source / "dea_input_snapshot", root / "dea_input_snapshot")
    original.to_csv(root / "fictive_candidates.csv", index=False)
    original.to_csv(root / "candidate_sampling.csv", index=False)
    necessary = pd.read_csv(root / "dea_input_snapshot/necessary_matrix.csv", index_col=0)
    routes, top, members, cover = build_pos_routes(necessary, target)
    pd.DataFrame(dict(name=top)).to_csv(root / "top_targets.csv", index=False)
    (root / "graph_routes.json").write_text(json.dumps(dict(routes=routes, top=top, components=members,
        cover={k: sorted(v) for k, v in cover.items()}), indent=2), encoding="utf-8")
    files = [source / "experiment_params.csv", source_run / "fictive_candidates.csv", source_run / "fictive_candidate_metrics.csv"]
    files += sorted((source / "dea_input_snapshot").glob("*.csv"))
    pd.DataFrame([dict(file=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]).to_csv(root / "source_manifest.csv", index=False)
    settings.update(source_experiment=str(source), comparison_root=str(root),
        goal_variants="pairwise,joint", intermediate_goal="POS over one target DMU on a NEC graph route",
        final_goal="all maximal NEC classes, globally; common for all branches", top_targets="|".join(top),
        candidate_limit="none" if candidate_limit is None else candidate_limit,
        sampling_mode="source_grid" if grid_levels is None else "dense_grid_with_jitter",
        grid_levels="source" if grid_levels is None else grid_levels,
        jitter_samples=jitter_samples if grid_levels is not None else 0,
        jitter_fraction=jitter_fraction, sampling_seed=sampling_seed,
        evaluated_candidate_count=len(original), evaluation_batch_size=500,
        peer_convention=PEER_CONVENTION, joint_tolerance=1e-9, weight_domain="nonnegative unrestricted CCR weights; positive IO",
        snapshot_reused=True, candidate_robust_metrics_recomputed=True, started_comparison_at=datetime.now(timezone.utc).isoformat())
    pd.DataFrame(settings.items(), columns=["parameter", "value"]).to_csv(root / "experiment_params.csv", index=False)
    timings = []

    def timed(label, call):
        before = time.perf_counter()
        print("Starting " + label, flush=True)
        result = call()
        timings.append(dict(step=label, elapsed_seconds=time.perf_counter()-before))
        pd.DataFrame(timings).to_csv(root / "execution_status.csv", index=False)
        return result

    ref_file = root / "dea_input_snapshot/decision_data.csv"
    timed("robust_candidate_metrics", lambda: evaluate_batches(original[["name", "state_type", *io]],
        root / "fictive_candidate_metrics.csv", lambda src, dst: export_candidate_robust_metrics_with_java(
            ref_file, src, dst, java_entry, "org.example.CsvCandidateRobustMetricsExporter")))
    candidates = pd.read_csv(root / "fictive_candidate_metrics.csv", keep_default_na=False)
    fresh = candidates.set_index("name")
    cached = old_metrics.set_index("name")
    shared = fresh.index.intersection(cached.index)
    for col in ("best_efficiency", "worst_efficiency", "best_rank", "worst_rank", "score_width", "rank_width"):
        if not np.allclose(fresh.loc[shared, col], cached.loc[shared, col], atol=1e-8, rtol=0):
            raise ValueError("Fresh robust metric differs from source: " + col)
    states = pd.concat([reference.loc[reference.name.eq(target), ["name", *io]], candidates[["name", *io]]], ignore_index=True)
    states.to_csv(root / "evaluation_states.csv", index=False)
    timed("pairwise_and_joint_goals", lambda: evaluate_batches(states, root / "possible_goals.csv",
        lambda src, dst: export_possible_goals_with_java(ref_file, src, root / "top_targets.csv", dst, java_entry), witnesses=True))
    goals = pd.read_csv(root / "possible_goals.csv", keep_default_na=False)
    if goals.name.duplicated().any() or set(goals.name) != set(states.name):
        raise ValueError("Incomplete goal evaluation")
    for _, row in candidates.iterrows():
        actual = goals.loc[goals.name.eq(row["name"]), "possible_over_refs"].iloc[0]
        if _parse_ref_set(actual) != _parse_ref_set(row.candidate_possible_over_refs):
            raise ValueError("Fresh pairwise POS differs from robust exporter")
    witness_audit = audit_weights(reference, states, goals, top)
    witness_audit.to_csv(root / "joint_weight_audit.csv", index=False)
    timed("peer_sets", lambda: evaluate_batches(states, root / "peer_sets.csv",
        lambda src, dst: export_peer_sets_with_java(ref_file, src, dst, java_entry)))
    peers = pd.read_csv(root / "peer_sets.csv")
    weights = {key: float(value) for key, value in (item.split("=") for item in settings["factor_weights"].split(","))}
    candidates = candidates.merge(goals, on="name", validate="one_to_one").merge(peers[["name", "peer_refs", "peer_theta"]], on="name", validate="one_to_one")
    candidates = add_effort_columns(candidates, baseline, columns, normalization_ranges=ranges, factor_weights=weights)
    provenance = [c for c in ("sampling_source", "sampling_center") if c in original]
    if provenance:
        candidates = candidates.merge(original[["name", *provenance]], on="name", validate="one_to_one")
        candidates["source"] = candidates.sampling_source + ": " + candidates.sampling_center.fillna("")
    for field in RELATION_SET_FIELDS:
        candidates[field] = candidates[field].replace("", "[]")
    for field, kind, transpose in [(RELATION_SET_FIELDS[0], "necessary", False), (RELATION_SET_FIELDS[1], "possible", False),
                                   (RELATION_SET_FIELDS[2], "necessary", True), (RELATION_SET_FIELDS[3], "possible", True)]:
        matrix = pd.read_csv(root / f"dea_input_snapshot/{kind}_matrix.csv", index_col=0)
        series = matrix[target] if transpose else matrix.loc[target]
        baseline[field] = "|".join(series.index[series.eq(1)]) or "[]"
    for field in ("peer_refs", "peer_theta"):
        baseline[field] = peers.set_index("name").loc[target, field]
    candidates.to_csv(root / "evaluated_candidates.csv", index=False)
    selected, summaries, all_metrics, coverage, checks = {}, [], [], [], []
    for variant in VARIANTS:
        run = root / variant
        run.mkdir()
        frame, metrics, pools, transitions, cap = timed("paths_"+variant, lambda: enumerate_pos_variant(
            baseline, candidates, routes, top, variant, inputs, outputs, columns, ranges, weights,
            points_per_transition=int(settings["points_per_transition"]), max_paths=int(settings["max_paths"]),
            max_stage_effort=None if settings["max_stage_effort"] == "none" else float(settings["max_stage_effort"]),
            max_factor_changes=None if settings["max_factor_changes"] == "none" else {k: float(v) for k, v in (part.split("=") for part in settings["max_factor_changes"].split(","))}))
        for filename, data in [("paths.csv", frame), ("path_metrics.csv", metrics), ("stage_candidates.csv", pools), ("transition_candidates.csv", transitions)]:
            data.to_csv(run / filename, index=False)
        candidates.to_csv(run / "fictive_candidate_metrics.csv", index=False)
        selection, _ = write_publication_selection(metrics, run, method_name="hasse_path", criteria=["msc", "tc", "swbp", "fw"])
        eligible = candidates.loc[candidates[f"{variant}_top"]]
        summary = dict(variant=variant, tested_candidates=len(candidates), final_goal_candidates=len(eligible),
            inefficient_final_candidates=int((eligible.best_efficiency < 1-1e-9).sum()),
            complete_path_records=len(frame), max_paths_reached=cap)
        if not metrics.empty:
            chosen = selection.loc[selection.selection_type.eq("lexicographic"), "path_id"].iloc[0]
            metric = metrics.set_index("path_id").loc[chosen].copy()
            metric["path_id"] = chosen
            path = frame.set_index("path_id").loc[chosen].copy()
            path["path_id"] = chosen
            selected[variant] = (metric, path, "#d97706" if variant == "pairwise" else "#2563b6")
            summary.update(selected_path=chosen, selected_tc=metric.tc, selected_msc=metric.msc,
                min_tc=float(metrics.tc.min()), final_best_efficiency=metric.best_efficiency_final,
                final_point=metric.final_name)
            statuses = metric_coverage(metrics, frame)
            statuses.insert(0, "variant", variant)
            statuses.to_csv(run / "metric_status.csv", index=False)
            coverage.append(statuses)
            metric_lookup = metrics.set_index("path_id")
            goal_lookup = goals.set_index("name")
            for _, p in frame.iterrows():
                previous = baseline
                total = 0.
                for h in range(1, int(p.path_length)+1):
                    prefix = f"stage_{h:02d}_"
                    current = pd.Series({k[len(prefix):]: v for k, v in p.items() if k.startswith(prefix)})
                    if not is_attainable_transition(previous, current, inputs, outputs) or current.transition_reference_name != previous["name"]:
                        raise ValueError("Invalid stepwise transition")
                    goal = current.milestone_target_name
                    result = goal_lookup.loc[current["name"]]
                    valid = result[f"{variant}_top"] if goal == "ALL_TOP" else goal in (_parse_ref_set(result.possible_over_refs) or set())
                    if not valid:
                        raise ValueError("Path violates POS milestone")
                    total += current.effort_from_previous
                    previous = current
                m = metric_lookup.loc[p.path_id]
                if not np.isclose(total, m.tc) or m.pc_pair_count != p.path_length or m.rc_pair_count != p.path_length:
                    raise ValueError("Metric/continuity audit failed")
            checks.append(dict(check=variant+"_all_paths_stepwise_goals_tc_pc_rc", passed=True))
        summaries.append(summary)
        all_metrics.append(metrics.assign(variant=variant))
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "variant_summary.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(root / "all_path_metrics.csv", index=False)
    if coverage:
        status = pd.concat(coverage, ignore_index=True)
        status.groupby(["variant", "metric", "status"]).size().reset_index(name="path_count").to_csv(root / "metric_coverage.csv", index=False)
    else:
        pd.DataFrame(columns=["variant", "metric", "status", "path_count"]).to_csv(root / "metric_coverage.csv", index=False)
    checks += [dict(check="joint_implies_pairwise", passed=bool((~goals.joint_top | goals.pairwise_top).all())),
               dict(check="fresh_scores_match_source", passed=True), dict(check="common_weight_witnesses", passed=True),
               dict(check="fresh_pairwise_matches_robust_exporter", passed=True)]
    pd.DataFrame(checks).to_csv(root / "validation_checks.csv", index=False)
    settings.update(completed_comparison_at=datetime.now(timezone.utc).isoformat())
    pd.DataFrame(settings.items(), columns=["parameter", "value"]).to_csv(root / "experiment_params.csv", index=False)
    from generate_hasse_pos_report import build_report
    timed("report", lambda: build_report(root, reference, candidates, selected, summary, settings, top))
    (root / "reproduction_command.json").write_text(json.dumps(sys.argv, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(root / "hasse_pos_comparison.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--java-entry", type=Path, default=Path(__file__).resolve().parents[1] / "java")
    parser.add_argument("--candidate-limit", type=int, help="Small integration smoke only; omitted uses the full saved grid")
    parser.add_argument("--grid-levels", type=int, help="Expand source bounds to this many levels per modified input")
    parser.add_argument("--jitter-samples", type=int, default=2)
    parser.add_argument("--jitter-fraction", type=float, default=.2, help="Local radius as a fraction of the new grid step")
    parser.add_argument("--sampling-seed", type=int, default=20261019)
    args = parser.parse_args()
    run_comparison(args.source.resolve(), args.output.resolve(), args.java_entry.resolve(), args.candidate_limit,
                   args.grid_levels, args.jitter_samples, args.jitter_fraction, args.sampling_seed)
