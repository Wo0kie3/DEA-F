"""Enrich saved paths with true CCR peer sets and recompute their complete metrics."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from collect_path_metrics import collect_path_metrics
from generate_five_method_robot_report import METHODS, _latest_run
from java_runner import export_peer_sets_with_java
from path_metrics import RELATION_SET_FIELDS, discover_stage_indices, summarize_paths
from path_pipeline_common import get_io_columns


PEER_CONVENTION = (
    "Input-oriented CCR, fixed original reference DMUs, radial phase-I optimum; "
    "union of peers with max lambda_j > 1e-7 over all radial optimal solutions; "
    "no phase-II slack optimization; peer_theta is not the robust best_efficiency."
)


def point_key(row, columns, prefix=""):
    return tuple(float(row[prefix + c]) for c in columns)


def enrich_paths(paths, stage_pool, reference, matrices, peers_by_point, columns):
    paths = paths.copy()
    reference = reference.set_index("name")
    candidates = stage_pool.drop_duplicates("name").set_index("name")
    specs = [("necessary", False), ("possible", False), ("necessary", True), ("possible", True)]
    for stage in discover_stage_indices(paths.columns):
        prefix = f"stage_{stage:02d}_"
        for index, row in paths.iterrows():
            if pd.isna(row.get(prefix + "name")):
                continue
            point = point_key(row, columns, prefix)
            peer = peers_by_point[point]
            paths.loc[index, prefix + "peer_refs"] = peer["peer_refs"]
            paths.loc[index, prefix + "peer_theta"] = peer["peer_theta"]
            name = str(row[prefix + "name"])
            for field, (kind, transpose) in zip(RELATION_SET_FIELDS, specs):
                if name in reference.index and point == point_key(reference.loc[name], columns):
                    matrix = matrices[kind]
                    series = matrix[name] if transpose else matrix.loc[name]
                    refs = "|".join(str(v) for v in series.index[series.eq(1)])
                else:
                    # Empty strings here are known empty sets, read with keep_default_na=False.
                    if name not in candidates.index or field not in candidates:
                        raise ValueError(f"No source relation set for {name}: {field}")
                    if not np.allclose(point_key(candidates.loc[name], columns), point, rtol=1e-12, atol=1e-12):
                        raise ValueError(f"Candidate coordinates differ from the saved path: {name}")
                    refs = str(candidates.loc[name, field])
                paths.loc[index, prefix + field] = refs or "[]"
    return paths


def save_with_backup(frame, path):
    backup = path.with_suffix(".before_metric_completion.csv")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    temp = path.with_suffix(".pending.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def complete_metrics(root, java_entry):
    started = time.perf_counter()
    snapshot = root / "dea_input_snapshot"
    reference = pd.read_csv(snapshot / "decision_data.csv")
    inputs, outputs = get_io_columns(reference)
    columns = inputs + outputs
    params_frame = pd.read_csv(root / "experiment_params.csv", keep_default_na=False)
    params = dict(zip(params_frame.parameter, params_frame.value))
    ranges = json.loads(params["normalization_ranges"])
    weights = {k: float(v) for k, v in (item.split("=") for item in params["factor_weights"].split(","))}
    if params.get("stage_weights", "equal") != "equal" or params.get("width_state_weights", "equal") != "equal":
        raise ValueError("This completion workflow requires explicit support for nonuniform stage/state weights")
    runs, points = [], {}
    for method, _, _ in METHODS:
        run = _latest_run(root / method)
        paths = pd.read_csv(run / "paths.csv")
        if paths.empty:
            continue
        runs.append((method, run, paths))
        for stage in discover_stage_indices(paths.columns):
            prefix = f"stage_{stage:02d}_"
            for _, row in paths.iterrows():
                if pd.notna(row.get(prefix + "name")):
                    key = point_key(row, columns, prefix)
                    points.setdefault(key, f"state_{len(points):06d}")
    if not points:
        if params.get('final_goal') == 'joint':
            (root/'metric_completion.json').write_text(json.dumps(dict(status='no_complete_paths',
                peer_convention=PEER_CONVENTION, unique_evaluated_states=0)),encoding='utf-8')
            return
        raise ValueError("No complete paths to enrich")
    states = pd.DataFrame([dict(name=name, **dict(zip(columns, point))) for point, name in points.items()])
    states_file = root / "peer_evaluation_states.csv"
    states.to_csv(states_file, index=False)
    peer_file = root / "peer_evaluation_results.csv"
    export_peer_sets_with_java((snapshot / "decision_data.csv").resolve(), states_file.resolve(),
                               peer_file.resolve(), java_entry.resolve())
    peers = pd.read_csv(peer_file).set_index("name")
    if peers.index.duplicated().any() or set(peers.index) != set(states.name):
        raise ValueError("Peer exporter did not return exactly all requested states")
    peers_by_point = {point: peers.loc[name] for point, name in points.items()}
    matrices = {kind: pd.read_csv(snapshot / f"{kind}_matrix.csv", index_col=0) for kind in ("necessary", "possible")}
    prepared = []
    for method, run, paths in runs:
        old = pd.read_csv(run / "path_metrics.csv").set_index("path_id")
        pool = pd.read_csv(run / "stage_candidates.csv", keep_default_na=False)
        enriched = enrich_paths(paths, pool, reference, matrices, peers_by_point, columns)
        rank_kind = method == "best_rank_path" or (method == "robustness_width_path" and params.get("width_kind") == "rank")
        costs = old.cdir.dropna().unique()
        scopes = old.cdir_scope.dropna().unique()
        if len(costs) != 1 or len(scopes) != 1:
            raise ValueError(f"Ambiguous original Cdir: {method}")
        parts = []
        for length, group in enriched.groupby('path_length'):
            keep = [c for c in group if not c.startswith('stage_') or int(c.split('_')[1]) <= int(length)]
            parts.append(summarize_paths(group[keep], method_name=method, io_columns=columns,
                normalization_ranges=ranges, adjustable_columns=params["dimensions"].split(","),
                factor_weights=weights, progress_field="best_rank" if rank_kind else "best_efficiency",
                width_field="rank_width" if rank_kind else "score_width",
                direct_target_cost=float(costs[0]), cdir_scope=scopes[0]))
        metrics = pd.concat(parts, ignore_index=True)
        if 'goal_variant' in old:
            metrics['goal_variant'] = metrics.path_id.map(old.goal_variant)
        new = metrics.set_index("path_id").reindex(old.index)
        # Recalculation may fill PC/RC, but must not change costs, selections or other existing metrics.
        unchanged = [c for c in old if c not in ("pc", "rc") and not c.startswith(("pc_", "rc_"))]
        pd.testing.assert_frame_equal(old[unchanged], new.reindex(columns=unchanged),
                                      check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-10)
        for metric in ("pc", "rc"):
            if metrics[metric].isna().any() or not np.allclose(metrics[f"{metric}_pair_count"], metrics.path_length):
                raise ValueError(f"Incomplete {metric} for {method}")
        prepared.append((run, enriched, metrics))
    for run, paths, metrics in prepared:
        save_with_backup(paths, run / "paths.csv")
        save_with_backup(metrics, run / "path_metrics.csv")
        # Refresh metric payloads without changing any selected IDs or selection criteria.
        for filename in ("path_selection.csv", "path_pareto.csv", "path_pareto_representatives.csv"):
            path = run / filename
            if path.exists():
                existing = pd.read_csv(path)
                lookup = metrics.set_index("path_id")
                for col in metrics:
                    if col != "path_id":
                        existing[col] = existing.path_id.map(lookup[col])
                save_with_backup(existing, path)
    all_metrics, summary = collect_path_metrics(root)
    all_metrics.to_csv(root / "all_path_metrics.csv", index=False)
    summary.to_csv(root / "method_summary.csv", index=False)
    source_paths = [Path(__file__), Path(__file__).with_name("path_metrics.py"), Path(__file__).with_name("java_runner.py"),
                    java_entry / "src/main/java/org/example/CsvPeerSetsExporter.java"]
    metadata = dict(peer_convention=PEER_CONVENTION, peer_lambda_tolerance=1e-7,
        peer_lp_verification_tolerance=1e-7, unique_evaluated_states=len(states),
        completed_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        path_records=sum(len(m) for _, _, m in prepared), elapsed_seconds=time.perf_counter()-started,
        reference_sha256=hashlib.sha256((snapshot / "decision_data.csv").read_bytes()).hexdigest(),
        command=sys.argv, rc_note="Full path including start; reflexive NEC/POS included consistently; empty sets serialized as [].")
    (root / "metric_completion.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    additions = {"peer_convention": PEER_CONVENTION, "peer_lambda_tolerance": "1e-7",
                 "continuity_scope": "all transitions including start; PC peers and RC relations separate"}
    updated = {**params, **additions}
    pd.DataFrame(updated.items(), columns=["parameter", "value"]).to_csv(root / "experiment_params.csv", index=False)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--java-entry", type=Path, default=Path(__file__).resolve().parents[1] / "java")
    args = parser.parse_args()
    complete_metrics(args.root.resolve(), args.java_entry)
