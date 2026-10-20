"""NEC graph routes with POS milestones and a common, branch-independent final goal."""

from itertools import product

import pandas as pd

from path_metrics import _parse_ref_set, summarize_paths
from path_pipeline_common import (
    build_worse_to_better_graph, strongly_connected_components, build_component_graph,
    transitive_reduce_dag, enumerate_paths_from_start, enumerate_state_paths, state_paths_to_frame,
)


VARIANTS = ("pairwise", "joint")


def build_pos_routes(necessary, target):
    graph = build_worse_to_better_graph(necessary)
    ids, members, dag = build_component_graph(strongly_connected_components(graph), graph, list(necessary.index))
    cover = transitive_reduce_dag(dag)
    # All maximal classes, including maxima not reachable from this target's component.
    top = sorted(dmu for component, successors in dag.items() if not successors for dmu in members[component])
    paths = enumerate_paths_from_start(ids[target], cover)
    routes = {}
    for path in paths:
        intermediate = path[1:-1]
        for targets in product(*(members[c] for c in intermediate)):
            key = tuple(zip(intermediate, targets))
            if key not in routes:
                routes[key] = dict(route_id=f"route_{len(routes)+1:04d}",
                    intermediate=[dict(component=c, target=t) for c, t in key], graph_paths=[])
            routes[key]["graph_paths"].append(path)
    return list(routes.values()), top, members, cover


def goal_pools(candidates, route, top, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown final goal variant: {variant}")
    result = []
    for stage, goal in enumerate([*route["intermediate"], None], start=1):
        if goal is None:
            mask = candidates[f"{variant}_top"].eq(True)
            target, component, required = "ALL_TOP", "ALL_TOP", top
            kind = f"{variant}_top"
        else:
            target, component, required = goal["target"], goal["component"], [goal["target"]]
            mask = candidates.possible_over_refs.apply(lambda refs: target in (_parse_ref_set(refs) or set()))
            kind = "pairwise_single_target"
        pool = candidates.loc[mask].copy()
        pool["stage"] = stage
        pool["milestone_route_id"] = route["route_id"]
        pool["milestone_target_name"] = target
        pool["milestone_component_id"] = component
        pool["milestone_members"] = "|".join(required)
        pool["milestone_goal_kind"] = kind
        pool["milestone_gap"] = 0.
        result.append(pool)
    return result


def enumerate_pos_variant(start, candidates, routes, top, variant, inputs, outputs, columns,
                          ranges, factor_weights, points_per_transition=15, max_paths=20000,
                          max_stage_effort=None, max_factor_changes=None):
    paths, pools, transitions = [], [], []
    limit_reached = False
    for route in routes:
        stages = goal_pools(candidates, route, top, variant)
        pools.extend(stages)
        remaining = None if max_paths is None else max_paths-len(paths)
        if remaining is not None and remaining <= 0:
            limit_reached = True
            continue
        found = enumerate_state_paths(start, [p.drop(columns="stage") for p in stages], inputs, outputs,
            max_paths=remaining, points_per_transition=points_per_transition, normalization_ranges=ranges,
            minimal_inputs=[c for c in inputs if c in columns], minimal_outputs=[c for c in outputs if c in columns],
            effort_inputs=[c for c in inputs if c in columns], effort_outputs=[c for c in outputs if c in columns],
            factor_weights=factor_weights, max_stage_effort=max_stage_effort,
            max_factor_changes=max_factor_changes, transition_log=transitions)
        paths.extend(found)
        if max_paths is not None and len(paths) >= max_paths:
            limit_reached = True
    frame = state_paths_to_frame(paths, inputs+outputs)
    if frame.empty:
        frame = pd.DataFrame(columns=["path_id", "path_length"])
    parts = []
    # Different NEC routes can have different lengths; never treat trailing NaNs as stages.
    for length, group in frame.groupby("path_length"):
        cols = [c for c in group if not c.startswith("stage_") or int(c.split("_")[1]) <= int(length)]
        parts.append(summarize_paths(group[cols], method_name="hasse_path", io_columns=inputs+outputs,
            normalization_ranges=ranges, adjustable_columns=columns, factor_weights=factor_weights,
            progress_field="best_efficiency", width_field="score_width"))
    metrics = pd.concat(parts, ignore_index=True) if parts else summarize_paths(frame, method_name="hasse_path")
    if not metrics.empty:
        cdir = float(metrics.cdir_endpoint.min())
        metrics["cdir"] = cdir
        metrics["cdir_scope"] = "complete_path_endpoints"
        metrics["dr"] = metrics.tc / cdir if abs(cdir) > 1e-12 else float("nan")
    metrics["goal_variant"] = variant
    return frame, metrics, pd.concat(pools, ignore_index=True), pd.DataFrame(transitions), limit_reached
