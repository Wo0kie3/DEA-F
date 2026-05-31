import argparse
import os
import re
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from java_runner import (
    evaluate_pairwise_possible_candidates_with_java,
    generate_preference_relations_with_java,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--columns", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--java-entry", required=True)
    parser.add_argument(
        "--preference-main-class",
        default="org.example.CsvPreferenceRelationsPreview",
    )
    parser.add_argument(
        "--possible-evaluator-main-class",
        default="org.example.CsvPairwisePossibleCandidateEvaluator",
    )
    parser.add_argument("--maven-executable", default="mvn")

    parser.add_argument("--pct-below", type=float, default=0.0)
    parser.add_argument("--pct-above", type=float, default=30.0)
    parser.add_argument("--step-pct", type=float, default=5.0)
    parser.add_argument("--step-abs", type=float, default=None)
    parser.add_argument("--max-candidates-per-step", type=int, default=25000)
    parser.add_argument("--max-paths", type=int, default=None)

    return parser.parse_args()


def create_run_output_dir(base_output_dir: str, method_name: str) -> Path:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / method_name / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def path_for_java(path_str: str, java_entry: str) -> str:
    abs_target = Path(path_str).resolve()
    abs_java = Path(java_entry).resolve()
    return os.path.relpath(abs_target, start=abs_java)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))


def to_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def get_io_columns(df: pd.DataFrame):
    inputs = sorted(
        [c for c in df.columns if c.startswith("i") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    outputs = sorted(
        [c for c in df.columns if c.startswith("o") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    return inputs, outputs


def build_relation_matrix(
    df_relations: pd.DataFrame,
    dmu_order: list[str],
    value_col: str,
) -> pd.DataFrame:
    matrix = (
        df_relations.pivot(
            index="source_dmu",
            columns="target_dmu",
            values=value_col,
        )
        .reindex(index=dmu_order, columns=dmu_order)
        .fillna(False)
    )

    return matrix.applymap(to_bool)


def build_worse_to_better_graph(
    necessary_matrix: pd.DataFrame,
) -> dict[str, set[str]]:
    dmus = necessary_matrix.index.tolist()
    graph = {dmu: set() for dmu in dmus}

    for better in dmus:
        for worse in dmus:
            if better == worse:
                continue
            if bool(necessary_matrix.at[better, worse]):
                graph[worse].add(better)

    return graph


def strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[list[str]]:
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlinks = {}
    components = []

    def visit(node: str):
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1

        stack.append(node)
        on_stack.add(node)

        for nxt in graph[node]:
            if nxt not in indices:
                visit(nxt)
                lowlinks[node] = min(lowlinks[node], lowlinks[nxt])
            elif nxt in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[nxt])

        if lowlinks[node] == indices[node]:
            component = []
            while True:
                current = stack.pop()
                on_stack.remove(current)
                component.append(current)
                if current == node:
                    break
            components.append(sorted(component))

    for node in graph:
        if node not in indices:
            visit(node)

    return components


def build_component_graph(
    components: list[list[str]],
    graph: dict[str, set[str]],
    dmu_order: list[str],
):
    component_id_by_dmu = {}
    for idx, members in enumerate(components, start=1):
        component_id = f"C{idx:03d}"
        for member in members:
            component_id_by_dmu[member] = component_id

    order_index = {name: idx for idx, name in enumerate(dmu_order)}
    component_members = {
        component_id_by_dmu[members[0]]: sorted(members, key=lambda x: order_index[x])
        for members in components
    }

    component_graph = {component_id: set() for component_id in component_members}
    for src, successors in graph.items():
        src_component = component_id_by_dmu[src]
        for dst in successors:
            dst_component = component_id_by_dmu[dst]
            if src_component != dst_component:
                component_graph[src_component].add(dst_component)

    return component_id_by_dmu, component_members, component_graph


def topological_sort(graph: dict[str, set[str]]) -> list[str]:
    indegree = {node: 0 for node in graph}
    for successors in graph.values():
        for nxt in successors:
            indegree[nxt] += 1

    ready = sorted([node for node, deg in indegree.items() if deg == 0])
    topo = []

    while ready:
        node = ready.pop(0)
        topo.append(node)
        for nxt in sorted(graph[node]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()

    if len(topo) != len(graph):
        raise ValueError("Component graph is not acyclic after SCC condensation.")

    return topo


def has_alternative_path(
    start: str,
    target: str,
    graph: dict[str, set[str]],
    skipped_edge: tuple[str, str],
) -> bool:
    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()
        for nxt in graph[node]:
            if (node, nxt) == skipped_edge:
                continue
            if nxt == target:
                return True
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)

    return False


def transitive_reduce_dag(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    reduced = {node: set(successors) for node, successors in graph.items()}

    for src in sorted(graph):
        for dst in sorted(list(graph[src])):
            if has_alternative_path(src, dst, reduced, (src, dst)):
                reduced[src].remove(dst)

    return reduced


def compute_rank_levels(component_graph: dict[str, set[str]]) -> dict[str, int]:
    topo = topological_sort(component_graph)
    levels = {}

    for node in reversed(topo):
        successors = component_graph[node]
        levels[node] = 1 if not successors else 1 + max(levels[nxt] for nxt in successors)

    return levels


def enumerate_paths_from_start(
    start_component: str,
    reduced_graph: dict[str, set[str]],
) -> list[list[str]]:
    paths = []

    def dfs(node: str, current_path: list[str]):
        successors = sorted(reduced_graph[node])
        if not successors:
            paths.append(current_path.copy())
            return

        for nxt in successors:
            current_path.append(nxt)
            dfs(nxt, current_path)
            current_path.pop()

    dfs(start_component, [start_component])
    return paths


def component_label(component_members: dict[str, list[str]], component_id: str) -> str:
    return "|".join(component_members[component_id])


def classify_pair_relation(
    candidate_over_reference_possible: bool,
    reference_over_candidate_possible: bool,
) -> str:
    if candidate_over_reference_possible and reference_over_candidate_possible:
        return "both_possible"
    if candidate_over_reference_possible:
        return "candidate_possible_over_reference"
    if reference_over_candidate_possible:
        return "reference_possible_over_candidate"
    return "no_possible_preference"


def build_axis_values(
    current_value,
    reference_value,
    step_pct,
    pct_below,
    pct_above,
    step_abs=None,
):
    current_value = float(current_value)
    reference_value = float(reference_value)

    lo = min(current_value, reference_value)
    hi = max(current_value, reference_value)

    span = hi - lo
    scale = max(span, abs(current_value), abs(reference_value), 1.0)

    start = lo - scale * (pct_below / 100.0)
    end = hi + scale * (pct_above / 100.0)
    if step_abs is not None:
        step = float(step_abs)
    else:
        step = scale * (step_pct / 100.0)

    if step <= 0:
        raise ValueError(f"Computed non-positive step: {step}")

    values = np.arange(start, end + step, step)
    values = np.unique(np.concatenate([values, [current_value, reference_value]])).astype(float)
    values.sort()

    return values


def generate_pairwise_samples(
    current_point_row: pd.Series,
    reference_row: pd.Series,
    columns_to_modify: list[str],
    io_cols: list[str],
    pct_below: float,
    pct_above: float,
    step_pct: float,
    step_abs: float | None = None,
    max_candidates_per_step: int | None = None,
) -> pd.DataFrame:
    grid = {}

    for col in columns_to_modify:
        if col not in current_point_row.index:
            raise ValueError(f"Column '{col}' missing in current_point_row")
        if col not in reference_row.index:
            raise ValueError(f"Column '{col}' missing in reference_row")

        current_val = float(current_point_row[col])
        reference_val = float(reference_row[col])

        values = build_axis_values(
            current_value=current_val,
            reference_value=reference_val,
            step_pct=step_pct,
            pct_below=pct_below,
            pct_above=pct_above,
            step_abs=step_abs,
        )

        grid[col] = values

        used_step = float(step_abs) if step_abs is not None else max(
            max(abs(max(current_val, reference_val) - min(current_val, reference_val)), abs(current_val), abs(reference_val), 1.0)
            * (step_pct / 100.0),
            0.0,
        )

        print(
            f"[GRID] {col}: current={current_val:.6f}, reference={reference_val:.6f}, "
            f"step={used_step:.6f}, points={len(values)}, min={values.min():.6f}, max={values.max():.6f}"
        )

    total_candidates = 1
    for values in grid.values():
        total_candidates *= len(values)

    if max_candidates_per_step is not None and total_candidates > max_candidates_per_step:
        raise ValueError(
            f"Sampling would generate {total_candidates} candidates, which exceeds "
            f"max_candidates_per_step={max_candidates_per_step}. "
            f"Reduce dimensions, increase step size, or set a higher limit explicitly."
        )

    if total_candidates <= 0:
        raise ValueError("No candidate combinations generated for pairwise sampling.")

    current_name = str(current_point_row["name"])
    reference_name = str(reference_row["name"])

    results = []
    for i, combo in enumerate(product(*grid.values())):
        row = {
            "name": f"{current_name}__to__{reference_name}_cand_{i}"
        }

        for c in io_cols:
            row[c] = current_point_row[c]

        for c, v in zip(columns_to_modify, combo):
            row[c] = v

        results.append(row)

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError("Pairwise sampling produced an empty dataframe.")

    return result_df


def export_single_reference_row(
    reference_row: pd.Series,
    io_cols: list[str],
    output_csv: str,
):
    row = {"name": reference_row["name"]}
    for col in io_cols:
        row[col] = reference_row[col]

    df = pd.DataFrame([row])
    ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False)


def euclidean_distance(
    candidate_row: pd.Series,
    current_point_row: pd.Series,
    columns: list[str],
) -> float:
    candidate = np.array([float(candidate_row[c]) for c in columns], dtype=float)
    current = np.array([float(current_point_row[c]) for c in columns], dtype=float)
    return float(np.linalg.norm(candidate - current))


def select_nearest_candidate(
    df_valid: pd.DataFrame,
    current_point_row: pd.Series,
    columns: list[str],
    zero_tol: float = 1e-12,
) -> tuple[pd.DataFrame, pd.Series]:
    out = df_valid.copy()
    out["distance_from_previous"] = out.apply(
        lambda row: euclidean_distance(row, current_point_row, columns),
        axis=1,
    )
    out["is_same_as_previous"] = out["distance_from_previous"] <= zero_tol

    selectable = out[~out["is_same_as_previous"]].copy()
    if selectable.empty:
        selectable = out.copy()

    selectable = selectable.sort_values(
        by=["distance_from_previous", "name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected = selectable.iloc[0].copy()
    out["selected_in_step"] = out["name"] == selected["name"]

    return out, selected


def build_next_point_row(selected_row: pd.Series, io_cols: list[str]) -> pd.Series:
    row = {"name": selected_row["name"]}
    for col in io_cols:
        row[col] = selected_row[col]
    return pd.Series(row)


def expand_component_paths_to_dmu_paths(
    component_paths: list[list[str]],
    component_members: dict[str, list[str]],
    start_dmu: str,
) -> list[list[str]]:
    dmu_paths = []

    for component_path in component_paths:
        partial_paths = [[start_dmu]]

        for component_id in component_path[1:]:
            members = component_members[component_id]
            next_partial_paths = []

            for partial in partial_paths:
                for member in members:
                    next_partial_paths.append(partial + [member])

            partial_paths = next_partial_paths

        dmu_paths.extend(partial_paths)

    return dmu_paths


def main():
    args = parse_args()
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    run_dir = create_run_output_dir(args.output_dir, "necessary_relation")
    df_input = pd.read_csv(args.input)

    if "name" not in df_input.columns:
        raise ValueError("Input CSV must contain a 'name' column.")

    if args.target not in df_input["name"].astype(str).tolist():
        raise ValueError(f"Target DMU '{args.target}' not found in input CSV.")

    inputs, outputs = get_io_columns(df_input)
    io_cols = inputs + outputs

    missing_columns = [col for col in columns if col not in io_cols]
    if missing_columns:
        raise ValueError(
            f"Columns passed in --columns must be DEA input/output columns. Missing/invalid: {missing_columns}"
        )

    dmu_order = df_input["name"].astype(str).tolist()
    start_row = df_input[df_input["name"] == args.target].iloc[0].copy()

    full_relations_csv = run_dir / "preference_relations_all.csv"

    print("Step 1: generating necessary/possible relations for all DMUs...")
    print(f"Run output directory: {run_dir}")

    generate_preference_relations_with_java(
        input_csv=path_for_java(args.input, args.java_entry),
        output_csv=path_for_java(str(full_relations_csv), args.java_entry),
        java_entry=args.java_entry,
        main_class=args.preference_main_class,
        maven_executable=args.maven_executable,
    )

    df_relations = pd.read_csv(full_relations_csv)
    necessary_matrix = build_relation_matrix(
        df_relations=df_relations,
        dmu_order=dmu_order,
        value_col="necessary_preferred",
    )
    possible_matrix = build_relation_matrix(
        df_relations=df_relations,
        dmu_order=dmu_order,
        value_col="possible_preferred",
    )

    necessary_matrix_path = run_dir / "necessary_matrix.csv"
    possible_matrix_path = run_dir / "possible_matrix.csv"
    necessary_matrix.to_csv(necessary_matrix_path)
    possible_matrix.to_csv(possible_matrix_path)

    print("Step 2: building necessary-relation ranking graph...")
    raw_graph = build_worse_to_better_graph(necessary_matrix)
    components = strongly_connected_components(raw_graph)
    component_id_by_dmu, component_members, component_graph = build_component_graph(
        components=components,
        graph=raw_graph,
        dmu_order=dmu_order,
    )

    reduced_component_graph = transitive_reduce_dag(component_graph)
    rank_levels = compute_rank_levels(reduced_component_graph)

    component_rows = []
    for component_id in sorted(component_members):
        members = component_members[component_id]
        component_rows.append(
            {
                "component_id": component_id,
                "members": "|".join(members),
                "component_size": len(members),
                "rank_level": rank_levels[component_id],
                "is_top_rank": len(reduced_component_graph[component_id]) == 0,
                "is_start_component": args.target in members,
            }
        )

    components_path = run_dir / "necessary_components.csv"
    pd.DataFrame(component_rows).to_csv(components_path, index=False)

    ranking_rows = []
    for dmu in dmu_order:
        better_than_count = int(necessary_matrix.loc[dmu].astype(int).sum() - 1)
        worse_than_count = int(necessary_matrix[dmu].astype(int).sum() - 1)
        component_id = component_id_by_dmu[dmu]

        ranking_rows.append(
            {
                "name": dmu,
                "component_id": component_id,
                "component_members": "|".join(component_members[component_id]),
                "rank_level": rank_levels[component_id],
                "is_top_rank": len(reduced_component_graph[component_id]) == 0,
                "better_than_count": better_than_count,
                "worse_than_count": worse_than_count,
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        by=["rank_level", "worse_than_count", "better_than_count", "name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    ranking_path = run_dir / "necessary_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)

    edge_rows = []
    for src_component in sorted(reduced_component_graph):
        for dst_component in sorted(reduced_component_graph[src_component]):
            edge_rows.append(
                {
                    "from_component_id": src_component,
                    "from_members": component_label(component_members, src_component),
                    "from_rank_level": rank_levels[src_component],
                    "to_component_id": dst_component,
                    "to_members": component_label(component_members, dst_component),
                    "to_rank_level": rank_levels[dst_component],
                }
            )

    cover_edges_path = run_dir / "necessary_cover_edges.csv"
    pd.DataFrame(edge_rows).to_csv(cover_edges_path, index=False)

    print("Step 3: enumerating all paths from the selected start DMU to top rank...")
    start_component = component_id_by_dmu[args.target]
    component_paths = enumerate_paths_from_start(
        start_component=start_component,
        reduced_graph=reduced_component_graph,
    )

    component_path_rows = []
    component_path_step_rows = []

    for idx, component_path in enumerate(component_paths, start=1):
        path_id = f"component_path_{idx:04d}"
        labels = [component_label(component_members, component_id) for component_id in component_path]

        component_path_rows.append(
            {
                "path_id": path_id,
                "start_dmu": args.target,
                "step_count": max(len(component_path) - 1, 0),
                "top_component_id": component_path[-1],
                "top_component_members": labels[-1],
                "path_string": " -> ".join(labels),
            }
        )

        for position, component_id in enumerate(component_path):
            component_path_step_rows.append(
                {
                    "path_id": path_id,
                    "path_position": position,
                    "component_id": component_id,
                    "component_members": component_label(component_members, component_id),
                    "rank_level": rank_levels[component_id],
                    "is_start": position == 0,
                    "is_top_rank": len(reduced_component_graph[component_id]) == 0,
                }
            )

    component_paths_path = run_dir / f"component_paths_from_{sanitize_name(args.target)}.csv"
    component_path_steps_path = run_dir / f"component_path_steps_from_{sanitize_name(args.target)}.csv"
    pd.DataFrame(component_path_rows).to_csv(component_paths_path, index=False)
    pd.DataFrame(component_path_step_rows).to_csv(component_path_steps_path, index=False)

    dmu_paths = expand_component_paths_to_dmu_paths(
        component_paths=component_paths,
        component_members=component_members,
        start_dmu=args.target,
    )

    if args.max_paths is not None:
        print(f"Limiting sampled DMU paths to first {args.max_paths}.")
        dmu_paths = dmu_paths[:args.max_paths]

    dmu_path_rows = []
    dmu_path_step_rows = []
    for idx, dmu_path in enumerate(dmu_paths, start=1):
        path_id = f"path_{idx:04d}"
        dmu_path_rows.append(
            {
                "path_id": path_id,
                "start_dmu": args.target,
                "step_count": max(len(dmu_path) - 1, 0),
                "top_dmu": dmu_path[-1],
                "path_string": " -> ".join(dmu_path),
            }
        )

        for position, dmu_name in enumerate(dmu_path):
            dmu_path_step_rows.append(
                {
                    "path_id": path_id,
                    "path_position": position,
                    "dmu_name": dmu_name,
                    "component_id": component_id_by_dmu[dmu_name],
                    "rank_level": rank_levels[component_id_by_dmu[dmu_name]],
                    "is_start": position == 0,
                    "is_top_rank": len(reduced_component_graph[component_id_by_dmu[dmu_name]]) == 0,
                }
            )

    dmu_paths_path = run_dir / f"paths_from_{sanitize_name(args.target)}.csv"
    dmu_path_steps_path = run_dir / f"path_steps_from_{sanitize_name(args.target)}.csv"
    pd.DataFrame(dmu_path_rows).to_csv(dmu_paths_path, index=False)
    pd.DataFrame(dmu_path_step_rows).to_csv(dmu_path_steps_path, index=False)

    print("Step 4: pairwise possible relations for all DMUs appearing on the paths...")
    unique_targets = []
    seen_targets = set()
    for dmu_path in dmu_paths:
        for target_name in dmu_path[1:]:
            if target_name not in seen_targets:
                seen_targets.add(target_name)
                unique_targets.append(target_name)

    pairwise_dir = run_dir / "pairwise_possible"
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows = []
    pairwise_by_target = {}

    for target_name in unique_targets:
        pair_dir = pairwise_dir / sanitize_name(target_name)
        pair_dir.mkdir(parents=True, exist_ok=True)

        pair_input_csv = pair_dir / "pair_input.csv"
        pair_output_csv = pair_dir / "pair_relations.csv"

        pair_df = (
            df_input[df_input["name"].astype(str).isin([args.target, target_name])]
            .copy()
        )
        pair_df["__order"] = pair_df["name"].map({args.target: 0, target_name: 1})
        pair_df = pair_df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

        if len(pair_df) != 2:
            raise ValueError(
                f"Expected exactly 2 DMUs for pair '{args.target}' vs '{target_name}', got {len(pair_df)}."
            )

        pair_df.to_csv(pair_input_csv, index=False)

        generate_preference_relations_with_java(
            input_csv=path_for_java(str(pair_input_csv), args.java_entry),
            output_csv=path_for_java(str(pair_output_csv), args.java_entry),
            java_entry=args.java_entry,
            main_class=args.preference_main_class,
            maven_executable=args.maven_executable,
        )

        df_pair = pd.read_csv(pair_output_csv)

        candidate_over_reference_possible = to_bool(
            df_pair[
                (df_pair["source_dmu"] == args.target)
                & (df_pair["target_dmu"] == target_name)
            ].iloc[0]["possible_preferred"]
        )
        reference_over_candidate_possible = to_bool(
            df_pair[
                (df_pair["source_dmu"] == target_name)
                & (df_pair["target_dmu"] == args.target)
            ].iloc[0]["possible_preferred"]
        )
        candidate_over_reference_necessary = to_bool(
            df_pair[
                (df_pair["source_dmu"] == args.target)
                & (df_pair["target_dmu"] == target_name)
            ].iloc[0]["necessary_preferred"]
        )
        reference_over_candidate_necessary = to_bool(
            df_pair[
                (df_pair["source_dmu"] == target_name)
                & (df_pair["target_dmu"] == args.target)
            ].iloc[0]["necessary_preferred"]
        )

        relation_row = {
            "start_dmu": args.target,
            "target_dmu": target_name,
            "candidate_over_reference_possible": candidate_over_reference_possible,
            "reference_over_candidate_possible": reference_over_candidate_possible,
            "candidate_over_reference_necessary": candidate_over_reference_necessary,
            "reference_over_candidate_necessary": reference_over_candidate_necessary,
            "possible_relation_label": classify_pair_relation(
                candidate_over_reference_possible=candidate_over_reference_possible,
                reference_over_candidate_possible=reference_over_candidate_possible,
            ),
            "pair_input_csv": str(pair_input_csv),
            "pair_output_csv": str(pair_output_csv),
        }

        pairwise_rows.append(relation_row)
        pairwise_by_target[target_name] = relation_row

    pairwise_summary_path = run_dir / f"pairwise_possible_summary_from_{sanitize_name(args.target)}.csv"
    pd.DataFrame(pairwise_rows).to_csv(pairwise_summary_path, index=False)

    print("Step 5: sampling along every concrete path and selecting the nearest valid point per step...")
    sampled_paths_dir = run_dir / "sampled_paths"
    sampled_paths_dir.mkdir(parents=True, exist_ok=True)

    aggregated_valid_candidates = []
    aggregated_selected_points = []
    aggregated_step_summary = []
    path_status_rows = []

    for path_idx, dmu_path in enumerate(dmu_paths, start=1):
        path_id = f"path_{path_idx:04d}"
        path_string = " -> ".join(dmu_path)
        path_dir = sampled_paths_dir / path_id
        path_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"SAMPLING PATH {path_id}")
        print(f"Path: {path_string}")

        current_point_row = start_row.copy()
        path_completed = True
        failure_reason = None

        for step_index, reference_name in enumerate(dmu_path[1:], start=1):
            reference_row = df_input[df_input["name"] == reference_name].iloc[0].copy()
            step_dir = path_dir / f"step_{step_index:02d}_{sanitize_name(reference_name)}"
            step_dir.mkdir(parents=True, exist_ok=True)

            reference_csv = step_dir / "reference.csv"
            samples_csv = step_dir / "samples.csv"
            results_csv = step_dir / "results.csv"
            possible_true_csv = step_dir / "possible_true.csv"
            selected_point_csv = step_dir / "selected_point.csv"

            print("-" * 80)
            print(f"Step {step_index}: {current_point_row['name']} -> {reference_name}")

            export_single_reference_row(
                reference_row=reference_row,
                io_cols=io_cols,
                output_csv=str(reference_csv),
            )

            sampled_df = generate_pairwise_samples(
                current_point_row=current_point_row,
                reference_row=reference_row,
                columns_to_modify=columns,
                io_cols=io_cols,
                pct_below=args.pct_below,
                pct_above=args.pct_above,
                step_pct=args.step_pct,
                step_abs=args.step_abs,
                max_candidates_per_step=args.max_candidates_per_step,
            )
            sampled_df.to_csv(samples_csv, index=False)

            evaluate_pairwise_possible_candidates_with_java(
                reference_csv=path_for_java(str(reference_csv), args.java_entry),
                candidates_csv=path_for_java(str(samples_csv), args.java_entry),
                results_csv=path_for_java(str(results_csv), args.java_entry),
                java_entry=args.java_entry,
                main_class=args.possible_evaluator_main_class,
                maven_executable=args.maven_executable,
            )

            df_results = pd.read_csv(results_csv)
            df_results["reference_over_candidate_possible"] = df_results[
                "reference_over_candidate_possible"
            ].map(to_bool)
            df_results["candidate_over_reference_possible"] = df_results[
                "candidate_over_reference_possible"
            ].map(to_bool)
            df_results["reference_over_candidate_necessary"] = df_results[
                "reference_over_candidate_necessary"
            ].map(to_bool)
            df_results["candidate_over_reference_necessary"] = df_results[
                "candidate_over_reference_necessary"
            ].map(to_bool)

            df_valid = df_results[df_results["candidate_over_reference_possible"]].copy()

            if df_valid.empty:
                path_completed = False
                failure_reason = (
                    f"No sampled point satisfied candidate_over_reference_possible for "
                    f"{current_point_row['name']} -> {reference_name}"
                )
                print(f"[STOP] {failure_reason}")

                aggregated_step_summary.append(
                    {
                        "path_id": path_id,
                        "path_string": path_string,
                        "step_index": step_index,
                        "from_point_name": current_point_row["name"],
                        "to_reference_name": reference_name,
                        "sample_count": len(df_results),
                        "possible_true_count": 0,
                        "selected_point_name": None,
                        "selected_distance": None,
                        "step_completed": False,
                        "failure_reason": failure_reason,
                    }
                )
                break

            df_valid, selected_row = select_nearest_candidate(
                df_valid=df_valid,
                current_point_row=current_point_row,
                columns=columns,
            )

            metadata = {
                "path_id": path_id,
                "path_string": path_string,
                "step_index": step_index,
                "from_point_name": current_point_row["name"],
                "to_reference_name": reference_name,
            }

            for key, value in metadata.items():
                df_valid[key] = value

            df_valid["possible_relation_label"] = df_valid.apply(
                lambda row: classify_pair_relation(
                    candidate_over_reference_possible=bool(row["candidate_over_reference_possible"]),
                    reference_over_candidate_possible=bool(row["reference_over_candidate_possible"]),
                ),
                axis=1,
            )

            df_valid.to_csv(possible_true_csv, index=False)

            selected_out = pd.DataFrame([selected_row]).copy()
            for key, value in metadata.items():
                selected_out[key] = value

            selected_out["possible_relation_label"] = classify_pair_relation(
                candidate_over_reference_possible=bool(selected_row["candidate_over_reference_possible"]),
                reference_over_candidate_possible=bool(selected_row["reference_over_candidate_possible"]),
            )
            selected_out.to_csv(selected_point_csv, index=False)

            aggregated_valid_candidates.append(df_valid)
            aggregated_selected_points.append(selected_out)
            aggregated_step_summary.append(
                {
                    "path_id": path_id,
                    "path_string": path_string,
                    "step_index": step_index,
                    "from_point_name": current_point_row["name"],
                    "to_reference_name": reference_name,
                    "sample_count": len(df_results),
                    "possible_true_count": len(df_valid),
                    "selected_point_name": selected_row["name"],
                    "selected_distance": float(selected_row["distance_from_previous"]),
                    "step_completed": True,
                    "failure_reason": None,
                }
            )

            print(
                f"Selected point: {selected_row['name']} | "
                f"distance={float(selected_row['distance_from_previous']):.10f}"
            )

            current_point_row = build_next_point_row(selected_row, io_cols)

        path_status_rows.append(
            {
                "path_id": path_id,
                "path_string": path_string,
                "path_completed": path_completed,
                "failure_reason": failure_reason,
                "steps_in_path": max(len(dmu_path) - 1, 0),
                "steps_completed": sum(
                    1
                    for row in aggregated_step_summary
                    if row["path_id"] == path_id and row["step_completed"]
                ),
            }
        )

    valid_all_df = (
        pd.concat(aggregated_valid_candidates, ignore_index=True)
        if aggregated_valid_candidates else pd.DataFrame()
    )
    selected_all_df = (
        pd.concat(aggregated_selected_points, ignore_index=True)
        if aggregated_selected_points else pd.DataFrame()
    )
    step_summary_df = pd.DataFrame(aggregated_step_summary)
    path_status_df = pd.DataFrame(path_status_rows)

    valid_all_path = run_dir / "possible_true_all_steps.csv"
    selected_all_path = run_dir / "selected_points_all_paths.csv"
    step_summary_path = run_dir / "path_step_summary.csv"
    path_status_path = run_dir / "path_status.csv"

    valid_all_df.to_csv(valid_all_path, index=False)
    selected_all_df.to_csv(selected_all_path, index=False)
    step_summary_df.to_csv(step_summary_path, index=False)
    path_status_df.to_csv(path_status_path, index=False)

    print("DONE")
    print(f"All relations:           {full_relations_csv}")
    print(f"Necessary matrix:        {necessary_matrix_path}")
    print(f"Possible matrix:         {possible_matrix_path}")
    print(f"Components:              {components_path}")
    print(f"Ranking:                 {ranking_path}")
    print(f"Cover edges:             {cover_edges_path}")
    print(f"Component paths:         {component_paths_path}")
    print(f"DMU paths:               {dmu_paths_path}")
    print(f"Pairwise summary:        {pairwise_summary_path}")
    print(f"Possible true all steps: {valid_all_path}")
    print(f"Selected points:         {selected_all_path}")
    print(f"Step summary:            {step_summary_path}")
    print(f"Path status:             {path_status_path}")


if __name__ == "__main__":
    main()
