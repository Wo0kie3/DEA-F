import math
import os
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd


def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def create_run_output_dir(base_output_dir: str, method_name: str) -> Path:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / method_name / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def path_for_java(path_str: str, java_entry: str) -> str:
    abs_target = Path(path_str).resolve()
    abs_java = Path(java_entry).resolve()
    return os.path.relpath(abs_target, start=abs_java)


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


def to_bool(value) -> bool:
    return str(value).strip().lower() == "true"


def resolve_first_present(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"None of the expected columns found: {candidates}")


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
    return matrix.apply(lambda column: column.map(to_bool))


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


def enumerate_paths_from_start(
    start_node: str,
    graph: dict[str, set[str]],
) -> list[list[str]]:
    paths = []

    def dfs(node: str, current_path: list[str]):
        successors = sorted(graph[node])
        if not successors:
            paths.append(current_path.copy())
            return

        for nxt in successors:
            current_path.append(nxt)
            dfs(nxt, current_path)
            current_path.pop()

    dfs(start_node, [start_node])
    return paths


def compute_front_map(component_graph: dict[str, set[str]]) -> dict[str, int]:
    remaining = {node: set(successors) for node, successors in component_graph.items()}
    front_map = {}
    front_idx = 1

    while remaining:
        maxima = sorted([node for node, successors in remaining.items() if not successors])
        if not maxima:
            raise ValueError("Could not compute fronts on component graph.")

        for node in maxima:
            front_map[node] = front_idx

        remaining = {
            node: {s for s in successors if s not in maxima}
            for node, successors in remaining.items()
            if node not in maxima
        }
        front_idx += 1

    return front_map


def enumerate_front_paths(
    start_component: str,
    component_graph: dict[str, set[str]],
    front_map: dict[str, int],
    require_edge_monotonicity: bool = False,
) -> list[list[str]]:
    start_front = front_map[start_component]
    target_fronts = list(range(start_front - 1, 0, -1))
    paths = []

    def dfs(current_component: str, target_idx: int, current_path: list[str]):
        if target_idx >= len(target_fronts):
            paths.append(current_path.copy())
            return

        next_front = target_fronts[target_idx]
        if require_edge_monotonicity:
            candidates = sorted(
                [nxt for nxt in component_graph[current_component] if front_map[nxt] == next_front]
            )
        else:
            candidates = sorted([node for node, front in front_map.items() if front == next_front])

        for nxt in candidates:
            current_path.append(nxt)
            dfs(nxt, target_idx + 1, current_path)
            current_path.pop()

    dfs(start_component, 0, [start_component])
    return paths


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


def limit_paths(paths: list[list[str]], max_paths: int | None) -> list[list[str]]:
    if max_paths is None or max_paths < 0:
        return paths
    return paths[:max_paths]


def rank_milestones(start_rank: int, target_rank: int, stages: int) -> list[int]:
    milestones = []
    for h in range(stages + 1):
        raw_value = start_rank - (h / stages) * (start_rank - target_rank)
        milestones.append(int(math.ceil(raw_value)))
    return milestones


def linear_milestones(start_value: float, target_value: float, stages: int) -> list[float]:
    return [
        float(start_value + (h / stages) * (target_value - start_value))
        for h in range(stages + 1)
    ]


def paths_to_frame(paths: list[list[str]]) -> pd.DataFrame:
    rows = []
    for path_idx, path in enumerate(paths, start=1):
        row = {
            "path_id": f"path_{path_idx:06d}",
            "path_length": len(path) - 1,
        }
        for stage_idx, name in enumerate(path):
            row[f"stage_{stage_idx:02d}_name"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def component_paths_to_frame(
    component_paths: list[list[str]],
    component_members: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for path_idx, path in enumerate(component_paths, start=1):
        row = {
            "path_id": f"path_{path_idx:06d}",
            "path_length": len(path) - 1,
        }
        for stage_idx, component_id in enumerate(path):
            row[f"stage_{stage_idx:02d}_component_id"] = component_id
            row[f"stage_{stage_idx:02d}_members"] = "|".join(component_members[component_id])
        rows.append(row)
    return pd.DataFrame(rows)


def write_stage_candidates(
    stage_candidates: list[pd.DataFrame],
    output_csv: str,
):
    frames = []
    for idx, frame in enumerate(stage_candidates, start=1):
        if frame.empty:
            continue
        tmp = frame.copy()
        tmp.insert(0, "stage", idx)
        frames.append(tmp)

    ensure_parent_dir(output_csv)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_csv, index=False)
    else:
        pd.DataFrame().to_csv(output_csv, index=False)


def parse_columns_arg(columns_arg: str | None, io_cols: list[str]) -> list[str]:
    if columns_arg is None or not str(columns_arg).strip():
        return list(io_cols)

    columns = [c.strip() for c in str(columns_arg).split(",") if c.strip()]
    missing = [c for c in columns if c not in io_cols]
    if missing:
        raise ValueError(f"Columns are not DEA input/output columns: {missing}")
    return columns


def _axis_values(start: float, end: float, step_pct: float, step_abs: float | None, min_points: int) -> list[float]:
    start = float(start)
    end = float(end)
    if math.isclose(start, end, rel_tol=1e-12, abs_tol=1e-12):
        return [start]

    lo = min(start, end)
    hi = max(start, end)
    span = hi - lo
    if step_abs is not None:
        step = float(step_abs)
    else:
        step = span * (float(step_pct) / 100.0)
    step = max(step, 1e-12)

    values = []
    current = lo
    while current <= hi + step * 0.5:
        values.append(float(current))
        current += step
    values.extend([lo, hi, start, end])

    if min_points is not None and min_points > 1 and len(values) < min_points:
        values.extend(
            lo + (hi - lo) * idx / (min_points - 1)
            for idx in range(min_points)
        )

    unique = sorted({round(float(v), 12) for v in values if lo - 1e-12 <= float(v) <= hi + 1e-12})
    if start > end:
        unique.reverse()
    return unique


def generate_attainable_fictive_candidates(
    df: pd.DataFrame,
    target_row: pd.Series,
    columns_to_modify: list[str],
    pct_above: float,
    step_pct: float,
    step_abs: float | None,
    min_points_per_dim: int,
    max_candidates: int,
    name_prefix: str,
) -> pd.DataFrame:
    inputs, outputs = get_io_columns(df)
    io_cols = inputs + outputs
    grid = {}

    for col in columns_to_modify:
        current = float(target_row[col])
        observed = df[col].astype(float)

        if col in inputs:
            end = max(0.0, float(observed.min()) * (1.0 - pct_above / 100.0))
        elif col in outputs:
            end = float(observed.max()) * (1.0 + pct_above / 100.0)
        else:
            raise ValueError(f"Column '{col}' is neither an input nor an output.")

        grid[col] = _axis_values(
            start=current,
            end=end,
            step_pct=step_pct,
            step_abs=step_abs,
            min_points=min_points_per_dim,
        )

    total = 1
    for values in grid.values():
        total *= len(values)
    if total > max_candidates:
        raise ValueError(
            f"Fictive candidate space has {total} points, above max_candidates={max_candidates}. "
            "Increase step size, reduce --columns, or raise --max-candidates explicitly."
        )

    rows = []
    for idx, combo in enumerate(product(*grid.values()), start=1):
        row = {"name": f"{name_prefix}_fictive_{idx:06d}", "state_type": "fictive"}
        for col in io_cols:
            row[col] = float(target_row[col])
        for col, value in zip(grid.keys(), combo):
            row[col] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def add_real_state_type(metrics: pd.DataFrame, df_input: pd.DataFrame) -> pd.DataFrame:
    inputs, outputs = get_io_columns(df_input)
    io_cols = inputs + outputs
    real = metrics.merge(df_input[["name", *io_cols]], on="name", how="left")
    real["state_type"] = "real"
    return real


def parse_ref_list(value) -> set[str]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {part for part in str(value).split("|") if part}


def component_requirement_mask(metrics: pd.DataFrame, required_members: list[str]) -> pd.Series:
    required = set(required_members)
    return metrics["candidate_necessary_over_refs"].apply(
        lambda value: required.issubset(parse_ref_list(value))
    )


def front_requirement_mask(metrics: pd.DataFrame, front_components: list[list[str]]) -> pd.Series:
    component_sets = [set(component) for component in front_components]

    def satisfies(value) -> bool:
        refs = parse_ref_list(value)
        return any(component.issubset(refs) for component in component_sets)

    return metrics["candidate_necessary_over_refs"].apply(satisfies)


def add_effort_columns(df: pd.DataFrame, target_row: pd.Series, io_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    ranges = {}
    for col in io_cols:
        values = out[col].astype(float)
        span = float(values.max() - values.min())
        if span <= 1e-12:
            span = max(abs(float(target_row[col])), 1.0)
        ranges[col] = span

    effort = []
    for _, row in out.iterrows():
        total = 0.0
        for col in io_cols:
            total += abs(float(row[col]) - float(target_row[col])) / ranges[col]
        effort.append(total)
    out["effort_from_start"] = effort
    return out


def is_attainable_transition(previous: pd.Series, current: pd.Series, inputs: list[str], outputs: list[str]) -> bool:
    for col in inputs:
        if float(current[col]) > float(previous[col]) + 1e-9:
            return False
    for col in outputs:
        if float(current[col]) + 1e-9 < float(previous[col]):
            return False
    return True


def enumerate_state_paths(
    start_row: pd.Series,
    stage_candidates: list[pd.DataFrame],
    inputs: list[str],
    outputs: list[str],
    max_paths: int | None,
) -> list[list[pd.Series]]:
    paths = []

    def dfs(stage_idx: int, current_path: list[pd.Series]):
        if max_paths is not None and len(paths) >= max_paths:
            return
        if stage_idx >= len(stage_candidates):
            paths.append(current_path.copy())
            return

        previous = current_path[-1]
        candidates = stage_candidates[stage_idx]
        for _, candidate in candidates.iterrows():
            if is_attainable_transition(previous, candidate, inputs, outputs):
                current_path.append(candidate)
                dfs(stage_idx + 1, current_path)
                current_path.pop()

    dfs(0, [start_row])
    return paths


def state_paths_to_frame(paths: list[list[pd.Series]], io_cols: list[str]) -> pd.DataFrame:
    rows = []
    for path_idx, path in enumerate(paths, start=1):
        row = {
            "path_id": f"path_{path_idx:06d}",
            "path_length": len(path) - 1,
        }
        for stage_idx, state in enumerate(path):
            prefix = f"stage_{stage_idx:02d}"
            row[f"{prefix}_name"] = state["name"]
            if "state_type" in state:
                row[f"{prefix}_state_type"] = state["state_type"]
            for metric in [
                "best_efficiency",
                "worst_efficiency",
                "best_rank",
                "worst_rank",
                "score_width",
                "rank_width",
                "milestone_gap",
                "effort_from_start",
            ]:
                if metric in state:
                    row[f"{prefix}_{metric}"] = state[metric]
            for col in io_cols:
                if col in state:
                    row[f"{prefix}_{col}"] = state[col]
        rows.append(row)
    return pd.DataFrame(rows)
