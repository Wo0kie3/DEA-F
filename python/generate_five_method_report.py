"""Self-contained, data-driven review of five DEA path experiments."""

import argparse
import hashlib
import html
import json
import platform
from importlib.metadata import version
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from generate_five_method_robot_report import METHODS, _latest_run, _selected_path
from generate_five_unit_2d_html import load_dea_snapshot
from metric_reporting import metric_coverage, selected_metric_table
from metric_calculation_details import explain_metrics, render_metric_table
from report_section_numbering import number_report_sections
from path_metrics import _parse_ref_set
from path_pipeline_common import (
    get_io_columns,
    is_attainable_transition,
    normalization_ranges_from_frame,
    select_transition_candidates,
)


LABELS = {
    "hasse_path": "1. Ścieżka Hassego",
    "front_path": "2. Warstwy rankingu NEC",
    "best_rank_path": "3. Najlepsza ranga",
    "best_efficiency_path": "4. Najlepsza efektywność",
    "robustness_width_path": "5. Szerokość oceny",
}
STAGE_COLORS = ["#df5a56", "#3978b5", "#43a06b", "#9a5eba", "#b77913"]


def table(frame):
    return "<div class='table-wrap'>" + frame.to_html(index=False, border=0, na_rep="—", float_format=lambda v: f"{v:.5g}", escape=True) + "</div>"


def joint_weight_section(witnesses, endpoints, top_targets, inputs, outputs):
    """Present the exported LP witnesses without changing their values or meaning."""
    indexed = witnesses.set_index('name', verify_integrity=True)
    rows = []
    for method, endpoint in endpoints.items():
        witness = indexed.loc[endpoint]
        satisfied = str(witness['joint_top']).strip().lower() == 'true'
        row = {'Metoda': LABELS[method], 'Stan końcowy': endpoint,
               'Wspólny cel': 'Spełniony' if satisfied else 'Niespełniony',
               'Wynik LP': float(witness['joint_score'])}
        for field, columns in [('joint_input_weights', inputs), ('joint_output_weights', outputs)]:
            weights = dict(item.split('=', 1) for item in str(witness[field]).split('|'))
            row.update({f'Waga {column}': float(weights[column]) for column in columns})
        rows.append(row)
    targets = ', '.join(html.escape(target) for target in top_targets)
    return ("<section id='joint-weight-witnesses'><h2>Wagi potwierdzające wspólny cel końcowy</h2>"
            f"<p><b>Jednostki docelowe:</b> {targets}.</p>"
            "<p>Każdy wiersz dotyczy końca ścieżki jednej metody. Solver znajduje dla tego punktu "
            "jeden zestaw wag nakładów i wyników, przy którym punkt spełnia warunek porównania "
            "ze wszystkimi jednostkami docelowymi jednocześnie. Zestaw wag może być inny w każdym wierszu.</p>"
            + table(pd.DataFrame(rows))
            + "<p><b>Jak czytać wynik LP?</b> Ważona suma nakładów badanego punktu jest normalizowana do 1. "
            "Solver maksymalizuje jego ważoną sumę wyników, ograniczoną od góry przez 1. "
            "Dla każdej jednostki docelowej, przy tych samych wagach, ważona suma wyników nie przekracza "
            "ważonej sumy nakładów. Wynik LP równy 1 (z tolerancją 10<sup>−9</sup>) potwierdza spełnienie "
            "wspólnego celu. Jest to wynik testu końcowego, odrębny od raportowanych best_efficiency i worst_efficiency.</p>"
            "<p><b>Jak czytać wagi?</b> To współczynniki modelu, a nie procentowe udziały ani zalecane wielkości zmian. "
            "Waga 0 oznacza, że dany czynnik nie uczestniczy w tym konkretnym porównaniu; model dopuszcza wagi nieujemne. "
            "Tabela pokazuje liczby z pięcioma cyframi znaczącymi; pełną precyzję zachowuje plik possible_goals.csv.</p></section>")


def plot(figure, identifier):
    figure.update_layout(template="plotly_white", margin=dict(l=55, r=30, t=65, b=90),
                         legend=dict(orientation="h", y=-0.13), height=650)
    if figure.layout.meta and 'score_color_range' in figure.layout.meta:
        from report_plot_style import style_search_layout
        figure = go.Figure(style_search_layout(figure.to_plotly_json()))
    return figure.to_html(full_html=False, include_plotlyjs=False, div_id=identifier,
                          config=dict(responsive=True, displaylogo=False, scrollZoom=True))


def scatter(frame, columns, name, color, mode="markers", **kwargs):
    cls = go.Scatter3d if len(columns) == 3 else go.Scatter
    coordinates = dict(x=frame[columns[0]], y=frame[columns[1]])
    if len(columns) == 3:
        coordinates["z"] = frame[columns[2]]
    metric_columns = ["best_efficiency", "worst_efficiency", "best_rank", "worst_rank", "score_width"]
    data = frame.reindex(columns=["name", *metric_columns, "source"]).fillna("—").to_numpy()
    coords_hover = "<br>".join(f"{column}=%{{{axis}:.5f}}" for column, axis in zip(columns, "xyz"))
    marker = dict(size=4 if len(columns) == 3 else 6, color=color, opacity=0.65)
    marker.update(kwargs.pop("marker", {}))
    return cls(**coordinates, name=name, mode=mode, marker=marker, customdata=data,
               hovertemplate="%{customdata[0]}<br>" + coords_hover +
               "<br>E*=%{customdata[1]} · E_=%{customdata[2]}<br>Rangi: %{customdata[3]}–%{customdata[4]}<br>Score width=%{customdata[5]}<br>Źródło/iteracja: %{customdata[6]}<extra>%{fullData.name}</extra>", **kwargs)


def axis_layout(columns, ranges):
    axes = {axis + "axis": dict(title=f"{column} (mniej = lepiej)", range=ranges[column])
            for column, axis in zip(columns, "xyz")}
    if len(columns) == 3:
        return dict(scene={**axes, "aspectmode": "data"})
    axes["yaxis"].update(scaleanchor="x", scaleratio=1)
    return axes


def hasse_figure(snapshot, target):
    components = {row["component_id"]: row for row in snapshot["components"]}
    graph = nx.DiGraph()
    graph.add_nodes_from(components)
    graph.add_edges_from((row["source_component"], row["target_component"]) for row in snapshot["edges"])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Necessary component graph is not a DAG")
    start = next(k for k, row in components.items() if target in str(row["members"]).split("|"))
    reachable = {start, *nx.descendants(graph, start)}
    positions = {}
    for rank in sorted({int(row["necessary_rank"]) for row in components.values()}):
        nodes = sorted(k for k, row in components.items() if int(row["necessary_rank"]) == rank)
        positions.update({k: (index - (len(nodes) - 1) / 2, -rank) for index, k in enumerate(nodes)})
    figure = go.Figure()
    for highlighted in (False, True):
        xs, ys = [], []
        for source, dest in graph.edges:
            if (source in reachable and dest in reachable) != highlighted:
                continue
            xs.extend([positions[source][0], positions[dest][0], None])
            ys.extend([positions[source][1], positions[dest][1], None])
        figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Ścieżki od targetu" if highlighted else "Pozostałe krawędzie NEC",
                                   line=dict(color="#3978b5" if highlighted else "#d1d8e0", width=3 if highlighted else 1), hoverinfo="skip"))
    nodes = list(graph)
    figure.add_trace(go.Scatter(x=[positions[k][0] for k in nodes], y=[positions[k][1] for k in nodes], mode="markers+text",
                               text=[components[k]["members"] for k in nodes], textposition="top center", name="Klasy NEC",
                               marker=dict(size=[18 if k == start else 11 for k in nodes], color=["#f0aa18" if k == start else "#3978b5" if k in reachable else "#b5bec9" for k in nodes]),
                               hovertemplate="%{text}<extra></extra>"))
    figure.update_layout(title="Diagram Hassego: poprawa biegnie ku górze", xaxis=dict(visible=False), yaxis=dict(title="Poziom NEC (R1 na górze)", tickvals=sorted({p[1] for p in positions.values()}), ticktext=[f"R{abs(v)}" for v in sorted({p[1] for p in positions.values()})]))

    @lru_cache(None)
    def path_count(node):
        return sum(path_count(child) for child in graph.successors(node)) if graph.out_degree(node) else 1

    return figure, path_count(start)


def evaluated_pool(run, stage):
    files = [run / "fictive_candidate_metrics.csv", run / "global_search_metrics.csv"]
    files += sorted(run.glob(f"stage_{stage:02d}_*_metrics.csv"))
    frames = []
    for path in files:
        if path.is_file():
            frame = pd.read_csv(path)
            if "name" in frame and "i1" in frame:
                if "source" not in frame:
                    frame["source"] = path.stem
                else:
                    frame["source"] = frame["source"].fillna(path.stem)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates("name", keep="last") if frames else pd.DataFrame()


def transition_front(pool, previous, current, method, reference, columns, params):
    """Reconstruct the exact branch's pre-limit front using the pipeline selector."""
    for key in ("milestone_component_id", "milestone_front", "milestone_route_id", "milestone_target_name", "milestone_goal_kind"):
        if key in current and pd.notna(current[key]) and key in pool:
            mask = pd.to_numeric(pool[key]).eq(float(current[key])) if key == "milestone_front" else pool[key].astype(str).eq(str(current[key]))
            pool = pool.loc[mask]
    required_key = {"hasse_path": "milestone_component_id", "front_path": "milestone_front"}.get(method)
    if required_key and (required_key not in current or pd.isna(current[required_key])):
        raise ValueError(f"Missing {required_key} in paths.csv: rerun structural method with milestone export")
    pool = pool.drop_duplicates("name").copy()
    inputs, outputs = get_io_columns(reference)
    predicate = None
    if method == "robustness_width_path":
        if params.get("width_kind", "score") == "score":
            predicate = lambda prev, cur: cur["best_efficiency"] + 1e-9 >= prev["best_efficiency"]
        else:
            predicate = lambda prev, cur: cur["best_rank"] <= prev["best_rank"] + 1e-9
    ranges = normalization_ranges_from_frame(reference, inputs + outputs, reference.iloc[0])
    def mapping(value):
        return {key: float(number) for key, number in (part.split("=") for part in str(value).split(","))} if value and str(value) != "none" else None
    budget = params.get("max_stage_effort", "none")
    front = select_transition_candidates(previous, pool, inputs, outputs, ranges,
        points_per_transition=None, minimal_inputs=columns, minimal_outputs=[],
        effort_inputs=columns, effort_outputs=[], transition_predicate=predicate,
        factor_weights=mapping(params.get("factor_weights")),
        max_stage_effort=None if str(budget) == "none" else float(budget),
        max_factor_changes=mapping(params.get("max_factor_changes")))
    return pool, front


def state_frame(path):
    states = []
    for stage in range(int(path["path_length"]) + 1):
        prefix = f"stage_{stage:02d}_"
        states.append({"stage": stage, **{key[len(prefix):]: value for key, value in path.items() if key.startswith(prefix)}})
    return pd.DataFrame(states)


def method_figure(run, method, path, reference, columns, ranges, params):
    from report_plot_style import score_limits, style_search_figure
    states = state_frame(path)
    stage_pool = pd.read_csv(run / "stage_candidates.csv")
    if method not in ("hasse_path", "front_path") and "milestone_target" not in states:
        states["milestone_target"] = float("nan")
        for stage in range(1, len(states)):
            goals = stage_pool.loc[stage_pool["stage"].eq(stage), "milestone_target"].dropna().unique()
            if len(goals) != 1:
                raise ValueError(f"Ambiguous numeric milestone for {method}, stage {stage}")
            states.loc[stage, "milestone_target"] = goals[0]
    figure = go.Figure()
    groups, audits = [], []
    for stage in range(1, len(states)):
        previous, current = states.iloc[stage - 1], states.iloc[stage]
        pool = stage_pool.loc[stage_pool["stage"].eq(stage)].copy()
        pool, front = transition_front(pool, previous, current, method, reference, columns, params)
        tested = evaluated_pool(run, stage)
        tested = pd.concat([pool, tested], ignore_index=True).drop_duplicates("name", keep="last")
        inputs, outputs = get_io_columns(reference)
        attainable = tested.apply(lambda row: is_attainable_transition(previous, row, inputs, outputs), axis=1)
        eligible = tested["name"].isin(pool["name"]) & attainable
        on_front = tested["name"].isin(front["name"])
        dominating = pool.loc[pool.apply(lambda row: is_attainable_transition(previous, row, inputs, outputs), axis=1)]
        if method == "robustness_width_path":
            if params.get("width_kind", "score") == "score":
                dominating = dominating.loc[dominating["best_efficiency"] + 1e-9 >= previous["best_efficiency"]]
            else:
                dominating = dominating.loc[dominating["best_rank"] <= previous["best_rank"] + 1e-9]
        # For minimized inputs, a closer point has no smaller input values.
        domination = (dominating[columns] >= current[columns].astype(float) - 1e-9).all(axis=1) & (dominating[columns] > current[columns].astype(float) + 1e-9).any(axis=1)
        audits.append(dict(method=method, stage=stage, predecessor=previous["name"], selected=current["name"],
                           tested_points=len(tested), milestone_pool=len(pool), front_size=len(front),
                           selected_on_front=str(current["name"]) in set(front["name"].astype(str)), dominating_points=int(domination.sum())))
        group = []
        categories = [
            ("Poza pulą celu / próby diagnostyczne", ~tested["name"].isin(pool["name"]), True),
            ("Cel spełniony, nieosiągalne z poprzednika", tested["name"].isin(pool["name"]) & ~attainable, True),
            ("Cel spełniony, poza frontem przejścia", eligible & ~on_front, True),
        ]
        for category, (label, mask, visibility) in enumerate(categories):
            frame = tested.loc[mask]
            figure.add_trace(scatter(frame, columns, label, frame["best_efficiency"], visible=visibility,
                                    marker=dict(colorscale="Viridis", cmin=0, cmax=1, size=3, opacity=0.3,
                                                showscale=category == 2, colorbar=dict(title="E*", len=0.5))))
            group.append((len(figure.data) - 1, visibility))
        figure.add_trace(scatter(front, columns, f"Front minimalnych zmian: etap {stage}", STAGE_COLORS[(stage-1) % len(STAGE_COLORS)], marker=dict(size=6, opacity=0.9)))
        group.append((len(figure.data) - 1, True))
        figure.add_trace(scatter(states.iloc[[stage-1, stage]], columns, "Aktualne przejście", "#f0aa18", mode="lines+markers", line=dict(width=8, color="#f0aa18"), marker=dict(size=8, opacity=1)))
        group.append((len(figure.data) - 1, True))
        groups.append(group)
    figure.add_trace(scatter(reference, columns, "Oryginalne DMU (rzut wejść)", "#aab4bf", visible=True))
    reference_index = len(figure.data) - 1
    figure.add_trace(scatter(states, columns, "Wybrana ścieżka", "#17324a", mode="lines+markers+text",
                            text=["Start", *map(str, range(1, len(states)))], textposition="top center",
                            line=dict(width=4, color="#17324a"), marker=dict(size=6, opacity=1)))
    path_index = len(figure.data) - 1
    buttons = []
    for stage, group in enumerate(groups, start=1):
        visibility = [False] * len(figure.data)
        visibility[reference_index] = True
        visibility[path_index] = True
        for index, shown in group:
            visibility[index] = shown
        buttons.append(dict(label=f"Etap {stage}", method="update", args=[dict(visible=visibility), dict(title=f"{LABELS[method]} · etap {stage}")]))
    if buttons:
        for trace, visible in zip(figure.data, buttons[0]["args"][0]["visible"]):
            trace.visible = visible
    figure.update_layout(**axis_layout(columns, ranges), title=f"{LABELS[method]} · etap 1",
                         updatemenus=[dict(buttons=buttons, direction="right", type="buttons", x=0, y=1.12)])
    limits = score_limits(pd.read_csv(run/'fictive_candidate_metrics.csv')['best_efficiency'])
    figure = go.Figure(style_search_figure(figure.to_plotly_json(),limits))
    return figure, states, audits


def build_report(root, input_csv, target, output, plotly_mode="embedded"):
    from metric_path_explorer import build_metric_explorer, explorer_assets
    reference = pd.read_csv(input_csv)
    snapshot_dir = root / "dea_input_snapshot"
    saved = pd.read_csv(snapshot_dir / "decision_data.csv")
    pd.testing.assert_frame_equal(reference, saved, check_dtype=False)
    snapshot = load_dea_snapshot(snapshot_dir, reference)
    results = pd.read_csv(snapshot_dir / "reference_dmu_results.csv")
    reference = reference.merge(results[["name", "best_efficiency", "worst_efficiency", "best_rank", "worst_rank", "score_width"]], on="name", validate="one_to_one")
    params_frame = pd.read_csv(root / "experiment_params.csv", keep_default_na=False)
    params = dict(zip(params_frame["parameter"], params_frame["value"]))
    source_root = Path(__file__).parent
    report_sources = [p for p in source_root.glob("*.py") if not p.name.startswith("test_")]
    report_sources += list((source_root / "report_assets").glob("metric_calculations.*"))
    pd.DataFrame([
        {"file": str(path.relative_to(source_root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(report_sources)
    ]).to_csv(root / "report_source_manifest.csv", index=False)
    pd.DataFrame([("python", platform.python_version()), *[(package, version(package)) for package in ("pandas", "numpy", "networkx", "plotly")]], columns=["package", "version"]).to_csv(root / "python_environment.csv", index=False)
    columns = str(params["dimensions"]).split(",")
    baseline = results.set_index("name").loc[target]
    selected = {}
    explanations = {}
    summary, audits, choices, sections, coverage_frames, explorer_choices = [], [], [], [], [], []
    ranges = {column: [float(reference[column].min()) * .90, float(reference[column].max()) * 1.05] for column in columns}
    for method, _, color in METHODS:
        run = _latest_run(root / method)
        try:
            metric, path, count, repr_count = _selected_path(run)
        except (pd.errors.EmptyDataError, ValueError, IndexError):
            summary.append(dict(method=LABELS[method], path_records=0, distinct_state_paths=0, status="Brak kompletnej ścieżki w ocenionej puli"))
            sections.append(f"<section><h2>{LABELS[method]}</h2><p>Brak kompletnej ścieżki w ocenionej puli. Wynik nie dowodzi nieosiągalności celu w całej przestrzeni.</p></section>")
            continue
        selected[method] = (metric, path, color)
        method_paths = pd.read_csv(run / "paths.csv")
        state_columns = [c for c in method_paths if c.startswith("stage_") and c.endswith("_name")]
        distinct_count = len(method_paths.drop_duplicates(state_columns))
        row = dict(method=LABELS[method], path_records=count, distinct_state_paths=distinct_count,
                   pareto_representatives=repr_count, selected_path=metric["path_id"])
        for key in ["tc", "msc", "md", "swbp", "cdir", "dr", "fw", "best_efficiency_final", "best_rank_final", "score_width_final", "rank_width_final"]:
            row[key] = metric.get(key)
        summary.append(row)
        fig, states, stage_audits = method_figure(run, method, path, reference, columns, ranges, params)
        if "peer_refs" in states:
            changes = [float("nan")]
            for previous, current in zip(states.peer_refs, states.peer_refs.iloc[1:]):
                a, b = _parse_ref_set(previous), _parse_ref_set(current)
                changes.append(float("nan") if a is None or b is None else 1 - len(a & b) / len(a | b) if a | b else 0.)
            states["peer_jaccard_distance"] = changes
        audits.extend(stage_audits)
        keep = ["stage", *columns, *snapshot["output_columns"], "best_efficiency", "worst_efficiency", "best_rank", "worst_rank", "score_width", "rank_width", "milestone_gap", "effort_from_previous", "peer_refs", "peer_theta", "peer_jaccard_distance"]
        milestone_columns = [c for c in states if c.startswith("milestone_") and c != "milestone_gap"]
        stage_table = states.reindex(columns=list(dict.fromkeys(keep + milestone_columns)))
        stage_table.to_csv(run / "selected_path_stages.csv", index=False)
        method_choices = pd.read_csv(run / "path_selection.csv")
        method_metrics = pd.read_csv(run / "path_metrics.csv")
        coverage = metric_coverage(method_metrics, method_paths)
        coverage.insert(0, "method", method)
        coverage.to_csv(run / "metric_status.csv", index=False)
        coverage_frames.append(coverage)
        lens = method_choices.merge(method_metrics, on="path_id", suffixes=("_selection", ""))
        lens.insert(0, "generator", LABELS[method])
        choices.append(lens)
        method_explanations = {
            "hasse_path": "Każdy etap ma osiągnąć przewagę necessary nad wszystkimi jednostkami wskazanej klasy na ścieżce Hassego. Tabela podaje klasę i jej członków.",
            "front_path": "Każdy etap ma osiągnąć przewagę necessary nad co najmniej jedną pełną klasą wskazanej warstwy rankingu NEC. Są to warstwy porządku NEC, odrębne od frontów efektywności.",
            "best_rank_path": "Etap spełnia best_rank ≤ milestone_target. Niższa ranga jest lepsza.",
            "best_efficiency_path": "Etap spełnia best_efficiency ≥ milestone_target. Wyższa efektywność jest lepsza.",
            "robustness_width_path": "Etap spełnia width ≤ milestone_target. Rodzaj szerokości: " + str(params.get("width_kind", "score")) + ". Dodatkowo nie wolno pogorszyć odpowiadającego jej głównego wskaźnika: najlepszej efektywności dla score width lub najlepszej rangi dla rank width.",
        }
        stage_pool = pd.read_csv(run / "stage_candidates.csv")
        if params.get('final_goal') == 'joint':
            method_explanations['hasse_path'] = 'Graf NEC wyznacza trasę. Etapy pośrednie wymagają POS nad pojedynczą DMU; ostatni cel obejmuje wszystkie maksymalne klasy NEC, niezależnie od gałęzi.'
            method_explanations[method] += ' Stan końcowy spełnia wspólny warunek wag: przy jednym nieujemnym wektorze wag CCR osiąga wynik co najmniej tak dobry jak wszystkie DMU z maksymalnych klas NEC jednocześnie. Wagi dobierane są dla danego stanu; poszczególne etapy mogą mieć różne wektory wag.'
            final_name = path[f"stage_{int(path.path_length):02d}_name"]
            goal_pool = pd.read_csv(run/'fictive_candidate_metrics.csv').set_index('name')
            if goal_pool.loc[final_name,'joint_top'] != True:
                raise ValueError('Selected path does not meet the common-weight final goal')
            row['joint_final_goal'] = True
        if metric.get("cdir_scope") == "evaluated_final_stage_pool":
            direct_pool = stage_pool.loc[stage_pool.stage.eq(stage_pool.stage.max()), ["name", "effort_from_start", *columns]].rename(columns={"effort_from_start": "direct_cost"})
        elif metric.get("cdir_scope") == "complete_path_endpoints":
            direct_pool = method_metrics[["path_id", "cdir_endpoint"]].rename(columns={"path_id": "name", "cdir_endpoint": "direct_cost"})
            for column in columns:
                endpoints = {row.path_id: row[f"stage_{int(row.path_length):02d}_{column}"] for _, row in method_paths.iterrows()}
                direct_pool[column] = direct_pool.name.map(endpoints)
        else:
            raise ValueError("Unknown Cdir scope for calculation details")
        explanations[method] = explain_metrics(metric, path, params, direct_pool, stage_pool)
        print('Building metric explorer: '+method,flush=True)
        explorer, selections = build_metric_explorer(run,method,metric,path,reference,columns,ranges,params)
        explorer_choices.append(selections)
        sections.append(f"<section id='{method}'><h2>{LABELS[method]}</h2><p>{html.escape(method_explanations[method])}</p>"
                        + explorer + "<details><summary>Kontrola minimalności domyślnej ścieżki</summary>" + table(pd.DataFrame(stage_audits)) + "</details></section>")
    overview = go.Figure()
    overview.add_trace(scatter(reference, columns, "Oryginalne DMU", "#aab4bf"))
    for method, (metric, path, color) in selected.items():
        states = state_frame(path)
        overview.add_trace(scatter(states, columns, LABELS[method], color, mode="lines+markers+text", text=["Start", *map(str, range(1, len(states)))], line=dict(color=color, width=5), marker=dict(size=6, opacity=1)))
    overview.update_layout(**axis_layout(columns, ranges), title="Pięć metod na wspólnych osiach")
    projections = ""
    if len(columns) == 3:
        pairs = [(columns[0], columns[1]), (columns[0], columns[2]), (columns[1], columns[2])]
        figure = make_subplots(rows=1, cols=3, subplot_titles=[f"{x} / {y}" for x, y in pairs])
        for index, pair in enumerate(pairs, start=1):
            for method, (_, path, color) in selected.items():
                figure.add_trace(scatter(state_frame(path), list(pair), LABELS[method], color, mode="lines+markers", showlegend=index == 1, legendgroup=method, line=dict(color=color)), row=1, col=index)
            figure.update_xaxes(title_text=pair[0], range=ranges[pair[0]], row=1, col=index)
            figure.update_yaxes(title_text=pair[1], range=ranges[pair[1]], scaleanchor="x" if index == 1 else f"x{index}", scaleratio=1, row=1, col=index)
        projections = "<section><h2>Rzuty 2D wybranych ścieżek</h2><p>Dominacja jest liczona we wszystkich trzech zmienianych inputach. Sam rzut 2D nie wystarcza do jej oceny.</p>" + plot(figure, "projections") + "</section>"
    hasse, count = hasse_figure(snapshot, target)
    matrices = ""
    for key, label in [("necessary", "NEC: przewaga konieczna"), ("possible", "POS: przewaga możliwa")]:
        frame = pd.read_csv(snapshot_dir / f"{key}_matrix.csv", index_col=0)
        fig = go.Figure(go.Heatmap(z=frame.values, x=frame.columns, y=frame.index, zmin=0, zmax=1,
                                 colorscale=[[0, "#f3f6f9"], [1, "#3978b5"]], showscale=False,
                                 hovertemplate="Wiersz %{y} nad kolumną %{x}: %{z}<extra></extra>"))
        fig.update_layout(title=label, yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-45))
        matrices += "<details><summary>" + label + "</summary>" + plot(fig, key + "-matrix") + table(frame.rename_axis("source / target").reset_index()) + "</details>"
    validation = pd.read_csv(root / "validation_checks.csv")
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(root / "selected_transition_audit.csv", index=False)
    if not audit_frame.empty and (not audit_frame["selected_on_front"].all() or audit_frame["dominating_points"].gt(0).any()):
        raise ValueError("Selected transition failed independent front reconstruction")
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(root / "selected_path_summary.csv", index=False)
    full_metrics = selected_metric_table(selected, LABELS)
    full_metrics.to_csv(root / "selected_path_all_metrics.csv", index=False)
    (root / "metric_calculation_details.json").write_text(json.dumps(explanations, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame(columns=["method", "metric", "status"])
    coverage_summary = coverage.groupby(["method", "metric", "status"]).size().reset_index(name="path_count")
    coverage_summary.to_csv(root / "metric_coverage.csv", index=False)
    exceptions = coverage.loc[~coverage.status.eq("computed")]
    exceptions.to_csv(root / "metric_exceptions.csv", index=False)
    status = "PASS" if validation["passed"].all() else "Wymaga przeglądu: sprawdź tabelę walidacji"
    script = f"<script>{get_plotlyjs()}</script>" if plotly_mode == "embedded" else "<script src='https://cdn.plot.ly/plotly-3.3.1.min.js'></script>"
    choice_frame = pd.concat(choices, ignore_index=True) if choices else pd.DataFrame()
    choice_columns = [c for c in ["generator", "selection_type", "selection_detail", "path_id", "tc", "msc", "md", "swbp", "fw"] if c in choice_frame]
    fixed = ", ".join(f"{column}={baseline[column]}" for column in str(params["fixed_columns"]).split(",") if column)
    report = """<!doctype html><html lang='pl'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>DEA: porównanie pięciu metod</title>""" + script + """<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#f3f6f9;color:#17324a;margin:0}main{max-width:1450px;margin:auto;padding:24px}section{background:white;border:1px solid #dce4eb;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:30px}h2{font-size:23px}p{line-height:1.55}.table-wrap{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #e0e6ed;text-align:right;white-space:nowrap}th{background:#f1f5f8;position:sticky;top:0}td:first-child,th:first-child{text-align:left}details{margin:15px 0}summary{cursor:pointer;font-weight:600}nav a{display:inline-block;padding:9px}code{font-family:Consolas,monospace}.note{border-left:4px solid #3978b5;padding-left:14px}@media print{body{background:white}section{break-inside:avoid}.table-wrap{max-height:none;overflow:visible}nav{display:none}}
</style></head><body><main>"""
    report += f"<h1>EDU: pięć metod poprawy dla {html.escape(target)}</h1><p>{len(reference)} jednostek referencyjnych. Zmieniane inputy: {html.escape(', '.join(columns))}. Stałe: {html.escape(fixed)}. Walidacja: <b>{status}</b>.</p>"
    report += "<nav>" + "".join(f"<a href='#{m}'>{LABELS[m]}</a>" for m, _, _ in METHODS) + "</nav>"
    report += "<section><h2>Jak czytać wyniki</h2><p><b>TC</b>: suma kosztów kroków; <b>MSC</b>: największy koszt kroku; <b>MD</b>: odchylenie od kamieni milowych; <b>SWBP</b>: kara za nierównomierny rozkład wysiłku; <b>FW</b>: końcowa szerokość oceny właściwa metodzie. Szczegółowe metryki są dostępne przy każdej ścieżce.</p><p>Koszt jest średnią z redukcji zmienianych inputów podzielonych przez ich stałe zakresy w danych referencyjnych. Koszt i minimalność każdego kroku liczymy od faktycznego poprzednika.</p><p>Wyświetlamy wybór leksykograficzny: metody 1–2 minimalizują kolejno MSC, TC, SWBP, FW; metody 3–5: MD, MSC, SWBP, FW. Tabela wyborów pokazuje także minimum TC i wariant ważony.</p><p class='note'>Metody realizują różne cele. Tańsza ścieżka innej metody nie musi realizować tego samego zadania. Wspólna siatka bazowa jest identyczna; metody 3–5 korzystają dodatkowo z refinementu i prób lokalnych, a metoda 4 z próbkowania globalnego. To porównanie zachowania generatorów, nie test na identycznych końcowych pulach kandydatów.</p></section>"
    glossary = pd.read_csv(Path(__file__).resolve().parents[1] / "templates/path_metric_descriptions.csv")
    report += "<details><summary>Pełny słownik metryk</summary>" + table(glossary) + "</details>"
    report += "<section><h2>1. Wejście i bazowe wyniki DEA</h2><p>Scores, rangi i relacje oblicza Java. Fronty efektywności powstają przez ponowne obliczanie scores po usunięciu jednostek efektywnych. Ranking NEC i diagram Hassego wynikają z relacji koniecznej.</p>" + table(results.drop(columns=[c for c in results if c.endswith("_refs")]))
    report += "<details><summary>Pełna macierz decyzyjna X/Y</summary>" + table(saved) + "</details>"
    report += "<details open><summary>Fronty efektywności: peeling</summary><p>F1: best_efficiency ≥ 1 − 10⁻⁹; następnie usuwamy te jednostki i przeliczamy pozostałe. Scores z kolejnych iteracji odnoszą się do pomniejszonego zbioru.</p>" + table(pd.read_csv(snapshot_dir / "efficiency_fronts.csv")) + "</details>"
    report += f"<h3>Ranking NEC i Hasse</h3><p>Liczba ścieżek od {target} do maksymalnych klas NEC: <b>{count}</b>. Kierunek poprawy: od dołu ku górze. Nie każda ścieżka strukturalna musi mieć realizację w ocenionej puli punktów.</p>" + plot(hasse, "hasse")
    report += "<details><summary>Klasy, poziomy i krawędzie NEC</summary>" + table(pd.read_csv(snapshot_dir / "necessary_components.csv")) + table(pd.read_csv(snapshot_dir / "necessary_cover_edges.csv")) + "</details><p>Macierze: wartość 1 oznacza, że jednostka w wierszu jest preferowana nad jednostką w kolumnie.</p>" + matrices + "</section>"
    report += "<section><h2>2. Porównanie ścieżek</h2><p>Wykres 3D można obracać, przybliżać i przesuwać. Kliknięcie legendy włącza lub wyłącza warstwę. Oryginalne DMU mają różne outputy; służą jako kontekst, a kandydaci zachowują output targetu.</p><p><code>path_records</code> to liczba rekordów algorytmu; <code>distinct_state_paths</code> to liczba różnych sekwencji nazw stanów. Różne cele strukturalne mogą prowadzić przez te same punkty, dlatego liczby te mogą się różnić. Reprezentanci Pareto grupują natomiast równoważne wektory metryk.</p>" + plot(overview, "overview") + table(summary_frame) + "</section>" + projections
    report += "<section id='all-metrics'><h2>Pełny zestaw metryk wybranych ścieżek</h2><p>Wszystkie 20 pozycji katalogu: metryki artykułu oraz pomocnicze Cdir_endpoint i RC. Nazwa, znaczenie i kierunek oceny są podane przed wynikami. <code>lower_better</code>: mniej = lepiej; <code>higher_better</code>: więcej = lepiej.</p><p>PYV/PYM opisują poprawę najlepszej rangi w metodzie 3, a najlepszej efektywności w pozostałych metodach (w metodzie 5 zależnie od width_kind). APW/FW/WW używają odpowiednio rank width albo score width. Wartości o różnych skalach nie są bezpośrednio porównywalne. Wagi etapów i stanów są równe w tym eksperymencie; WW może więc być równe APW.</p><p><b>Najedź na wynik lub przejdź do niego klawiszem Tab, aby zobaczyć wzór i podstawione wartości. Kliknięcie przypina okienko; Esc je zamyka.</b></p><div class='metric-table'>" + render_metric_table(selected, LABELS, explanations) + "</div>"
    report += "<h3>PC: ciągłość zbiorów benchmarków</h3><p>Zgodnie z przyjętą konwencją P(z) jest sumą zbiorów referencyjnych ze wszystkich radialnie optymalnych rozwiązań input-oriented CCR, na stałym zbiorze oryginalnych DMU. Minimalizujemy θ przy Xλ ≤ θx(z), Yλ ≥ y(z), λ ≥ 0; następnie ustalamy θ = θ* i osobno maksymalizujemy każdą λ_j. Peer należy do P(z), gdy max λ_j &gt; 10⁻⁷. Nie wykonujemy drugiej fazy optymalizacji slacków. PC jest średnią odległości Jaccarda kolejnych P(z), ze wszystkimi przejściami od startu.</p><p>Tabele etapów pokazują peer_refs, peer_theta i peer_jaccard_distance. Średnia ostatniej kolumny to PC. peer_theta odnosi się do stałej technologii i może przekroczyć 1 dla punktu poza nią; nie zastępuje robust best_efficiency. RC mierzy ciągłość zbiorów oryginalnych DMU, nad którymi kandydat ma przewagę necessary. Obliczenie obejmuje wszystkie przejścia od startu oraz znane puste zbiory.</p>"
    report += "<details><summary>Kompletność metryk wszystkich zapisanych ścieżek</summary><p><code>computed</code>: obliczono; <code>undefined_zero_step_cost</code>: PYV/PYM są niezdefiniowane, ponieważ co najmniej jeden koszt kroku jest zerowy; <code>undefined_zero_direct_cost</code>: DR ma zerowy mianownik; <code>missing_required_data</code>: brakuje danych. Żadnego z tych przypadków nie zastępujemy zerem.</p>" + table(coverage_summary) + "</details>"
    if not exceptions.empty:
        report += "<details><summary>Ścieżki z niezdefiniowanymi lub brakującymi metrykami</summary>" + table(exceptions) + "</details>"
    report += "</section>"
    report += "<section><h2>3. Punkty i etapy</h2><p>Przełącznik etapu pokazuje tylko badania dla tego etapu i front względem jego poprzednika. Linie prowadzą przez wybrane punkty. Warstwy diagnostyczne można włączyć z legendy; ich kolor koduje best_efficiency w skali Viridis 0–1. Hover podaje dokładne scores i rangi.</p><p>Front minimalnych zmian jest odrębny od frontu efektywności DEA i od warstw rankingu NEC. Próbki diagnostyczne nie są automatycznie kandydatami ścieżki.</p></section>" + "".join(sections)
    report += "<section><h2>4. Wybory według różnych kryteriów</h2>" + table(choice_frame[choice_columns]) + "</section>"
    assets = Path(__file__).parent / "report_assets"
    report += "<style>" + (assets / "metric_calculations.css").read_text(encoding="utf-8") + "</style>"
    report += '<aside id="metric-calculation-popup" role="dialog" aria-label="Sposób obliczenia metryki" hidden><div class="calculation-toolbar"><span class="calculation-hint"></span><button type="button" class="calculation-close" aria-label="Zamknij obliczenia">×</button></div><div class="calculation-content" tabindex="0"></div></aside>'
    report += "<script>" + (assets / "metric_calculations.js").read_text(encoding="utf-8") + "</script>"
    completion_file = root / "metric_completion.json"
    if completion_file.exists():
        completion = json.loads(completion_file.read_text(encoding="utf-8"))
        completion_table = pd.DataFrame([(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
                                         for key, value in completion.items()], columns=["parameter", "value"])
        report += "<details><summary>Przeliczenie metryk: konwencja, czas i pochodzenie danych</summary>" + table(completion_table) + "</details>"
    report += "<section><h2>5. Parametry i weryfikacja</h2><details><summary>Wszystkie parametry eksperymentu</summary>" + table(params_frame) + "</details><details><summary>Czas i status uruchomień</summary>" + table(pd.read_csv(root / "execution_status.csv").drop(columns="command_json")) + "</details><details><summary>Kontrole spójności pięciu metod</summary>" + table(validation) + "</details><p>Minimalność dotyczy ocenionej puli. Limity przejść i skończone próbkowanie mogą pominąć inne rozwiązania. Cdir jest minimum w zapisanym zakresie cdir_scope, a DR = TC/Cdir pozostaje aproksymacją. Konwencja PC jest jawnie podana wyżej; RC nie jest jej zamiennikiem.</p></section><style>.metric-table td:nth-child(2){white-space:normal;min-width:250px;max-width:380px;text-align:left}</style><script>document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', () => {if(d.open) d.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));}));</script></main></body></html>"
    if params.get('final_goal') == 'joint':
        from generate_hasse_pos_report import dataset_section
        report = report.replace('Metody realizują różne cele. Tańsza ścieżka innej metody nie musi realizować tego samego zadania. ', '')
        report = report.replace('Wspólna siatka bazowa jest identyczna; metody 3–5 korzystają dodatkowo z refinementu i prób lokalnych, a metoda 4 z próbkowania globalnego. To porównanie zachowania generatorów, nie test na identycznych końcowych pulach kandydatów.',
            'Pięć metod korzysta z tej samej ocenionej puli kandydatów i realizuje wspólny warunek końcowy. Metody różnią się celami etapów pośrednich oraz warunkami rangi, efektywności lub szerokości oceny. Koszty ścieżek porównujemy z uwzględnieniem tych wymagań.')
        note = "<section id='common-final-goal'><h2>Wspólny cel końcowy pięciu metod</h2><p>W każdej z pięciu metod stan końcowy musi osiągać wynik co najmniej tak dobry jak wszystkie DMU z maksymalnych klas grafu NEC jednocześnie, przy jednym wspólnym nieujemnym wektorze wag CCR. Wektor dobierany jest dla danego stanu końcowego; poszczególne etapy ścieżki mogą korzystać z różnych wektorów wag.</p><p>Etapy pośrednie wyznacza reguła danej metody: ścieżka Hassego wymaga POS nad wskazaną DMU, metoda warstw rankingu NEC wymaga przewagi necessary nad co najmniej jedną pełną klasą danej warstwy, a metody liczbowe realizują progi najlepszej rangi, najlepszej efektywności lub szerokości oceny. W metodach liczbowych stan końcowy spełnia zarówno wspólny warunek wag, jak i końcowy próg danej metody. Fronty efektywności wyznaczane przez iteracyjne usuwanie jednostek efektywnych stanowią osobny opis danych referencyjnych.</p><p>Wspólny warunek końcowy filtruje kandydatów przed wyznaczeniem frontu minimalnych zmian ostatniego przejścia. Wyniki obejmują kompletne, osiągalne ścieżki spełniające wszystkie warunki etapów; brak takiej ścieżki oznacza brak rozwiązania w ocenionej puli. Świadectwa wag zapisano we wspólnej puli źródłowej, a final_goal.json dokumentuje zbiór celów i pochodzenie danych.</p></section>"
        goal_source = Path(params['shared_pool'])
        witnesses = pd.read_csv(goal_source/'possible_goals.csv')
        endpoints = {method: str(path[f"stage_{int(path.path_length):02d}_name"])
                     for method, (_, path, _) in selected.items()}
        note += joint_weight_section(witnesses, endpoints, str(params['top_targets']).split('|'),
                                     snapshot['input_columns'], snapshot['output_columns'])
        report = report.replace('<section><h2>Jak czytać wyniki', dataset_section(saved,params)+note+'<section><h2>Jak czytać wyniki',1)
    if explorer_choices:
        pd.concat(explorer_choices,ignore_index=True).to_csv(root/'metric_explorer_selection.csv',index=False)
    report = report.replace('<section><h2>3. Punkty i etapy</h2>',
        '<section><h2>3. Wpływ metryki na wybór ścieżki</h2><p>Każda z pięciu metod ma osobny wybór jednej z 20 metryk. Wybór zmienia ścieżkę, fronty przejść, tabelę etapów oraz jej metryki z rachunkami. Zestawienie i wykres porównawczy powyżej zachowują domyślny wybór leksykograficzny. Wszystkie metryki policzono dla wszystkich zapisanych ścieżek w path_metrics.csv. Optimum w panelu dotyczy tej puli po przycinaniu frontów i limitach, nie wszystkich możliwych ścieżek w przestrzeni ciągłej. Brakujące i nieskończone wyniki są wyłączone z optymalizacji danej metryki. Remisy: tolerancja 1e-9, potem TC, MSC i ID. Pliki metric_explorer_selection.csv oraz metric_explorer_ties.csv zapisują wybory i wszystkie równorzędne ID w katalogach metod.</p>')
    report = report.replace('</main></body>',explorer_assets()+'</main></body>')
    from report_plot_style import describe_search_style
    report = describe_search_style(report)
    report = number_report_sections(report,LABELS.values())
    output.write_text(report, encoding="utf-8")
    def markdown(frame):
        columns = list(frame.columns)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        lines += ["| " + " | ".join("—" if pd.isna(v) else f"{v:.5g}" if isinstance(v, float) else str(v).replace("|", "/") for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
        return "\n".join(lines)
    (root / "README_results.md").write_text(f"# EDU: {target}, pięć metod\n\nZmieniane: {', '.join(columns)}. Stałe: {fixed}. Walidacja: {status}.\n\nOtwórz `five_method_report.html`: dane i Plotly są osadzone w pliku.\n\n## Porównanie\n\n" + markdown(summary_frame) + "\n\n## Założenia\n\nMetoda 2 używa warstw NEC, a fronty efektywności są osobnym wynikiem peelingu. Minimalność dotyczy przebadanej puli. Wszystkie koszty są krokowe.\n\n## Parametry\n\n" + markdown(params_frame), encoding="utf-8")
    with (root / "README_results.md").open("a", encoding="utf-8") as handle:
        handle.write("\n\n## Wszystkie metryki\n\n" + markdown(full_metrics) + "\n\nPC: union peer sets wszystkich radialnie optymalnych rozwiazan input-oriented CCR na stalym zbiorze oryginalnych DMU; prog lambda > 1e-7, bez drugiej fazy slackow. Szczegoly i zbiory etapow: HTML oraz peer_evaluation_results.csv. RC obejmuje wszystkie przejscia.\n\nStatusy wszystkich sciezek: metric_coverage.csv; wyjatki: metric_exceptions.csv. Niezdefiniowane PYV/PYM dla zerowego kosztu kroku nie sa zastepowane zerem.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plotly-mode", choices=["embedded", "cdn"], default="embedded")
    args = parser.parse_args()
    build_report(args.root.resolve(), args.input.resolve(), args.target, args.output.resolve(), args.plotly_mode)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
