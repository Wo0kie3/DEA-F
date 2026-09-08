from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

from minimality_certificate import find_selected_path
from visualize_minimality_2d import load_visualization_data


PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-3.3.1.min.js"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one self-contained interactive comparison of 2D DEA paths."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--x-label", default="Wejście i1")
    parser.add_argument("--y-label", default="Wejście i2")
    parser.add_argument(
        "--independent-axis-scales",
        action="store_true",
        help="Use separate padded ranges when the two inputs have different units or scales.",
    )
    parser.add_argument(
        "--normalize-display",
        action="store_true",
        help=(
            "Display both minimized inputs on a 0-100%% scale, where 0%% is the "
            "target start and 100%% is the smallest evaluated value per axis."
        ),
    )
    parser.add_argument(
        "--dea-snapshot-dir",
        type=Path,
        default=None,
        help="Directory produced by generate_dea_input_snapshot.py; defaults to ROOT/dea_input_snapshot.",
    )
    parser.add_argument(
        "--plotly-mode",
        choices=["embedded", "cdn"],
        default="embedded",
        help="Embed Plotly for a self-contained output or use the CDN for a small repo example.",
    )
    return parser.parse_args()


def round_value(value, digits=6):
    return round(float(value), digits)


def load_dea_snapshot(snapshot_dir: Path, input_frame: pd.DataFrame) -> dict:
    required = {
        "results": "reference_dmu_results.csv",
        "necessary": "necessary_matrix.csv",
        "possible": "possible_matrix.csv",
        "efficiency_fronts": "efficiency_fronts.csv",
        "components": "necessary_components.csv",
        "ranking": "necessary_ranking.csv",
        "edges": "necessary_cover_edges.csv",
    }
    missing = [filename for filename in required.values() if not (snapshot_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete DEA snapshot in {snapshot_dir}. Missing: {', '.join(missing)}. "
            "Run python/generate_dea_input_snapshot.py first."
        )

    results = pd.read_csv(snapshot_dir / required["results"])
    necessary = pd.read_csv(snapshot_dir / required["necessary"], index_col=0)
    possible = pd.read_csv(snapshot_dir / required["possible"], index_col=0)
    efficiency_fronts = pd.read_csv(snapshot_dir / required["efficiency_fronts"])
    components = pd.read_csv(snapshot_dir / required["components"])
    ranking = pd.read_csv(snapshot_dir / required["ranking"])
    edges = pd.read_csv(snapshot_dir / required["edges"])

    expected_names = input_frame["name"].astype(str).tolist()
    result_names = results["name"].astype(str).tolist()
    if result_names != expected_names:
        raise ValueError("DEA snapshot DMU order does not match the input CSV.")
    for table_name, frame in (
        ("efficiency_fronts", efficiency_fronts),
        ("necessary_ranking", ranking),
    ):
        if frame["name"].astype(str).tolist() != expected_names:
            raise ValueError(f"{table_name} does not match the input CSV DMUs.")
    for matrix_name, matrix in (("necessary", necessary), ("possible", possible)):
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        if matrix.index.tolist() != expected_names or matrix.columns.tolist() != expected_names:
            raise ValueError(f"{matrix_name} matrix does not match the input CSV DMUs.")

    input_columns = sorted(
        [column for column in results.columns if column.startswith("i") and column[1:].isdigit()],
        key=lambda column: int(column[1:]),
    )
    output_columns = sorted(
        [column for column in results.columns if column.startswith("o") and column[1:].isdigit()],
        key=lambda column: int(column[1:]),
    )
    numeric_columns = [
        *input_columns,
        *output_columns,
        "best_efficiency",
        "worst_efficiency",
        "score_width",
        "best_rank",
        "worst_rank",
        "rank_width",
        "efficiency_front",
        "necessary_rank",
        "necessary_over_count",
        "necessary_under_count",
    ]
    display_results = results[
        [
            "name",
            *numeric_columns,
            "necessary_component",
            "necessary_over_refs",
            "necessary_under_refs",
        ]
    ].copy()
    for column in numeric_columns:
        display_results[column] = pd.to_numeric(display_results[column], errors="raise").round(6)
    for column in ["necessary_component", "necessary_over_refs", "necessary_under_refs"]:
        display_results[column] = display_results[column].fillna("").astype(str)

    return {
        "names": expected_names,
        "input_columns": input_columns,
        "output_columns": output_columns,
        "results": display_results.to_dict("records"),
        "necessary_matrix": necessary.astype(int).values.tolist(),
        "possible_matrix": possible.astype(int).values.tolist(),
        "components": components.to_dict("records"),
        "efficiency_fronts": efficiency_fronts.to_dict("records"),
        "necessary_ranking": ranking.to_dict("records"),
        "edges": edges.fillna("").to_dict("records"),
        "orientation": "Wiersz/source jest preferowany nad kolumną/target.",
    }


def load_target(root: Path, target: str, input_frame: pd.DataFrame) -> dict:
    experiment_dir = root / target
    data = load_visualization_data(
        experiment_dir=experiment_dir,
        columns=["i1", "i2"],
        path_id=None,
        tolerance=1e-9,
    )
    metrics = pd.read_csv(experiment_dir / "all_path_metrics.csv")
    selected_metric, _ = find_selected_path(metrics, requested_path_id=None)
    certificate = pd.read_csv(experiment_dir / "minimality_certificate.csv")

    if int(selected_metric["attainable_transition_violations"]) != 0:
        raise ValueError(f"Selected path for {target} has unattainable transitions.")
    if not certificate["pareto_minimal"].astype(bool).all():
        raise ValueError(f"Selected path for {target} contains a dominated stage.")
    if not (pd.to_numeric(certificate["dominating_points"]) == 0).all():
        raise ValueError(f"Selected path for {target} has a dominating candidate.")

    effort_by_stage = {
        int(row["stage"]): round_value(row["effort_from_previous"])
        for _, row in certificate.iterrows()
    }
    path = []
    for point in data["path"]:
        stage = int(point["stage"])
        path.append(
            {
                "stage": stage,
                "label": "Start" if stage == 0 else f"Etap {stage}",
                "name": target if stage == 0 else f"punkt etapu {stage}",
                "x": round_value(point["x"]),
                "y": round_value(point["y"]),
                "efficiency": round_value(point["best_efficiency"]),
                "effort": None if stage == 0 else effort_by_stage[stage],
                "pareto": None if stage == 0 else True,
            }
        )

    fronts = {}
    candidates = {}
    for stage in range(1, 4):
        stage_points = [
            point for point in data["points"] if int(point["stage"]) == stage
        ]
        fronts[str(stage)] = [
            {
                "x": round_value(point["x"]),
                "y": round_value(point["y"]),
                "efficiency": round_value(point["best_efficiency"]),
            }
            for point in stage_points
            if bool(point["front"])
        ]
        candidates[str(stage)] = [
            {
                "name": point["name"],
                "x": round_value(point["x"]),
                "y": round_value(point["y"]),
                "best_efficiency": round_value(point["best_efficiency"]),
                "worst_efficiency": round_value(point["worst_efficiency"]),
                "score_width": round_value(point["score_width"]),
                "best_rank": int(point["best_rank"]),
                "worst_rank": int(point["worst_rank"]),
                "source": point["sampling_source"],
                "status": point["status"],
                "in_selection_pool": bool(point["in_selection_pool"]),
                "target_eligible": bool(point["target_eligible"]),
                "attainable": bool(point["attainable"]),
                "front": bool(point["front"]),
            }
            for point in stage_points
        ]

    raw_tc = float(selected_metric["tc"])
    tc = round_value(raw_tc)
    msc = round_value(selected_metric["msc"])
    cdir = round_value(selected_metric["cdir"])
    tc_ties = metrics[(pd.to_numeric(metrics["tc"]) - raw_tc).abs() <= 1e-9]
    return {
        "target": target,
        "path_id": str(selected_metric["path_id"]),
        "path_count": int(len(metrics)),
        "tc_ties": int(len(tc_ties)),
        "fixed_outputs": {
            column: round_value(
                input_frame.loc[input_frame["name"].astype(str) == target, column].iloc[0]
            )
            for column in input_frame.columns
            if column.startswith("o") and column[1:].isdigit()
        },
        "metrics": {
            "tc": tc,
            "msc": msc,
            "cdir": cdir,
            "dr": round_value(selected_metric["dr"]),
        },
        "path": path,
        "fronts": fronts,
        "candidates": candidates,
        "stage_summary": {
            str(int(row["stage"])): {
                "threshold": round_value(row["threshold"]),
                "evaluated": int(row["evaluated"]),
                "selection_pool": int(row["tested"]),
                "eligible": int(row["eligible"]),
                "front": int(row["front"]),
                "below_threshold": sum(
                    point["status"] == "below_threshold"
                    for point in candidates[str(int(row["stage"]))]
                ),
                "probes": sum(
                    point["status"] == "probe"
                    for point in candidates[str(int(row["stage"]))]
                ),
            }
            for row in data["stages"]
        },
        "front_counts": {
            str(int(row["stage"])): int(row["pareto_front_points"])
            for _, row in certificate.iterrows()
        },
    }


def build_html(
    payload: dict,
    output: Path,
    plotly_mode: str = "embedded",
    x_label: str = "Wejście i1",
    y_label: str = "Wejście i2",
    independent_axis_scales: bool = False,
    normalize_display: bool = False,
):
    if plotly_mode == "embedded":
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    elif plotly_mode == "cdn":
        plotly_tag = f'<script src="{PLOTLY_CDN_URL}" charset="utf-8"></script>'
    else:
        raise ValueError(f"Unsupported Plotly mode: {plotly_mode}")
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    input_labels = ", ".join(payload["dea"]["input_columns"])
    output_labels = ", ".join(payload["dea"]["output_columns"])
    x_values = [float(row["i1"]) for row in payload["dmus"]]
    y_values = [float(row["i2"]) for row in payload["dmus"]]
    for unit in payload["units"].values():
        x_values.extend(float(point["x"]) for point in unit["path"])
        y_values.extend(float(point["y"]) for point in unit["path"])
        for points in unit["fronts"].values():
            x_values.extend(float(point["x"]) for point in points)
            y_values.extend(float(point["y"]) for point in points)
        for points in unit["candidates"].values():
            x_values.extend(float(point["x"]) for point in points)
            y_values.extend(float(point["y"]) for point in points)

    def padded_range(values: list[float]) -> list[float]:
        low, high = min(values), max(values)
        padding = max((high - low) * 0.05, 1e-6)
        return [round(low - padding, 8), round(high + padding, 8)]

    x_range = padded_range(x_values)
    y_range = padded_range(y_values)
    if not independent_axis_scales:
        common_low = min(x_range[0], y_range[0])
        common_high = max(x_range[1], y_range[1])
        x_range = [common_low, common_high]
        y_range = [common_low, common_high]
    display_x_label = (
        f"Poziom redukcji w badanym zakresie: {x_label} (%)"
        if normalize_display
        else x_label
    )
    display_y_label = (
        f"Poziom redukcji w badanym zakresie: {y_label} (%)"
        if normalize_display
        else y_label
    )
    display_x_range = [-5, 105] if normalize_display else x_range
    display_y_range = [-5, 105] if normalize_display else y_range
    normalization_note = (
        " Osie są pokazane w skali względnej: 0% oznacza punkt startowy, a 100% "
        "najmniejszą wartość faktycznie ocenioną w danym inpucie. Oryginalne wartości "
        "pozostają w hoverze i tabeli."
        if normalize_display
        else ""
    )
    default_target = payload["order"][0]
    target_buttons = "\n".join(
        f'<button type="button" class="unit-button{" active" if target == default_target else ""}" '
        f'data-target="{html.escape(target)}" aria-pressed="{str(target == default_target).lower()}">{html.escape(target)}</button>'
        for target in payload["order"]
    )
    report_title = (
        f"Minimalna ścieżka DEA - {default_target}"
        if len(payload["order"]) == 1
        else f"Minimalne ścieżki DEA - porównanie {len(payload['order'])} jednostek"
    )
    document = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17324a;
      --muted: #607284;
      --line: #d7e0e7;
      --soft: #f5f8fa;
      --accent: #2878b5;
      --stage-1: #df5a56;
      --stage-2: #3978b5;
      --stage-3: #43a06b;
      --start: #f0aa18;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 36px;
    }}
    header {{
      display: flex;
      gap: 16px;
      align-items: end;
      justify-content: space-between;
      border-bottom: 2px solid var(--accent);
      padding-bottom: 12px;
      margin-bottom: 14px;
    }}
    h1 {{ margin: 0; font-size: clamp(22px, 3vw, 34px); font-weight: 600; }}
    .subtitle {{ margin: 5px 0 0; color: var(--muted); }}
    .unit-controls {{ display: flex; flex-wrap: wrap; gap: 7px; justify-content: flex-end; }}
    .unit-button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 14px;
      background: #ffffff;
      color: var(--ink);
      cursor: pointer;
      font: inherit;
    }}
    .unit-button.active {{ background: var(--ink); color: #ffffff; border-color: var(--ink); }}
    .plot-wrap {{ width: 100%; min-height: 610px; }}
    #plot {{ width: 100%; height: 610px; }}
    h2 {{ margin: 28px 0 8px; font-size: 22px; }}
    h3 {{ margin: 20px 0 8px; font-size: 17px; }}
    .section-note {{ margin: 0 0 12px; color: var(--muted); }}
    .dea-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
      margin: 12px 0 18px;
    }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .metric strong {{ font-size: 17px; }}
    details {{ border-top: 1px solid var(--line); padding: 10px 0; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .relation-controls {{ display: flex; gap: 12px; align-items: center; margin: 12px 0; }}
    select {{ border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px; background: #ffffff; color: var(--ink); }}
    .matrix-table {{ width: max-content; min-width: 100%; font-size: 12px; }}
    .matrix-table th, .matrix-table td {{ min-width: 38px; padding: 6px; text-align: center !important; }}
    .matrix-table td.relation-true {{ background: #dcefe5; font-weight: 600; }}
    .matrix-table .selected-axis {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
    .selected-dmu {{ background: #fff2cc !important; font-weight: 600; }}
    .structure-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    #hasse-plot {{ width: 100%; height: 620px; }}
    .result-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 9px 22px;
      align-items: baseline;
      margin: 8px 0 12px;
      color: var(--muted);
    }}
    .result-line strong {{ color: var(--ink); }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; }}
    th {{ background: var(--ink); color: #ffffff; font-weight: 600; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tbody tr:nth-child(even) {{ background: var(--soft); }}
    .status {{ text-align: center; }}
    footer {{ margin-top: 14px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 18px, 1180px); margin-top: 12px; }}
      header {{ align-items: flex-start; flex-direction: column; }}
      .unit-controls {{ justify-content: flex-start; }}
      .dea-summary {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
      .structure-grid {{ grid-template-columns: 1fr; }}
      #plot {{ height: 520px; }}
      .plot-wrap {{ min-height: 520px; }}
      .table-wrap {{ overflow-x: auto; }}
      table {{ min-width: 700px; }}
    }}
    @media print {{
      .unit-controls, .modebar, .updatemenu-container, .slider-container {{ display: none !important; }}
      main {{ width: 100%; margin: 0; }}
      #plot {{ height: 560px; }}
    }}
  </style>
  {plotly_tag}
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Minimalne ścieżki poprawy efektywności DEA</h1>
        <p class="subtitle">Dwa zmieniane wejścia: {html.escape(input_labels)}; wyjścia {html.escape(output_labels)} pozostają stałe dla wybranej jednostki.</p>
      </div>
      <nav class="unit-controls" aria-label="Wybór jednostki">
        {target_buttons}
      </nav>
    </header>
    <section aria-labelledby="dea-input-heading">
      <h2 id="dea-input-heading">1. Dane wejściowe i bazowy wynik robust DEA</h2>
      <p class="section-note">Najpierw pokazujemy macierz decyzyjną X/Y, przedziały score i rangi oraz strukturę relacji necessary. Są to wyniki dla oryginalnych jednostek referencyjnych, przed wygenerowaniem punktów fictive.</p>
      <div id="dea-summary" class="dea-summary"></div>
      <div class="table-wrap">
        <table aria-label="Dane i bazowe wyniki robust DEA">
          <thead>
            <tr id="dea-results-head"></tr>
          </thead>
          <tbody id="dea-results-table"></tbody>
        </table>
      </div>
      <details>
        <summary>Macierze relacji necessary i possible</summary>
        <p class="section-note">Wartość 1 oznacza, że jednostka z wiersza jest preferowana nad jednostką z kolumny. Wybrana DMU jest zaznaczona w wierszu i kolumnie.</p>
        <div class="relation-controls">
          <label for="relation-kind">Relacja:</label>
          <select id="relation-kind">
            <option value="necessary">Necessary</option>
            <option value="possible">Possible</option>
          </select>
        </div>
        <div id="relation-matrix" class="table-wrap"></div>
      </details>
      <details open>
        <summary>Fronty efektywności DEA</summary>
        <p class="section-note">Front F1 zawiera DMU z best efficiency równym 1 w pełnym zbiorze. Po ich usunięciu Java ponownie oblicza efektywność pozostałych jednostek, tworząc kolejno F2, F3 itd.</p>
        <div class="table-wrap"><table><thead><tr><th>Front</th><th>Liczba DMU</th><th>Jednostki</th></tr></thead><tbody id="efficiency-front-table"></tbody></table></div>
      </details>
      <details open>
        <summary>Ranking necessary i diagram Hassego</summary>
        <p class="section-note">Warstwy rankingu NEC wynikają z relacji necessary i nie są frontami efektywności. Krawędź diagramu Hassego oznacza bezpośrednią relację pokrycia po redukcji przechodniej. Dla wybranej DMU wyróżnione są wszystkie krawędzie i węzły lepsze, do których prowadzi relacja necessary.</p>
        <h3>Diagram Hassego relacji necessary</h3>
        <div id="hasse-summary" class="result-line"></div>
        <div id="hasse-plot" aria-label="Diagram Hassego relacji necessary"></div>
        <div class="structure-grid">
          <div>
            <h3>Klasy i warstwy rankingu NEC</h3>
            <div class="table-wrap"><table><thead><tr><th>Poziom</th><th>Klasa</th><th>DMU</th></tr></thead><tbody id="component-table"></tbody></table></div>
          </div>
          <div>
            <h3>Krawędzie pokrycia Hassego</h3>
            <div class="table-wrap"><table><thead><tr><th>Klasa gorsza</th><th>Klasa lepsza</th></tr></thead><tbody id="edge-table"></tbody></table></div>
          </div>
        </div>
      </details>
    </section>
    <section aria-labelledby="path-heading">
      <h2 id="path-heading">2. Minimalna ścieżka poprawy</h2>
      <p class="section-note">Suwak etapu pokazuje wszystkie punkty faktycznie ocenione dla tego etapu. Kolor oznacza best efficiency, a wyróżniony front Pareto jest liczony osobno względem faktycznego poprzednika. Punkty z pośrednich iteracji refinementu są widoczne jako próby diagnostyczne, lecz nie są automatycznie częścią puli selekcyjnej.{html.escape(normalization_note)}</p>
      <div id="stage-detail" class="result-line"></div>
    </section>
    <section class="plot-wrap" aria-label="Interaktywny wykres frontów Pareto i minimalnej ścieżki">
      <div id="plot"></div>
    </section>
    <section aria-label="Minimalna ścieżka wybranej jednostki">
      <div id="result-line" class="result-line"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Etap</th><th>Punkt</th><th>i1</th><th>i2</th><th>Efektywność</th><th>Wysiłek kroku</th><th>Status</th></tr>
          </thead>
          <tbody id="path-table"></tbody>
        </table>
      </div>
    </section>
    <footer>Pełna konfiguracja znajduje się w experiment_params.csv. Wybór ścieżki: minimalne TC, następnie minimalne MSC i stabilny path_id.</footer>
  </main>
  <script>
    const DATA = {data_json};
    const plot = document.getElementById('plot');
    const tableBody = document.getElementById('path-table');
    const resultLine = document.getElementById('result-line');
    const deaSummary = document.getElementById('dea-summary');
    const deaResultsHead = document.getElementById('dea-results-head');
    const deaResultsTable = document.getElementById('dea-results-table');
    const relationKind = document.getElementById('relation-kind');
    const relationMatrix = document.getElementById('relation-matrix');
    const efficiencyFrontTable = document.getElementById('efficiency-front-table');
    const componentTable = document.getElementById('component-table');
    const edgeTable = document.getElementById('edge-table');
    const hassePlot = document.getElementById('hasse-plot');
    const hasseSummary = document.getElementById('hasse-summary');
    const stageDetail = document.getElementById('stage-detail');
    const colors = {{1: '#df5a56', 2: '#3978b5', 3: '#43a06b'}};
    const symbols = {{1: 'circle-open', 2: 'square-open', 3: 'diamond-open'}};
    const normalizeDisplay = {str(normalize_display).lower()};
    const candidateStatuses = [
      ['below_threshold', 'Poniżej progu', 'x', 6, 0.32],
      ['probe', 'Próba diagnostyczna', 'cross-thin', 7, 0.42],
      ['unattainable', 'Nieosiągalny z poprzednika', 'triangle-down', 8, 0.5],
      ['dominated', 'Zdominowany', 'circle', 7, 0.55],
      ['front', 'Front Pareto', null, 10, 0.95]
    ];

    function format(value, digits = 4) {{
      if (value === null || value === undefined) return '–';
      return Number(value).toFixed(digits).replace('.', ',');
    }}

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }}

    function displayScale(unit) {{
      const start = unit.path[0];
      const points = Object.values(unit.candidates).flat();
      const minimumX = Math.min(...points.map(point => point.x), ...unit.path.map(point => point.x));
      const minimumY = Math.min(...points.map(point => point.y), ...unit.path.map(point => point.y));
      return {{
        startX: start.x, startY: start.y,
        spanX: Math.max(start.x - minimumX, 1e-12),
        spanY: Math.max(start.y - minimumY, 1e-12)
      }};
    }}

    function displayPoint(point, scale) {{
      if (!normalizeDisplay) return {{x: point.x, y: point.y}};
      return {{
        x: 100 * (scale.startX - point.x) / scale.spanX,
        y: 100 * (scale.startY - point.y) / scale.spanY
      }};
    }}

    function renderDeaSummary(target) {{
      const row = DATA.dea.results.find(item => item.name === target);
      deaSummary.innerHTML = `
        <div class="metric"><span>Wybrana DMU</span><strong>${{escapeHtml(row.name)}}</strong></div>
        <div class="metric"><span>Front efektywności</span><strong>F${{row.efficiency_front}}</strong></div>
        <div class="metric"><span>Poziom rankingu NEC</span><strong>R${{row.necessary_rank}}</strong></div>
        <div class="metric"><span>Score robust</span><strong>${{format(row.worst_efficiency)}} – ${{format(row.best_efficiency)}}</strong></div>
        <div class="metric"><span>Szerokość score</span><strong>${{format(row.score_width)}}</strong></div>
        <div class="metric"><span>Ranga robust</span><strong>${{row.best_rank}} – ${{row.worst_rank}}</strong></div>
        <div class="metric"><span>Szerokość rangi</span><strong>${{row.rank_width}}</strong></div>
        <div class="metric"><span>NEC nad innymi</span><strong>${{row.necessary_over_count}}</strong><span>${{escapeHtml(row.necessary_over_refs || 'brak')}}</span></div>
        <div class="metric"><span>Inne NEC nad DMU</span><strong>${{row.necessary_under_count}}</strong><span>${{escapeHtml(row.necessary_under_refs || 'brak')}}</span></div>`;
    }}

    function renderDeaResults(target) {{
      deaResultsHead.innerHTML = [
        'DMU', ...DATA.dea.input_columns, ...DATA.dea.output_columns,
        'Best score', 'Worst score', 'Score width', 'Best rank', 'Worst rank', 'Rank width', 'Front efektywności', 'Poziom NEC'
      ].map(label => `<th>${{escapeHtml(label)}}</th>`).join('');
      deaResultsTable.innerHTML = DATA.dea.results.map(row => `
        <tr class="${{row.name === target ? 'selected-dmu' : ''}}">
          <td>${{escapeHtml(row.name)}}</td>
          ${{[...DATA.dea.input_columns, ...DATA.dea.output_columns].map(column => `<td>${{format(row[column])}}</td>`).join('')}}
          <td>${{format(row.best_efficiency)}}</td><td>${{format(row.worst_efficiency)}}</td><td>${{format(row.score_width)}}</td>
          <td>${{row.best_rank}}</td><td>${{row.worst_rank}}</td><td>${{row.rank_width}}</td><td>F${{row.efficiency_front}}</td><td>R${{row.necessary_rank}}</td>
        </tr>`).join('');
    }}

    function renderRelationMatrix(target) {{
      const matrix = relationKind.value === 'possible'
        ? DATA.dea.possible_matrix
        : DATA.dea.necessary_matrix;
      const header = DATA.dea.names.map(name =>
        `<th class="${{name === target ? 'selected-axis' : ''}}">${{escapeHtml(name)}}</th>`
      ).join('');
      const rows = DATA.dea.names.map((source, rowIndex) => {{
        const cells = DATA.dea.names.map((column, columnIndex) => {{
          const value = matrix[rowIndex][columnIndex];
          const classes = [
            value ? 'relation-true' : '',
            source === target || column === target ? 'selected-axis' : ''
          ].filter(Boolean).join(' ');
          return `<td class="${{classes}}" title="${{escapeHtml(source)}} → ${{escapeHtml(column)}}">${{value}}</td>`;
        }}).join('');
        return `<tr><th class="${{source === target ? 'selected-axis' : ''}}">${{escapeHtml(source)}}</th>${{cells}}</tr>`;
      }}).join('');
      relationMatrix.innerHTML = `<table class="matrix-table"><thead><tr><th>wiersz ↓ / kolumna →</th>${{header}}</tr></thead><tbody>${{rows}}</tbody></table>`;
    }}

    function renderDeaStructure() {{
      const efficiencyGroups = new Map();
      DATA.dea.efficiency_fronts.forEach(row => {{
        const front = Number(row.efficiency_front);
        if (!efficiencyGroups.has(front)) efficiencyGroups.set(front, []);
        efficiencyGroups.get(front).push(row.name);
      }});
      efficiencyFrontTable.innerHTML = [...efficiencyGroups.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([front, members]) => `
          <tr><td>F${{front}}</td><td>${{members.length}}</td><td>${{escapeHtml(members.join(' | '))}}</td></tr>`
        ).join('');
      componentTable.innerHTML = DATA.dea.components.map(row => `
        <tr><td>R${{row.necessary_rank}}</td><td>${{escapeHtml(row.component_id)}}</td><td>${{escapeHtml(row.members)}}</td></tr>`
      ).join('');
      edgeTable.innerHTML = DATA.dea.edges.length
        ? DATA.dea.edges.map(row => `
            <tr><td>${{escapeHtml(row.source_component)}}: ${{escapeHtml(row.source_members)}}</td><td>${{escapeHtml(row.target_component)}}: ${{escapeHtml(row.target_members)}}</td></tr>`
          ).join('')
        : '<tr><td colspan="2">Brak krawędzi między różnymi klasami.</td></tr>';
    }}

    function renderHasse(target) {{
      const targetResult = DATA.dea.results.find(row => row.name === target);
      const targetComponent = targetResult.necessary_component;
      const components = new Map(DATA.dea.components.map(row => [row.component_id, row]));
      const adjacency = new Map([...components.keys()].map(id => [id, []]));
      DATA.dea.edges.forEach(edge => adjacency.get(edge.source_component).push(edge.target_component));

      const reachable = new Set();
      const stack = [targetComponent];
      while (stack.length) {{
        const component = stack.pop();
        if (reachable.has(component)) continue;
        reachable.add(component);
        adjacency.get(component).forEach(next => stack.push(next));
      }}

      const memo = new Map();
      function pathCount(component) {{
        if (memo.has(component)) return memo.get(component);
        const successors = adjacency.get(component).filter(next => reachable.has(next));
        const count = successors.length
          ? successors.reduce((total, next) => total + pathCount(next), 0)
          : 1;
        memo.set(component, count);
        return count;
      }}

      const visibleComponents = [...reachable]
        .map(id => components.get(id))
        .sort((a, b) => a.necessary_rank - b.necessary_rank || a.component_id.localeCompare(b.component_id));
      const maximumRank = Math.max(...visibleComponents.map(row => row.necessary_rank));
      const positions = new Map();
      for (let rank = 1; rank <= maximumRank; rank += 1) {{
        const level = visibleComponents.filter(row => row.necessary_rank === rank);
        level.forEach((row, index) => positions.set(row.component_id, {{
          x: (index + 1) / (level.length + 1),
          y: maximumRank - rank + 1
        }}));
      }}

      const edgeX = [];
      const edgeY = [];
      DATA.dea.edges
        .filter(edge => reachable.has(edge.source_component) && reachable.has(edge.target_component))
        .forEach(edge => {{
          const source = positions.get(edge.source_component);
          const destination = positions.get(edge.target_component);
          edgeX.push(source.x, destination.x, null);
          edgeY.push(source.y, destination.y, null);
        }});

      const nodeX = visibleComponents.map(row => positions.get(row.component_id).x);
      const nodeY = visibleComponents.map(row => positions.get(row.component_id).y);
      const nodeText = visibleComponents.map(row => row.members.split('|').join('<br>'));
      const customdata = visibleComponents.map(row => [
        row.component_id, row.necessary_rank, row.members_count,
        row.component_id === targetComponent ? 'wybrana DMU' : 'necessary-lepsza od targetu'
      ]);
      const nodeColors = visibleComponents.map(row =>
        row.component_id === targetComponent ? '#f0aa18' : '#2878b5'
      );
      const nodeSizes = visibleComponents.map(row =>
        row.component_id === targetComponent ? 24 : 17
      );

      const traces = [
        {{
          type: 'scatter', mode: 'lines', name: 'Krawędź pokrycia NEC',
          x: edgeX, y: edgeY,
          line: {{color: '#7c91a4', width: 2.2}}, hoverinfo: 'skip'
        }},
        {{
          type: 'scatter', mode: 'markers+text', name: 'Klasy NEC',
          x: nodeX, y: nodeY, text: nodeText, textposition: 'top center',
          customdata,
          marker: {{color: nodeColors, size: nodeSizes, line: {{color: '#17324a', width: 1.5}}}},
          hovertemplate: '<b>%{{text}}</b><br>klasa=%{{customdata[0]}}<br>poziom NEC=R%{{customdata[1]}}<br>liczba DMU=%{{customdata[2]}}<br>%{{customdata[3]}}<extra></extra>'
        }}
      ];
      const layout = {{
        autosize: true, height: 620,
        margin: {{l: 50, r: 40, t: 42, b: 45}},
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
        title: {{text: `Ścieżki Hassego od ${{target}} ku klasom necessary-lepszym`, x: 0.01, xanchor: 'left'}},
        hovermode: 'closest', showlegend: false,
        xaxis: {{visible: false, range: [0, 1]}},
        yaxis: {{
          title: {{text: 'Poziom rankingu necessary'}},
          tickmode: 'array',
          tickvals: Array.from({{length: maximumRank}}, (_, index) => maximumRank - index),
          ticktext: Array.from({{length: maximumRank}}, (_, index) => `R${{index + 1}}`),
          range: [0.55, maximumRank + 0.45], zeroline: false, gridcolor: '#d7e0e7'
        }}
      }};
      Plotly.react(hassePlot, traces, layout, {{responsive: true, displaylogo: false}});
      hasseSummary.innerHTML = `
        <span>Wybrana DMU: <strong>${{escapeHtml(target)}}</strong></span>
        <span>poziom NEC: <strong>R${{targetResult.necessary_rank}}</strong></span>
        <span>osiągalne klasy necessary-lepsze: <strong>${{reachable.size - 1}}</strong></span>
        <span>ścieżki Hassego do klas maksymalnych: <strong>${{pathCount(targetComponent)}}</strong></span>`;
    }}

    function candidateTrace(unit, scale, stage, status, label, symbol, size, opacity) {{
      const points = unit.candidates[String(stage)]
        .filter(point => point.status === status)
        .map(point => ({{...point, display: displayPoint(point, scale)}}))
        .sort((a, b) => a.display.x - b.display.x);
      const isFront = status === 'front';
      const threshold = unit.stage_summary[String(stage)].threshold;
      return {{
        type: 'scatter', mode: isFront ? 'lines+markers' : 'markers', name: label,
        x: points.map(point => point.display.x), y: points.map(point => point.display.y),
        customdata: points.map(point => [
          point.name, point.best_efficiency, point.worst_efficiency, point.score_width,
          point.best_rank, point.worst_rank, point.source,
          point.in_selection_pool ? 'tak' : 'nie', threshold, point.x, point.y
        ]),
        line: isFront ? {{color: colors[stage], width: 2.5}} : undefined,
        marker: {{
          color: points.map(point => point.best_efficiency), colorscale: 'Viridis',
          cmin: 0, cmax: 1, showscale: isFront,
          colorbar: isFront ? {{title: {{text: 'Best efficiency'}}, thickness: 14, len: 0.62}} : undefined,
          size, symbol: isFront ? symbols[stage] : symbol,
          line: {{color: isFront ? colors[stage] : '#6f7d8b', width: isFront ? 2.2 : 0.6}}
        }},
        opacity,
        visible: stage === 3,
        legendgroup: `candidates-${{stage}}`,
        hovertemplate:
          `<b>%{{customdata[0]}}</b><br>status: ${{label}}<br>źródło: %{{customdata[6]}}` +
          (normalizeDisplay
            ? '<br>poziom redukcji i1=%{{x:.1f}}%<br>poziom redukcji i2=%{{y:.1f}}%<br>oryginalne i1=%{{customdata[9]:.5f}}<br>oryginalne i2=%{{customdata[10]:.5f}}'
            : '<br>i1=%{{x:.5f}}<br>i2=%{{y:.5f}}') +
          '<br>best efficiency=%{{customdata[1]:.5f}}<br>worst efficiency=%{{customdata[2]:.5f}}' +
          '<br>score width=%{{customdata[3]:.5f}}<br>ranga=%{{customdata[4]}}–%{{customdata[5]}}' +
          '<br>próg etapu=%{{customdata[8]:.5f}}<br>w puli selekcyjnej: %{{customdata[7]}}<extra></extra>'
      }};
    }}

    function renderStageDetail(unit, stage) {{
      if (!stage) {{
        stageDetail.innerHTML = '<span>Widok startowy: wybierz etap, aby zobaczyć ocenione punkty i ich efektywności.</span>';
        return;
      }}
      const summary = unit.stage_summary[String(stage)];
      stageDetail.innerHTML = `
        <span>Etap: <strong>${{stage}}</strong></span>
        <span>próg best efficiency: <strong>${{format(summary.threshold, 5)}}</strong></span>
        <span>ocenione punkty: <strong>${{summary.evaluated}}</strong></span>
        <span>poniżej progu: <strong>${{summary.below_threshold}}</strong></span>
        <span>próby diagnostyczne: <strong>${{summary.probes}}</strong></span>
        <span>pula selekcyjna: <strong>${{summary.selection_pool}}</strong></span>
        <span>osiągalne i spełniające próg: <strong>${{summary.eligible}}</strong></span>
        <span>front przejścia: <strong>${{summary.front}}</strong></span>`;
    }}

    function pathTraces(unit, scale) {{
      const displayPath = unit.path.map(point => displayPoint(point, scale));
      return [
        {{
          type: 'scatter', mode: 'lines', name: 'Minimalna ścieżka',
          x: displayPath.map(point => point.x), y: displayPath.map(point => point.y),
          line: {{color: '#17324a', width: 4}}, hoverinfo: 'skip'
        }},
        {{
          type: 'scatter', mode: 'markers+text', name: 'Wybrane punkty',
          x: displayPath.map(point => point.x), y: displayPath.map(point => point.y),
          text: ['Start', '1', '2', '3'],
          textposition: ['top center', 'top center', 'top right', 'bottom right'],
          customdata: unit.path.map(point => [point.label, point.efficiency, point.x, point.y]),
          marker: {{
            color: ['#f0aa18', colors[1], colors[2], colors[3]],
            symbol: ['diamond', 'circle', 'square', 'diamond'],
            size: [16, 15, 15, 16], line: {{color: '#17324a', width: 2}}
          }},
          hovertemplate: '%{{customdata[0]}}' +
            (normalizeDisplay
              ? '<br>poziom redukcji i1=%{{x:.1f}}%<br>poziom redukcji i2=%{{y:.1f}}%<br>oryginalne i1=%{{customdata[2]:.5f}}<br>oryginalne i2=%{{customdata[3]:.5f}}'
              : '<br>i1=%{{x:.4f}}<br>i2=%{{y:.4f}}') +
            '<br>efektywność=%{{customdata[1]:.4f}}<extra></extra>'
        }}
      ];
    }}

    function renderTable(unit) {{
      tableBody.innerHTML = unit.path.map(point => `
        <tr>
          <td>${{point.stage}}</td><td>${{point.name}}</td>
          <td>${{format(point.x)}}</td><td>${{format(point.y)}}</td>
          <td>${{format(point.efficiency)}}</td><td>${{format(point.effort, 6)}}</td>
          <td class="status">${{point.pareto === null ? 'start' : 'Pareto-min.'}}</td>
        </tr>`).join('');
      resultLine.innerHTML = `
        <span>Jednostka: <strong>${{unit.target}}</strong></span>
        ${{Object.entries(unit.fixed_outputs).map(([name, value]) => `<span>stałe ${{escapeHtml(name)}}: <strong>${{format(value)}}</strong></span>`).join('')}}
        <span>TC: <strong>${{format(unit.metrics.tc, 6)}}</strong></span>
        <span>MSC: <strong>${{format(unit.metrics.msc, 6)}}</strong></span>
        <span>DR: <strong>${{format(unit.metrics.dr, 3)}}</strong></span>
        <span>sprawdzone ścieżki: <strong>${{unit.path_count.toLocaleString('pl-PL')}}</strong></span>`;
    }}

    function render(target) {{
      const unit = DATA.units[target];
      const scale = displayScale(unit);
      renderDeaSummary(target);
      renderDeaResults(target);
      renderRelationMatrix(target);
      renderHasse(target);
      const reference = {{
        type: 'scatter', mode: 'markers', name: 'Jednostki referencyjne',
        x: DATA.dmus.map(point => displayPoint({{x: point.i1, y: point.i2}}, scale).x),
        y: DATA.dmus.map(point => displayPoint({{x: point.i1, y: point.i2}}, scale).y),
        text: DATA.dmus.map(point => point.name),
        customdata: DATA.dmus.map(point => [point.i1, point.i2, ...DATA.dea.output_columns.map(column => point[column])]),
        marker: {{color: '#8a99a8', size: 8, opacity: 0.52}},
        hovertemplate: '%{{text}}' +
          (normalizeDisplay ? '<br>poziom redukcji i1=%{{x:.1f}}%<br>poziom redukcji i2=%{{y:.1f}}%' : '') +
          '<br>oryginalne i1=%{{customdata[0]:.3f}}<br>oryginalne i2=%{{customdata[1]:.3f}}' +
          DATA.dea.output_columns.map((column, index) => `<br>${{column}}=%{{customdata[${{index + 2}}]:.3f}}`).join('') + '<extra></extra>'
      }};
      const traces = [reference];
      const candidateTraceIndexes = {{1: [], 2: [], 3: []}};
      for (let stage = 1; stage <= 3; stage += 1) {{
        for (const statusInfo of candidateStatuses) {{
          candidateTraceIndexes[stage].push(traces.length);
          traces.push(candidateTrace(unit, scale, stage, ...statusInfo));
        }}
      }}
      const pathLineIndex = traces.length;
      traces.push(...pathTraces(unit, scale));
      const pathPointIndex = pathLineIndex + 1;
      const frameNames = unit.path.map(point => `${{target}}-${{point.stage}}`);
      const frames = unit.path.map((point, index) => {{
        const frameTraceIndexes = [];
        const frameData = [];
        for (let stage = 1; stage <= 3; stage += 1) {{
          for (const traceIndex of candidateTraceIndexes[stage]) {{
            frameTraceIndexes.push(traceIndex);
            frameData.push({{visible: point.stage === stage, showlegend: point.stage === stage}});
          }}
        }}
        frameTraceIndexes.push(pathLineIndex, pathPointIndex);
        frameData.push(
          {{x: unit.path.slice(0, index + 1).map(item => displayPoint(item, scale).x), y: unit.path.slice(0, index + 1).map(item => displayPoint(item, scale).y)}},
          {{
            x: unit.path.slice(0, index + 1).map(item => displayPoint(item, scale).x),
            y: unit.path.slice(0, index + 1).map(item => displayPoint(item, scale).y),
            text: ['Start', '1', '2', '3'].slice(0, index + 1),
            customdata: unit.path.slice(0, index + 1).map(item => [item.label, item.efficiency, item.x, item.y]),
            marker: {{
              color: ['#f0aa18', colors[1], colors[2], colors[3]].slice(0, index + 1),
              symbol: ['diamond', 'circle', 'square', 'diamond'].slice(0, index + 1),
              size: [16, 15, 15, 16].slice(0, index + 1), line: {{color: '#17324a', width: 2}}
            }}
          }}
        );
        return {{name: frameNames[index], traces: frameTraceIndexes, data: frameData}};
      }});
      const layout = {{
        autosize: true, margin: {{l: 64, r: 24, t: 34, b: 124}},
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: {{color: '#17324a'}},
        title: {{text: `Jednostka ${{target}} · ` + Object.entries(unit.fixed_outputs).map(([name, value]) => `${{name}} = ${{format(value)}}`).join(' · '), x: 0.01, xanchor: 'left'}},
        hovermode: 'closest', dragmode: 'pan',
        legend: {{orientation: 'h', x: 0.5, xanchor: 'center', y: -0.16, yanchor: 'top'}},
        xaxis: {{title: {{text: {json.dumps(display_x_label, ensure_ascii=False)}}}, range: {json.dumps(display_x_range)}, ticksuffix: normalizeDisplay ? '%' : '', zeroline: false, gridcolor: '#d7e0e7', linecolor: '#d7e0e7', mirror: true, constrain: 'domain'}},
        yaxis: {{title: {{text: {json.dumps(display_y_label, ensure_ascii=False)}}}, range: {json.dumps(display_y_range)}, ticksuffix: normalizeDisplay ? '%' : '', zeroline: false, gridcolor: '#d7e0e7', linecolor: '#d7e0e7', mirror: true{", scaleanchor: 'x', scaleratio: 1" if normalize_display or not independent_axis_scales else ""}}},
        updatemenus: [{{
          type: 'buttons', direction: 'left', x: 0, xanchor: 'left', y: -0.27, yanchor: 'top', showactive: false,
          buttons: [
            {{label: 'Odtwórz ścieżkę', method: 'animate', args: [frameNames, {{fromcurrent: false, frame: {{duration: 800, redraw: true}}, transition: {{duration: 500}}, mode: 'immediate'}}]}},
            {{label: 'Pauza', method: 'animate', args: [[null], {{frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}, mode: 'immediate'}}]}}
          ]
        }}],
        sliders: [{{
          active: 3, x: 0.32, len: 0.68, y: -0.265, yanchor: 'top', currentvalue: {{prefix: 'Widok: '}},
          steps: ['Start', 'Etap 1', 'Etap 2', 'Etap 3'].map((label, index) => ({{
            label, method: 'animate', args: [[frameNames[index]], {{mode: 'immediate', frame: {{duration: 400, redraw: true}}, transition: {{duration: 300}}}}]
          }}))
        }}]
      }};
      const config = {{responsive: true, scrollZoom: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'], toImageButtonOptions: {{format: 'png', filename: `minimalna-sciezka-${{target}}`, scale: 2}}}};

      const existingFrameCount = plot._transitionData?._frames?.length || 0;
      const clearFrames = existingFrameCount
        ? Plotly.deleteFrames(plot, Array.from({{length: existingFrameCount}}, (_, index) => index))
        : Promise.resolve();
      clearFrames
        .then(() => Plotly.react(plot, traces, layout, config))
        .then(() => Plotly.addFrames(plot, frames))
        .then(() => {{
          if (plot.removeAllListeners) plot.removeAllListeners('plotly_sliderchange');
          plot.on('plotly_sliderchange', event => renderStageDetail(unit, event.slider.active));
        }});
      renderStageDetail(unit, 3);
      renderTable(unit);
    }}

    document.querySelectorAll('.unit-button').forEach(button => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('.unit-button').forEach(item => {{
          const active = item === button;
          item.classList.toggle('active', active);
          item.setAttribute('aria-pressed', String(active));
        }});
        render(button.dataset.target);
      }});
    }});
    relationKind.addEventListener('change', () => {{
      renderRelationMatrix(document.querySelector('.unit-button.active').dataset.target);
    }});
    renderDeaStructure();
    render(DATA.order[0]);
  </script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    input_path = args.input.resolve()
    root = args.root.resolve()
    snapshot_dir = (
        args.dea_snapshot_dir.resolve()
        if args.dea_snapshot_dir is not None
        else root / "dea_input_snapshot"
    )
    targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    input_frame = pd.read_csv(input_path)
    dea_snapshot = load_dea_snapshot(snapshot_dir, input_frame)
    io_columns = [*dea_snapshot["input_columns"], *dea_snapshot["output_columns"]]
    payload = {
        "order": targets,
        "dmus": input_frame[["name", *io_columns]].round(6).to_dict("records"),
        "dea": dea_snapshot,
        "units": {
            target: load_target(root, target, input_frame)
            for target in targets
        },
    }
    build_html(
        payload,
        args.output.resolve(),
        plotly_mode=args.plotly_mode,
        x_label=args.x_label,
        y_label=args.y_label,
        independent_axis_scales=args.independent_axis_scales,
        normalize_display=args.normalize_display,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
