from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

from minimality_certificate import find_selected_path
from visualize_minimality_2d import load_visualization_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one self-contained interactive comparison of 2D DEA paths."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def round_value(value, digits=6):
    return round(float(value), digits)


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
    for stage in range(1, 4):
        fronts[str(stage)] = [
            {
                "x": round_value(point["x"]),
                "y": round_value(point["y"]),
                "efficiency": round_value(point["best_efficiency"]),
            }
            for point in data["points"]
            if int(point["stage"]) == stage and bool(point["front"])
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
        "o1": round_value(
            input_frame.loc[input_frame["name"].astype(str) == target, "o1"].iloc[0]
        ),
        "metrics": {
            "tc": tc,
            "msc": msc,
            "cdir": cdir,
            "dr": round_value(selected_metric["dr"]),
        },
        "path": path,
        "fronts": fronts,
        "front_counts": {
            str(int(row["stage"])): int(row["pareto_front_points"])
            for _, row in certificate.iterrows()
        },
    }


def build_html(payload: dict, output: Path):
    plotly_js = get_plotlyjs()
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    default_target = payload["order"][0]
    target_buttons = "\n".join(
        f'<button type="button" class="unit-button{" active" if target == default_target else ""}" '
        f'data-target="{html.escape(target)}" aria-pressed="{str(target == default_target).lower()}">{html.escape(target)}</button>'
        for target in payload["order"]
    )
    document = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Minimalne ścieżki DEA - porównanie pięciu jednostek</title>
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
  <script>{plotly_js}</script>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Minimalne ścieżki poprawy efektywności DEA</h1>
        <p class="subtitle">Dwa zmieniane wejścia: i1, i2; wyjście o1 pozostaje stałe dla wybranej jednostki.</p>
      </div>
      <nav class="unit-controls" aria-label="Wybór jednostki">
        {target_buttons}
      </nav>
    </header>
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
    <footer>Wspólne parametry: cel efektywności 0,92; 3 etapy; globalny grid 20×20; lokalne siatki 10×10; wybór: minimalne TC, następnie minimalne MSC.</footer>
  </main>
  <script>
    const DATA = {data_json};
    const plot = document.getElementById('plot');
    const tableBody = document.getElementById('path-table');
    const resultLine = document.getElementById('result-line');
    const colors = {{1: '#df5a56', 2: '#3978b5', 3: '#43a06b'}};
    const symbols = {{1: 'circle-open', 2: 'square-open', 3: 'diamond-open'}};

    function format(value, digits = 4) {{
      if (value === null || value === undefined) return '–';
      return Number(value).toFixed(digits).replace('.', ',');
    }}

    function frontTrace(unit, stage) {{
      const points = [...unit.fronts[String(stage)]].sort((a, b) => a.x - b.x);
      return {{
        type: 'scatter', mode: 'lines+markers', name: `Front etapu ${{stage}}`,
        x: points.map(point => point.x), y: points.map(point => point.y),
        customdata: points.map(point => point.efficiency),
        line: {{color: colors[stage], width: 2}},
        marker: {{color: colors[stage], size: 8, symbol: symbols[stage], line: {{color: colors[stage], width: 2}}}},
        opacity: 0.88,
        hovertemplate: `Front etapu ${{stage}}<br>i1=%{{x:.4f}}<br>i2=%{{y:.4f}}<br>efektywność=%{{customdata:.4f}}<extra></extra>`
      }};
    }}

    function pathTraces(unit) {{
      return [
        {{
          type: 'scatter', mode: 'lines', name: 'Minimalna ścieżka',
          x: unit.path.map(point => point.x), y: unit.path.map(point => point.y),
          line: {{color: '#17324a', width: 4}}, hoverinfo: 'skip'
        }},
        {{
          type: 'scatter', mode: 'markers+text', name: 'Wybrane punkty',
          x: unit.path.map(point => point.x), y: unit.path.map(point => point.y),
          text: ['Start', '1', '2', '3'],
          textposition: ['top center', 'top center', 'top right', 'bottom right'],
          customdata: unit.path.map(point => [point.label, point.efficiency]),
          marker: {{
            color: ['#f0aa18', colors[1], colors[2], colors[3]],
            symbol: ['diamond', 'circle', 'square', 'diamond'],
            size: [16, 15, 15, 16], line: {{color: '#17324a', width: 2}}
          }},
          hovertemplate: '%{{customdata[0]}}<br>i1=%{{x:.4f}}<br>i2=%{{y:.4f}}<br>efektywność=%{{customdata[1]:.4f}}<extra></extra>'
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
        <span>stałe o1: <strong>${{format(unit.o1)}}</strong></span>
        <span>TC: <strong>${{format(unit.metrics.tc, 6)}}</strong></span>
        <span>MSC: <strong>${{format(unit.metrics.msc, 6)}}</strong></span>
        <span>DR: <strong>${{format(unit.metrics.dr, 3)}}</strong></span>
        <span>sprawdzone ścieżki: <strong>${{unit.path_count.toLocaleString('pl-PL')}}</strong></span>`;
    }}

    function render(target) {{
      const unit = DATA.units[target];
      const reference = {{
        type: 'scatter', mode: 'markers', name: 'Jednostki referencyjne',
        x: DATA.dmus.map(point => point.i1), y: DATA.dmus.map(point => point.i2),
        text: DATA.dmus.map(point => point.name), customdata: DATA.dmus.map(point => point.o1),
        marker: {{color: '#8a99a8', size: 8, opacity: 0.52}},
        hovertemplate: '%{{text}}<br>i1=%{{x:.3f}}<br>i2=%{{y:.3f}}<br>o1=%{{customdata:.3f}}<extra></extra>'
      }};
      const traces = [reference, frontTrace(unit, 1), frontTrace(unit, 2), frontTrace(unit, 3), ...pathTraces(unit)];
      const frameNames = unit.path.map(point => `${{target}}-${{point.stage}}`);
      const frames = unit.path.map((point, index) => ({{
        name: frameNames[index], traces: [4, 5], data: [
          {{x: unit.path.slice(0, index + 1).map(item => item.x), y: unit.path.slice(0, index + 1).map(item => item.y)}},
          {{
            x: unit.path.slice(0, index + 1).map(item => item.x),
            y: unit.path.slice(0, index + 1).map(item => item.y),
            text: ['Start', '1', '2', '3'].slice(0, index + 1),
            customdata: unit.path.slice(0, index + 1).map(item => [item.label, item.efficiency]),
            marker: {{
              color: ['#f0aa18', colors[1], colors[2], colors[3]].slice(0, index + 1),
              symbol: ['diamond', 'circle', 'square', 'diamond'].slice(0, index + 1),
              size: [16, 15, 15, 16].slice(0, index + 1), line: {{color: '#17324a', width: 2}}
            }}
          }}
        ]
      }}));
      const layout = {{
        autosize: true, margin: {{l: 64, r: 24, t: 34, b: 124}},
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: {{color: '#17324a'}},
        title: {{text: `Jednostka ${{target}} · o1 = ${{format(unit.o1)}}`, x: 0.01, xanchor: 'left'}},
        hovermode: 'closest', dragmode: 'pan',
        legend: {{orientation: 'h', x: 0.5, xanchor: 'center', y: -0.16, yanchor: 'top'}},
        xaxis: {{title: {{text: 'Wejście i1'}}, range: [-0.5, 9.8], zeroline: false, gridcolor: '#d7e0e7', linecolor: '#d7e0e7', mirror: true, constrain: 'domain'}},
        yaxis: {{title: {{text: 'Wejście i2'}}, range: [-0.1, 9.8], zeroline: false, gridcolor: '#d7e0e7', linecolor: '#d7e0e7', mirror: true, scaleanchor: 'x', scaleratio: 1}},
        updatemenus: [{{
          type: 'buttons', direction: 'left', x: 0, xanchor: 'left', y: -0.27, yanchor: 'top', showactive: false,
          buttons: [
            {{label: 'Odtwórz ścieżkę', method: 'animate', args: [frameNames, {{fromcurrent: false, frame: {{duration: 800, redraw: false}}, transition: {{duration: 500}}, mode: 'immediate'}}]}},
            {{label: 'Pauza', method: 'animate', args: [[null], {{frame: {{duration: 0, redraw: false}}, transition: {{duration: 0}}, mode: 'immediate'}}]}}
          ]
        }}],
        sliders: [{{
          active: 3, x: 0.32, len: 0.68, y: -0.265, yanchor: 'top', currentvalue: {{prefix: 'Widok: '}},
          steps: ['Start', 'Etap 1', 'Etap 2', 'Etap 3'].map((label, index) => ({{
            label, method: 'animate', args: [[frameNames[index]], {{mode: 'immediate', frame: {{duration: 400, redraw: false}}, transition: {{duration: 300}}}}]
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
        .then(() => Plotly.addFrames(plot, frames));
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
    targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    input_frame = pd.read_csv(input_path)
    payload = {
        "order": targets,
        "dmus": input_frame[["name", "i1", "i2", "o1"]].round(6).to_dict("records"),
        "units": {
            target: load_target(root, target, input_frame)
            for target in targets
        },
    }
    build_html(payload, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
