from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


METHODS = [
    ("hasse_path", "Hasse path", "#7a5195"),
    ("front_path", "Necessary-layer path", "#ef5675"),
    ("best_rank_path", "Best-rank milestones", "#ffa600"),
    ("best_efficiency_path", "Best-efficiency milestones", "#2f4b7c"),
    ("robustness_width_path", "Robustness-width milestones", "#00a6a6"),
]


def _latest_run(method_dir: Path) -> Path:
    runs = sorted(path for path in method_dir.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run_* directory below {method_dir}.")
    return runs[-1]


def _fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, str):
        return html.escape(value)
    return f"{float(value):.{digits}f}"


def _stage_indices(row: pd.Series) -> list[int]:
    indices = []
    for column in row.index:
        if column.startswith("stage_") and column.endswith("_name"):
            try:
                indices.append(int(column.split("_")[1]))
            except ValueError:
                pass
    return sorted(set(indices))


def _selected_path(run_dir: Path) -> tuple[pd.Series, pd.Series, int, int]:
    metrics = pd.read_csv(run_dir / "path_metrics.csv")
    paths = pd.read_csv(run_dir / "paths.csv")
    selection = pd.read_csv(run_dir / "path_selection.csv")
    selected = selection[selection["selection_type"] == "lexicographic"]
    if selected.empty:
        raise ValueError(f"No lexicographic selection in {run_dir}.")
    path_id = str(selected.iloc[0]["path_id"])
    metric_row = metrics[metrics["path_id"].astype(str) == path_id].iloc[0]
    path_row = paths[paths["path_id"].astype(str) == path_id].iloc[0]
    pareto_count = len(pd.read_csv(run_dir / "path_pareto.csv"))
    representatives_file = run_dir / "path_pareto_representatives.csv"
    representative_count = (
        len(pd.read_csv(representatives_file))
        if representatives_file.is_file()
        else pareto_count
    )
    return metric_row, path_row, len(metrics), representative_count


def build_report(
    root: Path,
    input_csv: Path,
    output_html: Path,
    plotly_mode: str,
    target: str,
) -> None:
    reference = pd.read_csv(input_csv)
    snapshot = pd.read_csv(root / "dea_input_snapshot" / "reference_dmu_results.csv")
    baseline = snapshot[snapshot["name"].astype(str) == target].iloc[0]

    traces = [
        {
            "type": "scatter",
            "mode": "markers",
            "name": "Reference robots",
            "x": reference["i1"].astype(float).tolist(),
            "y": reference["i2"].astype(float).tolist(),
            "text": reference["name"].astype(str).tolist(),
            "marker": {"size": 7, "color": "#bcc5cf", "opacity": 0.55},
            "hovertemplate": "%{text}<br>Cost i1=%{x:.4f}<br>Repeatability i2=%{y:.4f}<extra></extra>",
        }
    ]
    overview_rows = []
    selection_rows = []
    stage_sections = []
    selected_x = []
    selected_y = []

    for method_dir_name, label, color in METHODS:
        run_dir = _latest_run(root / method_dir_name)
        metric, path, path_count, representative_count = _selected_path(run_dir)
        method_metrics = pd.read_csv(run_dir / "path_metrics.csv")
        method_selection = pd.read_csv(run_dir / "path_selection.csv")
        metrics_by_path = method_metrics.set_index(method_metrics["path_id"].astype(str))
        for selection_type, selection_label in [
            ("single_metric", "Minimum TC"),
            ("lexicographic", "Lexicographic"),
            ("normalized_weighted", "Normalized weighted"),
        ]:
            choice = method_selection[
                method_selection["selection_type"].eq(selection_type)
            ]
            if selection_type == "single_metric":
                choice = choice[choice["selection_detail"].eq("tc")]
            if choice.empty:
                continue
            choice_row = choice.iloc[0]
            choice_metric = metrics_by_path.loc[str(choice_row["path_id"])]
            selection_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td><td>{selection_label}</td>"
                f"<td><code>{html.escape(str(choice_metric['path_id']))}</code></td>"
                f"<td>{_fmt(choice_metric.get('tc'))}</td>"
                f"<td>{_fmt(choice_metric.get('msc'))}</td>"
                f"<td>{_fmt(choice_metric.get('md'))}</td>"
                f"<td>{_fmt(choice_metric.get('swbp'))}</td>"
                f"<td>{_fmt(choice_metric.get('fw'))}</td>"
                f"<td>{_fmt(choice_metric.get('best_efficiency_final'))}</td>"
                f"<td>{_fmt(choice_metric.get('best_rank_final'), 0)}</td>"
                "</tr>"
            )
        stages = _stage_indices(path)
        x_values = [float(path[f"stage_{stage:02d}_i1"]) for stage in stages]
        y_values = [float(path[f"stage_{stage:02d}_i2"]) for stage in stages]
        selected_x.extend(x_values)
        selected_y.extend(y_values)
        names = [str(path[f"stage_{stage:02d}_name"]) for stage in stages]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers+text",
                "name": label,
                "x": x_values,
                "y": y_values,
                "text": ["S" if stage == 0 else str(stage) for stage in stages],
                "customdata": names,
                "textposition": "top right",
                "line": {"color": color, "width": 3},
                "marker": {"color": color, "size": 11, "line": {"color": "white", "width": 1.5}},
                "hovertemplate": "%{customdata}<br>Cost i1=%{x:.5f}<br>Repeatability i2=%{y:.5f}<extra>" + label + "</extra>",
            }
        )
        overview_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td><td>{path_count}</td><td>{representative_count}</td>"
            f"<td><code>{html.escape(str(metric['path_id']))}</code></td>"
            f"<td>{_fmt(metric.get('tc'))}</td><td>{_fmt(metric.get('msc'))}</td>"
            f"<td>{_fmt(metric.get('cdir'))}</td><td>{_fmt(metric.get('dr'))}</td>"
            f"<td>{_fmt(metric.get('md'))}</td><td>{_fmt(metric.get('swbp'))}</td>"
            f"<td>{_fmt(metric.get('best_efficiency_final'))}</td>"
            f"<td>{_fmt(metric.get('best_rank_final'), 0)}</td>"
            f"<td>{_fmt(metric.get('score_width_final'))}</td>"
            f"<td>{_fmt(metric.get('rank_width_final'), 0)}</td>"
            "</tr>"
        )

        stage_rows = []
        for stage in stages:
            prefix = f"stage_{stage:02d}"
            stage_rows.append(
                "<tr>"
                f"<td>{stage}</td><td><code>{html.escape(str(path[f'{prefix}_name']))}</code></td>"
                f"<td>{_fmt(path.get(f'{prefix}_i1'), 5)}</td>"
                f"<td>{_fmt(path.get(f'{prefix}_i2'), 5)}</td>"
                f"<td>{_fmt(path.get(f'{prefix}_best_efficiency'))}</td>"
                f"<td>{_fmt(path.get(f'{prefix}_best_rank'), 0)}</td>"
                f"<td>{_fmt(path.get(f'{prefix}_score_width'))}</td>"
                f"<td>{_fmt(path.get(f'{prefix}_rank_width'), 0)}</td>"
                f"<td>{_fmt(path.get(f'{prefix}_milestone_gap'))}</td>"
                f"<td>{_fmt(path.get(f'{prefix}_effort_from_previous'))}</td>"
                "</tr>"
            )
        stage_sections.append(
            f"<details><summary>{html.escape(label)} — <code>{html.escape(str(metric['path_id']))}</code></summary>"
            "<div class='table-wrap'><table><thead><tr><th>Stage</th><th>State</th><th>i1</th><th>i2</th>"
            "<th>Best eff.</th><th>Best rank</th><th>Score width</th><th>Rank width</th>"
            "<th>Milestone gap</th><th>Step effort</th></tr></thead><tbody>"
            + "".join(stage_rows)
            + "</tbody></table></div></details>"
        )

    x_span = max(selected_x) - min(selected_x)
    y_span = max(selected_y) - min(selected_y)
    x_padding = max(0.05 * x_span, 0.1)
    y_padding = max(0.08 * y_span, 0.01)
    layout = {
        "title": f"{target} — five robust improvement-path generators",
        "xaxis": {
            "title": "i1: Cost ($10,000), lower is better",
            "range": [max(0.0, min(selected_x) - x_padding), max(selected_x) + x_padding],
        },
        "yaxis": {
            "title": "i2: Repeatability (mm), lower is better",
            "range": [max(0.0, min(selected_y) - y_padding), max(selected_y) + y_padding],
        },
        "template": "plotly_white",
        "hovermode": "closest",
        "legend": {"orientation": "h", "y": -0.22},
        "margin": {"l": 75, "r": 30, "t": 70, "b": 130},
    }
    plotly_script = (
        f"<script>{get_plotlyjs()}</script>"
        if plotly_mode == "embedded"
        else "<script src='https://cdn.plot.ly/plotly-3.3.1.min.js'></script>"
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DEA-F controlled five-method example</title>{plotly_script}
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#17212b}}main{{max-width:1500px;margin:auto;padding:28px}}
h1{{margin-bottom:4px}}.subtitle{{color:#596879;margin-top:0}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:22px 0}}
.card,section,details{{background:white;border:1px solid #dfe5ec;border-radius:12px;box-shadow:0 3px 14px #2030400b}}.card{{padding:16px}}.card b{{display:block;font-size:1.35rem;margin-top:5px}}
section{{padding:20px;margin:18px 0}}#plot{{height:720px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.88rem}}th,td{{padding:8px 9px;border-bottom:1px solid #e5e9ef;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#f7f9fb;position:sticky;top:0}}
details{{padding:12px 16px;margin:10px 0}}summary{{font-weight:650;cursor:pointer}}code{{font-family:Consolas,monospace;font-size:.86em}}.note{{border-left:4px solid #2f4b7c;padding-left:14px;color:#405064}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}#plot{{height:560px}}}}
</style></head><body><main>
<h1>Controlled five-method DEA-F example</h1><p class="subtitle">{html.escape(target)} · two adjustable inputs (i1, i2) · fixed outputs (o1, o2) · lexicographic publication choices</p>
<div class="cards"><div class="card">Start best efficiency<b>{_fmt(baseline['best_efficiency'])}</b></div><div class="card">Start best rank<b>{_fmt(baseline['best_rank'],0)}</b></div><div class="card">Start score width<b>{_fmt(baseline['score_width'])}</b></div><div class="card">Start rank width<b>{_fmt(baseline['rank_width'],0)}</b></div></div>
<section><p class="note">Default view is zoomed to the attainable rectangle of the selected paths. Use Autoscale in the plot toolbar to restore all 27 reference robots.</p><div id="plot"></div></section>
<section><h2>Selected-path comparison</h2><p class="note">Path counts refer to complete paths in the evaluated finite pool. “Pareto repr.” collapses paths with the same criterion vector to one representative.</p><div class="table-wrap"><table><thead><tr><th>Method</th><th>Paths</th><th>Pareto repr.</th><th>Selected</th><th>TC</th><th>MSC</th><th>Cdir</th><th>DR</th><th>MD</th><th>SWBP</th><th>Final E*</th><th>Final rank*</th><th>Final score width</th><th>Final rank width</th></tr></thead><tbody>{''.join(overview_rows)}</tbody></table></div></section>
<section><h2>Selection lenses</h2><p class="note">The minimum-TC row exposes the cheapest admissible path. The lexicographic row follows the paper's priority order; the normalized weighted row exposes trade-offs after min-max scaling. Coincident rows are genuine agreement, not deduplication.</p><div class="table-wrap"><table><thead><tr><th>Method</th><th>Selection rule</th><th>Path</th><th>TC</th><th>MSC</th><th>MD</th><th>SWBP</th><th>FW</th><th>Final E*</th><th>Final rank*</th></tr></thead><tbody>{''.join(selection_rows)}</tbody></table></div></section>
<section><h2>Stage-by-stage paths</h2>{''.join(stage_sections)}</section>
<section><h2>How to read this example</h2><ul><li>Hasse and necessary-layer milestones encode robust necessary-order structure; the latter are not DEA efficiency fronts.</li><li>Best-rank and best-efficiency milestones track numerical targets; robustness-width minimizes rank width while forbidding deterioration of best rank.</li><li>TC and MSC use only i1 and i2 with reference-data normalization and factor weights 0.6/0.4. Fixed outputs remain part of the attainability check.</li><li>The best-efficiency comparison demonstrates the central selection trade-off: a very cheap path may overshoot the intermediate milestones, while the lexicographic choice pays more to follow them closely.</li><li>The width path repeats a state after reaching the discrete target early. This is an informative limitation of a coarse finite candidate pool, not evidence of three distinct operational changes.</li><li>Cdir is a finite-pool estimate identified by cdir_scope, not an analytical global optimum. PC is intentionally absent because the Java adapter does not export genuine DEA peer sets; relation continuity is RC.</li><li>Every displayed path is selected from candidates evaluated in this run. The result supports an algorithmic comparison, not a global minimality theorem for the continuous space.</li></ul></section>
<script>Plotly.newPlot('plot',{json.dumps(traces, ensure_ascii=False)},{json.dumps(layout, ensure_ascii=False)},{{responsive:true,displaylogo:false}});</script>
</main></body></html>"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="Robot_18")
    parser.add_argument("--plotly-mode", choices=["embedded", "cdn"], default="embedded")
    args = parser.parse_args()
    build_report(
        args.root.resolve(),
        args.input.resolve(),
        args.output.resolve(),
        args.plotly_mode,
        args.target,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
