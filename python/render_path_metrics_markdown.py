import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DESCRIPTION_PATH = Path("templates/path_metric_descriptions.csv")
IDENTIFIER_COLUMNS = [
    "method",
    "path_id",
    "start_name",
    "final_name",
    "path_length",
]


def markdown_escape(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return markdown_escape(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return markdown_escape(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6g}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return "_No matching columns available._"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        rows.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def read_experiment_params(experiment_dir: Path | None) -> pd.DataFrame:
    if experiment_dir is None:
        return pd.DataFrame()
    params_path = experiment_dir / "experiment_params.csv"
    if not params_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(params_path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def discover_method_status(experiment_dir: Path | None) -> pd.DataFrame:
    if experiment_dir is None:
        return pd.DataFrame()

    rows = []
    for metric_file in sorted(experiment_dir.rglob("path_metrics.csv")):
        try:
            frame = pd.read_csv(metric_file)
            row_count = len(frame)
        except pd.errors.EmptyDataError:
            row_count = 0

        run_dir = metric_file.parent
        method = run_dir.parent.name
        rows.append(
            {
                "method": method,
                "run_dir": run_dir.name,
                "path_count": row_count,
                "status": "OK" if row_count > 0 else "brak ścieżek",
            }
        )

    return pd.DataFrame(rows)


def compact_method_summary(method_summary: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "method",
        "path_count",
        "tc_mean",
        "msc_mean",
        "dr_mean",
        "bp_mean",
        "mcp_mean",
        "md_mean",
        "pyv_mean",
        "pym_mean",
        "apw_mean",
        "fw_mean",
        "pc_mean",
        "opp_mean",
        "rr_mean",
    ]
    return method_summary[[column for column in wanted if column in method_summary.columns]]


def best_path_notes(metrics: pd.DataFrame, descriptions: pd.DataFrame) -> list[str]:
    notes = []
    for _, desc in descriptions.iterrows():
        metric = desc["metric"]
        if metric not in metrics.columns:
            continue
        values = pd.to_numeric(metrics[metric], errors="coerce")
        if not values.notna().any():
            continue

        direction = str(desc.get("direction", "")).strip()
        if direction == "lower_better":
            idx = values.idxmin()
            label = "min"
        elif direction == "higher_better":
            idx = values.idxmax()
            label = "max"
        else:
            continue

        path_id = metrics.loc[idx, "path_id"] if "path_id" in metrics.columns else idx
        method = metrics.loc[idx, "method"] if "method" in metrics.columns else ""
        method_prefix = f"`{method}` / " if method else ""
        notes.append(f"- `{metric}` {label}: {method_prefix}`{path_id}` = {format_value(values.loc[idx])}")
    return notes


def render_report(metrics: pd.DataFrame, descriptions: pd.DataFrame, source_name: str) -> str:
    return render_experiment_report(metrics, descriptions, source_name)


def render_experiment_report(
    metrics: pd.DataFrame,
    descriptions: pd.DataFrame,
    source_name: str,
    method_summary: pd.DataFrame | None = None,
    method_status: pd.DataFrame | None = None,
    experiment_params: pd.DataFrame | None = None,
    title: str = "Raport Metryk Ścieżek DEA",
    display_metrics: pd.DataFrame | None = None,
    sample_limit: int | None = None,
    sample_random_state: int | None = None,
) -> str:
    display_metrics = metrics if display_metrics is None else display_metrics
    lines = [
        f"# {title}",
        "",
        f"Źródłowy plik metryk: `{source_name}`",
        "",
        "## Zakres",
        "",
        f"- Liczba ścieżek w raporcie: **{len(metrics)}**",
    ]
    if len(display_metrics) != len(metrics):
        lines.append(f"- Liczba ścieżek pokazywanych w tabelach: **{len(display_metrics)}**")
        if sample_limit is not None:
            lines.append(f"- Sposób prezentacji: losowa próbka max **{sample_limit}** ścieżek")
        if sample_random_state is not None:
            lines.append(f"- Ziarno losowania próbki: **{sample_random_state}**")

    if "method" in metrics.columns and not metrics.empty:
        method_count = metrics["method"].nunique(dropna=True)
        lines.append(f"- Liczba metod z niepustymi ścieżkami: **{method_count}**")

    lines.append("")

    if experiment_params is not None and not experiment_params.empty:
        lines.extend(
            [
                "## Parametry Eksperymentu",
                "",
                markdown_table(experiment_params, ["parameter", "value"]),
                "",
            ]
        )

    if method_status is not None and not method_status.empty:
        lines.extend(
            [
                "## Status Metod",
                "",
                markdown_table(method_status, ["method", "run_dir", "path_count", "status"]),
                "",
            ]
        )

    if method_summary is not None and not method_summary.empty:
        compact_summary = compact_method_summary(method_summary)
        lines.extend(
            [
                "## Podsumowanie Metod",
                "",
                markdown_table(compact_summary, compact_summary.columns.tolist()),
                "",
            ]
        )

    lines.extend(
        [
            "## Ścieżki Kandydackie Według Grup Metryk",
            "",
        ]
    )

    for group, group_descriptions in descriptions.groupby("group", sort=False):
        group_metrics = [
            metric
            for metric in group_descriptions["metric"].tolist()
            if metric in metrics.columns
        ]
        if not group_metrics:
            continue
        lines.extend(
            [
                f"### {group}",
                "",
                "**Opis metryk w tej grupie**",
                "",
                markdown_table(
                    group_descriptions,
                    [
                        "metric",
                        "direction",
                        "plain_description",
                        "paper_formula",
                        "notes",
                    ],
                ),
                "",
                "**Wartości dla ścieżek**",
                "",
                markdown_table(display_metrics, [*IDENTIFIER_COLUMNS, *group_metrics]),
                "",
            ]
        )

    notes = best_path_notes(metrics, descriptions)
    if notes:
        lines.extend(["## Szybkie Wskazówki Selekcji", "", *notes, ""])

    lines.extend(
        [
            "## Uwagi",
            "",
            "- Metryki `lower_better` są kryteriami kosztowymi.",
            "- Metryki `higher_better` są kryteriami korzyści.",
            "- Opisy metryk są umieszczone bezpośrednio przed tabelą wartości dla danej grupy.",
            "- Jeśli raport używa próbki, tabele ścieżek pokazują tylko wylosowane rekordy; podsumowania metod liczone są z pełnego CSV.",
            "- Puste komórki oznaczają, że wymagane kolumny źródłowe nie były dostępne w wejściowym `paths.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="CSV generated by python/path_metrics.py")
    parser.add_argument("--descriptions", default=str(DEFAULT_DESCRIPTION_PATH))
    parser.add_argument("--method-summary", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--experiment-params", default=None)
    parser.add_argument("--title", default="Raport Metryk Ścieżek DEA")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-paths", type=int, default=None, help="Show at most this many random paths in path tables.")
    parser.add_argument("--sample-random-state", type=int, default=42, help="Random seed used with --sample-paths.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics)
    descriptions_path = Path(args.descriptions)
    output_path = Path(args.output) if args.output else metrics_path.with_suffix(".md")
    method_summary_path = Path(args.method_summary) if args.method_summary else None
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    explicit_params_path = Path(args.experiment_params) if args.experiment_params else None

    metrics = pd.read_csv(metrics_path)
    display_metrics = metrics
    if args.sample_paths is not None and args.sample_paths > 0 and len(metrics) > args.sample_paths:
        display_metrics = metrics.sample(n=args.sample_paths, random_state=args.sample_random_state).sort_index()
    descriptions = pd.read_csv(descriptions_path)
    method_summary = pd.read_csv(method_summary_path) if method_summary_path else None
    method_status = discover_method_status(experiment_dir)
    if explicit_params_path:
        experiment_params = pd.read_csv(explicit_params_path, dtype=str)
    else:
        experiment_params = read_experiment_params(experiment_dir)

    report = render_experiment_report(
        metrics=metrics,
        descriptions=descriptions,
        source_name=str(metrics_path),
        method_summary=method_summary,
        method_status=method_status,
        experiment_params=experiment_params,
        title=args.title,
        display_metrics=display_metrics,
        sample_limit=args.sample_paths,
        sample_random_state=args.sample_random_state if len(display_metrics) != len(metrics) else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote Markdown report: {output_path}")


if __name__ == "__main__":
    main()
