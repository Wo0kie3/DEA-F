import argparse
import unicodedata
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DEFAULT_DESCRIPTION_PATH = Path("templates/path_metric_descriptions.csv")
DEFAULT_FONT = Path(r"C:\Windows\Fonts\arial.ttf")
DEFAULT_BOLD_FONT = Path(r"C:\Windows\Fonts\arialbd.ttf")


def ascii_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def read_csv_or_empty(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_text_csv_or_empty(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def discover_method_status(experiment_dir: Path | None) -> pd.DataFrame:
    if experiment_dir is None:
        return pd.DataFrame()

    rows = []
    for metric_file in sorted(experiment_dir.rglob("path_metrics.csv")):
        frame = read_csv_or_empty(metric_file)
        count = len(frame)
        rows.append(
            {
                "method": metric_file.parent.parent.name,
                "run_dir": metric_file.parent.name,
                "path_count": count,
                "status": "OK" if count else "brak sciezek",
            }
        )
    return pd.DataFrame(rows)


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return ascii_text(value)
    try:
        numeric = float(value)
    except Exception:
        return ascii_text(value)
    if abs(numeric - round(numeric)) < 1e-10:
        return str(int(round(numeric)))
    return f"{numeric:.5g}"


def build_styles():
    pdfmetrics.registerFont(TTFont("Arial", str(DEFAULT_FONT)))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(DEFAULT_BOLD_FONT)))

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyPL",
            parent=styles["BodyText"],
            fontName="Arial",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitlePL",
            parent=styles["Title"],
            fontName="Arial-Bold",
            fontSize=16,
            leading=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingPL",
            parent=styles["Heading2"],
            fontName="Arial-Bold",
            fontSize=12,
            leading=14,
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallPL",
            parent=styles["BodyText"],
            fontName="Arial",
            fontSize=6,
            leading=7,
        )
    )
    return styles


def paragraph(value, styles, style_name="SmallPL"):
    text = fmt(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(text, styles[style_name])


def make_table(frame: pd.DataFrame, columns: list[str], styles, font_size=6):
    columns = [column for column in columns if column in frame.columns]
    data = [[paragraph(column, styles) for column in columns]]
    for _, row in frame[columns].iterrows():
        data.append([paragraph(row[column], styles) for column in columns])

    available_width = landscape(A4)[0] - 2 * cm
    col_width = available_width / max(len(columns), 1)
    table = Table(data, colWidths=[col_width] * len(columns), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Arial"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E2F3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FBFF")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def compact_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    return summary[[column for column in columns if column in summary.columns]]


def render_pdf(
    metrics: pd.DataFrame,
    descriptions: pd.DataFrame,
    output_pdf: Path,
    title: str,
    experiment_dir: Path | None = None,
    method_summary: pd.DataFrame | None = None,
    experiment_params: pd.DataFrame | None = None,
    display_metrics: pd.DataFrame | None = None,
    sample_limit: int | None = None,
    sample_random_state: int | None = None,
):
    styles = build_styles()
    display_metrics = metrics if display_metrics is None else display_metrics
    descriptions = descriptions.copy()
    for column in descriptions.columns:
        descriptions[column] = descriptions[column].map(ascii_text)

    story = [
        Paragraph(ascii_text(title), styles["TitlePL"]),
        Paragraph(ascii_text(f"Katalog eksperymentu: {experiment_dir or ''}"), styles["BodyPL"]),
        Paragraph(ascii_text(f"Liczba sciezek: {len(metrics)}"), styles["BodyPL"]),
    ]
    if len(display_metrics) != len(metrics):
        story.append(Paragraph(ascii_text(f"Liczba sciezek pokazywanych w tabelach: {len(display_metrics)}"), styles["BodyPL"]))
        if sample_limit is not None:
            story.append(Paragraph(ascii_text(f"Sposob prezentacji: losowa probka max {sample_limit} sciezek"), styles["BodyPL"]))
        if sample_random_state is not None:
            story.append(Paragraph(ascii_text(f"Ziarno losowania probki: {sample_random_state}"), styles["BodyPL"]))
    story.append(Spacer(1, 0.3 * cm))

    if experiment_params is not None and not experiment_params.empty:
        story.append(Paragraph("Parametry eksperymentu", styles["HeadingPL"]))
        story.append(make_table(experiment_params, ["parameter", "value"], styles, font_size=6.4))
        story.append(Spacer(1, 0.4 * cm))

    status = discover_method_status(experiment_dir)
    if not status.empty:
        story.append(Paragraph("Status metod", styles["HeadingPL"]))
        story.append(make_table(status, ["method", "run_dir", "path_count", "status"], styles, font_size=7))
        story.append(Spacer(1, 0.4 * cm))

    if method_summary is not None and not method_summary.empty:
        summary = compact_method_summary(method_summary)
        story.append(Paragraph("Podsumowanie metod", styles["HeadingPL"]))
        story.append(make_table(summary, summary.columns.tolist(), styles, font_size=5.7))
        story.append(PageBreak())

    id_cols = ["method", "path_id", "start_name", "final_name", "path_length"]
    for group, group_desc in descriptions.groupby("group", sort=False):
        group_metrics = [metric for metric in group_desc["metric"].tolist() if metric in metrics.columns]
        if not group_metrics:
            continue
        story.append(Paragraph(ascii_text(group), styles["HeadingPL"]))
        story.append(Paragraph("Opis metryk w tej grupie", styles["BodyPL"]))
        story.append(
            make_table(
                group_desc,
                ["metric", "direction", "plain_description", "paper_formula", "notes"],
                styles,
                font_size=5.2,
            )
        )
        story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph("Wartosci dla sciezek", styles["BodyPL"]))
        story.append(make_table(display_metrics, id_cols + group_metrics, styles, font_size=5.4))
        story.append(PageBreak())

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(A4),
        rightMargin=0.5 * cm,
        leftMargin=0.5 * cm,
        topMargin=0.5 * cm,
        bottomMargin=0.5 * cm,
    )
    doc.build(story)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--descriptions", default=str(DEFAULT_DESCRIPTION_PATH))
    parser.add_argument("--method-summary", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--experiment-params", default=None)
    parser.add_argument("--title", default="Raport eksperymentu DEA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-paths", type=int, default=None, help="Show at most this many random paths in path tables.")
    parser.add_argument("--sample-random-state", type=int, default=42, help="Random seed used with --sample-paths.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    display_metrics = metrics
    if args.sample_paths is not None and args.sample_paths > 0 and len(metrics) > args.sample_paths:
        display_metrics = metrics.sample(n=args.sample_paths, random_state=args.sample_random_state).sort_index()
    descriptions = pd.read_csv(args.descriptions)
    method_summary = read_csv_or_empty(Path(args.method_summary)) if args.method_summary else None
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    if args.experiment_params:
        experiment_params = read_text_csv_or_empty(Path(args.experiment_params))
    elif experiment_dir is not None:
        experiment_params = read_text_csv_or_empty(experiment_dir / "experiment_params.csv")
    else:
        experiment_params = None

    render_pdf(
        metrics=metrics,
        descriptions=descriptions,
        output_pdf=Path(args.output),
        title=args.title,
        experiment_dir=experiment_dir,
        method_summary=method_summary,
        experiment_params=experiment_params,
        display_metrics=display_metrics,
        sample_limit=args.sample_paths,
        sample_random_state=args.sample_random_state if len(display_metrics) != len(metrics) else None,
    )
    print(f"Wrote PDF report: {args.output}")


if __name__ == "__main__":
    main()
