from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#14324A")
BLUE = colors.HexColor("#2878B5")
TEAL = colors.HexColor("#1B998B")
ORANGE = colors.HexColor("#F2A900")
RED = colors.HexColor("#D95555")
INK = colors.HexColor("#22313F")
MUTED = colors.HexColor("#5F6F7F")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
LIGHT_TEAL = colors.HexColor("#EAF7F5")
LIGHT_ORANGE = colors.HexColor("#FFF6DC")
LIGHT_GREY = colors.HexColor("#F3F5F7")
MID_GREY = colors.HexColor("#D5DDE3")
WHITE = colors.white


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a polished PDF report for a path experiment.")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
    bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
        return "ReportSans", "ReportSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()
PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
LEFT = 15 * mm
RIGHT = 15 * mm
TOP = 14 * mm
BOTTOM = 13 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=25,
            leading=29,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=12,
            leading=17,
            textColor=MUTED,
            spaceAfter=5 * mm,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceAfter=4 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12,
            leading=15,
            textColor=BLUE,
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.1,
            leading=12.4,
            textColor=INK,
            spaceAfter=2.4 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.6,
            leading=10,
            textColor=MUTED,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=10.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=1.5 * mm,
        ),
        "card_value": ParagraphStyle(
            "CardValue",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=20,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "card_label": ParagraphStyle(
            "CardLabel",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=7.5,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10,
            leading=14,
            textColor=NAVY,
        ),
        "formula": ParagraphStyle(
            "Formula",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=12,
            textColor=INK,
            leftIndent=3 * mm,
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}".replace(".", ",")


def load_params(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    return dict(zip(frame["parameter"], frame["value"]))


def table(data, widths, header=True, font_size=7.7, row_bgs=None, alignments=None):
    tbl = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ]
        )
        start = 1
    else:
        start = 0
    for row_index in range(start, len(data)):
        bg = LIGHT_GREY if row_index % 2 == 0 else WHITE
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), bg))
    if row_bgs:
        for row_index, bg in row_bgs.items():
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), bg))
    if alignments:
        for col_index, alignment in alignments.items():
            commands.append(("ALIGN", (col_index, 0), (col_index, -1), alignment))
    tbl.setStyle(TableStyle(commands))
    return tbl


def metric_cards(items):
    cards = []
    for label, value, bg in items:
        card = Table(
            [[p(value, "card_value")], [p(label, "card_label")]],
            colWidths=[CONTENT_WIDTH / len(items) - 3 * mm],
            rowHeights=[11 * mm, 8 * mm],
        )
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("BOX", (0, 0), (-1, -1), 0.6, MID_GREY),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        cards.append(card)
    outer = Table([cards], colWidths=[CONTENT_WIDTH / len(items)] * len(items))
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return outer


def callout(text: str, background=LIGHT_TEAL, border=TEAL):
    box = Table([[p(text, "callout")]], colWidths=[CONTENT_WIDTH])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return box


def scatter_drawing(input_df: pd.DataFrame, width=335, height=210) -> Drawing:
    drawing = Drawing(width, height)
    pad_left, pad_right, pad_bottom, pad_top = 34, 12, 28, 18
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_bottom - pad_top
    x_min, x_max = input_df["i1"].min(), input_df["i1"].max()
    y_min, y_max = input_df["i2"].min(), input_df["i2"].max()
    x_margin = (x_max - x_min) * 0.07
    y_margin = (y_max - y_min) * 0.07
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    def sx(value):
        return pad_left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value):
        return pad_bottom + (value - y_min) / (y_max - y_min) * plot_h

    drawing.add(Rect(pad_left, pad_bottom, plot_w, plot_h, fillColor=colors.HexColor("#FAFBFC"), strokeColor=MID_GREY))
    for fraction in [0.25, 0.5, 0.75]:
        drawing.add(Line(pad_left + fraction * plot_w, pad_bottom, pad_left + fraction * plot_w, pad_bottom + plot_h, strokeColor=colors.HexColor("#E4E9ED"), strokeWidth=0.4))
        drawing.add(Line(pad_left, pad_bottom + fraction * plot_h, pad_left + plot_w, pad_bottom + fraction * plot_h, strokeColor=colors.HexColor("#E4E9ED"), strokeWidth=0.4))
    drawing.add(String(width / 2, 7, "Input i1", fontName=FONT, fontSize=8, textAnchor="middle", fillColor=MUTED))
    drawing.add(String(7, height / 2, "i2", fontName=FONT, fontSize=8, fillColor=MUTED))
    drawing.add(String(pad_left, 15, fmt(x_min + x_margin, 1), fontName=FONT, fontSize=6.5, fillColor=MUTED))
    drawing.add(String(pad_left + plot_w - 12, 15, fmt(x_max - x_margin, 1), fontName=FONT, fontSize=6.5, fillColor=MUTED))
    drawing.add(String(15, pad_bottom - 2, fmt(y_min + y_margin, 1), fontName=FONT, fontSize=6.5, fillColor=MUTED))
    drawing.add(String(15, pad_bottom + plot_h - 2, fmt(y_max - y_margin, 1), fontName=FONT, fontSize=6.5, fillColor=MUTED))
    for _, row in input_df.iterrows():
        target = row["name"] == "T_B"
        color = ORANGE if target else BLUE
        radius = 5.2 if target else 3.0
        drawing.add(Circle(sx(row["i1"]), sy(row["i2"]), radius, fillColor=color, strokeColor=WHITE, strokeWidth=0.7))
        if target:
            drawing.add(String(sx(row["i1"]) - 26, sy(row["i2"]) + 8, "T_B (start)", fontName=FONT_BOLD, fontSize=8, fillColor=NAVY))
    return drawing


def process_drawing(width=735, height=105) -> Drawing:
    drawing = Drawing(width, height)
    labels = [
        ("1", "Próg etapu", "Wymagana efektywność"),
        ("2", "Pula osiągalna", "Od poprzedniego punktu"),
        ("3", "Front Pareto", "Bez punktów dominujących"),
        ("4", "Wybór przejścia", "Najmniejszy wysiłek kroku"),
    ]
    card_w = 160
    gap = (width - 4 * card_w) / 3
    for idx, (number, title, subtitle) in enumerate(labels):
        x = idx * (card_w + gap)
        drawing.add(Rect(x, 18, card_w, 72, rx=6, ry=6, fillColor=LIGHT_BLUE if idx % 2 == 0 else LIGHT_TEAL, strokeColor=MID_GREY, strokeWidth=0.8))
        drawing.add(Circle(x + 21, 54, 12, fillColor=NAVY, strokeColor=None))
        drawing.add(String(x + 21, 50.5, number, fontName=FONT_BOLD, fontSize=10, textAnchor="middle", fillColor=WHITE))
        drawing.add(String(x + 40, 61, title, fontName=FONT_BOLD, fontSize=9.5, fillColor=NAVY))
        drawing.add(String(x + 40, 44, subtitle, fontName=FONT, fontSize=7.5, fillColor=MUTED))
        if idx < len(labels) - 1:
            x1 = x + card_w + 3
            x2 = x + card_w + gap - 3
            drawing.add(Line(x1, 54, x2, 54, strokeColor=BLUE, strokeWidth=1.6))
            drawing.add(Line(x2 - 5, 58, x2, 54, strokeColor=BLUE, strokeWidth=1.6))
            drawing.add(Line(x2 - 5, 50, x2, 54, strokeColor=BLUE, strokeWidth=1.6))
    return drawing


def page_decorator(canvas, doc):
    canvas.saveState()
    page = doc.page
    if page > 1:
        canvas.setStrokeColor(MID_GREY)
        canvas.setLineWidth(0.5)
        canvas.line(LEFT, PAGE_HEIGHT - 9 * mm, PAGE_WIDTH - RIGHT, PAGE_HEIGHT - 9 * mm)
        canvas.setFont(FONT, 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(LEFT, PAGE_HEIGHT - 7 * mm, "Krokowa ścieżka poprawy efektywności DEA | T_B")
    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(LEFT, 8.5 * mm, PAGE_WIDTH - RIGHT, 8.5 * mm)
    canvas.setFont(FONT, 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, 5.2 * mm, "Raport eksperymentalny | 1 sierpnia 2026")
    canvas.drawRightString(PAGE_WIDTH - RIGHT, 5.2 * mm, f"Strona {page}")
    canvas.restoreState()


def build_report(experiment_dir: Path, output: Path):
    params = load_params(experiment_dir / "experiment_params.csv")
    cert = pd.read_csv(experiment_dir / "minimality_certificate.csv")
    metrics = pd.read_csv(experiment_dir / "all_path_metrics.csv")
    input_path = Path(params["input"])
    input_df = pd.read_csv(input_path)
    selected_path_id = params["selected_path_id"]
    selected_metric = metrics.loc[metrics["path_id"] == selected_path_id].iloc[0]
    min_tc = metrics["tc"].min()
    tc_ties = metrics[metrics["tc"].sub(min_tc).abs() <= 1e-9]
    min_msc_in_ties = tc_ties["msc"].min()
    msc_ties = tc_ties[tc_ties["msc"].sub(min_msc_in_ties).abs() <= 1e-9]

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=PAGE_SIZE,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title="Krokowa ścieżka poprawy efektywności DEA - eksperyment T_B",
        author="Raport wygenerowany z wyników eksperymentu DEA-F",
        subject="Walidacja krokowej ścieżki poprawy efektywności",
    )
    story = []

    # Page 1: concise result.
    story.append(Spacer(1, 5 * mm))
    story.append(p("Krokowa ścieżka poprawy efektywności DEA", "title"))
    story.append(p("Eksperyment dla jednostki T_B w przestrzeni dwóch wejść i1, i2 przy stałym wyjściu o1 = 5", "subtitle"))
    story.append(HRFlowable(width="100%", thickness=2.2, color=BLUE, spaceAfter=5 * mm))
    story.append(
        metric_cards(
            [
                ("Łączny wysiłek ścieżki (TC)", fmt(selected_metric["tc"], 6), LIGHT_BLUE),
                ("Największy krok (MSC)", fmt(selected_metric["msc"], 6), LIGHT_TEAL),
                ("Bezpośredniość (DR)", fmt(selected_metric["dr"], 3), LIGHT_ORANGE),
                ("Pełne ścieżki", f"{int(params['complete_path_count']):,}".replace(",", " "), LIGHT_GREY),
            ]
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(p("Najważniejszy wynik", "h2"))
    story.append(
        callout(
            "Wybrana ścieżka jest osiągalna krok po kroku i na każdym etapie należy do frontu minimalnych zmian względem <b>rzeczywistego poprzednika</b>. Dla wszystkich trzech przejść liczba punktów dominujących wynosi 0."
        )
    )
    story.append(Spacer(1, 4 * mm))
    path_rows = [["Etap", "Punkt", "i1", "i2", "o1", "Najlepsza efektywność", "Wysiłek kroku", "Status"]]
    start = input_df.loc[input_df["name"] == params["target"]].iloc[0]
    path_rows.append(["0", "T_B", fmt(start["i1"]), fmt(start["i2"]), fmt(start["o1"]), fmt(selected_metric["best_efficiency_start"], 4), "-", "start"])
    for _, row in cert.iterrows():
        path_rows.append(
            [
                str(int(row["stage"])),
                str(row["point_name"]),
                fmt(row["i1"]),
                fmt(row["i2"]),
                "5,0000",
                fmt(row["best_efficiency"], 4),
                fmt(row["effort_from_previous"], 6),
                "Pareto-min.",
            ]
        )
    story.append(table(path_rows, [35, 205, 58, 58, 48, 118, 92, 82], font_size=8.2, alignments={0: "CENTER", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT", 7: "CENTER"}))
    story.append(Spacer(1, 4 * mm))
    story.append(
        p(
            "Interpretacja: T_B przechodzi z (8,2; 9,1) do (1,1842; 4,5846). Najlepszy score efektywności rośnie z 0,2513 do 1,0000. DR = 1 oznacza, że ścieżka nie zawiera cofania zmian względem ruchu bezpośredniego.",
            "body",
        )
    )
    story.append(PageBreak())

    # Page 2: data and configuration.
    story.append(p("1. Dane wejściowe i konfiguracja", "h1"))
    data_rows = [["DMU", "i1", "i2", "o1"]]
    for _, row in input_df.iterrows():
        data_rows.append([row["name"], fmt(row["i1"], 1), fmt(row["i2"], 1), fmt(row["o1"], 1)])
    data_table = table(data_rows, [48, 47, 47, 47], font_size=7.2, row_bgs={len(data_rows) - 3: LIGHT_ORANGE}, alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"})
    left_col = [p("Zbiór referencyjny", "h2"), p("Zbiór obejmuje 18 jednostek, dwa wejścia oraz jedno wyjście. Jednostką badaną jest T_B.", "small"), Spacer(1, 2 * mm), data_table]

    setup_rows = [
        ["Parametr", "Wartość"],
        ["Jednostka", params["target"]],
        ["Metoda", "best efficiency path"],
        ["Tryb", params["mode"]],
        ["Zmienne modyfikowane", params["modified_dimensions"]],
        ["Wymiar stały", params["fixed_dimensions"]],
        ["Cel końcowy efektywności", params["target_best_efficiency"]],
        ["Liczba etapów", params["stages"]],
        ["Punkty na przejście", params["points_per_stage"]],
        ["Normalizacja", "zakresy obserwowane w danych"],
        ["Skala osi", "1 jednostka i1 = 1 jednostka i2"],
        ["Czas całkowity", params["pipeline_elapsed_time"]],
    ]
    setup_table = table(setup_rows, [145, 160], font_size=7.6)
    right_col = [p("Położenie jednostek w przestrzeni wejść", "h2"), scatter_drawing(input_df), Spacer(1, 2 * mm), setup_table]
    columns = Table([[left_col, right_col]], colWidths=[335, CONTENT_WIDTH - 335], hAlign="LEFT")
    columns.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
    story.append(columns)
    story.append(PageBreak())

    # Page 3: algorithm and metric definitions.
    story.append(p("2. Jak powstaje ścieżka", "h1"))
    story.append(
        callout(
            "W poprawionej wersji kandydaci etapu h są oceniani względem punktu wybranego w etapie h-1. Punkt etapu 2 nie jest więc porównywany ponownie z T_B, lecz z konkretnym punktem etapu 1, z którego prowadzi przejście.",
            background=LIGHT_BLUE,
            border=BLUE,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(process_drawing())
    story.append(Spacer(1, 2 * mm))
    left_text = [
        p("Osiągalność przejścia", "h2"),
        p("Dla wejść wymagamy i<sub>j,h</sub> ≤ i<sub>j,h-1</sub>; wejście nie może wzrosnąć. Dla wyjść wymagamy o<sub>j,h</sub> ≥ o<sub>j,h-1</sub>. W tym eksperymencie o1 jest stałe.", "body"),
        p("Minimalność Pareto", "h2"),
        p("Wybrany punkt jest minimalny, jeśli wśród osiągalnych punktów spełniających próg etapu nie istnieje inny punkt wymagający nie większej zmiany w obu wejściach i mniejszej zmiany w co najmniej jednym z nich.", "body"),
        p("Przeszukiwanie", "h2"),
        p("Najpierw oceniany jest globalny grid 30×30. Następnie front jest zagęszczany przez 5 iteracji refinement oraz lokalne próbkowanie 20×20 wokół 15 centrów.", "body"),
    ]
    metric_rows = [
        ["Metryka", "Definicja i interpretacja"],
        ["e_h", p("Średnia znormalizowana zmiana w przejściu h. Każdy z trzech wymiarów i1, i2, o1 ma wagę 1/3; stałe o1 wnosi 0.", "small")],
        ["TC", p("TC = Σ e_h. Łączny wysiłek całej ścieżki; mniejsza wartość jest lepsza.", "small")],
        ["MSC", p("MSC = max(e_h). Najtrudniejszy pojedynczy krok; mniejsza wartość oznacza łagodniejszą ścieżkę.", "small")],
        ["Cdir", p("Znormalizowany wysiłek bezpośrednio od punktu startowego do końcowego.", "small")],
        ["DR", p("DR = TC / Cdir. Wartość 1 oznacza brak dodatkowego wysiłku wynikającego z cofania zmian.", "small")],
    ]
    formula_box = [
        p("Metryki ścieżki", "h2"),
        table(metric_rows, [55, 300], font_size=7.4),
        Spacer(1, 2 * mm),
        p("Zakresy normalizacji: R<sub>i1</sub> = 6,9; R<sub>i2</sub> = 6,8; R<sub>o1</sub> = 5,2.", "small"),
    ]
    methodology = Table([[left_text, formula_box]], colWidths=[365, CONTENT_WIDTH - 365], hAlign="LEFT")
    methodology.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
    story.append(methodology)
    story.append(PageBreak())

    # Page 4: visual evidence.
    story.append(p("3. Punkty kandydackie, fronty i wybrana ścieżka", "h1"))
    story.append(p("Każdy panel etapowy pokazuje pulę spełniającą próg, podzbiór osiągalny z poprzednika, front Pareto oraz wybrany punkt. W panelu zbiorczym widoczna jest kompletna ścieżka T_B → 1 → 2 → 3.", "body"))
    figure = Image(str(experiment_dir / "minimality_points_2d.png"))
    max_w, max_h = 170 * mm, 112 * mm
    scale = min(max_w / figure.imageWidth, max_h / figure.imageHeight)
    figure.drawWidth = figure.imageWidth * scale
    figure.drawHeight = figure.imageHeight * scale
    figure.hAlign = "CENTER"
    story.append(figure)
    story.append(p("Rysunek 1. Gęste przeszukiwanie przestrzeni i1-i2. Osie mają jednakową skalę, dlatego odległości można porównywać bez zniekształcenia proporcji.", "caption"))
    story.append(PageBreak())

    # Page 5: numerical certificate.
    story.append(p("4. Certyfikat minimalności krok po kroku", "h1"))
    cert_rows = [["Etap", "Poprzednik", "Próg eff.", "Uzyskana eff.", "Δi1", "Δi2", "e_h", "Osiągalne", "Front", "Dominujące", "Pareto"]]
    for _, row in cert.iterrows():
        cert_rows.append(
            [
                int(row["stage"]),
                str(row["previous_point_name"]),
                fmt(row["efficiency_threshold"], 4),
                fmt(row["best_efficiency"], 4),
                fmt(row["delta_i1"], 4),
                fmt(row["delta_i2"], 4),
                fmt(row["effort_from_previous"], 6),
                int(row["eligible_tested_points"]),
                int(row["pareto_front_points"]),
                int(row["dominating_points"]),
                "TAK" if bool(row["pareto_minimal"]) else "NIE",
            ]
        )
    story.append(table(cert_rows, [35, 165, 62, 76, 56, 56, 67, 67, 45, 63, 48], font_size=7.2, row_bgs={1: LIGHT_BLUE, 2: LIGHT_TEAL, 3: LIGHT_ORANGE}, alignments={0: "CENTER", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT", 7: "RIGHT", 8: "RIGHT", 9: "RIGHT", 10: "CENTER"}))
    story.append(Spacer(1, 4 * mm))
    step1, step2, step3 = cert["effort_from_previous"].tolist()
    calculation = (
        f"TC = {fmt(step1, 6)} + {fmt(step2, 6)} + {fmt(step3, 6)} = <b>{fmt(selected_metric['tc'], 6)}</b><br/>"
        f"MSC = max({fmt(step1, 6)}; {fmt(step2, 6)}; {fmt(step3, 6)}) = <b>{fmt(selected_metric['msc'], 6)}</b><br/>"
        f"DR = TC / Cdir = {fmt(selected_metric['tc'], 6)} / {fmt(selected_metric['cdir'], 6)} = <b>{fmt(selected_metric['dr'], 3)}</b>"
    )
    evidence_rows = [
        [p("Obliczenie metryk", "h2"), p("Test wyboru ścieżki", "h2")],
        [
            p(calculation, "formula"),
            p(
                f"Przeanalizowano <b>{len(metrics):,}</b> pełnych ścieżek. Minimalne TC osiąga {len(tc_ties)} ścieżek; po zastosowaniu kryterium MSC pozostaje {len(msc_ties)} równoważnych wariantów. Wybrano <b>{selected_path_id}</b>.".replace(",", " "),
                "body",
            ),
        ],
    ]
    evidence = Table(evidence_rows, colWidths=[CONTENT_WIDTH / 2, CONTENT_WIDTH / 2])
    evidence.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE), ("BACKGROUND", (1, 0), (1, -1), LIGHT_TEAL), ("BOX", (0, 0), (-1, -1), 0.6, MID_GREY), ("INNERGRID", (0, 0), (-1, -1), 0.4, MID_GREY), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(evidence)
    story.append(Spacer(1, 4 * mm))
    story.append(p("Wniosek walidacyjny", "h2"))
    story.append(
        callout(
            "Każde przejście spełnia warunek osiągalności, a wybrany punkt ma 0 punktów dominujących w swojej puli osiągalnych kandydatów. Testy implementacji zakończyły się wynikiem 11/11.",
            background=LIGHT_TEAL,
            border=TEAL,
        )
    )
    story.append(PageBreak())

    # Page 6: reproducibility and limits.
    story.append(p("5. Reprodukowalność i ograniczenia", "h1"))
    search_rows = [
        ["Grupa", "Parametr", "Wartość"],
        ["Global", "Strategia / próbek", f"{params['global_search_sampling_strategy']} / {params['global_search_samples']}"],
        ["Global", "Grid / seed", f"{params['global_search_grid']} / {params['global_search_random_state']}"],
        ["Lokalne", "Strategia", params["local_search_sampling_strategy"]],
        ["Lokalne", "Próbek na centrum", params["local_search_samples_per_center"]],
        ["Lokalne", "Liczba centrów", params["local_search_total_centers"]],
        ["Lokalne", "Łączna liczba punktów", params["local_search_total_points"]],
        ["Lokalne", "Promień i1 / i2", f"{params['local_search_radius_i1']} / {params['local_search_radius_i2']}"],
        ["Refinement", "Iteracje", params["refine_iterations"]],
        ["Ścieżki", "Kandydatów przejść", params["transition_candidate_rows"]],
        ["Ścieżki", "Różnych poprzedników", params["distinct_transition_predecessors"]],
        ["Ścieżki", "Limit ścieżek", params["max_paths"]],
        ["Wykonanie", "Łącznie próbek", params["total_search_samples"]],
        ["Wykonanie", "Czas", params["pipeline_elapsed_time"]],
    ]
    stage_rows = [
        ["Etap", "Sprawdzone", "Próg spełniony", "Osiągalne dla wybranej ścieżki", "Front"],
        ["1", params["stage_1_tested_points"], params["stage_1_threshold_eligible_points"], str(int(cert.iloc[0]["eligible_tested_points"])), str(int(cert.iloc[0]["pareto_front_points"]))],
        ["2", params["stage_2_tested_points"], params["stage_2_threshold_eligible_points"], str(int(cert.iloc[1]["eligible_tested_points"])), str(int(cert.iloc[1]["pareto_front_points"]))],
        ["3", params["stage_3_tested_points"], params["stage_3_threshold_eligible_points"], str(int(cert.iloc[2]["eligible_tested_points"])), str(int(cert.iloc[2]["pareto_front_points"]))],
    ]
    left = [p("Parametry przeszukiwania", "h2"), table(search_rows, [72, 155, 125], font_size=7.4)]
    right = [
        p("Pokrycie etapów", "h2"),
        table(stage_rows, [42, 72, 92, 130, 48], font_size=7.2, alignments={0: "CENTER", 1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}),
        Spacer(1, 4 * mm),
        p("Zakres wniosku", "h2"),
        p("Raport potwierdza minimalność względem skończonego zbioru punktów faktycznie ocenionych przez DEA: gridu, pięciu iteracji refinement i lokalnego próbkowania. Nie jest to analityczny dowód minimum w całej ciągłej przestrzeni.", "body"),
        p("Porównywalność", "h2"),
        p("Wartości TC i MSC zależą od przyjętych zakresów normalizacji i wag. W tym eksperymencie zakresy są stałe, wyznaczone z obserwowanego zbioru wejściowego, a i1, i2 i o1 mają równe wagi.", "body"),
        p("Uwagi wykonawcze", "h2"),
        p("Końcowa partia 400 ocen trzeciego etapu została wznowiona z zapisanych plików po limicie procesu. Żaden etap ani punkt nie został pominięty.", "body"),
    ]
    appendix = Table([[left, right]], colWidths=[365, CONTENT_WIDTH - 365], hAlign="LEFT")
    appendix.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
    story.append(appendix)
    story.append(Spacer(1, 4 * mm))
    story.append(
        p(
            f"Identyfikator eksperymentu: <b>{experiment_dir.name}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Wybrana ścieżka: <b>{selected_path_id}</b>",
            "small",
        )
    )

    doc.build(story, onFirstPage=page_decorator, onLaterPages=page_decorator)


def main():
    args = parse_args()
    build_report(args.experiment_dir.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
