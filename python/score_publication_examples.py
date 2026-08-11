import argparse
import math
from pathlib import Path

import pandas as pd


COORD_COLUMNS = ["i1", "i2", "o1"]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def extract_stages(path_row: pd.Series) -> list[dict]:
    stages = []
    stage_idx = 0
    while f"stage_{stage_idx:02d}_name" in path_row.index:
        prefix = f"stage_{stage_idx:02d}_"
        stage = {
            column.removeprefix(prefix): value
            for column, value in path_row.items()
            if column.startswith(prefix)
        }
        if stage.get("name"):
            stages.append(stage)
        stage_idx += 1
    return stages


def vector_angle(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def path_geometry(path_row: pd.Series, input_frame: pd.DataFrame) -> dict:
    stages = extract_stages(path_row)
    start = stages[0]
    capacities = {
        "i1": max(float(start["i1"]) - float(input_frame["i1"].min()), 1e-12),
        "i2": max(float(start["i2"]) - float(input_frame["i2"].min()), 1e-12),
        "o1": max(float(input_frame["o1"].max()) - float(start["o1"]), 1e-12),
    }

    step_vectors = []
    for previous, current in zip(stages, stages[1:]):
        step_vectors.append(
            [
                max(0.0, float(previous["i1"]) - float(current["i1"])) / capacities["i1"],
                max(0.0, float(previous["i2"]) - float(current["i2"])) / capacities["i2"],
                max(0.0, float(current["o1"]) - float(previous["o1"])) / capacities["o1"],
            ]
        )

    totals = [sum(vector[idx] for vector in step_vectors) for idx in range(3)]
    total_movement = sum(totals)
    shares = [value / total_movement if total_movement > 1e-12 else 0.0 for value in totals]
    axis_coverage = sum(value >= 0.05 for value in totals)
    movement_balance = 1.0 - max(shares) if shares else 0.0
    angles = [vector_angle(left, right) for left, right in zip(step_vectors, step_vectors[1:])]
    max_turn_angle = max(angles, default=0.0)
    dominant_dimensions = {
        max(range(3), key=lambda idx: vector[idx])
        for vector in step_vectors
        if sum(vector) > 1e-12
    }

    coordinate_keys = [
        tuple(round(float(stage[column]), 10) for column in COORD_COLUMNS)
        for stage in stages
    ]
    repeated_state_count = len(coordinate_keys) - len(set(coordinate_keys))

    visual_score = (
        2.0 * axis_coverage
        + 3.0 * movement_balance
        + 1.5 * min(max_turn_angle / 90.0, 1.0)
        + 0.5 * len(dominant_dimensions)
        - 2.0 * repeated_state_count
    )
    return {
        "visual_score": visual_score,
        "axis_coverage": axis_coverage,
        "movement_balance": movement_balance,
        "max_turn_angle_deg": max_turn_angle,
        "dominant_dimension_count": len(dominant_dimensions),
        "repeated_state_count_geometry": repeated_state_count,
        "i1_movement_share": shares[0],
        "i2_movement_share": shares[1],
        "o1_movement_share": shares[2],
    }


def stage_counts(run_dir: Path) -> dict[int, int]:
    frame = read_csv(run_dir / "stage_candidates.csv")
    if frame.empty or "stage" not in frame.columns:
        return {}
    return {
        int(stage): int(count)
        for stage, count in frame.groupby("stage").size().items()
    }


def score_runs(root: Path, input_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for paths_file in sorted(root.rglob("paths.csv")):
        run_dir = paths_file.parent
        paths = read_csv(paths_file)
        if paths.empty:
            continue
        metrics = read_csv(run_dir / "path_metrics.csv")
        metrics_by_path = metrics.set_index("path_id") if not metrics.empty else pd.DataFrame()
        counts = stage_counts(run_dir)
        relative = run_dir.relative_to(root)
        scenario = relative.parts[0]
        method = relative.parts[-2] if len(relative.parts) >= 2 else ""

        for _, path_row in paths.iterrows():
            path_id = str(path_row["path_id"])
            geometry = path_geometry(path_row, input_frame)
            row = {
                "scenario": scenario,
                "method": method,
                "run_dir": str(run_dir),
                "path_id": path_id,
                "stage_candidate_min": min(counts.values(), default=0),
                "stage_candidate_mean": (
                    sum(counts.values()) / len(counts) if counts else 0.0
                ),
                **geometry,
            }
            row["visual_score"] += min(row["stage_candidate_min"] / 10.0, 1.0)
            if not metrics.empty and path_id in metrics_by_path.index:
                metric_row = metrics_by_path.loc[path_id]
                if isinstance(metric_row, pd.DataFrame):
                    metric_row = metric_row.iloc[0]
                for column in [
                    "tc",
                    "msc",
                    "dr",
                    "mcp",
                    "total_input_reduction",
                    "total_output_increase",
                    "best_efficiency_improvement",
                    "attainable_transition_violations",
                    "total_milestone_gap",
                ]:
                    if column in metric_row.index:
                        row[column] = metric_row[column]
            milestone_gap = float(row.get("total_milestone_gap", 0.0) or 0.0)
            violations = float(row.get("attainable_transition_violations", 0.0) or 0.0)
            row["selection_score"] = row["visual_score"] - 10.0 * milestone_gap - 2.0 * violations
            rows.append(row)
    return pd.DataFrame(rows)


def format_number(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join(format_number(row.get(column, "")) for column in columns) + " |"
        )
    return "\n".join(lines)


def write_report(scores: pd.DataFrame, output_path: Path):
    ranked = scores.sort_values(
        ["selection_score", "visual_score", "movement_balance", "max_turn_angle_deg"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    scenario_best = ranked.groupby("scenario", as_index=False).first()
    best = ranked.iloc[0]

    scenario_columns = [
        "scenario",
        "path_id",
        "selection_score",
        "visual_score",
        "axis_coverage",
        "movement_balance",
        "max_turn_angle_deg",
        "dominant_dimension_count",
        "stage_candidate_min",
    ]
    path_columns = [
        "scenario",
        "path_id",
        "selection_score",
        "visual_score",
        "i1_movement_share",
        "i2_movement_share",
        "o1_movement_share",
        "max_turn_angle_deg",
        "tc",
        "msc",
    ]
    lines = [
        "# Screening przykładu publikacyjnego",
        "",
        "## Kryterium wyboru",
        "",
        "Wynik premiuje zmianę wszystkich trzech współrzędnych, zbilansowany udział osi, "
        "widoczne załamanie ścieżki, różne dominujące kierunki etapów oraz wystarczającą "
        "liczbę niezdominowanych kandydatów. Powtórzone stany, naruszenia osiągalności "
        "i odchylenie od kamieni milowych są karane.",
        "",
        "## Najlepszy wariant każdego scenariusza",
        "",
        markdown_table(scenario_best, scenario_columns),
        "",
        "## Najlepsze ścieżki ogółem",
        "",
        markdown_table(ranked.head(10), path_columns),
        "",
        "## Wybrany przypadek",
        "",
        f"Najwyższy wynik uzyskał `{best['scenario']}` / `{best['path_id']}` "
        f"z wynikiem wyboru `{best['selection_score']:.4f}` "
        f"i wynikiem geometrycznym `{best['visual_score']:.4f}`.",
        "",
        f"Katalog przebiegu: `{best['run_dir']}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    input_frame = pd.read_csv(args.input)
    scores = score_runs(root, input_frame)
    if scores.empty:
        raise ValueError(f"No complete paths found below {root}")
    scores = scores.sort_values("selection_score", ascending=False).reset_index(drop=True)
    output_csv = Path(args.output_csv).resolve()
    output_md = Path(args.output_md).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_csv, index=False)
    write_report(scores, output_md)
    print(f"Scored paths: {len(scores)}")
    print(f"CSV: {output_csv}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
