import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--columns", required=True)
    parser.add_argument("--path-id", default=None)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def fmt(value, digits=4):
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def find_selected_path(metrics, requested_path_id, tolerance=1e-9):
    valid = metrics[
        pd.to_numeric(metrics["attainable_transition_violations"], errors="coerce").fillna(0) == 0
    ].copy()
    if valid.empty:
        raise ValueError("No attainable paths found.")

    if requested_path_id:
        selected = valid[valid["path_id"].astype(str) == str(requested_path_id)]
        if selected.empty:
            raise ValueError(f"Path {requested_path_id} not found.")
        return selected.iloc[0], False

    valid["_tc"] = pd.to_numeric(valid["tc"], errors="coerce")
    valid["_msc"] = pd.to_numeric(valid["msc"], errors="coerce")
    minimum_tc = valid["_tc"].min()
    tied = valid[(valid["_tc"] - minimum_tc).abs() <= tolerance].copy()
    tied = tied.sort_values(["_msc", "path_id"], ascending=[True, True])
    return tied.iloc[0], True


def load_selection_pool(run_dir, stage):
    stage_candidates_path = run_dir / "stage_candidates.csv"
    if stage_candidates_path.exists():
        stage_candidates = pd.read_csv(stage_candidates_path)
        if "stage" in stage_candidates.columns:
            stage_pool = stage_candidates[
                pd.to_numeric(stage_candidates["stage"], errors="coerce") == stage
            ].copy()
            if not stage_pool.empty:
                return stage_pool.drop_duplicates(
                    subset=["name"],
                    keep="last",
                ).reset_index(drop=True)

    prefix = f"stage_{stage:02d}_eff"
    files = [
        run_dir / f"{prefix}_refined_final_metrics.csv",
        run_dir / f"{prefix}_local_search_metrics.csv",
        run_dir / "global_search_metrics.csv",
    ]
    frames = [pd.read_csv(path) for path in files if path.exists()]
    if not frames:
        raise FileNotFoundError(f"No selection-pool files found for stage {stage}.")
    pool = pd.concat(frames, ignore_index=True, sort=False)
    return pool.drop_duplicates(subset=["name"], keep="last").reset_index(drop=True)


def dominating_mask(pool, selected, columns, tolerance):
    no_more_reduction = pd.Series(True, index=pool.index)
    strictly_less_change = pd.Series(False, index=pool.index)
    for column in columns:
        values = pd.to_numeric(pool[column], errors="coerce")
        selected_value = float(selected[column])
        no_more_reduction &= values + tolerance >= selected_value
        strictly_less_change |= values > selected_value + tolerance
    return no_more_reduction & strictly_less_change


def pareto_front_mask(pool, columns, tolerance):
    keep = []
    for _, candidate in pool.iterrows():
        keep.append(not dominating_mask(pool, candidate, columns, tolerance).any())
    return pd.Series(keep, index=pool.index)


def relative_reduction_sum(row, previous, columns):
    total = 0.0
    for column in columns:
        denominator = max(abs(float(previous[column])), 1e-12)
        total += max(0.0, float(previous[column]) - float(row[column])) / denominator
    return total * 100.0


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    columns = [part.strip() for part in args.columns.split(",") if part.strip()]
    if len(columns) != 2:
        raise ValueError("This compact certificate expects exactly two input columns.")

    metrics = pd.read_csv(experiment_dir / "all_path_metrics.csv")
    selected_metric, selected_by_min_tc = find_selected_path(
        metrics,
        args.path_id,
        args.tolerance,
    )
    run_dir = Path(selected_metric["run_dir"])
    if not run_dir.is_absolute():
        run_dir = experiment_dir / run_dir
    paths = pd.read_csv(run_dir / "paths.csv")
    path_row = paths[paths["path_id"].astype(str) == str(selected_metric["path_id"])]
    if path_row.empty:
        raise ValueError(f"Path {selected_metric['path_id']} not found in paths.csv.")
    path_row = path_row.iloc[0]

    milestones = pd.read_csv(run_dir / "efficiency_milestones.csv").set_index("stage")
    start = {
        column: float(path_row[f"stage_00_{column}"])
        for column in columns
    }

    stage_rows = []
    for stage in range(1, int(path_row["path_length"]) + 1):
        previous = {
            "name": path_row[f"stage_{stage - 1:02d}_name"],
            **{
                column: float(path_row[f"stage_{stage - 1:02d}_{column}"])
                for column in columns
            },
        }
        selected = {
            "name": path_row[f"stage_{stage:02d}_name"],
            "best_efficiency": float(path_row[f"stage_{stage:02d}_best_efficiency"]),
            **{
                column: float(path_row[f"stage_{stage:02d}_{column}"])
                for column in columns
            },
        }
        threshold = float(milestones.loc[stage, "milestone_best_efficiency"])
        pool = load_selection_pool(run_dir, stage)
        pool = pool[
            pd.to_numeric(pool["best_efficiency"], errors="coerce") + args.tolerance >= threshold
        ].copy()
        pool = pool.dropna(subset=columns)
        for column in columns:
            pool = pool[
                pd.to_numeric(pool[column], errors="coerce")
                <= previous[column] + args.tolerance
            ].copy()

        dominators = pool[dominating_mask(pool, selected, columns, args.tolerance)]
        front_mask = pareto_front_mask(pool, columns, args.tolerance)
        pool["_relative_reduction_sum_pct"] = pool.apply(
            lambda row: relative_reduction_sum(row, previous, columns),
            axis=1,
        )
        selected_reduction_sum = relative_reduction_sum(selected, previous, columns)
        scalar_rank = 1 + int(
            (pool["_relative_reduction_sum_pct"] < selected_reduction_sum - args.tolerance).sum()
        )

        stage_rows.append(
            {
                "stage": stage,
                "previous_point_name": previous["name"],
                "efficiency_threshold": threshold,
                "best_efficiency": selected["best_efficiency"],
                "point_name": selected["name"],
                columns[0]: selected[columns[0]],
                columns[1]: selected[columns[1]],
                f"delta_{columns[0]}": previous[columns[0]] - selected[columns[0]],
                f"delta_{columns[1]}": previous[columns[1]] - selected[columns[1]],
                "relative_reduction_sum_pct": selected_reduction_sum,
                "effort_from_previous": path_row.get(
                    f"stage_{stage:02d}_effort_from_previous",
                    None,
                ),
                "eligible_tested_points": len(pool),
                "pareto_front_points": int(front_mask.sum()),
                "dominating_points": len(dominators),
                "pareto_minimal": len(dominators) == 0,
                "scalar_effort_rank": scalar_rank,
            }
        )

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else Path(args.output_md).with_suffix(".csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stage_rows).to_csv(output_csv, index=False)

    table_rows = []
    for row in stage_rows:
        table_rows.append(
            {
                "etap": row["stage"],
                "od punktu": row["previous_point_name"],
                "prog eff.": fmt(row["efficiency_threshold"]),
                "uzyskana eff.": fmt(row["best_efficiency"]),
                columns[0]: fmt(row[columns[0]]),
                columns[1]: fmt(row[columns[1]]),
                f"redukcja {columns[0]}": fmt(row[f"delta_{columns[0]}"]),
                f"redukcja {columns[1]}": fmt(row[f"delta_{columns[1]}"]),
                "wysilek kroku": fmt(row["effort_from_previous"], 6),
                "sprawdzone": row["eligible_tested_points"],
                "punkty dominujace": row["dominating_points"],
                "Pareto-min.": "TAK" if row["pareto_minimal"] else "NIE",
                "ranking sumy redukcji": (
                    f"{row['scalar_effort_rank']}/{row['eligible_tested_points']}"
                ),
            }
        )

    min_tc = pd.to_numeric(metrics["tc"], errors="coerce").min()
    tc_ties = int(
        (
            pd.to_numeric(metrics["tc"], errors="coerce")
            .sub(min_tc)
            .abs()
            <= args.tolerance
        ).sum()
    )
    final_stage = stage_rows[-1]
    final_delta_parts = [
        f"{column}: {fmt(start[column])} - {fmt(final_stage[column])} = "
        f"{fmt(start[column] - final_stage[column])}"
        for column in columns
    ]

    selection_text = (
        "Sciezka zostala wybrana automatycznie: najpierw najmniejsze TC w granicy "
        "tolerancji, a przy remisie najmniejsze MSC."
        if selected_by_min_tc
        else "Sciezka zostala wskazana przez parametr --path-id."
    )
    parameter_lines = []
    parameters_path = experiment_dir / "experiment_params.csv"
    if parameters_path.exists():
        parameters = pd.read_csv(parameters_path)
        parameter_map = dict(zip(parameters["parameter"], parameters["value"]))
        parameter_keys = [
            "input",
            "target",
            "modified_dimensions",
            "target_best_efficiency",
            "stages",
            "points_per_stage",
            "points_per_stage_semantics",
            "transition_reference",
            "normalization_ranges",
            "global_search_sampling_strategy",
            "global_search_grid",
            "global_search_samples",
            "global_search_bounds_i1",
            "global_search_bounds_i2",
            "local_search_sampling_strategy",
            "local_search_grid_per_center",
            "local_search_samples_per_center",
            "local_search_total_centers",
            "local_search_total_points",
            "local_search_step_multiplier",
            "local_search_radius_i1",
            "local_search_radius_i2",
            "axis_scale",
            "complete_path_count",
            "pipeline_elapsed_time",
        ]
        parameter_rows = [
            {"parametr": key, "wartosc": parameter_map[key]}
            for key in parameter_keys
            if key in parameter_map
        ]
        if parameter_rows:
            parameter_lines = [
                "## Parametry probkowania",
                "",
                markdown_table(parameter_rows, ["parametr", "wartosc"]),
                "",
            ]

    lines = [
        "# Prosty certyfikat minimalnosci zmiany",
        "",
        f"- Eksperyment: `{experiment_dir}`",
        f"- Wybrana sciezka: `{selected_metric['path_id']}`",
        f"- Zmieniane wymiary: `{', '.join(columns)}`",
        f"- Staly output: `o1 = {fmt(path_row['stage_00_o1'])}`",
        f"- Punkt startowy: `{columns[0]}={fmt(start[columns[0]])}, "
        f"{columns[1]}={fmt(start[columns[1]])}`",
        "",
        *parameter_lines,
        *(
            [
                "## Wizualizacja 2D",
                "",
                f"![Punkty kandydackie, fronty Pareto i wybrana sciezka]({args.plot_path})",
                "",
            ]
            if args.plot_path
            else []
        ),
        "## Co znaczy minimalna zmiana",
        "",
        "Minimalnosc jest sprawdzana osobno dla kazdego przejscia `z_(h-1) -> z_h`. "
        "Do puli etapu trafiaja tylko punkty osiagalne z poprzednio wybranego punktu. "
        f"Dla inputow mniejsza zmiana oznacza pozostawienie wiekszej wartosci. "
        f"Punkt B dominowalby wybrany punkt A, gdyby `B.{columns[0]} >= A.{columns[0]}` "
        f"i `B.{columns[1]} >= A.{columns[1]}`, przy co najmniej jednej ostrej nierownosci, "
        "a jednoczesnie B osiagalby wymagany prog efektywnosci.",
        "",
        "Jezeli liczba punktow dominujacych wynosi 0, punkt jest minimalny w sensie Pareto "
        "wzgledem kandydatow ocenionych w danym etapie i osiagalnych z jego poprzednika.",
        "",
        "## Wynik etap po etapie",
        "",
        markdown_table(table_rows, list(table_rows[0].keys())),
        "",
        "Wynik: wszystkie etapy maja `punkty dominujace = 0`, wiec kazdy wybrany punkt "
        "nalezy do frontu minimalnych zmian swojego przejscia."
        if all(row["pareto_minimal"] for row in stage_rows)
        else "Wynik: co najmniej jeden etap nie jest minimalny w sensie Pareto.",
        "",
        "`Ranking sumy redukcji` jest dodatkowym, skalarnym porzadkiem opartym na sumie "
        "procentowych redukcji obu inputow. Pareto-minimalnosc nie wymaga pierwszego "
        "miejsca w tym rankingu: rozne proporcje redukcji tworza rozne minimalne kompromisy.",
        "",
        "## Minimalnosc calej sciezki",
        "",
        selection_text,
        "",
        f"- `TC = {fmt(selected_metric['tc'], 6)}`",
        f"- Najmniejsze TC wsrod {len(metrics)} kompletnych sciezek: `{fmt(min_tc, 6)}`",
        f"- Liczba sciezek z tym samym minimalnym TC: `{tc_ties}`",
        f"- Najwiekszy pojedynczy krok `MSC = {fmt(selected_metric['msc'], 6)}`",
        f"- Naruszenia osiagalnosci: `{int(float(selected_metric['attainable_transition_violations']))}`",
        f"- Zmiana start-koniec: {'; '.join(final_delta_parts)}",
        "",
        "## Ograniczenie wniosku",
        "",
        "Jest to certyfikat wzgledem skonczonego zbioru punktow faktycznie ocenionych przez DEA "
        "(grid, refinement i local search). Nie jest to analityczny dowod minimum w calej "
        "ciaglej przestrzeni. W eksperymencie 2D granice tego przyblizenia sa jednak latwe "
        "do pokazania i zageszczania.",
        "",
    ]
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote minimality certificate: {output_md}")
    print(f"Wrote certificate data: {output_csv}")


if __name__ == "__main__":
    main()
