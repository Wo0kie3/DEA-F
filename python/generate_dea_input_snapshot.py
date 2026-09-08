from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from java_runner import (
    export_extreme_efficiencies_with_java,
    export_extreme_ranks_with_java,
    generate_preference_relations_with_java,
)
from path_pipeline_common import (
    build_component_graph,
    build_relation_matrix,
    build_worse_to_better_graph,
    compute_front_map,
    get_io_columns,
    path_for_java,
    strongly_connected_components,
    transitive_reduce_dag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the reference DEA data, robust metrics and preference structures."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--java-entry", required=True, type=Path)
    parser.add_argument(
        "--preference-main-class",
        default="org.example.CsvPreferenceRelationsPreview",
    )
    parser.add_argument(
        "--efficiencies-main-class",
        default="org.example.CsvExtremeEfficienciesExporter",
    )
    parser.add_argument(
        "--ranks-main-class",
        default="org.example.CsvExtremeRanksExporter",
    )
    parser.add_argument("--maven-executable", default="mvn")
    return parser.parse_args()


def _validate_reference_data(input_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    if "name" not in input_frame.columns:
        raise ValueError("Input CSV must contain a 'name' column.")
    if input_frame["name"].isna().any() or input_frame["name"].astype(str).str.strip().eq("").any():
        raise ValueError("Input CSV contains an empty DMU name.")
    if input_frame["name"].astype(str).duplicated().any():
        raise ValueError("Input CSV contains duplicate DMU names.")

    inputs, outputs = get_io_columns(input_frame)
    if not inputs or not outputs:
        raise ValueError("Input CSV must contain at least one iN input and one oN output.")
    return inputs, outputs


def peel_efficiency_fronts(
    input_frame: pd.DataFrame,
    evaluator: Callable[[pd.DataFrame, int], pd.DataFrame],
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    dmu_order = input_frame["name"].astype(str).tolist()
    remaining = input_frame.copy()
    rows = []
    front = 1

    while not remaining.empty:
        evaluated = evaluator(remaining.copy(), front).copy()
        evaluated["name"] = evaluated["name"].astype(str)
        expected_names = remaining["name"].astype(str).tolist()
        if set(evaluated["name"]) != set(expected_names):
            raise ValueError(
                f"Efficiency-front iteration {front} returned a different DMU set."
            )
        evaluated = evaluated.set_index("name").loc[expected_names].reset_index()
        best = pd.to_numeric(evaluated["best_efficiency"], errors="raise")
        efficient = evaluated[best >= 1.0 - tolerance].copy()
        if efficient.empty:
            raise ValueError(
                f"Efficiency-front iteration {front} contains no DMU with "
                "best_efficiency equal to 1 within tolerance."
            )

        for _, row in efficient.iterrows():
            rows.append(
                {
                    "name": str(row["name"]),
                    "efficiency_front": front,
                    "front_best_efficiency": float(row["best_efficiency"]),
                    "front_worst_efficiency": float(row["worst_efficiency"]),
                }
            )
        efficient_names = set(efficient["name"].astype(str))
        remaining = remaining[
            ~remaining["name"].astype(str).isin(efficient_names)
        ].copy()
        front += 1

    return (
        pd.DataFrame(rows)
        .set_index("name")
        .loc[dmu_order]
        .reset_index()
    )


def build_snapshot_tables(
    input_frame: pd.DataFrame,
    relations: pd.DataFrame,
    efficiencies: pd.DataFrame,
    ranks: pd.DataFrame,
    efficiency_fronts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    inputs, outputs = _validate_reference_data(input_frame)
    dmu_order = input_frame["name"].astype(str).tolist()

    necessary_matrix = build_relation_matrix(relations, dmu_order, "necessary_preferred")
    possible_matrix = build_relation_matrix(relations, dmu_order, "possible_preferred")

    graph = build_worse_to_better_graph(necessary_matrix)
    components = strongly_connected_components(graph)
    component_id_by_dmu, component_members, component_graph = build_component_graph(
        components=components,
        graph=graph,
        dmu_order=dmu_order,
    )
    necessary_rank_map = compute_front_map(component_graph)
    reduced_graph = transitive_reduce_dag(component_graph)

    metrics = input_frame[["name", *inputs, *outputs]].copy()
    metrics["name"] = metrics["name"].astype(str)
    metrics = metrics.merge(
        efficiencies[["name", "best_efficiency", "worst_efficiency"]],
        on="name",
        how="left",
        validate="one_to_one",
    )
    metrics = metrics.merge(
        efficiency_fronts[
            [
                "name",
                "efficiency_front",
                "front_best_efficiency",
                "front_worst_efficiency",
            ]
        ],
        on="name",
        how="left",
        validate="one_to_one",
    )
    metrics = metrics.merge(
        ranks[["name", "best_rank", "worst_rank"]],
        on="name",
        how="left",
        validate="one_to_one",
    )
    metric_columns = [
        "best_efficiency",
        "worst_efficiency",
        "best_rank",
        "worst_rank",
        "efficiency_front",
    ]
    if metrics[metric_columns].isna().any().any():
        missing = metrics.loc[metrics[metric_columns].isna().any(axis=1), "name"].tolist()
        raise ValueError(f"Missing Java robust metrics for DMUs: {missing}")

    metrics["score_width"] = metrics["best_efficiency"] - metrics["worst_efficiency"]
    metrics["rank_width"] = metrics["worst_rank"] - metrics["best_rank"]
    metrics["necessary_component"] = metrics["name"].map(component_id_by_dmu)
    metrics["necessary_rank"] = metrics["necessary_component"].map(necessary_rank_map)

    for relation_name, matrix in (("necessary", necessary_matrix), ("possible", possible_matrix)):
        without_self = matrix.copy()
        for name in dmu_order:
            without_self.at[name, name] = False
        metrics[f"{relation_name}_over_count"] = metrics["name"].map(
            without_self.sum(axis=1).astype(int)
        )
        metrics[f"{relation_name}_over_refs"] = metrics["name"].map(
            lambda source: "|".join(
                target for target in dmu_order if bool(without_self.at[source, target])
            )
        )
        metrics[f"{relation_name}_under_count"] = metrics["name"].map(
            without_self.sum(axis=0).astype(int)
        )
        metrics[f"{relation_name}_under_refs"] = metrics["name"].map(
            lambda target: "|".join(
                source for source in dmu_order if bool(without_self.at[source, target])
            )
        )

    component_rows = []
    for component_id, members in sorted(
        component_members.items(), key=lambda item: (necessary_rank_map[item[0]], item[0])
    ):
        component_rows.append(
            {
                "component_id": component_id,
                "necessary_rank": necessary_rank_map[component_id],
                "members_count": len(members),
                "members": "|".join(members),
            }
        )

    ranking_rows = [
        {
            "name": name,
            "component_id": component_id_by_dmu[name],
            "necessary_rank": necessary_rank_map[component_id_by_dmu[name]],
        }
        for name in dmu_order
    ]

    edge_rows = []
    for source, targets in sorted(reduced_graph.items()):
        for target in sorted(targets):
            edge_rows.append(
                {
                    "source_component": source,
                    "source_members": "|".join(component_members[source]),
                    "target_component": target,
                    "target_members": "|".join(component_members[target]),
                }
            )

    return {
        "decision_data": input_frame[["name", *inputs, *outputs]].copy(),
        "reference_dmu_results": metrics,
        "necessary_matrix": necessary_matrix.astype(int),
        "possible_matrix": possible_matrix.astype(int),
        "efficiency_fronts": efficiency_fronts.copy(),
        "necessary_components": pd.DataFrame(component_rows),
        "necessary_ranking": pd.DataFrame(ranking_rows),
        "necessary_cover_edges": pd.DataFrame(
            edge_rows,
            columns=[
                "source_component",
                "source_members",
                "target_component",
                "target_members",
            ],
        ),
    }


def write_snapshot_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=name.endswith("_matrix"))


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    java_entry = args.java_entry.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_frame = pd.read_csv(input_path)
    inputs, outputs = _validate_reference_data(input_frame)

    relations_path = output_dir / "preference_relations.csv"
    efficiencies_path = output_dir / "extreme_efficiencies.csv"
    ranks_path = output_dir / "extreme_ranks.csv"

    generate_preference_relations_with_java(
        input_csv=path_for_java(str(input_path), str(java_entry)),
        output_csv=path_for_java(str(relations_path), str(java_entry)),
        java_entry=str(java_entry),
        main_class=args.preference_main_class,
        maven_executable=args.maven_executable,
    )
    export_extreme_efficiencies_with_java(
        input_csv=path_for_java(str(input_path), str(java_entry)),
        output_csv=path_for_java(str(efficiencies_path), str(java_entry)),
        java_entry=str(java_entry),
        main_class=args.efficiencies_main_class,
        maven_executable=args.maven_executable,
    )
    export_extreme_ranks_with_java(
        input_csv=path_for_java(str(input_path), str(java_entry)),
        output_csv=path_for_java(str(ranks_path), str(java_entry)),
        java_entry=str(java_entry),
        main_class=args.ranks_main_class,
        maven_executable=args.maven_executable,
    )

    efficiency_iterations_dir = output_dir / "efficiency_front_iterations"
    efficiency_iterations_dir.mkdir(parents=True, exist_ok=True)
    full_efficiencies = pd.read_csv(efficiencies_path)

    def evaluate_efficiency_front(
        remaining_frame: pd.DataFrame,
        front: int,
    ) -> pd.DataFrame:
        iteration_input = efficiency_iterations_dir / f"front_{front:02d}_input.csv"
        iteration_metrics = efficiency_iterations_dir / f"front_{front:02d}_metrics.csv"
        remaining_frame.to_csv(iteration_input, index=False)
        if front == 1:
            evaluated = full_efficiencies.copy()
            evaluated.to_csv(iteration_metrics, index=False)
            return evaluated
        export_extreme_efficiencies_with_java(
            input_csv=path_for_java(str(iteration_input), str(java_entry)),
            output_csv=path_for_java(str(iteration_metrics), str(java_entry)),
            java_entry=str(java_entry),
            main_class=args.efficiencies_main_class,
            maven_executable=args.maven_executable,
        )
        return pd.read_csv(iteration_metrics)

    efficiency_fronts = peel_efficiency_fronts(
        input_frame=input_frame,
        evaluator=evaluate_efficiency_front,
    )

    tables = build_snapshot_tables(
        input_frame=input_frame,
        relations=pd.read_csv(relations_path),
        efficiencies=pd.read_csv(efficiencies_path),
        ranks=pd.read_csv(ranks_path),
        efficiency_fronts=efficiency_fronts,
    )
    write_snapshot_tables(tables, output_dir)

    metadata = pd.DataFrame(
        [
            ("source_input", str(input_path), "Absolute source CSV used by Java"),
            ("dmu_count", len(input_frame), "Number of reference DMUs"),
            ("input_columns", "|".join(inputs), "DEA inputs, minimized"),
            ("output_columns", "|".join(outputs), "DEA outputs, maximized"),
            (
                "relation_orientation",
                "row/source preferred over column/target",
                "Orientation of necessary_matrix.csv and possible_matrix.csv",
            ),
            (
                "efficiency_front_rule",
                "iteratively remove DMUs with best_efficiency >= 1 - 1e-9",
                "DEA efficiency-front peeling rule",
            ),
            (
                "necessary_rank_1",
                "best",
                "Rank 1 contains maximal components of the necessary relation",
            ),
            ("preference_main_class", args.preference_main_class, "Java exporter"),
            ("efficiencies_main_class", args.efficiencies_main_class, "Java exporter"),
            ("ranks_main_class", args.ranks_main_class, "Java exporter"),
        ],
        columns=["parameter", "value", "description"],
    )
    metadata.to_csv(output_dir / "snapshot_metadata.csv", index=False)
    print(output_dir)


if __name__ == "__main__":
    main()
