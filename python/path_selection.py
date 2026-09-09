import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TOLERANCE = 1e-9
STRUCTURAL_METHODS = {"hasse_path", "front_path", "necessary_layer_path"}


def default_criteria(method_name: str | None) -> list[str]:
    if method_name in STRUCTURAL_METHODS:
        return ["msc", "tc", "swbp", "fw"]
    return ["md", "msc", "swbp", "fw"]


def _usable(metrics: pd.DataFrame, criteria: list[str]) -> tuple[pd.DataFrame, list[str]]:
    available = [criterion for criterion in criteria if criterion in metrics.columns]
    if not available:
        return metrics.iloc[0:0].copy(), []
    numeric = metrics.copy()
    for criterion in available:
        numeric[criterion] = pd.to_numeric(numeric[criterion], errors="coerce")
    return numeric.dropna(subset=available).copy(), available


def tolerance_lexicographic_choice(
    metrics: pd.DataFrame,
    criteria: list[str],
    tolerance: float = TOLERANCE,
) -> pd.Series | None:
    pool, available = _usable(metrics, criteria)
    if pool.empty:
        return None
    for criterion in available:
        best = float(pool[criterion].min())
        pool = pool[pool[criterion] <= best + tolerance].copy()
    return pool.sort_values("path_id", kind="stable").iloc[0]


def pareto_minimal_mask(
    metrics: pd.DataFrame,
    criteria: list[str],
    tolerance: float = TOLERANCE,
) -> pd.Series:
    mask = pd.Series(False, index=metrics.index)
    pool, available = _usable(metrics, criteria)
    if pool.empty:
        return mask
    values = pool[available].to_numpy(dtype=float)
    keep = np.ones(len(pool), dtype=bool)
    for index, value in enumerate(values):
        no_worse = np.all(values <= value + tolerance, axis=1)
        strictly_better = np.any(values < value - tolerance, axis=1)
        no_worse[index] = False
        if np.any(no_worse & strictly_better):
            keep[index] = False
    mask.loc[pool.index] = keep
    return mask


def normalized_weighted_choice(
    metrics: pd.DataFrame,
    criteria: list[str],
    weights: dict[str, float] | None = None,
    tolerance: float = TOLERANCE,
) -> tuple[pd.Series | None, pd.Series]:
    pool, available = _usable(metrics, criteria)
    scores = pd.Series(np.nan, index=metrics.index, dtype=float)
    if pool.empty:
        return None, scores

    raw_weights = {
        criterion: float(weights.get(criterion, 1.0) if weights else 1.0)
        for criterion in available
    }
    if any(value < 0 for value in raw_weights.values()) or sum(raw_weights.values()) <= 0:
        raise ValueError("Selection weights must be non-negative with a positive sum.")
    total_weight = sum(raw_weights.values())
    normalized_weights = {
        criterion: value / total_weight for criterion, value in raw_weights.items()
    }

    normalized = pd.DataFrame(index=pool.index)
    for criterion in available:
        minimum = float(pool[criterion].min())
        maximum = float(pool[criterion].max())
        span = maximum - minimum
        normalized[criterion] = (
            0.0 if span <= tolerance else (pool[criterion] - minimum) / span
        )
    pool_scores = sum(
        normalized_weights[criterion] * normalized[criterion]
        for criterion in available
    )
    scores.loc[pool.index] = pool_scores
    best = float(pool_scores.min())
    candidates = pool.loc[pool_scores <= best + tolerance].copy()
    return candidates.sort_values("path_id", kind="stable").iloc[0], scores


def select_publication_paths(
    metrics: pd.DataFrame,
    method_name: str | None = None,
    criteria: list[str] | None = None,
    weights: dict[str, float] | None = None,
    tolerance: float = TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = criteria or default_criteria(method_name)
    pool, available = _usable(metrics, requested)
    selection_rows = []
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    single_metric_criteria = list(dict.fromkeys(["tc", *available]))
    for criterion in single_metric_criteria:
        single_pool, single_available = _usable(metrics, [criterion])
        if single_pool.empty or not single_available:
            continue
        best = float(single_pool[criterion].min())
        winners = single_pool[single_pool[criterion] <= best + tolerance].sort_values(
            "path_id", kind="stable"
        )
        winner = winners.iloc[0]
        selection_rows.append(
            {
                "selection_type": "single_metric",
                "selection_detail": criterion,
                "path_id": winner["path_id"],
                "selection_value": float(winner[criterion]),
                "tie_count": len(winners),
            }
        )

    lex = tolerance_lexicographic_choice(pool, available, tolerance)
    if lex is not None:
        selection_rows.append(
            {
                "selection_type": "lexicographic",
                "selection_detail": " > ".join(available),
                "path_id": lex["path_id"],
                "selection_value": None,
                "tie_count": None,
            }
        )

    weighted, scores = normalized_weighted_choice(
        pool,
        available,
        weights=weights,
        tolerance=tolerance,
    )
    if weighted is not None:
        selection_rows.append(
            {
                "selection_type": "normalized_weighted",
                "selection_detail": " + ".join(available),
                "path_id": weighted["path_id"],
                "selection_value": float(scores.loc[weighted.name]),
                "tie_count": None,
            }
        )

    pareto = pool.loc[pareto_minimal_mask(pool, available, tolerance)].copy()
    pareto.insert(0, "selection_criteria", "|".join(available))
    return pd.DataFrame(selection_rows), pareto.reset_index(drop=True)


def write_publication_selection(
    metrics: pd.DataFrame,
    output_dir: str | Path,
    method_name: str | None = None,
    criteria: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection, pareto = select_publication_paths(
        metrics,
        method_name=method_name,
        criteria=criteria,
        weights=weights,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    selection.to_csv(directory / "path_selection.csv", index=False)
    pareto.to_csv(directory / "path_pareto.csv", index=False)
    if pareto.empty:
        representatives = pareto.copy()
    else:
        pareto_criteria = str(pareto.iloc[0]["selection_criteria"]).split("|")
        grouping_columns = []
        grouped = pareto.copy()
        for criterion in pareto_criteria:
            key = f"__rounded_{criterion}"
            grouped[key] = pd.to_numeric(grouped[criterion], errors="coerce").round(9)
            grouping_columns.append(key)
        representatives = (
            grouped.sort_values("path_id", kind="stable")
            .groupby(grouping_columns, dropna=False, sort=False)
            .first()
            .reset_index(drop=True)
        )
        equivalence_counts = grouped.groupby(
            grouping_columns, dropna=False, sort=False
        ).size().tolist()
        representatives.insert(1, "metric_equivalence_count", equivalence_counts)
    representatives.to_csv(
        directory / "path_pareto_representatives.csv",
        index=False,
    )
    return selection, pareto


def _parse_weights(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    parsed = {}
    for item in value.split(","):
        name, raw = (part.strip() for part in item.split("=", 1))
        parsed[name] = float(raw)
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--criteria", default=None)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    criteria = (
        [part.strip() for part in args.criteria.split(",") if part.strip()]
        if args.criteria
        else None
    )
    write_publication_selection(
        pd.read_csv(args.path_metrics),
        args.output_dir,
        method_name=args.method_name,
        criteria=criteria,
        weights=_parse_weights(args.weights),
    )


if __name__ == "__main__":
    main()
