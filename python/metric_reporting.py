"""Explicit metric coverage, including mathematically undefined values."""

from pathlib import Path

import numpy as np
import pandas as pd

from path_metrics import EPS, discover_stage_indices


def metric_catalog():
    return pd.read_csv(Path(__file__).resolve().parents[1] / "templates/path_metric_descriptions.csv")


def metric_status(metric, value, path, metrics):
    if pd.notna(value) and np.isfinite(float(value)):
        return "computed"
    if metric in ("pyv", "pym") or metric.startswith(("pyv_", "pym_")):
        for stage in discover_stage_indices(path.index):
            cost = path.get(f"stage_{stage:02d}_effort_from_previous")
            if stage and pd.notna(cost) and abs(float(cost)) <= EPS:
                return "undefined_zero_step_cost"
    if metric == "dr" and pd.notna(metrics.get("cdir")) and abs(float(metrics["cdir"])) <= EPS:
        return "undefined_zero_direct_cost"
    return "missing_required_data"


def metric_coverage(metrics, paths):
    paths = paths.set_index("path_id")
    names = metric_catalog().metric.tolist()
    records = []
    for _, row in metrics.iterrows():
        path = paths.loc[row.path_id]
        for name in names:
            value = row.get(name, float("nan"))
            records.append(dict(path_id=row.path_id, metric=name, value=value,
                                status=metric_status(name, value, path, row)))
    return pd.DataFrame(records, columns=["path_id", "metric", "value", "status"])


def selected_metric_table(selected, labels):
    catalog = metric_catalog()
    result = catalog[["metric", "plain_description", "direction"]].copy()
    for method, (metrics, path, _) in selected.items():
        values = []
        for name in catalog.metric:
            value = metrics.get(name, float("nan"))
            status = metric_status(name, value, path, metrics)
            values.append(float(value) if status == "computed" else status)
        result[labels[method]] = values
    return result
