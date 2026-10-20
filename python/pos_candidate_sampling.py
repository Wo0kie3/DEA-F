"""Reproducible dense grids with bounded local perturbations for POS comparisons."""

from itertools import product

import numpy as np
import pandas as pd


def expand_grid(original, columns, levels=9, samples_per_center=2, jitter_fraction=.2, seed=20261019):
    if levels < 2 or samples_per_center < 0 or not 0 < jitter_fraction <= 1:
        raise ValueError("Require levels >= 2, samples >= 0 and 0 < jitter_fraction <= 1")
    lower = original[columns].min().to_numpy(float)
    upper = original[columns].max().to_numpy(float)
    step = (upper-lower)/(levels-1)
    rng = np.random.default_rng(seed)
    rows, seen = [], {}

    def add(values, source, center="", existing=None):
        key = tuple(np.round(values, 12))
        if key in seen:
            return seen[key]
        row = original.iloc[0].to_dict() if existing is None else existing.to_dict()
        row.update(zip(columns, values))
        if existing is None:
            row["name"] = f"pos_{source}_{len(rows):07d}"
        row.update(sampling_source=source, sampling_center=center)
        seen[key] = row["name"]
        rows.append(row)
        return row["name"]

    for _, row in original.iterrows():
        add(row[columns].to_numpy(float), "source_grid", existing=row)
    centers = []
    for values in product(*(np.linspace(a, b, levels) for a, b in zip(lower, upper))):
        values = np.array(values)
        centers.append((values, add(values, "dense_grid")))
    for values, name in centers:
        # Sample the intersection directly: clipping would pile points on boundaries.
        lo = np.maximum(lower, values-jitter_fraction*step)
        hi = np.minimum(upper, values+jitter_fraction*step)
        for _ in range(samples_per_center):
            add(rng.uniform(lo, hi), "jitter", name)
    return pd.DataFrame(rows)
