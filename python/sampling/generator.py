from itertools import product
import numpy as np
import pandas as pd


def _get_io_columns(df):
    inputs = sorted(
        [c for c in df.columns if c.startswith("i")],
        key=lambda x: int(x[1:])
    )
    outputs = sorted(
        [c for c in df.columns if c.startswith("o")],
        key=lambda x: int(x[1:])
    )
    return inputs, outputs


def _validate_columns(df, columns_to_modify):
    missing = [c for c in columns_to_modify if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")


def _make_grid_values(start, end, step):
    if step <= 0:
        raise ValueError(f"Step must be positive, got {step}")

    lo = min(start, end)
    hi = max(start, end)

    values = np.arange(lo, hi + step, step)

    if values.size == 0:
        values = np.array([lo, hi], dtype=float)

    return values


def generate_frontier_samples(
    df,
    columns_to_modify,
    target_front=None,
    pct_below=20.0,
    pct_above=15.0,
    step_pct=5.0,
    target_name=None,
    target_row=None,
):
    inputs, outputs = _get_io_columns(df)
    io_cols = inputs + outputs

    _validate_columns(df, columns_to_modify)

    if target_row is None and target_name is None:
        raise ValueError("Provide either target_name or target_row.")

    if target_row is None:
        target_df = df[df["name"] == target_name]
        if target_df.empty:
            raise ValueError(f"Target '{target_name}' not found in dataset.")
        target = target_df.iloc[0]
    else:
        target = target_row

    if target_front is None:
        if "frontier_layer" not in target.index:
            raise ValueError(
                "target_front is None and target_row does not contain 'frontier_layer'."
            )
        target_front = int(target["frontier_layer"]) - 1

    frontier = df[df["frontier_layer"] == target_front]
    if frontier.empty:
        raise ValueError(f"No rows found for frontier_layer={target_front}")

    grid = {}

    for col in columns_to_modify:
        t = float(target[col])

        # step liczony ze skali frontieru
        frontier_col = frontier[col].astype(float)
        fmin = float(frontier_col.min())
        fmax = float(frontier_col.max())
        fspan = max(abs(fmax - fmin), abs(fmax), abs(fmin), abs(t), 1.0)
        step = fspan * (step_pct / 100.0)

        if col.startswith("i"):
            # input: zwykle chcemy iść w dół (mniej = lepiej),
            # ale robimy zakres odporny niezależnie od relacji target/frontier
            start = fmin * (1.0 - pct_below / 100.0)
            end = t * (1.0 + pct_above / 100.0)
        elif col.startswith("o"):
            # output: zwykle chcemy iść w górę (więcej = lepiej)
            start = t * (1.0 - pct_below / 100.0)
            end = fmax * (1.0 + pct_above / 100.0)
        else:
            raise ValueError(f"Column '{col}' is neither input nor output.")

        values = _make_grid_values(start, end, step)

        if values.size == 0:
            raise ValueError(
                f"Empty grid for column '{col}'. "
                f"target={t}, fmin={fmin}, fmax={fmax}, start={start}, end={end}, step={step}"
            )

        grid[col] = values

        print(
            f"[GRID] {col}: target={t:.6f}, frontier_min={fmin:.6f}, frontier_max={fmax:.6f}, "
            f"start={start:.6f}, end={end:.6f}, step={step:.6f}, points={len(values)}"
        )

    combos = list(product(*grid.values()))

    if not combos:
        raise ValueError("No candidate combinations generated. Check grid ranges.")

    results = []

    base_name = str(target["name"]) if "name" in target.index else "target"

    for i, combo in enumerate(combos):
        row = {"name": f"{base_name}_cand_{i}"}

        for c in io_cols:
            row[c] = target[c]

        for c, v in zip(columns_to_modify, combo):
            row[c] = v

        results.append(row)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise ValueError("Generated samples dataframe is empty.")

    return result_df