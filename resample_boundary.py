import argparse
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# =========================================================
# CORE FUNCTION
# =========================================================

def resample_boundary(
    df,
    cols_3d,
    full_cols,
    k_neighbors=5,
    n_interpolations=3,
    noise_scale=0.05,
    only_opposite_class=False,
):
    """
    Generuje nowe punkty i zwraca pełne rekordy (wszystkie kolumny DEA)
    """

    X = df[cols_3d].to_numpy(dtype=float)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    new_rows = []
    counter = 0

    for i in range(len(X)):
        for j_idx in indices[i][1:]:

            if only_opposite_class and "candidate_efficient" in df.columns:
                if df.iloc[i]["candidate_efficient"] == df.iloc[j_idx]["candidate_efficient"]:
                    continue

            p1 = X[i]
            p2 = X[j_idx]

            base_row = df.iloc[i]

            # ===== INTERPOLATION =====
            for alpha in np.linspace(0.2, 0.8, n_interpolations):
                new_vals = p1 * (1 - alpha) + p2 * alpha

                row = build_full_row(base_row, new_vals, cols_3d, full_cols, counter)
                new_rows.append(row)
                counter += 1

            # ===== NOISE =====
            if noise_scale > 0:
                noise = np.random.normal(scale=noise_scale, size=len(cols_3d))
                new_vals = p1 + noise

                row = build_full_row(base_row, new_vals, cols_3d, full_cols, counter)
                new_rows.append(row)
                counter += 1

    if not new_rows:
        return pd.DataFrame(columns=full_cols)

    df_new = pd.DataFrame(new_rows)

    # usuwamy duplikaty po przestrzeni DEA
    df_new = df_new.drop_duplicates(subset=["i1", "i2", "i3", "i4", "o1", "o2"])

    return df_new


def build_full_row(base_row, new_vals, cols_3d, full_cols, idx):
    """
    Buduje pełny wiersz DEA
    """

    row = {}

    # ===== name =====
    row["name"] = f"RZE_resampled_{idx}"

    # ===== kopiujemy wszystkie =====
    for col in full_cols:
        if col not in ["name", "i1", "i4", "o2"]:
            row[col] = base_row.get(col, np.nan)

    # ===== nadpisujemy 3D =====
    for c, v in zip(cols_3d, new_vals):
        row[c] = float(v)

    return row


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Resample boundary with full DEA columns")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output-file", default="boundary_resampled_full.csv")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-interp", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--only-opposite", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    print(f"\nLoading: {args.input}")
    df = pd.read_csv(args.input)

    cols_3d = ["i1", "i4", "o2"]
    full_cols = ["name", "i1", "i2", "i3", "i4", "o1", "o2"]

    # ===== fallback jeśli brak kolumn =====
    if "i2" not in df.columns:
        df["i2"] = 6.0
    if "i3" not in df.columns:
        df["i3"] = 11.3
    if "o1" not in df.columns:
        df["o1"] = 0.3
    if "name" not in df.columns:
        df["name"] = [f"orig_{i}" for i in range(len(df))]

    print(f"Input points: {len(df)}")

    df_new = resample_boundary(
        df=df,
        cols_3d=cols_3d,
        full_cols=full_cols,
        k_neighbors=args.k,
        n_interpolations=args.n_interp,
        noise_scale=args.noise,
        only_opposite_class=args.only_opposite,
    )

    print(f"Generated points: {len(df_new)}")

    df_new.to_csv(output_path, index=False)

    print("\n=== DONE ===")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()