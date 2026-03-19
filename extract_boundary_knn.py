import argparse
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def to_bool_series(s: pd.Series) -> pd.Series:
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if mapped.isna().any():
        raise ValueError("Invalid values in candidate_efficient column")
    return mapped


def extract_boundary_knn(df, cols_3d, k=15):
    """
    Zostawia punkty, które mają wśród k najbliższych sąsiadów
    punkt z przeciwną klasą.
    """

    X = df[cols_3d].to_numpy(dtype=float)
    y = df["candidate_efficient"].to_numpy()

    # model kNN
    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 bo pierwszy to sam punkt
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    boundary_mask = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        neighbor_ids = indices[i][1:]  # pomijamy siebie

        neighbor_labels = y[neighbor_ids]

        # jeśli istnieje sąsiad z przeciwną klasą → punkt graniczny
        if np.any(neighbor_labels != y[i]):
            boundary_mask[i] = True

    return boundary_mask


def main():
    parser = argparse.ArgumentParser(description="Boundary detection using k-NN")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output-file", default="boundary_knn.csv")
    parser.add_argument("--k", type=int, default=3)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)

    # konwersja bool
    df["candidate_efficient"] = to_bool_series(df["candidate_efficient"])

    cols_3d = ["i1", "i4", "o2"]

    # walidacja
    for c in cols_3d:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    print(f"Total points: {len(df)}")

    # usuwamy duplikaty przestrzenne (ważne!)
    df_unique = df.drop_duplicates(subset=cols_3d).copy()
    print(f"After deduplication: {len(df_unique)}")

    # boundary detection
    mask = extract_boundary_knn(df_unique, cols_3d, k=args.k)

    boundary_df = df_unique.loc[mask].copy()

    # oznaczenie (opcjonalne)
    boundary_df["boundary_flag"] = True

    # zapis
    boundary_df.to_csv(output_path, index=False)

    print("\n=== DONE ===")
    print(f"Boundary points: {len(boundary_df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()