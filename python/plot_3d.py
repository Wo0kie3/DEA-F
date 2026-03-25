import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output HTML")
    parser.add_argument("--x", required=True, help="Column for X axis")
    parser.add_argument("--y", required=True, help="Column for Y axis")
    parser.add_argument("--z", required=True, help="Column for Z axis")

    parser.add_argument(
        "--color",
        default=None,
        help="Optional column for point color, e.g. frontier_layer or candidate_efficient",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Optional column for point symbol",
    )
    parser.add_argument(
        "--title",
        default="3D Scatter Plot",
        help="Plot title",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=6,
        help="Marker size",
    )
    parser.add_argument(
        "--hover-name",
        default="name",
        help="Column displayed as point label in hover",
    )

    return parser.parse_args()


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col is not None and col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    validate_columns(
        df,
        [args.x, args.y, args.z, args.color, args.symbol, args.hover_name],
    )

    fig = px.scatter_3d(
        df,
        x=args.x,
        y=args.y,
        z=args.z,
        color=args.color,
        symbol=args.symbol,
        hover_name=args.hover_name if args.hover_name in df.columns else None,
        title=args.title,
        opacity=0.8,
    )

    fig.update_traces(marker=dict(size=args.size))
    fig.update_layout(
        scene=dict(
            xaxis_title=args.x,
            yaxis_title=args.y,
            zaxis_title=args.z,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    print(f"Saved plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main()