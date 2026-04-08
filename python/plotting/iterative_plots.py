from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def save_boundary_plot(boundary_csv: str, output_html: str, x: str, y: str, z: str):
    df = pd.read_csv(boundary_csv)

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="iteration",
        hover_data=["name", "reference_frontier"],
        title="Boundary true points across iterations",
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved boundary plot: {output_html}")


def save_selected_points_plot(selected_csv: str, output_html: str, x: str, y: str, z: str):
    df = pd.read_csv(selected_csv)

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="iteration",
        hover_data=[
            "name",
            "reference_frontier",
            "selected_rank",
            "candidate_efficiency",
            "efficiency_sum",
        ],
        title="Selected points across iterations",
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved selected-points plot: {output_html}")


def save_best_points_plot(best_csv: str, output_html: str, x: str, y: str, z: str):
    df = pd.read_csv(best_csv)

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="iteration",
        hover_data=["name", "reference_frontier", "efficiency_sum"],
        title="Best points across iterations",
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved best points plot: {output_html}")


def save_combined_iterative_plot(
    boundary_csv: str,
    best_csv: str,
    output_html: str,
    x: str,
    y: str,
    z: str,
    start_point: dict | None = None,
):
    df_boundary = pd.read_csv(boundary_csv)
    df_best = pd.read_csv(best_csv)

    fig = go.Figure()

    # Boundary points
    for iteration in sorted(df_boundary["iteration"].dropna().unique()):
        dfi = df_boundary[df_boundary["iteration"] == iteration]
        fig.add_trace(
            go.Scatter3d(
                x=dfi[x],
                y=dfi[y],
                z=dfi[z],
                mode="markers",
                name=f"Boundary iter {int(iteration)}",
                text=dfi["name"],
                customdata=dfi[["reference_frontier"]],
                marker=dict(size=3),
                hovertemplate=(
                    "name=%{text}<br>"
                    f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<br>"
                    "frontier=%{customdata[0]}<extra></extra>"
                ),
            )
        )

    # Best points
    fig.add_trace(
        go.Scatter3d(
            x=df_best[x],
            y=df_best[y],
            z=df_best[z],
            mode="markers+lines+text",
            name="Best points path",
            text=df_best["iteration"].astype(str),
            customdata=df_best[["name", "reference_frontier", "efficiency_sum"]],
            marker=dict(size=7, symbol="diamond"),
            hovertemplate=(
                "name=%{customdata[0]}<br>"
                "iteration=%{text}<br>"
                f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<br>"
                "frontier=%{customdata[1]}<br>"
                "eff_sum=%{customdata[2]}<extra></extra>"
            ),
        )
    )

    # Start point
    if start_point is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[start_point[x]],
                y=[start_point[y]],
                z=[start_point[z]],
                mode="markers+text",
                name="Start point",
                text=[start_point.get("name", "start")],
                textposition="top center",
                marker=dict(size=9, symbol="x"),
                hovertemplate=(
                    "start=%{text}<br>"
                    f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Iterative DEA path: boundary true points and best points",
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
        ),
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved combined iterative plot: {output_html}")


def save_all_results_plot(results_csv: str, output_html: str, x: str, y: str, z: str):
    df = pd.read_csv(results_csv)

    if "candidate_efficient" in df.columns:
        df["candidate_efficient"] = df["candidate_efficient"].astype(str).str.lower()

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="candidate_efficient",
        hover_data=[
            "name",
            "candidate_efficiency",
            "candidate_efficient",
        ],
        title="All evaluated grid points (true/false)",
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved all-results plot: {output_html}")


def save_boundary_flag_plot(results_csv: str, output_html: str, x: str, y: str, z: str):
    df = pd.read_csv(results_csv)

    if "candidate_efficient" in df.columns:
        df["candidate_efficient"] = df["candidate_efficient"].astype(str).str.lower()

    if "boundary_has_false_neighbor" in df.columns:
        df["boundary_has_false_neighbor"] = df["boundary_has_false_neighbor"].map(
            lambda v: "boundary" if str(v).lower() == "true" else "not_boundary"
        )

    df["boundary_plot_group"] = (
        df["boundary_has_false_neighbor"].astype(str) + "_" + df["candidate_efficient"].astype(str)
    )

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="boundary_plot_group",
        symbol="candidate_efficient",
        hover_data=[
            "name",
            "candidate_efficiency",
            "candidate_efficient",
            "boundary_has_false_neighbor",
            "boundary_false_neighbor_count",
        ],
        title="Grid points with boundary-neighbor flag",
        color_discrete_map={
            "boundary_true": "#d62728",
            "boundary_false": "#ff9896",
            "not_boundary_true": "#1f77b4",
            "not_boundary_false": "#2ca02c",
        },
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved boundary-flag plot: {output_html}")
