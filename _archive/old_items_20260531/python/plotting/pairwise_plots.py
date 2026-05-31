from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def save_pairwise_boundary_plot(
    boundary_csv: str,
    output_html: str,
    x: str,
    y: str,
    z: str,
):
    df = pd.read_csv(boundary_csv)

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="iteration",
        hover_data=["name", "from_point_name", "to_reference_name", "path_id"],
        title="Pairwise boundary true points",
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved pairwise boundary plot: {output_html}")


def save_pairwise_best_points_plot(
    best_csv: str,
    output_html: str,
    x: str,
    y: str,
    z: str,
):
    df = pd.read_csv(best_csv)

    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color="iteration",
        hover_data=[
            "name",
            "from_point_name",
            "to_reference_name",
            "path_id",
            "efficiency_sum",
        ],
        title="Pairwise best points",
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved pairwise best points plot: {output_html}")


def save_pairwise_tree_plot(
    best_csv: str,
    output_html: str,
    x: str,
    y: str,
    z: str,
    start_point: dict | None = None,
):
    df_best = pd.read_csv(best_csv)

    required_cols = {"node_id", "parent_node_id", "path_id", x, y, z, "name", "iteration"}
    missing = required_cols - set(df_best.columns)
    if missing:
        raise ValueError(f"Missing required columns in best_csv: {sorted(missing)}")

    fig = go.Figure()

    # punkty best
    fig.add_trace(
        go.Scatter3d(
            x=df_best[x],
            y=df_best[y],
            z=df_best[z],
            mode="markers+text",
            name="Best points",
            text=df_best["iteration"].astype(str),
            customdata=df_best[["name", "from_point_name", "to_reference_name", "path_id", "efficiency_sum"]],
            marker=dict(size=6, symbol="diamond"),
            hovertemplate=(
                "name=%{customdata[0]}<br>"
                "iteration=%{text}<br>"
                "from=%{customdata[1]}<br>"
                "to ref=%{customdata[2]}<br>"
                "path=%{customdata[3]}<br>"
                "eff_sum=%{customdata[4]}<br>"
                f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<extra></extra>"
            ),
        )
    )

    # mapowanie node_id -> punkt
    node_map = {}
    for _, row in df_best.iterrows():
        node_map[row["node_id"]] = row

    # linie parent -> child
    for _, row in df_best.iterrows():
        parent_id = row["parent_node_id"]

        if pd.isna(parent_id) or parent_id is None or str(parent_id).strip() == "":
            if start_point is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=[start_point[x], row[x]],
                        y=[start_point[y], row[y]],
                        z=[start_point[z], row[z]],
                        mode="lines",
                        name="Path",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            continue

        if parent_id in node_map:
            parent_row = node_map[parent_id]
            fig.add_trace(
                go.Scatter3d(
                    x=[parent_row[x], row[x]],
                    y=[parent_row[y], row[y]],
                    z=[parent_row[z], row[z]],
                    mode="lines",
                    name="Path",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        elif start_point is not None and str(parent_id) == "node_root":
            fig.add_trace(
                go.Scatter3d(
                    x=[start_point[x], row[x]],
                    y=[start_point[y], row[y]],
                    z=[start_point[z], row[z]],
                    mode="lines",
                    name="Path",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # start point
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
                marker=dict(size=8, symbol="x"),
                hovertemplate=(
                    "start=%{text}<br>"
                    f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Pairwise DEA path tree",
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
        ),
    )

    _ensure_parent(output_html)
    fig.write_html(output_html)
    print(f"Saved pairwise tree plot: {output_html}")