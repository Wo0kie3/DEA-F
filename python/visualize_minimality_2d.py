import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from minimality_certificate import (
    find_selected_path,
    load_selection_pool,
    pareto_front_mask,
)


STAGE_COLORS = {
    1: "#e45756",
    2: "#4c78a8",
    3: "#54a24b",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--columns", required=True)
    parser.add_argument("--path-id", default=None)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def load_visualization_data(experiment_dir, columns, path_id, tolerance):
    metrics = pd.read_csv(experiment_dir / "all_path_metrics.csv")
    selected_metric, _ = find_selected_path(metrics, path_id, tolerance)
    run_dir = Path(selected_metric["run_dir"])
    if not run_dir.is_absolute():
        run_dir = experiment_dir / run_dir

    paths = pd.read_csv(run_dir / "paths.csv")
    path_row = paths[paths["path_id"].astype(str) == str(selected_metric["path_id"])]
    if path_row.empty:
        raise ValueError(f"Path {selected_metric['path_id']} not found in paths.csv.")
    path_row = path_row.iloc[0]
    milestones = pd.read_csv(run_dir / "efficiency_milestones.csv").set_index("stage")

    path_points = []
    for stage in range(int(path_row["path_length"]) + 1):
        path_points.append(
            {
                "stage": stage,
                "name": str(path_row[f"stage_{stage:02d}_name"]),
                "x": round(float(path_row[f"stage_{stage:02d}_{columns[0]}"]), 6),
                "y": round(float(path_row[f"stage_{stage:02d}_{columns[1]}"]), 6),
                "best_efficiency": round(
                    float(path_row[f"stage_{stage:02d}_best_efficiency"]),
                    6,
                ),
            }
        )

    points = []
    stage_summary = []
    for stage in range(1, int(path_row["path_length"]) + 1):
        threshold = float(milestones.loc[stage, "milestone_best_efficiency"])
        previous = path_points[stage - 1]
        pool = load_selection_pool(run_dir, stage)
        pool = pool.dropna(subset=columns).reset_index(drop=True)
        target_mask = (
            pd.to_numeric(pool["best_efficiency"], errors="coerce") + tolerance >= threshold
        )
        attainable_mask = pd.Series(True, index=pool.index)
        for column, previous_value in zip(columns, [previous["x"], previous["y"]]):
            attainable_mask &= (
                pd.to_numeric(pool[column], errors="coerce")
                <= previous_value + tolerance
            )
        eligible_mask = target_mask & attainable_mask
        eligible_pool = pool[eligible_mask].copy().reset_index(drop=True)
        eligible_front_mask = pareto_front_mask(eligible_pool, columns, tolerance)
        front_names = set(eligible_pool.loc[eligible_front_mask, "name"].astype(str))
        selected_name = path_row[f"stage_{stage:02d}_name"]

        for row_idx, row in pool.iterrows():
            target_eligible = bool(target_mask.iloc[row_idx])
            attainable = bool(attainable_mask.iloc[row_idx])
            eligible = target_eligible and attainable
            points.append(
                {
                    "stage": stage,
                    "name": str(row["name"]),
                    "x": round(float(row[columns[0]]), 6),
                    "y": round(float(row[columns[1]]), 6),
                    "best_efficiency": round(float(row["best_efficiency"]), 6),
                    "target_eligible": target_eligible,
                    "attainable": attainable,
                    "eligible": eligible,
                    "front": eligible and str(row["name"]) in front_names,
                    "selected": str(row["name"]) == str(selected_name),
                }
            )

        stage_summary.append(
            {
                "stage": stage,
                "threshold": threshold,
                "tested": len(pool),
                "target_eligible": int(target_mask.sum()),
                "eligible": int(eligible_mask.sum()),
                "front": int(eligible_front_mask.sum()),
            }
        )

    return {
        "path_id": str(selected_metric["path_id"]),
        "tc": float(selected_metric["tc"]),
        "columns": columns,
        "points": points,
        "path": path_points,
        "stages": stage_summary,
    }


def render_png(data, output_path):
    columns = data["columns"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.flatten()

    all_x = [point["x"] for point in data["points"]] + [point["x"] for point in data["path"]]
    all_y = [point["y"] for point in data["points"]] + [point["y"] for point in data["path"]]
    x_pad = max(max(all_x) - min(all_x), 1.0) * 0.06
    y_pad = max(max(all_y) - min(all_y), 1.0) * 0.06
    x_limits = (min(all_x) - x_pad, max(all_x) + x_pad)
    y_limits = (min(all_y) - y_pad, max(all_y) + y_pad)

    for stage, axis in enumerate(axes[:3], start=1):
        stage_points = [point for point in data["points"] if point["stage"] == stage]
        rejected = [point for point in stage_points if not point["target_eligible"]]
        unattainable = [
            point
            for point in stage_points
            if point["target_eligible"] and not point["attainable"]
        ]
        dominated = [
            point
            for point in stage_points
            if point["eligible"] and not point["front"]
        ]
        front = [point for point in stage_points if point["front"]]
        previous = data["path"][stage - 1]
        selected = data["path"][stage]
        color = STAGE_COLORS[stage]
        summary = data["stages"][stage - 1]

        axis.scatter(
            [point["x"] for point in rejected],
            [point["y"] for point in rejected],
            s=16,
            c="#d9dde4",
            alpha=0.35,
            marker="x",
            linewidths=0.5,
            label="ponizej progu",
        )
        axis.scatter(
            [point["x"] for point in unattainable],
            [point["y"] for point in unattainable],
            s=18,
            c="#f2b701",
            alpha=0.4,
            marker="x",
            linewidths=0.7,
            label="nieosiagalne z poprzednika",
        )
        axis.scatter(
            [point["x"] for point in dominated],
            [point["y"] for point in dominated],
            s=18,
            c="#9fa8b5",
            alpha=0.45,
            linewidths=0,
            label="zdominowane",
        )
        axis.scatter(
            [point["x"] for point in front],
            [point["y"] for point in front],
            s=38,
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            label="front Pareto",
        )
        axis.plot(
            [previous["x"], selected["x"]],
            [previous["y"], selected["y"]],
            color="#333333",
            linewidth=1.2,
            alpha=0.7,
        )
        axis.scatter(
            [previous["x"]],
            [previous["y"]],
            s=90,
            c="#f2b701",
            marker="s",
            edgecolors="#333333",
            linewidths=1,
            label="poprzednik",
            zorder=5,
        )
        axis.scatter(
            [selected["x"]],
            [selected["y"]],
            s=150,
            c=color,
            marker="*",
            edgecolors="#222222",
            linewidths=1.2,
            label="wybrany punkt",
            zorder=6,
        )
        axis.set_title(
            f"Etap {stage}: prog {summary['threshold']:.3f} | "
            f"test {summary['tested']}, prog {summary['target_eligible']}, "
            f"osiagalne {summary['eligible']}, "
            f"front {summary['front']}"
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(columns[0])
        axis.set_ylabel(columns[1])
        axis.grid(True, color="#d9dde4", linewidth=0.6, alpha=0.8)

    combined = axes[3]
    for stage in range(1, 4):
        front = [
            point
            for point in data["points"]
            if point["stage"] == stage and point["front"]
        ]
        combined.scatter(
            [point["x"] for point in front],
            [point["y"] for point in front],
            s=34,
            facecolors="none",
            edgecolors=STAGE_COLORS[stage],
            linewidths=1.2,
            label=f"front etapu {stage}",
        )
    path_x = [point["x"] for point in data["path"]]
    path_y = [point["y"] for point in data["path"]]
    combined.plot(path_x, path_y, color="#222222", linewidth=2.2, zorder=5)
    for point in data["path"]:
        color = "#f2b701" if point["stage"] == 0 else STAGE_COLORS[point["stage"]]
        marker = "s" if point["stage"] == 0 else "*"
        combined.scatter(
            [point["x"]],
            [point["y"]],
            s=120,
            c=color,
            marker=marker,
            edgecolors="#222222",
            linewidths=1,
            zorder=6,
        )
        combined.annotate(
            str(point["stage"]),
            (point["x"], point["y"]),
            xytext=(7, 7),
            textcoords="offset points",
        )
    combined.set_title(f"Fronty i minimalna sciezka {data['path_id']}")
    combined.set_xlim(*x_limits)
    combined.set_ylim(*y_limits)
    combined.set_aspect("equal", adjustable="box")
    combined.set_xlabel(columns[0])
    combined.set_ylabel(columns[1])
    combined.grid(True, color="#d9dde4", linewidth=0.6, alpha=0.8)
    combined.legend(loc="best", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle(
        "Test minimalnosci w dwoch inputach (o1 stale = 5)",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_html(data, output_path):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    title = html.escape(f"Minimalnosc 2D - {data['path_id']}")
    html_text = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #172033; background: #fff; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 12px; }}
    button {{ border: 1px solid #c7cfdb; border-radius: 999px; padding: 7px 12px; background: #f6f8fb; cursor: pointer; }}
    button.active {{ color: #fff; background: #31466b; border-color: #31466b; }}
    canvas {{ width: min(100%, 1000px); height: auto; border: 1px solid #d4dae4; border-radius: 10px; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 10px 0; font-size: 13px; }}
    .swatch {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 5px; }}
    .muted {{ color: #596579; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Minimalnosc sciezki w przestrzeni i1 x i2</h1>
  <div class="controls">
    <button type="button" data-stage="all" class="active">Wszystkie etapy</button>
    <button type="button" data-stage="1">Etap 1</button>
    <button type="button" data-stage="2">Etap 2</button>
    <button type="button" data-stage="3">Etap 3</button>
    <label><input id="showRejected" type="checkbox" checked> pokaz punkty ponizej progu i nieosiagalne</label>
    <label><input id="showDominated" type="checkbox" checked> pokaz spelniajace prog, ale zdominowane</label>
    <button type="button" id="resetView">Reset widoku</button>
  </div>
  <div class="legend">
    <span><i class="swatch" style="background:#d9dde4"></i>ponizej progu</span>
    <span><i class="swatch" style="background:#f2b701"></i>prog spelniony, ale nieosiagalny z poprzednika</span>
    <span><i class="swatch" style="background:#9fa8b5"></i>prog spelniony, ale zdominowany</span>
    <span><i class="swatch" style="background:#e45756"></i>etap 1</span>
    <span><i class="swatch" style="background:#4c78a8"></i>etap 2</span>
    <span><i class="swatch" style="background:#54a24b"></i>etap 3</span>
    <span>☆ wybrana sciezka</span>
  </div>
  <canvas id="plot" width="900" height="900" aria-label="Punkty kandydackie, fronty Pareto i wybrana sciezka w wymiarach i1 i i2"></canvas>
  <p id="detail" class="muted">Najedz na punkt. Przeciagaj, aby przesuwac; kolkiem zmieniaj skale.</p>
  <script>
const DATA = {payload};
const canvas = document.getElementById('plot');
const ctx = canvas.getContext('2d');
const COLORS = {{1:'#e45756',2:'#4c78a8',3:'#54a24b'}};
const MARGIN = {{left:72,right:28,top:24,bottom:62}};
let activeStage = null;
let dragging = false;
let lastMouse = null;
let projected = [];

const allX = [...DATA.points.map(p=>p.x), ...DATA.path.map(p=>p.x)];
const allY = [...DATA.points.map(p=>p.y), ...DATA.path.map(p=>p.y)];
const baseBounds = {{
  minX: Math.min(...allX), maxX: Math.max(...allX),
  minY: Math.min(...allY), maxY: Math.max(...allY)
}};
let view = null;
function resetView() {{
  const px = Math.max(baseBounds.maxX-baseBounds.minX,1)*0.06;
  const py = Math.max(baseBounds.maxY-baseBounds.minY,1)*0.06;
  view = {{minX:baseBounds.minX-px,maxX:baseBounds.maxX+px,minY:baseBounds.minY-py,maxY:baseBounds.maxY+py}};
  equalizeView();
  draw();
}}
function equalizeView() {{
  const plotWidth=canvas.width-MARGIN.left-MARGIN.right;
  const plotHeight=canvas.height-MARGIN.top-MARGIN.bottom;
  const pixelRatio=plotWidth/plotHeight;
  const xRange=view.maxX-view.minX;
  const yRange=view.maxY-view.minY;
  const dataRatio=xRange/yRange;
  if(dataRatio>pixelRatio) {{
    const targetY=xRange/pixelRatio;
    const centerY=(view.minY+view.maxY)/2;
    view.minY=centerY-targetY/2;view.maxY=centerY+targetY/2;
  }} else {{
    const targetX=yRange*pixelRatio;
    const centerX=(view.minX+view.maxX)/2;
    view.minX=centerX-targetX/2;view.maxX=centerX+targetX/2;
  }}
}}
function sx(x) {{ return MARGIN.left + (x-view.minX)/(view.maxX-view.minX)*(canvas.width-MARGIN.left-MARGIN.right); }}
function sy(y) {{ return canvas.height-MARGIN.bottom - (y-view.minY)/(view.maxY-view.minY)*(canvas.height-MARGIN.top-MARGIN.bottom); }}
function stageVisible(stage) {{ return activeStage === null || Number(stage) === Number(activeStage); }}
function drawGrid() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.font='12px Arial'; ctx.fillStyle='#4e596b'; ctx.strokeStyle='#dde2ea'; ctx.lineWidth=1;
  for(let i=0;i<=8;i++) {{
    const x=MARGIN.left+i*(canvas.width-MARGIN.left-MARGIN.right)/8;
    const value=view.minX+i*(view.maxX-view.minX)/8;
    ctx.beginPath();ctx.moveTo(x,MARGIN.top);ctx.lineTo(x,canvas.height-MARGIN.bottom);ctx.stroke();
    ctx.fillText(value.toFixed(2),x-16,canvas.height-MARGIN.bottom+22);
  }}
  for(let i=0;i<=7;i++) {{
    const y=MARGIN.top+i*(canvas.height-MARGIN.top-MARGIN.bottom)/7;
    const value=view.maxY-i*(view.maxY-view.minY)/7;
    ctx.beginPath();ctx.moveTo(MARGIN.left,y);ctx.lineTo(canvas.width-MARGIN.right,y);ctx.stroke();
    ctx.fillText(value.toFixed(2),MARGIN.left-48,y+4);
  }}
  ctx.fillStyle='#172033';ctx.font='bold 14px Arial';
  ctx.fillText(DATA.columns[0],canvas.width/2,canvas.height-14);
  ctx.save();ctx.translate(18,canvas.height/2);ctx.rotate(-Math.PI/2);ctx.fillText(DATA.columns[1],0,0);ctx.restore();
}}
function circle(point,radius,fill,stroke,lineWidth=1) {{
  const x=sx(point.x), y=sy(point.y);
  ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);
  ctx.fillStyle=fill;ctx.fill();
  ctx.strokeStyle=stroke;ctx.lineWidth=lineWidth;ctx.stroke();
  projected.push({{...point,sx:x,sy:y,radius}});
}}
function drawPath() {{
  const path = activeStage === null ? DATA.path : DATA.path.filter(p=>p.stage===0 || p.stage<=activeStage);
  ctx.beginPath();
  path.forEach((p,i)=>i===0?ctx.moveTo(sx(p.x),sy(p.y)):ctx.lineTo(sx(p.x),sy(p.y)));
  ctx.strokeStyle='#20242b';ctx.lineWidth=3;ctx.stroke();
  path.forEach(p=>{{
    circle(p,p.stage===0?8:10,p.stage===0?'#f2b701':COLORS[p.stage],'#20242b',2);
    ctx.fillStyle='#172033';ctx.font='bold 13px Arial';ctx.fillText(String(p.stage),sx(p.x)+12,sy(p.y)-10);
  }});
}}
function draw() {{
  projected=[];drawGrid();
  const showRejected=document.getElementById('showRejected').checked;
  const showDominated=document.getElementById('showDominated').checked;
  DATA.points.filter(p=>stageVisible(p.stage))
    .filter(p=>p.eligible ? (p.front || showDominated) : showRejected)
    .forEach(p=>{{
    if(p.front) circle(p,p.selected?9:5,'rgba(255,255,255,0.85)',COLORS[p.stage],p.selected?3:2);
    else if(p.eligible) circle(p,3,'rgba(125,136,151,0.38)','rgba(125,136,151,0.2)',1);
    else if(p.target_eligible) circle(p,3,'rgba(242,183,1,0.35)','rgba(242,183,1,0.5)',1);
    else circle(p,2.5,'rgba(205,211,220,0.32)','rgba(205,211,220,0.18)',1);
  }});
  drawPath();
}}
document.querySelectorAll('[data-stage]').forEach(button=>button.addEventListener('click',()=>{{
  activeStage=button.dataset.stage==='all'?null:Number(button.dataset.stage);
  document.querySelectorAll('[data-stage]').forEach(item=>item.classList.toggle('active',item===button));
  draw();
}}));
document.getElementById('showDominated').addEventListener('change',draw);
document.getElementById('showRejected').addEventListener('change',draw);
document.getElementById('resetView').addEventListener('click',resetView);
canvas.addEventListener('mousedown',event=>{{dragging=true;lastMouse={{x:event.clientX,y:event.clientY}};}});
window.addEventListener('mouseup',()=>{{dragging=false;lastMouse=null;}});
canvas.addEventListener('mousemove',event=>{{
  if(dragging&&lastMouse) {{
    const rect=canvas.getBoundingClientRect();
    const dx=(event.clientX-lastMouse.x)*(view.maxX-view.minX)/(rect.width-MARGIN.left-MARGIN.right);
    const dy=(event.clientY-lastMouse.y)*(view.maxY-view.minY)/(rect.height-MARGIN.top-MARGIN.bottom);
    view.minX-=dx;view.maxX-=dx;view.minY+=dy;view.maxY+=dy;
    lastMouse={{x:event.clientX,y:event.clientY}};draw();return;
  }}
  const rect=canvas.getBoundingClientRect();
  const mx=(event.clientX-rect.left)*canvas.width/rect.width;
  const my=(event.clientY-rect.top)*canvas.height/rect.height;
  let hit=null,dist=Infinity;
  projected.forEach(p=>{{const d=Math.hypot(p.sx-mx,p.sy-my);if(d<p.radius+5&&d<dist){{hit=p;dist=d;}}}});
  if(hit) document.getElementById('detail').textContent=`etap ${{hit.stage}} | ${{hit.name}} | i1=${{hit.x}}, i2=${{hit.y}}, eff=${{hit.best_efficiency}}${{hit.front?' | front Pareto wzgledem poprzednika':(hit.eligible?' | osiagalny, ale zdominowany':(hit.target_eligible?' | prog spelniony, ale nieosiagalny z poprzednika':' | ponizej progu'))}}`;
}});
canvas.addEventListener('wheel',event=>{{
  event.preventDefault();
  const factor=event.deltaY<0?0.86:1.16;
  const cx=(view.minX+view.maxX)/2,cy=(view.minY+view.maxY)/2;
  const hx=(view.maxX-view.minX)*factor/2,hy=(view.maxY-view.minY)*factor/2;
  view={{minX:cx-hx,maxX:cx+hx,minY:cy-hy,maxY:cy+hy}};equalizeView();draw();
}},{{passive:false}});
resetView();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    columns = [part.strip() for part in args.columns.split(",") if part.strip()]
    if len(columns) != 2:
        raise ValueError("Visualization expects exactly two columns.")
    data = load_visualization_data(
        experiment_dir,
        columns,
        args.path_id,
        args.tolerance,
    )
    render_png(data, Path(args.output_png))
    render_html(data, Path(args.output_html))
    print(f"Wrote PNG visualization: {args.output_png}")
    print(f"Wrote interactive visualization: {args.output_html}")


if __name__ == "__main__":
    main()
