import argparse
import html
import json
import random
from pathlib import Path

import pandas as pd


COORD_COLUMNS = ["i1", "i2", "o1"]
DISPLAY_COLUMNS = ["i1", "i2", "o1"]


def read_csv_or_empty(path: Path | None, dtype=None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-10:
            return str(int(round(value)))
        return f"{value:.6g}"
    return str(value)


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_Brak danych._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [fmt(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def split_refs(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [part for part in str(value).split("|") if part]


def choose_path(metrics: pd.DataFrame, metric: str, method: str | None) -> pd.Series:
    frame = metrics.copy()
    if method:
        frame = frame[frame["method"] == method]
    if frame.empty:
        raise ValueError("No paths available for the requested method.")
    if metric not in frame.columns:
        raise ValueError(f"Metric column not found: {metric}")
    values = pd.to_numeric(frame[metric], errors="coerce")
    if not values.notna().any():
        raise ValueError(f"Metric has no numeric values: {metric}")
    idx = values.idxmin()
    return frame.loc[idx]


def extract_path_stages(path_row: pd.Series) -> pd.DataFrame:
    stages = []
    stage = 0
    while f"stage_{stage:02d}_name" in path_row.index:
        prefix = f"stage_{stage:02d}_"
        row = {"stage": stage}
        for column in path_row.index:
            if column.startswith(prefix):
                row[column.removeprefix(prefix)] = path_row[column]
        if row.get("name"):
            stages.append(row)
        stage += 1
    return pd.DataFrame(stages)


def find_run_dir(experiment_dir: Path, selected_metric_row: pd.Series) -> Path:
    run_dir_value = str(selected_metric_row.get("run_dir", ""))
    if run_dir_value:
        path = experiment_dir / run_dir_value
        if path.exists():
            return path
    method = str(selected_metric_row["method"])
    candidates = sorted((experiment_dir / method).glob("run_*"))
    if not candidates:
        raise FileNotFoundError(f"No run directory found for method {method}")
    return candidates[-1]


def first_csv(experiment_dir: Path, file_name: str, preferred_run_dir: Path) -> pd.DataFrame:
    preferred = preferred_run_dir / file_name
    frame = read_csv_or_empty(preferred)
    if not frame.empty:
        return frame
    for path in sorted(experiment_dir.rglob(file_name)):
        frame = read_csv_or_empty(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def load_original_points(input_path: Path, experiment_dir: Path, run_dir: Path) -> pd.DataFrame:
    original = read_csv_or_empty(input_path)
    if original.empty:
        return original

    eff = first_csv(experiment_dir, "extreme_efficiencies.csv", run_dir)
    ranks = first_csv(experiment_dir, "extreme_ranks.csv", run_dir)
    if not eff.empty:
        original = original.merge(eff, on="name", how="left")
    if not ranks.empty:
        original = original.merge(ranks, on="name", how="left")
    return original


def compact_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "path_count",
        "tc_mean",
        "tc_min",
        "msc_mean",
        "bp_mean",
        "mcp_mean",
        "pc_mean",
        "opp_mean",
        "best_efficiency_improvement_mean",
        "best_rank_improvement_mean",
        "score_width_reduction_mean",
    ]
    return summary[[column for column in columns if column in summary.columns]]


def load_candidates(run_dir: Path) -> pd.DataFrame:
    candidates = read_csv_or_empty(run_dir / "fictive_candidate_metrics.csv")
    if candidates.empty:
        candidates = read_csv_or_empty(run_dir / "fictive_candidates.csv")
    return candidates


def load_refinement_points(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*refine_iter_*_metrics.csv")):
        frame = read_csv_or_empty(path)
        if frame.empty:
            continue
        parts = path.stem.split("_")
        stage = ""
        iteration = ""
        if "stage" in parts:
            idx = parts.index("stage")
            if idx + 1 < len(parts):
                stage = parts[idx + 1]
        if "iter" in parts:
            idx = parts.index("iter")
            if idx + 1 < len(parts):
                iteration = parts[idx + 1]
        frame["refinement_stage"] = stage
        frame["refinement_iteration"] = iteration
        frame["refinement_file"] = path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_local_search_points(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*local_search_metrics.csv")):
        frame = read_csv_or_empty(path)
        if frame.empty:
            continue
        parts = path.stem.split("_")
        stage = ""
        if "stage" in parts:
            idx = parts.index("stage")
            if idx + 1 < len(parts):
                stage = parts[idx + 1]
        frame["local_search_stage"] = stage
        frame["local_search_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_local_search_centers(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*refined_final_metrics.csv")):
        frame = read_csv_or_empty(path)
        if frame.empty:
            continue
        parts = path.stem.split("_")
        stage = ""
        if "stage" in parts:
            idx = parts.index("stage")
            if idx + 1 < len(parts):
                stage = parts[idx + 1]
        frame["local_search_stage"] = stage
        frame["local_search_center_name"] = frame["name"]
        frame["local_search_center_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_stage_candidates(run_dir: Path) -> pd.DataFrame:
    return read_csv_or_empty(run_dir / "stage_candidates.csv")


def dominated_mask(frame: pd.DataFrame) -> list[bool]:
    if frame.empty or not all(column in frame.columns for column in COORD_COLUMNS):
        return [False for _ in range(len(frame))]

    coords = frame[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    mask = []
    records = coords.to_dict("records")
    for idx, candidate in enumerate(records):
        if any(pd.isna(candidate[column]) for column in COORD_COLUMNS):
            mask.append(False)
            continue

        dominated = False
        for other_idx, other in enumerate(records):
            if idx == other_idx or any(pd.isna(other[column]) for column in COORD_COLUMNS):
                continue

            no_more_input_reduction = (
                float(other["i1"]) + 1e-9 >= float(candidate["i1"])
                and float(other["i2"]) + 1e-9 >= float(candidate["i2"])
            )
            no_more_output_increase = float(other["o1"]) <= float(candidate["o1"]) + 1e-9
            strictly_less_change = (
                float(other["i1"]) > float(candidate["i1"]) + 1e-9
                or float(other["i2"]) > float(candidate["i2"]) + 1e-9
                or float(other["o1"]) + 1e-9 < float(candidate["o1"])
            )
            if no_more_input_reduction and no_more_output_increase and strictly_less_change:
                dominated = True
                break
        mask.append(dominated)
    return mask


def mark_dominated(frame: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    out["dominated"] = False
    if group_column and group_column in out.columns:
        for _, group in out.groupby(group_column, dropna=False):
            out.loc[group.index, "dominated"] = dominated_mask(group)
    else:
        out["dominated"] = dominated_mask(out)
    return out


def points_for_html(frame: pd.DataFrame, kind: str, label_column: str = "name") -> list[dict]:
    points = []
    for _, row in frame.iterrows():
        if not all(column in row.index for column in COORD_COLUMNS):
            continue
        if any(pd.isna(row[column]) for column in COORD_COLUMNS):
            continue
        point = {
            "name": str(row.get(label_column, "")),
            "kind": kind,
            "stage": "" if "stage" not in row.index or pd.isna(row.get("stage")) else int(row.get("stage")),
            "x": round(float(row["i1"]), 6),
            "y": round(float(row["i2"]), 6),
            "z": round(float(row["o1"]), 6),
            "best_efficiency": fmt(row.get("best_efficiency", "")),
            "worst_efficiency": fmt(row.get("worst_efficiency", "")),
            "best_rank": fmt(row.get("best_rank", "")),
            "worst_rank": fmt(row.get("worst_rank", "")),
            "score_width": fmt(row.get("score_width", "")),
            "rank_width": fmt(row.get("rank_width", "")),
            "refinement_stage": fmt(row.get("refinement_stage", "")),
            "refinement_iteration": fmt(row.get("refinement_iteration", "")),
            "refinement_file": fmt(row.get("refinement_file", "")),
            "local_search_stage": fmt(row.get("local_search_stage", row.get("stage", ""))),
            "local_search_file": fmt(row.get("local_search_file", "")),
            "local_search_center_name": fmt(row.get("local_search_center_name", "")),
            "local_search_sample_index": fmt(row.get("local_search_sample_index", "")),
            "local_search_step_i1": fmt(row.get("local_search_step_i1", "")),
            "local_search_step_i2": fmt(row.get("local_search_step_i2", "")),
            "local_search_step_o1": fmt(row.get("local_search_step_o1", "")),
            "dominated": bool(row.get("dominated", False)),
        }
        points.append({key: value for key, value in point.items() if value != ""})
    return points


def local_cube_points(
    stages: pd.DataFrame,
    bounds_frame: pd.DataFrame,
    samples_per_stage: int,
    radius_pct: float,
    seed: int,
) -> pd.DataFrame:
    if stages.empty or samples_per_stage <= 0:
        return pd.DataFrame()

    rng = random.Random(seed)
    rows = []
    coord_ranges = {}
    for col in COORD_COLUMNS:
        values = pd.to_numeric(bounds_frame[col], errors="coerce") if col in bounds_frame.columns else pd.Series(dtype=float)
        values = values.dropna()
        if values.empty:
            center_values = pd.to_numeric(stages[col], errors="coerce").dropna()
            values = center_values
        if values.empty:
            coord_ranges[col] = (0.0, 1.0, 0.1)
            continue
        lo = float(values.min())
        hi = float(values.max())
        span = max(hi - lo, abs(hi), 1.0)
        coord_ranges[col] = (lo, hi, span * float(radius_pct))

    for _, stage in stages.iterrows():
        if int(stage.get("stage", 0)) == 0:
            continue
        for sample_idx in range(1, samples_per_stage + 1):
            row = {
                "name": f"local_cube_s{int(stage['stage'])}_{sample_idx:03d}",
                "state_type": "local_cube_sample",
                "stage": int(stage["stage"]),
                "best_efficiency": stage.get("best_efficiency", ""),
                "worst_efficiency": stage.get("worst_efficiency", ""),
                "best_rank": stage.get("best_rank", ""),
                "worst_rank": stage.get("worst_rank", ""),
                "score_width": stage.get("score_width", ""),
                "rank_width": stage.get("rank_width", ""),
            }
            for col in COORD_COLUMNS:
                center = float(stage[col])
                lo, hi, radius = coord_ranges[col]
                row[col] = min(hi, max(lo, center + rng.uniform(-radius, radius)))
            rows.append(row)

    return pd.DataFrame(rows)


def stage_rows_for_report(stages: pd.DataFrame) -> list[dict]:
    rows = []
    for _, stage in stages.iterrows():
        rows.append(
            {
                "stage": stage.get("stage"),
                "name": stage.get("name"),
                "type": stage.get("state_type"),
                "i1": stage.get("i1"),
                "i2": stage.get("i2"),
                "o1": stage.get("o1"),
                "best_eff": stage.get("best_efficiency"),
                "worst_eff": stage.get("worst_efficiency"),
                "best_rank": stage.get("best_rank"),
                "worst_rank": stage.get("worst_rank"),
                "score_width": stage.get("score_width"),
                "rank_width": stage.get("rank_width"),
                "necessary_over": len(split_refs(stage.get("candidate_necessary_over_refs", ""))),
                "possible_over": len(split_refs(stage.get("candidate_possible_over_refs", ""))),
            }
        )
    return rows


def build_html(
    output_path: Path,
    title: str,
    original_points: list[dict],
    candidate_points: list[dict],
    stage_candidate_points: list[dict],
    refinement_points: list[dict],
    local_center_points: list[dict],
    local_cube_points_data: list[dict],
    path_points: list[dict],
    selected_summary: dict,
):
    payload = {
        "original": original_points,
        "candidates": candidate_points,
        "stageCandidates": stage_candidate_points,
        "refinement": refinement_points,
        "localCenters": local_center_points,
        "localCube": local_cube_points_data,
        "path": path_points,
        "summary": selected_summary,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    safe_title = html.escape(title)
    html_text = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #172033; }}
    .wrap {{ display: grid; grid-template-columns: 920px 1fr; gap: 22px; align-items: start; }}
    canvas {{ border: 1px solid #d5dbe8; border-radius: 10px; background: #fbfcff; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    .panel {{ border: 1px solid #d5dbe8; border-radius: 10px; padding: 14px; background: #ffffff; }}
    label {{ display: block; margin: 10px 0 4px; font-weight: 600; }}
    input[type="range"] {{ width: 100%; }}
    .legend span {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; border-radius: 50%; }}
    .small {{ color: #556070; font-size: 13px; }}
    .stage-buttons {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }}
    .stage-buttons button {{ border: 1px solid #c9d3e4; border-radius: 999px; background: #f6f8fc; padding: 6px 10px; cursor: pointer; }}
    .stage-buttons button.active {{ background: #f04d4d; color: white; border-color: #f04d4d; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    td, th {{ border-bottom: 1px solid #edf0f5; padding: 4px 6px; text-align: left; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <p class="small">Osie wykresu: X=i1, Y=i2, Z=o1. W tej wizualizacji pracujemy tylko w tych trzech wymiarach.</p>
  <div class="wrap">
    <canvas id="plot" width="900" height="650"></canvas>
    <div class="panel">
      <h2>Warstwy</h2>
      <label><input id="showOriginal" type="checkbox" checked> oryginalne DMU</label>
      <label><input id="showCandidates" type="checkbox" checked> kandydaci fikcyjni metody</label>
      <label><input id="showStageCandidates" type="checkbox" checked> kandydaci etapu po froncie niezdominowanym</label>
      <label><input id="showRefinement" type="checkbox" checked> punkty przeszukiwania refinementu</label>
      <label><input id="showRefinementLabels" type="checkbox"> podpisy iteracji refinementu</label>
      <label><input id="showLocalCenters" type="checkbox" checked> centra local search</label>
      <label><input id="showLocalCube" type="checkbox" checked> ocenione punkty local search wokol etapow</label>
      <label><input id="showDominated" type="checkbox" checked> pokazuj punkty zdominowane</label>
      <label><input id="showPath" type="checkbox" checked> wybrana najlepsza sciezka</label>
      <h2>Etapy sciezki</h2>
      <div id="stageButtons" class="stage-buttons"></div>
      <p class="small">Kliknij etap, aby go podswietlic i pokazac punkty refinementu, local search oraz kandydatow etapu tylko z tego etapu. Przeciagaj wykres mysza, a kolkiem zmieniaj zoom.</p>
      <label>Obrot poziomy</label>
      <input id="yaw" type="range" min="-180" max="180" value="-35">
      <label>Obrot pionowy</label>
      <input id="pitch" type="range" min="-80" max="80" value="25">
      <label>Skala</label>
      <input id="scale" type="range" min="60" max="180" value="115">
      <h2>Legenda</h2>
      <p class="legend"><span style="background:#7d8798"></span>oryginalne DMU</p>
      <p class="legend"><span style="background:#7db7ff"></span>kandydaci fikcyjni</p>
      <p class="legend"><span style="background:#ff8a00"></span>kandydaci etapu po froncie niezdominowanym</p>
      <p class="legend"><span style="background:#9b6bff"></span>punkty przeszukiwania refinementu</p>
      <p class="legend"><span style="background:#ffffff;border:3px solid #f04d4d;box-sizing:border-box"></span>centra local search</p>
      <p class="legend"><span style="background:#2fbf71"></span>ocenione punkty local search</p>
      <p class="legend"><span style="background:#f04d4d"></span>punkty sciezki</p>
      <p class="legend"><span style="background:#ffbf00"></span>target/start</p>
      <h2>Wybrana sciezka</h2>
      <table id="summary"></table>
      <p id="hover" class="small">Najedz kursorem na punkt, aby zobaczyc szczegoly.</p>
    </div>
  </div>
  <script>
const DATA = {data_json};
const canvas = document.getElementById('plot');
const ctx = canvas.getContext('2d');
const controls = ['showOriginal', 'showCandidates', 'showStageCandidates', 'showRefinement', 'showRefinementLabels', 'showLocalCenters', 'showLocalCube', 'showDominated', 'showPath', 'yaw', 'pitch', 'scale'].map(id => document.getElementById(id));
const hover = document.getElementById('hover');
let projected = [];
let selectedStage = null;
let dragging = false;
let lastMouse = null;

function allPoints() {{ return [...DATA.original, ...DATA.candidates, ...DATA.stageCandidates, ...DATA.refinement, ...DATA.localCenters, ...DATA.localCube, ...DATA.path]; }}
function stageColor(stage) {{
  const colors = ['#ffbf00', '#e45756', '#4c78a8', '#54a24b', '#b279a2', '#f58518', '#72b7b2'];
  const idx = Math.max(0, Number(stage) || 0);
  return colors[idx % colors.length];
}}
function keepDominance(p) {{
  return document.getElementById('showDominated').checked || !p.dominated;
}}
function bounds() {{
  const pts = allPoints();
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y), zs = pts.map(p => p.z);
  return {{ minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys), minZ: Math.min(...zs), maxZ: Math.max(...zs) }};
}}
const B = bounds();
function norm(p) {{
  const nx = (p.x - B.minX) / Math.max(B.maxX - B.minX, 1e-9) - 0.5;
  const ny = (p.y - B.minY) / Math.max(B.maxY - B.minY, 1e-9) - 0.5;
  const nz = (p.z - B.minZ) / Math.max(B.maxZ - B.minZ, 1e-9) - 0.5;
  return {{x:nx, y:ny, z:nz}};
}}
function project(p) {{
  const yaw = Number(document.getElementById('yaw').value) * Math.PI / 180;
  const pitch = Number(document.getElementById('pitch').value) * Math.PI / 180;
  const sc = Number(document.getElementById('scale').value) * 4;
  let q = norm(p);
  let x1 = q.x * Math.cos(yaw) - q.z * Math.sin(yaw);
  let z1 = q.x * Math.sin(yaw) + q.z * Math.cos(yaw);
  let y1 = q.y * Math.cos(pitch) - z1 * Math.sin(pitch);
  let z2 = q.y * Math.sin(pitch) + z1 * Math.cos(pitch);
  return {{ sx: canvas.width/2 + x1*sc, sy: canvas.height/2 - y1*sc, depth: z2 }};
}}
function drawAxes() {{
  const origin = {{x:B.minX, y:B.minY, z:B.minZ, name:'origin'}};
  const axes = [
    [origin, {{x:B.maxX, y:B.minY, z:B.minZ}}, 'i1'],
    [origin, {{x:B.minX, y:B.maxY, z:B.minZ}}, 'i2'],
    [origin, {{x:B.minX, y:B.minY, z:B.maxZ}}, 'o1'],
  ];
  ctx.strokeStyle = '#aeb7c8'; ctx.fillStyle = '#586277'; ctx.lineWidth = 1;
  axes.forEach(([a,b,label]) => {{
    const pa = project(a), pb = project(b);
    ctx.beginPath(); ctx.moveTo(pa.sx, pa.sy); ctx.lineTo(pb.sx, pb.sy); ctx.stroke();
    ctx.fillText(label, pb.sx + 5, pb.sy - 5);
  }});
}}
function drawPoint(p, color, radius) {{
  const pp = project(p);
  const selected = selectedStage !== null && p.kind === 'path' && Number(p.stage) === Number(selectedStage);
  const r = selected ? radius + 7 : radius;
  ctx.beginPath();
  ctx.arc(pp.sx, pp.sy, r, 0, Math.PI*2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = selected ? '#172033' : '#ffffff';
  ctx.lineWidth = selected ? 3 : 1;
  ctx.stroke();
  if (p.kind === 'path') {{
    ctx.fillStyle = '#172033';
    ctx.font = selected ? 'bold 16px Arial' : 'bold 13px Arial';
    ctx.fillText(String(p.stage), pp.sx + r + 4, pp.sy - r - 2);
  }}
  if (p.kind === 'refinement' && document.getElementById('showRefinementLabels').checked) {{
    ctx.fillStyle = '#4a2f93';
    ctx.font = '10px Arial';
    ctx.fillText(`s${{p.refinement_stage}}/i${{p.refinement_iteration}}`, pp.sx + r + 3, pp.sy + 3);
  }}
  projected.push({{...p, sx:pp.sx, sy:pp.sy, radius:r}});
}}
function drawCenter(p) {{
  const pp = project(p);
  const color = stageColor(p.local_search_stage);
  const radius = 9;
  ctx.beginPath();
  ctx.arc(pp.sx, pp.sy, radius, 0, Math.PI*2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pp.sx - 4, pp.sy); ctx.lineTo(pp.sx + 4, pp.sy);
  ctx.moveTo(pp.sx, pp.sy - 4); ctx.lineTo(pp.sx, pp.sy + 4);
  ctx.stroke();
  projected.push({{...p, sx:pp.sx, sy:pp.sy, radius}});
}}
function draw() {{
  projected = [];
  ctx.clearRect(0,0,canvas.width,canvas.height);
  drawAxes();
  if (document.getElementById('showCandidates').checked) DATA.candidates.filter(keepDominance).forEach(p => drawPoint(p, '#7db7ff', 4));
  if (document.getElementById('showStageCandidates').checked) {{
    DATA.stageCandidates
      .filter(p => selectedStage === null || Number(p.stage) === Number(selectedStage))
      .filter(keepDominance)
      .forEach(p => drawPoint(p, stageColor(p.stage), 7));
  }}
  if (document.getElementById('showRefinement').checked) {{
    DATA.refinement
      .filter(p => selectedStage === null || Number(p.refinement_stage) === Number(selectedStage))
      .filter(keepDominance)
      .forEach(p => drawPoint(p, stageColor(p.refinement_stage), 3));
  }}
  if (document.getElementById('showLocalCube').checked) {{
    DATA.localCube
      .filter(p => selectedStage === null || Number(p.local_search_stage) === Number(selectedStage))
      .filter(keepDominance)
      .forEach(p => drawPoint(p, stageColor(p.local_search_stage), 3));
  }}
  if (document.getElementById('showLocalCenters').checked) {{
    DATA.localCenters
      .filter(p => selectedStage === null || Number(p.local_search_stage) === Number(selectedStage))
      .forEach(drawCenter);
  }}
  if (document.getElementById('showOriginal').checked) DATA.original.forEach(p => drawPoint(p, p.name === DATA.summary.target ? '#ffbf00' : '#7d8798', 6));
  if (document.getElementById('showPath').checked) {{
    const pts = DATA.path.map(project);
    ctx.strokeStyle = '#f04d4d'; ctx.lineWidth = 4;
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.sx, p.sy) : ctx.lineTo(p.sx, p.sy));
    ctx.stroke();
    DATA.path.forEach((p, i) => drawPoint(p, i === 0 ? '#ffbf00' : stageColor(p.stage), 9));
  }}
}}
function updateSummary() {{
  const s = DATA.summary;
  document.getElementById('summary').innerHTML = Object.keys(s).map(k => `<tr><th>${{k}}</th><td>${{s[k]}}</td></tr>`).join('');
}}
function updateStageButtons() {{
  const box = document.getElementById('stageButtons');
  box.innerHTML = '';
  const all = document.createElement('button');
  all.textContent = 'wszystkie';
  all.className = selectedStage === null ? 'active' : '';
  all.onclick = () => {{ selectedStage = null; updateStageButtons(); draw(); }};
  box.appendChild(all);
  DATA.path.forEach(p => {{
    const b = document.createElement('button');
    b.textContent = `etap ${{p.stage}}: ${{p.name}}`;
    b.className = Number(selectedStage) === Number(p.stage) ? 'active' : '';
    b.onclick = () => {{ selectedStage = p.stage; updateStageButtons(); hover.textContent = stageText(p); draw(); }};
    box.appendChild(b);
  }});
}}
function stageText(p) {{
  return `etap ${{p.stage}} | ${{p.name}} | i1=${{p.x}}, i2=${{p.y}}, o1=${{p.z}}, best_eff=${{p.best_efficiency}}, rank=${{p.best_rank}}-${{p.worst_rank}}`;
}}
function pointText(p) {{
  if (p.kind === 'path') return stageText(p);
  if (p.kind === 'stage_candidate') return `kandydat etapu ${{p.stage}} po froncie | ${{p.name}} | i1=${{p.x}}, i2=${{p.y}}, o1=${{p.z}}, best_eff=${{p.best_efficiency}}, rank=${{p.best_rank}}-${{p.worst_rank}}, width=${{p.score_width}}`;
  if (p.kind === 'refinement') return `refinement stage=${{p.refinement_stage}}, iter=${{p.refinement_iteration}} | ${{p.name}} | i1=${{p.x}}, i2=${{p.y}}, o1=${{p.z}}, best_eff=${{p.best_efficiency}}, rank=${{p.best_rank}}-${{p.worst_rank}}, width=${{p.score_width}}`;
  if (p.kind === 'local_center') return `centrum local search stage=${{p.local_search_stage}} | ${{p.name}} | i1=${{p.x}}, i2=${{p.y}}, o1=${{p.z}}, best_eff=${{p.best_efficiency}}, rank=${{p.best_rank}}-${{p.worst_rank}}, width=${{p.score_width}}`;
  if (p.kind === 'local_cube') return `local search stage=${{p.local_search_stage}} | ${{p.name}} | center=${{p.local_search_center_name}}, sample=${{p.local_search_sample_index}} | i1=${{p.x}}, i2=${{p.y}}, o1=${{p.z}}, step=±(${{p.local_search_step_i1}}, ${{p.local_search_step_i2}}, ${{p.local_search_step_o1}}), best_eff=${{p.best_efficiency}}, rank=${{p.best_rank}}-${{p.worst_rank}}, width=${{p.score_width}}`;
  return `${{p.name}} | i1=${{p.x}}, i2=${{p.y}}, o1=${{p.z}}, best_eff=${{p.best_efficiency}}, rank=${{p.best_rank}}-${{p.worst_rank}}`;
}}
canvas.addEventListener('mousemove', ev => {{
  if (dragging && lastMouse) {{
    const dx = ev.clientX - lastMouse.x;
    const dy = ev.clientY - lastMouse.y;
    const yaw = document.getElementById('yaw');
    const pitch = document.getElementById('pitch');
    yaw.value = Math.max(-180, Math.min(180, Number(yaw.value) + dx * 0.45));
    pitch.value = Math.max(-80, Math.min(80, Number(pitch.value) - dy * 0.35));
    lastMouse = {{x: ev.clientX, y: ev.clientY}};
    draw();
    return;
  }}
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  let hit = null;
  for (const p of projected) {{
    if (Math.hypot(p.sx-x, p.sy-y) <= p.radius + 4) hit = p;
  }}
  if (hit) hover.textContent = pointText(hit);
}});
canvas.addEventListener('mousedown', ev => {{ dragging = true; lastMouse = {{x: ev.clientX, y: ev.clientY}}; }});
window.addEventListener('mouseup', () => {{ dragging = false; lastMouse = null; }});
canvas.addEventListener('wheel', ev => {{
  ev.preventDefault();
  const scale = document.getElementById('scale');
  scale.value = Math.max(60, Math.min(180, Number(scale.value) + (ev.deltaY < 0 ? 8 : -8)));
  draw();
}}, {{passive:false}});
canvas.addEventListener('click', ev => {{
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  let hit = null;
  for (const p of projected) {{
    if (p.kind === 'path' && Math.hypot(p.sx-x, p.sy-y) <= p.radius + 5) hit = p;
  }}
  if (hit) {{
    selectedStage = hit.stage;
    updateStageButtons();
    hover.textContent = stageText(hit);
    draw();
  }}
}});
controls.forEach(c => c.addEventListener('input', draw));
updateSummary();
updateStageButtons();
draw();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def build_markdown(
    output_path: Path,
    title: str,
    experiment_dir: Path,
    selected_metric_row: pd.Series,
    run_dir: Path,
    stages: pd.DataFrame,
    method_summary: pd.DataFrame,
    original: pd.DataFrame,
    candidates: pd.DataFrame,
    refinement_points: pd.DataFrame,
    local_search_centers: pd.DataFrame,
    local_search_points: pd.DataFrame,
    html_path: Path,
):
    target = stages.iloc[0]["name"] if not stages.empty else ""
    selected = {
        "method": selected_metric_row.get("method", ""),
        "path_id": selected_metric_row.get("path_id", ""),
        "tc": selected_metric_row.get("tc", ""),
        "msc": selected_metric_row.get("msc", ""),
        "dr": selected_metric_row.get("dr", ""),
        "bp": selected_metric_row.get("bp", ""),
        "mcp": selected_metric_row.get("mcp", ""),
        "pc": selected_metric_row.get("pc", ""),
        "opp": selected_metric_row.get("opp", ""),
    }
    original_rows = []
    if not original.empty:
        cols = ["name", *DISPLAY_COLUMNS, "best_efficiency", "worst_efficiency", "best_rank", "worst_rank"]
        for _, row in original.head(30).iterrows():
            original_rows.append({column: row.get(column, "") for column in cols})

    efficiency_front_rows = []
    if "best_efficiency" in original.columns:
        best_eff = pd.to_numeric(original["best_efficiency"], errors="coerce")
        front = original[best_eff >= 0.999999].copy()
        for _, row in front.iterrows():
            efficiency_front_rows.append(
                {
                    "name": row.get("name", ""),
                    "i1": row.get("i1", ""),
                    "i2": row.get("i2", ""),
                    "o1": row.get("o1", ""),
                    "best_efficiency": row.get("best_efficiency", ""),
                    "best_rank": row.get("best_rank", ""),
                    "worst_rank": row.get("worst_rank", ""),
                }
            )

    candidate_summary = {
        "candidate_count": len(candidates),
        "min_best_efficiency": pd.to_numeric(candidates.get("best_efficiency", pd.Series(dtype=float)), errors="coerce").min(),
        "max_best_efficiency": pd.to_numeric(candidates.get("best_efficiency", pd.Series(dtype=float)), errors="coerce").max(),
        "min_best_rank": pd.to_numeric(candidates.get("best_rank", pd.Series(dtype=float)), errors="coerce").min(),
        "max_best_rank": pd.to_numeric(candidates.get("best_rank", pd.Series(dtype=float)), errors="coerce").max(),
    }
    candidate_rows = [candidate_summary]

    lines = [
        f"# {title}",
        "",
        f"Katalog eksperymentu: `{experiment_dir}`",
        f"Interaktywna wizualizacja 3D: `{html_path}`",
        "",
        "## Co jest pokazane",
        "",
        "- Szare punkty: oryginalne jednostki DMU z pliku wejsciowego.",
        "- Niebieskie punkty: kandydaci fikcyjni wygenerowani dla metody, z ktorej pochodzi wybrana sciezka.",
        "- Kolorowe wieksze punkty: kandydaci danego etapu po ograniczeniu do frontu niezdominowanego.",
        "- Fioletowe punkty: punkty testowane lokalnie podczas refinementu kandydatow.",
        "- Czerwona linia: wybrana najlepsza sciezka.",
        "- Zolty punkt: start/target.",
        "- Osie 3D: `i1`, `i2`, `o1`. Zmienna `i3` nie jest tu uzywana jako wymiar eksperymentu.",
        "",
        "## Wybrana sciezka",
        "",
        md_table([selected], list(selected.keys())),
        "",
        "## Etapy wybranej sciezki",
        "",
        md_table(
            stage_rows_for_report(stages),
            [
                "stage",
                "name",
                "type",
                "i1",
                "i2",
                "o1",
                "best_eff",
                "worst_eff",
                "best_rank",
                "worst_rank",
                "score_width",
                "rank_width",
                "necessary_over",
                "possible_over",
            ],
        ),
        "",
        "## Jak czytac mechanizm tworzenia sciezki",
        "",
        "1. Z pliku wejsciowego bierzemy realne DMU oraz target, tutaj `" + str(target) + "`.",
        "2. Generator tworzy kandydatow fikcyjnych przez ruch w wybranych wymiarach eksperymentu: `i1`, `i2`, `o1`.",
        "3. Dla kandydatow liczone sa relacje preferencji: czy kandydat koniecznie lub mozliwie przewyzsza referencje.",
        "4. Na tej podstawie kandydaci dostaja wyniki: `best_efficiency`, `worst_efficiency`, `best_rank`, `worst_rank`, `score_width`, `rank_width`.",
        "5. Kandydaci dla kazdego etapu sa wyznaczani wzgledem punktu startowego i celu danego etapu, a potem dociskani/refinowani do granicy celu.",
        "6. Po zebraniu kandydatow etapowych pipeline odrzuca punkty zdominowane w przestrzeni `i1`, `i2`, `o1`.",
        "7. Sama sciezka jest skladana krok-po-kroku: punkt etapu 2 musi byc osiagalny z punktu etapu 1, etap 3 z etapu 2 itd.",
        "8. Metryki sciezki oceniaja koszt, rownomiernosc i stabilnosc gotowej sekwencji.",
        "",
        "## Punkty wejsciowe i wyniki DEA",
        "",
        md_table(
            original_rows,
            ["name", "i1", "i2", "o1", "best_efficiency", "worst_efficiency", "best_rank", "worst_rank"],
        ),
        "",
        "## Front efektywnosci w danych wejsciowych",
        "",
        "Ponizsza tabela pokazuje jednostki z `best_efficiency` rownym 1, czyli punkty, do ktorych kandydaci moga probowac dojsc w sensie score.",
        "",
        md_table(
            efficiency_front_rows,
            ["name", "i1", "i2", "o1", "best_efficiency", "best_rank", "worst_rank"],
        ),
        "",
        "## Kandydaci fikcyjni - zakres wynikow",
        "",
        md_table(
            candidate_rows,
            ["candidate_count", "min_best_efficiency", "max_best_efficiency", "min_best_rank", "max_best_rank"],
        ),
        "",
        "## Punkty przeszukiwania refinementu",
        "",
        "Te punkty pochodza z iteracji lokalnego dociskania kandydatow do granicy celu. Na wykresie sa pokazane na fioletowo.",
        "",
        md_table(
            [
                {
                    "refinement_point_count": len(refinement_points),
                    "refinement_dominated_count": int(refinement_points.get("dominated", pd.Series(dtype=bool)).sum()) if not refinement_points.empty else 0,
                    "iteration_files": refinement_points["refinement_file"].nunique() if "refinement_file" in refinement_points.columns else 0,
                }
            ],
            ["refinement_point_count", "refinement_dominated_count", "iteration_files"],
        ),
        "",
        "## Punkty local search",
        "",
        "Te punkty sa losowymi probkami z lokalnych szescianow wokol kandydatow po refinementcie. Centra sa zaznaczone pierscieniem z krzyzykiem. Punkty zostaly przeliczone przez Jave i moga wejsc do kandydatow etapowych, jesli spelniaja cel.",
        "",
        md_table(
            [
                {
                    "local_search_center_count": len(local_search_centers),
                    "local_search_point_count": len(local_search_points),
                    "local_search_dominated_count": int(local_search_points.get("dominated", pd.Series(dtype=bool)).sum()) if not local_search_points.empty else 0,
                    "local_search_files": local_search_points["local_search_file"].nunique() if "local_search_file" in local_search_points.columns else 0,
                }
            ],
            ["local_search_center_count", "local_search_point_count", "local_search_dominated_count", "local_search_files"],
        ),
        "",
        "## Podsumowanie metod",
        "",
        md_table(compact_method_summary(method_summary).to_dict("records"), compact_method_summary(method_summary).columns.tolist()) if not method_summary.empty else "_Brak podsumowania metod._",
        "",
        "## Pliki zrodlowe uzyte w tej wizualizacji",
        "",
        f"- `all_path_metrics.csv`: `{experiment_dir / 'all_path_metrics.csv'}`",
        f"- `paths.csv`: `{run_dir / 'paths.csv'}`",
        f"- `fictive_candidate_metrics.csv`: `{run_dir / 'fictive_candidate_metrics.csv'}`",
        f"- `extreme_efficiencies.csv`: `{run_dir / 'extreme_efficiencies.csv'}`",
        f"- `extreme_ranks.csv`: `{run_dir / 'extreme_ranks.csv'}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--input", default="input/EDU.csv")
    parser.add_argument("--metric", default="tc")
    parser.add_argument("--method", default=None)
    parser.add_argument("--path-id", default=None)
    parser.add_argument("--output-html", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--title", default="Wizualizacja 3D sciezki DEA")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    input_path = Path(args.input)
    metrics = read_csv_or_empty(experiment_dir / "all_path_metrics.csv")
    method_summary = read_csv_or_empty(experiment_dir / "method_summary.csv")
    if metrics.empty:
        raise FileNotFoundError(f"No metrics found in {experiment_dir}")

    if args.path_id:
        selected_frame = metrics[metrics["path_id"].astype(str) == str(args.path_id)]
        if args.method:
            selected_frame = selected_frame[selected_frame["method"] == args.method]
        if selected_frame.empty:
            raise ValueError(f"Path {args.path_id} not found in combined metrics.")
        selected_metric_row = selected_frame.iloc[0]
    else:
        selected_metric_row = choose_path(metrics, args.metric, args.method)
    run_dir = find_run_dir(experiment_dir, selected_metric_row)
    paths = read_csv_or_empty(run_dir / "paths.csv")
    if paths.empty:
        raise FileNotFoundError(f"No paths.csv found in {run_dir}")
    selected_path_id = selected_metric_row["path_id"]
    selected_path = paths[paths["path_id"] == selected_path_id]
    if selected_path.empty:
        raise ValueError(f"Path {selected_path_id} not found in {run_dir / 'paths.csv'}")

    stages = extract_path_stages(selected_path.iloc[0])
    original = load_original_points(input_path, experiment_dir, run_dir)
    candidates = mark_dominated(load_candidates(run_dir))
    stage_candidates = mark_dominated(load_stage_candidates(run_dir), "stage")
    refinement_points = mark_dominated(load_refinement_points(run_dir), "refinement_stage")
    local_search_centers = load_local_search_centers(run_dir)
    local_search_points = mark_dominated(load_local_search_points(run_dir), "local_search_stage")

    html_path = Path(args.output_html) if args.output_html else experiment_dir / "path_3d_visualization.html"
    md_path = Path(args.output_md) if args.output_md else experiment_dir / "path_formation_report.md"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    selected_summary = {
        "target": str(stages.iloc[0]["name"] if not stages.empty else ""),
        "method": str(selected_metric_row.get("method", "")),
        "path_id": str(selected_metric_row.get("path_id", "")),
        "selection_metric": "path_id" if args.path_id else args.metric,
        "selection_value": str(args.path_id) if args.path_id else fmt(selected_metric_row.get(args.metric, "")),
        "run_dir": str(run_dir),
    }

    build_html(
        output_path=html_path,
        title=args.title,
        original_points=points_for_html(original, "original"),
        candidate_points=points_for_html(candidates, "candidate"),
        stage_candidate_points=points_for_html(stage_candidates, "stage_candidate"),
        refinement_points=points_for_html(refinement_points, "refinement"),
        local_center_points=points_for_html(local_search_centers, "local_center"),
        local_cube_points_data=points_for_html(local_search_points, "local_cube"),
        path_points=points_for_html(stages, "path"),
        selected_summary=selected_summary,
    )
    build_markdown(
        output_path=md_path,
        title=args.title,
        experiment_dir=experiment_dir,
        selected_metric_row=selected_metric_row,
        run_dir=run_dir,
        stages=stages,
        method_summary=method_summary,
        original=original,
        candidates=candidates,
        refinement_points=refinement_points,
        local_search_centers=local_search_centers,
        local_search_points=local_search_points,
        html_path=html_path,
    )
    stages.to_csv(experiment_dir / "selected_path_stages.csv", index=False)
    print(f"Wrote HTML visualization: {html_path}")
    print(f"Wrote path formation report: {md_path}")
    print(f"Wrote selected path stages: {experiment_dir / 'selected_path_stages.csv'}")


if __name__ == "__main__":
    main()
