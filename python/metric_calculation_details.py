"""Auditable, path-specific arithmetic for the twenty report metrics."""

import html
import json
import math

import pandas as pd

from metric_reporting import metric_catalog
from path_metrics import EPS, PEER_SET_FIELDS, RELATION_SET_FIELDS, _parse_ref_set, discover_io_columns


def number(value):
    return "niezdefiniowane" if value is None or pd.isna(value) else f"{float(value):.10g}"


def normalized_weights(raw, count):
    values = [1.] * count if raw is None or str(raw) == "equal" else (
        json.loads(raw) if isinstance(raw, str) and raw.startswith("[") else
        [float(v) for v in raw.split(",")] if isinstance(raw, str) else raw)
    if len(values) != count or any(v < 0 for v in values) or sum(values) <= EPS:
        raise ValueError("Invalid explanation weights")
    return [v / sum(values) for v in values]


def explain_metrics(metrics, path, params, direct_pool, stage_pool):
    """Reconstruct arithmetic and reject an explanation inconsistent with the saved value."""
    length = int(path.path_length)
    factors = params["dimensions"].split(",")
    ranges = json.loads(params["normalization_ranges"])
    raw_weights = dict(part.split("=") for part in params["factor_weights"].split(","))
    weights = normalized_weights([float(raw_weights.get(f, 1)) for f in factors], len(factors))
    stage_weights = normalized_weights(params.get("stage_weights"), length)
    state_weights = normalized_weights(params.get("width_state_weights"), length + 1)
    method = metrics["method"]
    rank = method == "best_rank_path" or (method == "robustness_width_path" and params.get("width_kind") == "rank")
    progress_field, width_field = ("best_rank", "rank_width") if rank else ("best_efficiency", "score_width")
    details = {}

    def state(h, field):
        return path.get(f"stage_{h:02d}_{field}")

    def series(field):
        return [float(state(h, field)) for h in range(length + 1)]

    def add(name, formula, lines, result, note=""):
        saved = metrics.get(name)
        finite = result is not None and math.isfinite(result)
        if finite and (pd.isna(saved) or not math.isclose(float(saved), result, rel_tol=1e-8, abs_tol=1e-10)):
            raise ValueError(f"Calculation detail disagrees with {method}/{path.path_id}/{name}: {result} vs {saved}")
        if not finite and pd.notna(saved):
            raise ValueError(f"Undefined explanation for a defined metric: {name}")
        details[name] = dict(metric=name, path_id=str(path.path_id), method=method,
            formula=formula, steps=lines, computed_value=result if finite else None,
            recorded_value=float(saved) if pd.notna(saved) else None, note=note)

    delta = []
    cost_lines = ["Kolumny modyfikowane: " + ", ".join(factors) + ". Stałe czynniki nie zwiększają kosztu.",
                  "Stałe zakresy R_j: " + "; ".join(f"{f}={number(ranges[f])}" for f in factors),
                  "Znormalizowane wagi a_j: " + "; ".join(f"{f}={number(w)}" for f, w in zip(factors, weights))]
    costs = []
    for h in range(1, length + 1):
        changes = []
        for f in factors:
            previous, current = float(state(h-1, f)), float(state(h, f))
            a, b = (previous, current) if f.startswith("i") else (current, previous)
            changes.append((a - b) / ranges[f])
            cost_lines.append(f"Etap {h}, {f}: ({number(a)} − {number(b)}) / {number(ranges[f])} = {number(changes[-1])}")
        delta.append(changes)
        costs.append(sum(w * d for w, d in zip(weights, changes)))
        cost_lines.append(f"c_{h} = " + " + ".join(f"{number(w)} × {number(d)}" for w, d in zip(weights, changes)) + f" = {number(costs[-1])}")
    cost_list = ", ".join(map(number, costs))
    add("tc", "TC = Σ c_h; c_h = Σ a_j × Δ_hj", cost_lines + ["TC = " + " + ".join(map(number, costs))], sum(costs))
    add("msc", "MSC = max(c_1, …, c_L)", cost_lines + [f"MSC = max({cost_list}); największy koszt: etap {costs.index(max(costs))+1}."], max(costs))

    direct_values = [sum(d[j] for d in delta) for j in range(len(factors))]
    endpoint_lines = []
    for f, w, d in zip(factors, weights, direct_values):
        a, b = (float(state(0, f)), float(state(length, f))) if f.startswith("i") else (float(state(length, f)), float(state(0, f)))
        endpoint_lines.append(f"{f}: {number(w)} × ({number(a)} − {number(b)}) / {number(ranges[f])} = {number(w*d)}")
    add("cdir_endpoint", "Cdir_endpoint = Σ a_j × Δ(start, koniec)_j", endpoint_lines,
        sum(w*d for w, d in zip(weights, direct_values)), "Dotyczy końca tej ścieżki, nie minimum wspólnej puli celów.")

    candidates = direct_pool.dropna(subset=["direct_cost"])
    minimum = candidates.loc[candidates.direct_cost.idxmin()] if len(candidates) else None
    cdir = float(minimum.direct_cost) if minimum is not None else None
    direct_lines = [f"Zakres minimum: {metrics.get('cdir_scope')}", f"Liczba ocenionych rekordów w tym zakresie: {len(candidates)}."]
    if minimum is not None:
        direct_lines += [f"Punkt / ścieżka osiągająca minimum: {minimum['name']}",
                         f"min(kosztów bezpośrednich) = {number(cdir)}"]
        witness_cost = 0.
        for f, w in zip(factors, weights):
            a, b = (float(state(0, f)), float(minimum[f])) if f.startswith("i") else (float(minimum[f]), float(state(0, f)))
            contribution = w * (a-b) / ranges[f]
            witness_cost += contribution
            direct_lines.append(f"{f}, koszt do tego minimum: {number(w)} × ({number(a)} − {number(b)}) / {number(ranges[f])} = {number(contribution)}")
        if not math.isclose(witness_cost, cdir, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Cdir witness does not reproduce its recorded direct cost")
    add("cdir", "Cdir = min kosztu start → cel w zapisanej puli", direct_lines, cdir,
        "Minimum skończonej ocenionej puli, nie dowód minimum w całej przestrzeni.")
    add("dr", "DR = TC / Cdir", [f"{number(sum(costs))} / {number(cdir)}", *direct_lines],
        sum(costs) / cdir if cdir is not None and abs(cdir) > EPS else None,
        "Przy Cdir = 0 iloraz jest niezdefiniowany. Cdir nie jest Cdir_endpoint.")

    gaps, gap_lines = [], []
    for h in range(1, length + 1):
        gap = float(state(h, "milestone_gap"))
        gaps.append(gap)
        if method in ("best_rank_path", "best_efficiency_path", "robustness_width_path"):
            goal = state(h, "milestone_target")
            if goal is None or pd.isna(goal):
                goals = stage_pool.loc[stage_pool.stage.eq(h), "milestone_target"].dropna().unique()
                if len(goals) != 1:
                    raise ValueError("Ambiguous milestone in calculation detail")
                goal = goals[0]
            field = width_field if method == "robustness_width_path" else progress_field
            value = float(state(h, field))
            if not math.isclose(abs(value - float(goal)), gap, abs_tol=1e-9):
                raise ValueError("Milestone arithmetic inconsistent")
            gap_lines.append(f"Etap {h}, {field}: |{number(value)} − {number(goal)}| = {number(gap)}")
        else:
            gap_lines.append(f"Etap {h}: gap = {number(gap)}; cel strukturalny: " + str(state(h, "milestone_members") or state(h, "milestone_front")))
    add("md", "MD = (gap_1 + … + gap_L) / L", gap_lines + [f"({ ' + '.join(map(number, gaps)) }) / {length}"], sum(gaps)/length)

    active = [j for j, total in enumerate(direct_values) if total > EPS]
    if active:
        active_weights = normalized_weights([weights[j] for j in active], len(active))
        terms = {j: [(d[j] / direct_values[j] - 1/length)**2 for d in delta] for j in active}
        balance_lines = [f"L = {length}; K = {len(active)} czynników z dodatnią łączną zmianą.",
                         "Wagi etapów β: " + ", ".join(map(number, stage_weights)),
                         "Wagi aktywnych czynników α: " + ", ".join(f"{factors[j]}={number(w)}" for j, w in zip(active, active_weights))]
        for j in active:
            balance_lines.append(f"{factors[j]}: łączna zmiana D_j = {number(direct_values[j])}")
            for h, d in enumerate(delta, start=1):
                balance_lines.append(f"{factors[j]}, etap {h}: t_hj = ({number(d[j])}/{number(direct_values[j])} − 1/{length})² = {number(terms[j][h-1])}")
        bp = sum(sum(v) for v in terms.values())/(len(active)*length)
        wbp = sum(w*sum(terms[j]) for j, w in zip(active, active_weights))/length
        sbp = sum(sum(b*t for b, t in zip(stage_weights, terms[j])) for j in active)/len(active)
        swbp = sum(w*sum(b*t for b, t in zip(stage_weights, terms[j])) for j, w in zip(active, active_weights))
        for key, formula, result in [("bp", "BP = Σ_j Σ_h t_hj / (K × L)", bp),
            ("wbp", "WBP = Σ_j α_j Σ_h t_hj / L", wbp),
            ("sbp", "SBP = Σ_j Σ_h β_h t_hj / K", sbp),
            ("swbp", "SWBP = Σ_j α_j Σ_h β_h t_hj", swbp)]:
            add(key, formula, balance_lines, result, "Wagi czynników są ponownie normalizowane na czynnikach z dodatnią zmianą; równe wagi mogą dać identyczne wyniki.")
        concentrations = [d[j]/direct_values[j] for j in active for d in delta]
        add("mcp", "MCP = max_h,j(Δ_hj / D_j)",
            [f"{factors[j]}, etap {h}: {number(d[j])} / {number(direct_values[j])} = {number(d[j]/direct_values[j])}" for j in active for h, d in enumerate(delta, start=1)], max(concentrations))
    else:
        for key in ("bp", "wbp", "sbp", "swbp", "mcp"):
            add(key, "Wymaga dodatniej łącznej zmiany co najmniej jednego czynnika", ["Brak aktywnych czynników."], None)

    progress = series(progress_field)
    yields, yield_lines = [], ["Wskaźnik postępu: " + progress_field + ("; mniejsza ranga jest lepsza." if rank else "; większa efektywność jest lepsza.")]
    for h, cost in enumerate(costs, start=1):
        a, b = (progress[h-1], progress[h]) if rank else (progress[h], progress[h-1])
        y = (a-b)/cost if abs(cost) > EPS else None
        yields.append(y)
        yield_lines.append(f"η_{h} = ({number(a)} − {number(b)}) / {number(cost)} = {number(y)}")
    if None in yields:
        add("pyv", "PYV = Σ(η_h − średnia η)² / L", yield_lines, None, "Co najmniej jeden koszt kroku jest zerowy; nie podstawiamy zera za iloraz.")
        add("pym", "PYM = min η_h", yield_lines, None, "Co najmniej jeden koszt kroku jest zerowy.")
    else:
        mean = sum(yields)/length
        add("pyv", "PYV = Σ(η_h − średnia η)² / L", yield_lines + [f"Średnia η = {number(mean)}", "Kwadraty odchyleń: " + ", ".join(number((y-mean)**2) for y in yields), f"Suma kwadratów / {length}; wariancja populacyjna."], sum((y-mean)**2 for y in yields)/length)
        add("pym", "PYM = min η_h", yield_lines + ["min(" + ", ".join(map(number, yields)) + ")"], min(yields))

    widths = series(width_field)
    width_lines = [f"Właściwa szerokość dla tej metody: {width_field}."]
    for h, value in enumerate(widths):
        a, b = (state(h, "worst_rank"), state(h, "best_rank")) if rank else (state(h, "best_efficiency"), state(h, "worst_efficiency"))
        width_lines.append(f"Stan {h}: {number(a)} − {number(b)} = {number(value)}")
    add("apw", "APW = Σ width(z_h) / (L+1), łącznie ze startem", width_lines + [f"Suma {number(sum(widths))} / {length+1}"], sum(widths)/len(widths))
    add("fw", "FW = width(z_L)", width_lines[-1:], widths[-1])
    add("ww", "WW = Σ γ_h × width(z_h), Σ γ_h = 1", width_lines + ["Wagi γ: " + ", ".join(map(number, state_weights)), " + ".join(f"{number(w)} × {number(v)}" for w, v in zip(state_weights, widths))], sum(w*v for w, v in zip(state_weights, widths)))

    for name, fields in [("pc", PEER_SET_FIELDS), ("rc", RELATION_SET_FIELDS)]:
        chosen = next((f for f in fields if all(_parse_ref_set(state(h, f)) is not None for h in range(length+1))), None)
        if chosen is None:
            add(name, "Średnia odległości Jaccarda wszystkich kolejnych zbiorów", ["Brak kompletnego szeregu zbiorów, w tym startu."], None)
            continue
        sets = [_parse_ref_set(state(h, chosen)) for h in range(length+1)]
        lines = ["Źródłowe pole: " + chosen] + [f"Stan {h}: {{" + ", ".join(sorted(s)) + "}" for h, s in enumerate(sets)]
        distances = []
        for h, (a, b) in enumerate(zip(sets, sets[1:]), start=1):
            distances.append(1-len(a & b)/len(a | b) if a | b else 0.)
            lines.append(f"Etap {h}: 1 − {len(a & b)} / {len(a | b)} = {number(distances[-1])}" if a | b else f"Etap {h}: oba zbiory puste; odległość = 0.")
        lines.append(f"Średnia: ({' + '.join(map(number, distances))}) / {length}")
        add(name, f"{name.upper()} = średnia(1 − |P_poprz ∩ P_akt| / |P_poprz ∪ P_akt|)", lines, sum(distances)/length,
            "PC: wszystkie radialnie optymalne peery CCR na stałych oryginalnych DMU, max λ > 1e-7; bez fazy slacków." if name == "pc" else "RC dotyczy relacji, nie zbiorów peerów.")

    inputs, outputs = discover_io_columns(path.index)
    profiles, profile_lines = [], ["Udziały liczymy osobno w inputach i outputach, także tych stałych; porównujemy każdy stan ze startem."]
    for h in range(length+1):
        shares = []
        for group in (inputs, outputs):
            total = sum(float(state(h, f)) for f in group)
            for f in group:
                value = float(state(h, f))/total
                shares.append(value)
                profile_lines.append(f"Stan {h}, {f}: {number(state(h, f))}/{number(total)} = {number(value)}")
        profiles.append(shares)
    distances = [sum(abs(a-b) for a, b in zip(profile, profiles[0])) for profile in profiles[1:]]
    profile_lines += [f"Etap {h}: suma |udział_h − udział_0| = {number(v)}" for h, v in enumerate(distances, start=1)]
    add("opp", "OPP = Σ_h ||profil(z_h) − profil(z_0)||₁", profile_lines, sum(distances))
    types = [str(state(h, "state_type")) for h in range(1, length+1)]
    real_count = types.count("real")
    add("rr", "RR = liczba etapów typu real / L (bez startu)", [f"Etap {h}: {t}" for h, t in enumerate(types, start=1)] + [f"{real_count} / {length}"], real_count/length,
        "Zgodnie z implementacją korzystamy z zapisanej etykiety state_type.")
    return details


def render_metric_table(selected, labels, explanations):
    """Escaped HTML with keyboard-accessible triggers and inert explanation templates."""
    escape = lambda value: html.escape(str(value), quote=True)
    header = "<tr><th>Metryka</th><th>Opis</th><th>Kierunek</th>" + "".join(f"<th>{escape(labels[m])}</th>" for m in selected) + "</tr>"
    rows, templates = [], []
    for definition in metric_catalog().itertuples(index=False):
        cells = [f"<td>{escape(definition.metric)}</td>", f"<td>{escape(definition.plain_description)}</td>", f"<td>{escape(definition.direction)}</td>"]
        for method in selected:
            item = explanations[method][definition.metric]
            identifier = f"calc-{method}-{definition.metric}"
            display = "niezdef." if item["recorded_value"] is None else f"{item['recorded_value']:.5g}"
            cells.append(f'<td><button type="button" class="metric-value" data-calculation="{escape(identifier)}" aria-haspopup="dialog" aria-controls="metric-calculation-popup" aria-expanded="false" aria-label="{escape(definition.metric.upper() + ", " + labels[method] + ": " + display + ". Pokaż obliczenia")}">{escape(display)}</button></td>')
            body = f"<h3>{escape(definition.metric.upper())} · {escape(labels[method])}</h3><p>Ścieżka: <code>{escape(item['path_id'])}</code></p><p><b>{escape(item['formula'])}</b></p><ol>"
            body += "".join(f"<li>{escape(step)}</li>" for step in item["steps"]) + "</ol>"
            body += f"<p class='calculation-result'>Wynik: <b>{escape(number(item['recorded_value']))}</b></p><p>{escape(item['note'])}</p><small>Obliczenia używają pełnej precyzji danych. Liczby wyświetlane w tabeli i krokach są zaokrąglone.</small>"
            templates.append(f'<template id="{escape(identifier)}">{body}</template>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-wrap"><table><thead>' + header + "</thead><tbody>" + "".join(rows) + "</tbody></table></div>" + "".join(templates)
