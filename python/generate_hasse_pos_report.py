"""Self-contained comparison of two final POS quantifiers on NEC graph routes."""

import html
import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from generate_five_method_report import table, plot, scatter, axis_layout, hasse_figure, method_figure, state_frame
from generate_five_unit_2d_html import load_dea_snapshot
from metric_calculation_details import explain_metrics, render_metric_table
from metric_reporting import selected_metric_table, metric_coverage
from generate_five_method_robot_report import METHODS, _latest_run, _selected_path
from generate_five_method_report import LABELS as FIVE_LABELS


LABELS = {"pairwise": "POS osobno dla każdej jednostki", "joint": "Przewaga przy wspólnych wagach"}


def markdown(frame):
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def dataset_section(reference, params):
    from path_pipeline_common import get_io_columns
    inputs, outputs = get_io_columns(reference)
    target = reference.set_index('name').loc[params['target']]
    modified = params['dimensions'].split(',')
    name = Path(str(params.get('input', 'dataset.csv'))).stem
    rows = [dict(zmienna=c, rola='Nakład (input)' if c in inputs else 'Wynik (output)',
                 kierunek='mniej = lepiej' if c in inputs else 'więcej = lepiej',
                 minimum=reference[c].min(), maksimum=reference[c].max(),
                 wartosc_startowa=target[c], zmieniana='tak' if c in modified else 'nie')
            for c in inputs+outputs]
    metadata_file = Path(__file__).resolve().parents[1]/'input/EDU.metadata.json'
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        canonical = json.dumps(reference.to_dict('records'),sort_keys=True,separators=(',',':'))
        if hashlib.sha256(canonical.encode()).hexdigest() == metadata['canonical_records_sha256']:
            for row in rows:
                definition = metadata['variables'][row['zmienna']]
                row.update(znaczenie=definition['meaning'],nazwa_zrodlowa=definition['source_name'])
            source = metadata['source']
            article = metadata['related_article']
            return (f"<section id='dataset-description'><h2>O zbiorze danych: EDU</h2>"
                "<p><b>Jednostkami są lekarze pediatrycznego oddziału ratunkowego CHEO w Ottawie.</b> "
                "Badanie łączyło ocenę wykorzystania zasobów z jakością opieki; pacjentów grupowano według dolegliwości. "
                f"Kontekst: <a href='{html.escape(article['url'],quote=True)}'>Fiallos i in. (2017)</a>.</p>"
                f"<p>Nasze 20 wierszy odpowiada MD1–MD20 w tabeli na s. {source['table_page']} "
                f"<a href='{html.escape(source['url'],quote=True)}'>pracy Fiallosa ({source['year']})</a>: "
                "ból brzucha i zaparcia, dane z 2011 r. Zgodność: 80/80 wartości. "
                "Definicje zmiennych: s. 48; pochodzenie danych: s. 55.</p>"
                + table(pd.DataFrame(rows))
                + f"<p>Eksperyment dotyczy {html.escape(params['target'])}; output pozostaje stały. "
                "Ścieżki przedstawiają wyznaczone przez model kolejne etapy redukcji nakładów i1, i2 i i3 "
                "przy zachowaniu udziału wizyt bez powrotu pacjenta w ciągu 72 godzin. "
                "Każdy etap określa poziom nakładów spełniający cel danej metody.</p></section>")
    return (f"<section id='dataset-description'><h2>O zbiorze danych: {html.escape(name)}</h2>"
            f"<p>Zbiór zawiera <b>{len(reference)} jednostek decyzyjnych (DMU)</b>, opisanych przez "
            f"{len(inputs)} nakłady i {len(outputs)} wynik. Każdy wiersz reprezentuje jedną jednostkę. "
            "DEA służy tu do porównania wyników jednostek z wykorzystanymi nakładami.</p>"
            f"<p>Badamy <b>{html.escape(params['target'])}</b>: szukamy kolejnych stanów wymagających "
            f"zmniejszenia nakładów {html.escape(', '.join(modified))}, przy niezmienionych pozostałych zmiennych. "
            "Punkty siatki i szumu to hipotetyczne warianty tej jednostki, nie nowe obserwacje. "
            "Oryginalne DMU stanowią stały zbiór referencyjny.</p>"
            "<p><b>Zakres informacji o danych:</b> plik wejściowy zawiera wyłącznie identyfikatory, "
            "symbole zmiennych i wartości. W repozytorium nie udokumentowano znaczenia dziedzinowego "
            "zmiennych, jednostek miary, sposobu skalowania ani publikacji źródłowej. "
            "Poniższe zakresy dotyczą wartości zapisanych w pliku, nie skali wykresu.</p>"
            + table(pd.DataFrame(rows)) + "</section>")


def build_report(root, reference, candidates, selected, summary, params, top, five_method_root=None):
    labels = dict(LABELS)
    selected = dict(selected)
    runs = {key: root/key for key in selected}
    methods = {key: 'hasse_path' for key in selected}
    five_summary = []
    if five_method_root is not None:
        for method, _, color in METHODS:
            labels[method] = FIVE_LABELS[method] + (' (historyczny cel NEC)' if method in ('hasse_path','front_path') else '')
            run = _latest_run(five_method_root/method)
            runs[method], methods[method] = run, method
            pool = pd.read_csv(run/'fictive_candidate_metrics.csv')
            pd.testing.assert_frame_equal(candidates[['name', *params['dimensions'].split(',')]].reset_index(drop=True),
                pool[['name', *params['dimensions'].split(',')]].reset_index(drop=True), check_dtype=False)
            stage_pool = pd.read_csv(run/'stage_candidates.csv')
            if not set(stage_pool.name).issubset(set(candidates.name)):
                raise ValueError('Method-specific candidates found in shared-pool report')
            try:
                metric, path, count, _ = _selected_path(run)
            except (ValueError, IndexError, pd.errors.EmptyDataError):
                five_summary.append(dict(method=labels[method], paths=0))
                continue
            selected[method] = (metric, path, color)
            five_summary.append(dict(method=labels[method], paths=count, limit_reached=count >= int(params['max_paths']),
                tc=metric.tc, msc=metric.msc, final_best_efficiency=metric.best_efficiency_final))
    repo = Path(__file__).resolve().parents[1]
    sources = [p for p in (repo / "python").glob("*.py") if not p.name.startswith("test_")]
    sources += list((repo / "python/report_assets").glob("metric_calculations.*"))
    sources += list((repo / "java/src/main/java/org/example").glob("*.java")) + [repo / "java/pom.xml"]
    pd.DataFrame([dict(file=str(p.relative_to(repo)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(sources)]).to_csv(root / "implementation_manifest.csv", index=False)
    columns = params["dimensions"].split(",")
    snapshot = load_dea_snapshot(root / "dea_input_snapshot", reference)
    baseline = pd.read_csv(root / "dea_input_snapshot/reference_dmu_results.csv")
    ref_plot = reference.merge(baseline[["name", "best_efficiency", "worst_efficiency", "best_rank", "worst_rank", "score_width"]], on="name", validate="one_to_one")
    ranges = {c: [float(min(reference[c].min(), candidates[c].min()))*.95,
                  float(max(reference[c].max(), candidates[c].max()))*1.05] for c in columns}
    overview = go.Figure()
    for name, mask, color in [("Cel końcowy niespełniony", ~candidates.pairwise_top, "#b7c0ca"),
                               ("Tylko POS parami", candidates.pairwise_top & ~candidates.joint_top, "#d97706"),
                               ("Spełnia wspólne wagi (i POS parami)", candidates.joint_top, "#30946b")]:
        overview.add_trace(scatter(candidates.loc[mask], columns, name, color, marker=dict(opacity=.45, size=4)))
    sections, details, audits, coverage_frames = [], {}, [], []
    for variant, (metric, path, color) in selected.items():
        run = runs[variant]
        states = state_frame(path)
        overview.add_trace(scatter(states, columns, labels[variant], color, mode="lines+markers+text",
            text=["Start", *map(str, range(1, len(states)))], line=dict(color=color, width=6), marker=dict(size=7, opacity=1)))
        fig, states, stage_audits = method_figure(run, methods[variant], path, ref_plot, columns, ranges, params)
        fig.update_layout(title=labels[variant]+" · etap 1")
        for i, button in enumerate(fig.layout.updatemenus[0].buttons, start=1):
            button.args[1]["title.text"] = labels[variant]+f"<br>etap {i}"
        for row in stage_audits:
            row["variant"] = variant
        audits.extend(stage_audits)
        stage_columns = ["stage", "name", *columns, *snapshot["output_columns"], "milestone_target_name",
                         "milestone_members", "milestone_goal_kind", "best_efficiency", "worst_efficiency",
                         "best_rank", "worst_rank", "effort_from_previous", "peer_refs"]
        stage_columns += [c for c in states if c.startswith('milestone_') and c not in stage_columns]
        stages = states.reindex(columns=stage_columns)
        stages.to_csv(run / "selected_path_stages.csv", index=False)
        zero_steps = sum(abs(float(v)) <= 1e-12 for v in states.effort_from_previous.iloc[1:])
        sections.append(f"<section><h2>{labels[variant]}</h2><p>Wybrana ścieżka: {html.escape(str(path.path_id))}. "
            f"Kroki o zerowym koszcie: {zero_steps}. POS może być spełnione już przed zmianą; nie wymuszamy sztucznego dodatniego kroku.</p>"
            + plot(fig, "stages-"+variant) + table(stages) + "<details><summary>Audyt frontów minimalnych zmian</summary>"
            + table(pd.DataFrame(stage_audits)) + "</details></section>")
        metrics = pd.read_csv(run / "path_metrics.csv")
        paths = pd.read_csv(run / "paths.csv")
        coverage_frames.append(metric_coverage(metrics, paths).assign(variant=variant))
        direct = metrics[["path_id", "cdir_endpoint"]].rename(columns={"path_id": "name", "cdir_endpoint": "direct_cost"})
        for c in columns:
            by_id = {p.path_id: p[f"stage_{int(p.path_length):02d}_{c}"] for _, p in paths.iterrows()}
            direct[c] = direct.name.map(by_id)
        stage_pool = pd.read_csv(run/'stage_candidates.csv')
        if metric.cdir_scope == 'evaluated_final_stage_pool':
            direct = stage_pool.loc[stage_pool.stage.eq(stage_pool.stage.max()), ['name', 'effort_from_start', *columns]].rename(columns={'effort_from_start':'direct_cost'})
        details[variant] = explain_metrics(metric, path, params, direct, stage_pool)
    for variant in labels.keys()-selected.keys():
        sections.append(f"<section><h2>{labels[variant]}</h2><p>Brak kompletnej ścieżki w ocenionej puli. Nie jest to dowód nieosiągalności celu w przestrzeni ciągłej.</p></section>")
    audit = pd.DataFrame(audits)
    audit.to_csv(root / "selected_transition_audit.csv", index=False)
    if not audit.empty and (not audit.selected_on_front.all() or audit.dominating_points.gt(0).any()):
        raise ValueError("Selected transition front reconstruction failed")
    all_metrics = selected_metric_table(selected, labels)
    all_metrics.to_csv(root / "selected_path_all_metrics.csv", index=False)
    if coverage_frames:
        coverage = pd.concat(coverage_frames, ignore_index=True)
        coverage.groupby(['variant','metric','status']).size().reset_index(name='path_count').to_csv(root/'metric_coverage.csv',index=False)
        coverage.loc[~coverage.status.eq('computed')].to_csv(root/'metric_exceptions.csv',index=False)
    (root / "metric_calculation_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    overview.update_layout(**axis_layout(columns, ranges), title="Ta sama pula punktów: pięć metod i dwa warianty POS" if five_method_root else "Ta sama pula punktów, dwa warunki końcowe")
    hasse, count = hasse_figure(snapshot, params["target"])
    differences = candidates.loc[candidates.pairwise_top & ~candidates.joint_top,
        ["name", *columns, *snapshot["output_columns"], "best_efficiency", "worst_efficiency", "joint_score", "effort_from_start"]]
    differences.to_csv(root / "pairwise_only_candidates.csv", index=False)
    witnesses = pd.read_csv(root / "possible_goals.csv.witnesses.csv")
    selected_names = [str(p[1][f"stage_{int(p[1].path_length):02d}_name"]) for p in selected.values()]
    common = pd.read_csv(root / "possible_goals.csv")
    weight_audit = pd.read_csv(root / "joint_weight_audit.csv")
    matrices = ""
    for kind in ("necessary", "possible"):
        matrix = pd.read_csv(root / f"dea_input_snapshot/{kind}_matrix.csv")
        matrices += f"<details><summary>Macierz {kind}: wiersz preferowany nad kolumną</summary>" + table(matrix) + "</details>"
    report = "<!doctype html><html lang='pl'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Hasse: POS parami i wspólne wagi</title><script>" + get_plotlyjs() + "</script><style>body{font-family:Segoe UI,Arial,sans-serif;background:#f3f6f9;color:#17324a;margin:0}main{max-width:1450px;margin:auto;padding:24px}section{background:white;border:1px solid #dce4eb;border-radius:12px;padding:22px;margin:20px 0}p{line-height:1.6}.table-wrap{overflow:auto;max-height:650px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce4eb;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#edf4fa}td:first-child{text-align:left}.metric-table td:nth-child(2){white-space:normal;min-width:240px;text-align:left}details{margin:16px 0}summary{cursor:pointer;font-weight:bold}@media print{.table-wrap{max-height:none}section{break-inside:avoid}}</style></head><body><main>"
    report += f"<h1>{html.escape(params['target'])}: dwa warianty celu POS</h1><p>Zmieniane: {html.escape(', '.join(columns))}; outputy stałe. {len(candidates)} identycznych punktów w obu wariantach. "
    report += "Ścieżki wyznacza graf NEC, nie warstwy rankingu. Etapy pośrednie: POS nad pojedynczą jednostką celu. Ostatni etap: ten sam zbiór wszystkich maksymalnych klas NEC, niezależnie od gałęzi.</p>"
    report += "<section><h2>Co dokładnie porównujemy</h2><p><b>POS parami:</b> dla każdej jednostki celu istnieją wagi dające przewagę; różne porównania mogą używać różnych wag. <b>Wspólne wagi:</b> istnieje jeden układ wag spełniający wszystkie porównania jednocześnie. Przewaga jest słaba (co najmniej tak dobry), w tolerancji solvera.</p><p>Wspólne LP: max u·y(z), przy v·x(z)=1, u·y(z)≤1, u·y(t)≤v·x(t) dla każdego t ze szczytu, u,v≥0. Cel uznajemy za spełniony przy optimum ≥1−10⁻⁹. Gdy optimum wynosi 1, te same wagi dają kandydatowi score 1 i wszystkim celom score ≤1. To nie jest koniunkcja niezależnych zapytań POS. Model dotyczy dodatnich danych i nieograniczonych nieujemnych wag CCR; nie jest certyfikatem silnej efektywności Pareto.</p>"
    report += "<p>Wybór ścieżki: leksykograficznie MSC, TC, SWBP, FW. Minimalność jest liczona od faktycznego poprzednika w ocenionej puli. Zachowujemy limity źródłowego eksperymentu; przycięcie do frontu oraz limity są heurystyką. W tabeli pokazujemy też najmniejsze TC wśród zapisanych ścieżek.</p></section>"
    report += "<section><h2>Wejście DEA i graf NEC</h2>" + table(baseline) + "<details><summary>Dane X/Y</summary>" + table(reference) + "</details><details><summary>Fronty efektywności z peelingu</summary>" + table(pd.read_csv(root / "dea_input_snapshot/efficiency_fronts.csv")) + "</details>" + matrices
    report += f"<p>Liczba oryginalnych ścieżek grafu od startu: {count}. Końcowe cele wspólne dla wszystkich gałęzi: {html.escape(', '.join(top))}. Gałęzie różniące się tylko końcowym wierzchołkiem są scalone po zastąpieniu końca wspólnym celem.</p>" + table(baseline.loc[baseline.name.isin(top)]) + plot(hasse, "hasse") + "</section>"
    report += "<section><h2>Wynik porównania</h2><p>Wspólne wagi są warunkiem silniejszym: pula końcowa joint jest podzbiorem pairwise. Liczba zapisanych ścieżek nie musi jednak maleć: po zmianie puli ponownie wyznaczamy fronty minimalnych zmian, które mogą ujawnić inne gałęzie. Nie jest to pełna enumeracja wszystkich osiągalnych ścieżek.</p>" + table(summary) + plot(overview, "overview") + "<details><summary>Punkty spełniające tylko POS parami</summary>" + table(differences) + "</details></section>"
    if five_method_root is not None:
        report += "<section><h2>Przywrócone pięć metod</h2><p>Wszystkie siedem wariantów korzysta z tej samej ocenionej puli. Nie dodawano osobnego refinementu metod 3–5. Dwie historyczne metody NEC pozostają punktami odniesienia, nie zastępują poprawionej interpretacji Hasse/POS. Warstwy NEC nie są frontami efektywności. Metody liczbowe realizują cele z parametrów: best_rank, best_efficiency i width; nie ten sam cel końcowy co POS.</p><p>Wybór historycznych metod 3–5: MD, MSC, SWBP, FW; pozostałych: MSC, TC, SWBP, FW. PYV/PYM oraz szerokości mają skalę właściwą metodzie. Limit ścieżek oznacza ograniczony zbiór wyników, nie pełną enumerację.</p>" + table(pd.DataFrame(five_summary))
        report += '<details><summary>Audyt pięciu metod</summary>' + table(pd.read_csv(five_method_root/'validation_checks.csv')) + '</details></section>'
    report += "<section><h2>Wszystkie metryki wybranych ścieżek</h2><p>Najedź na wynik lub użyj Tab. Kliknij, aby przypiąć obliczenia; Esc zamyka. Zerowy koszt kroku oznacza niezdefiniowane PYV/PYM, a nie zero.</p><div class='metric-table'>" + render_metric_table(selected, labels, details) + "</div></section>" + "".join(sections)
    report += "<section><h2>Wagi dla punktów końcowych</h2><p>Dla wariantu parami każda pozycja target może mieć własne wagi. W tabeli wspólnej jeden wiersz zawiera wagi dla wszystkich celów; joint_top=false oznacza, że nie są świadectwem osiągnięcia celu.</p><h3>Osobne porównania</h3>" + table(witnesses.loc[witnesses.name.isin(selected_names)]) + "<h3>Jedno wspólne rozwiązanie</h3>" + table(common.loc[common.name.isin(selected_names)]) + "<details><summary>Kontrola ilorazów przy wspólnych wagach, wszystkie DMU</summary>" + table(weight_audit.loc[weight_audit.name.isin(selected_names)]) + "</details></section>"
    report += "<section><h2>Parametry i audyt</h2><p>Wykorzystano dokładnie siatkę źródłowego eksperymentu Hassego i jego snapshot wejścia. Robust scores i rangi kandydatów przeliczono w Javie i porównano ze źródłem; obie definicje celu i peer sets oceniono ponownie. Nie łączymy tego porównania z dawnymi wynikami celu NEC. Pliki source_manifest.csv i implementation_manifest.csv zapisują pochodzenie i hashe danych oraz kodu.</p>" + table(pd.read_csv(root / "validation_checks.csv")) + "<details><summary>Kompletność metryk</summary>" + table(pd.read_csv(root / "metric_coverage.csv")) + "</details><details><summary>Czasy obliczeń</summary>" + table(pd.read_csv(root / "execution_status.csv")) + "</details><details><summary>Parametry</summary>" + table(pd.read_csv(root / "experiment_params.csv")) + "</details></section>"
    assets = Path(__file__).parent / "report_assets"
    report += "<style>" + (assets / "metric_calculations.css").read_text(encoding="utf-8") + "</style>"
    report += '<aside id="metric-calculation-popup" role="dialog" aria-label="Sposób obliczenia metryki" hidden><div class="calculation-toolbar"><span class="calculation-hint"></span><button type="button" class="calculation-close" aria-label="Zamknij obliczenia">×</button></div><div class="calculation-content" tabindex="0"></div></aside>'
    report += "<script>" + (assets / "metric_calculations.js").read_text(encoding="utf-8") + "</script><script>document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{if(d.open)d.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.Plots.resize(p));}));</script></main></body></html>"
    report = report.replace("Wykorzystano dokładnie siatkę źródłowego eksperymentu Hassego i jego snapshot wejścia. Robust scores i rangi kandydatów przeliczono w Javie i porównano ze źródłem;", "Zachowano snapshot i zakres źródłowego eksperymentu Hassego. Budowę puli opisują parametry i candidate_sampling.csv. Robust scores i rangi wszystkich kandydatów przeliczono w Javie; punkty zachowane ze źródła porównano ze starymi wynikami;")
    if params.get("sampling_mode") == "dense_grid_with_jitter":
        sampling = "<section><h2>Zagęszczona siatka i małe odchylenia</h2><p>"
        sampling += f"Poziomy na osi: {params['grid_levels']}; próbek na centrum: {params['jitter_samples']}; promień: {params['jitter_fraction']} kroku nowej siatki; seed: {params['sampling_seed']}. "
        sampling += "Losowanie jednostajne w przecięciu lokalnego pudełka z zakresem źródłowej siatki. Wszystkie próbki oceniono w Javie i należą do puli selekcyjnej obu wariantów. Output pozostaje stały. Szum dotyczy współrzędnych kandydatów, nie wyników DEA. Nie jest to dowód minimum w przestrzeni ciągłej.</p>"
        sampling += table(candidates.groupby('sampling_source').size().reset_index(name='liczba')) + "</section>"
        report = report.replace("<section><h2>Wejście DEA", sampling + "<section><h2>Wejście DEA")
    if five_method_root is not None:
        report = report.replace(f"{html.escape(params['target'])}: dwa warianty celu POS", f"{html.escape(params['target'])}: pięć metod i dwa warianty Hasse/POS")
        report = report.replace('Hasse: POS parami i wspólne wagi</title>', 'DEA: pięć metod i dwa warianty POS</title>')
        report = report.replace('</main>', '<section><h2>Czasy pięciu metod</h2>' + table(pd.read_csv(five_method_root/'execution_status.csv')) + '</section></main>')
    report = report.replace('<section><h2>Co dokładnie porównujemy', dataset_section(reference, params) + '<section><h2>Co dokładnie porównujemy', 1)
    (root / "hasse_pos_comparison.html").write_text(report, encoding="utf-8")
    (root / "README_results.md").write_text("# Hasse: dwa warianty POS\n\nOtworz hasse_pos_comparison.html.\n\n" + markdown(summary)
        + "\n\nGraf NEC wyznacza sciezki. Etapy posrednie: POS nad pojedyncza DMU. Ostatni cel jest wspolny dla wszystkich galezi: "
        + ", ".join(top) + ". Dwa kwantyfikatory: osobne wagi dla kazdej DMU albo jedne wspolne wagi.\n\nMinimalnosc dotyczy ocenionej puli. Zerowe kroki nie sa sztucznie usuwane; PYV/PYM moga byc niezdefiniowane.\n", encoding="utf-8")
