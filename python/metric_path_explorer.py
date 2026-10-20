"""Single-criterion path choices and self-contained interactive report panels."""

import base64
import gzip
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_reporting import metric_catalog
from metric_calculation_details import explain_metrics, render_metric_table
from path_selection import TOLERANCE, tolerance_lexicographic_choice


def metric_choices(metrics, catalog=None):
    """Optimize each catalog criterion on all finite saved values, including ties."""
    catalog = metric_catalog() if catalog is None else catalog
    if metrics.path_id.duplicated().any():
        raise ValueError('Metric explorer requires unique path IDs')
    records, ties = [], []
    for definition in catalog.itertuples(index=False):
        direction = definition.direction
        if direction not in ('lower_better', 'higher_better'):
            raise ValueError('Unknown metric direction: '+direction)
        values = pd.to_numeric(metrics.get(definition.metric, pd.Series(np.nan,index=metrics.index)), errors='coerce')
        finite = np.isfinite(values)
        usable = metrics.loc[finite]
        record = dict(metric=definition.metric, direction=direction, description=definition.plain_description,
                      tested_paths=len(metrics), finite_paths=int(finite.sum()), excluded_paths=int((~finite).sum()),
                      tie_count=0, constant=False, path_id=None, optimum=None, selected_value=None)
        if not usable.empty:
            optimum = float(values[finite].min() if direction == 'lower_better' else values[finite].max())
            mask = finite & ((values-optimum).abs() <= TOLERANCE)
            winners = metrics.loc[mask]
            choice = tolerance_lexicographic_choice(winners, ['tc','msc'])
            if choice is None:
                choice = winners.sort_values('path_id',kind='stable').iloc[0]
            record.update(path_id=str(choice.path_id), optimum=optimum,
                          selected_value=float(values.loc[choice.name]), tie_count=len(winners),
                          constant=bool(float(values[finite].max()-values[finite].min()) <= TOLERANCE))
            ties.extend(dict(metric=definition.metric, path_id=str(row.path_id), value=float(values.loc[index]))
                        for index,row in winners.iterrows())
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(ties,columns=['metric','path_id','value'])


def pack_bundle(bundle):
    raw = json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    return base64.b64encode(gzip.compress(raw,mtime=0)).decode('ascii')


def build_metric_explorer(run, method, default_metric, default_path, reference, columns, ranges, params):
    # Local import avoids a cycle with the report that hosts these panels.
    from generate_five_method_report import LABELS, method_figure, table

    metrics = pd.read_csv(run/'path_metrics.csv')
    paths = pd.read_csv(run/'paths.csv')
    selection, ties = metric_choices(metrics)
    selection.to_csv(run/'metric_explorer_selection.csv',index=False)
    ties.to_csv(run/'metric_explorer_ties.csv',index=False)
    by_metric = metrics.set_index('path_id',drop=False)
    by_path = paths.set_index('path_id',drop=False)
    stage_pool = pd.read_csv(run/'stage_candidates.csv')
    if default_metric.cdir_scope == 'evaluated_final_stage_pool':
        direct = stage_pool.loc[stage_pool.stage.eq(stage_pool.stage.max()), ['name','effort_from_start',*columns]].rename(columns={'effort_from_start':'direct_cost'})
    else:
        direct = metrics[['path_id','cdir_endpoint']].rename(columns={'path_id':'name','cdir_endpoint':'direct_cost'})
        for column in columns:
            endpoints = {row.path_id: row[f'stage_{int(row.path_length):02d}_{column}'] for _,row in paths.iterrows()}
            direct[column] = direct.name.map(endpoints)
    key = 'explorer-'+method
    bundles, audit_rows = {}, []
    representatives = sorted({str(default_path.path_id),*selection.path_id.dropna().astype(str)})
    for path_id in representatives:
        metric, path = by_metric.loc[path_id], by_path.loc[path_id]
        figure, states, audits = method_figure(run,method,path,reference,columns,ranges,params)
        if any(not row['selected_on_front'] or row['dominating_points'] for row in audits):
            raise ValueError(f'Invalid explorer transition front: {method}/{path_id}')
        audit_rows += [dict(path_id=path_id,**row) for row in audits]
        figure.update_layout(template='plotly_white',height=780,margin=dict(l=55,r=30,t=180,b=90),
                             legend=dict(orientation='h',y=-.13),uirevision=key)
        details = explain_metrics(metric,path,params,direct,stage_pool)
        metric_html = render_metric_table({key:(metric,path,'#17324a')},{key:LABELS[method]},{key:details})
        keep = ['stage','name',*columns,*[c for c in states if c.startswith('o') and c[1:].isdigit()],
                'best_efficiency','worst_efficiency','best_rank','worst_rank','score_width','rank_width',
                'effort_from_previous','transition_reference_name','peer_refs']
        keep += [c for c in states if c.startswith('milestone_')]
        bundle = dict(path_id=path_id,figure=json.loads(figure.to_json()),
                      stages_html=table(states.reindex(columns=list(dict.fromkeys(keep)))),
                      metrics_html=metric_html)
        bundles[path_id] = pack_bundle(bundle)
    pd.DataFrame(audit_rows).to_csv(run/'metric_explorer_transition_audit.csv',index=False)
    payload = dict(method=method,label=LABELS[method],default_path=str(default_path.path_id),
                   choices=selection.astype(object).where(pd.notna(selection),None).to_dict('records'),bundles=bundles)
    # JSON is an inert data block, but HTML parsers still recognize a closing script tag.
    encoded = json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    options = ['<option value="default">Dotychczasowy wybór leksykograficzny</option>']
    for row in selection.itertuples(index=False):
        disabled = ' disabled' if row.path_id is None else ''
        goal = 'minimum' if row.direction == 'lower_better' else 'maksimum'
        label = f'{row.metric.upper()} - {goal}' + (' (brak wartości skończonych)' if disabled else '')
        options.append(f'<option value="{html.escape(row.metric)}"{disabled}>{html.escape(label)}</option>')
    count = len(representatives)
    panel = f'<div class="path-explorer" id="{key}"><h3>Wybór ścieżki według metryki</h3>'
    panel += f'<p>Oceniono {len(metrics)} zapisanych ścieżek. Poszczególne kryteria i wybór domyślny wskazują {count} różnych reprezentantów (według ID). Zakres: ścieżki po selekcji frontów i limitach eksperymentu.</p>'
    panel += f'<label for="{key}-criterion">Metryka wyboru ścieżki: </label><select id="{key}-criterion" class="explorer-criterion">'+''.join(options)+'</select>'
    panel += '<p class="explorer-status" role="status" aria-live="polite">Wybierz metrykę lub uruchom wykres.</p>'
    panel += '<button type="button" class="explorer-load">Uruchom wykres</button>'
    panel += f'<div id="{key}-plot" class="explorer-plot"></div><div class="explorer-stages"></div>'
    panel += '<details><summary>Wszystkie metryki aktualnie wybranej ścieżki i rachunki</summary><div class="explorer-metrics metric-table"></div></details>'
    panel += '<details><summary>Najlepsze wyniki według wszystkich 20 metryk</summary>'+table(selection)+'</details>'
    panel += f'<script type="application/json" class="explorer-data">{encoded}</script></div>'
    return panel, selection.assign(method=method)


def explorer_assets():
    assets = Path(__file__).with_name('report_assets')
    return ('<style>.path-explorer select{font:inherit;padding:8px;max-width:100%}.explorer-load{padding:8px 14px;cursor:pointer}'
            '.explorer-status{min-height:3em}.explorer-plot{min-height:30px}</style><script>'
            +(assets/'metric_path_explorer.js').read_text(encoding='utf-8')+'</script>')
