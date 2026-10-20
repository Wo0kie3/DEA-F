"""Presentation-only color normalization and stage visibility for search figures."""

import argparse
import base64
import gzip
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


def score_limits(values):
    values = pd.to_numeric(pd.Series(values),errors='coerce')
    values = values[np.isfinite(values)]
    return (float(values.min()),float(values.max())) if len(values) else (0.,1.)


def describe_search_style(report):
    return report.replace('Warstwy diagnostyczne można włączyć z legendy; ich kolor koduje best_efficiency w skali Viridis 0–1.',
        'Wszystkie kategorie punktów wybranego etapu oraz oryginalne DMU są domyślnie widoczne. '
        'Kolory E* wykorzystują pełną skalę 0–1 w zakresie ocenionej puli danej metody: minimum mapujemy na 0, maksimum na 1. '
        'Legenda i podpowiedzi pokazują rzeczywiste wartości efektywności. Kliknięcie legendy pozwala ukrywać kategorie.')


def style_search_figure(figure, limits):
    low, high = map(float,limits)
    for trace in figure['data']:
        marker = trace.get('marker',{})
        if 'colorscale' not in marker:
            continue
        scores = [float(row[1]) for row in trace['customdata']]
        marker['color'] = [max(0.,min(1.,(value-low)/(high-low))) if high > low else .5 for value in scores]
        marker.update(cmin=0,cmax=1,opacity=.6)
        ticks = np.linspace(0,1,5).tolist()
        marker['colorbar'].update(title=dict(text='Efektywność E*'),tickvals=ticks,
            ticktext=[f'{low+t*(high-low):.4g}' for t in ticks])
    menus = figure['layout'].get('updatemenus',[])
    for menu in menus:
        for button in menu.get('buttons',[]):
            visibility = button['args'][0].get('visible')
            if visibility is not None:
                button['args'][0]['visible'] = [True if shown == 'legendonly' else shown for shown in visibility]
    if menus and menus[0].get('buttons'):
        active = menus[0].get('active',0)
        for trace,shown in zip(figure['data'],menus[0]['buttons'][active]['args'][0]['visible']):
            trace['visible'] = shown
    figure['layout'].setdefault('meta',{})['score_color_range'] = [low,high]
    style_search_layout(figure)
    return figure


def style_search_layout(figure):
    layout = figure['layout']
    title = layout.get('title',{})
    if isinstance(title,str):
        title = dict(text=title)
    title.update(x=0,xref='paper',xanchor='left',y=.94,yref='container',yanchor='top',font=dict(size=18))
    title['text'] = title.get('text','').replace(' · ','<br>')
    layout['title'] = title
    layout['height'] = 780
    layout.setdefault('margin',{}).update(t=180,b=90)
    for menu in layout.get('updatemenus',[]):
        menu.update(x=0,xanchor='left',y=1.04,yanchor='bottom',pad=dict(t=0,b=6,l=0,r=0))
        for button in menu.get('buttons',[]):
            update = button['args'][1]
            text = update.pop('title',None)
            if isinstance(text,dict):
                text = text.get('text')
            if text is not None:
                update['title.text'] = text
            if 'title.text' in update:
                update['title.text'] = update['title.text'].replace(' · ','<br>')
    return figure


def restyle_report(report, limits):
    """Refresh embedded explorer figures without repeating scientific calculations."""
    from metric_path_explorer import pack_bundle
    offsets = [0]
    for line in report.splitlines(keepends=True):
        offsets.append(offsets[-1]+len(line))
    class Blocks(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.start = None
            self.edits = []
        def position(self):
            line,col = self.getpos()
            return offsets[line-1]+col
        def handle_starttag(self,tag,attrs):
            a = dict(attrs)
            if tag == 'script' and a.get('class') == 'explorer-data':
                self.start = self.position()+len(self.get_starttag_text())
        def handle_endtag(self,tag):
            if tag != 'script' or self.start is None:
                return
            end = self.position()
            data = json.loads(report[self.start:end])
            for path_id,encoded in data['bundles'].items():
                bundle = json.loads(gzip.decompress(base64.b64decode(encoded)))
                bundle['figure'] = style_search_figure(bundle['figure'],limits)
                data['bundles'][path_id] = pack_bundle(bundle)
            text = json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
            self.edits.append((self.start,end,text))
            self.start = None
    parser = Blocks()
    parser.feed(report)
    for start,end,text in reversed(parser.edits):
        report = report[:start]+text+report[end:]
    return describe_search_style(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    args = parser.parse_args()
    limits = score_limits(pd.read_csv(args.candidates)['best_efficiency'])
    original = args.report.read_text(encoding='utf-8')
    updated = restyle_report(original,limits)
    args.report.write_text(updated,encoding='utf-8')
