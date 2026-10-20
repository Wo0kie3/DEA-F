"""Number report headings while preserving scripts, templates and figure data."""

import argparse
import html
from html.parser import HTMLParser
from pathlib import Path
import re


def unnumber(title):
    return re.sub(r'^\d+(?:\.\d+)*\.?\s+', '', title.strip())


def number_report_sections(document, method_labels):
    methods = {unnumber(label) for label in method_labels}
    offsets = [0]
    for line in document.splitlines(keepends=True):
        offsets.append(offsets[-1]+len(line))

    class Headings(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.blocked = 0
            self.heading = None
            self.edits = []
            self.main = self.sub = self.leaf = 0
            self.in_method = False
            self.in_nav = False
            self.link = None
            self.links = []
            self.method_headings = {}

        def source_position(self):
            line, column = self.getpos()
            return offsets[line-1]+column

        def handle_starttag(self, tag, attrs):
            if tag == 'nav':
                self.in_nav = True
            if tag == 'a' and self.in_nav:
                self.link = (self.source_position()+len(self.get_starttag_text()),[])
            if tag in ('script','style','template','aside'):
                self.blocked += 1
            if not self.blocked and tag in ('h2','h3','h4'):
                self.heading = (tag,self.source_position(),[])

        def handle_data(self, data):
            if self.heading is not None:
                self.heading[2].append(data)
            if self.link is not None:
                self.link[1].append(data)

        def handle_entityref(self, name):
            self.handle_data(html.unescape('&'+name+';'))

        def handle_charref(self, name):
            self.handle_data(html.unescape('&#'+name+';'))

        def handle_endtag(self, tag):
            if tag == 'a' and self.link is not None:
                self.links.append((self.link[0],self.source_position(),unnumber(''.join(self.link[1]))))
                self.link = None
            if tag == 'nav':
                self.in_nav = False
            if self.heading is not None and tag == self.heading[0]:
                _, start, parts = self.heading
                title = unnumber(''.join(parts))
                if title in methods:
                    self.sub += 1
                    self.leaf = 0
                    self.in_method = True
                    level, number = 'h3',f'{self.main}.{self.sub}'
                    self.method_headings[title] = f'{number}. {title}'
                elif tag == 'h2':
                    self.main += 1
                    self.sub = self.leaf = 0
                    self.in_method = False
                    level, number = 'h2',str(self.main)
                elif self.in_method:
                    self.leaf += 1
                    level, number = 'h4',f'{self.main}.{self.sub}.{self.leaf}'
                else:
                    self.sub += 1
                    level, number = 'h3',f'{self.main}.{self.sub}'
                replacement = f'<{level}>{number}. {html.escape(title)}</{level}>'
                self.edits.append((start,self.source_position()+len('</'+tag+'>'),replacement))
                self.heading = None
            if tag in ('script','style','template','aside'):
                self.blocked -= 1

    parser = Headings()
    parser.feed(document)
    for start,end,title in parser.links:
        if title in parser.method_headings:
            parser.edits.append((start,end,html.escape(parser.method_headings[title])))
    for start,end,replacement in sorted(parser.edits,reverse=True):
        document = document[:start]+replacement+document[end:]
    return document


if __name__ == '__main__':
    from generate_five_method_report import LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args = parser.parse_args()
    original = args.input.read_text(encoding='utf-8')
    args.input.write_text(number_report_sections(original,LABELS.values()),encoding='utf-8')
