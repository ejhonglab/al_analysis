#!/usr/bin/env python3
"""Builds the HTML page of the report markdown in this folder ("APL and the ORN→KC remapping
of odor–odor correlations.md"), for publishing as the report artifact.

The markdown is the source of truth: when new experiments are added to the shared
`experiment_priorities.md`, they are appended to the report's "In silico experiments" list
(with a note saying when and from what they were conceived), this script is run, and the
resulting page is republished to the existing artifact (see CLAUDE.md, "Report artifact").

Handles the subset of markdown the report uses: `#` / `##` headings, paragraphs, pipe tables,
numbered lists (numbering preserved, so a list may continue after a note paragraph), bullet
and task lists, bold, inline code, links (http(s) URLs and in-page `#id` anchors; a heading ending
in `{#id}` gets that id, e.g. `## Footnotes {#footnotes}`), and the escapes `\\~`, `\\[`, `\\]`,
`\\_`. A paragraph starting with `*Added` is styled as a provenance note.

Usage: ./venv/bin/python build_report_artifact.py [--out PATH]
"""
import argparse
import datetime as dt
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / 'apl_orn_to_kc_remapping_report.md'
DEFAULT_OUT = HERE / 'outputs' / 'report_artifact' / 'report.html'

CSS = """
  :root { --ink: #1f2328; --muted: #57606a; --line: #d8dee4; --bg: #ffffff; --soft: #f6f8fa; --accent: #2a78d6; }
  @media (prefers-color-scheme: dark) { :root { --ink: #e6edf3; --muted: #9da7b3; --line: #30363d; --bg: #0d1117; --soft: #161b22; } }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 15.5px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  main { max-width: 880px; margin: 0 auto; padding: 32px 22px 70px; }
  h1 { font-size: 1.7em; line-height: 1.25; margin: 0 0 6px; }
  h2 { font-size: 1.25em; margin: 36px 0 10px; padding-top: 12px; border-top: 1px solid var(--line); }
  p { margin: 10px 0; }
  .meta { color: var(--muted); font-size: 0.9em; }
  table { border-collapse: collapse; width: 100%; font-size: 0.88em; margin: 12px 0 16px; }
  th, td { border-bottom: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  ol, ul { padding-left: 24px; } li { margin: 8px 0; }
  li.task { list-style: none; margin-left: -18px; }
  code { font-size: 0.88em; background: var(--soft); padding: 1px 4px; border-radius: 4px; }
  a { color: var(--accent); }
  .added { background: var(--soft); border-left: 3px solid var(--accent); padding: 8px 12px; margin: 14px 0 6px; }
  ol.added-list { border-left: 3px solid var(--accent); padding-left: 32px; margin-left: 0; }
"""


def inline(s: str) -> str:
    s = s.replace('\\~', '~').replace('\\[', '[').replace('\\]', ']').replace('\\_', '_')
    s = html.escape(s, quote=False).replace('&amp;#91;', '&#91;')
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<![*\w])\*([^*\n]+?)\*(?![*\w])', r'<i>\1</i>', s)
    s = re.sub(r'\[([^\]]+)\]\((https?://[^)\s]+|#[A-Za-z0-9_-]+)\)', r'<a href="\2">\1</a>', s)
    return s


def heading(tag: str, text: str) -> str:
    """A heading element; a trailing `{#id}` in the markdown becomes its id (the target of `[...](#id)`
    links, e.g. the Footnotes section)."""
    m = re.search(r'\s*\{#([A-Za-z0-9_-]+)\}\s*$', text)
    if m:
        return f'<{tag} id="{m.group(1)}">{inline(text[:m.start()])}</{tag}>'
    return f'<{tag}>{inline(text)}</{tag}>'


def convert(md: str) -> str:
    lines = md.split('\n')
    out, i = [], 0
    max_exp = 0
    after_note = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('# '):
            out.append(heading('h1', ln[2:])); i += 1; continue
        if ln.startswith('## '):
            out.append(heading('h2', ln[3:])); after_note = False; i += 1; continue
        if ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
            t = ['<table>', '<tr>' + ''.join(f'<th>{inline(c)}</th>' for c in cells[0]) + '</tr>']
            for r in cells[2:]:
                t.append('<tr>' + ''.join(
                    f'<td class="{"num" if re.fullmatch(r"[−+\-]?[0-9.]+(\s*\[.*\])?", c) else "txt"}">'
                    f'{inline(c)}</td>' for c in r) + '</tr>')
            t.append('</table>')
            out.append('\n'.join(t)); continue
        m = re.match(r'^(\d+)\. (.*)', ln)
        if m:
            start = int(m.group(1))
            items = []
            while i < len(lines) and re.match(r'^\d+\. ', lines[i]):
                body = re.match(r'^\d+\. (.*)', lines[i]).group(1)
                items.append(body)
                em = re.match(r'\*\*E(\d+)', body)
                if em:
                    max_exp = max(max_exp, int(em.group(1)))
                i += 1
            cls = ' class="added-list"' if after_note else ''
            out.append(f'<ol start="{start}"{cls}>' + ''.join(f'<li>{inline(x)}</li>' for x in items)
                + '</ol>')
            after_note = False
            continue
        if ln.startswith('- '):
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(lines[i][2:]); i += 1
            li = [f'<li class="task">&#9744; {inline(x[4:])}</li>' if x.startswith('[ ] ')
                else f'<li>{inline(x)}</li>' for x in items]
            out.append('<ul>' + ''.join(li) + '</ul>'); continue
        if ln.strip() == '':
            i += 1; continue
        para = [ln]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||\d+\. |- )', lines[i]):
            para.append(lines[i]); i += 1
        text = ' '.join(para)
        if text.startswith('*Added'):
            out.append(f'<p class="added">{inline(text)}</p>'); after_note = True
        elif text.startswith('Nov 3, 2026'):
            # the experiment count is filled in once the whole document has been scanned
            out.append(f'<p class="meta">{inline(text)} · page rebuilt {dt.date.today().isoformat()}'
                '</p>')
        else:
            out.append(f'<p>{inline(text)}</p>')
    body = '\n'.join(out)
    if max_exp:
        body = body.replace('page rebuilt ' + dt.date.today().isoformat(),
            f'page rebuilt {dt.date.today().isoformat()} (experiments through E{max_exp})')
    title = re.search(r'<h1[^>]*>(.*?)</h1>', body).group(1)
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n'
        f'</main>\n</body>\n</html>\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    page = convert(MD.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    n_items = [len(re.findall(r'<li>', ol)) for ol in re.findall(r'<ol[^>]*>.*?</ol>', page, flags=re.S)]
    print(f'wrote {args.out} ({len(page) / 1e3:.0f} kB); ordered lists: {n_items}; '
        f'sections: {re.findall(r"<h2[^>]*>(.*?)</h2>", page)}')


if __name__ == '__main__':
    main()
