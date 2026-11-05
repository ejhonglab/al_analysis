#!/usr/bin/env python3
"""Converts one of our markdown notes into a self-contained artifact page (HTML).

Handles the markdown our notes use: `#` / `##` / `###` headings, paragraphs and list items
wrapped over several lines (joined, so no line breaks mid-sentence), numbered and bulleted
lists (continuation lines indented or simply following the item until a blank line or the
next item), pipe tables (numeric cells right-aligned), figures written as
`![caption](path.png)` on their own line (embedded as data URIs, so the page is one file),
`---` rules, collapsible blocks (a line `??? Title` opens one, a line `???` closes it; rendered
as <details>, collapsed by default), hover definitions (a block opened by a line `%%% hover` and
closed by `%%%`, each line `term: definition`; every whole-word occurrence of the term in
paragraphs, list items and table cells gets a dotted underline and shows the definition on
hover; headings and captions are left alone), heading anchors (a heading ending in `{#id}` gets
that id; others get a slug of their text) and internal links `[text](#id)` besides http links,
a table of contents (a line `[TOC]` is replaced by a two-level nested list of the page's
`##` and `###` headings, linked to their anchors), and inline `code`, **bold**, *italic*. The
first paragraph after the title is styled as the byline. After converting, `lint` fails on anything that still looks
like markdown (leftover `#`, `**`, backticks, `](`), which is how the "### heading" and
split-sentence defects of hand-converted pages are caught.

Usage: ./venv/bin/python md_to_artifact.py note.md page.html [--figdir DIR] [--title T]
Import: convert_markdown(md_text, figdir) -> html; lint(html) -> list of problems.
"""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import re
import sys
from pathlib import Path
from typing import List, Optional

CSS = """
  :root { --ink: #1f2328; --muted: #57606a; --line: #d8dee4; --bg: #ffffff; --soft: #f6f8fa; --accent: #2a78d6; }
  @media (prefers-color-scheme: dark) { :root { --ink: #e6edf3; --muted: #9da7b3; --line: #30363d; --bg: #0d1117; --soft: #161b22; } }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 15.5px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  main { max-width: 920px; margin: 0 auto; padding: 32px 22px 70px; }
  h1 { font-size: 1.7em; line-height: 1.25; margin: 0 0 6px; }
  h2 { font-size: 1.25em; margin: 36px 0 10px; padding-top: 12px; border-top: 1px solid var(--line); }
  h3 { font-size: 1.05em; margin: 24px 0 6px; }
  p { margin: 9px 0; }
  p.meta { color: var(--muted); font-size: 0.9em; }
  table { border-collapse: collapse; width: 100%; font-size: 0.84em; margin: 12px 0 16px; }
  th, td { border-bottom: 1px solid var(--line); padding: 5px 7px; text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  ol, ul { padding-left: 24px; } li { margin: 5px 0; }
  code { font-size: 0.9em; background: var(--soft); padding: 1px 4px; border-radius: 4px; }
  figure { margin: 18px 0; }
  figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
  figcaption { color: var(--muted); font-size: 0.86em; margin-top: 6px; }
  hr { border: 0; border-top: 1px solid var(--line); margin: 24px 0; }
  details.fold { margin: 10px 0 18px; border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; background: var(--soft); }
  details.fold summary { cursor: pointer; font-weight: 600; color: var(--muted); }
  details.fold ul { margin: 8px 0 4px; }
  a { color: var(--accent); }
  span.def { border-bottom: 1px dotted var(--muted); cursor: help; text-decoration: none; }
  nav.toc { margin: 10px 0 18px; padding: 10px 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--soft); font-size: 0.92em; }
  nav.toc ol { margin: 0; padding-left: 20px; } nav.toc li { margin: 2px 0; } nav.toc > ol > li { margin: 4px 0; }
  nav.toc a { text-decoration: none; } nav.toc a:hover { text-decoration: underline; }
"""

_ITEM_RE = re.compile(r'^(\d+)\. (.*)')
_BULLET_RE = re.compile(r'^[-*] (.*)')
_FIG_RE = re.compile(r'^!\[(.*?)\]\((\S+?)\)\s*$')


def inline(s: str) -> str:
    s = html_mod.escape(s, quote=False)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<![*\w])\*([^*\n]+?)\*(?![*\w])', r'<i>\1</i>', s)
    s = re.sub(r'\[([^\]]+)\]\(((?:https?://|#)[^)\s]+)\)', r'<a href="\2">\1</a>', s)
    s = re.sub(r'(?<![">])(https?://[^\s<)]+)', r'<a href="\1">\1</a>', s)
    return s


def _is_block_start(ln: str) -> bool:
    return bool(re.match(r'^(#{1,3} |\||\d+\. |[-*] |---\s*$|!\[|\?\?\?|%%%|\[TOC\]\s*$)', ln))


def build_toc(body: str) -> str:
    """A two-level nested list (h2 > h3) of the body's headings, linked to their ids."""
    heads = [(int(l), i, re.sub(r'<[^>]+>', '', t)) for l, i, t in re.findall(r'<h([23]) id="([^"]+)">(.*?)</h[23]>', body, flags=re.S)]
    if not heads:
        return ''
    out = ['<nav class="toc"><ol>']
    open_sub = False
    for level, hid, text in heads:
        if level == 2:
            if open_sub:
                out.append('</ol>'); open_sub = False
            if len(out) > 1:
                out.append('</li>')
            out.append(f'<li><a href="#{hid}">{text}</a>')
        else:
            if len(out) == 1:
                out.append('<li>')
            if not open_sub:
                out.append('<ol>'); open_sub = True
            out.append(f'<li><a href="#{hid}">{text}</a></li>')
    if open_sub:
        out.append('</ol>')
    out.append('</li></ol></nav>')
    return ''.join(out)


def _slug(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:60] or 'section'


def with_defs(html: str, defs: dict) -> str:
    """Wraps whole-word occurrences of each defined term (outside tags) in a hover span."""
    if not defs:
        return html
    parts = re.split(r'(<[^>]+>)', html)
    for k, part in enumerate(parts):
        if part.startswith('<'):
            continue
        for term, definition in defs.items():
            part = re.sub(r'(?<![\w-])(' + re.escape(term) + r')(?![\w-])',
                lambda m: f'<span class="def" title="{html_mod.escape(definition, quote=True)}">{m.group(1)}</span>', part)
        parts[k] = part
    return ''.join(parts)


def _data_uri(path: Path) -> str:
    mime = 'image/svg+xml' if path.suffix == '.svg' else f'image/{path.suffix.lstrip(".").replace("jpg", "jpeg")}'
    return f'data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}'


def convert_body(md: str, figdir: Optional[Path] = None, *, byline: bool = True, defs: Optional[dict] = None) -> str:
    """Converts markdown to the body HTML (no page wrapper)."""
    lines = md.split('\n')
    out: List[str] = []
    i = 0
    seen_h1 = False
    byline_done = not byline
    defs = dict(defs or {})
    # hover-definition blocks are collected first (they may sit anywhere) and removed
    kept, j, in_defs = [], 0, False
    for ln in lines:
        if ln.strip() == '%%% hover':
            in_defs = True; continue
        if in_defs and ln.strip() == '%%%':
            in_defs = False; continue
        if in_defs:
            if ':' in ln:
                term, definition = ln.split(':', 1); defs[term.strip()] = definition.strip()
            continue
        kept.append(ln)
    lines = kept

    def text_html(s: str) -> str:
        return with_defs(inline(s), defs)

    def take_item(first: str) -> str:
        """Returns the item text with its continuation lines joined."""
        nonlocal i
        parts = [first]
        i += 1
        while i < len(lines) and lines[i].strip() and not _is_block_start(lines[i].lstrip()) \
                and not _ITEM_RE.match(lines[i]) and not _BULLET_RE.match(lines[i]):
            parts.append(lines[i].strip()); i += 1
        return ' '.join(parts)

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s == '':
            i += 1; continue
        m = re.match(r'^(#{1,3}) (.*)', s)
        if m:
            level = len(m.group(1))
            htxt = m.group(2)
            idm = re.search(r'\s*\{#([A-Za-z0-9_-]+)\}\s*$', htxt)
            hid = idm.group(1) if idm else _slug(htxt)
            if idm:
                htxt = htxt[:idm.start()]
            out.append(f'<h{level} id="{hid}">{inline(htxt)}</h{level}>')
            if level == 1:
                seen_h1 = True
            else:
                byline_done = True   # a byline is only a paragraph directly under the title
            i += 1; continue
        if re.match(r'^---\s*$', s):
            out.append('<hr>'); i += 1; continue
        if re.match(r'^\[TOC\]\s*$', s):
            out.append('@@TOC@@'); i += 1; continue
        if s.startswith('??? '):
            title_txt = s[4:].strip(); inner = []; i += 1
            while i < len(lines) and lines[i].strip() != '???':
                inner.append(lines[i]); i += 1
            i += 1   # the closing ???
            out.append(f'<details class="fold"><summary>{inline(title_txt)}</summary>'
                + convert_body('\n'.join(inner), figdir, byline=False, defs=defs) + '</details>')
            continue
        fm = _FIG_RE.match(s)
        if fm:
            cap, src = fm.group(1), fm.group(2)
            p = Path(src)
            if not p.is_absolute():
                p = (figdir or Path('.')) / p
            if not p.exists():
                raise FileNotFoundError(f'figure not found: {p}')
            out.append(f'<figure><img src="{_data_uri(p)}" alt="{html_mod.escape(cap)}">'
                + (f'<figcaption>{inline(cap)}</figcaption>' if cap else '') + '</figure>')
            i += 1; continue
        if s.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip()); i += 1
            cells = [[c.strip() for c in r.strip('|').split('|')] for r in rows]
            body_rows = [r for r in cells[1:] if not all(re.fullmatch(r':?-{2,}:?', c) for c in r)]
            t = ['<table>', '<tr>' + ''.join(f'<th>{text_html(c)}</th>' for c in cells[0]) + '</tr>']
            for r in body_rows:
                t.append('<tr>' + ''.join(
                    f'<td class="{"num" if re.fullmatch(r"[−+\-]?\d[\d.,]*(\s*[\[(].*)?|—|-", c) else "txt"}">'
                    f'{text_html(c)}</td>' for c in r) + '</tr>')
            t.append('</table>')
            out.append('\n'.join(t)); continue
        m = _ITEM_RE.match(s)
        if m:
            start = int(m.group(1)); items = []
            while i < len(lines) and _ITEM_RE.match(lines[i].strip()):
                items.append(take_item(_ITEM_RE.match(lines[i].strip()).group(2)))
                while i < len(lines) and lines[i].strip() == '' and i + 1 < len(lines) \
                        and _ITEM_RE.match(lines[i + 1].strip()):
                    i += 1   # blank line between items of the same list
            out.append(f'<ol start="{start}">' + ''.join(f'<li>{text_html(x)}</li>' for x in items) + '</ol>')
            continue
        m = _BULLET_RE.match(s)
        if m:
            items = []
            while i < len(lines) and _BULLET_RE.match(lines[i].strip()):
                items.append(take_item(_BULLET_RE.match(lines[i].strip()).group(1)))
                while i < len(lines) and lines[i].strip() == '' and i + 1 < len(lines) \
                        and _BULLET_RE.match(lines[i + 1].strip()):
                    i += 1
            out.append('<ul>' + ''.join(f'<li>{text_html(x)}</li>' for x in items) + '</ul>')
            continue
        para = [s]; i += 1
        while i < len(lines) and lines[i].strip() and not _is_block_start(lines[i].strip()):
            para.append(lines[i].strip()); i += 1
        text = ' '.join(para)
        if seen_h1 and not byline_done:
            out.append(f'<p class="meta">{inline(text)}</p>'); byline_done = True
        else:
            out.append(f'<p>{text_html(text)}</p>')
    return '\n'.join(out)


def convert_markdown(md: str, figdir: Optional[Path] = None, title: Optional[str] = None) -> str:
    body = convert_body(md, figdir)
    if '@@TOC@@' in body:
        body = body.replace('@@TOC@@', build_toc(body.replace('@@TOC@@', '')))
    if title is None:
        mt = re.search(r'<h1[^>]*>(.*?)</h1>', body)
        title = re.sub(r'<[^>]+>', '', mt.group(1)) if mt else 'page'
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n'
        f'</main>\n</body>\n</html>\n')


def lint(page: str) -> List[str]:
    """Problems that mean markdown survived conversion (empty list = clean)."""
    text = re.sub(r'<img [^>]*>', '', page)
    text = re.sub(r'<style>.*?</style>', '', text, flags=re.S)
    # heading text may legitimately start with "1." or "-": drop headings before the checks
    text = re.sub(r'<h[1-3][^>]*>.*?</h[1-3]>', '', text, flags=re.S)
    visible = re.sub(r'<[^>]+>', '\n', text)
    probs = []
    for ln in visible.split('\n'):
        t = ln.strip()
        if not t:
            continue
        if re.match(r'^#{1,6} ', t):
            probs.append(f'heading marker left in text: {t[:70]!r}')
        if '**' in t:
            probs.append(f'bold marker left in text: {t[:70]!r}')
        if '`' in t:
            probs.append(f'backtick left in text: {t[:70]!r}')
        if re.search(r'\]\(', t):
            probs.append(f'link syntax left in text: {t[:70]!r}')
        if t.startswith('???'):
            probs.append(f'collapsible-block marker left in text: {t[:70]!r}')
        if t.startswith('%%%'):
            probs.append(f'hover-block marker left in text: {t[:70]!r}')
        if t == '[TOC]' or t == '@@TOC@@':
            probs.append('table-of-contents marker left in text')
        if re.search(r'\{#[A-Za-z0-9_-]+\}', t):
            probs.append(f'heading id marker left in text: {t[:70]!r}')
        if re.match(r'^(\d+\. |[-*] )', t) and len(t) > 3:
            probs.append(f'list marker left in text: {t[:70]!r}')
    if re.search(r'<li>[^<]*<br', page):
        probs.append('line break inside a list item')
    return probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('markdown', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--figdir', type=Path, default=None, help='directory figure paths are relative to (default: the markdown file\'s)')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()
    page = convert_markdown(args.markdown.read_text(), args.figdir or args.markdown.parent, args.title)
    probs = lint(page)
    args.out.write_text(page)
    counts = {tag: len(re.findall(f'<{tag}[ >]', page)) for tag in ('h1', 'h2', 'h3', 'p', 'ol', 'ul', 'li', 'table', 'figure')}
    print(f'wrote {args.out} ({len(page) // 1024} KB); elements {counts}')
    if probs:
        print('LINT PROBLEMS:'); print('\n'.join(' - ' + p for p in probs)); sys.exit(1)
    print('lint clean')


if __name__ == '__main__':
    main()
