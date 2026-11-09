#!/usr/bin/env python3
"""Converts one of our markdown notes into a self-contained artifact page (HTML).

Handles the markdown our notes use: `#` / `##` / `###` headings, paragraphs and list items
wrapped over several lines (joined, so no line breaks mid-sentence), numbered and bulleted
lists (continuation lines indented or simply following the item until a blank line or the
next item), pipe tables (every row before the `|---|` separator is a header row, a header
cell ending in `{span=N}` spans N columns centered; a column whose body cells are mostly
numbers is right-aligned as a whole, header included, and kept narrow so values sit close
together; `<br>` inside a cell is a line break, used to put each parameter of a header that
lists parameters on its own line; a table whose first header cell is "statistic" and whose
other columns are all numeric is a table of statistics compared across columns: its first
column is right-aligned, next to the numbers, and the table is centered on the page, up to
the full window width, scrolling sideways only if wider), figures written as
`![caption](path.png)` on their own line (embedded as data URIs, so the page is one file;
clicking one opens it full screen, closed by a click, Esc or the x),
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
A line `%%% html <path>` (relative to --figdir) inserts that file's HTML verbatim (the summary's
progress plot from summary_progress_plot.py); lint ignores <script> blocks.
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
  th, td { border-bottom: 1px solid var(--line); padding: 5px 7px; text-align: left; vertical-align: bottom; }
  th { color: var(--muted); font-weight: 600; }
  th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; width: 1%; padding-left: 14px; }
  th.span { text-align: center; border-bottom: 1px solid var(--line); }
  tr.subhead th { border-top: 0; }
  div.statwrap { margin: 12px calc(50% - 50vw + 8px) 16px; overflow-x: auto; }
  div.statwrap table.stats { width: auto; margin: 0 auto; }
  table.stats th.num, table.stats td.num { width: auto; }
  table.stats td:first-child, table.stats th:first-child { text-align: right; }
  table.stats td:first-child { min-width: 12em; }
  ol, ul { padding-left: 24px; } li { margin: 5px 0; }
  code { font-size: 0.9em; background: var(--soft); padding: 1px 4px; border-radius: 4px; }
  figure { margin: 18px 0; }
  figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
  figcaption { color: var(--muted); font-size: 0.86em; margin-top: 6px; }
  figure img.zoomable { cursor: zoom-in; }
  .zoom-overlay { display: none; position: fixed; inset: 0; z-index: 1000; background: rgba(0, 0, 0, 0.92); cursor: zoom-out; align-items: center; justify-content: center; }
  .zoom-overlay.open { display: flex; }
  .zoom-overlay img { width: 98vw; height: 98vh; object-fit: contain; background: #fff; border-radius: 4px; }
  .zoom-close { position: absolute; top: 10px; right: 12px; z-index: 1; width: 38px; height: 38px; border-radius: 50%; font-size: 26px; line-height: 36px; padding: 0; color: #fff; background: rgba(0, 0, 0, 0.65); border: 1px solid rgba(255, 255, 255, 0.5); cursor: pointer; }
  hr { border: 0; border-top: 1px solid var(--line); margin: 24px 0; }
  details.fold { margin: 10px 0 18px; border: 1px solid var(--line); border-radius: 8px; padding: 6px 12px; background: var(--soft); }
  details.fold summary { cursor: pointer; font-weight: 600; color: var(--muted); }
  details.fold ul { margin: 8px 0 4px; }
  a { color: var(--accent); }
  span.def { border-bottom: 1px dotted var(--muted); cursor: help; text-decoration: none; }
  span.pv { cursor: help; border-bottom: 1px dotted rgba(127, 127, 127, 0.25); }
  nav.toc { margin: 10px 0 18px; padding: 10px 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--soft); font-size: 0.92em; }
  nav.toc ol { margin: 0; padding-left: 20px; } nav.toc li { margin: 2px 0; } nav.toc > ol > li { margin: 4px 0; }
  nav.toc a { text-decoration: none; } nav.toc a:hover { text-decoration: underline; }
"""

_ITEM_RE = re.compile(r'^(\d+)\. (.*)')
_BULLET_RE = re.compile(r'^[-*] (.*)')
_FIG_RE = re.compile(r'^!\[(.*?)\]\((\S+?)\)\s*$')
_BR_RE = re.compile(r'<br\s*/?>', re.I)


# Click-to-fullscreen for the static figures (img.zoomable): an overlay over the page that
# also asks the browser for real full screen (refused inside an iframe without the fullscreen
# permission, in which case the overlay alone fills the frame); a click, Esc or the x closes it.
ZOOM_JS = """<script>
(function () {
  var imgs = document.querySelectorAll('img.zoomable');
  if (!imgs.length) return;
  var ov = document.createElement('div');
  ov.className = 'zoom-overlay'; ov.setAttribute('role', 'dialog'); ov.setAttribute('aria-modal', 'true');
  ov.innerHTML = '<button class="zoom-close" aria-label="close">\u00d7</button><img alt="">';
  document.body.appendChild(ov);
  var big = ov.querySelector('img');
  function hide() { ov.classList.remove('open'); big.removeAttribute('src'); }
  function close() {
    if (document.fullscreenElement === ov && document.exitFullscreen) document.exitFullscreen().catch(function () {});
    hide();
  }
  function open(img) {
    big.src = img.src; big.alt = img.alt; ov.classList.add('open');
    if (ov.requestFullscreen) ov.requestFullscreen().catch(function () {});
  }
  imgs.forEach(function (img) {
    img.addEventListener('click', function () { open(img); });
    img.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(img); } });
  });
  ov.addEventListener('click', close);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && ov.classList.contains('open')) close(); });
  document.addEventListener('fullscreenchange', function () { if (!document.fullscreenElement && ov.classList.contains('open')) hide(); });
})();
</script>"""


PV = re.compile(r'⟦pv:(.*?)§(.*?)⟧')


def strip_pv(s: str) -> str:
    """A filled report value `⟦pv:value§source⟧` (fill_report.py) reduced to its value."""
    return PV.sub(r'\1', s)


def inline(s: str) -> str:
    s = html_mod.escape(s, quote=False)
    s = PV.sub(lambda m: f'<span class="pv" title="{m.group(2).replace(chr(34), chr(39))}">{m.group(1)}</span>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<![*\w])\*([^*\n]+?)\*(?![*\w])', r'<i>\1</i>', s)
    s = re.sub(r'\[([^\]]+)\]\(((?:https?://|#)[^)\s]+)\)', r'<a href="\2">\1</a>', s)
    s = re.sub(r'(?<![">])(https?://[^\s<)]+)', r'<a href="\1">\1</a>', s)
    return s


def _is_block_start(ln: str) -> bool:
    return bool(re.match(r'^(#{1,3} |\||\d+\. |[-*] |---\s*$|!\[|\?\?\?|%%%|\[TOC\]\s*$)', ln))


def build_toc(body: str) -> str:
    """A two-level nested list (h2 > h3) of the body's headings, linked to their ids. The
    Footnotes section (`## Footnotes`, id `footnotes`) is listed without its sub-entries: every
    footnote is reached from the text that cites it (`lint` checks that each one is linked)."""
    heads = [(int(l), i, re.sub(r'<[^>]+>', '', t)) for l, i, t in re.findall(r'<h([23]) id="([^"]+)">(.*?)</h[23]>', body, flags=re.S)]
    kept, in_foot = [], False
    for level, hid, text in heads:
        if level == 2:
            in_foot = hid == 'footnotes' or text.strip().lower() == 'footnotes'
        elif in_foot:
            continue
        kept.append((level, hid, text))
    heads = kept
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


def _span_of(cell: str):
    """('text', N) for a header cell ending in {span=N}; ('text', 1) otherwise."""
    m = re.search(r'\s*\{span=(\d+)\}\s*$', cell)
    return (cell[:m.start()], int(m.group(1))) if m else (cell, 1)


def _slug(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:60] or 'section'


def with_defs(html: str, defs: dict) -> str:
    """Wraps whole-word occurrences of each defined term (outside tags) in a hover span.

    Longer terms are wrapped first and a wrapped occurrence is never matched again (it is
    replaced by a marker until the end), so terms may contain other terms ("oracle gain" and
    "oracle"): each occurrence gets the longest term that covers it, and no span is nested in
    another or inserted into a span's title."""
    if not defs:
        return html
    spans: List[str] = []

    def wrap(m, definition: str) -> str:
        spans.append(f'<span class="def" title="{html_mod.escape(definition, quote=True)}">{m.group(1)}</span>')
        return f'\x00{len(spans) - 1}\x00'

    parts = re.split(r'(<[^>]+>)', html)
    for k, part in enumerate(parts):
        if part.startswith('<'):
            continue
        for term in sorted(defs, key=len, reverse=True):
            part = re.sub(r'(?<![\w-])(' + re.escape(term) + r')(?![\w-])',
                lambda m, d=defs[term]: wrap(m, d), part)
        parts[k] = part
    return re.sub(r'\x00(\d+)\x00', lambda m: spans[int(m.group(1))], ''.join(parts))


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

    def cell_html(s: str) -> str:
        """A table cell: `<br>` splits it into lines, each converted on its own."""
        return '<br>'.join(text_html(part.strip()) for part in _BR_RE.split(s))

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
        m_html = re.match(r'^%%% html (\S+)$', s)
        if m_html:
            # a prepared HTML fragment (e.g. the progress plot), inserted verbatim
            p = (figdir or Path('.')) / m_html.group(1)
            if not p.exists():
                raise FileNotFoundError(f'html block not found: {p}')
            out.append(p.read_text())
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
            out.append(f'<figure><img class="zoomable" tabindex="0" src="{_data_uri(p)}" alt="{html_mod.escape(cap)}" title="click to view full screen">'
                + (f'<figcaption>{inline(cap)}</figcaption>' if cap else '') + '</figure>')
            i += 1; continue
        if s.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip()); i += 1
            cells = [[c.strip() for c in r.strip('|').split('|')] for r in rows]
            is_sep = [all(re.fullmatch(r':?-{2,}:?', c) for c in r) for r in cells]
            sep_at = is_sep.index(True) if True in is_sep else 0
            head_rows = cells[:sep_at] if sep_at > 0 else cells[:1]
            body_rows = [r for k, r in enumerate(cells) if k > sep_at or (sep_at == 0 and k > 0)]
            body_rows = [r for r in body_rows if not all(re.fullmatch(r':?-{2,}:?', c) for c in r)]
            ncol = max([len(r) for r in body_rows] + [sum(_span_of(c)[1] for c in r) for r in head_rows])
            numeric_like = re.compile(r'^([−+\-]?\d|[—–-]$|undefined$|n/?a$)')
            col_num = []
            for c in range(ncol):
                vals = [strip_pv(r[c]) for r in body_rows if c < len(r) and r[c] != '']
                col_num.append(bool(vals) and sum(bool(numeric_like.match(v)) for v in vals) >= 0.5 * len(vals))
            stats_table = ncol > 1 and all(col_num[1:]) and any(
                r and _span_of(r[0])[0].strip().lower() == 'statistic' for r in head_rows)
            t = ['<div class="statwrap"><table class="stats">' if stats_table else '<table>']
            for hk, hr in enumerate(head_rows):
                cells_html, c = [], 0
                for cell in hr:
                    text, span = _span_of(cell)
                    if span > 1:
                        cells_html.append(f'<th class="span" colspan="{span}">{cell_html(text)}</th>')
                    else:
                        cells_html.append(f'<th class="{"num" if c < ncol and col_num[c] else "txt"}">{cell_html(text)}</th>')
                    c += span
                t.append(f'<tr class="{"subhead" if hk else "head"}">' + ''.join(cells_html) + '</tr>')
            for r in body_rows:
                t.append('<tr>' + ''.join(
                    f'<td class="{"num" if c < ncol and col_num[c] else "txt"}">{cell_html(cell)}</td>' for c, cell in enumerate(r)) + '</tr>')
            t.append('</table></div>' if stats_table else '</table>')
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
        f'</main>\n' + (ZOOM_JS + '\n' if 'class="zoomable"' in body else '') + '</body>\n</html>\n')


def lint(page: str) -> List[str]:
    """Problems that mean markdown survived conversion (empty list = clean)."""
    text = re.sub(r'<img [^>]*>', '', page)
    text = re.sub(r'<style>.*?</style>', '', text, flags=re.S)
    text = re.sub(r'<script>.*?</script>', '', text, flags=re.S)
    text = re.sub(r"<figure class=\"prog\"[^>]*>", '', text)
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
        if re.search(r'&lt;br\s*/?&gt;', t, re.I):
            probs.append(f'escaped line break left in text (<br> works in table cells only): {t[:70]!r}')
        if re.search(r'\{span=\d+\}', t):
            probs.append(f'column-span marker left in text: {t[:70]!r}')
        if re.match(r'^(\d+\. |[-*] )', t) and len(t) > 3:
            probs.append(f'list marker left in text: {t[:70]!r}')
    if re.search(r'<li>[^<]*<br', page):
        probs.append('line break inside a list item')
    # every footnote (an h3 under the Footnotes h2) must be linked from the text
    m = re.search(r'<h2 id="footnotes"[^>]*>.*?</h2>(.*)$', page, flags=re.S)
    if m:
        for fid in re.findall(r'<h3 id="([^"]+)"', m.group(1)):
            refs = re.findall(r'href="#' + re.escape(fid) + '"', page.replace(m.group(1), ''))
            if not refs:
                probs.append(f'footnote #{fid} is not linked from the text')
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
