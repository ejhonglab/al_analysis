#!/usr/bin/env python3
"""Fill a report template's `{{key ~ spec}}` placeholders (report_values.py), write the filled
markdown and a sidecar of every value's source, and optionally build the HTML page.

- fill: template -> filled markdown (+ `<out>.values.json` sidecar). With tooltips (default), each
  value is written as `⟦pv:value§source⟧`, which md_to_artifact.py renders as a span whose hover
  text names the source. `~ spec` is a Python format spec (`.2f`, `+.3f`, `d`); negatives get a
  Unicode minus. (`|` cannot be used inside placeholders: it separates table cells.)
- --check SIDECAR: re-resolves every key of a sidecar and lists values whose sources changed.
- --compare CURRENT_MD: compares the filled markdown (tooltips stripped) with another markdown
  line by line and lists every differing table cell / sentence with its row, column and key.
- lint: in "KC data" rows and the best-models table, a number outside a placeholder is an error
  (`{{lit:<text> # <source>}}` is the escape).
Usage: ./venv/bin/python fill_report.py reports/summary_template.md OUT.md [--html OUT.html]
       [--no-tooltips] [--compare CURRENT.md] [--report DIFF.md]
       ./venv/bin/python fill_report.py --check OUT.values.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import kc_remapping_harness as H
import report_values as R

PH = re.compile(r'\{\{(.+?)\}\}')
PV = re.compile(r'⟦pv:(.*?)§(.*?)⟧')
SPEC = re.compile(r'^[+\- ]?[#0]?\d*[,_]?(\.\d+)?[a-z%]?$')


def split_placeholder(body: str) -> Tuple[str, str]:
    key, sep, spec = body.rpartition('~')
    if sep and SPEC.match(spec.strip()) and spec.strip():
        return key.strip(), spec.strip()
    return body.strip(), ''


def fill(template: str, *, tooltips: bool = True):
    """Returns (filled markdown, sidecar list of {line, key, spec, text, value, source})."""
    side, errors = [], []
    out_lines = []
    for n, line in enumerate(template.split('\n'), 1):
        def sub(m):
            key, spec = split_placeholder(m.group(1))
            try:
                res = R.resolve(key)
            except Exception as e:
                errors.append(f'line {n}: {key}: {e}')
                return '??'
            text = R.format_value(res.value, spec)
            side.append(dict(line=n, key=key, spec=spec, text=text,
                value=res.value if not hasattr(res.value, 'item') else res.value.item(), source=res.source))
            if not tooltips:
                return text
            tip = R.tooltip(res).replace('§', ' ').replace('⟧', ')').replace('|', '/').replace('"', "'")
            return f'⟦pv:{text}§{tip}⟧'
        out_lines.append(PH.sub(sub, line))
    if errors:
        raise KeyError('unresolved placeholders:\n' + '\n'.join(errors))
    return '\n'.join(out_lines), side


def _tables(md: str):
    """Yields (first line index, header labels per column, rows) for every markdown table."""
    lines = md.split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('|'):
            start = i
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
            sep = next((k for k, r in enumerate(cells) if all(re.fullmatch(r':?-{2,}:?', c) for c in r)), 0)
            heads = cells[:sep] if sep else cells[:1]
            ncol = max(len(r) for r in cells)
            labels = []
            for c in range(ncol):
                parts = [h[c] for h in heads if c < len(h) and h[c] and not h[c].endswith('}')]
                labels.append(' / '.join(p.replace('<br>', ' ') for p in parts) or f'column {c + 1}')
            yield start, sep, labels, cells
        else:
            i += 1


def lint(template: str) -> List[str]:
    """Bare numbers in KC-data rows and the best-models table (list of messages)."""
    probs = []
    lines = template.split('\n')
    best = None
    for k, l in enumerate(lines):
        if l.startswith('## Where the best models stand'):
            best = k
            break
    best_end = next((k for k in range((best or 0) + 1, len(lines)) if lines[k].startswith('## ')), len(lines)) if best is not None else -1
    for start, sep, labels, cells in _tables(template):
        in_best = best is not None and best < start < best_end
        for r, row in enumerate(cells):
            if r <= sep:
                continue
            if not (in_best or row[0].replace('*', '').strip() == 'KC data'):
                continue
            for c, cell in enumerate(row[1:], 1):
                bare = PH.sub('', cell)
                bare = re.sub(r'<br>', ' ', bare)
                bare = re.sub(r'!button\[[^\]]*\]\([^)]*\)', '', bare)   # matrix buttons (md_to_artifact.py)
                if re.search(r'\d', bare):
                    probs.append(f'line {start + r + 1}: {row[0][:30]!r} / {labels[c][:40]!r}: bare number in {cell[:60]!r}')
    return probs


def compare(filled: str, current: str, template: str) -> List[dict]:
    """Differences between the filled markdown (tooltips stripped) and the current markdown,
    cell by cell for table rows, word by word for other lines, with the template's keys."""
    f_lines = PV.sub(r'\1', filled).split('\n')
    c_lines = current.split('\n')
    t_lines = template.split('\n')
    labels_at = {}
    for start, sep, labels, cells in _tables(current):
        for r in range(len(cells)):
            labels_at[start + r] = labels
    diffs = []
    sm = difflib.SequenceMatcher(a=c_lines, b=f_lines, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == 'equal':
            continue
        if op != 'replace' or (a1 - a0) != (b1 - b0):
            diffs.append(dict(kind='structure', current_lines=f'{a0 + 1}-{a1}', filled_lines=f'{b0 + 1}-{b1}',
                current='\n'.join(c_lines[a0:a1])[:300], filled='\n'.join(f_lines[b0:b1])[:300]))
            continue
        for a, b in zip(range(a0, a1), range(b0, b1)):
            cl, fl, tl = c_lines[a], f_lines[b], t_lines[b] if b < len(t_lines) else ''
            if cl.strip().startswith('|') and fl.strip().startswith('|'):
                cc = [x.strip() for x in cl.strip().strip('|').split('|')]
                ff = [x.strip() for x in fl.strip().strip('|').split('|')]
                tt = [x.strip() for x in tl.strip().strip('|').split('|')]
                labels = labels_at.get(a, [])
                for k in range(max(len(cc), len(ff))):
                    x = cc[k] if k < len(cc) else ''
                    y = ff[k] if k < len(ff) else ''
                    if x != y:
                        diffs.append(dict(kind='cell', line=a + 1, row=cc[0][:60] if cc else '', column=(labels[k] if k < len(labels) else f'column {k + 1}')[:90],
                            current=x, filled=y, keys=[split_placeholder(m)[0] for m in PH.findall(tt[k] if k < len(tt) else '')]))
            else:
                cw, fw = re.findall(r'\S+', cl), re.findall(r'\S+', fl)
                ws = difflib.SequenceMatcher(a=cw, b=fw, autojunk=False)
                for op2, x0, x1, y0, y1 in ws.get_opcodes():
                    if op2 != 'equal':
                        ctx = ' '.join(cw[max(0, x0 - 6):x0])
                        diffs.append(dict(kind='text', line=a + 1, context=ctx[-80:], current=' '.join(cw[x0:x1]), filled=' '.join(fw[y0:y1]),
                            keys=[split_placeholder(m)[0] for m in PH.findall(tl)]))
    return diffs


def check(sidecar_path: Path) -> List[str]:
    side = json.loads(Path(sidecar_path).read_text())
    changed = []
    for e in side['values']:
        try:
            res = R.resolve(e['key'])
        except Exception as ex:
            changed.append(f"line {e['line']}: {e['key']}: no longer resolves ({ex})")
            continue
        text = R.format_value(res.value, e['spec'])
        if text != e['text']:
            changed.append(f"line {e['line']}: {e['key']}: {e['text']} -> {text}")
        elif res.source.get('sha256') != e['source'].get('sha256'):
            changed.append(f"line {e['line']}: {e['key']}: source file changed, value unchanged ({text})")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('template', nargs='?', type=Path)
    ap.add_argument('out', nargs='?', type=Path)
    ap.add_argument('--html', type=Path, default=None)
    ap.add_argument('--no-tooltips', action='store_true')
    ap.add_argument('--compare', type=Path, default=None, help='current markdown to compare the filled one with')
    ap.add_argument('--report', type=Path, default=None, help='write the comparison as markdown here')
    ap.add_argument('--check', type=Path, default=None, help='sidecar to re-check')
    args = ap.parse_args()
    if args.check:
        ch = check(args.check)
        print('\n'.join(ch) if ch else 'all values unchanged')
        sys.exit(1 if ch else 0)
    template = args.template.read_text()
    probs = lint(template)
    for p in probs:
        print('LINT', p)
    filled, side = fill(template, tooltips=not args.no_tooltips)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(filled)
    sidecar = args.out.with_suffix('.values.json')
    sidecar.write_text(json.dumps(dict(template=str(args.template), generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        n_values=len(side), lint=probs, values=side), indent=1, ensure_ascii=False, default=str) + '\n')
    print(f'filled {len(side)} placeholders -> {args.out} (sidecar {sidecar.name}); lint: {len(probs)} bare numbers')
    if args.html:
        r = subprocess.run([sys.executable, str(H.HERE / 'md_to_artifact.py'), str(args.out), str(args.html), '--figdir', str(H.shared_dir())],
            capture_output=True, text=True)
        print(r.stdout.strip().split('\n')[-1] if r.stdout.strip() else r.stderr[-500:])
    if args.compare:
        diffs = compare(filled, args.compare.read_text(), template)
        print(f'{len(diffs)} differences from {args.compare.name}')
        if args.report:
            rows = ['| line | where | current | templated | key |', '|---|---|---|---|---|']
            for d in diffs:
                where = f"{d.get('row', '')} / {d.get('column', '')}" if d['kind'] == 'cell' else (d.get('context', '') if d['kind'] == 'text' else 'structure')
                rows.append(f"| {d.get('line', d.get('current_lines'))} | {where.replace('|', '/')} | {str(d.get('current', '')).replace('|', '/')[:120]} | "
                    f"{str(d.get('filled', '')).replace('|', '/')[:120]} | {', '.join(d.get('keys', []))[:150]} |")
            args.report.write_text('\n'.join(rows) + '\n')
            print('comparison ->', args.report)


if __name__ == '__main__':
    main()
