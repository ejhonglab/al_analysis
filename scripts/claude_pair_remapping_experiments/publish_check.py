#!/usr/bin/env python3
"""Before republishing a shared page (the summary, the report): does the live page hold anything the new
build lacks?

Several hosts publish the same artifact, each from a page built from its own checkout, and the Artifact
tool refuses a publish that was not built on the live version. This compares the live page (saved with the
Artifact tool's read, `path: index.html`) with the new build at the level of HTML tags and sentences,
ignoring what every rebuild changes (the build and completion timestamps in the page's meta line, the
`[commit ...]` hashes in the value tooltips, the server's wrapper around the page), and prints the segments
the live page has that the new build does not: if there are none, the new build is a superset and can be
published once; if there are some, the template the build came from is behind someone's publish (pull, or
ask in chat), and the publish must wait. The reverse (what only the new build has) is printed with --new.

Usage: ./venv/bin/python publish_check.py <live index.html> <new index.html> [--new] [--width 300]
Exit status 0 when the live page has nothing the new build lacks, 1 otherwise.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
from pathlib import Path

# segments longer than this (inline plot data, embedded images) are compared by difflib's quick_ratio, an
# upper bound: the exact character-level ratio is quadratic in their length and took over ten minutes on
# the summary page (2026-11-10)
LONG = 20000
DATA_URI = re.compile(r'data:([a-z]+/[a-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)')

WRAPPER_START = re.compile(r'^\s*<!doctype html>\s*<html>\s*<head>.*?<body>\s*', re.S | re.I)
WRAPPER_END = re.compile(r'\s*</body>\s*</html>\s*$', re.S | re.I)


def normalize(html: str) -> str:
    """The page without what every rebuild changes."""
    # the server wraps the uploaded page in its own document; the uploaded page then starts with its own doctype
    m = WRAPPER_START.match(html)
    if m and html.count('<!DOCTYPE html>') + html.count('<!doctype html>') >= 2:
        html = html[m.end():]
        html = WRAPPER_END.sub('', html)
    # an embedded figure as a short hash of its bytes (a changed figure still shows as a changed segment)
    html = DATA_URI.sub(lambda m: f'data:{m.group(1)};sha1={hashlib.sha1(m.group(2).encode()).hexdigest()[:12]}', html)
    html = re.sub(r'\[commit [0-9a-f]{6,40}\]', '[commit]', html)
    html = re.sub(r'(title="time this page was built">)[^<]*', r'\1T', html)
    html = re.sub(r'(title="completion time of the latest completed experiment \()E\d+(\) in [^"]*">)[^<]*', r'\1E\2T', html)
    return html


def segments(html: str) -> list:
    return [s for s in re.split(r'(?<=[>.])\s*', html) if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('live', type=Path)
    ap.add_argument('new', type=Path)
    ap.add_argument('--new', dest='show_new', action='store_true', help='also print what only the new build has')
    ap.add_argument('--width', type=int, default=300)
    args = ap.parse_args()
    a = segments(normalize(args.live.read_text(encoding='utf-8')))
    b = segments(normalize(args.new.read_text(encoding='utf-8')))
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    removed, changed, added = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'delete':
            removed.append(' '.join(a[i1:i2]))
        elif op == 'insert':
            added.append(' '.join(b[j1:j2]))
        elif op == 'replace':
            old, new = ' '.join(a[i1:i2]), ' '.join(b[j1:j2])
            # a segment the new build rewrote (a status word, a sentence) against one it dropped and replaced
            sm2 = difflib.SequenceMatcher(None, old, new, autojunk=False)
            if (sm2.quick_ratio() if len(old) + len(new) > LONG else sm2.ratio()) > 0.5:
                changed.append((old, new))
            else:
                removed.append(old)
                added.append(new)
    w = args.width
    if removed:
        print(f'REMOVED: the live page has {len(removed)} segment(s) the new build drops (publish only if you meant to drop them):')
        for s in removed:
            print('  LIVE:', s[:w].replace('\n', ' '))
    if changed:
        print(f'CHANGED: {len(changed)} segment(s) the new build rewrites (publish if these are your edits):')
        for old, new in changed:
            print('  LIVE:', old[:w].replace('\n', ' '))
            print('  NEW: ', new[:w].replace('\n', ' '))
    if not removed and not changed:
        print('the live page has nothing the new build lacks or rewrites: publish')
    if args.show_new:
        print(f'only in the new build: {len(added)} segment(s)')
        for s in added:
            print('  NEW:', s[:w].replace('\n', ' '))
    sys.exit(1 if removed else (2 if changed else 0))


if __name__ == '__main__':
    main()
