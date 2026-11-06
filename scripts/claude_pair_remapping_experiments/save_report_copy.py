#!/usr/bin/env python3
"""Saves a self-contained copy of a published artifact page to the shared reports/ folder.

Our artifact pages are single HTML files with figures embedded as data URIs, published with
`publish_files` from a local file; the server serves that file unchanged, so a copy of it is
the same document the artifact's share > export download gives. Run this after every publish
or republish (CLAUDE.md, "Report copies"):

    ./venv/bin/python save_report_copy.py <index.html or directory> <name> <artifact url>

Writes <shared>/reports/<name>.html (or <name>.zip with index.html and the other files when a
directory is given, i.e. a multi-file artifact) and upserts a row in reports/README.md
(name, title from the page's <title>, artifact URL, when and from which host it was copied).
Republishing overwrites the same file: reports/ holds the latest version of each page.

Summary pages (names starting with `summary_`) also update the symlink
reports/current-summary.html -> <name>.html, so one bookmark always opens the latest summary
as the date in the name changes; `--current <link name>` does the same for any page (e.g.
`--current current-report.html` for the report page). A copy is written instead if the
filesystem refuses the symlink.
"""
from __future__ import annotations

import argparse
import html as html_mod
import re
import shutil
import socket
import zipfile
from datetime import datetime
from pathlib import Path

import kc_remapping_harness as H

README_HEADER = """# Report copies

Self-contained copies of every artifact page published for this project (the same HTML the
artifact serves, i.e. what share > export downloads), written by `save_report_copy.py` in
the repo folder after each publish. One file per page, latest version only; multi-file
artifacts are zipped (index.html plus the other files).

`current-summary.html` is a symlink to the latest summary page (updated by the script).

| name | title | artifact | copied | host |
|---|---|---|---|---|
"""


def page_title(html: str) -> str:
    m = re.search(r'<title>(.*?)</title>', html, flags=re.S)
    return html_mod.unescape(re.sub(r'\s+', ' ', m.group(1)).strip()) if m else ''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('source', type=Path, help='the published index.html, or the directory of a multi-file artifact')
    ap.add_argument('name', help='file stem in reports/, e.g. E17_panel_general_apl, report_apl_remapping')
    ap.add_argument('url', help='the artifact URL')
    ap.add_argument('--current', default=None, metavar='LINK',
        help='also point reports/<LINK> at this copy (default: current-summary.html for '
        'names starting with summary_)')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.name):
        raise SystemExit(f'name must be a plain file stem: {args.name!r}')
    reports = H.shared_dir() / 'reports'
    reports.mkdir(exist_ok=True)
    src = args.source
    if src.is_dir():
        index = src / 'index.html'
        if not index.exists():
            raise SystemExit(f'{src} has no index.html')
        title = page_title(index.read_text(errors='replace'))
        dest = reports / f'{args.name}.zip'
        with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(src.rglob('*')):
                if p.is_file():
                    z.write(p, p.relative_to(src).as_posix())
        for stale in (reports / f'{args.name}.html',):
            if stale.exists():
                stale.unlink()
    else:
        title = page_title(src.read_text(errors='replace'))
        dest = reports / f'{args.name}.html'
        shutil.copyfile(src, dest)
        stale = reports / f'{args.name}.zip'
        if stale.exists():
            stale.unlink()
    link_name = args.current or ('current-summary.html' if args.name.startswith('summary_') else None)
    if link_name is not None:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', link_name) or link_name == dest.name:
            raise SystemExit(f'bad --current link name: {link_name!r}')
        link = reports / link_name
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(dest.name)   # relative, so the link works from any mount point
            print(f'{link} -> {dest.name}')
        except OSError as e:
            shutil.copyfile(dest, link)
            print(f'{link}: copy of {dest.name} (symlink refused: {e})')
    readme = reports / 'README.md'
    text = readme.read_text() if readme.exists() else README_HEADER
    if link_name is not None and 'current-summary.html' not in text:
        text = text.replace('| name | title | artifact | copied | host |',
            '`current-summary.html` is a symlink to the latest summary page (updated by the script).\n\n'
            '| name | title | artifact | copied | host |', 1)
    if '| name | title |' not in text:
        text = README_HEADER + text
    row = (f'| {args.name} | {title.replace("|", "/")} | {args.url} | '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")} | {socket.gethostname()} |')
    lines = text.rstrip('\n').split('\n')
    lines = [l for l in lines if not l.startswith(f'| {args.name} |')]
    lines.append(row)
    readme.write_text('\n'.join(lines) + '\n')
    print(f'{dest} ({dest.stat().st_size // 1024} KB); README row: {row}')


if __name__ == '__main__':
    main()
