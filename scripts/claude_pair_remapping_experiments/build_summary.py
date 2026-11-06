#!/usr/bin/env python3
"""Build the all-experiments summary report from its template (reports/summary_template.md).

1. regenerates the progress plot (summary_progress_plot.py);
2. fills the template from the data-target registry and the experiments' results tables
   (fill_report.py; fails on any unresolved placeholder, prints lint for bare numbers in KC-data
   rows and the best-models table);
3. writes the plain filled markdown (no tooltip markup) to the shared directory's
   add_to_reports/summary_all_experiments_2026-11-05.md (generated: edit the template, never
   that file), and the sidecar of every value's source next to it
   (summary_all_experiments_2026-11-05.values.json);
4. builds the page with source tooltips (md_to_artifact.py) at
   outputs/_artifacts/summary/index.html.
Publishing and save_report_copy.py stay manual (CLAUDE.md, "Summary reports").
Usage: ./venv/bin/python build_summary.py [--skip-progress-plot] [--data-targets]
       (--data-targets first rebuilds the registry, data_targets.py)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

import kc_remapping_harness as H
import fill_report as F

TEMPLATE = H.HERE / 'reports' / 'summary_template.md'
OUT_MD = H.shared_dir() / 'add_to_reports' / 'summary_all_experiments_2026-11-05.md'
OUT_HTML = H.HERE / 'outputs' / '_artifacts' / 'summary' / 'index.html'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--skip-progress-plot', action='store_true')
    ap.add_argument('--data-targets', action='store_true')
    args = ap.parse_args()
    py = sys.executable
    if args.data_targets:
        subprocess.run([py, str(H.HERE / 'data_targets.py')], check=True)
    if not args.skip_progress_plot:
        subprocess.run([py, str(H.HERE / 'summary_progress_plot.py')], check=True)
    template = TEMPLATE.read_text()
    probs = F.lint(template)
    for p in probs:
        print('LINT', p)
    plain, side = F.fill(template, tooltips=False)
    tipped, _ = F.fill(template, tooltips=True)
    OUT_MD.write_text(plain)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    OUT_MD.with_suffix('.values.json').write_text(json.dumps(dict(template=str(TEMPLATE.relative_to(H.REPO)), template_commit=commit,
        generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), n_values=len(side), lint=probs, values=side),
        indent=1, ensure_ascii=False, default=str) + '\n')
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    tipped_md = OUT_HTML.parent / 'summary_filled.md'
    tipped_md.write_text(tipped)
    r = subprocess.run([py, str(H.HERE / 'md_to_artifact.py'), str(tipped_md), str(OUT_HTML), '--figdir', str(H.shared_dir())],
        capture_output=True, text=True)
    print('\n'.join(l for l in (r.stdout + r.stderr).split('\n') if l and 'Warning' not in l and 'warnings.warn' not in l)[-600:])
    print(f'{len(side)} values filled; {len(probs)} lint problems; markdown -> {OUT_MD}; page -> {OUT_HTML}')
    if r.returncode != 0 or 'LINT PROBLEMS' in r.stdout:
        sys.exit(1)


if __name__ == '__main__':
    main()
