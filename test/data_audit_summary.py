#!/usr/bin/env python3
"""Summarizes a data-audit CSV (from `AL_ANALYSIS_DATA_AUDIT=<csv> pytest ...`).

Prints, for each folder under al_analysis/data and test/test_data, which side of the
planned al_analysis / pyolfsysm split opened files in it:
- by code: 'model' if mb_model or wpnkc_shuffles was on the stack, 'al' if one of the
  AL modules (al_analysis, plot_roi*, load_antennal_csv) was (and no model module),
  'al_util' if only al_util was, 'test' if only test code was;
- by test file: model tests vs AL tests (see `MODEL_TEST_FILES`).

Usage: ./venv/bin/python test/data_audit_summary.py <audit.csv> [<audit2.csv> ...]
"""

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd


MODEL_TEST_FILES = ('test_mb_model.py', 'test_characterize_model.py',
    'test_wpnkc_shuffles.py', 'test_model_apl_taum.py', 'test_model_provenance.py',
    'test_model_timing.py', 'test_claw_bouton_options.py', 'test_e43_presentation.py',
    'test_multi_condition_loss.py', 'test_olfsysm_save_buffers.py',
    'test_pn_ln_params.py'
)
AL_TEST_FILES = ('test_al_analysis.py', 'test_al_util.py', 'test_characterize_al.py',
    'test_characterize_output_helpers.py'
)


def test_file(test: str) -> str:
    test = test.replace('<collect> ', '')
    return Path(test.split('::')[0]).name


def test_group(test: str) -> str:
    name = test_file(test)
    if name in MODEL_TEST_FILES:
        return 'model tests'
    if name in AL_TEST_FILES:
        return 'AL tests'
    return 'other'


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['folder'] = df['root'].map({'data': 'data/', 'test_data': 'test_data/'}) + \
        df['folder'].astype(str)
    df['test_group'] = df['test'].map(test_group)
    df['opened'] = df['event'].str.startswith('open')

    rows = []
    for folder, fdf in df.groupby('folder'):
        opened = fdf[fdf.opened]
        rows.append(dict(
            folder=folder,
            sides_opening=', '.join(sorted(opened.side.unique())),
            test_groups_opening=', '.join(sorted(opened.test_group.unique())),
            n_files_opened=opened.path.nunique(),
            listed_only_by=', '.join(sorted(set(fdf[~fdf.opened].side) -
                set(opened.side))
            ),
            example_callers='; '.join(sorted(opened.caller.dropna().unique())[:3]),
        ))
    return pd.DataFrame(rows).set_index('folder')


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('csvs', nargs='+', type=Path)
    parser.add_argument('-f', '--files', action='store_true',
        help='also list every file opened, with the sides opening it'
    )
    args = parser.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in args.csvs], ignore_index=True)
    summary = summarize(df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
            'display.width', 250, 'display.max_colwidth', 80):
        print(summary.drop(columns='example_callers').to_string())

        both = summary[summary.sides_opening.str.contains('model') &
            summary.sides_opening.str.contains(r'\bal\b')
        ]
        print()
        print('folders opened by both model and AL code:',
            list(both.index) if len(both) else 'none'
        )
        if args.files:
            files = df[df.event.str.startswith('open')].groupby(['root', 'path']
                ).side.agg(lambda x: ', '.join(sorted(set(x))))
            print()
            print(files.to_string())


if __name__ == '__main__':
    main(sys.argv[1:])
