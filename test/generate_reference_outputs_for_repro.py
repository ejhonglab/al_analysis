#!/usr/bin/env python3
"""Generates reference model outputs (via `fit_and_plot_mb_model`), for each element in
`test_mb_model.FITANDPLOT_MODEL_KW_LIST`, to capture the current behavior of the model
(for each of those sets of parameters).

These reference outputs (saved under `test_mb_model.reference_output_dir`) will be used
by `test_mb_model.test_fitandplot_repro` to detect whether model behavior has changed?
"""

import argparse
from pathlib import Path
from datetime import datetime
from pprint import pformat
import shutil
import traceback

from tqdm import tqdm
from termcolor import cprint

from hong2p.util import format_date
from al_util import warn
from mb_model import read_param_csv, format_model_params

# how to import this? (couldn't quickly figure it out when this script was originally in
# a test/scripts subdir, but don't see huge issues w/ having directly under test/ dir)
from test_mb_model import (FITANDPLOT_MODEL_KW_LIST, reference_output_root, _orn_deltas,
    _fit_and_plot_mb_model, assert_param_dicts_equal, assert_param_csv_matches_returned,
    get_expected_missing_csv_keys, get_latest_reference_output_dir
)


# TODO any easy way to generate (+ serialize) outputs for fit_mb_model alone, or too
# much duplicated work w/ serialization done already in fit_and_plot_mb_model?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--write-into-last', action='store_true', help='writes '
        'into most recent output directory, instead of generating a new one. will skip '
        'any existing model output directories'
    )
    parser.add_argument('-x', '--xfail-only', action='store_true', help='will *only* '
        'run model parameters pytest marked as expected to fail (xfail). without this,'
        ' xfail cases are *not* run'
    )
    parser.add_argument('-d', '--delete-on-err', action='store_true',
        help='delete any model_output_dir created, if model call that made them fails'
    )
    parser.add_argument('-b', '--debug', action='store_true', help='raises error on '
        'failure of any model call. if this script is called like:\n'
        '`python -m ipdb -c continue <script.py> <args>`\n'
        '...that will allow post-mortem debugging'
    )
    args = parser.parse_args()
    write_into_last = args.write_into_last
    xfail_only = args.xfail_only
    delete_on_err = args.delete_on_err
    debug = args.debug

    reference_output_root.mkdir(exist_ok=True)

    if not write_into_last:
        reference_output_dir = reference_output_root / format_date(datetime.today())
        while reference_output_dir.exists():
            warn(f'reference_output_dir {reference_output_dir} existed! '
                'adding/incrementing number on end to avoid overwriting!'
            )
            curr_name = reference_output_dir.name
            sep = '.'
            if sep in curr_name:
                date_part, num_part = curr_name.split(sep)
                # don't want to have to worry about overwriting, and probably want to
                # always test against the latest in the test
                n = int(num_part) + 1
            else:
                date_part = curr_name
                n = 0

            # Seems I can let the  _fit_and_plot... calls below make all these
            # directories (and subdirectories).
            reference_output_dir = reference_output_dir.parent / f'{date_part}{sep}{n}'
    else:
        reference_output_dir = get_latest_reference_output_dir()

    # TODO copy code involved (at least mb_model.py, and maybe olfsysm.cpp?)? git
    # commit(s) (+ diff?)

    print(f'saving outputs under:\n{reference_output_dir}\n')

    # NOTE: could not call the fixture `orn_deltas()` (that wraps `_orn_deltas()`)
    # directly here. would cause an error (not supposed to call them directly), so had
    # to break into two fns, the core and the fixture wrapper.
    orn_df = _orn_deltas()

    # TODO filter to xfailed only before tqdm? (minor)

    failed_model_kws = []
    skipped_xfail_model_kws = []
    for model_kws in tqdm(FITANDPLOT_MODEL_KW_LIST, unit='test-cases'):
        xfail = False
        # TODO factor this into test_mb_model? or some other util library?
        # hong2p.testing?
        if type(model_kws) is not dict:
            # should be a pytest ParameterSet then, which I currently use to mark
            # specific elements as e.g. expected to fail.
            # (with a type(...) like <class '_pytest.mark.structures.ParameterSet'>)
            #
            # should be able to recover the dict this way
            values = model_kws.values
            assert type(values) is tuple and len(values) == 1
            model_kws = values[0]
            assert type(model_kws) is dict
            #

            xfail = True
            # TODO actually check mark is xfail, and not something else?
            # (that is the only mark i'm actually using though...)
        # TODO actually check type is as expected here?

        print(f'{model_kws=}')

        # TODO some way to simplify this conditional?
        if not xfail_only and xfail:
            skipped_xfail_model_kws.append(model_kws)
            warn('skipping this test case because marked xfail (expected to fail)!')
            continue
        elif xfail_only and not xfail:
            warn('skipping non-xfail test case (because -x/--xfail-only)')
            continue

        expected_model_output_dir_name = format_model_params(model_kws)
        model_output_dir = reference_output_dir / expected_model_output_dir_name
        if write_into_last:
            if model_output_dir.exists():
                # TODO orange or something other than red here? back to yellow?
                cprint(f'skipping {expected_model_output_dir_name} b/c already existed!'
                    ' delete it manually (and re-run), if you wish.', 'red'
                )
                try:
                    # this will only be able to remove empty directories. will err, if
                    # non-empty.
                    model_output_dir.rmdir()
                except IOError as err:
                    assert 'directory not empty' in str(err).lower(), f'{err=}'

                continue
        else:
            # since we should have been writing into a fresh directory in that case, and
            # all model output dir names should be unique within a run of this script
            assert not model_output_dir.exists()

        try:
            param_dict = _fit_and_plot_mb_model(reference_output_dir, orn_deltas=orn_df,
                **model_kws
            )
        except Exception as err:
            cprint(f'{model_kws=} failed with:\n{traceback.format_exc()}', 'red')

            # TODO some way to also do this after raise below (if whole this called from
            # debugger)? atexit? not sure it's worth it...
            if delete_on_err:
                warn(f'deleting directory {model_output_dir.name} after this error, '
                    'because -d/--delete-on-err'
                )
                shutil.rmtree(model_output_dir)

            if debug:
                # can be used for post-mortem debugging, if whole script is run with
                # debugger (pdb/ipdb). see -h/--help CLI message for this arg.
                raise

            failed_model_kws.append(model_kws)
            continue

        # would mean we couldn't reliably check existance of directory before
        # _fit_and_plot... call above, which we need for write_into_last=True
        assert param_dict['output_dir'] == expected_model_output_dir_name, \
            f'{param_dict["output_dir"]=} != {expected_model_output_dir_name=}'

        expected_missing_csv_keys = get_expected_missing_csv_keys(model_kws)

        assert_param_csv_matches_returned(reference_output_dir, param_dict,
            expected_missing_csv_keys=expected_missing_csv_keys
        )

    # TODO if write_into_last=True, also warn about any directory names present but not
    # seen in loop above

    # NOTE: test-id-suffix will be component in brackets at end of pytest parametrized
    # tests
    if len(skipped_xfail_model_kws) > 0:
        skipped_xfail_test_ids = map(format_model_params, skipped_xfail_model_kws)
        warn('failed (parameters, test-id-suffix):\n'
            f'{pformat(list(zip(skipped_xfail_model_kws, skipped_xfail_test_ids)))}'
        )

    if len(failed_model_kws) > 0:
        failed_test_ids = map(format_model_params, failed_model_kws)
        warn('failed (parameters, test-id-suffix):\n'
            f'{pformat(list(zip(failed_model_kws, failed_test_ids)))}'
        )


if __name__ == '__main__':
    main()

