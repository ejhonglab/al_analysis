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

from hong2p.util import format_date, symlink
from al_util import warn
from mb_model import read_param_csv, format_model_params

# how to import this? (couldn't quickly figure it out when this script was originally in
# a test/scripts subdir, but don't see huge issues w/ having directly under test/ dir)
from test_mb_model import (FITANDPLOT_MODEL_KW_LIST, reference_output_root,
    megamat_orn_deltas, _fit_and_plot_mb_model, assert_param_dicts_equal,
    assert_param_csv_matches_returned, assert_fit_and_plot_outputs_equal,
    get_expected_missing_csv_keys, get_latest_reference_output_dir
)


# TODO any easy way to generate (+ serialize) outputs for fit_mb_model alone, or too
# much duplicated work w/ serialization done already in fit_and_plot_mb_model?

def main():
    # TODO TODO add argument for substring to match against test ID (use -k, like
    # pytest?)
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--write-into-last', action='store_true', help='writes '
        'into most recent output directory, instead of generating a new one. will skip '
        'any existing model output directories'
    )
    # TODO err/warn if no test cases marked xfail?
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
        '...that will allow post-mortem debugging. without this, will continue to try '
        'all remaining parameters parameter, reporting errors in passing and a '
        'summary at  the end.'
    )
    parser.add_argument('-q', '--quick', action='store_true',
        help='uses shorter QUICK_FITANDPLOT_MODEL_KW_LIST, instead of '
        'FITANDPLOT_MODEL_KW_LIST'
    )
    parser.add_argument('-k', '--substring', action='store', default=None,
        help='only param combos with test IDs containing this substring will be run'
    )
    parser.add_argument('-y', '--save-dynamics', action='store_true',
        help='will also save dynamics (currently into pickles) in each model output dir'
    )
    parser.add_argument('-c', '--check-against-last', action='store_true',
        help='if True, will check against last output directory, and make symlinks from'
        ' new output directory to old one, for all directories where old output was '
        'preserved. any non-matching directories will be written anew in new output '
        'reference output directory.'
    )
    args = parser.parse_args()
    write_into_last = args.write_into_last
    xfail_only = args.xfail_only
    delete_on_err = args.delete_on_err
    debug = args.debug
    quick = args.quick
    substring = args.substring
    save_dynamics = args.save_dynamics
    check_against_last = args.check_against_last

    if substring is not None:
        assert type(substring) is str and len(substring) > 0

    if check_against_last:
        assert not write_into_last, 'only specify one of these'

    # TODO TODO add arg to check existing outputs match, and write to new dir if not
    # (then also write / print which dirs had matching input. could (for now at least)
    # manually `git mv` matching old outputs to new directory, maybe adding a text file
    # with other dates they existed in? i assume git doesn't handle symlinks super well
    # across platforms?)
    # TODO TODO actually, try symlinking to old dirs (+ committing the links too). seems
    # like it should work well enough, and still give me something to work with even on
    # systems that don't support symlinks

    reference_output_root.mkdir(exist_ok=True)

    last_reference_output_dir = None
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

            if check_against_last:
                last_reference_output_dir = get_latest_reference_output_dir()
    else:
        reference_output_dir = get_latest_reference_output_dir()
        last_reference_output_dir = reference_output_dir

    # TODO copy code involved (at least mb_model.py, and maybe olfsysm.cpp?)? git
    # commit(s) (+ diff?) (eh, nah)

    if not check_against_last:
        print(f'saving outputs under:\n{reference_output_dir}\n')
    else:
        print(f'checking against old outputs under:\n{last_reference_output_dir}')
        print('saving non-matching outputs (and making symlinks to matching old outputs'
            f') under:\n{reference_output_dir}\n'
        )

    # NOTE: could not call the fixture `orn_deltas()` (that wraps
    # `megamat_orn_deltas()`) directly here. would cause an error (not supposed to call
    # them directly), so had to break into two fns, the core and the fixture wrapper.
    orn_df = megamat_orn_deltas()

    if not quick:
        # TODO TODO delete
        model_kw_list = FITANDPLOT_MODEL_KW_LIST[::-1]
        # TODO TODO TODO restore
        #model_kw_list = FITANDPLOT_MODEL_KW_LIST
    else:
        model_kw_list = QUICK_FITANDPLOT_MODEL_KW_LIST

    skipped_xfail_model_kws = []
    skipped_because_existing = []
    not_present_in_last = []
    nonskipped_kws_and_dirs = []
    for model_kws in model_kw_list:
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
        if substring is not None and substring not in expected_model_output_dir_name:
            warn(f'skipping because {substring=} was not in (what would be) directory '
                f'name ({expected_model_output_dir_name})'
            )
            continue

        model_output_dir = reference_output_dir / expected_model_output_dir_name
        if write_into_last:
            if model_output_dir.exists():
                try:
                    # this will only be able to remove empty directories. will err, if
                    # non-empty.
                    model_output_dir.rmdir()

                except IOError as err:
                    assert 'directory not empty' in str(err).lower(), f'{err=}'
                    warn(f'skipping {expected_model_output_dir_name} b/c already '
                        'existed! delete it manually (and re-run), if you wish.'
                    )
                    skipped_because_existing.append(model_kws)
                    continue
        else:
            # since we should have been writing into a fresh directory in that case, and
            # all model output dir names should be unique within a run of this script
            assert not model_output_dir.exists()

        nonskipped_kws_and_dirs.append((model_kws, model_output_dir))


    if substring is not None and nonskipped_kws_and_dirs:
        raise RuntimeError(f'{substring=} did not match any model parameters!')

    curr_dirnames = None
    if write_into_last:
        curr_dirnames = set(skipped_because_existing)

    elif check_against_last:
        curr_dirnames = {d.name for _, d in nonskipped_kws_and_dirs}

    failed_model_kws = []
    failed_test_ids = []
    failed_tracebacks = []
    for model_kws, model_output_dir in tqdm(nonskipped_kws_and_dirs, unit='test-cases'):
        print(f'{model_kws=}')
        expected_model_output_dir_name = model_output_dir.name

        last_output_dir = None
        if last_reference_output_dir is not None:
            last_output_dir = last_reference_output_dir / expected_model_output_dir_name
            if (not last_output_dir.exists() or
                    # this is last CSV saved in fit_and_plot... so if it doesn't exist,
                    # things terminated before it finished
                    not (last_output_dir / 'spike_counts.csv').exists()
                ):

                # still going to generate anew (if check_against_last=True), just won't
                # have anything to compare to
                not_present_in_last.append(model_kws)
                # so we don't try to call assert_fit_and_plot_outputs_equal with this
                # output below, which currently does not reliably fail for directories
                # missing files
                last_output_dir = None

        curr_tb = None
        try:
            # TODO atexit / catch KeyboardInterrupt to delete last model_output_dir
            # created, if process interrupted? (or originally write to diff name, then
            # move to actual target directory once succesfully completed?)
            #
            # TODO only save dynamics for those in QUICK* subset? check current total
            # size without restricting to subset
            # TODO (eh, prob not worth anymore?) subset dynamics (after the fact
            # load+resave? or ig i'd otherwise have to thread some hack thru?) (like 1-3
            # cells/ 1-2 odors or something?)
            # TODO TODO make sure repro test ignores all the dynamics, one way or
            # another. don't want to *have* to save+check those (at least, w/o some
            # separate test or flag)
            params = _fit_and_plot_mb_model(reference_output_dir, orn_deltas=orn_df,
                return_dynamics=save_dynamics, **model_kws
            )
        except Exception as err:
            curr_tb = traceback.format_exc()
            cprint(f'{model_kws=} failed with:\n{curr_tb}', 'red')

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
            failed_test_ids.append(expected_model_output_dir_name)
            assert curr_tb is not None
            failed_tracebacks.append(curr_tb)
            continue

        if check_against_last and last_output_dir is not None:
            assert model_output_dir.is_dir()
            try:
                # TODO want to pass any kwargs here?
                # TODO TODO TODO fix how this seems isn't failing when input dir is
                # basically empty (just some irrelevant plots)
                # TODO or at least only proceed to this check if we have some file
                # created at end of successful model run
                assert_fit_and_plot_outputs_equal(reference_output_root, params,
                    plot_root2=last_reference_output_dir
                )
            except AssertionError as err:
                print(f'deleting new output:\n{model_output_dir}\n...because it matched'
                    f' old output:\n{last_output_dir}'
                )
                shutil.rmtree(model_output_dir)
                # TODO test case where target is already a symlink? that fine, or
                # need to reference original file?
                print('creating symlink to matching old output')
                symlink(last_output_dir, model_output_dir, relative=True)

        # would mean we couldn't reliably check existance of directory before
        # _fit_and_plot... call above, which we need for write_into_last=True
        assert params['output_dir'] == expected_model_output_dir_name, \
            f'{params["output_dir"]=} != {expected_model_output_dir_name=}'

        # TODO delete?
        expected_missing_csv_keys = get_expected_missing_csv_keys(model_kws)

        assert_param_csv_matches_returned(reference_output_dir, params,
            expected_missing_csv_keys=expected_missing_csv_keys
        )
        # TODO warn/err if any outputs have hardcoded learning rates?
        # (would be more annoying to repro in future)
        # should prob update defaults soon though...


    def format_kw_list(kw_seq) -> str:
        id_seq = map(format_model_params, kw_seq)
        return '\n'.join(id_seq)
        # TODO delete?
        # TODO print each element individually, rather than showing as tuples
        #return pformat(list(zip(kw_seq, id_seq)))

    # NOTE: test-id-suffix will be component in brackets at end of pytest parametrized
    # tests
    if len(skipped_xfail_model_kws) > 0:
        warn('xfailed (parameters, test-id-suffix):\n'
            f'{format_kw_list(skipped_xfail_model_kws)}'
        )

    if curr_dirnames is not None:
        assert last_reference_output_dir is not None
        last_dirnames = {
            x.name for x in last_reference_output_dir.glob('*/') if x.is_dir()
        }
        only_present_in_last = last_dirnames - curr_dirnames
        if len(only_present_in_last) > 0:
            cprint(f'only present in last:\n{pformat(only_present_in_last)}',
                'red'
            )

    if len(not_present_in_last) > 0:
        cprint(f'not present in last (last_reference_output_dir):\n'
            f'{format_kw_list(not_present_in_last)}', 'red'
        )

    assert len(failed_model_kws) == len(failed_test_ids) == len(failed_tracebacks)
    if len(failed_model_kws) > 0:
        print()
        cprint('failed model params and tracebacks:', 'red')
        for kws, tid, tb in zip(failed_model_kws, failed_test_ids, failed_tracebacks):
            cprint(f'{tid}\nfailed with:\n{curr_tb}', 'red')

        cprint('failed test IDs (again, without tracebacks):\n'
            f'{format_kw_list(failed_test_ids)}', 'red'
        )
        print('')


if __name__ == '__main__':
    main()

