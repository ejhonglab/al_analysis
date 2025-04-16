
from copy import deepcopy
import difflib
from functools import wraps
import filecmp
import itertools
import os
from os.path import getmtime
from pathlib import Path
import pickle
from pprint import pformat
import sys
from tempfile import NamedTemporaryFile
import time
import traceback
from typing import Type, Union, Optional, Callable, List, Set
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# TODO need to install something else for this, in a new env?
# (might have manually installed before in current one...)
from matplotlib.testing import compare as mpl_compare
from matplotlib.testing.exceptions import ImageComparisonFailure
#
import seaborn as sns
from termcolor import cprint, colored

from hong2p import olf, util, viz
from hong2p.types import Pathlike
import natmix


bootstrap_seed = 1337

verbose = False

# TODO replace this + use of warnings.warn w/ logging.warning (w/ logger probably
# currently just configured to output to stdout/stderr)
def formatwarning_msg_only(msg, category, *args, **kwargs):
    """Format warning without line/lineno (which are often not the relevant line)
    """
    warn_type = category.__name__ if category.__name__ != 'UserWarning' else 'Warning'
    return colored(f'{warn_type}: {msg}\n', 'yellow')

# TODO do just in main?
# TODO TODO maybe also toggle this w/ -v CLI flag (still have it colored orange tho...)
# (or a dedicated flag for this?)
warnings.formatwarning = formatwarning_msg_only


# TODO maybe log all warnings?
# TODO TODO replace w/ logging.warning (have init_logger just hook into warnings.warn?
# some standard mechanism for that?)
def warn(msg):
    warnings.warn(str(msg))


# TODO move to hong2p.util?
def format_mtime(mtime_or_path: Union[float, Pathlike], *, year: bool = False,
    seconds: bool = False) -> str:
    """Formats mtime like default `ls -l` output (e.g. 'Oct 11 18:24').
    """
    if isinstance(mtime_or_path, float):
        mtime = mtime_or_path
    else:
        mtime = getmtime(mtime_or_path)

    fstr = '%b %d %H:%M'
    if seconds:
        fstr += ':%S'
    if year:
        fstr += ' %Y'

    return time.strftime(fstr, time.localtime(mtime))


# True|False|'nonmain'
check_outputs_unchanged = False
# hack so al_analysis can edit this to add functions i currently have defined under main
# (e.g. save_method_csvs)
_consider_as_main = []

# TODO TODO move to hong2p.util
# TODO unit test?
#
# TODO default verbose=None and try to use default of wrapped fn then
# (or True otherwise?)
# (still need to test behavior when wrapped fn has existing verbose kwarg)
# TODO make this an attribute of this/one of inner fns (rather than module level)?
_fn2seen_inputs = dict()
# TODO what is _fn for again? keep?
def produces_output(_fn=None, *, verbose=True):
    # for how to make a decorator with optional  arg:
    # https://realpython.com/primer-on-python-decorators

    # TODO what would be a good name for this?
    def wrapper_helper(fn):

        assert fn.__name__ not in _fn2seen_inputs, (
            'seen set would have been overwritten'
        )
        # TODO some reason to use lists like i was in savefig? was that just for easier
        # use in multiprocessing access (no set equiv of IPC data type?)?
        # that matter anymore?
        _fn2seen_inputs[fn.__name__] = set()

        @wraps(fn)
        # TODO delete *args (if assertion it's unused passes for a wihle)
        def wrapped_fn(data, path: Pathlike, *args, verbose: bool = verbose,
            multiple_saves_per_run_ok: bool = False, **kwargs):
            """
            Args:
                multiple_saves_per_run_ok: if False, will raise
                    MultipleSavesPerRunException if same absolute path is written to
                    twice in one run (which is assumed to usually be a programming
                    mistake)
            """
            # TODO delete (probably delete *args in sig above if so)
            assert len(args) == 0
            #

            # TODO easy to check type of matching positional arg is Path/Pathlike
            # (if specified)?
            # see: https://stackoverflow.com/questions/71082545 for one way

            path = Path(path)

            # so that no matter how input is specified, check on whether we already
            # wrote to it (via `seen_inputs`) not have any false negatives.
            #
            # from `Path.resolve` docs:
            # "Make the path absolute, resolving any symlinks."
            # "'..' components are also eliminated"
            normalized_path = path.resolve()

            # TODO add option (for use during debugging) that checks outputs
            # have not changed since last run (to the extent the format allows it...)
            # (currently have this via global `check_outputs_unchanged`. do i actually
            # want it to specifically be a kwarg added by the wrapper instead?)

            assert fn.__name__ in _fn2seen_inputs
            # TODO probably don't want different fns to be able to save to same path
            # either tho... (not that they currently would). maybe seen_inputs should be
            # one global?
            seen_inputs = _fn2seen_inputs[fn.__name__]

            if not multiple_saves_per_run_ok:
                if normalized_path in seen_inputs:
                    raise MultipleSavesPerRunException('would have overwritten output '
                        f'{path} (previously written elsewhere in this run)!'
                    )

            seen_inputs.add(normalized_path)

            if check_outputs_unchanged and path.exists():
                try:
                    # TODO wrapper kwarg (that would be added to some/all decorations)
                    # for disabling this for some calls? e.g. for formats where they can
                    # change for uninteresting reasons, like creation time
                    _check_output_would_not_change(path, fn, data, **kwargs)
                    return

                except RuntimeError as err:
                    # TODO also need to apply this logic in savefig? or just replace
                    # savefig w/ a simpler version (wrapped w/ @produces_output)?
                    # no plots really saved directly under main (at least not
                    # unconditionally...)
                    calling_fn = caller_info().name
                    if calling_fn == 'main' or calling_fn in _consider_as_main:
                        if check_outputs_unchanged == 'nonmain':
                            # TODO refactor to share mtime/etc parts of message w/ other
                            # main places similar message (w/ 'would have changed')
                            # defined? not sure it matters enough here...
                            warn(f'{path} would have changed!\nnot failing because -C '
                                'passed (instead of -c), and this output saved directly'
                                ' in main().\nremove -C/-c and re-run to overwrite.'
                            )
                            # TODO option to not return (which would allow the `fn` call
                            # below to overwrite this output)?
                            return

                        elif check_outputs_unchanged == True:
                            assert len(err.args) == 1
                            curr_msg = err.args[0]
                            err = RuntimeError(f'{curr_msg}\n\nor pass -C instead of '
                                '-c, to ignore outputs (like this one) saved directly '
                                'from main()'
                            )

                    write_output = _output_change_prompt_or_err(err, path)
            else:
                write_output = True

            if write_output:
                # TODO test! (and test arg kwarg actually useable on wrapped fn, whether
                # or not already wrapped fn has this kwarg. can start by assuming it
                # doesn't have this kwarg tho...)!
                #
                # (have already manually tested cases where wrapped fns do not have
                # existin verbose= kwarg. just need to test case where wrapped fn DOES
                # have existing verbose= kwarg now.)
                if verbose:
                    print(f'writing {path}')

                fn(data, path, **kwargs)

        return wrapped_fn

    # TODO what is this for again?
    if _fn is None:
        return wrapper_helper
    else:
        return wrapper_helper(_fn)


# if one output would get written two twice in one run of this script.
# for most outputs, we only intend to write them once, and this indicates an error.
class MultipleSavesPerRunException(IOError):
    pass


# set True by al_analysis if -P passed as CLI arg
prompt_if_changed = False

def _output_change_prompt_or_err(err: RuntimeError, path: Path) -> bool:
    """Returns whether `path` should be written to with new version.
    """
    assert err is not None
    if not prompt_if_changed:
        raise err

    assert len(err.args) == 1
    msg = err.args[0]
    # TODO still show lineno (of fn calling wrapped_fn) in this case?
    warn(msg)

    overwrite = None
    # TODO try to have Ctrl-c behave same as 'n'?
    while True:
        # TODO option to accept all future prompts too? (would still want to warn for
        # each)
        # TODO maybe for anything <= current rms (prompting again if new max)?
        # TODO option to start debugger here (to step up and inspect difference)?
        response = input(f'overwrite {path}?\n[y]es / [n]o (quit) / [s]kip (continue) '
            '(press enter after selection)'
        )
        response = response.lower()

        if response == 'y':
            overwrite = True
            break

        elif response == 'n':
            raise err

        # [s]kip
        elif response == 's':
            overwrite = False
            break
        else:
            print('invalid response. expected y/n/s.')

    assert overwrite is not None
    return overwrite


def caller_info() -> traceback.FrameSummary:
    """Returns FrameSummary from one stack level above.

    FrameSummary objects have attributes: filename, lineno, name, line (as well as
    locals).
    """
    stack = traceback.extract_stack()
    assert len(stack) >= 4
    # NOTE: .name of each element is the function name of that element
    # of the stack. if at module level (not within a function), .name
    # should be (literal, not w/ actual module name) '<module>'
    assert stack[0].name == '<module>'
    assert stack[-1].name == 'caller_info' and not any(
        x.name == 'caller_info' for x in stack[:-1]
    )
    # n_levels_up=1 would just put us in the caller of caller_info, which is not useful.
    # we want the caller of caller_info's caller.
    # TODO expose as kwarg? no use less than 2 tho
    n_levels_up = 2
    frame_summary = stack[-1 - n_levels_up]
    return frame_summary


@produces_output
# input could be at least Series|DataFrame
def to_csv(data, path: Path, **kwargs) -> None:
    """
    NOTE: `produces_output` wrapper modifies fn to allow `Pathlike` for path arg
    """
    data.to_csv(path, **kwargs)


@produces_output(verbose=False)
# input could be at least Series|DataFrame
# TODO add flag (maybe via changes to wrapper?) that allow overwriting same thing
# written already in current run
def to_pickle(data, path: Path) -> None:
    """
    NOTE: `produces_output` wrapper modifies fn to allow `Pathlike` for path arg
    """
    if isinstance(data, xr.DataArray):
        path = Path(path)
        # read via: pickle.loads(path.read_bytes())
        # (note lack of need to specify protocol)
        # just specifying protocol b/c docs say it is (sometimes?) much faster
        path.write_bytes(pickle.dumps(data, protocol=-1))
        return

    if hasattr(data, 'to_pickle'):
        # TODO maybe do this if instance DataFrame/Series, but otherwise fall back to
        # something like generic read_pickle?
        data.to_pickle(path)

    path.write_bytes(pickle.dumps(data))


# TODO move to hong2p?
def read_pickle(path: Pathlike):
    path = Path(path)
    return pickle.loads(path.read_bytes())


@produces_output(verbose=False)
def np_save(data: np.ndarray, path: Path, **kwargs) -> None:
    """
    NOTE: opposite order of args to `np.save`, which has path first and data second.
    necessary to work w/ my `produces_output` wrapper.
    """
    np.save(path, data, **kwargs)


def text_diff(f1: Path, f2: Path) -> str:
    lines1 = f1.read_text().strip().splitlines(keepends=True)
    lines2 = f2.read_text().strip().splitlines(keepends=True)

    # TODO add git diff style coloring (would conflict w/ how current output is returned
    # as a str which gets often displayed via a `warn` call that already colors output
    # yellow. could return to calling sys.stderr.writelines in here, but would
    # complicate keeping style consistent w/ warning in general)?
    #
    # for reference, default ubuntu diff output on same two files as below:
    # $ diff .../params.csv /tmp/tmph87sfj7j.csv
    # 10c10
    # < used_model_cache,True
    # ---
    # > used_model_cache,False
    #
    # example output:
    # --- .../params.csv
    # +++ /tmp/tmph87sfj7j.csv
    # @@ -10 +10 @@
    # -used_model_cache,True
    # +used_model_cache,False
    #
    # TODO also populate fromdate [+todate?] fields? (w/ mtime? what format?)
    # (seems they take str, so pre-formatted as i want)
    #
    # NOTE: do need to convert path to str for fromfile/tofile
    diff = difflib.unified_diff(lines1, lines2, fromfile=str(f1), tofile=str(f2), n=0)

    # elements in `diff` seem to already have linesep chars at end, so don't need
    # '\n'.join(diff)
    return ''.join(diff)


_save_fn_name2diff_fn = {
    'to_csv': text_diff,
}
# TODO TODO unit test all combinations of changed/unchanged w/ CSV / pickle / some plot
# formats
# TODO option to use np.isclose or something instead of exact file comparison?
# (mainly thinking for CSVs, where the specific way the computation is done might change
# and just lead to a non-important numerical change. hasn't been an issue so far though)
# TODO change so save_fn is the optional one (rather than data) and automatically use
# .savefig if that positional argument has that attribute (or is a Figure/FacetGrid?
# checking for savefig attr probably better...)
# TODO also support objects w/ .save(path) method? (e.g. statsmodels models)
# TODO option to touch files-that-would-be-unchanged to have mtime as if they were just
# written?
def _check_output_would_not_change(path: Path, save_fn: Callable, data=None, **kwargs
    ) -> None:
    """Raises RuntimeError if output would change.

    Args:
        path: must already exist (raises IOError if not)
        *args, **kwargs: passed to `save_fn`
    """
    if not path.exists():
        raise IOError(f'{path} did not exist!')

    # TODO derive name in deterministic way from path, so that same input will always
    # overwrite any pre-existing temp outputs? not a huge priority, just some temp files
    # could get left around as-is (which is probably fine. generally get deleted on
    # reboot)
    #
    # TODO test w/ input where output filename doesn't already have suffix?
    # support that?
    # (if input doesn't have '.' in name, suffix is ''. if input name starts with '.',
    # and there isn't another '.', suffix is also '')
    #
    # need to use existing suffix when savefig is matplotlib Figure.savefig, as it will
    # save to `temp_file + '.png'` instead of `temp_file`.
    # path.suffix for e.g. Path('x/y/z.pdf') is '.pdf'.
    temp_file = NamedTemporaryFile(delete=False, suffix=path.suffix)
    # also includes directory
    temp_file_path = Path(temp_file.name)

    # TODO move some/all of use_mpl_comparison/is_pickle def to just before conditional
    # (as part of factoring out a file comparison fn from within this)
    use_mpl_comparison = False

    # for save_fn input like:
    # <bound method Figure.savefig of <Figure size 1920x1440 with 1 Axes>>
    # I wasn't actually seeing __name__ in dir(save_fn) for that, but accessing
    # .__name__ still worked (providing 'savefig'). __func__ / __self__ may be the
    # function and bound instance, but not sure, and this seems like it might work ok.
    #
    # TODO test w/ seaborn input
    if save_fn.__name__ == 'savefig':
        use_mpl_comparison = True

    is_pickle = False
    if path.suffix == '.p':
        # TODO could also check if save_fn is to_pickle (or has 'pickle' in name, or
        # name split on '_'?). i always name pickles w/ .p, but if anyone else uses this
        # fn (for pickles), might matter.
        # TODO more generally, check if there is a read_<x> fn defined in same scope as
        # input to_<x> fn name (-> use read fn to compare if filecmp fails). or accept
        # read fn as optional input?
        assert not use_mpl_comparison
        is_pickle = True

    # TODO set verbose=False if save_fn already had that?
    # assuming we don't need to for now. any other issues w/ calling wrapped
    # fn twice?
    # TODO possible to refactor to not need to check data (maybe using *args, and
    # changing all fns using produces_output to swap order of data and path args?)?
    if data is not None:
        save_fn(data, temp_file_path, **kwargs)
    else:
        # TODO assert use_mpl_comparison here (and only here?)? (should be only case i'm
        # currently not passing in data, and no real plans for that to change)
        save_fn(temp_file_path, **kwargs)

    temp_file.close()
    # TODO assert some bytes have been written to file?
    # (would have caught save_fn appending suffix issue before i was using existing
    # suffix)

    to_delete = []
    mtime_str = format_mtime(path, year=True, seconds=True)
    err_msg = f'{path} ({mtime_str}) would have changed! (run without -c/-C to ignore)'

    # TODO factor out file comparison fn(s) from this?
    # (-> use to check certain key outputs same as what is committed to repo, e.g.
    # CSVs w/ model responses in data/sent_to_anoop vs current ones)
    # TODO + make available as CLI?

    if not use_mpl_comparison:
        # https://stackoverflow.com/questions/1072569
        unchanged = filecmp.cmp(path, temp_file_path, shallow=False)

        # could also *always* do this comparison, instead of filecmp above, but I'm
        # assuming actually loading the pickles is more expensive than the above.
        # if the filecmp approach *usually* works for pickles (as it seems to), then
        # it's probably better to only load+compare the pickles if that check fails.
        if is_pickle and not unchanged:
            # TODO these read_pickle fns always return output consistent w/
            # pd.read_pickle (in only case checked so far, yes)?
            old = read_pickle(path)
            new = read_pickle(temp_file_path)

            # TODO what about if new has 'equals' and old doesn't (could still get
            # ValueError about truth value ambiguous, if new is series?)
            if hasattr(old, 'equals'):
                unchanged = old.equals(new)
            else:
                # TODO also catch possible ValueError here
                # (why need cast to bool in some but not all cases? Series vs
                # DataFrames?)
                # (this was b/c old and new were both dicts w/ some values that were
                # pd.Series in old and np.array in new)
                # (also got it when new was Series and old was np.array)
                try:
                    unchanged = old == new
                except ValueError:
                    # TODO delete
                    print()
                    print(f'{type(old)=}')
                    print(f'{type(new)=}')
                    import ipdb; ipdb.set_trace()
                    #

                try:
                    unchanged = bool(unchanged)

                # will trigger if old/new are like numpy arrays, like:
                # ValueError: The truth value of an array with more than one element is
                # ambiguous. Use a.any() or a.all()
                #
                # even though the `old.equals(new)` check above should be used for
                # pandas objects (and probably anything else that would have this type
                # of error...), a similarly worded error would be emitted if trying to
                # coerce pandas elementwise comparisons to a single bool. e.g.
                # ValueError: The truth value of a DataFrame is ambiguous. Use a.empty,
                # a.bool(), a.item(), a.any() or a.all().
                except ValueError as err:
                    # TODO also filter on err msg, only trying to coerce check w/ .all()
                    # if msg matches [some parts of] expected err msg?
                    unchanged = np.all(unchanged)
    else:
        # TODO TODO when factoring out file comparison fn, def use_mpl_comparison from
        # whether extension is in `mpl_compare.comparable_formats()`?
        # (currently ['png', 'pdf', 'eps', 'svg'])
        #
        # TODO can i remove this? what happens if compare_images gets an input w/ a
        # non-comparable format?
        assert path.suffix[1:].lower() in mpl_compare.comparable_formats()

        # TODO want to actually use tolerance > 0 ever? (reading mpl's code, if tol is
        # 0, np.array_equal is used, rather than mpl_compare.calculate_rms)
        #
        # 2025-02-19: getting some PDFs i can't visually tell apart, w/ rms ~3.84
        # (mb_modeling/hist_hallem.pdf)
        tolerance = 0
        # TODO keep something like this? worried it'd also miss a point meaningfully
        # moving around / similar... (as opposed to weird text spacing changes that
        # seemed non-deterministic)
        #tolerance = 15

        # TODO fix ImageComparisonFailure (can i repro?):
        # ...
        #   File "./al_analysis.py", line 4095, in plot_corr
        #     savefig(fig, plot_dir, prefix, bbox_inches='tight', debug=verbose, **_save_kws)
        #   File "./al_analysis.py", line 2988, in savefig
        #     _check_output_would_not_change(fig_path, save_fn, **kwargs)
        #   File "./al_analysis.py", line 2503, in _check_output_would_not_change
        #     diff_dict = mpl_compare.compare_images(path, str(temp_file_path), tolerance,
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/testing/compare.py", line 466, in compare_images
        #     rms = calculate_rms(expected_image, actual_image)
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/testing/compare.py", line 363, in calculate_rms
        #     raise ImageComparisonFailure(
        # matplotlib.testing.exceptions.ImageComparisonFailure: Image sizes do not match expected size: (359, 382, 3) actual size (359, 383, 3)

        # compare_images seems to save a png alongside input image, and i don't think
        # there are options to not do that, so i'm just deleting those files after
        #
        # https://matplotlib.org/devdocs/api/testing_api.html#module-matplotlib.testing.compare
        #
        unchanged = None
        try:
            # from docs: "Return None if the images are equal within the given
            # tolerance."
            diff_dict = mpl_compare.compare_images(path, str(temp_file_path), tolerance,
                # TODO remove this, since i couldn't really use it to find files to
                # delete anyway (since it's None if no diff)?
                in_decorator=True
            )

        except ImageComparisonFailure as err:
            # TODO fix
            warn(err)
            #
            unchanged = False

        # paths in diff_dict (for temp_file_path=/tmp/tmpdcsvhhzb.pdf) should look like:
        # 'actual': '/tmp/tmpdcsvhhzb_pdf.png',
        # 'diff': '/tmp/tmpdcsvhhzb_pdf-failed-diff.png',
        # 'expected': 'pebbled_6f/pdf/ijroi/mb_modeling/hist_hallem_pdf.png',

        # TODO try to move the to_delete handling to after this conditional
        # (so i can factor this whole conditional, w/ some of earlier stuff, into fn for
        # just comparing files, not doing any of the temp file creation / cleanup)

        # bit more flexible than Path.with_suffix
        def with_suffix(filepath, suffix):
            return filepath.parent / f'{filepath.stem}{suffix}'

        def temp_with_suffix(suffix):
            return with_suffix(temp_file_path, suffix)

        if plot_fmt != 'png':
            # don't think i need to worry about these two if plot_fmt == 'png'
            to_delete.extend([
                temp_with_suffix(f'_{plot_fmt}.png'),
                with_suffix(path, f'_{plot_fmt}.png')
            ])

        if unchanged is None:
            if diff_dict is None:
                unchanged = True
            else:
                # NOTE: <x>_failed-diff[_<plot-format>].png created in this case, but we
                # probably want to keep it for inspection, so not adding to to_delete
                unchanged = False
                err_msg += ('\n\nmatplotlib.testing.compare.compare_images output:\n'
                    f'{pformat(diff_dict)}\n\ndiff image kept for inspection'
                )
                # TODO assert diff_dict['actual'] is same as temp_file_path?

                # TODO also open up (xdg-open / whatever) temp png matplotlib wrote (the
                # diff image)?
                # (and yes, it does convert everything to PNG for this, named like
                # /tmp/tmpa3hi5nxy_pdf-failed-diff.png (for diff),
                # /tmp/tmpa3hi5nxy_pdf.png)

    # want to leave this file around if changed, so we can compare to existing output
    if unchanged:
        to_delete.append(temp_file_path)

    for temp_path in to_delete:
        assert temp_path.exists(), f'{temp_path=} did not exist!'
        temp_path.unlink()

    if unchanged:
        if verbose:
            print(f'{path} would be unchanged')

        return

    # TODO (delete?) move temp file into place, if DOESN'T match (after warning)
    # (would need new value for -c, to warn instead of err. not sure i want)?
    # (also, would have to handle in callers)

    save_fn_name = save_fn.__name__
    if save_fn_name in _save_fn_name2diff_fn:
        diff_fn = _save_fn_name2diff_fn[save_fn_name]
        diff_str = diff_fn(path, temp_file_path)
        err_msg += f'\n\n{diff_str}'
    else:
        err_msg += ('\n\ncompare with what would have been new output, saved at: '
            f'{temp_file_path}'
        )

    raise RuntimeError(err_msg)




# TODO clarify how these behave if something is missing (in comment)
panel2name_order = deepcopy(natmix.panel2name_order)
panel_order = list(natmix.panel_order)

diag_panel_str = 'glomeruli_diagnostics'

# TODO any reason for this order (i think it might be same as order in yaml [which more
# or less goes from odors activating glomeruli in higher planes to lower planes], so
# maybe could load from there now?)? just use order loaded from config /
# glomeruli_diagnostics.yaml?
#
# TODO actually load from generator config (union of all loaded w/ this panel?)
# -> use associated glomeruli keys of odors to sort
#
# Sorted manually to roughly alphabetically sort by names of glomeruli we are trying to
# target with these diagnostics.
panel2name_order[diag_panel_str] = [
    '3mtp',
    'va',
    # DL5
    't2h',
    'ms',
    'a-terp',
    'geos',
    # DM5
    'e3hb',
    'mhex',
    # DM4
    'ma',
    # DM1
    'ea',
    # VM7d
    '2-but',
    '2h',
    'ga',
    'HCl',
    'carene',
    'farn',
    '4h-ol',
    '2but',
    # VA2
    '2,3-b',
    'p-cre',
    'aphe',

    # TODO was this working before? didn't i have my own abbrev as 6ol? or did something
    # else overwrite that?
    '1-6ol',
    'paa',
    'fench',
    'CO2',

    # VM2/VA2/?
    'ecrot',

    # VC4 (+VA2)
    'elac',
    # ~VM3
    'acetoin',
]

panel_order = [diag_panel_str] + panel_order

# TODO get order from yaml files if not specified? per-panel flag to do this?
# TODO (as w/ other stuff), load from either generated or generator-input YAMLs
# (in this case, tom_olfactometer_configs/megamat0.yaml) (should have code for initial
# part of this now)
panel2name_order['megamat'] = [
    '2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al',
    't2h', '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms'
]
# Putting this before my 'control' panel, so that shared odors are plotted in correct
# order for this data
panel_order.insert(1, 'megamat')

panel_order.append('validation2')
# TODO TODO which of these did i actually want? check all odors match up in my data
# (maybe fix some of my abbrevs to match remy's?)
panel2name_order['validation2'] = [
    '+pul',
    'menth',
    'long',
    'sab',
    '-bCar',
    '-aPine',
    'euc',
    '2-mib',
    'geos',
    'guai',
    'mchav',
    'PEA',

    # NOTE: diff from remy's 'PAA'
    'paa',
    'PAA',

    'bbenz',
    'B-myr',
    'ger',

    # NOTE: diff from remy's '1-prop'
    '1-3ol',
    '1-prop',

    '1p3one',
    '1o3one',
    'EtOct',
    '2-mba',
    'Z4-7al',
]

# TODO also let this take a list of odors? or somehow use output + input to do something
# that is effectively like an argsort, and use that to index some other type of object
# (where we convert it to a DataFrame just for sorting)
# TODO olf.sort_odors allow specifying axis?
# TODO TODO TODO how to get this to not sort control panel '2h @ -5 + oct @ -3' (air
# mix, using odor2 level) right after '2h @ -5'? (and same for kiwi)
def sort_odors(df: pd.DataFrame, add_panel: Optional[str] = None, **kwargs
    ) -> pd.DataFrame:

    # TODO add to whichever axis has odor info automatically? or too complicated.
    # currently doesn't work if odors are in columns.
    if add_panel is not None:
        assert type(add_panel) is str
        # TODO assert 'panel' not already in (relevant axis) index level names
        # (or just ignore if it is, at least if already there)
        df = util.addlevel(df, 'panel', add_panel)

    return olf.sort_odors(df, panel_order=panel_order,
        # TODO what does if_panel_missing=None do? comment explaining (/ change code to
        # not require an explanation...)
        panel2name_order=panel2name_order, if_panel_missing=None, **kwargs
    )


megamat_odor_names = set(panel2name_order['megamat'])

def odor_is_megamat(odor: str) -> bool:
    """Takes odor str (with conc) like 'va @ -3' to whether it's among 17 megamat odors.
    """
    return olf.parse_odor_name(odor) in megamat_odor_names


# changed in al_analysis.main (exposed as command line arguments)
# (to a set of strings)
ignore_existing: Union[bool, Set[str]] = False

# TODO distinguish between these two in CLI doc (or delete blanket -i)
# TODO should i even have a blanket '-i' option?
ignore_if_explicitly_requested = ('json',)
also_ignore_if_ignore_existing_true = ('nonroi', 'ijroi', 'suite2p', 'dff2spiking',
    'model'
)
ignore_existing_options = (
    ignore_if_explicitly_requested + also_ignore_if_ignore_existing_true
)

def should_ignore_existing(name: str, explicit_only: bool = False) -> bool:
    """
    Args:
        name: step to check whether it should be recomputed.
            see `ignore_existing_options`.

        explicit_only: will only ignore `name` if it is in `ignore_existing`, not if
            `ignore_existing == True`
    """
    if type(ignore_existing) is bool:

        # Some steps are unlikely enough to need recomputation, we will only do so if we
        # explicitly request that step be recomputed.
        if name in ignore_if_explicitly_requested:
            return False

        return ignore_existing
    else:
        assert name in ignore_existing_options
        # (if it's not a bool, it should be guaranteed to be a set-of-str here)
        return name in ignore_existing # pylint: disable=unsupported-membership-test


# TODO test still works when used from al_analysis (al_analysis fns still using this
# _dirs_to_delete_if_empty, referring to it w/ `al_util.` prefix now)
_dirs_to_delete_if_empty = []
# TODO work w/ pathlib input?
def makedirs(d):
    """Make directory if it does not exist, and register for deletion if empty.
    """
    # TODO make sure if we make a directory as well as some of its parent directories,
    # that if (all) the leaf dirs are empty, the whole empty tree gets deleted
    # TODO shortcircuit to returning if we already made it this run, to avoid the checks
    # on subsequent calls? they probably aren't a big deal though...
    os.makedirs(d, exist_ok=True)
    # TODO only do this if we actually made the directory in the above call?
    # not sure we really care for empty dirs in any circumstances tho...
    _dirs_to_delete_if_empty.append(d)


save_figs = True

# TODO support multiple (don't know if i want this to be default though, cause extra
# time)
# TODO add CLI arg to override this?
plot_fmt = os.environ.get('plot_fmt', 'pdf')

exit_after_saving_fig_containing = None
# TODO CLI flag to (or just always?) warn if there are old figs in any/some of the dirs
# we saved figs in (would only want if very verbose...)?
# TODO maybe refactor to automatically prefix path with '{plot_fmt}/' in here?
#
# Especially running process_recording in parallel, the many-figures-open memory
# warning will get tripped at the default setting, hence `close=True`.
#
# TODO why Type[sns.axisgrid.Grid] instead of just sns.axisgrid.Grid?
# (is below comment why? i assume so...)
# sns.FacetGrid and sns.ClusterGrid should both be subclasses of sns.axisgrid.Grid
# (can't seem to import sns.ClusterGrid anyway... maybe it's somewhere else?)
# TODO TODO try to replace logic w/ this decorator
#@produces_output(verbose=False)
_savefig_seen_paths = set()
# TODO possible to use verbose kwarg alongside global verbose?
# (for using this fn in other scripts, would rather have kwarg, but also don't want to
# have to pass verbose=verbose everywhere in here, nor use global as default kwarg,
# which wouldn't be update by CLI arg correctly...)
def savefig(fig_or_seaborngrid: Union[Figure, Type[sns.axisgrid.Grid]],
    fig_dir: Pathlike, desc: str, *, close: bool = True, normalize_fname: bool = True,
    debug: bool = False, **kwargs) -> Path:

    global exit_after_saving_fig_containing

    # TODO delete (after checking i never actually added code that actually used the
    # plot_fmt kwarg i had on this fn for a little bit late 2024, removed in december)
    assert 'plot_fmt' not in kwargs

    # TODO delete
    if plot_fmt == 'pdf':
        # even needed in current mpl? not referenced in current docs, and not obviously
        # in settings current mpl testing code enforces for tests
        kwargs['metadata'] = {'creationDate': None}
    #

    if normalize_fname:
        prefix = util.to_filename(desc)
    else:
        # util.to_filename in branch above adds a '.' suffix by default
        prefix = f'{desc}.'

    # TODO also allow input to have extension (use that if passed)?
    # (meh, already exposed plot_fmt kwarg)
    # TODO actually modify to_filename to not throw out '.', and manually remove that in
    # any remaining cases where i didn't want it? for concentrations like '-3.5', this
    # makes them more confusing to read... (-> '-35')
    basename = prefix + plot_fmt

    # TODO delete / fix
    makedirs(fig_dir)
    #
    fig_path = Path(fig_dir) / basename

    # TODO share logic w/ to_csv above (meaning also want to resolve() in
    # produces_output? or no? already doing that?)
    abs_fig_path = fig_path.resolve()

    # TODO delete try/except (can i repro failure?)
    # 2023-12-04:
    # $ ./al_analysis.py -t 2022-02-03 -e 2022-04-03 -v -s model
    #thorimage_dir: 2022-02-22/1/kiwi_ea_eb_only
    #thorsync_dir: 2022-02-22/1/SyncData003
    #yaml_path: 20220222_184517_stimuli/20220222_184517_stimuli_0.yaml
    #TIFF (/ motion correction) changed. updating non-ROI outputs.
    #ImageJ ROIs were modified. re-analyzing.
    #...
    #merging ROI VM7d?
    #selecting input ROI 10 as best plane
    #dropping other input ROIs [9]
    #           roi_quality
    #roi_index
    #9             0.049183
    #10            0.054798
    #
    #Uncaught exception
    #Traceback (most recent call last):
    #  File "./al_analysis.py", line 10766, in <module>
    #    main()
    #  File "./al_analysis.py", line 9580, in main
    #    was_processed = list(starmap(process_recording, keys_and_paired_dirs))
    #  File "./al_analysis.py", line 3949, in process_recording
    #    ij_trial_df, best_plane_rois, full_rois = ij_trace_plots(analysis_dir,
    #  File "./al_analysis.py", line 3096, in ij_trace_plots
    #    trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
    #  File "./al_analysis.py", line 3044, in trace_plots
    #    savefig(fig, roi_plot_dir, str(roi))
    #  File "./al_analysis.py", line 1406, in savefig
    #    assert abs_fig_path not in _savefig_seen_paths
    #AssertionError
    # TODO delete
    # TODO TODO why is it trying to save this twice?
    if '/'.join(abs_fig_path.parts[-3:]) in (
        '2022-02-22_1_kiwi_ea_eb_only/ijroi/DL1.png',
        # TODO TODO fix this too. can repro by same command as above
        # (w/o need for `-i ijroi`)
        # other stuff also affected here, also (prob not exclusively):
        # corr_certain_only/kiwi/2022-03-31_1.png
        'corr_certain_only/kiwi/2022-03-30_1.png'
        ):

        print('SAVING ONE OF WHAT WILL BE A DUPLICATE FIGURE NAME')
        traceback.print_stack(file=sys.stdout)
        #import ipdb; ipdb.set_trace()
    #

    # TODO (option to) also save traceback at each, for easier debugging?

    try:
        assert abs_fig_path not in _savefig_seen_paths
    except AssertionError:
        print(f'{abs_fig_path=}')
        print(f'{desc=}')
        # TODO TODO TODO fix:
        # no uncertain ROIs. not generating uncertain_by_max_resp fig
        # done
        # Warning: correlation shapes unequal (in plot_corrs input)! shapes->counts: {(33, 33): 10, (72, 72): 5}
        # some mean correlations will have more N than others!
        # PREFIX DRIVER/INDICATOR IF I CAN (WORTH PASSING? COMPUTE FROM DIR)
        # RESTORE CSV SAVING (AFTER DE-DUPING...)
        # writing pebbled_6f/pdf/ijroi/corr_certain_only/control_flies.csv
        # PREFIX DRIVER/INDICATOR IF I CAN (WORTH PASSING? COMPUTE FROM DIR)
        # RESTORE CSV SAVING (AFTER DE-DUPING...)
        # writing pebbled_6f/pdf/ijroi/corr_certain_only/kiwi_flies.csv
        # abs_fig_path=PosixPath('/home/tom/src/al_analysis/pebbled_6f/pdf/ijroi/corr_certain_only/kiwi/2024-09-03_1.pdf')
        # desc='2024-09-03_1'
        # > /home/tom/src/al_analysis/al_analysis.py(2581)savefig()
        # -> 2581     _savefig_seen_paths.add(abs_fig_path)
        # ipdb> u
        # > /home/tom/src/al_analysis/al_analysis.py(8313)plot_corrs()
        # -> 8313         fig_path = savefig(fig, panel_dir, fly_plot_prefix)
        import ipdb; ipdb.set_trace()
    #

    _savefig_seen_paths.add(abs_fig_path)

    _skip_saving = False
    # TODO delete if i manage to restore use of produces_output wrapper around savefig
    if check_outputs_unchanged and fig_path.exists():
        save_fn = fig_or_seaborngrid.savefig
        try:
            # TODO wrapper kwarg (that would be added to some/all decorations)
            # for disabling this for some calls? e.g. for formats where they can
            # change for uninteresting reasons, like creation time
            _check_output_would_not_change(fig_path, save_fn, **kwargs)
            _skip_saving = True

        except RuntimeError as err:
            overwrite = _output_change_prompt_or_err(err, fig_path)
            _skip_saving = not overwrite
    #

    # TODO should i be passing metadata={'creationDate': None} in pdf case?
    # would that help make diffing easier / more reliable? still needed?
    # https://matplotlib.org/2.1.1/users/whats_new.html#reproducible-ps-pdf-and-svg-output
    if save_figs and not _skip_saving:
        fig_or_seaborngrid.savefig(fig_path, **kwargs)

        # TODO delete
        # TODO TODO why this call always seem to fail (w/ below exception)?
        # TODO TODO fix:
        # ...
        #     diff_dict = mpl_compare.compare_images(fig_path, fig_path, tolerance,
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/testing/compare.py", line 445, in compare_images
        #     actual = convert(actual, cache=True)
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/testing/compare.py", line 310, in convert
        #     convert(path, newpath)
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/testing/compare.py", line 135, in __call__
        #     raise ImageComparisonFailure(
        # matplotlib.testing.exceptions.ImageComparisonFailure: Processing pages 1 through 1.
        # Page 1
        # GS>
        #tolerance = 0
        #diff_dict = mpl_compare.compare_images(fig_path, fig_path, tolerance,
        #    in_decorator=True
        #)
        #assert diff_dict is None, f'{fig_path} NOT equal to itself (compare_images)'
        #print(f'{fig_path} was equal to itself (according to compare_images)')
        #

    fig = None
    if isinstance(fig_or_seaborngrid, Figure):
        fig = fig_or_seaborngrid

    elif isinstance(fig_or_seaborngrid, sns.axisgrid.Grid):
        fig = fig_or_seaborngrid.fig

    assert fig is not None

    # al_util.verbose flag set True/False by CLI in al_analysis.main
    if (verbose and not _skip_saving) or debug:
        # TODO may shorten to remove first two components of path by default
        # (<driver>_<indicator>/<plot_fmt> in most/all cases)?
        color = 'light_blue'
        cprint(fig_path, color)

    if (exit_after_saving_fig_containing and
        exit_after_saving_fig_containing in str(fig_path)):

        warn('exiting after saving fig matching '
            f"'{exit_after_saving_fig_containing}':\n{fig_path}"
        )
        sys.exit()

    # TODO warn if any figs are wider than would fit in a sheet of paper?
    # at least for a subset of figs marked as for publications?
    # (e.g. diagnostic example plot_rois fig i'm working on now)

    if close:
        if fig is None:
            raise ValueError(
                f'cannot close unknown plot object of type={type(fig_or_seaborngrid)}'
            )

        plt.close(fig)

    return fig_path


# TODO refactor (duped in pn_convergence_vs_hallem_corr.py)
# (and may want to use elsewhere in here for picking odors to use for constructing dF/F
# -> spike rate fn)
# TODO TODO hardcode abbrevs for all hallem odors (or at least all we've ever used?)
# in olf.odor2abbrev, so they can all be found here (input names could be diff from what
# we routinely use) (in the meantime, check all odors i have decent data for are NOT in
# hallem)
# TODO fn to combine this w/ loading of hallem data (-> only use that wrapper)?
def abbrev_hallem_odor_index(df: pd.DataFrame, axis='index') -> pd.DataFrame:
    """Abbreviates Hallem odor names in single-level row index.
    """
    # TODO assert some got replaced (as a check axis was correct)

    # - a-terp (a-terpineol)
    # - 3mtp (3-methylthio-1-propanol)
    # - carene ((1S)-(+)-3-carene)
    #
    # - (don't currently have data, but may soon) o-cre (2-methylphenol)
    df = df.rename({
            # TODO just move all these into olf.odor2abbrev
            'b-citronellol': 'B-citronellol',
            'isopentyl acetate': 'isoamyl acetate',
            'E2-hexenal': 'trans-2-hexenal',
            'a-terpineol': 'alpha-terpineol',
            '3-methylthio-1-propanol': '3-methylthiopropanol',
            '(1S)-(+)-3-carene': '(1S)-(+)-carene',
            '2-methylphenol': 'o-cresol',

            # these should be the only 7 odors in validation2 panel that are in Hallem,
            # courtesy of what Remy sent me on slack 2024-02-12.
            # already abbreviated (concentration in validation2 panel in parens):
            # 1-propanol (-3), ethyl octanoate (-2), geraniol (-2),
            # phenylacetaldehyde (-4)
            #
            # (-2)
            'b-myrcene': 'beta-myrcene',
            # (-1.5)
            'a-pinene': '(-)-alpha-pinene',
            # (-3)
            '(-)-trans-caryophyllene': '(-)-beta-caryophyllene',

            # TODO restore -> re-run hallem modelling -> check nothing broke
            # (-> maybe remove some/all other hacks to odor abbrev handling in creation
            # of my versions of 2E plot)
            #
            ## abbreviations Remy uses in some of 2E data she sent me.
            ## (for the 24 - 17 odors she is had some data for, beyond megamat17).
            ## all should be in hallem.
            ## I already had an abbreviation for the 7th ('1-penten-3-ol' -> '1p3ol').
            #'eugenol': 'eug',
            #'ethyl cinnamate': 'ECin',
            #'propyl acetate': 'PropAc',
            ## these two had full names w/ 'gamma'/'delta' prefix in some of Remy's
            ## things, but these LHS values are what I need to convert from in my Hallem
            ## representation.
            #'g-hexalactone': 'g-6lac',
            #'d-decalactone': 'd-dlac',
            ## TODO TODO might need 'moct' -> 'MethOct' (assuming my abbrev has
            ## already been applied [it prob wasn't, considering it's called below...].
            ## how have i been handling any other existing abbrev conflicts? are there
            ## any others?)
            ## TODO TODO how do i want to handle this? leaning towards just renaming
            ## to 'moct' when loading remy's 2e data...
            ##'methyl octanoate': 'MethOct',
        },
        axis=axis
    )
    df = df.rename(olf.odor2abbrev, axis=axis)
    return df


# TODO move to hong2p.util or something?
def n_choose_2(n: int) -> int:
    ret = (n - 1) * n / 2
    assert np.isclose(ret, int(ret))
    return int(ret)


# TODO refactor to hong2p (replace util.melt_symmetric [used once in here])?
# TODO TODO test that a given pair will always be returned in a fixed order, so that no
# matter the set of odors we pass in here, we can average across outputs from diff calls
# to this fn. relevant for now loading old megamat data. not sure current implementation
# has this property... (as long as i'm always using ordered_pairs where needed, should
# be ok... kind of a hack though)
# TODO TODO add option to pass sequence of odors, to generate something like ordered
# pairs? (for convenience when averaging over groups w/ diff sets of odors/pairs)
def corr_triangular(corr_df, *, ordered_pairs=None):
    assert corr_df.index.equals(corr_df.columns)

    # TODO this causing difficulties later? alternatives?
    # (w/ needing to sort again)
    #
    # sorting index so that diff calls to this will produce mergeable outputs.
    # otherwise, though the set of pairs will be the same, some may be represented
    # in the other order in this index (and thus merging won't find a matching
    # pair).
    # TODO .loc still work w/ list(...)?
    pairs = list(itertools.combinations(corr_df.index.sort_values(), 2))

    corr_ser = corr_df.stack(dropna=False)

    if ordered_pairs is not None:
        # does fail in call from end of load_remy_2e...
        # (only happened to be true for my first use case. don't think it actually
        # matters)
        #assert set(ordered_pairs) - set(corr_ser.index) == set()

        pairs = [(b,a) if (b,a) in ordered_pairs else (a,b) for a,b in pairs]

    # itertools.combinations will not give us any combinations of an element with
    # itself. in other words, we won't be keeping the identity correlations.
    assert not any(a == b for a, b in pairs)

    # itertools.combinations essentially selects one triangular, excluding diagonal
    corr_ser = corr_ser.loc[pairs]

    # TODO switch to assertion(s) on input index/column names?
    # (just to fail sooner / be more clear)
    #
    # TODO make more general than assuming 'odor' prefix?
    assert len(corr_ser.index.names) == 2

    # TODO delete?
    assert all(x.startswith('odor') for x in corr_ser.index.names), \
        f'{corr_ser.index.names=}'

    # TODO do 'a','b' instead? other suffix ('_row','_col')? (to not confused w/
    # 'odor1'/'odor2' used in many other MultiIndex levels in here, where 'odor2' is a
    # almost-never-used-anymore optional 2nd odor, where 2 delivered at same time
    # (most recently in kiwi/control 2-component ramp experiments).
    corr_ser.index.names = ['odor1', 'odor2']

    # TODO sort output so odors appear in same order as in input (within each component
    # of pair, at least)?

    return corr_ser


def invert_corr_triangular(corr_ser, diag_value=1., _index=None, name='odor'):
    if _index is None:
        for_odor_index = corr_ser.index
    else:
        for_odor_index = _index

    # TODO make more general than assuming 'odor' prefix?
    # TODO + factor to share w/ what corr_triangual sets by default (at least), in case
    # i change suffix added there
    #assert for_odor_index.names == ['odor1', 'odor2']
    # TODO rename all "odor" stuff to be more general (now that i'm not requiring
    # 'odor1'/'odor2')
    assert len(for_odor_index.names) == 2

    # unique values for odor1 and odor2 will not be same (each should have one value not
    # in other). could just sort, for purposes of generating one combined order.
    # for now, assuming (correctly, it seems) that first value in odor1 and last value
    # in odor2 are the only non shared, and that otherwise we want to keep the order
    #
    # pandas <Series>.unique() keeps order of input (assuming all are adjacent, at
    # least)
    odor1 = for_odor_index.get_level_values(0).unique()
    odor2 = for_odor_index.get_level_values(1).unique()

    # TODO TODO try to make work without these assertions (would need to change how
    # odor_indx is defined below). these seem to work if index is sorted, but not for
    # (at least some) indices before sort_index call.
    assert all(odor2[:-1] == odor1[1:])
    assert odor1[0] not in set(odor2)
    assert odor2[-1] not in set(odor1)

    # TODO maybe columns and index should have diff names? keep odor1/odor2?
    # TODO get shared prefix of cols for name=? accept as kwarg?
    odor_index = pd.Index(list(odor1) + [odor2[-1]], name=name)

    square_corr = pd.DataFrame(index=odor_index, columns=odor_index, data=float('nan'))
    for a in odor_index:
        for b in odor_index:
            if a == b:
                assert (a, b) not in corr_ser
                square_corr.at[a, b] = diag_value
                continue

            # TODO clean up
            try:
                if (a, b) in corr_ser:
                    assert (b, a) not in corr_ser
                    c = corr_ser.at[a, b]
                else:
                    assert (b, a) in corr_ser
                    c = corr_ser.at[b, a]
            except AssertionError:
                #print(f'{a=}')
                #print(f'{b=}')
                #import ipdb; ipdb.set_trace()
                c = float('nan')
            #

            square_corr.at[a, b] = c

    return square_corr


# TODO TODO use inside plot_corr, for its calc when input has multiple flies (/seeds)
# TODO add option to drop nonresponders (whether glomeruli or KCs), before computing?
# could only do stuff that is actually 0 (either via filling or in model KC responses)
# though, not just close to 0, or would have to decide how to threshold...
# TODO refactor part of internals out into a new fn that keeps id_cols metadata, but has
# corr_triangular pairs (as the 2 level MultiIndex it returns) on opposite (column) axis
# index (-> use that to calc mean -> invert_corr_triangular in here)
def mean_of_fly_corrs(df: pd.DataFrame, *, id_cols: Optional[List[str]] = None,
    square: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Args:
        df: DataFrame with odor level on row index, and levels from id_cols
            in column index.

        id_cols: column index levels, unique combinations of which identify individual
            flies (/ experimental units). if None, defaults to ['date', 'fly_num'].

        square: if True, returns square correlation matrix as a DataFrame. otherwise,
            returns one triangular (excluding diagonal) as a Series.
    """

    # TODO also allow selecting 'fly_id' as default, if there?
    if id_cols is None:
        id_cols = ['date', 'fly_num']

    # TODO TODO also work w/ 'odor' level (have in loaded model responses)
    # (or just expose as kwarg?)
    # TODO only do if 'repeat' level present? assert it's there?
    #
    # assumes 'odor2' level, if present, doesn't vary.
    # TODO assert assumption about possible 'odor2' level?
    trialmean_df = df.groupby(level='odor1', sort=False).mean()
    n_odors = len(trialmean_df)

    # TODO also want to  keep track of and append metadata?
    fly_corrs = []
    for fly, fly_df in trialmean_df.groupby(level=id_cols, axis='columns', sort=False):
        # TODO TODO pass in ordered_pairs? just expose as kwarg to this fn? (or fix
        # corr_triangular to always have a pair in a fixed order? even possible?)
        corr = corr_triangular(fly_df.T.corr())
        fly_corrs.append(corr)

    # TODO delete (just notes re-assuring myself this fn is working properly, when run
    # on megamat input)
    #
    # in case of Hallem subset of ORN data (when called from top-level modelling fn):
    # ipdb> trialmean_df.shape
    # (17, 140)
    #
    # ipdb> [num_null(x) for _, x in trialmean_df.groupby(level=id_cols,
    #     axis='columns', sort=False)]
    # [32, 30, 0, 0, 0, 0, 0, 0, 0]
    #
    # ipdb> [x.shape for _, x in trialmean_df.groupby(level=id_cols, axis='columns',
    #     sort=False)]
    # [(17, 16), (17, 15), (17, 16), (17, 15), (17, 15), (17, 16), (17, 16), (17, 16), (17, 15)]
    #
    # ipdb> [x.columns for _, x in trialmean_df.groupby(level=id_cols, axis='columns',
    #     sort=False)][0]
    # MultiIndex([('2023-04-22', 2,  'DC1'),
    #             ('2023-04-22', 2,  'DL1'),
    #             ('2023-04-22', 2,  'DL5'),
    #             ('2023-04-22', 2,  'DM2'),
    #             ('2023-04-22', 2,  'DM3'),
    #             ('2023-04-22', 2,  'DM4'),
    #             ('2023-04-22', 2,  'DM5'),
    #             ('2023-04-22', 2,  'DM6'),
    #             ('2023-04-22', 2,  'VA5'),
    #             ('2023-04-22', 2,  'VA6'),
    #             ('2023-04-22', 2,  'VC3'),
    #             ('2023-04-22', 2,  'VC4'),
    #             ('2023-04-22', 2,  'VM2'),
    #             ('2023-04-22', 2,  'VM3'),
    #             ('2023-04-22', 2, 'VM5d'),
    #             ('2023-04-22', 2, 'VM5v')],
    #            names=['date', 'fly_num', 'roi'])

    n_flies = len(fly_corrs)

    corrs = pd.concat(fly_corrs, axis='columns', verify_integrity=True)
    assert corrs.shape[1] == n_flies

    n_odors_choose_2 = n_choose_2(n_odors)
    # n choose 2. 136 for all non-identity combinations of 17 odors.
    assert len(corrs) == n_odors_choose_2

    # excludes NaN (e.g. va/aa in first 2 megamat flies)
    mean_corr_triangular = corrs.mean(axis='columns')
    assert mean_corr_triangular.shape == (n_odors_choose_2,)

    odor_order = trialmean_df.index
    assert not odor_order.duplicated().any()
    # TODO similar assertion to below, but against mean_corr_triangular?
    # (don't think i can...)

    if not square:
        return mean_corr_triangular

    mean_corr = invert_corr_triangular(mean_corr_triangular)
    assert mean_corr.shape == (n_odors, n_odors)

    assert set(odor_order) == set(mean_corr.index) == set(mean_corr.columns)

    # re-ordering odors to keep same order as input
    mean_corr = mean_corr.loc[odor_order, odor_order].copy()
    return mean_corr


# started w/ 'RdBu_r', but Remy had wanted to change to 'vlag', which she said is
# supposed to be similar, but perceptually uniform
#
# TODO maybe still use 'vlag' for diagnostic ROI vs dF/F image plots (version i
# generated was that...)
#
# actually, not switching to 'vlag', to not give Remy more work regenerating old figs
diverging_cmap = plt.get_cmap('RdBu_r').copy()

# since default set_bad seems to be ~white, which was fine for 'plasma' (which doesn't
# contain white), but now is not distinct from cmap midpoint (0.)
#
# lighter gray than 'gray', and according to Sam, B takes less issue with this color
diverging_cmap.set_bad((0.8, 0.8, 0.8))

# TODO actually set colors for stuff outside diverging_cmap range?
# (just need cmap.set_[over|under] and extend='both' (or 'min'|'max', if only want
# one of the two))

# TODO could try TwoSlopeNorm ([-0.5, 0] and [0, 2.0]?), but would probably want to
# define bounds per fly (or else compute in another pass / plot these after
# aggregating?)
# TODO rename to diverging_cmap_kws
diverging_cmap_kwargs = dict(
    cmap=diverging_cmap,

    # TODO TODO test default clip behavior of this is OK. (and contrast w/ e.g.
    # CenteredNorm, which actually has a clip=True/False kwarg, like most)
    # TODO TODO test what happens if diverging_cmap_kwargs used w/o vmin/vmax
    # specified (as diag_example_kws currently adds below) (want to get from data)
    # NOTE: specifying norm classes this way only works because some of my hong2p.viz
    # wrappers
    norm='two-slope',

    # TODO TODO want clip=True for diag_example_kws?
    # TODO TODO with clip=False (the default), is this colormap even clearly
    # showing over/under cmap range values distinctly (not sure this was ever that
    # related to clip, but pretty sure we need extend='both' set on colorbar creation
    # for this)?
)


def plot_corr(df: pd.DataFrame, plot_dir: Path, prefix: str, *, title: str = '',
    as_corr_dist: bool = False, verbose: bool = False, _save_kws=None, **kwargs
    ) -> pd.DataFrame:

    # otherwise, we assume input is already a correlation (/ difference of correlations)
    if not df.columns.equals(df.index):
        # TODO delete?
        if len(df.columns) == len(df.index):
            print('double check input is not already a correlation [diff]!')
            import ipdb; ipdb.set_trace()
        #

        # TODO TODO use new al_util.mean_of_fly_corrs instead (when appropriate, e.g. when input
        # has multiple flies [/ model seeds])?
        corr = df.corr()
    else:
        corr = df.copy()
        # to check not a corr dist input
        # TODO also check that range is consistent w/ corr and not corr-dist?
        #
        # TODO delete. won't work w/ corr_diff input i'm using in one place.
        # TODO TODO TODO fix (prob w/ new mean_of_... input)
        '''
        try:
            assert (corr.max() == 1).all()
        except AssertionError:
            import ipdb; ipdb.set_trace()
        '''

    if not as_corr_dist:
        vmin = -1.0
        vmax = 1.0
        vcenter = 0.0
        cmap = diverging_cmap
        to_plot = corr
        # TODO have cbar ticks be [-1, -0.5, 0, 0.5, 1.0] in this case
        # (in both cases i just want multiples of 0.5)
    else:
        vmin = 0.0
        vcenter = 1.0
        vmax = 2.0
        cmap = diverging_cmap.reversed()
        corr_dist = 1 - corr
        # should be this on the diagonals
        assert (corr_dist.min() == 0).all()
        to_plot = corr_dist
        # TODO have cbar ticks be [0, 0.5, 1, 1.5, 2] in this case
        # (in both cases i just want multiples of 0.5)

    # TODO TODO check that results w/ norm/vcenter not passed equiv to new results w/
    # norm='two-slope'+vcenter=0
    fig, _ = viz.matshow(to_plot, cmap=cmap, vmin=vmin, vcenter=vcenter, vmax=vmax,
        # just using 'two-slope', since 'centered' norm code would require
        # modification to get it to work w/ vcenter != 0.
        norm='two-slope', **kwargs
    )

    if len(title) > 0:
        fig.suptitle(title)

    if _save_kws is None:
        _save_kws = dict()

    # TODO any downside to hardcoding bbox_inches='tight'? was unspecified before
    savefig(fig, plot_dir, prefix, bbox_inches='tight', debug=verbose, **_save_kws)

    return corr


# TODO use/delete (and maybe refactor to include much/all of the kwargs used in
# plot_all...? see also mean_df/etc plotting in main, that recreates much of those
# kwargs for use w/ viz.matshow)
def plot_responses(df: pd.DataFrame, plot_dir: Path, prefix: str, *,
    vmin=None, vmax=None, title: str = '', _save_kws=None, **kwargs) -> None:

    fig, _ = viz.matshow(df, vmin=vmin, vmax=vmax, **diverging_cmap_kwargs, **kwargs)

    if len(title) > 0:
        fig.suptitle(title)

    if _save_kws is None:
        _save_kws = dict()

    # TODO any downside to hardcoding bbox_inches='tight'? was unspecified before
    savefig(fig, plot_dir, prefix, bbox_inches='tight', **_save_kws)


# TODO also break out a plot_responses fn from first part (-> use here)?
def plot_responses_and_corr(df: pd.DataFrame, plot_dir: Path, prefix: str, *,
    vmin=None, vmax=None, title: str = '', **kwargs) -> pd.DataFrame:

    # TODO expose bbox_inches (or remove kwarg in plot_[responses|corr], and just accept
    # the hardcode to bbox_inches='tight' in both of those, if no downside)? ever need
    # diff between the two calls?

    plot_responses(df, plot_dir, prefix, vmin=vmin, vmax=vmax, title=title, **kwargs)

    # TODO thread thru bbox_inches kwarg here (for savefig call)?
    return plot_corr(df, plot_dir, f'{prefix}_corr', title=title, **kwargs)


# TODO replace some use of this w/ diverging_cmap_kwargs?
# (e.g. in response matrix plots)
#cmap = 'plasma'
# to match remy
cmap = 'magma'

# TODO why does this decorator not seem required anymore? mpl/sns version thing?
#
# decorator to fix "There are no gridspecs with layoutgrids" warning that would
# otherwise happen in any following savefig calls
#@no_constrained_layout
# TODO find where sns.ClusterGrid is actually defined and use that as return type?
# shouldn't need any more generality (Grid was used above to include FacetGrid)
def cluster_rois(df: pd.DataFrame, title=None, odor_sort: bool = True, cmap=cmap,
    return_linkages: bool = False, **kwargs) -> sns.axisgrid.Grid:
    """
    Args:
        return_linkages: passed to `hong2p.viz.clustermap`

        **kwargs: passed to `hong2p.viz.clustermap`
    """
    # TODO doc expectations on what rows / columns of input are

    # TODO why transposing? stop doing that (-> change all inputs). will just cause
    # confusion if i use [row/col]_colors...
    if odor_sort:
        # TODO why olf.sort_odors not one defined in here?
        # After transpose: columns=odors
        df = olf.sort_odors(df.T)
    else:
        # just in case we didn't sort above. we still don't want to modify input...
        df = df.T.copy()

    # TODO plus, this will screw up [h|v]line_level_fn stuff...
    # TODO warn/fail if this is not the case? accidentally didn't hit this when there
    # was a bug in above
    if 'odor1' in df.columns.names:
        # TODO why? comment explaining? just use an appropriate index-value-dict ->
        # formatted str fn? isn't that what i do w/ hong2p.viz.matshow calls?
        df.columns = df.columns.get_level_values('odor1')

    # TODO TODO add option to color rows by fly (-> generate row_colors Series in here)
    # (values of series should be colors)
    # (see natmix_data/analysis.py get_fly_color_series)

    ret = viz.clustermap(df, col_cluster=False, cmap=cmap,
        return_linkages=return_linkages, **kwargs
    )
    if not return_linkages:
        cg = ret
    else:
        cg, row_linkage, col_linkage = ret

    ax = cg.ax_heatmap
    ax.set_yticks([])
    ax.set_xlabel('odor')

    if title is not None:
        ax.set_title(title)
        #cg.fig.suptitle(title)

    return ret


# TODO maybe some of below (stuff re: remy's kc data) should be moved into
# mb_model, rather than here in al_util? (al_analysis can import from mb_model, mb_model
# just can't import from al_analysis)

# TODO also anchor path to script dir? would only be to support running from elsewhere,
# which i prob don't care about
remy_data_dir = Path('data/from_remy')

n_final_megamat_kc_flies = 4

# NOTE: contains sparsities in top level CSVs, as well as individual fly binarized
# responses in subdirectories
#
# downloaded from Dropbox folder:
# Remy/odor_space_collab/analysis_outputs/multistage/multiregion_data/\
#     response_breadth/by_trialavg_ref_stim/megamat
# (Remy sent me a link on Slack 2024-04-04)
remy_sparsity_dir = (remy_data_dir /
    'response_rates/refstim__ep_at_-3.0__median__0.120-0.130'
)

# TODO refactor to share?
remy_date_col = 'date_imaged'
# she uses 'fly_num' same as I do, to number flies within each day (i.e. the numbers not
# unique across days). these two can generally be used to compute/lookup values for
# remy_fly_id.
remy_fly_cols = [remy_date_col, 'fly_num']

# 0, 1, ..., 3 (or all 0 in one of the CSVs, by accident, but that CSV redundant
# anyway)
remy_fly_id = 'acq'

# e.g. '1-5ol @ -3.0'
remy_odor_col = 'stim'

remy_conc_str = ' @ -3.0'
# TODO refactor to share w/ some similar fns?
def _strip_remy_concs(x):
    assert x.str.endswith(remy_conc_str).all()
    return x.str.replace(remy_conc_str, '', regex=False)


# contains CSVs remy made from the pickles she sent earlier under 'by_acq'
# (which i couldn't load)
remy_binary_response_dir = remy_sparsity_dir / 'by_acq_csvs'

# looks like the top-level sparsity CSVs are computed from peak amplitude alone
# (rather than options involving "std"), so only going to load these pickles
# (out of the 3 options in each fly_dir)
remy_fly_binary_response_fname = 'df_stim_responders_from_peak.csv'


# TODO add verbose kwarg and set False when calling for debug purposes from kc response
# loading code
def load_remy_fly_binary_responses(fly_sparsity_path: Path,
    acq_ledger: Optional[pd.DataFrame]=None, *, reset_index: bool = True,
    _seen_date_fly_combos=None) -> pd.DataFrame:
    # TODO check row/col is correct
    """Loads single fly NetCDF file to boolean (cell row X odor column) DataFrame.

    Args:
        acq_ledger: if passed in, has `index.names == remy_fly_cols`, and has
            `remy_fly_id in df.columns`

            maps ['date', 'fly_num'] to int 'acq' (`remy_fly_id`) (0-indexed).

        reset_index: if True, only return int 'cell' index. otherwise, retains all
            metadata columns in source CVS (i.e. row index names in the DataFrame)
    """
    fly_dirname = fly_sparsity_path.parent.name

    if verbose:
        print(fly_sparsity_path)

    # sep/index_col params from snippet Remy sent me
    #
    # there do only seem to be 5 index columns in these CSVs. all columns after are
    # odor names (e.g. '1-5ol @ -3.0')
    fly_df = pd.read_csv(fly_sparsity_path, sep='\t', index_col=list(range(5)))

    # 17 columns, 1 for each megamat odor.
    # index levels: cells_level_0 iscell iscell_xid0 xid0 embedding0
    #
    # iscells seems to all be 1, so no need to subset based on this/keep
    assert fly_df.index.get_level_values('iscell').astype(bool).all()

    # TODO delete (probably unimportant anyway, as no flies have only some elements
    # True, so this never has any info...)
    # TODO why is this true for some but not all?
    '''
    try:
        # not sure how this is different (from above), but at least it's also all True
        assert fly_df.index.get_level_values('iscell_xid0').all()
        print('iscell_xid0 assertion pass')
    except AssertionError:
        # NOTE: all False for this one fly (2022-11-10/1 [/megamat0_dsub3])
        # (all True for the other 3 final megamat flies)
        print('iscell_xid0 assertion fail!')
    '''

    # so i guess if cells were subset, they were renumbered (or more likely cells
    # were not subset at this point)
    cells_level_0 = fly_df.index.get_level_values('cells_level_0')
    assert pd.RangeIndex(cells_level_0.max() + 1).equals(cells_level_0)

    # don't think there's anything worth checking about 'embedding0'

    # TODO TODO do something w/ 'xid0' index level? as part of comparison wrt
    # response data (the stuff loaded in KC corr loading fn)?
    # TODO see if which xid0 cells have are consistent w/ those in responses i'm
    # loading elsewhere?
    #import ipdb; ipdb.set_trace()

    assert set(np.unique(fly_df.values)) == {0, 1}
    fly_df = fly_df.astype(bool)

    # TODO can i recreate these from the data Remy sent to anoop (what should
    # still be the final 4 megamat KC flies), which is in our Dropbox at:
    # Remy/odor_space_collab/for_mendy/data/megamat17
    # (using thresholds from CSV above)?

    # NOTE: there must be double counted cells. don't think Remy makes any attempt
    # to de-duplicate them.
    n_cells = len(fly_df)

    silent_cells = ~ (fly_df.any(axis='columns'))
    assert len(silent_cells) == n_cells
    n_silent = silent_cells.sum()

    parts = fly_dirname.split('__')
    assert len(parts) >= 3
    date_str, fly_str = parts[:2]

    date = pd.Timestamp(date_str)

    # e.g. 'fly01' -> 1
    fly_num = int(fly_str.replace('fly', ''))

    if _seen_date_fly_combos is not None:
        # we already know the ledger doesn't have duplicate flies (or 'acq' values),
        # from checks in main caller of this code.
        #
        # now we are just checking the individual files we are loading also don't ever
        # refer to the same fly in >1 of the files (could accidentally be 2 recordings
        # of same fly).
        assert (date, fly_num) not in _seen_date_fly_combos
        _seen_date_fly_combos.add( (date, fly_num) )

    if reset_index:
        fly_df = fly_df.reset_index(drop=True)
        # TODO what was it before? (maybe still 'cell', the reset_index() call seems to
        # clear this)
        # TODO still do this in else case?
        fly_df.index.name = 'cell'

    if acq_ledger is not None:
        curr_acq = acq_ledger.loc[(date, fly_num), remy_fly_id]
        fly_df = util.addlevel(fly_df, remy_fly_id, curr_acq)

        msg = f'{fly_dirname} ({remy_fly_id}={curr_acq})'
    else:
        msg = f'{fly_dirname}'

    if verbose:
        print(msg)

        # silent = responds to no odors
        #
        # NOTE: not sure if silent/not cells are equally represented (i.e. equally
        # double counted), so maybe fraction of silent cells is off?
        print(f'{n_silent} silent / {n_cells} cells ({(n_silent / n_cells):.3f})')
        print()

    return fly_df


_remy_megamat_kc_binary_responses = None
def load_remy_megamat_kc_binary_responses() -> pd.DataFrame:
    # TODO doc

    global _remy_megamat_kc_binary_responses
    if _remy_megamat_kc_binary_responses is not None:
        return _remy_megamat_kc_binary_responses

    # should be able to map acq to fly CSV dir using first 4 cols:
    # ['acq' (=remy_fly_id), 'date_imaged' (=remy_date_col), 'fly_num',
    # 'thorimage_name'] (last 2 cols not important)
    acq_ledger = pd.read_csv(remy_sparsity_dir / 'df_acqs.csv')
    acq_ledger[remy_date_col] = pd.to_datetime(acq_ledger[remy_date_col])

    acq_ledger = acq_ledger.set_index(remy_fly_cols, verify_integrity=True)
    assert not acq_ledger[remy_fly_id].duplicated().any()

    if verbose:
        print()
        print('loading Remy megamat KC binarized responses (for sparsity + S1C):')

    _seen_date_fly_combos = set()
    fly_binary_responses_list = []

    for fly_sparsity_path in sorted(
            remy_binary_response_dir.glob(f'*/{remy_fly_binary_response_fname}')
        ):

        fly_df = load_remy_fly_binary_responses(fly_sparsity_path, acq_ledger,
            _seen_date_fly_combos=_seen_date_fly_combos
        )
        fly_binary_responses_list.append(fly_df)

    binary_responses = pd.concat(fly_binary_responses_list, verify_integrity=True)
    binary_responses.columns.name = remy_odor_col

    binary_responses.columns = _strip_remy_concs(binary_responses.columns)
    assert binary_responses.shape[1] == len(megamat_odor_names)

    binary_responses = binary_responses.sort_index()

    assert _remy_megamat_kc_binary_responses is None
    _remy_megamat_kc_binary_responses = binary_responses

    return binary_responses


# TODO factor to hong2p.viz
# TODO simpler way?
def rotate_xticklabels(ax, rotation=90):
    for x in ax.get_xticklabels():
        x.set_rotation(rotation)


def remy_megamat_sparsity() -> float:
    """Returns mean response rate in Remy's final megamat KC data.

    Weights each fly (of 4) equally, regardless of number of cells per fly.
    """
    # md5 of this and
    # response_rates/old_megamat_sparsities/refstim__ep_at_-3.0__median__0.120-0.130
    # match, so they are the same.
    megamat_csv = remy_sparsity_dir / 'tidy_sparsities_ascending.csv'

    # columns: ['acq', 'stim', 'peak_amp_thresh', 'max_sparsity', 'min_sparsity',
    # 'kc_soma']
    #
    # 'kc_soma' should be between or equal to '[max|min]_sparsity' limits (which should
    # correspond to range of thresholds referenced in path name). 'kc_soma' is what Remy
    # said I should use.
    df = pd.read_csv(megamat_csv)
    sparsity_col = 'kc_soma'

    # TODO what are units/meaning of "0.120-0.130" in path? how does it relate to
    # 'peak_amp_thresh' col (min=0.95, max=1.75)?

    # TODO delete? only important column in this should match the corresponding (diff
    # named) column in other csv
    megamat_csv2 = remy_sparsity_dir / 'df_sparsity_recomputed.csv'
    df2 = pd.read_csv(megamat_csv2)
    sparsity_col2 = 'sparsity_from_peak_thr'

    assert df2[remy_odor_col].equals(df[remy_odor_col])
    assert df2[sparsity_col2].equals(df[sparsity_col])
    # df2[remy_fly_id] won't match. df2[remy_fly_id].unique() == [0] (through a
    # mistake on her end, Remy created this CSV w/ ID 0 for all the data in this CSV,
    # even though it actually has data from 4 flies. other CSV includes the correct IDs,
    # and rest of data matches)
    del df2

    assert not df.isna().any().any()
    assert df[remy_fly_id].nunique() == n_final_megamat_kc_flies
    assert set(x.stim.nunique() for _, x in df.groupby(remy_fly_id)) == {
        len(megamat_odor_names)
    }

    df[remy_odor_col] = _strip_remy_concs(df[remy_odor_col])

    # low->high response rate
    odors_in_sparsity_order = df.groupby(remy_odor_col, sort=False)[sparsity_col].mean(
        ).sort_values().index

    df = df.sort_values(remy_odor_col, kind='stable',
        key=lambda x: x.map(odors_in_sparsity_order.get_loc)
    )
    # TODO just force ep to be last? sort order actually diff (-> imply data is diff?)?
    # maybe just slightly though... (?)

    # TODO TODO TODO save this again + compare to model response rates
    # (or did i have another plot directly making that comparison?)
    plot = False
    if plot:
        fig, ax = plt.subplots()
        sns.pointplot(ax=ax, data=df, x=remy_odor_col, y=sparsity_col, hue=remy_fly_id)
        ax.set_xlabel('odor')
        ax.set_ylabel('response rate')
        ax.set_ylim([0, 0.2])
        rotate_xticklabels(ax, 90)
        # https://stackoverflow.com/questions/44620013
        ax.get_legend().set_title('fly')

        # should match preprint S1B
        fig, ax = plt.subplots()
        # should have the same 95% CI as reported in paper
        sns.pointplot(ax=ax, data=df, x=remy_odor_col, y=sparsity_col, color='black')
        ax.set_xlabel('odor')
        ax.set_ylabel('response rate')
        ax.set_ylim([0, 0.2])
        rotate_xticklabels(ax, 90)

        # TODO put mean_sparsity in xlabel, like in sparsity_per_odor plots
        # (silent cells too?) (delete comment? still care?)

        # TODO save plots? return figs?

    # TODO TODO recreate model KC response rates plot, on model tuned on hallem, but
    # subset to 17 megamat odors (using this code too, in case there was an issue w/
    # other code?) (done? still want?)

    mean_of_odor_means = df.set_index([remy_odor_col, remy_fly_id]).groupby(
        level=remy_odor_col, sort=False).mean()[sparsity_col].mean()

    mean_sparsity = df[sparsity_col].mean()
    # as expected from the math. numerically slightly (but not consequentially) diff
    assert np.isclose(mean_sparsity, mean_of_odor_means)

    binary_responses = load_remy_megamat_kc_binary_responses()
    assert set(binary_responses.columns) == set(df[remy_odor_col])

    recomputed_df = binary_responses.groupby(remy_fly_id).mean().melt(
        value_name=sparsity_col, ignore_index=False).reset_index()

    recomputed_df = recomputed_df.set_index([remy_fly_id, remy_odor_col],
        verify_integrity=True).sort_index()

    df = df.set_index([remy_fly_id, remy_odor_col], verify_integrity=True).sort_index()
    # changing index affects outcome, but only in irrelevant numerical way
    assert np.isclose(df[sparsity_col].mean(), mean_sparsity)

    for acq, recomputed_fly_df in recomputed_df.groupby(remy_fly_id):
        recomputed_ser = recomputed_fly_df.droplevel(remy_fly_id).iloc[:, 0]
        remy_fly_ser = df.loc[acq, sparsity_col]
        assert np.allclose(recomputed_ser, remy_fly_ser)

    # NOTE: can't just do binarized_responses.mean().mean(), as there are different
    # numbers of cells for different flies, so we need to average within each fly first
    # (to not weight some flies more than others)
    assert np.isclose(recomputed_df[sparsity_col].mean(), mean_sparsity)

    return mean_sparsity
