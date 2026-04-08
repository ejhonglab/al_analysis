
from copy import deepcopy
from collections import namedtuple
import difflib
from functools import wraps
import filecmp
import inspect
import itertools
import json
from math import factorial
import os
from os.path import getmtime
from pathlib import Path
import pickle
from pprint import pformat
import psutil
import sys
from tempfile import NamedTemporaryFile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.testing.exceptions import ImageComparisonFailure
# TODO replace w/ conditional import of seaborn / scipy as needed?
# if trying to run tests w/ valgrind, this import currently seems to raise:
# SystemError: initialization of beta_ufunc raised unreported exception
# from a from scipy.stats._boost.beta_ufunc import
# TODO can i assert we are not TYPE_CHECKING here, and then put the imports that would
# fail (for typing hinting commented below) behind that flag? and maybe define the
# types for type hinting conditional on PYTHONMALLOC and/or TYPE_CHECKING?
if os.getenv('PYTHONMALLOC') != 'malloc':
    import seaborn as sns
else:
    warnings.warn('could not import scipy (and dependent packages), because '
        'PYTHONMALLOC=malloc'
    )
from termcolor import cprint, colored

from hong2p import olf, util, viz
from hong2p.olf import format_mix_from_strs, solvent_str
from hong2p.roi import is_ijroi_named
from hong2p.util import pd_allclose
from hong2p.viz import dff_latex
from hong2p.types import Pathlike
import natmix


# TODO move to hong2p.types? already have something like that there?
ParamDict = Dict[str, Any]

bootstrap_seed: int = 1337

verbose: bool = False

fly_cols: List[str] = ['date', 'fly_num']
# typically, ROI = 'glomerulus' (mb_model.glomerulus_col)
# TODO move glomerulus_col here. could prob use in both al_analysis and mb_model
flyroi_cols: List[str] = fly_cols + ['roi']

diag_panel_str: str = 'glomeruli_diagnostics'

# TODO test
# TODO type hint
def sign_preserving_maxabs(x):
    """Returns value with maximum absolute value.
    """
    idxmax = np.abs(x).idxmax(axis=0)
    # TODO update w/ sam's version that should also work w/ >2D input?
    # not sure it also works for 2D input though... here's the code he used to make
    # dF/F images using [what should be an equivalent] calculation:
    # newshape = dff_response.shape[1:]
    # oldshape = [1] + [i for i in newshape]
    # maxabs_idxs = np.abs(dff_response).argmax(axis=0).reshape(oldshape)
    # maxabs_dff = np.take_along_axis(dff_response, maxabs_idxs, axis=0).reshape(newshape)
    # (and i think I'm fine leaving those images as computed with mean, instead of this
    # type of calculation)
    df = pd.DataFrame([
        x.loc[absmaxidx, idx] for idx, absmaxidx in zip(idxmax.index.values, idxmax)
    ])
    # TODO keep as pandas type (old stat=mean seemed to, as last bit of _checks=True
    # code in trial_response_traces seemed to assume Series types)?
    # TODO why .flatten? replace w/ .squeeze() at least, if that's the issue?
    return df.values.flatten()

response_stat_fn = sign_preserving_maxabs
# mean was what I had used for a while (also with n_volumes_for_response=2, I believe),
# including to generate Remy-paper outputs, and inputs to modelling for that.
#response_stat_fn = np.mean

if response_stat_fn.__name__ == 'mean':
    # TODO want to avoid just 'mean', to avoid confusion w/ 'mean ' prefixes that might
    # get prepended, once we start dealing with means over either trials or flies?
    # would get pretty verbose...
    # (currently, this just refers to mean over timepoints in a response window, used to
    # get one number for a given unit [e.g. glomerulus, pixel, etc] and trial)
    trial_stat_desc = 'mean'

# currently, np.max seems to have __name__ of amax, but checking both in case that isn't
# always true.
elif response_stat_fn.__name__ in ('max', 'amax'):
    trial_stat_desc = 'peak'

elif response_stat_fn.__name__ == 'sign_preserving_maxabs':
    trial_stat_desc = '±peak'

else:
    assert False, f'unrecognized {response_stat_fn.__name__=}'

# TODO TODO delete (output does not make that much sense. also already have per-panel
# trace zscoring imlemented now, from cached raw traces. output there also doesn't make
# much sense)
#
# NOTE: currently will warn + exit before running model, if this is True. assuming for
# now I probably don't want to run model with these inputs. (but can't currently tell if
# cached responses were computed that way...)
zscore_traces_per_recording: bool = False
#

if not zscore_traces_per_recording:
    response_desc: str = f'{trial_stat_desc} {dff_latex}'
else:
    # TODO TODO also mention the additional baseline subtraction step? add delta symbol?
    response_desc: str = f'{trial_stat_desc} Z-scored F'

# never saying "mean mean <x>", just specifying outer mean separately when
# trial_stat_desc == 'max'
if trial_stat_desc != 'mean':
    # going to make no effort to clarify whether it's a mean over trials or flies.
    # should be clear from plot, and will get too verbose + create too many variables
    mean_response_desc: str = f'mean {response_desc}'
else:
    mean_response_desc: str = response_desc

response_calc_params_json_name: str = 'response_calc_params.json'


# TODO adapt -> share w/ (at least) drop_redone_odors?
# TODO type hint Mapping? can it be Series or Dict?
def format_panel(x) -> str:
    panel = x.get('panel')
    # TODO maybe return None if it'd make more consistent vline_level_fn usage
    # possible (across single/multi panel cases)? would prob need to handle in viz
    # regardless
    if not panel:
        return ''

    if panel == diag_panel_str:
        # Less jarring than 'glomeruli_diagnostics'
        return 'diagnostics'

    assert type(panel) is str
    return panel


def roi_label(index_dict) -> str:
    roi = index_dict['roi']
    if is_ijroi_named(roi):
        return roi
    # Don't want numbered ROIs to be grouped together in plots, as the numbers
    # aren't meanginful across flies.
    return ''


# TODO rename now that i'm also allowing input w/o date/fly_num attributes?
def fly_roi_id(row: pd.Series, *, fly_only: bool = False) -> str:
    """
    Args:
        fly_only: if False, will include date, fly, and ROI information.
            If True, will exclude ROI information, so long as the ROI was given a name
            (and not autonamed/numbered).
    """
    # NOTE: assuming no need to use row.thorimage_id (or a panel / something like
    # that), as assumed this will only be used within a context where that is
    # context (e.g. a plot w/ kiwi data, but no control data)
    try:
        parts = []

        if hasattr(row, 'fly_id') and pd.notnull(row.fly_id):
            fly_id = row.fly_id
            assert type(fly_id) is str
            parts.append(fly_id)

        # TODO option to have these (or fly_id) parenthetical, and still show all 3
        # vars?
        else:
            if hasattr(row, 'date') and pd.notnull(row.date):
                date_str = f'{row.date:%-m-%d}'
                parts.append(date_str)

            if hasattr(row, 'fly_num') and pd.notnull(row.fly_num):
                fly_num_str = f'{row.fly_num:0.0f}'
                parts.append(fly_num_str)

        # TODO also support a 'fly'/'fly_id' key in place of (date, fly[_num])?
        # (for lettered/sequential simplified IDs, for nicer plots)

        roi = row.roi
        if not is_ijroi_named(roi):
            fly_only = False

        if not fly_only:
            parts.append(str(roi))

        # fly_only=True for when [h|v]line_[level_fn+group_text] code is drawing ROI
        # labels
        return '/'.join(parts)

    except AttributeError:
        assert not fly_only, "no fly ID vars ('fly_id' or ['date', 'fly_num'])"
        return f'{row.roi}'


# TODO refactor inches_per_cell (+extra_figsize) to share w/ plot_roi_util?
# just move into plot_all_... default? (would need extra_figsize[1] == 1.0 to work here)
# TODO rename to clarify these aren't for plot_rois calls...
roi_plot_kws = dict(
    inches_per_cell=0.08,
    # TODO adjust based on whether we have extra text labels / title / etc?
    # 1.7 better for the single panel plots, 2.0 needed for ijrois_certain.png, etc
    extra_figsize=(2.0, 0.0),

    fontsize=4.5,

    linewidth=0.5,
    # TODO just set this in rcParams (or patching them at module level up top?)?
    dpi=300,

    # controls spacing from glomerulus names and <date>/<fly_num> IDs in yticklabels
    # (only relevant for plots w/ data from multiple flies).
    # default 0.12 was also OK here, just a bit much.
    #
    # was using this before trying to find a value for kiwi/control matrices
    #hgroup_label_offset=0.095,
    #
    # TODO TODO can one value work for both megamat/validation/etc (w/ many odors), and
    # just kiwi/control (when shown w/o diags)? or define dynamically based on number of
    # odors? layout method that is agnostic?
    # TODO this even doing anything? .15 was not clearly diff from .095
    hgroup_label_offset=0.5,

    # controls spacing between odor xticklabels and panel strings above them (for matrix
    # plots where we have data from multiple panels)
    #
    # though 0.08 works for roimean_plot_kws below (where flies are averaged over, and
    # thus less rows), since figure is so tall when there is a row per fly X glomerulus,
    # i think both axes and data coordinates have the problem of scaling with the height
    # of the figure, so this spacing would lead to a silly gap
    #
    # good for ijroi/certain.pdf, but didn't check much else
    # TODO also check ijroi/by_panel/megamat/with-diags_certain.pdf (may need to use
    # something like figure, rather than axes, coords; or special case the value for
    # diff plots w/ diff heights)
    vgroup_label_offset=0.0145,

    # TODO can the following values (from load_antennal_csv.py) also work here?
    # i assume not, at least not for hgroup_label_offset.
    #hgroup_label_offset=0.12,
    #vgroup_label_offset=0.03,

    # TODO define separate ones for colorbar + title/ylabel (+ check colorbar one is
    # working)
    bigtext_fontsize_scaler=1.5,

    cbar_label=mean_response_desc,

    odor_sort=False,
    # TODO need to prevent hline_level_fn from triggering in case where we already
    # only have one row per roi?
    #  'roi' should always be a level
    hline_level_fn=roi_label,
    vline_level_fn=format_panel,
    # TODO try to delete levels_from_labels (switching to only == False case),
    # (inside viz)
    levels_from_labels=False,
)

roimean_plot_kws = dict(roi_plot_kws)
roimean_plot_kws['inches_per_cell'] = 0.15
roimean_plot_kws['extra_figsize'] = (1.0, 0.0)
roimean_plot_kws['vgroup_label_offset'] = 0.1


# TODO delete? sort_odors below not do what i wanted in some pair data stuff?
def sort_concs(df: pd.DataFrame) -> pd.DataFrame:
    return olf.sort_odors(df, sort_names=False)


# TODO flag to select whether ROI or (date, fly) take priority?
# TODO move to hong2p + test
def sort_fly_roi_cols(df: pd.DataFrame, flies_first: bool = False, sort_first_on=None
    ) -> pd.DataFrame:
    # TODO delete key if i can do w/o it (by always just sorting a second time when i
    # want some outer level)
    # TODO is doc for sort_first_on right (or is it maybe just describing one use case
    # for it?)?
    """Sorts column MultiIndex with `flyroi_cols` levels.

    Args:
        df: data to sort by fly/ROI column values

        flies_first: if True, sorts on `fly_cols` columns primarily, followed by 'roi'
            ROI names.

        sort_first_on: sequence of same length as df.columns, used to order ROIs.
            Within each level of this key, will sort on the default date/fly_num ->
            with higher priority than roi, but will then group all "named" ROIs before
            all numbered/autonamed ROIs.
    """
    index_names = df.columns.names
    # TODO replace w/ def from flyroi_cols? or just also assert it's in flyroi_cols?
    assert 'roi' in index_names or 'roi' == df.columns.name

    levels = ['not_named', 'roi']
    # TODO replace date/fly_num usage w/ all fly_cols
    if 'date' in index_names and 'fly_num' in index_names:
        if not flies_first:
            levels = levels + fly_cols
        else:
            levels = fly_cols + levels

    levels_to_drop = []
    to_concat = [df.columns.to_frame(index=False)]

    assert 'not_named' not in df.columns.names
    # TODO option to do certain instead of named?
    not_named = df.columns.get_level_values('roi').map(
        lambda x: not is_ijroi_named(x)).to_frame(index=False, name='not_named')

    levels_to_drop.append('not_named')
    to_concat.append(not_named)

    if sort_first_on is not None:
        # NOTE: for now, just gonna support this being of-same-length as df.columns

        # TODO delete try/except
        # triggered when trying to adapt each_fly diag resp matrix code to across fly
        # case
        try:
            assert len(sort_first_on) == len(df.columns)
        except AssertionError:
            print(f'{sort_first_on=}')
            print(f'{df.columns=}')
            print(f'{len(sort_first_on)=}')
            print(f'{len(df.columns)=}')
            import ipdb; ipdb.set_trace()

        # Seems to also work when input is a list of tuples (so you can list(zip(...))
        # multiple iterables of keys, in the order you want them to take priority).
        sort_first_on = pd.Series(list(sort_first_on), name='_sort_first_on').to_frame()

        levels = ['_sort_first_on'] + levels
        levels_to_drop.append('_sort_first_on')
        to_concat.append(sort_first_on)

    df.columns = pd.MultiIndex.from_frame(pd.concat(to_concat, axis='columns'))

    # TODO get numbers to actually sort like numbers, for any numbered ROIs
    # (or maybe just set name to NaN there, just for the sort, and rely on them already
    # being in order?) (would probably have to sort the numbered section separately from
    # the named one, casting the numbered ROI names to ints)

    # the order of level here determines the sort-priority of each level.
    sorted_df = df.sort_index(level=levels, sort_remaining=False, kind='stable',
        axis='columns').droplevel(levels_to_drop, axis='columns')

    # index_names were column names at input
    assert sorted_df.columns.names == index_names

    return sorted_df


# TODO replace this + use of warnings.warn w/ logging.warning (w/ logger probably
# currently just configured to output to stdout/stderr)
def formatwarning_msg_only(msg, category, *args, **kwargs):
    """Format warning without line/lineno (which are often not the relevant line)
    """
    warn_type = category.__name__ if category.__name__ != 'UserWarning' else 'Warning'
    return colored(f'{warn_type}: {msg}\n', 'yellow')

# TODO do just in main? (/ after some call?)
warnings.formatwarning = formatwarning_msg_only


def in_pytest() -> bool:
    """Returns whether code is currently being executed by pytest.
    """
    # https://docs.pytest.org/en/latest/example/simple.html#pytest-current-test-environment-variable
    return 'PYTEST_CURRENT_TEST' in os.environ


def is_pytest_capturing_stderr() -> bool:
    """Returns whether pytest is capturing stderr (as can be disabled by -s).

    There are probably other ways to disable capture other than `-s`, which disables all
    capturing (stdout as well). This method will probably also work in that context.
    """
    # TODO does pytest directly expose it's capturing method in and env var or
    # something? use that instead, if so?
    # TODO maybe it also exposes whether it will show capture at end? use that too?

    # from my testing (w/ pytest 8.3.5 and python 3.8.12), it seems .name on sys.stderr
    # can distinguish whether pytest was called with `-s` or not.
    #
    # with -s (--capture=no), it's: '<stderr>'
    #
    # without -s, it's like: "<_io.FileIO name=8 mode='rb+' closefd=True>"
    stderr = sys.stderr
    # should work for --capture=fd  (default), =no (-s), and =sys , but get:
    # `AttributeError: '_io.BytesIO' object has no attribute 'name'` if using
    # --capture=sys-tee
    if hasattr(stderr, 'name'):
        return stderr.name != '<stderr>'

    # assuming we are in --capture=sys-tee or --capture=sys case here
    # (both currently hit this code path on my system)
    #
    # (not sure --capture=sys-tee works w/ rest of my system here... i.e. that prints
    # and warnings are still interleaved as i want [and ideally printed in real time],
    # and that debugger interface can still be used normally. sys-tee seems like it
    # might interfere w/ debugger being used normally, tho it does also print in real
    # time [in addition to capturing])
    #
    # TODO TODO need a test case w/ some interleaved prints and warnings before a
    # breakpoint, then try all the capturing methods and see which are available
    return True


# TODO maybe log all warnings?
def warn(msg) -> None:
    color = 'yellow'

    # since i couldn't otherwise figure out how to get pytest to show warnings before
    # debugger breakpoint, especially if pytest is configured to ignore warnings, so now
    # we'll just also print it before pytest cleanup.
    # TODO check pytest config options again for something to disable warning-exclusive
    # capturing / formatting / output control?
    #
    # if pytest is capturing stderr, we probably aren't debugging, and thus shouldn't
    # need these extra prints.
    # TODO once per run, warn we are changing warning behavior based on this?
    if in_pytest() and not is_pytest_capturing_stderr():
        print(colored(str(msg), color))

    # TODO replace w/ logging.warning? (have init_logger just hook into warnings.warn?
    # some standard mechanism for that?)
    # TODO how to get color to work here too? possible?
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


# TODO doc what this does in a comment (are any outputs actually written, or all just
# written to temp paths and compared to any existing current outputs at same paths?)
# True|False|'nonmain'
check_outputs_unchanged = False
# hack so al_analysis can edit this to add functions i currently have defined under main
# (e.g. save_method_csvs)
# TODO just for -c/-C right? explain better the need for this
_consider_as_main = []

# TODO TODO move to hong2p.util
# TODO unit test?
#
# TODO default verbose=None and try to use default of wrapped fn then
# (or True otherwise?)
# (still need to test behavior when wrapped fn has existing verbose kwarg)
_fn_name2seen_inputs: Dict[str, Path] = dict()
_fn_name2wrapped_fn: Dict[str, Callable] = dict()
# TODO also include savefig plots in these (partially to make replacing custom logic in
# there w/ wrapping it w/ @produces_output easier later)
_all_seen_inputs: Set[Path] = set()
CodeContext = namedtuple('CodeContext', 'filename lineno fn_name')
_saved_path2last_save_code_context: Dict[Path, CodeContext] = dict()
# TODO what is _fn for again? keep?
def produces_output(_fn=None, *, verbose=True):
    # for how to make a decorator with optional  arg:
    # https://realpython.com/primer-on-python-decorators

    # TODO what would be a good name for this?
    def wrapper_helper(fn):
        assert fn.__name__ not in _fn_name2seen_inputs, (
            f'{fn.__name__=} seen set would have been overwritten'
        )
        # TODO some reason to use lists like i was in savefig? was that just for easier
        # use in multiprocessing access (no set equiv of IPC data type?)?
        # that matter anymore?
        _fn_name2seen_inputs[fn.__name__] = set()

        @wraps(fn)
        # TODO delete *args (if assertion it's unused passes for a wihle)
        def wrapped_fn(data, path: Pathlike, *args, verbose: bool = verbose,
            ignore_output_change_check: Union[bool, str] = False,
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

            assert ignore_output_change_check in (True, False, 'warn')

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

            assert fn.__name__ in _fn_name2seen_inputs
            # TODO probably don't want different fns to be able to save to same path
            # either tho... (not that they currently would). maybe seen_inputs should be
            # one global?
            seen_inputs = _fn_name2seen_inputs[fn.__name__]

            if not multiple_saves_per_run_ok and (
                (normalized_path in seen_inputs or normalized_path in _all_seen_inputs)
                ):
                context = _saved_path2last_save_code_context[normalized_path]
                raise MultipleSavesPerRunException('would have overwritten output '
                    f'{path}\npreviously written elsewhere in this run, at:\n'
                    f'{context.filename}, line {context.lineno} (in {context.fn_name})'
                    '\nadd multiple_saves_per_run_ok=True to call to override, but this'
                    ' is likely a mistake'
                )

            seen_inputs.add(normalized_path)
            # TODO why did i need a fn specific cache anyway? paths should be unique
            # across all fns anyway, right? i don't want two fns overwriting each others
            # outputs within a run... simplify by replacing all fn specific sets w/ this
            # one global one?
            _all_seen_inputs.add(normalized_path)

            last_frame = inspect.currentframe().f_back
            # "index" (last argument) is "the index of the current line being executed
            # in the code_context list", but not sure what that means, and don't seem to
            # need it for what i want
            filename, lineno, fn_name, _, _ = inspect.getframeinfo(last_frame)
            _saved_path2last_save_code_context[normalized_path] = CodeContext(
                filename=filename, lineno=lineno, fn_name=fn_name
            )

            write_output = True
            if check_outputs_unchanged and path.exists():
                try:
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

                    # TODO add kwarg to only ignore if diff matches a certain value?
                    _warn_only = False
                    if ignore_output_change_check and check_outputs_unchanged != False:
                        _warn_only = True

                    # 'warn'|False will still enter this conditional
                    if ignore_output_change_check != True:
                        write_output = _output_change_prompt_or_err(err, path,
                            _warn_only=_warn_only
                        )

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

        _fn_name2wrapped_fn[fn.__name__] = wrapped_fn
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

def _output_change_prompt_or_err(err: RuntimeError, path: Path, *,
    _warn_only: bool = False) -> bool:
    """Returns whether `path` should be written to with new version.

    If `al_util.prompt_if_changed = True`, prompts user to choose what to do with file
    (whether to overwrite/skip/exit). If `al_util.prompt_if_changed = False`, raises
    `err` passed in.
    """
    assert err is not None
    if not prompt_if_changed and not _warn_only:
        raise err

    assert len(err.args) == 1
    msg = err.args[0]
    # TODO still show lineno (of fn calling wrapped_fn) in this case?
    warn(msg)
    if _warn_only:
        return True

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
    assert path.suffix == '.csv', f"{path.suffix} should be '.csv'"
    data.to_csv(path, **kwargs)


_SERIES_NAME_PLACEHOLDER: str = '__UNNAMED-SERIES'
def read_parquet(path: Path, *, squeeze: bool = True) -> Union[pd.DataFrame, pd.Series]:
    # TODO try changing default engine (='fastparquet'? or ='pyarrow'? latter should be
    # default, if both installed [at least in pandas 3.0...])? that fix any of column
    # level type handling i add below?
    data = pd.read_parquet(path)

    # TODO need an outer need len check? prob not practically, but maybe for some edge
    # cases, if trying to support saving empty stuff?
    # TODO refactor to share w/ similar code in to_parquet?
    move_first_column_to_index = False
    i00 = data.iloc[0, 0]
    # TODO also exclude anything that isn't explicitly a sequence of float/int?
    if not isinstance(i00, str):
        try:
            len(i00)
            move_first_column_to_index = True
        except TypeError as err:
            assert str(err).endswith('has no len()')
    #

    # TODO assert no sequence values in index already? (would not be possible here, b/c
    # read_parquet above would fail. may want something like that in to_parquet)

    if move_first_column_to_index:
        assert len(data.columns) > 1, \
            'still need some columns left after moving one to index'

        c0 = data.columns[0]
        # we need to make sequences hashable to move to index. tuple is the way i had
        # done this with bouton_ids, and the most straightforward.
        data[c0] = data[c0].map(tuple)
        data = data.set_index(c0, append=True)

    # TODO do for non-MultIndex indices as well? (only [currently] converting levels on
    # save for MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        index_arrays = []
        for c in data.columns.names:
            xs = data.columns.get_level_values(c)
            # TODO anything other types besides numeric/datetime?
            try:
                # seems to automatically convert to int if it can (haven't tested float,
                # especially not with NaN, but assume that would remain float too)
                xs = pd.to_numeric(xs)

            # ValueError: Unable to parse string <x> at position <n>
            except ValueError:
                try:
                    xs = pd.to_datetime(xs)
                except ValueError:
                    pass

            # each will still have .name defined
            index_arrays.append(xs)

        data.columns = pd.MultiIndex.from_arrays(index_arrays)


    # all Series written with to_parquet above (if they had None for .name), should have
    # this as last column
    if data.columns[-1] == _SERIES_NAME_PLACEHOLDER:
        data = data.squeeze().rename(None)

    # TODO remove kwarg? always squeeze? would then prob want to assert no DataFrame
    # inputs with only one column (which there probably aren't...)
    elif len(data.columns) == 1 and squeeze:
        return data.squeeze()

    return data


@produces_output(verbose=True)
def to_parquet(data: Union[pd.DataFrame, pd.Series], path: Path, *, check: bool = True
    ) -> None:
    """Write `data` to parquet at `path`, with default check loaded value matches input.
    """
    assert path.suffix == '.parquet', f"{path.suffix} should be '.parquet'"
    # TODO make check=False the default eventually?
    if check:
        orig = data

    if isinstance(data, pd.Series):
        assert not hasattr(data, 'to_parquet')
        # since DataFrame.to_parquet(...) requires str column names, but we often have
        # Series with .name == None.
        # Not planning to support non-str for column names other than just a single
        # missing Series name.
        if data.name is None:
            data = data.to_frame(name=_SERIES_NAME_PLACEHOLDER)
        else:
            data = data.to_frame()

    # TODO remove outer len check?
    if len(data) > 0:
        i00 = data.iloc[0, 0]
        # TODO also exclude anything that isn't explicitly a sequence of float/int?
        if not isinstance(i00, str):
            try:
                len(i00)
                raise ValueError('data already had sequence elements in first column. '
                    'would be inferred as a final tuple-of-int index level with current'
                    ' scheme'
                )
            except TypeError as err:
                assert str(err).endswith('has no len()')

    # pd.MultiIndex objects have levels. non-MultiIndex pd.Index objects do NOT.
    if hasattr(data.columns, 'levels'):
        str_levels = [all(type(x) is str for x in xs) for xs in data.columns.levels]

        # TODO some way to avoid the copy here? (without changing input)
        data = data.copy()

        # TODO assert all non-str are all int/float? (basically, some list of types we
        # should be able to detect and convert back on read)
        # TODO try to support datetime/Timestamp here too? anything else?
        data.columns = data.columns.map(lambda xs:
            tuple(x if was_str else str(x) for x, was_str in zip(xs, str_levels))
        )
        # TODO  assert no already-str levels would be detected convertible to int/float
        # (/whatever other types we are converting in read) here?
    else:
        # NOTE: currently assuming any non-MultIndex input will already just have str
        # columns (could implement without too much effort)
        assert all(type(x) is str for x in data.columns)

    # TODO how would this interact with MultiIndex columns? assert we don't have both?
    # test some cases there?
    #
    # currently only planning on supporting automatic tuple-of-int (index) <->
    # array-of-int (column, as supported by my current pd.read_parquet) conversion for a
    # final index level (e.g. bouton_id, which I'm currently assuming will always be
    # last level, when present).
    #
    # I assume that, for anything more complicated than one level at the end of index,
    # the easiest thing to do will be to add an argument to allow caller to specify
    # which columns to convert (and probably need another one for column order, at least
    # so long as I'm doing automated round trip checks, since how else will we know what
    # order to put columns back into index in?)
    if type(data.index.get_level_values(-1)[0]) is tuple:
        # moving last index level to first column
        data = data.reset_index(level=-1)

    # TODO accept engine argument here? changing engine actually avoid any of my current
    # need for custom conversions in to_parquet/read_parquet?
    data.to_parquet(path)

    if check:
        d2 = read_parquet(path)
        try:
            assert d2.equals(orig)
        # TODO TODO was i doing this was just b/c of need for isclose in index? why can
        # we assert values are equal and not just close then?
        except AssertionError:
            # TODO warn here?

            # TODO also convert indices to frames and check those w/ allclose (like
            # columns)? or is parquet MultiIndex reading not broken for those? i had to
            # add manual type conversions for MultiIndex column level values in my
            # read_parquet fn.
            assert d2.index.equals(orig.index)

            c1 = orig.columns.to_frame(index=False)
            c2 = d2.columns.to_frame(index=False)
            # TODO handle case where date is datetime64[ns] in one, and object in the
            # other. (now that i added datetime column level handling in read_parquet,
            # should be fine. delete)
            # TODO is it just float levels that are allclose that are causing failure
            # for claws_sims_[sums|maxs]? (seems so, yes)
            assert pd_allclose(c2, c1)

            # TODO TODO fix. failing w/ consensus_df saving w/:
            # ./al_analysis.py -d pebbled -n 6f -t 2023-04-22 -e 2024-01-05 -s corr,intensity,ijroi,model-seeds,model-sensitivity -v -i model -M
            # (equal_nan=True here seems to fix it, but pd_allclose(d2, orig,
            # equal_nan=True) does *not* work? how come?)
            try:
                assert np.array_equal(d2.values, orig.values, equal_nan=True)
            except AssertionError:
                breakpoint()


@produces_output(verbose=False)
# input could be at least Series|DataFrame
# TODO delete write_parquet kwarg eventually? after adding explicit calls where i want
def to_pickle(data, path: Path, *, write_parquet: bool = True) -> None:
    """Writes input to pickle at `path`.

    Args:
        write_parquet: if True, will also write any `pd.Series|DataFrame` data to
            `path.with_suffix('.parquet')` (i.e. '<x>/<y>.parquet', for input
            '<x>/<y>.p').

    NOTE: `produces_output` wrapper modifies fn to allow `Pathlike` for path arg
    """
    # just checking we aren't accidentally calling to_pickle with a path we intend for
    # one of the other calls
    assert path.suffix not in ('.csv', '.parquet', '.nc', '.json')

    if isinstance(data, xr.DataArray):
        path = Path(path)
        # read via: pickle.loads(path.read_bytes())
        # (note lack of need to specify protocol)
        # just specifying protocol b/c docs say it is (sometimes?) much faster
        path.write_bytes(pickle.dumps(data, protocol=-1))
        return

    # TODO delete eventually (replace calls [that i can] of to_pickle w/ to_parquet
    # first)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if write_parquet:
            # replacing .p w/ .parquet
            parquet_path = path.with_suffix('.parquet')
            # TODO also include line of calling code? easily possible?
            warn(f'also saving to {parquet_path} via to_parquet call inside to_pickle. '
                'replace with explicit to_parquet call in the future!'
            )
            to_parquet(data, parquet_path)
    else:
        # write_parquet=False must have been manually specified, which indicates i
        # thought that input was a DataFrame/Series that would have been written as a
        # parquet file otherwise, which is incorrect if we are in this `else`
        assert write_parquet, (f'{type(data)=} was not the DataFrame/Series you '
            'seemed to expect, by setting explicit write_parquet=False'
        )
    #

    if hasattr(data, 'to_pickle'):
        # TODO maybe do this if instance DataFrame/Series, but otherwise fall back to
        # something like generic read_pickle?
        data.to_pickle(path)

    path.write_bytes(pickle.dumps(data))


# TODO move to hong2p?
def read_pickle(path: Pathlike):
    path = Path(path)
    return pickle.loads(path.read_bytes())


def read_json(path: Path) -> ParamDict:
    """Returns dict contained within JSON at `path`
    """
    with open(path, 'r') as f:
        return json.load(f)


@produces_output
def to_json(param_dict: ParamDict, path: Path, *, check: bool = True) -> None:
    """Write `param_dict` to JSON at `path`, with default round trip check.
    """
    assert path.suffix == '.json', f"{path.suffix} should be '.json'"
    with open(path, 'w') as f:
        json.dump(param_dict, f, indent=4)

    if check:
        round_trip = read_json(path)
        assert param_dict == round_trip, f'{param_dict=}\n...!=...\n{round_trip=}'


@produces_output(verbose=False)
def np_save(data: np.ndarray, path: Path, **kwargs) -> None:
    # TODO add check=True option to round trip test load the data?
    """
    NOTE: opposite order of args to `np.save`, which has path first and data second.
    necessary to work w/ my `produces_output` wrapper.
    """
    np.save(path, data, **kwargs)


# TODO maybe don't require is_pair? panel?
# TODO rename?
#
# can have more 'odor<x>' row index levels if there are multiple odors presented at once
# (via air mixing). will have a number of levels equal to maximum presented at once.
required_index_levels: List[str] = ['panel', 'is_pair', 'odor1', 'repeat']

# TODO provide fn to invert zero filling i had done for some new outputs (dropping
# glomeruli w/ all 0s or NaN)

def drop_old_odor_index_levels(df: pd.DataFrame, *, warn_: bool = True
    ) -> pd.DataFrame:
    # TODO doc
    # for dropping metadata intended for binary mixture experiments
    to_drop = []

    if 'is_pair' in df.index.names:
        if set(df.index.get_level_values('is_pair')) == {False}:
            to_drop.append('is_pair')
        else:
            if warn_:
                warn('index had some is_pair=True entries! not dropping is_pair level!')

    if 'odor2' in df.index.names:
        if set(df.index.get_level_values('odor2')) == {solvent_str}:
            to_drop.append('odor2')
        else:
            if warn_:
                warn(f'index had some odor2 != {solvent_str} entries! not dropping '
                    'odor2 level!'
                )

    if len(to_drop) > 0:
        df.index = df.index.droplevel(to_drop)

    return df


# TODO rename to include dff in name? or antennal? it's not just the counterpart to my
# wrapped to_csv fn, which is intended to writing general CSVs, not just this specific
# format of them
# TODO TODO tho should i test this fn can read CSVs written by my to_csv fn?
def read_csv(csv: Pathlike, *, drop_old_odor_levels: bool = True,
    check_vs_pickle: bool = True, warn_: bool = True, verbose: bool = True
    ) -> pd.DataFrame:
    """
    Args:
        warn_: passed to `drop_old_odor_index_levels`, if called
    """
    # TODO doc output format (w/ example str repr)
    # TODO does this work on both ij_certain-roi_stats.csv and ij_roi_stats.csv outputs?

    csv = Path(csv)
    assert csv.exists(), f'CSV {csv} did not exist!'

    if verbose:
        print(f'loading {csv}')
        print(f'modified {format_mtime(getmtime(csv), year=True)}')
        print()

    # this line will start with the level names for the row index, e.g.
    # ['panel', 'is_pair', 'odor1', 'repeat', nan, ...]
    # (for data with a maximum of one odor component mixed-in-air)
    #
    # if there are additional components, there will be additional 'odor<N>' index
    # levels, up to the maximum # of components presented at once (via air mixing. odors
    # mixed in vial are represented by their own unique name, not via separate levels).
    # most I've ever used (via air mixing) is 2.
    #
    # not sure if there might be data after these level names, or if it will always be
    # NaN for everything else in that row (but doesn't matter for this).
    index_row = pd.read_csv(csv, skiprows=3, nrows=1, header=None).squeeze()
    # 'repeat' should always be the last level, so if we find that, we know we only have
    # to read index_col up to there (in next read_csv call)
    eq_repeat = index_row == 'repeat'
    assert eq_repeat.sum() == 1
    # index starts at 0, so we will need to read 1 more index_col level past this
    repeat_idx = eq_repeat.idxmax()
    n_index_col_levels = repeat_idx + 1

    df = pd.read_csv(csv,
        # can't pass list of names here when specifying column MultiIndex via header.
        # trying raises ValueError to that effect.
        index_col=list(range(n_index_col_levels)),

        # doesn't take list of str, and names= seems to only set column level names,
        # rather than finding them by name.
        header=list(range(len(flyroi_cols)))
    )
    # this is assumed in section above that determines how many index levels there are
    assert df.index.names[-1] == 'repeat'

    assert df.columns.names == flyroi_cols
    assert set(required_index_levels) - set(df.index.names) == set()

    # the only row index levels not in required_index_levels should be extra air-mix
    # components (with names in 'odor<N>' form)
    prefix = 'odor'
    for x in set(df.index.names) - set(required_index_levels):
        assert x.startswith(prefix)
        try:
            component_num = int(x[len(prefix):])
        # only intending to catch int parsing failure like:
        # ValueError: invalid literal for int() with base 10: ...
        except ValueError:
            # TODO include better message about malformed index level name
            raise

        # TODO assert all contiguous?
        assert component_num >= 1

    # TODO refactor?
    assert df.columns.names[0] == fly_cols[0]
    df.columns = df.columns.set_levels(pd.to_datetime(df.columns.levels[0]),
        level=0, verify_integrity=True
    )

    assert df.columns.names[1] == fly_cols[1]
    df.columns = df.columns.set_levels(df.columns.levels[1].astype(int), level=1,
        verify_integrity=True
    )

    if check_vs_pickle:
        # just some checking i was doing against a parallel pickle version i had, mainly
        # to make sure i was loading CSV correctly (with same dtype info)
        pickle_path = csv.with_suffix('.p')
        if pickle_path.exists():
            pdf = pd.read_pickle(pickle_path)
            assert pd_allclose(df, pdf, equal_nan=True)
            if verbose:
                print(f'CSV data matches pickle {pickle_path}\n')
        else:
            if verbose:
                print(f'no pickle at {pickle_path}. could not check against CSV.\n')

    if drop_old_odor_levels:
        df = drop_old_odor_index_levels(df, warn_=warn_)

    return df


# TODO unit test this?
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
def _check_output_would_not_change(path: Path, save_fn: Callable,
    data: Optional[Any] = None, **kwargs) -> None:
    """Raises RuntimeError if output would change.

    Args:
        path: must already exist (raises IOError if not)

        data: if passed, will be 1st `save_fn` argument, otherwise file path is 1st
            argument, and it's assumed that the data to save is in `kwargs`

        **kwargs: passed to `save_fn`
    """
    # TODO also raise if not file? don't think anything below is equipped to handle e.g.
    # a directory
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

            # TODO TODO refactor to use my hong2p.util.equals now?
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
        # TODO need to install something else for this, in a new env?
        # (might have manually installed before in current one...)
        # (and not sure i've tested this path in a fresh install, since import is
        # conditional)
        # TODO TODO add unit test that hits this path
        # TODO when factoring out file comparison fn, def use_mpl_comparison from
        # conditional import for this b/c it can hang sometimes (for unclear reasons).
        # see comment by original import up top (was there up to f1b56dc4). not sure if
        # the hanging will ever come up again, or how to repro.
        from matplotlib.testing import compare as mpl_compare

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

        # bit more flexible than Path.with_suffix (how so? just extra '.' chars?
        # delete?)
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


# TODO move up top?
# TODO clarify how these behave if something is missing (in comment)
panel2name_order = deepcopy(natmix.panel2name_order)
panel_order = list(natmix.panel_order)

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
# TODO TODO how to get this to not sort control panel '2h @ -5 + oct @ -3' (air
# mix, using odor2 level) right after '2h @ -5'? (and same for kiwi)
def sort_odors(df: pd.DataFrame, add_panel: Optional[str] = None, **kwargs
    ) -> pd.DataFrame:

    # TODO TODO check in here whether panel(s) are in panel2name_order (rather than
    # requiring that check many other places)

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
_dirs_to_delete_if_empty: List[Path] = []
# TODO add test confirming it also makes parents (i think that was the whole point of
# using os.makedirs)
def makedirs(path: Pathlike) -> Path:
    """Make directory if it does not exist, and register for deletion if empty.
    """
    path = Path(path)
    # TODO make sure if we make a directory as well as some of its parent directories,
    # that if (all) the leaf dirs are empty, the whole empty tree gets deleted
    # TODO shortcircuit to returning if we already made it this run, to avoid the checks
    # on subsequent calls? they probably aren't a big deal though...
    os.makedirs(path, exist_ok=True)
    # TODO only do this if we actually made the directory in the above call?
    # not sure we really care for empty dirs in any circumstances tho...
    _dirs_to_delete_if_empty.append(path)
    return path


save_figs = True

# TODO support multiple (don't know if i want this to be default though, cause extra
# time)
# TODO add CLI arg to override this?
plot_fmt: str = os.environ.get('plot_fmt', 'pdf')

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
# TODO try to restore type hint w/ seaborn type. commenting for now to try to be able to
# not import sns, to use some extra C++ debugging tools that it (and scipy) are
# incompatible with
#def savefig(fig_or_seaborngrid: Union[Figure, Type[sns.axisgrid.Grid]],
def savefig(fig_or_seaborngrid,
    fig_dir: Pathlike, desc: str, *, close: bool = True, normalize_fname: bool = True,
    debug: bool = False, multiple_saves_per_run_ok: bool = False,
    fmt: Optional[str] = None, **kwargs) -> Path:
    """
    Args:
        fmt: if None, defaults to `al_util.plot_fmt` (typically 'pdf')
    """
    # TODO doc what this is set by (al_analysis CLI args i assume?)
    global exit_after_saving_fig_containing

    if fmt is None:
        fmt = plot_fmt

    if fmt == 'pdf':
        kwargs['metadata'] = {
            # TODO delete? test whether this actually improves reproducibility of output
            # files? other blockers on that?
            # TODO anything similar i want for svg, now that i'm also trying that?
            # even needed in current mpl? not referenced in current docs, and not
            # obviously in settings current mpl testing code enforces for tests
            'creationDate': None,

            # TODO some other way to specify (doesn't seem so)? delete
            # matplotlib doesn't seem to set this by default (grepping an example PDF
            # does contain CreationDate, but nothing case-insensitive matching "interp".
            # hopefully this makes the viewers I typically use behave better when
            # rending imshow/matshow plots with many small cells (so I don't need to
            # make huge figures to even have a chance, e.g. 650 inches tall in current
            # plot_aligned_dynamics plot. for that particular plot, this alone didn't
            # seem to help me reduce size. or maybe my reader is ignoring this flag, as
            # apparently they are free to do)
            # https://stackoverflow.com/questions/27512326
            # if trying to specify this way, get
            # Warning: Unknown infodict keyword: 'interpolate'. Must be one of {'Title',
            # 'ModDate', 'Keywords', 'Trapped', 'CreationDate', 'Author', 'Producer',
            # 'Creator', 'Subject'}.
            # and there is no interpolate=<bool> kwarg
            #'interpolate': False
        }
    #

    if normalize_fname:
        prefix = util.to_filename(desc)
    else:
        # util.to_filename in branch above adds a '.' suffix by default
        prefix = f'{desc}.'

    basename = prefix + fmt

    makedirs(fig_dir)
    fig_path = Path(fig_dir) / basename

    # TODO share logic w/ to_csv above (meaning also want to resolve() in
    # produces_output? or no? already doing that?)
    abs_fig_path = fig_path.resolve()

    # duped from produces_output
    last_frame = inspect.currentframe().f_back
    filename, lineno, fn_name, _, _ = inspect.getframeinfo(last_frame)
    _saved_path2last_save_code_context[abs_fig_path] = CodeContext(
        filename=filename, lineno=lineno, fn_name=fn_name
    )
    _all_seen_inputs.add(abs_fig_path)
    #

    # TODO (option to) also save traceback at each, for easier debugging?
    # TODO some way to patch avoid this for testing? (support plot_dir=None?
    # have everything reset _savefig_seen_paths for each offending call? actually
    # just save all figs into distinct plot dirs, even for testing...?)
    if not multiple_saves_per_run_ok and abs_fig_path in _savefig_seen_paths:
        # TODO (delete?) fix:
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
        context = _saved_path2last_save_code_context[abs_fig_path]
        # TODO maybe i do need more context (previous stack frames).
        # current issue is same line in two calls to the same fn.
        raise MultipleSavesPerRunException('would have overwritten output '
            f'{fig_path}\npreviously written elsewhere in this run, at:\n'
            f'{context.filename}, line {context.lineno} (in {context.fn_name})'
            '\nadd multiple_saves_per_run_ok=True to call to override, but this'
            ' is likely a mistake'
        )
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

    fig = None
    if isinstance(fig_or_seaborngrid, Figure):
        fig = fig_or_seaborngrid

    # TODO any other types of seaborn objects we might want to support? don't think i've
    # encounterd any so far...
    elif isinstance(fig_or_seaborngrid, sns.axisgrid.Grid):
        fig = fig_or_seaborngrid.fig

    assert fig is not None, (f'{type(fig)=} may not have been an instance of '
        'Figure or sns.axisgrid.Grid (above) (or somehow otherwise fig was None)'
    )

    # al_util.verbose flag set True/False by CLI in al_analysis.main
    if (verbose and not _skip_saving) or debug:
        # TODO may shorten to remove first two components of path by default
        # (<driver>_<indicator>/<fmt> in most/all cases)?
        color = 'light_blue'
        cprint(fig_path, color)

    # TODO move this into produces_output, and work on all filetypes saved w/ that
    # wrapper?
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
    # TODO assert n > 0? > 1?
    ret = (n - 1) * n / 2
    assert np.isclose(ret, int(ret))
    return int(ret)


def n_multichoose_k(n: int, k: int) -> int:
    """Returns number of size-k unordered "multisets" that can be drawn from n options.

    Multisets can contain one element multiple times, but are still unordered. This can
    answer how many ways a population can be resampled with replacement.

    https://en.wikipedia.org/wiki/Multiset#Counting_multisets
    """
    ret = factorial(n + k - 1) / (factorial(k) * factorial(n - 1))
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

    # TODO TODO support panel level being in this index (or at least drop, and make sure
    # that handling is correct. may at least need to also assert same odor doesn't show
    # up in >1 panel then)

    # TODO this causing difficulties later? alternatives?
    # (w/ needing to sort again)
    #
    # sorting index so that diff calls to this will produce mergeable outputs.
    # otherwise, though the set of pairs will be the same, some may be represented
    # in the other order in this index (and thus merging won't find a matching
    # pair).
    # TODO .loc still work w/ list(...)?
    pairs = list(itertools.combinations(corr_df.index.sort_values(), 2))

    # need to stack all levels, in case we also have 'panel' in index names
    # TODO why dropna=False? matter?
    corr_ser = corr_df.stack(level=corr_df.columns.names, dropna=False)

    # TODO delete this branch? make sure it also supports panel level in index?
    if ordered_pairs is not None:
        # does fail in call from end of load_remy_2e...
        # (only happened to be true for my first use case. don't think it actually
        # matters)
        #assert set(ordered_pairs) - set(corr_ser.index) == set()

        pairs = [(b,a) if (b,a) in ordered_pairs else (a,b) for a,b in pairs]

    # itertools.combinations will not give us any combinations of an element with
    # itself. in other words, we won't be keeping the identity correlations.
    assert not any(a == b for a, b in pairs)

    if 'panel' in corr_df.index.names:
        flat_tuples = []
        # e.g. (('control', '1o3ol @ -3'), ('control', '1o3ol+2h @ 0'))
        for p in pairs:
            assert type(p) is tuple and len(p) == 2
            assert all(type(x) is tuple and len(x) == 2 for x in p)
            # makes one 4-element tuple, which should be able to index corr_ser below
            # e.g. ('control', '1o3ol @ -3', 'control', '1o3ol+2h @ 0')
            flat_tuples.append(p[0] + p[1])
        pairs = flat_tuples
    else:
        for p in pairs:
            assert type(p) is tuple and len(p) == 2
            assert all(type(x) is str for x in p)

    # itertools.combinations essentially selects one triangular, excluding diagonal
    corr_ser = corr_ser.loc[pairs]

    if 'panel' not in corr_df.index.names:
        # TODO switch to assertion(s) on input index/column names?
        # (just to fail sooner / be more clear)
        #
        # TODO make more general than assuming 'odor' prefix?
        assert len(corr_ser.index.names) == 2

        # TODO delete?
        assert all(x.startswith('odor') for x in corr_ser.index.names), \
            f'{corr_ser.index.names=}'

        # TODO do 'a','b' instead? other suffix ('_row','_col')? (to not confused w/
        # 'odor1'/'odor2' used in many other MultiIndex levels in here, where 'odor2' is
        # a almost-never-used-anymore optional 2nd odor, where 2 delivered at same time
        # (most recently in kiwi/control 2-component ramp experiments).
        corr_ser.index.names = ['odor1', 'odor2']
    else:
        assert len(corr_ser.index.names) == 4
        odor_var = olf.first_odor_level(corr_df.index)
        assert corr_ser.index.names == ['panel', odor_var, 'panel', odor_var]
        # TODO again, would prefer to use _a/_b suffixes
        corr_ser.index.names = ['panel1', 'odor1', 'panel2', 'odor2']

    # TODO sort output so odors appear in same order as in input (within each component
    # of pair, at least)?

    return corr_ser


# TODO unit test
# TODO remove "private" '_' prefix of _index kwarg?
def invert_corr_triangular(corr_ser, diag_value=1., _index=None, name='odor'):
    if _index is None:
        for_odor_index = corr_ser.index
    else:
        for_odor_index = _index

    have_panels = False
    if any(x.startswith('panel') for x in for_odor_index.names):
        have_panels = True
        # TODO relax these assertions?

    if not have_panels:
        # TODO make more general than assuming 'odor' prefix?
        # TODO + factor to share w/ what corr_triangual sets by default (at least), in
        # case i change suffix added there
        #assert for_odor_index.names == ['odor1', 'odor2']
        # TODO rename all "odor" stuff to be more general (now that i'm not requiring
        # 'odor1'/'odor2')
        assert len(for_odor_index.names) == 2

        # unique values for odor1 and odor2 will not be same (each should have one value
        # not in other). could just sort, for purposes of generating one combined order.
        # for now, assuming (correctly, it seems) that first value in odor1 and last
        # value in odor2 are the only non shared, and that otherwise we want to keep the
        # order
        #
        # pandas <Series>.unique() keeps order of input (assuming all are adjacent, at
        # least)
        assert for_odor_index.names[0].startswith('odor')
        assert for_odor_index.names[1].startswith('odor')
        odor1 = for_odor_index.get_level_values(0).unique()
        odor2 = for_odor_index.get_level_values(1).unique()

        # TODO try to make work without these assertions (would need to change how
        # odor_index is defined below). these seem to work if index is sorted, but not
        # for (at least some) indices before sort_index call.
        index_vals1 = odor1
        index_vals2 = odor2
    else:
        assert for_odor_index.names[0].startswith('panel')
        assert for_odor_index.names[1].startswith('odor')

        index_vals1 = [tuple(x) for x in
            for_odor_index.to_frame(index=False).iloc[:, 0:2].drop_duplicates(
            ).itertuples(index=False)
        ]

        assert for_odor_index.names[2].startswith('panel')
        assert for_odor_index.names[3].startswith('odor')

        index_vals2 = [tuple(x) for x in
            for_odor_index.to_frame(index=False).iloc[:, 2:].drop_duplicates(
            ).itertuples(index=False)
        ]

    assert np.array_equal(index_vals2[:-1], index_vals1[1:])
    assert index_vals1[0] not in set(index_vals2)
    assert index_vals2[-1] not in set(index_vals1)

    if not have_panels:
        # single element list does NOT work here
        odor_index = pd.Index(list(index_vals1) + [index_vals2[-1]], name=name)
    else:
        names = ['panel', name]
        odor_index = pd.MultiIndex.from_tuples(list(index_vals1) + [index_vals2[-1]],
            names=names
        )

    # TODO maybe columns and index should have diff names? keep odor1/odor2?


    square_corr = pd.DataFrame(index=odor_index, columns=odor_index, data=float('nan'))
    for a in odor_index:
        for b in odor_index:
            if not have_panels:
                pair = (a, b)
                reverse_pair = (b, a)
            else:
                pair = tuple(a) + tuple(b)
                reverse_pair = tuple(b) + tuple(a)

            if a == b:
                assert pair not in corr_ser
                square_corr.at[a, b] = diag_value
                continue

            # TODO clean up
            try:
                if pair in corr_ser:
                    assert reverse_pair not in corr_ser
                    c = corr_ser.at[pair]
                else:
                    assert reverse_pair in corr_ser
                    c = corr_ser.at[reverse_pair]
            # TODO delete this try/except if i can't trigger this case in any tests.
            # probably indicates a bug anyway
            except AssertionError:
                raise
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
    # TODO after testing, try to have keep_panel default to True?
    square: bool = True, keep_panel: bool = False) -> Union[pd.Series, pd.DataFrame]:
    """
    Args:
        df: DataFrame with odor level on row index, and levels from id_cols
            in column index.

        id_cols: column index levels, unique combinations of which identify individual
            flies (/ experimental units). if None, defaults to ['date', 'fly_num'].

        square: if True, returns square correlation matrix as a DataFrame. otherwise,
            returns one triangular (excluding diagonal) as a Series.
    """
    # TODO add checks= kwarg to disable all the assertions?

    # TODO also allow selecting 'fly_id' as default, if there?
    if id_cols is None:
        id_cols = fly_cols

    # TODO TODO also work w/ 'odor' level (have in loaded model responses)
    # (or just expose as kwarg?)
    # TODO only do if 'repeat' level present? assert it's there?
    #
    # assumes 'odor2' level, if present, doesn't vary.
    # TODO TODO assert assumption about possible 'odor2' level?
    # TODO TODO assert no variation in any level other than odor1 and repeat?
    # TODO TODO add some option to group by panel too, if available?
    odor_var = olf.first_odor_level(df.index)
    if not keep_panel or 'panel' not in df.index.names:
        by = odor_var
    else:
        by = ['panel', odor_var]

    trialmean_df = df.groupby(level=by, sort=False).mean()
    n_odors = len(trialmean_df)

    first_row_index = None
    # TODO also want to  keep track of and append metadata?
    fly_corrs = []
    for fly, fly_df in trialmean_df.groupby(level=id_cols, axis='columns', sort=False):
        if first_row_index is None:
            first_row_index = fly_df.index.copy()
        else:
            assert fly_df.index.equals(first_row_index)

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

    # TODO TODO separate check that there are no duplicates in columns? think i might
    # have come across some other cases where verify_integrity=True did not do what i
    # expected when concatenating across columns
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

    # TODO TODO need to do something different here if panel level is included above?
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


# TODO don't require prefix, and save to plot_dir / 'corr.<plot_fmt>' by default?
def plot_corr(df: pd.DataFrame, plot_dir: Path, prefix: str, *, title: str = '',
    as_corr_dist: bool = False, verbose: bool = False, _save_kws=None, **kwargs
    ) -> pd.DataFrame:
    """Saves odor-odor correlation plot under <plot_dir>/<prefix>.<plot_fmt>
    """
    # otherwise, we assume input is already a correlation (/ difference of correlations)
    if not df.columns.equals(df.index):
        # TODO delete?
        if len(df.columns) == len(df.index):
            print('double check input is not already a correlation [diff]!')
            import ipdb; ipdb.set_trace()
        #

        # TODO TODO use new al_util.mean_of_fly_corrs instead (when appropriate, e.g.
        # when input has multiple flies [/ model seeds])?
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
    #return_linkages: bool = False, **kwargs) -> sns.axisgrid.Grid:
    return_linkages: bool = False, **kwargs):
    # TODO update doc on row_colors. what are other ways? if i add features to
    # viz.clustermap, just reference that doc here?
    """
    Args:
        return_linkages: passed to `hong2p.viz.clustermap`

        **kwargs: passed to `hong2p.viz.clustermap`

    One way `row_colors` can be passed is as a `Series` with index matching
    `df.columns`, where values are color (RGB?) 3-tuples.
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

    # TODO add option to color rows by fly (-> generate row_colors Series in here)
    # (values of series should be colors)
    # (see natmix_data/analysis.py get_fly_color_series)
    # (2025-11-05: planning to add some related code to viz.clustermap, for use in
    # natmix_data/analysis.py)

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


# TODO move to hong2p?
# TODO TODO use in fn to check whether outputs werer created since proc start
def curr_proc_start_time() -> float:
    """Returns float start time of current process, comparable to `time.time()`
    """
    # seems like this should be in same format/offset as time.time(). creator of package
    # uses it as time.strftime(..., time.localtime(p.create_time())) in an example:
    # https://stackoverflow.com/questions/2598145
    proc = psutil.Process()
    # TODO module-level cache?
    create_time = proc.create_time()
    assert 0 < create_time < time.time()
    return create_time


def written_since_proc_start(path: Pathlike) -> bool:
    """Returns whether path was written since start of current process.

    Also returns False if path does not exist.
    """
    if not Path(path).exists():
        return False

    # getmtime output should also be comparable to time.time()
    return getmtime(path) > curr_proc_start_time()


# TODO delete (/move to hong2p util?)
def print_curr_mem_usage(end: str = '\n') -> None:
    """Prints memory usage (MiB) of current process.

    Prints both resident set size (RSS) and virtual memory size (VMS).
    """
    # https://stackoverflow.com/questions/938733
    proc = psutil.Process()
    byte2MiB = 1024**2
    # NOTE: "resident set size" (rss) is probably what I want, but may also
    # consider "virtual memory size" (vms), which would also include a few other
    # (mostly not actively being used) things.
    # see: https://stackoverflow.com/questions/7880784
    #
    # memory_info().rss should be current memory usage in bytes
    rss = proc.memory_info().rss / byte2MiB
    vms = proc.memory_info().vms / byte2MiB
    print(f'memory usage (MiB): rss={rss:.2f} vms={vms:.2f}', end=end)
#


def get_gsheet_metadata() -> pd.DataFrame:
    """Downloads and formats Google Sheet experiment/fly metadata.

    Loads 'metadata_gsheet_link.txt' from directory containing this script, which should
    contain the full URL to your metadata Google Sheet.

    Most important columns in this sheet are:
    - 'Date': YYYY-MM-DD format dates for when experiments were conducted

    - 'Fly': integers counting up from 1, numbering flies within each date

    - 'Driver': the driver being used to drive indicator expression in this fly
       (e.g. 'pebbled' for pebbled-Gal4, our standard all-ORN driver)

    - 'Indicator': abbrevation for indicator the fly is expressing
       (e.g. '6f' for UAS-GCaMP6f)

    - 'Exclude': a checkbox-column where a check indicates the analysis should not be
       run on this experiment

    - 'Side': values should all be either 'right'|'left'|empty (I use a dropdown to
       enforce this). My recordings have all been imaging only one hemisphere of the
       brain at a time (either the left or the right), but we want to flip them all into
       a standard orientation to make the spatial patterns more easily comparable across
       experiments. All recordings will be flipped to `standard_side_orientation`, if
       not already in that orientation.

    My sheet is called 'tom_antennal_lobe_data' in the Hong lab Google Drive. Sam has
    his own. New users of the pipeline should probably start by copying one of ours, to
    get the right column names, data validation, etc.
    """
    script_dir = Path(__file__).resolve().parent

    # TODO set bool_fillna_false=False (kwarg to gsheet_to_frame) and manually fix any
    # unintentional NaN in these columns if I need to use the missing data for early
    # diagnostic panels (w/o some of the odors only in newest set) for anything
    #
    # This file is intentionally not tracked in git, so you will need to create it and
    # paste in the link to this Google Sheet as the sole contents of that file. The
    # sheet is located on our drive at:
    # 'Hong Lab documents/Tom - odor mixture experiments/tom_antennal_lobe_data'
    #
    # Sam has his own sheet following a similar format, as should any extra user of this
    # pipeline.
    df = util.gsheet_to_frame('metadata_gsheet_link.txt', normalize_col_names=True,
        # so that the .txt file can be found no matter where we run this code from
        # (hong2p defaults to checking current working dir and a hong2p root)
        extra_search_dirs=[script_dir]
    )
    df.set_index(['date', 'fly'], verify_integrity=True, inplace=True)

    # Currently has some explicitly labelled 'pebbled' (for new megamat experiments
    # where I also have some some 'GH146' data), but all other data should come from
    # pebbled flies.
    df.driver = df.driver.fillna('pebbled')

    # TODO if i don't switch off 8m for the PN experiments, first fillna w/ '8m' for
    # GH146 flies
    df.indicator = df.indicator.fillna('6f')

    return df


# TODO delete odor_min_max_scale if i don't end up using
# TODO TODO rename this, and similar w/ 'roi_' (any others?), to exclude that?
# what else would i be plotting responses of? this is the main type of response i'd want
# to plot...
# TODO TODO always(/option to) break each of these into a number of plots such that we
# can always see the xticklabels (at the top), without having to scroll up?
# TODO add option to chop off odor concentrations in odor matshow xticklabels IF it's
# all the same on the input data? maybe default to that?
def plot_all_roi_mean_responses(trial_df: pd.DataFrame, title=None, roi_sort=True,
    sort_rois_first_on=None, odor_sort=True, keep_panels_separate=True,
    roi_min_max_scale=False, odor_min_max_scale=False,

    use_diverging_cmap: bool = True,

    # TODO delete hack!
    yticklabels=None,

    # TODO keep?
    avg_repeats: bool = True,

    single_fly: bool = False,
    odor_glomerulus_combos_to_highlight: Optional[List[Dict]] = None, **kwargs):
    # TODO rename odor_sort -> conc_sort (or delete altogether)
    """Plots odor x ROI data displayed with odors as columns and ROI means as rows.

    Args:
        trial_df: ['odor1', 'odor2', 'repeat'] index names and a column for each ROI.
            ['odor2', 'repeat'] are optional, and 'odor' may be used in place of
            'odor1'.

        roi_sort: whether to sort columns

        sort_rois_first_on: passed to sort_fly_roi_cols's sort_first_on kwarg

        keep_panels_separate: if 'panel' is among trial_df index level names, and there
            are any odors shared by multiple panels, this will prevent data from
            different panels from being averaged together

        roi_min_max_scale: if True, scales data within each ROI to [0, 1].
            if `cbar_label` is in `kwargs`, will append '[0,1] scaled per ROI'.

        odor_min_max_scale: if True, scales data within each odor to [0, 1].
            if `cbar_label` is in `kwargs`, will append '[0,1] scaled per odor'.

        odor_glomerulus_combos_to_highlight: list of dicts with 'odor' and 'glomerulus'
            keys. cells where `odor1` matches 'odor' (with no odor in `odor2`) and `roi`
            matches 'glomerulus' will have a red box drawn around them.

        **kwargs: passed thru to hong2p.viz.matshow
    """
    # TODO factor out this odor-index checking to hong2p.olf?
    # may also have 'panel', 'repeat', 'odor2', and arbitrary other metadata levels.
    if 'odor' in trial_df.index.names:
        assert 'odor1' not in trial_df.index.names
        odor_var = 'odor'
    else:
        assert 'odor1' in trial_df.index.names
        odor_var = 'odor1'

    # TODO also check ROI index (and also factor that to hong2p)
    # TODO maybe also support just 'fly' on the column index (where plot title might be
    # the glomerulus name, and we are showing all fly data for a particular glomerulus)

    avg_levels = [odor_var]
    # TODO handle in a way agnostic to # of components? e.g. supporting also 'odor3',
    # etc, if present
    if 'odor2' in trial_df.index.names:
        avg_levels.append('odor2')

    # TODO unsupport keep_panels_separate=False?
    if keep_panels_separate and 'panel' in trial_df.index.names:
        # TODO TODO TODO warn/err if any null panel values. will silently be dropped as
        # is.
        # TODO or change fn to handle them gracefully (sorting alphabetically w/in?)
        avg_levels = ['panel'] + avg_levels

    if trial_df.index.name == odor_var:
        # assuming input is mean already columns probably are still just 'roi', as I
        # assume is also true in most cases below (as we are only ever computing
        # groupby->mean across row groups in this fn)
        mean_df = trial_df.copy()
    else:
        avg_levels = [x for x in avg_levels if x in trial_df.index.names]

        if not avg_repeats:
            assert 'repeat' in trial_df.index.names
            assert 'repeat' not in avg_levels
            avg_levels.append('repeat')

        # This will throw away any metadata in multiindex levels other than these
        # (so can't just add metadata once at beginning and have it propate through
        # here, without extra work at least)
        mean_df = trial_df.groupby(avg_levels, sort=False).mean()

    # TODO might wanna drop 'panel' level after mean in keep_panels_separate case, so
    # that we don't get the format_mix_from_strs warning about other levels (or just
    # delete that warning...) (still relevant?)

    if roi_min_max_scale:
        assert not odor_min_max_scale

        # TODO may need to check vmin/vmax aren't in kwargs and change if so

        # The .min()/.max() functions should return Series where index elements are ROI
        # labels (or at least it won't be the odor axis based on above assertions...).
        # equivalent to mean_df.[min|max](axis='rows')
        mean_df -= mean_df.min()
        mean_df /= mean_df.max()

        assert np.isclose(mean_df.min().min(), 0)
        assert np.isclose(mean_df.max().max(), 1)

        # TODO set this as full title if not in kwargs?
        if 'cbar_label' in kwargs:
            # (won't modify input)
            kwargs['cbar_label'] += '\n[0,1] scaled per ROI'

    if odor_min_max_scale:
        mean_df = mean_df.T.copy()
        # I tried passing axis='columns' (without transposing first), but then the
        # subtracting didn't seem able to align (nor would other ops, probably)
        mean_df -= mean_df.min()
        mean_df /= mean_df.max()

        mean_df = mean_df.T.copy()

        assert np.isclose(mean_df.min().min(), 0)
        assert np.isclose(mean_df.max().max(), 1)

        if 'cbar_label' in kwargs:
            kwargs['cbar_label'] += '\n[0,1] scaled per odor'

    if odor_sort:
        mean_df = sort_concs(mean_df)

    # TODO TODO also add option to fillna, adding rows until the rows match (or at least
    # include) all the hemibrain glomeruli? maybe sort those not in ANY of my data down
    # below (though would be more complicated...)?
    if roi_sort:
        mean_df = sort_fly_roi_cols(mean_df, sort_first_on=sort_rois_first_on)

    # TODO deal w/ warning this is currently producing (not totally sure it's this call
    # tho)
    xticklabels = format_mix_from_strs

    # TODO TODO numbered ROIs should be shown as before, and not have number shown
    # as an ROI group label (via hline_* stuff) (ideally in same plot w/ some named ROIs
    # grouped, but maybe just disable if not all certain/named)
    # (which plots currently affected by this? still relevant?)
    # (did *uncertain.<plot_fmt> roi matrix plots used to group non-numbered ROIs
    # together? don't think it's doing that now...)

    # TODO try to move some of this logic into viz.matshow?
    # (the automatic enabling of hline_group_text if we have levels_from_labels?)
    # (also, just to not have to redefine the default value of levels_from_labels...)
    hline_group_text = False

    # TODO delete hack?
    if yticklabels is None:
        if not single_fly:
            # (assuming it's a valid callable if so)
            if 'hline_level_fn' in kwargs and not kwargs.get('levels_from_labels',True):
                if all([x in trial_df.columns.names for x in fly_cols]):
                    # TODO maybe still check if there is >1 fly too (esp if this path
                    # produces bad looking figures in that case)

                    # will show the ROI label only once for each group of rows where the
                    hline_group_text = True

            # TODO allow overriding w/ kwarg for case where i wanna call this w/ single
            # fly data? (to make diag examples, but calling as part of
            # acrossfly_response_matrix_plots)
            yticklabels = lambda x: fly_roi_id(x, fly_only=hline_group_text)
        else:
            # TODO factor out to a is_single_fly check or something?
            if all(x in trial_df.columns for x in fly_cols):
                n_flies = len(
                    trial_df.columns.to_frame(index=False)[fly_cols].drop_duplicates()
                )
                assert n_flies == 1
            else:
                assert not any(x in trial_df.columns for x in fly_cols)
            #
            yticklabels = lambda x: x.roi

    vline_group_text = kwargs.pop('vline_group_text', 'panel' in trial_df.index.names)

    mean_df = mean_df.T

    # TODO maybe put lines between levels of sortkey if int (e.g. 'iplane')
    # (and also show on plot as second label above/below roi labels?)

    if roi_min_max_scale or odor_min_max_scale:
        # TODO TODO TODO change [h|vline]s to black in this case
        use_diverging_cmap = False
        # TODO assert no norm / diverging cmap in kwargs?
        if cmap not in kwargs:
            kwargs['cmap'] = cmap

    # TODO detect (using viz.is_diverging_cmap?)?
    if use_diverging_cmap:
        kwargs = {**diverging_cmap_kwargs, **kwargs}
        # center of diverging cmap should be white, so we'll use black lines here
        kwargs['linecolor'] = 'k'

    fig, _ = viz.matshow(mean_df, title=title, xticklabels=xticklabels,
        yticklabels=yticklabels, hline_group_text=hline_group_text,
        vline_group_text=vline_group_text, **kwargs
    )

    if odor_glomerulus_combos_to_highlight is not None:
        # colorbar should be fig.axes[1] if it's there at all
        ax = fig.axes[0]

        # this seems to be default colorbar label. for other ax (one i want), default
        # label seems to be '' here.
        assert '<colorbar>' != ax.get_label()

        # TODO factor this box drawing into some hong2p.viz fn?
        # (use for some plots of sensitivity analysis, to highlight the tuned param
        # combo stepped around? like the one in here, or the one in
        # natmix_data/analysis.py?)
        for combo in odor_glomerulus_combos_to_highlight:
            # TODO also check odor_glomerulus_combos_to_highlight for 'odor' vs 'odor1'?
            # assuming for now it will always be the former
            odor = combo['odor']
            roi = combo['glomerulus']

            matching_roi = mean_df.index.get_level_values('roi') == roi

            matching_odor = mean_df.columns.get_level_values(odor_var) == odor
            if 'odor2' in mean_df.columns.names:
                matching_odor &= (
                    mean_df.columns.get_level_values('odor2') == solvent_str
                )

            if matching_roi.sum() == 0 or matching_odor.sum() == 0:
                continue

            # TODO TODO if there are a few adjacent, find outer edge and just draw one
            # rect?
            # (for highlighting same on plots that have multiple fly data)
            #
            # other cases not currently supported (would have to think about handling)

            # should be fine to ignore / delete, or significantly weaken
            # TODO TODO not true actually, as i'm currently only drawing box around
            # FIRST matching index pair
            #'''
            assert matching_odor.sum() == 1
            # TODO delete try/except
            try:
                assert matching_roi.sum() == 1
            except AssertionError:
                # TODO be more descriptive (say which plot(s) affected) in warning
                warn(f'{matching_roi.sum()=} > 1. disabling box drawing!')
                continue
                #import ipdb; ipdb.set_trace()
            #'''

            # these will get index of first True value
            odor_index = np.argmax(matching_odor)
            roi_index = np.argmax(matching_roi)

            # TODO possible to compute good value for this tho (for flush w/ edge of
            # cell)?
            # since the rect (+ path effect) extend a bit beyond each cell, and it looks
            # kinda bad. 0.05 seems to produce good results (w/ linewidth=1, or
            # linewith=0.5 + patheffects w/ lw=1.0).
            # TODO decrease shrink slightly? some (but--for some reason--not all) boxes
            # seem to show a tiny bit of underlying color on right edge. seemed to
            # happen more on the bright yellow ones. not sure it's consistent...
            # (may only be an issue w/ png too?)
            shrink = 0.05

            # https://stackoverflow.com/questions/37435369
            # -0.5 seems needed for both in order to center box on each matshow cell
            anchor = (odor_index - 0.5 + shrink, roi_index - 0.5 + shrink)
            box_size = 1 - 2 * shrink
            # linewidth: 0.75 bit too much
            rect = patches.Rectangle(anchor, box_size, box_size, facecolor='none',

                # TODO try something other than white/red (that doesn't need PathEffect
                # black outline maybe?) (red pretty bad on magma cmap i'm using).
                # dotted (first try was pretty bad)?
                #
                # don't like w/ edgecolor='r', lw=0.4, PathEffect lw=1.0
                # OK (w/ edgecolor='w', lw=0.5, PathEffect lw=1.0). lw=0.3 prob too low.
                #edgecolor='w', linewidth=0.4,
                #path_effects=[
                #    # w/ linewidth=1 above: 0.75 too little, 2.0 a bit too much
                #    # w/ linewidth=0.75 above: 1.5 OK, but maybe highlights that either
                #    # rect or path effect is offset very slightly (~1px)?
                #    PathEffects.withStroke(linewidth=1.0, foreground='black')
                #],

                # OK (try a brighter green? might need path effect then...)
                #edgecolor='g', linewidth=1.0,

                # OK. maybe my fav so far?
                #edgecolor='k', linewidth=1.0,

                # OK. bit too close to yellow (too light) maybe?.
                # gray ('1.0' = white, '0.0' = black)
                #edgecolor='0.6', linewidth=1.0,

                # among my favorites.
                #edgecolor='0.4', linewidth=1.0,

                # TODO restore
                # think i'll stick with this one for now
                edgecolor='0.5', linewidth=1.0,

                # bad. at least with the linewidth=1.0 (too much) and not densely dotted
                #edgecolor='k', linewidth=1.0, linestyle='dotted',
                # cyan?
            )
            ax.add_patch(rect)

    # TODO just mean across trials right? do i actually use this anywhere?
    # would probably make more sense to just recompute, considering how often i find
    # myself writing `fig, _ = plot...`
    return fig, mean_df


def count_n_per_odor_and_glom(df: pd.DataFrame, *, count_zero: bool = True
    ) -> pd.DataFrame:
    # TODO doc

    if not count_zero:
        # TODO need to do anything special to keep rows for things that would then be
        # fully NaN?
        #
        # since one call of this happens downstream of some 0-filling betty wanted, that
        # i don't think should count for this
        # TODO assert there are actually some 0.0 vals? or warn if not?
        df = df.replace(0.0, np.nan)
    else:
        if (df == 0.0).any().any():
            warn('count_n_per_odor_and_glom: have exact 0.0 data values. also counting '
                'them, though they may just be fill values. pass count_zero=False to '
                'exclude'
            )

    # TODO might need level= here, for sort=False to work as expected?
    # (didn't have it where i copied it from)
    n_per_odor_and_glom = df.notna().groupby(level='roi', sort=False,
        axis='columns').sum()

    return n_per_odor_and_glom


# TODO rename (either this or others) to be consistent about "plot_*" fns either saving
# outputs or not? use decorator to add save (option?) to plot fns that dont save
# (having all either just return fig, or at least having fig as first var returned?)?
def plot_n_per_odor_and_glom(df: pd.DataFrame, *, input_already_counts: bool = False,
    count_zero: bool = True, cmap: str = 'cividis', zero_color='white',
    title: bool = True, title_prefix='', **kwargs) -> Tuple[Figure, pd.DataFrame]:

    if not input_already_counts:
        n_per_odor_and_glom = count_n_per_odor_and_glom(df, count_zero=count_zero)
    else:
        n_per_odor_and_glom = df

    # TODO at least for panels below, show min N for each glomerulus?
    # (maybe as a separate single-column matshow w/ it's own colorbar?)
    # (only relevant in plots that take mean across flies)

    # TODO hong2p.viz tricks to set cmap max dynamically according to data?
    # possible? clean enough to be worth? would also want to intercept vmin/vmax, and
    # set the ticks in cbar_kws as i'm doing below...
    max_n = n_per_odor_and_glom.max().max()
    # discrete colormap: https://stackoverflow.com/questions/14777066
    cmap = plt.get_cmap(cmap, max_n)
    # want to display 0 as distinct (white rather than dark blue)
    cmap.set_under(zero_color)

    n_roi_plot_kws = dict(roimean_plot_kws)
    n_roi_plot_kws['cbar_label'] = 'number of flies (n)'

    if title:
        if len(title_prefix) > 0:
            title_prefix = f'{title_prefix}\n'

        # TODO de-dupe w/ cbar_label? just title_prefix?
        n_roi_plot_kws['title'] = \
            f'{title_prefix}sample size (n) per (glomerulus X odor)'
    else:
        assert len(title_prefix) == 0

    n_roi_plot_kws.update(kwargs)

    fig, _ = plot_all_roi_mean_responses(n_per_odor_and_glom, cmap=cmap,

        # TODO more elegant solution -> delete (detect whether cmap diverging inside
        # plot_all*?)
        use_diverging_cmap=False,

        # TODO why isn't 0 in the bar tho? if the data had 0, would there be?
        #
        # vmin has to be > 0, so that zero_color set correctly via cmap's set_under
        vmin=0.5, vmax=(max_n + 0.5),
        cbar_kws=dict(ticks=np.arange(1, max_n + 1)), **n_roi_plot_kws
    )

    # TODO delete
    # unuseable as is. font too big and may not be transposed and/or aligned correctly.
    #
    # TODO was it constrained layout that was causing (most of?) the issues?
    # can i do without it?
    # TODO would probably have to move this into plot_all_roi_mean...
    # (or otherwise ensure plotted order matches order of n_per_odor_and_glom
    # (as averaged / sorted here) for purposes of drawing N on each cell)
    # n_per_odor_and_glom.groupby([x for x in df.index.names if x != 'repeat'],
    #     sort=False).mean()
    # .max(axis='rows') above, and save to csv (for now)?
    '''
    # TODO implement in such a way that we don't just assume the first axes is the
    # non-colorbar one? it probably always will be tho...
    # not sure i could trust plt.gca() any more either...
    assert len(fig.axes) == 2
    ax = fig.axes[0]

    # TODO need to transpose n_per_odor_and_glom?
    #
    # https://stackoverflow.com/questions/20998083
    for (i, j), n in np.ndenumerate(n_per_odor_and_glom):
        # TODO color visible enough? way to put white behind?
        # or just use some color distinguishable from whole colormap?
        ax.text(j, i, n, ha='center', va='center')
    '''

    return fig, n_per_odor_and_glom


# TODO maybe some of below (stuff re: remy's kc data) should be moved into
# mb_model, rather than here in al_util? (al_analysis can import from mb_model, mb_model
# just can't import from al_analysis)

data_root: Path = Path(__file__).resolve().parent / 'data'

sent_to_remy: Path = data_root / 'sent_to_remy'

# TODO TODO regenerate these (and recommit in new dir, leaving old data), so we also
# have newer parquet / response_calc_json / etc outputs?
# TODO TODO add tests we can re-run al_analysis (w/ appropriate response calcs) and
# regenerate this as well as similar paper outputs (w/ older response calc)
#
# Contains subdirs [kiwi_control|megamat|validation2]_signed-max, each with dF/F
# CSV/pickle outputs and plots committed.
signedmax_orn_dff_dir: Path = sent_to_remy / '2025-09-30_tom_orn_data_signed-max'

# TODO TODO add tests that we can successfully load data with all these fns (both dff
# and est spike deltas). will be mainly important when trying to move data handling from
# stuff in an editable repo on disk to files provided by importlib_resources.
#
# TODO add similar fns for megamat/control (and both for paper response calc and newer
# signedmax one)
# TODO use in tests to check we can repro mean_est_spike_deltas for all, and everything
# downstream
def load_natmix_dff(**kwargs) -> pd.DataFrame:
    data_dir = signedmax_orn_dff_dir / 'kiwi_control_signed-max'

    # TODO also load parquet (and prefer that, if we have it). need to regen and commit
    # new outputs first
    # TODO factor out this name to share w/ mb_model/al_analysis
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)

    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'kiwi', 'control'
    }
    return df


# TODO assert matches subset of
# data/internal/for_dff_to_spiking_fn/ij_certain-roi_stats.parquet?
# TODO use in tests
def load_megamat_dff(**kwargs) -> pd.DataFrame:
    data_dir = signedmax_orn_dff_dir / 'megamat_signed-max'
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)
    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'megamat'
    }
    return df


# TODO assert matches subset of
# data/internal/for_dff_to_spiking_fn/ij_certain-roi_stats.parquet?
# TODO use in tests
def load_validation2_dff(**kwargs) -> pd.DataFrame:
    data_dir = signedmax_orn_dff_dir / 'validation2_signed-max'
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)
    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'validation2'
    }
    return df


# TODO implement this one too (may need to recalc and commit? or instead have flags for
# validation / megamat load fns?
#def load_remypaper_dff(**kwargs) -> pd.DataFrame:


# TODO also anchor path to script dir? would only be to support running from elsewhere,
# which i prob don't care about
remy_data_dir: Path = data_root / 'from_remy'

n_final_megamat_kc_flies: int = 4

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
    # TODO update doc? is it actually reading netcdf? loooks like i'm reading csvs...
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
