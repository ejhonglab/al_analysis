
import inspect
from pathlib import Path
import traceback
from typing import Any, Callable, Dict, List, Set
from contextlib import contextmanager

import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from hong2p.xarray import get_example_orn_dynamics
from hong2p.viz import matshow
from hong2p.olf import solvent_str

from al_analysis.al_util import (MultipleSavesPerRunException, read_json, plot_fmt,
    ParamDict, load_natmix_dff, mean_of_fly_corrs, diverging_cmap, savefig
)
from al_analysis import al_util
# just to include in produces_output test
from al_analysis.mb_model import save_dataarray

# TODO better way?
from .conftest import test_data_dir


# TODO some builtin pytest way of monkey patching module-level variables like this?
@contextmanager
def temporarily_set_al_util_var(var_name: str, tmp_value: Any):
    # assuming we don't need to copy / deepcopy or anything
    orig_value = getattr(al_util, var_name)
    setattr(al_util, var_name, tmp_value)
    try:
        yield
    finally:
        setattr(al_util, var_name, orig_value)


# TODO restore / delete (currently defining all example test data in module level, for
# lack of understanding an easy way to have a fixture that returns output based on an
# argument, or how to manage multiple fixtures [whichever would make sense])
#@pytest.fixture(scope='session')
#def df() -> pd.DataFrame:
#    # TODO use one of my own committed example CSVs? or drosolf.orns.orns() (hallem)?
#    # TODO TODO something with/all the row/column multiindices i might have, to check
#    # parquet / csv round tripping in more relevant cases
#    return sns.load_dataset('iris')
#
#
#@pytest.fixture(scope='session')
#def orns() -> pd.DataFrame:
#    return get_example_orn_dynamics()
#

# TODO add another test for round tripping with read_csv (which is NOT at all as generic
# as name currently suggests, and is for loading specifically my format of antennal
# data) (use load_natmix_dff output for that? that fn uses read_csv tho, so mostly
# tautological...) (could just have a test that we can still read it? maybe that it
# still compares to reading a similar saved parquet file [do i already have such a
# parquet file committed? do so if not])

def assert_wrapped_by_produces_output(fn: Callable) -> None:
    assert fn.__name__ in al_util._fn_name2seen_inputs, \
        f'{fn.__name__} not wrapped by produces_output'


save_fn_name2suffix: Dict[str, str] = {
    'to_csv': '.csv',
    'to_parquet': '.parquet',
    'to_pickle': '.p',
    'save_dataarray': '.nc',
    'to_json': '.json',

    'np_save': '.npy',

    # NOTE: not currently wrapped by produces_output, but would like to make it so it is
    'savefig': f'.{plot_fmt}',
    # TODO just f'.{plot_fmt}', or also other formats? separate test for that?
}
# TODO some way to have values of dict be fixtures instead? or how would it make sense
# to do this?
# TODO convert to fixture?
# TODO add read_csv arg to drop odor2 != solvent anyway? thats what warning i'm
# silencing is about, and i do that everywhere i use this in here currently
df: pd.DataFrame = load_natmix_dff(warn_=False, verbose=False)

def test_mean_of_fly_corrs():
    # TODO use some other fn to drop these?
    # TODO want to try adding support for these to corr fns? may not...
    no_mixtures = df.loc[df.index.get_level_values('odor2') == solvent_str]

    one_panel_no_mixtures = df.loc['kiwi']
    corr1 = mean_of_fly_corrs(one_panel_no_mixtures)
    # TODO delete
    #print()
    #print('corr1:')
    #print(corr1)

    # TODO actually check anything? mainly wanted to check that it all runs,
    # as branch w/ 'panel' didn't at inception of this test, and not sure i'll ever
    # support odor2 != solvent

    # TODO want to check anything w/ keep_panel=False here? odors duplicated or averaged
    # over across panels? want to fail if there are duplicate odors across panels?
    # TODO check shape is same regardless of keep_panel=True/False? (assuming duplicate
    # odors, after dropping panels, have order preserved [which is probably not true...)
    corr2 = mean_of_fly_corrs(no_mixtures, keep_panel=True)
    # TODO delete
    #print()
    #print('corr2:')
    #print(corr2)
    #breakpoint()


# TODO convert to fixture?
def get_example_corr(df: pd.DataFrame) -> pd.DataFrame:
    no_mixtures = df.loc[df.index.get_level_values('odor2') == solvent_str]
    corr = mean_of_fly_corrs(no_mixtures, keep_panel=True)
    return corr


def make_corr_fig(corr: pd.DataFrame) -> Figure:
    # TODO lines between panels?
    fig, _ = matshow(corr, vmin=-1, vcenter=0, vmax=1, cmap=diverging_cmap,
        norm='two-slope'
    )
    return fig


# TODO restore (/ put behind fixtures / in fn that needs them)
orns: xr.DataArray = get_example_orn_dynamics()

# this one might be a bit tautological... was probably saved w/ check=True in first
# place
example_params: ParamDict = read_json(
    test_data_dir / 'example_saved_model_params/params.json'
)

# TODO maybe a fixture that takes an argument (fn [name?]), and loads/computes the data
# and returns as needed? (then pass indirect arg to that fixture, from parametrized
# test)
save_fn_name2test_data: Dict[str, Any] = {
    'to_csv': df,
    'to_parquet': df,
    'to_pickle': df,
    'save_dataarray': orns,
    'to_json': example_params,
    'np_save': df.values.copy(),

    # TODO add savefig if i also include it (maybe in separate tests tho?)
    # (generate a fig here, maybe viz.matshow corr from df, esp if one of my dfs)
    'savefig': make_corr_fig(get_example_corr(df)),
}
fn_names_with_check_flag: Set[str] = set()
def get_produces_output_wrapped_fns() -> List[Callable]:
    """Returns list of functions wrapped by `@produces_output` decorator.

    Also populates `fn_names_with_check_flag`, so that flag can be set True for some
    tests later.
    """
    # NOTE: for now, just asserting there is only a bool flag spelled "check" and not
    # "checks", and then support that later if i realllllly want, but should probably
    # change all code to use consistent spelling, if ever find something with "checks"
    # spelling.
    fns = []
    for name, fn in al_util._fn_name2wrapped_fn.items():
        fn_name = fn.__name__
        params = inspect.signature(fn).parameters
        assert 'checks' not in params, "expecting 'check' spelling, not 'checks'"
        if 'check' in params:
            check = params['check']
            assert type(check.default) is bool, (f'expected check= to have bool default'
                f'. got {type(check.default)=}'
            )
            # inspect._empty seems to be what keyword arguments with no type annotation
            # get
            if check.annotation != inspect._empty:
                assert check.annotation is bool, (f'expected check= to have bool type '
                    'annotation, if any. got {check.annotation=}'
                )
            fn_names_with_check_flag.add(fn_name)
        else:
            # TODO need to also check it's actually right type? can't find anything
            # obvious to check, other than that repr is <Parameter "**kwargs">.
            # not sure it's worth
            # checking if **kwargs seems to support same type of check[s] flag, as maybe
            # the wrapped fn is calling something else with that flag. assuming that if
            # we don't get a TypeError complaining about it missing, we probably want
            # to set it to true
            if 'kwargs' in params and fn_name in save_fn_name2test_data:
                data = save_fn_name2test_data[fn_name]
                # seems that to_csv does not complain about checks=True. and not sure
                # that's b/c the pandas function actually does something with it...
                # blacklist that one?
                suffix = save_fn_name2suffix.get(fn_name, '')

                try:
                    fn(data, (Path('.') / 'test1').with_suffix(suffix), checks=True,
                        verbose=False
                    )
                    assert False, "expecting 'check' spelling, not 'checks'"
                # TypeError: to_csv() got an unexpected keyword argument 'checks'
                except TypeError:
                    pass

                try:
                    fn(data, (Path('.') / 'test2').with_suffix(suffix), check=True,
                        verbose=False
                    )
                    # TODO also inspect and check type of this final fn that has the
                    # check flag? if i could do that (possible via inspect?) could skip
                    # all this try/except logic for kwargs anyway...
                    fn_names_with_check_flag.add(fn_name)
                # TypeError: to_csv() got an unexpected keyword argument 'check'
                except TypeError:
                    pass

        fns.append(fn)

    return fns


save_fn_name2extra_kws = {
    'to_pickle': dict(write_parquet=False),
}
wrapped_fns = get_produces_output_wrapped_fns()
@pytest.mark.parametrize('save_fn', wrapped_fns)
def test_produces_output(tmp_path, save_fn):
    # NOTE: save_fn.__name__ is still 'to_csv'/etc, as needed for some of below to work
    assert_wrapped_by_produces_output(save_fn)

    fn_name = save_fn.__name__

    assert fn_name in save_fn_name2suffix, ('add entry in save_fn_name2suffix, '
        f'for (what is presumably a new @produces_output wrapped function: {fn_name})'
    )
    suffix = save_fn_name2suffix[fn_name]
    if fn_name in save_fn_name2extra_kws:
        kws = save_fn_name2extra_kws[fn_name]
    else:
        kws = dict()

    kws['verbose'] = False

    # produces_output will also resolve() paths before using them as keys
    output_path = (tmp_path / 'test').with_suffix(suffix).resolve()

    # TODO xfail if missing? or just fail? (nothing should be missing now, at least,
    # unless more functions are registered with @produces_output)
    assert fn_name in save_fn_name2test_data, ('add entry in save_fn_name2test_data, '
        f'for (what is presumably a new @produces_output wrapped function: {fn_name})'
    )
    data = save_fn_name2test_data[fn_name]

    assert output_path not in al_util._saved_path2last_save_code_context
    assert output_path not in al_util._all_seen_inputs

    first_kws = dict(kws)
    if fn_name in fn_names_with_check_flag:
        first_kws['check'] = True

    # this line must be right before the to_csv call, so that lineno+1 should match the
    # lineno of the to_csv call
    filename, lineno, fn_name, _, _ = inspect.getframeinfo(inspect.currentframe())
    save_fn(data, output_path, **first_kws)

    code_context = al_util._saved_path2last_save_code_context[output_path]
    # NOTE: these are both str ('/home/tom/src/al_analysis/test/test_al_util.py')
    assert code_context.filename == __file__
    assert code_context.lineno == lineno + 1
    assert code_context.fn_name == test_produces_output.__name__
    assert output_path in al_util._all_seen_inputs

    with pytest.raises(MultipleSavesPerRunException):
        save_fn(data, output_path, **kws)

    save_fn(data, output_path, multiple_saves_per_run_ok=True, **kws)

    assert al_util.check_outputs_unchanged == False
    # both of these should have the same behavior, since none of this code should be
    # considered as in main
    for x in (True, 'nonmain'):
        with temporarily_set_al_util_var('check_outputs_unchanged', x):
            assert al_util.check_outputs_unchanged == x

            # multiple_saves_per_run_ok=True is still needed even with
            # al_util.check_outputs_unchanged=True
            save_fn(data, output_path, multiple_saves_per_run_ok=True, **kws)

            # TODO TODO TODO implement similar things for data of type other than
            # dataframe
            new_value = 0
            if isinstance(data, pd.DataFrame):
                changed_data = data.copy()
                assert changed_data.iat[0, 0] != new_value
                changed_data.iat[0, 0] = new_value

            elif isinstance(data, np.ndarray):
                changed_data = data.copy()
                assert changed_data[0, 0] != new_value
                changed_data[0, 0] = new_value

            elif isinstance(data, xr.DataArray):
                assert len(data.values.shape) == 3
                changed_data = data.copy()
                assert changed_data.values[0,0,0] != new_value
                changed_data.values[0,0,0] = new_value

            elif isinstance(data, dict):
                assert 'x' not in data
                changed_data = dict(data)
                changed_data['x'] = 1

            # currently won't be reached, because savefig is not currently wrapped by
            # produces output, and it has a test duplicated from this one
            elif isinstance(data, Figure):
                # assuming data is a fig that came from:
                # make_corr_fig(get_example_corr(df))
                corr2 = get_example_corr(df)
                assert corr2.iat[0, 0] != new_value
                corr2.iat[0, 0] = new_value
                fig = make_corr_fig(corr2)
                changed_data = fig
            else:
                assert False, f'{type(data)=} not supported'

            # TODO should this be a more specific custom error instead?
            # WouldHaveChangedError?
            with pytest.raises(RuntimeError):
                save_fn(changed_data, output_path, multiple_saves_per_run_ok=True,
                    **kws
                )

    assert al_util.check_outputs_unchanged == False


# TODO TODO dedupe w/ test_produces_output (or delete entirely, if making savefig
# wrapped w/ @produces_output) (if i ever do that...)
# TODO test savefig w/ multiple plot formats? (currently will just use al_util.plot_fmt,
# which is currently 'pdf') (would have to use other suffix values than one from
# save_fn_name2suffix)
@pytest.mark.parametrize('save_fn', [savefig])
def test_savefig(tmp_path, save_fn):
    fn_name = save_fn.__name__

    suffix = save_fn_name2suffix[fn_name]
    if fn_name in save_fn_name2extra_kws:
        kws = save_fn_name2extra_kws[fn_name]
    else:
        kws = dict()

    # produces: TypeError: print_pdf() got an unexpected keyword argument 'verbose'
    # in at least some calls below (probably first one)
    #kws['verbose'] = False

    # produces_output will also resolve() paths before using them as keys
    output_path = (tmp_path / 'test').with_suffix(suffix).resolve()

    data = save_fn_name2test_data[fn_name]

    assert output_path not in al_util._saved_path2last_save_code_context
    assert output_path not in al_util._all_seen_inputs

    # this line must be right before the to_csv call, so that lineno+1 should match the
    # lineno of the to_csv call
    filename, lineno, fn_name, _, _ = inspect.getframeinfo(inspect.currentframe())
    save_fn(data, output_path.parent, output_path.stem, **kws)

    code_context = al_util._saved_path2last_save_code_context[output_path]
    # NOTE: these are both str ('/home/tom/src/al_analysis/test/test_al_util.py')
    assert code_context.filename == __file__
    assert code_context.lineno == lineno + 1
    assert code_context.fn_name == test_savefig.__name__
    assert output_path in al_util._all_seen_inputs

    with pytest.raises(MultipleSavesPerRunException):
        save_fn(data, output_path.parent, output_path.stem, **kws)

    save_fn(data, output_path.parent, output_path.stem, multiple_saves_per_run_ok=True,
        **kws
    )

    assert al_util.check_outputs_unchanged == False
    # both of these should have the same behavior, since none of this code should be
    # considered as in main
    for x in (True, 'nonmain'):
        with temporarily_set_al_util_var('check_outputs_unchanged', x):
            assert al_util.check_outputs_unchanged == x

            # multiple_saves_per_run_ok=True is still needed even with
            # al_util.check_outputs_unchanged=True
            # TODO will this even work w/o adding a multiple_saves_per_run_ok flag
            # to savefig? (or just switching to using produces_output wrapper)
            # (no, i needed to add this flag to savefig, since it didn't previously have
            # one, and i didn't want to go through the presumably larger work of
            # removing duplicated produces_output functionality in savefig, adding
            # wrapper, and making sure that didn't break anything)
            save_fn(data, output_path.parent, output_path.stem,
                multiple_saves_per_run_ok=True, **kws
            )

            new_value = 0
            # currently won't be reached, because savefig is not currently wrapped by
            # produces output, and it has a test duplicated from this one
            if isinstance(data, Figure):
                # assuming data is a fig that came from:
                # make_corr_fig(get_example_corr(df))
                corr2 = get_example_corr(df)
                assert corr2.iat[0, 0] != new_value
                corr2.iat[0, 0] = new_value
                fig = make_corr_fig(corr2)
                changed_data = fig
            else:
                assert False, f'{type(data)=} not supported'

            # TODO should this be a more specific custom error instead?
            # WouldHaveChangedError?
            with pytest.raises(RuntimeError):
                save_fn(changed_data, output_path.parent, output_path.stem,
                    multiple_saves_per_run_ok=True, **kws
                )

    assert al_util.check_outputs_unchanged == False

