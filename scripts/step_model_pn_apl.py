#!/usr/bin/env python3
"""
After `pip install`-ing `al_analysis`, you should be able to invoke this script via:
`step_model_pn_apl`, and it will save outputs in current directory (can be quite large,
if saving dynamics via `-d/--save-dynamics` flag).

Does not depend on any input data. Loads precomputed megamat ORN spike delta estimates,
and runs those through models with parameters in `MODEL_TUNE_KWS` (one instantiation per
entry in that list).
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from itertools import product
from pathlib import Path
from pprint import pformat, pprint
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hong2p.viz import matshow
from hong2p.util import symlink, subset_same_in_all_dicts, shorten_path
import al_util
from al_util import savefig, ParamDict, warn, read_parquet, to_json, read_json
from mb_model import (fit_and_plot_mb_model, megamat_orn_deltas, dict_seq_product,
    format_weights, format_model_params, get_thr_and_APL_weights, glomerulus_col,
    save_and_remove_from_param_dict, drop_silent_model_cells, load_and_plot_dynamics,
    update_var2range, MinMaxDict
)


MODEL_TUNE_KWS: List[ParamDict] = dict_seq_product(
    [
        dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
            use_connectome_APL_weights=True
        )
    ],
    # pn_claw_to_apl=False is the default, and could normally be omitted, but doing it
    # this way produces nicer directory names when using subset_same_in_all_dicts to
    # exclude params
    [dict(pn_claw_to_apl=True), dict(pn_claw_to_apl=False)]
)

OUTPUT_ROOT_NAME: str = 'PNAPL_stepping'

# TODO tuple, to make sure this doesn't get mutated?
STEPS = [100, 20, 1.0, 0.5, 10, .1]

def analyze_outputs(plot_dir: Path, *, plot_dynamics: bool = False,
    corners_only: bool = False) -> None:
    # TODO doc

    kstr = plot_dir.name
    cols = ['wAPLPN', 'wPNAPL', 'sparsity', 'n_silent_cells', 'hept_pent_corr',
        'avg_lts', 'n_avg_odors_responded_to'
    ]
    cols_computed_over_nonresponders = ['avg_lts', 'n_avg_odors_responded_to',
        'hept_pent_corr'
    ]
    col2range = {
        # yes, it could technically be negative, but don't care about that,
        # and unlikely (except maybe noise. could maybe set lower bound to -0.05 or so?)
        # TODO check what actual min is, and at least warn if it's too far below?
        # TODO use const > 0 min (e.g. .2)?
        'hept_pent_corr': (0, 1),

        # TODO assert 1732 matches shape component of all rs below (it should)
        'n_silent_cells': (0, 1732),

        # TODO mechanism for fixing min to 0, but letting max float w/ data?
        # TODO or what kind of max i want for n_avg_odors_responded to?

        # TODO TODO TODO have diverging cmap around 0.1
        # TODO TODO have diverging cmap around tuned value for all of these?
        #'sparsity': (0, 1),
        'sparsity': (0, .15),

        # TODO assert no data exceeds max here
        'n_avg_odors_responded_to': (0, 5),

        # TODO is this correct? verify
        'avg_lts': (0, 1),
    }
    vals = []
    plot_suffix: str = '.pdf'
    d0_dynamics_plotnames: Optional[Set[str]] = None
    d0_dynamics_plot_dirnames: Set[str] = set()
    n_corners_seen = 0
    var2range: MinMaxDict = dict()

    dir_iter = list(plot_dir.glob('*/'))
    existing_var2range = None
    if plot_dynamics:
        dir_iter = tqdm(dir_iter, unit='model-dir', total=4)

        var2range_json = plot_dir / 'var2range.json'
        if var2range_json.exists():
            # TODO refactor to share tuple conversion w/ check below? or change type to
            # always have list (instead of tuple) pairs as values
            existing_var2range = {
                k: tuple(v) for k, v in read_json(var2range_json).items()
            }
        else:
            warn(f'{var2range_json} did not exist yet, so can not set consistent scale '
                'for plot_dynamics across directories. should generate one if this run '
                'finishes.'
            )

    for d in dir_iter:
        if not d.is_dir():
            continue

        # all other directories should be named like: 'wAPLPN-0.10_wPNAPL-20.00'
        if d.name in {'model_internals', 'dynamics'} | d0_dynamics_plot_dirnames:
            continue

        # TODO pad all numbers for symlinking (or in general?), so sorting is
        # consistent? (actually, happens to be fine as-is, for current steps at least)
        try:
            a2p, p2a = d.name.split('_')
        # ValueError: too many values to unpack (expected 2)
        # probably would be b/c an old plot has a link dir setup, but
        # d0_dynamics_plot_dirnames doesn't currently include that plot
        except ValueError:
            warn(f'delete old plot link dir: {shorten_path(d, n=2)}\nnot among current '
                f'plot names in first directory{pformat(d0_dynamics_plot_dirnames)}'
            )
            # TODO assert all contents are symlinks? (or provide diff warning / err?)
            continue

        a2p = float(a2p.split('-')[-1])
        p2a = float(p2a.split('-')[-1])
        if corners_only:
            limits = (min(STEPS), max(STEPS))
            if not (a2p in limits and p2a in limits):
                warn(f'skipping {d.name} because not among corners (and '
                    'corners_only=True)'
                )
                continue
            n_corners_seen += 1

        # this step is slow, so want to be after corners_only check
        if plot_dynamics:
            curr_var2range = load_and_plot_dynamics(d, var2range=existing_var2range)
            update_var2range(var2range, curr_var2range)

        if d0_dynamics_plotnames is None:
            # assuming this should be same for all subdirs
            d0_dynamics_plotnames = set(
                x.name for x in (d / 'dynamics').glob(f'*{plot_suffix}')
            )
            for p in d0_dynamics_plotnames:
                curr_plot_link_dir = (plot_dir / p).with_suffix('')
                curr_plot_link_dir.mkdir(exist_ok=True)
                d0_dynamics_plot_dirnames.add(curr_plot_link_dir.name)

        for p in d0_dynamics_plotnames:
            dynamics_plot_dir = d / 'dynamics'
            assert dynamics_plot_dir.is_dir(), f'{dynamics_plot_dir=}'
            src = dynamics_plot_dir / p
            # TODO convert to warning? or fine as long as this happens after
            # plot_dynamics (as it does now)?
            assert src.is_file() and not src.is_symlink()

            curr_plot_link_dir = (plot_dir / p).with_suffix('')
            assert curr_plot_link_dir.is_dir()

            # can't use with_suffix on something w/ d.name as name, or it will strip the
            # last bit of the rightmost float parameter from name (after decimal)
            link = curr_plot_link_dir / f'{d.name}{plot_suffix}'
            if link.exists():
                assert link.is_symlink()
                link.unlink()

            # TODO verbose=True (wouldn't currently do what i want)? print something?
            symlink(src, link)

        # TODO also compute + save/load (to json) + use min/max limits for all vars
        # below (should only relevant if doing a higher dimensional sweep, where i'll be
        # plotting a grid of those grids)

        rs = read_parquet(d / 'responses.parquet')
        sp = rs.mean().mean()

        # NOTE: all quantities computed using rs_nosilent (as opposed to responses that
        # still have non-responding cells) should have their column name manually added
        # to cols_computed_over_nonresponders above
        rs_nosilent = drop_silent_model_cells(rs)
        n_silent = len(rs) - len(rs_nosilent)
        # TODO average over corr w/ 1-6ol too? also eb/ep vs 1-5ol/1-6ol block?
        # TODO + maybe ratio/diff w/ corrs in rest of off diag, if needed (but prob not)
        hept_pent_corr = rs_nosilent.corr().loc['1-5ol @ -3', '2h @ -3']
        # TODO TODO say what remy's value (computed on real KCs) is here, for reference?

        xs = rs_nosilent
        n_odors = (xs > 0).T.sum()
        avg_n_odors = n_odors.mean()

        # TODO add some assertions verifying this (what exactly? bounds? which cases
        # produce min/max values?)
        xs = xs.T
        L = pow(xs.sum(), 2.0)/(xs*xs).sum()
        L =  (1.0 - L/len(xs))/(1.0 - 1.0/len(xs))
        L = L.fillna(1.0)
        assert len(L) == len(n_odors)

        vals.append((a2p, p2a, sp, n_silent, hept_pent_corr, L.mean(), avg_n_odors))

    if corners_only:
        # TODO change if needed, if i sweep over more than 2 dims (i.e. adding wAPLKC
        # and wKCAPL)
        assert n_corners_seen == 4, f'{n_corners_seen=} != 4'

    df = pd.DataFrame.from_records(vals, columns=cols)
    df = df.set_index(['wAPLPN', 'wPNAPL'], verify_integrity=True)
    if len(df) == 0:
        raise IOError(f'found no stepped model output subdirectories under {plot_dir}')

    print()
    # {'Is_from_kcs': (1.5808925149559592, 77.70825644115786),
    # 'Is_from_pns': (0.028954122482549444, 2248.2127184891215),
    # 'Is_sims': (0.0, 0.0),
    # 'bouton_sims': (0.0, 177.49935256444556),
    # 'claw_sims': (0.0, 177.49935256444556),
    # 'inh_sims': (0.24619630866805273, 844.2746926926052),
    # 'vm_sims': (0.0, 595.8084413722237)}
    print('var2range:')
    pprint(var2range)
    # TODO change type of this to use lists instead of tuples for the ranges?
    # that's why check=True path is failing, b/c they are converted to lists on reading
    to_json(var2range, var2range_json, check=False)
    v2r2 = {k: tuple(v) for k, v in read_json(var2range_json).items()}
    assert v2r2 == var2range

    # TODO rotate xticks to horizontal (+ put on bottom, or put xlabel in title
    # instead?)

    # TODO TODO start all (/most?) color scales from 0 (i.e. n_avg_odors_responses_to,
    # sparsity, corr, # silent cells. lifetime sparseness?)?
    # TODO TODO fixed max for each too?

    # TODO TODO draw box around tuned or similar? (refactor to share code that is doing
    # something similar in al_analysis.py, where it's used to box particular (glomeruli,
    # odor) combos?)
    # TODO pass this in separately? (/define at module level?)
    plot_root = plot_dir.parent
    for c in df.columns:
        print(f'{c=}')
        fig, ax = plt.subplots()
        mat = df[c].unstack()
        # TODO colorbar?
        mat.columns = mat.columns.astype(str)
        mat.index = mat.index.astype(str)

        assert c in col2range
        vmin, vmax = col2range[c]
        assert mat.min().min() >= vmin, f'{c=} {mat.min().min()=} {vmin}'
        assert mat.max().max() <= vmax, f'{c=} {mat.max().max()=} {vmax}'

        # TODO fail earlier (w/ better err message) in viz.matshow, if mat is empty.
        # currently get: `ZeroDivisionError: float division by zero`
        #   File "/home/tom/src/hong2p/hong2p/viz.py", line 1741, in matshow
        # fontsize = min(10.0, 240.0 / max(df.shape[0], df.shape[1]))
        matshow(mat, ax=ax, xticklabels=True, yticklabels=True, cbar_label=c,
            vmin=vmin, vmax=vmax
        )

        # wPNAPL
        ax.set_ylabel(mat.index.name, fontsize=10)
        # wAPLPN
        ax.set_xlabel(mat.columns.name, fontsize=10)

        title = f'{kstr}\n{c}'
        if c in cols_computed_over_nonresponders:
            title += '\nsilent cells dropped'
        else:
            title += '\nall cells, including silent'
        # TODO include other model params?
        ax.set_title(title)

        savefig(fig, plot_root, f'{kstr}__{c}', bbox_inches='tight')


def step_pn_apl_weights_around_tuned(plot_dir: Path, orn_deltas: pd.DataFrame,
    kws: ParamDict, *, ignore_existing: bool = False, save_dynamics: bool = False,
    tuned_only: bool = False, corners_only: bool = False) -> None:
    # TODO doc
    """Runs `orn_deltas`
    Args:
        orn_deltas: glomerulus (rows) X odors (columns) estimated spike delta DataFrame

        kws: passed to all `fit_and_plot_mb_model` calls, and passed thru
            `format_model_params` to create output subdirectory names

        ignore_existing: if False, will attempt to load cached model outputs (already in
            directories that would be created), rather than re-running models. If True,
            will always re-run models.

        save_dynamics: if True, will save DataArray pickles of all model internal
            dynamic quantities (e.g. membrane potential of KCs over time, to each odor)

        corners_only: if True, only analyzes combinations of min/max step for each
            paramerter
    """
    output_kws = dict(
        # if return_dynamics is True, fit_and_plot_mb_model will write DataArrays
        # containing dynamics as pickles, before popping them from returned param dict.
        # plot_example_dynamics will make some internal plots using the same data, but
        # then will not return them from fit_mb_model (so they will not be saved).
        return_dynamics=save_dynamics,
        # TODO delete make_plots=True? just make sure plot_example_dynamics sets
        # that? or do i really want (/have) a flag controlling plots other than
        # plot_example_dynamics (i.e. internal corrs)? rename, if that's what
        # it's for?
        plot_example_dynamics=True, make_plots=True, connectome_weight_plots=False
    )

    # TODO add fit_mb_model option to assert output is still within target sparsity,
    # when passing in fixed_thr and wAPLKC (to check we are still within what would
    # converge tuning, even if skipping it)? (-> use here, if not tuning)
    # TODO or otherwise assert in here that if we re-run calls from scratch, we get
    # whatever hardcoding parameters we might sometimes use to skip tuning (within
    # tolerance)
    # TODO TODO refactor to share test precalc_weights=True code w/ this (if i get that
    # working). would skip a lot of time spent in calls below

    # TODO try to get this False after getting code recalculating (what?) below working
    # (should be fine now (?))
    # TODO delete eventually?
    return_olfsysm_vars = True
    #
    delete_pretime = True

    # TODO make CLI arg for this?
    # didn't work (2026-03-15): 50 (but it was still oscillating a lot. try lower
    # initial sp_lr_coeff?)
    max_iters = 100

    plot_root = plot_dir.parent
    # TODO TODO also step around fixed_thr/wAPLKC from previous tuning, e.g. a
    # similar model w/o the PN<>APL weights?
    # TODO try a hypergrid stepping thr and APL independently too? (w/ same steps
    # for PN>APL and APL>PN weights)
    params = fit_and_plot_mb_model(plot_root, plot_dirname=plot_dir.name,
        orn_deltas=orn_deltas, verbose=True, try_cache=not ignore_existing,
        **kws, **output_kws, return_olfsysm_vars=return_olfsysm_vars,
        delete_pretime=delete_pretime, max_iters=max_iters
    )
    if tuned_only:
        warn('skipping all PN<>APL weight sweeping, because tuned_only=True')
        return

    # TODO add option just to reanalyze any saved dynamics, if i factor out that
    # plotting code from fit_mb_model? (do have plot-dynamics CLI for that now)
    thr_and_apl_kws = get_thr_and_APL_weights(params, kws)
    print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')
    wAPLPN_scale = thr_and_apl_kws['wAPLPN']
    wAPLKC_scale = thr_and_apl_kws['wAPLKC']

    wPNKC = read_parquet(plot_dir / 'wPNKC.parquet')

    # currently 389
    n_boutons = len(wPNKC.columns)

    # should be 54
    n_gloms = wPNKC.columns.get_level_values(glomerulus_col).nunique()
    assert n_gloms < n_boutons, f'{n_gloms=} >= {n_boutons=}'

    # rv/mp will only be in params if return_olfsysm_vars=True was set in
    # fit_mb_model call above (and will never be in cached outputs)
    if 'rv' in params:
        rv = params['rv']
        mp = params['mp']
        assert mp.pn.n_total_boutons > 0

        assert n_boutons == mp.pn.n_total_boutons

        # these will not currently be in thr_and_apl_kws (assumed each can be
        # calculated from the from-APL weights), so need to get separately
        responses = read_parquet(plot_dir / 'responses.parquet')
        n_kcs = mp.kc.N
        assert n_kcs == len(responses)

        wKCAPL_scale = rv.kc.wKCAPL_scale
        # NOTE: this one may change to have n_claws as denominator, if I change all
        # handling to be consistent eventually
        assert np.isclose(wAPLKC_scale / n_kcs, wKCAPL_scale)

        wPNAPL_scale = rv.pn.wPNAPL_scale
        assert np.isclose(wAPLPN_scale / n_boutons, wPNAPL_scale)

        # just how initial implementation initialized things
        assert np.isclose(wAPLKC_scale, wAPLPN_scale)
    else:
        # TODO warn about how we are calculating wPNAPL_scale (and wKCAPL_scale, if
        # used)
        wPNAPL_scale = wAPLPN_scale / n_boutons

    print()
    print('stepping wAPLPN & wPNAPL around tuned value:')
    # TODO TODO also try at a few diff wKCAPL/wAPLKC scales? (paper hemibrain was
    # wAPLKC=4.63/wKCAPL=0.00252, for ref)
    # TODO TODO worth trying w/ change in how thr is calculated, so it's not
    # relative to spont in? (how to even do? what's that look like w/ other things
    # same?)
    # TODO worth trying w/ a couple diff sp_factor_pre_APL? (1.5 / 3.0?)
    # TODO TODO these are ultimately sorted before plots, right?
    if not corners_only:
        steps = STEPS
    else:
        steps = [min(STEPS), max(STEPS)]

    # TODO provide warning / fail early if we can estimate we won't have enough disk
    # space (if return_dynamics / plot_example_dynamics)?
    for ap, pa in tqdm(list(product(steps, steps)), unit='param-combo'):
        # TODO save plots/dynamics from previous call, and link to that dir?
        # (do definitely want to include it, one way or another)
        #if ap == 1 and pa == 1:
        #    continue

        step = dict(thr_and_apl_kws)
        step['wAPLPN'] = wAPLPN_scale * ap
        # TODO TODO make sure this one is also in format_model_params output,
        # esp when not derivable from wAPLPN (not using format_model_params for now)
        step['wPNAPL'] = wPNAPL_scale * pa
        # TODO remove part about wAPLKC?
        #param_dir = plot_dir / format_model_params(step)

        # TODO TODO TODO actually try per bouton/claw[/KC?] inh dynamics (prob both for
        # KC claws and PN boutons)
        # TODO TODO only matter if i also have a per-bouton/claw synaptic depression (or
        # some other kind of saturation?) add that too?

        param_dir = plot_dir / (
            # TODO factor ', ' stripping into option for format_weights (/ another fn?)
            # with orig values scaled, could get duplicate plot dir names, b/c some
            # values too small for .3f float format
            # TODO change float formatting in format_weights to fix that
            # TODO delete?
            #format_weights(step['wAPLPN'], 'wAPLPN').strip(', ') + '_' +
            #format_weights(step['wPNAPL'], 'wPNAPL').strip(', ')
            format_weights(ap, 'wAPLPN').strip(', ') + '_' +
            format_weights(pa, 'wPNAPL').strip(', ')
        ).replace('=', '-')

        print(f'{param_dir.name}')

        # TODO add arg to skip CSV saving + plotting for cached outputs (preferable)? or
        # handle cache checking in here, to avoid calling (would rather not)?
        # TODO (delete) also pass try_cache=(not ignore_existing) here (prob, yea)?
        # or would i prefer to do the checking in here (prob not)?
        fit_and_plot_mb_model(plot_dir, plot_dirname=param_dir.name,
            try_cache=not ignore_existing, orn_deltas=orn_deltas,
            **step, **kws, **output_kws
        )

        # TODO (delete?) breadth across KCs (vs max) (make sense?) (something else that
        # might be diff across KCs to account for sparsity diff?)?


def main():
    # TODO how to preserve newlines in this description again?
    parser = ArgumentParser(description='will run models with the following '
        f'parameters:\n{pformat(MODEL_TUNE_KWS)}\n...on precomputed megamat est spike '
        'deltas, varying scales of PN>APL and APL>PN weights in a grid around tuned '
        'values. Initial "tuned" values are chosen at somewhat arbitrary initial offset'
        f' from APL<>KC weight scales.\n\nA directory {repr(OUTPUT_ROOT_NAME)} will be '
        'created in the current path, and model outputs will be stored in '
        'sub-directories within.', formatter_class=RawTextHelpFormatter
    )
    # TODO add (+implement) -c flag to check outputs match existing ones
    # (can i use existing fns / code for that? want subset of behavior al_analysis.py
    # supports with -c/-C)
    parser.add_argument('-i', '--ignore-existing', action='store_true',
        help='re-runs model (and at each parameter step), rather than just doing '
        'downstream analysis on existing saved outputs'
    )
    # TODO provide disk space usage estimate as we proceed through this one?
    parser.add_argument('-d', '--save-dynamics', action='store_true',
        help='saves DataArray pickles of internal model dynamics (in '
        'fit_and_plot_mb_model, via setting fit_mb_model return_dynamics=True)'
    )
    parser.add_argument('-t', '--tuned-only', action='store_true',
        help='only runs the initial tuned version of each model parameters, skipping '
        'all of the stepping of PN>APL and APL>PN weight scales. mainly for testing.'
    )
    parser.add_argument('-o', '--only-analyze-outputs', action='store_true',
        help='skip even checking that all model directories are created. only run '
        'analyze_outputs on model output directories that are immediate children '
        f'of {repr(OUTPUT_ROOT_NAME)}'
    )
    parser.add_argument('-c', '--corners-only', action='store_true',
        help='only analyzes the corners of the sweep, also excluding the tuned values. '
        'for quick tests of extreme behavior.'
    )
    parser.add_argument('-p', '--plot-dynamics', action='store_true',
        help='loads and plots saved dynamics (in the analyze_outputs call, so '
        'works with -o/--only-analyze-outputs)'
    )
    args = parser.parse_args()
    ignore_existing = args.ignore_existing
    save_dynamics = args.save_dynamics
    tuned_only = args.tuned_only
    only_analyze_outputs = args.only_analyze_outputs
    corners_only = args.corners_only
    plot_dynamics = args.plot_dynamics

    if only_analyze_outputs:
        assert not (save_dynamics or tuned_only or ignore_existing), \
            'all of these incompatible with -o/--only-analyze-outputs'

    # TODO is this required to see when we are saving figs (think so)? change so that's
    # not the case (and for saving other things, if necessary)?
    al_util.verbose = True

    # should now be loading the new signed absmax response calc version
    orn_deltas = megamat_orn_deltas()

    curr_dir = Path('.').resolve()
    if curr_dir.name == OUTPUT_ROOT_NAME:
        raise IOError('you probably made a mistake by calling from within '
            f'{OUTPUT_ROOT_NAME}. call from one level above (the directory containing '
            'that directory).'
        )

    # TODO pass in? or define module level?
    # outputs can be big and want to be able to save in arbitrary paths. just run script
    # from the folder you want the outputs in.
    plot_root = curr_dir / OUTPUT_ROOT_NAME

    # TODO or just exclude hardcoded list, so directory names won't change if i add
    # more params to the list (which would change subset that is same across all)?
    same_in_all = set(subset_same_in_all_dicts(MODEL_TUNE_KWS).keys())

    plot_dir2kws: Dict[Path, ParamDict] = dict()
    for kws in MODEL_TUNE_KWS:
        print(f'{kws=}')

        plot_dirname = format_model_params(kws, exclude=same_in_all)
        plot_dir = plot_root / plot_dirname
        assert plot_dir not in plot_dir2kws, f'duplicate {plot_dir=}'
        plot_dir2kws[plot_dir] = kws

        if not only_analyze_outputs:
            step_pn_apl_weights_around_tuned(plot_dir, orn_deltas, kws,
                ignore_existing=ignore_existing, save_dynamics=save_dynamics,
                tuned_only=tuned_only, corners_only=corners_only
            )

    if tuned_only:
        warn('not calling analyze_outputs on any output directory, because '
            '-t/--tuned-only. stepped (subdirectory) outputs may be older than tuned '
            'outputs (out of date).'
        )
        return

    for plot_dir, kws in plot_dir2kws.items():
        # kws not actually used by analyze_outputs, so could just keep a list of
        # plot_dirs...
        analyze_outputs(plot_dir, plot_dynamics=plot_dynamics,
            corners_only=corners_only
        )


if __name__ == '__main__':
    main()

