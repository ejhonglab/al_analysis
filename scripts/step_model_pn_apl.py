#!/usr/bin/env python3
"""
After `pip install`-ing `al_analysis`, you should be able to invoke this script via:
`step_model_pn_apl`, and it will save outputs in current directory (can be quite large,
if saving dynamics via `-d/--save-dynamics` flag).

Does not depend on any input data. Loads precomputed megamat ORN spike delta estimates,
and runs those through models with parameters in `MODEL_TUNE_KWS` (one instantiation per
entry in that list).
"""

import argparse
from itertools import product
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hong2p.viz import matshow
from hong2p.util import symlink, subset_same_in_all_dicts
import al_util
from al_util import savefig, ParamDict
from mb_model import (fit_and_plot_mb_model, megamat_orn_deltas, dict_seq_product,
    format_weights, format_model_params, get_thr_and_APL_weights,
    save_and_remove_from_param_dict, drop_silent_model_cells, glomerulus_col
)


MODEL_TUNE_KWS = dict_seq_product(
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

def analyze_outputs(plot_dir: Path) -> None:
    # TODO doc

    kstr = plot_dir.name
    cols = ['wAPLPN', 'wPNAPL', 'sparsity', 'n_silent_cells', 'hept_pent_corr',
        'avg_lts', 'n_avg_odors_responded_to'
    ]
    vals = []
    for d in list(plot_dir.glob('*/')):
        if not d.is_dir():
            continue

        # all other directories should be named like: 'wAPLPN-0.10_wPNAPL-20.00'
        if d.name in {'model_internals'}:
            continue

        a2p, p2a = d.name.split('_')
        a2p = float(a2p.split('-')[-1])
        p2a = float(p2a.split('-')[-1])
        # TODO use parquet instead?
        rs = pd.read_pickle(d / 'responses.p')
        sp = rs.mean().mean()
        rs_nosilent = drop_silent_model_cells(rs)
        n_silent = len(rs) - len(rs_nosilent)
        # TODO TODO average over corr w/ 1-6ol too? also eb/ep vs 1-5ol/1-6ol block?
        # TODO + maybe ratio/diff w/ corrs in rest of off diag, if needed (but prob not)
        hept_pent_corr = rs_nosilent.corr().loc['1-5ol @ -3', '2h @ -3']

        xs = rs_nosilent
        n_odors = (xs > 0).T.sum()
        avg_n_odors = n_odors.mean()

        # TODO add some assertions verifying this
        xs = xs.T
        L = pow(xs.sum(), 2.0)/(xs*xs).sum()
        L =  (1.0 - L/len(xs))/(1.0 - 1.0/len(xs))
        L = L.fillna(1.0)
        assert len(L) == len(n_odors)

        vals.append((a2p, p2a, sp, n_silent, hept_pent_corr, L.mean(), avg_n_odors))

    df = pd.DataFrame.from_records(vals, columns=cols)
    df = df.set_index(['wAPLPN', 'wPNAPL'], verify_integrity=True)

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

        # TODO fail earlier (w/ better err message) in viz.matshow, if mat is empty.
        # currently get: `ZeroDivisionError: float division by zero`
        #   File "/home/tom/src/hong2p/hong2p/viz.py", line 1741, in matshow
        # fontsize = min(10.0, 240.0 / max(df.shape[0], df.shape[1]))
        matshow(mat, ax=ax, xticklabels=True, yticklabels=True, cbar_label=c)

        # wPNAPL
        ax.set_ylabel(mat.index.name, fontsize=10)
        # wAPLPN
        ax.set_xlabel(mat.columns.name, fontsize=10)

        # TODO include other model params?
        ax.set_title(f'{kstr}\n{c}')

        savefig(fig, plot_root, f'{kstr}__{c}', bbox_inches='tight')


def step_pn_apl_weights_around_tuned(orn_deltas: pd.DataFrame, kws: ParamDict, *,
    ignore_existing: bool = False, save_dynamics: bool = False) -> None:
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
    """
    # outputs can be big and want to be able to save in arbitrary paths. just run script
    # from the folder you want the outputs in.
    plot_root = Path('.').resolve() / OUTPUT_ROOT_NAME

    # TODO or just exclude hardcoded list, so directory names won't change if i add more
    # params to the list (which would change subset that is same across all)?
    same_in_all = set(subset_same_in_all_dicts(MODEL_TUNE_KWS).keys())
    plot_dirname = format_model_params(kws, exclude=same_in_all)
    plot_dir = plot_root / plot_dirname

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

    # TODO delete? this should not really be a meaningful amount of total time now
    # TODO set fixed_thr/wAPLKC/etc instead? (would be faster)
    plot_dirname2sp_lr_coeff = {
        'pn-claw-to-apl_True': 5.90222,
        # TODO TODO add one for *_False
    }
    dirname = plot_dir.name
    # TODO change handling in there so it doesn't matter (and so we can set any of these
    # None to same effect)?
    # NOTE: with how tuned (output) and input params are checked for equality currently
    # in fit_and_plot_mb_model, we need to actually not pass `sp_lr_coeff`, rather than
    # setting it None like we might otherwise do for default
    sp_lr_coeff = None
    if dirname in plot_dirname2sp_lr_coeff:
        # TODO TODO warn if we are using this (/delete?)
        sp_lr_coeff = plot_dirname2sp_lr_coeff[dirname]

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

    # TODO TODO also step around fixed_thr/wAPLKC from previous tuning, e.g. a
    # similar model w/o the PN<>APL weights?
    # TODO try a hypergrid stepping thr and APL independently too? (w/ same steps
    # for PN>APL and APL>PN weights)
    params = fit_and_plot_mb_model(plot_root, plot_dirname=plot_dir.name,
        orn_deltas=orn_deltas, verbose=True, try_cache=not ignore_existing,
        **kws, **output_kws, sp_lr_coeff=sp_lr_coeff,
        return_olfsysm_vars=return_olfsysm_vars, delete_pretime=delete_pretime,
        max_iters=max_iters
    )
    # TODO add option just to reanalyze any saved dynamics, if i factor out that
    # plotting code from fit_mb_model?
    thr_and_apl_kws = get_thr_and_APL_weights(params, kws)
    print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')
    wAPLPN_scale = thr_and_apl_kws['wAPLPN']
    wAPLKC_scale = thr_and_apl_kws['wAPLKC']

    # TODO use parquet instead?
    wPNKC = pd.read_pickle(plot_dir / 'wPNKC.p')

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
        responses = pd.read_pickle(plot_dir / 'responses.p')
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
    steps = [100, 20, 1.0, 0.5, 10, .1]
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

    analyze_outputs(plot_dir)


def main():
    parser = argparse.ArgumentParser('will run models with the following '
        f'parameters:\n{pformat(MODEL_TUNE_KWS)}\n...on precomputed megamat est spike '
        'deltas, varying scales of PN>APL and APL>PN weights in a grid around tuned '
        'values. Initial "tuned" values are chosen at somewhat arbitrary initial offset'
        f' from APL<>KC weight scales.\n\nA directory {repr(OUTPUT_ROOT_NAME)} will be '
        'created in the current path, and model outputs will be stored in '
        'sub-directories within.'
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
    # TODO add flag to reanalyze / plot dynamics (if i factor out the
    # plot_example_dynamics code from fit_mb_model)?
    args = parser.parse_args()
    ignore_existing = args.ignore_existing
    save_dynamics = args.save_dynamics

    # TODO is this required to see when we are saving figs (think so)? change so that's
    # not the case (and for saving other things, if necessary)?
    al_util.verbose = True

    # should now be loading the new signed absmax response calc version
    orn_deltas = megamat_orn_deltas()

    for kws in MODEL_TUNE_KWS:
        print(f'{kws=}')
        step_pn_apl_weights_around_tuned(orn_deltas, kws,
            ignore_existing=ignore_existing, save_dynamics=save_dynamics
        )


if __name__ == '__main__':
    main()

