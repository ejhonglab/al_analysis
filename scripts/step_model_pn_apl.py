#!/usr/bin/env python3

import argparse
from itertools import product
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hong2p.viz import matshow
from hong2p.util import symlink
import al_util
from al_util import savefig
from mb_model import (fit_and_plot_mb_model, megamat_orn_deltas, dict_seq_product,
    format_weights, format_model_params, get_thr_and_APL_weights,
    save_and_remove_from_param_dict, drop_silent_model_cells, glomerulus_col
)


model_tune_kws = dict_seq_product(
    [
        dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
            use_connectome_APL_weights=True
        )
    ],
    # TODO switch order (maybe not until figuring out latest issues?)?
    [dict(pn_claw_to_APL=True), dict()]
)


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


def step_pn_apl_weights_around_tuned(orn_deltas, kws, *, ignore_existing: bool = False
    ) -> None:
    # TODO doc

    # TODO move this path creation to script root?
    plot_root = Path('.').resolve() / 'PNAPL_stepping'

    # TODO TODO other params in here? just format all params?
    # (won't be able to use my own dir name here, unless i add support for hardcoding
    # dir name [or list of params to use for formatting] in fit_and_plot..., now that
    # i'm using  that instead of fit_mb_model. i did add plot_dirname for that.)
    # TODO TODO all except exclude list of expected (+add format_model_params kwarg for
    # that)? just hardcode in the expected kws here, rather than even having them passed
    # in?
    plot_dir = plot_root / format_model_params({
        'pn_claw_to_APL': kws.get('pn_claw_to_APL', 'False')
    })

    output_kws = dict(
        # TODO delete (or restore + actually use these. ran out of space on device b/c i
        # was saving them)
        #return_dynamics=True,
        #
        # TODO delete make_plots=True? just make sure plot_example_dynamics sets
        # that? or do i really want (/have) a flag controlling plots other than
        # plot_example_dynamics (i.e. internal corrs)? rename, if that's what
        # it's for?
        plot_example_dynamics=True, make_plots=True, connectome_weight_plots=False
    )

    # TODO set fixed_thr/wAPLKC/etc instead? (would be faster)
    plot_dirname2sp_lr_coeff = {
        # TODO TODO TODO add one for *_False
        'pn-claw-to-APL_True': 5.96433,
    }
    dirname = plot_dir.name
    # TODO change handling in there so it doesn't matter (and so we can set any of these
    # None to same effect)?
    # NOTE: with how tuned (output) and input params are checked for equality currently
    # in fit_and_plot_mb_model, we need to actually not pass `sp_lr_coeff`, rather than
    # setting it None like we might otherwise do for default
    sp_lr_coeff = None
    if dirname in plot_dirname2sp_lr_coeff:
        # TODO TODO warn if we are using this? (/delete?)
        sp_lr_coeff = plot_dirname2sp_lr_coeff[dirname]

    # TODO add fit_mb_model option to assert output is still within target sparsity,
    # when passing in fixed_thr and wAPLKC (to check we are still within what would
    # converge tuning, even if skipping it)? (-> use here, if not tuning)
    # TODO or otherwise assert in here that if we re-run calls from scratch, we get
    # whatever hardcoding parameters we might sometimes use to skip tuning (within
    # tolerance)
    # TODO delete (were same parameters from one tuning output, hardcoded for speed)
    '''
    thr_and_apl_kws = {'fixed_thr': 207.42859388292763, 'wAPLKC': 1.93701}
    wAPLPN_scale = 1.93701
    wPNAPL_scale = 0.00497946
    wAPLKC_scale = 1.93701
    wKCAPL_scale = 0.00111837
    '''
    # TODO TODO refactor to share test precalc_weights=True code w/ this (if i get that
    # working). would skip a lot of time spent in calls below

    # TODO try to get this False after getting code recalculating below working
    # (should be fine now)
    # TODO restore
    #return_olfsysm_vars = False
    # TODO delete eventually
    return_olfsysm_vars = True
    #
    delete_pretime = True

    # TODO delete
    #if ignore_existing or not plot_dir.exists():
    #    plot_dir.mkdir(exist_ok=True, parents=True)
    #    ret = fit_mb_model(orn_deltas=orn_deltas, **kws, **output_kws,
    #        # TODO TODO try to get this False after getting code recalculating below
    #        # working (so delete_pretime is not set =False inside fit_mb_model)
    #        plot_dir=plot_dir, return_olfsysm_vars=return_olfsysm_vars,
    #        delete_pretime=delete_pretime, verbose=True, sp_lr_coeff=sp_lr_coeff
    #    )
    #    # TODO why remove=False? going to analyze here? delete?
    #    # TODO just call fit_and_plot... instead? (want to be able to load
    #    # parameters and wPNKC...)
    #    save_and_remove_from_param_dict(params, plot_dir, remove=False)
    #    params = ret[-1]
    #    wPNKC = ret[-2]
    #else:
    #    #wPNAPL_scale =
    #    breakpoint()
    #    assert 'rv' not in params
    #

    params = fit_and_plot_mb_model(plot_root, plot_dirname=plot_dir.name,
        orn_deltas=orn_deltas, verbose=True, try_cache=not ignore_existing,
        **kws, **output_kws, sp_lr_coeff=sp_lr_coeff,
        return_olfsysm_vars=return_olfsysm_vars, delete_pretime=delete_pretime
    )
    thr_and_apl_kws = get_thr_and_APL_weights(params, kws)
    print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')
    wAPLPN_scale = thr_and_apl_kws['wAPLPN']
    # TODO use / delete
    wAPLKC_scale = thr_and_apl_kws['wAPLKC']
    #
    # TODO delete
    #breakpoint()
    #

    # TODO TODO why are we still getting killed below? thought i wasn't saving
    # claw_sims? is it something else? just low free memory now?
    # (maybe issue is solved now?)

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
        # TODO delete
        #responses = ret[0]
        responses = pd.read_pickle(plot_dir / 'responses.p')
        n_kcs = mp.kc.N
        assert n_kcs == len(responses)
        # TODO use (/delete)
        wKCAPL_scale = rv.kc.wKCAPL_scale
        #
        # NOTE: this one may change to have n_claws as denominator, if I change all
        # handling to be consistent eventually
        assert np.isclose(wAPLKC_scale / n_kcs, wKCAPL_scale)
        #
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
    # TODO TODO TODO worth trying w/ change in how thr is calculated, so it's not
    # relative to spont in? (how to even do? what's that look like w/ other things
    # same?)
    # TODO TODO TODO worth trying w/ a couple diff sp_factor_pre_APL? (1.5 / 3.0?)
    steps = [100, 20, 1.0, 0.5, 10, .1]
    for ap, pa in tqdm(list(product(steps, steps)), unit='param-combo'):
        # TODO or just include, to make some plotting that includes this easier?
        # TODO TODO save plots/dynamics from previous call, and link to that dir?
        #if ap == 1 and pa == 1:
        #    continue

        step = dict(thr_and_apl_kws)
        step['wAPLPN'] = wAPLPN_scale * ap
        # TODO TODO make sure this one is also in format_model_params output,
        # esp when not derivable from wAPLPN (not using format_model_params for now)
        step['wPNAPL'] = wPNAPL_scale * pa
        # TODO remove part about wAPLKC?
        #param_dir = plot_dir / format_model_params(step)

        # TODO TODO TODO TODO actually try per bouton inh dynamics (prob both for KC
        # and PN)

        param_dir = plot_dir / (
            # TODO factor ', ' stripping into option for format_weights (/
            # another fn?)
            # with orig values scaled, could get duplicate plot dir names, b/c some
            # values too small for .3f float format
            # TODO change float formatting in format_weights to fix that
            #format_weights(step['wAPLPN'], 'wAPLPN').strip(', ') + '_' +
            #format_weights(step['wPNAPL'], 'wPNAPL').strip(', ')
            format_weights(ap, 'wAPLPN').strip(', ') + '_' +
            format_weights(pa, 'wPNAPL').strip(', ')
        ).replace('=', '-')

        # TODO TODO how were these dirs (as originally created, w/ commented
        # fit_mb_model code) writing responses.p / spike_counts.p?
        # were those outputs in param dict returned (or internally, at points
        # save_and_remove... was called)? they shouldn't have been, right?
        #
        # TODO delete
        #if ignore_existing or not param_dir.exists():
        #    param_dir.mkdir(exist_ok=True, parents=True)
        #    # TODO delete?
        #    print()
        #    print()
        #    print(f'{ap=}')
        #    print(f'{pa=}')
        #    print()
        #    #
        #    # TODO delete
        #    #curr_ret = fit_mb_model(orn_deltas=orn_deltas,
        #    #    # TODO use same (simpler) syntax as above for multiple kwarg dicts?
        #    #    **{**step, **kws, **output_kws}, plot_dir=param_dir
        #    #)
        #    #rs, ss, _, ps = curr_ret
        #    ## TODO why remove=False? going to analyze here? delete?
        #    ## TODO just call fit_and_plot... instead? (now also needing to save
        #    ## responses / spike_counts separately...)
        #    #save_and_remove_from_param_dict(ps, param_dir, remove=False)
        #    #
        #

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

        # TODO delete
        #print()

    analyze_outputs(plot_dir)


def main():
    parser = argparse.ArgumentParser()
    # TODO TODO add (+implement) -c flag to check outputs match existing ones
    parser.add_argument('-i', '--ignore-existing', action='store_true',
        help='re-runs model (and at each parameter step), rather than using existing '
        'saved  outputs'
    )
    args = parser.parse_args()
    ignore_existing = args.ignore_existing

    # TODO is this required to see when we are saving figs (think so)? change so that's
    # not the case (and for saving other things, if necessary)?
    al_util.verbose = True

    orn_deltas = megamat_orn_deltas()
    for kws in model_tune_kws:
        print(f'{kws=}')
        step_pn_apl_weights_around_tuned(orn_deltas, kws,
            ignore_existing=ignore_existing
        )


if __name__ == '__main__':
    main()

