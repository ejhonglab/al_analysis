#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint, pformat
from typing import Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from al_analysis import al_util
from al_analysis.al_util import warn
from al_analysis.mb_model import (fit_and_plot_mb_model, connectome_wPNKC, ParamDict,
    connectome_APL_weights, read_parquet, megamat_orn_deltas, pd_allclose,
    get_connectome_wPNKC_params, BOUTON_MODEL_KW_LIST, NONCLAW_MODEL_KW_LIST,
    dict_seq_product, check_model_kws_unique, assert_fit_and_plot_outputs_equal
)


def run_tuned_and_multireponsder_apl_boosted(plot_root: Path, orn_deltas: pd.DataFrame,
    name2weights: ParamDict, kws: ParamDict, *, try_cache: bool = True,
    ignore_lr_cache: bool = False, checks: bool = False) -> None:

    # TODO delete. just to regen corr plots w/ different calculation
    kws = dict(kws)
    # TODO work? did i have some other flag to control whether to make plots after
    # loading cache?
    # TODO TODO restore? make CLI flag for this?
    kws['only_return_params'] = False
    #

    try_lr_cache = not ignore_lr_cache

    # TODO TODO re-organize outputs of this script, so the tuned dirs are the only ones
    # at top level, and all multi-responder tweaked versions of the model are
    # subdirectories within each of those?
    # TODO or parallel set of directories for each of the three, with each tree
    # containing leaf dirs/links w/ same name (that of tuned dir)?

    # TODO make module-level set of paths instead?
    output_dirs: Set[str] = set()
    # NOTE: using this rather than _fit_mb_model, so that onestep LR cache can be used
    # (if QUICK=1). not available in fit_mb_model.
    # TODO add CLI flag have this try_cache separate from rest?
    ret = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, try_cache=try_cache,
        try_lr_cache=try_lr_cache, **kws
    )
    output_dir = ret['output_dir']
    output_dirs.add(output_dir)

    # TODO (delete?) use mb_model.get_APL_weights / whatever (since in weight-divisor_20
    # case, and others without connectome APL, just have e.g. wAPLKC=<float>, and no
    # wAPLKC_scale param) (or reimplement here, but would rather not. eh, that other fn
    # doesn't currently return all though, not that it necessary matters [just wAPLKC
    # and wAPLPN, not the scales for the weights in the other direction, which it
    # assumes are derivable from others])
    scale_param_names = tuple(
        k for k in ret.keys() if k.startswith('w') and k.endswith('_scale')
    )
    if len(scale_param_names) == 0:
        scale_param_names = tuple(
            k for k in ret.keys() if k.startswith('w') and 'APL' in k
        )
    # TODO more than that? assert some minimum set?
    assert len(scale_param_names) >= 2, f'only had {scale_param_names=}'

    # this is only case where fit_mb_model currently scales weights to mean of 1 (which
    # does NOT currently happen in the `_wAPLKC is not None` case, so we need to
    # replicate that here)
    one_row_per_claw = kws.get('one_row_per_claw', False)

    # will need to replace values for weight vectors with all ones here
    use_connectome_APL_weights = kws.get('use_connectome_APL_weights', False)

    if not use_connectome_APL_weights:
        assert not one_row_per_claw, ('have not tested/thought about whether this case '
            'would be correct without modifications'
        )
        name2uniform_weights = dict()
        for name, weights in name2weights.items():
            if weights is None:
                name2uniform_weights[name] = None
                continue

            weights = weights.copy()
            weights.values[:] = 1
            name2uniform_weights[name] = weights

        name2weights = name2uniform_weights

    scaled_series_weight_kws = dict()
    normed_not_scaled_series_weight_kws = dict()
    for k, scale in ret.items():
        if k not in scale_param_names:
            continue

        assert isinstance(scale, float), \
            f'expected type float for {k=}. got {type(scale)=}'

        if '_' in k:
            # e.g. 'wAPLKC_scale' -> 'wAPLKC'
            n = k.split('_')[0]
        else:
            n = k

        normed_but_not_tuning_scaled = name2weights[n]
        # TODO is this the case (and the only one) that is not already scaled
        # to mean of 1 before fitting starts (seems so)? fix that?
        # TODO if fixed_inh_params* tests are currently passing in the
        # weight-divisor_20__connectome-APL_True case, then maybe it shouldn't be scaled
        # first? (no, probably should be, but not sure whether i can easily add a test
        # in here that we can recreate same output from this?)
        if not one_row_per_claw:
            # see comment by one_row_per_claw def above for why we need to do this here
            normed_but_not_tuning_scaled = (
                normed_but_not_tuning_scaled / normed_but_not_tuning_scaled.mean()
            )

        weight_parquet = plot_root / output_dir / f'{n}.parquet'
        if weight_parquet.exists():
            scaled = read_parquet(weight_parquet)

            # TODO assert here that normed_but_not_tuning_scaled has mean close to 1
            # now? (now that i've fixed the one_row_per_claw case, but duplicating what
            # fit_mb_model currently does in that case)
            assert pd_allclose(normed_but_not_tuning_scaled * scale, scaled), f'{k=}'
        else:
            assert not use_connectome_APL_weights, \
                f'{weight_parquet} should exist otherwise'
            # TODO this what i want?
            scaled = normed_but_not_tuning_scaled * scale

        scaled_series_weight_kws[f'_{n}'] = scaled
        # TODO avoid need for this? or do it internally somewhere?
        # TODO (delete?) and how should it interact w/ (TBD) flag(s) to control whether
        # tuning happens or not?
        scaled_series_weight_kws[n] = 1.0

        normed_not_scaled_series_weight_kws[f'_{n}'] = normed_but_not_tuning_scaled
        # the scaling factors are expected to be passed in as e.g. `wAPLKC=<float>`,
        # NOT as `wAPLKC_scale=<float>`
        normed_not_scaled_series_weight_kws[n] = scale

    # this is required if we are hardcoding weights (would have to tune otherwise, but
    # would actually just raise an error before that)
    fixed_thr = ret['fixed_thr']

    for_model_id = ''
    if not use_connectome_APL_weights:
        kws = dict(kws)
        # necessary for calls below to work. see other comments about this above.
        # NOTE: do currently need to set use_connectome_APL_weights=True just to be
        # able to pass the weights as Series behind _<weight>, but that's fine. It
        # won't actually be recomputing connnectome APL weights, since these Series
        # weights will be passed in
        kws['use_connectome_APL_weights'] = True

        # hack to be clear (for any caches/whatever that include model ID), that it's
        # not *actually* connectome APL, even though we currently have to set
        # use_connectome_APL_weights=True (as easiest way to get this path to work,
        # despite the fact that the input weight vectors have had their values replaced
        # with a constant value [all 1, or scaled from there])
        for_model_id = '_UNIFORM-APL'

    if checks:
        if not use_connectome_APL_weights:
            # TODO TODO write temporary code to delete the directories these made (or
            # manually delete them...), and change code to (in the future) write them to
            # a temp dir

            # TODO check that:
            # AssertionError: param_dir=PosixPath('megamat_multiresponder_apl_boost/
            # wAPLKC-multiresponder-pretune-10x_scale-pre-tuning_True__weight-divisor_20
            # __connectome-APL_True__fixed-thr_279__wAPLKC_1.00') already seen!
            # was only because i had not added the param_dir_prefix= kwargs below (and
            # that the dupe was only between the two calls within this block)

            # TODO move these to tests? nice to have here too tho... (dupe to test(s)
            # instead probably)
            ret0 = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
                fixed_thr=fixed_thr, try_cache=try_cache, try_lr_cache=try_lr_cache,
                **kws, **normed_not_scaled_series_weight_kws,
                param_dir_prefix=f'check-normed_'
            )
            exclude_params = ('use_connectome_APL_weights', 'wAPLKC_scale',
                'wKCAPL_scale', 'wAPLKC', 'wKCAPL'
            )
            stems_to_ignore = {'wAPLKC', 'wKCAPL'}
            assert_fit_and_plot_outputs_equal(plot_root, ret, ret0,
                ignore_tuning_params=True, file_stems_to_ignore=stems_to_ignore,
                exclude_params=exclude_params
            )

            ret1 = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
                fixed_thr=fixed_thr, try_cache=try_cache, try_lr_cache=try_lr_cache,
                **kws, **scaled_series_weight_kws, param_dir_prefix=f'check-scaled_'
            )
            assert_fit_and_plot_outputs_equal(plot_root, ret, ret1,
                ignore_tuning_params=True, file_stems_to_ignore=stems_to_ignore,
                exclude_params=exclude_params
            )
    else:
        warn('checks disabled! pass -C/--checks CLI arg to enable (which should set '
            'checks=True)'
        )

    # TODO TODO loop over a few boost factors?
    boost = 10
    #boost = 5
    #boost = 100

    # TODO also try adding some small offset to all APL weights of them, since many
    # might be 0? that's probably doing too much...

    tune_dir = plot_root / output_dir
    assert tune_dir.is_dir()
    spike_counts = read_parquet(tune_dir / 'spike_counts.parquet')

    n_spikes_for_response = kws.get('n_spikes_for_response', 1)
    # TODO refactor to share this calc w/ mb_model (/test code)? i assume at least one
    # of those uses similar code for some check at least?
    n_odors = (spike_counts > (n_spikes_for_response - 1)).T.sum()

    # TODO TODO TODO try scaling proportional to # odors, instead of just a flat scale?
    # TODO TODO and/or proportional to # of spikes? how to weight?
    multiresponders = n_odors >= 2
    multiresponder_index = multiresponders[multiresponders].index.droplevel('kc_type')

    pretune_apl_boost_kws = dict(normed_not_scaled_series_weight_kws)
    for k, v in name2weights.items():
        if v is None:
            continue
        # these scales would have been the ones post-tuning (from ret), but we want to
        # start them around 1.0 for pre-tuning scaling
        # TODO check that =None does same thing? (would require slight changes in
        # mb_model)
        # TODO remove need to explicitly specify this, and have it all implied when
        # passing in _<n> weight vectors?
        pretune_apl_boost_kws[k] = 1.0

    # TODO TODO TODO plot similar stats as step_model_pn_apl_weights.py? maybe in
    # particular that average elevated off-diagonal correlation, ideally against uniform
    # model and remy's (w/ error bar, or quantiles, on remy's)
    # TODO TODO refactor to share w/ that script

    # TODO TODO TODO explicitly compute / plot reduction in correlation (the off
    # diagonal one at least?)
    # TODO precompute and save corrs so i can realize from those easily?
    # TODO also compare to remy KC corr? compute + commit latest version of that (+
    # provide al_analysis fn to load it)?

    # TODO TODO TODO compare to effects of multiresponder_APL_boost w/
    # _multiresponder_mask defined from this. can i reproduce
    # 2025-07-31_apl_boosted_within_multiresponders_2 slides? and that was pre-tuning,
    # right?
    # TODO TODO + see i can repro those plots (with whichever code paths? both?)
    _wAPLKC = pretune_apl_boost_kws['_wAPLKC'].copy()
    _wAPLKC.loc[multiresponder_index] *= boost
    pretune_apl_boost_kws['_wAPLKC'] = _wAPLKC
    plot_dirname = f'wAPLKC-multiresponder-pretune-{boost:.0f}x'
    # are these never using LR cache? fix, if true (oh, cause i was explicitly
    # specifying sp_lr_coeff=1.2 here...)
    # TODO can i still save a value to cache despite having it passed in? it's not
    # dependent on initial sp_lr_coeff, right? (or at least check i already am. may be.
    # doesn't seem so?)
    ret2 = fit_and_plot_mb_model(tune_dir, plot_dirname=plot_dirname,
        orn_deltas=orn_deltas, try_cache=try_cache, try_lr_cache=try_lr_cache,
        fixed_thr=fixed_thr, scale_pre_tuning=True, max_iters=200,
        # NOTE: need to specify this separately from plot_dirname, since it will be
        # included in model ID genereted from parameters (and thus will be used in some
        # caches, etc, like onestep_lr_cache)
        param_dir_prefix=f'{plot_dirname}{for_model_id}_',
        **pretune_apl_boost_kws, **kws
    )

    # TODO TODO TODO plot clustered responses in a way that tracks the same cells from
    # tuning and the calls below (use fixed order, from clustering on tuning) (factor
    # out any fns for this? already have code for it in natmix_data/analysis.py?)
    # TODO TODO + plots like in 2025-07-31_apl_boosted_within_multiresponders_2, where
    # we can see the conversions on a cluster[/class] level?
    # TODO TODO TODO and plot spike counts (log color scale?) instead of binarized
    # "responses" anyway

    posttune_apl_boost_kws = dict(scaled_series_weight_kws)
    _wAPLKC = posttune_apl_boost_kws['_wAPLKC'].copy()
    _wAPLKC.loc[multiresponder_index] *= boost
    posttune_apl_boost_kws['_wAPLKC'] = _wAPLKC
    plot_dirname = f'wAPLKC-multiresponder-posttune-{boost:.0f}x'
    ret3 = fit_and_plot_mb_model(tune_dir, plot_dirname=plot_dirname,
        orn_deltas=orn_deltas, try_cache=try_cache, try_lr_cache=try_lr_cache,
        fixed_thr=fixed_thr, param_dir_prefix=f'{plot_dirname}{for_model_id}_',
        **posttune_apl_boost_kws, **kws
    )

    # TODO TODO actually do anything here? check sparsity change? other plotting?
    # TODO delete
    #breakpoint()

    print()


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--use-cache', action='store_true', help='uses cache for '
        'all model outputs, when available. this may mean no models are re-run.'
    )
    parser.add_argument('-r', '--ignore-lr-cache', action='store_true', help='uses no '
        'values in any cache of learning rates (e.g. sp_lr_coeff) (but will still write'
        ' to them in the same circumstances as normal)'
    )
    parser.add_argument('-s', '--skip-first-n', action='store', default=0, type=int,
        help='skips the first N model parameter combinations (as seen in progress bar)'
    )
    parser.add_argument('-C', '--checks', action='store_true', help='enables checks')
    args = parser.parse_args()
    use_cache = args.use_cache
    ignore_lr_cache = args.ignore_lr_cache
    skip_first_n = args.skip_first_n
    checks = args.checks
    assert skip_first_n >= 0

    plot_root = Path('megamat_multiresponder_apl_boost')
    plot_root.mkdir(exist_ok=True)

    # TODO still needed? thought i saw some output of plots being saved (well, not
    # seeing now, when i tried removing)?
    # (here and in other scripts that set it just for that reason)
    al_util.verbose = True

    orn_deltas = megamat_orn_deltas()

    # should be the case that all the weights are the same for all entries within each
    # of these lists
    kw_lists_sharing_weights = [BOUTON_MODEL_KW_LIST, NONCLAW_MODEL_KW_LIST]

    for i, xs in enumerate(kw_lists_sharing_weights):
        wPNKC_param_set = set()
        for kws in xs:
            wPNKC_params = get_connectome_wPNKC_params(kws)
            # TODO need to also assert that kws is only missing
            # use_connectome_APL_weights=True for cases where we also know the
            # non-connectome APL implementation works here? (going to implement support
            # for non-claw case first, and maybe only ever that, so
            # one_row_per_claw=True should maybe cause failure if any non-connectome
            # APL?)
            assert 'use_connectome_APL_weights' not in wPNKC_params
            wPNKC_param_set.add(frozenset(tuple(x) for x in wPNKC_params.items()))
        assert len(wPNKC_param_set) == 1, (f'at index {i} in kw_lists_sharing_weights:'
            f'\n{pformat(wPNKC_param_set)}'
        )

    # TODO TODO refactor to share w/ step_model_pn_apl.py?
    # TODO TODO also try claw_dynamics (w/ diff time constants?)? diff APL time
    # constants?
    try_each_with = dict_seq_product(
        [dict(), dict(target_sparsity=0.05)],
        [dict(), dict(n_spikes_for_response=2)]
    )
    all_model_kw_list = []
    for kw_list in kw_lists_sharing_weights:
        all_model_kw_list.extend(dict_seq_product(kw_list, try_each_with))
    check_model_kws_unique(all_model_kw_list)

    print('will run tuned and APL boosted versions of the following model '
        'instantiations:'
    )
    pprint(all_model_kw_list)
    assert skip_first_n < len(all_model_kw_list), f'{len(all_model_kw_list)=}'

    pbar = tqdm(unit='model-params', total=len(all_model_kw_list))

    print(f'{skip_first_n=}')
    n_to_skip = skip_first_n
    print(f'{n_to_skip=}')
    for kw_list in kw_lists_sharing_weights:
        kws = kw_list[0]

        wPNKC_params = get_connectome_wPNKC_params(kws)
        # NOTE: one_row_per_claw is not an argument to connectome_wPNKC, nor is
        # use_connectome_APL_weights
        # TODO change how this adds a connectome='hemibrain' we don't have as input?
        # why can't i just let that be handled by default? (any other uses currently
        # depend on that? just check no tests fail after removing [once less tests
        # in general are failing...])
        #assert all(x in kws for x in wPNKC_params)
        wPNKC = connectome_wPNKC(**wPNKC_params)
        # TODO delete
        print(f'{wPNKC_params=}')
        #

        wPNKC_only_params = {'weight_divisor'}
        apl_weight_params = {
            k: v for k, v in wPNKC_params.items() if k not in wPNKC_only_params
        }
        weights = connectome_APL_weights(wPNKC=wPNKC, **apl_weight_params)
        # TODO delete
        # True as of 2026-04-14 (and maybe not always after), when scaling (in
        # python) each weight vector by `len(weights) / weights.sum()`
        #assert all(np.isclose(ws.mean(), 1) for ws in weights if ws is not None), (
        #    'still scaling (in python) each weight vector to mean of 1?'
        #)
        # TODO update connectome_APL_weights to also scale in the
        # one_row_per_claw=False case, then restore assertion here (+ remove
        # separate handling for that in run_tuned_...)?
        if not all(np.isclose(ws.mean(), 1) for ws in weights if ws is not None):
            # at least (and maybe only) default cases (like wd20) will hit this
            warn('some/all APL weight vectors had mean not close to 1. '
                'may be fine.'
            )

        # not yet scaled by tuning in olfsysm, just "normalization" in python
        wAPLKC, wKCAPL, wAPLPN, wPNAPL = weights

        name2weights = {
            'wAPLKC': wAPLKC,
            'wKCAPL': wKCAPL,
            'wAPLPN': wAPLPN,
            'wPNAPL': wPNAPL,
        }
        for kws in dict_seq_product(kw_list, try_each_with):
            if n_to_skip > 0:
                n_to_skip -= 1
                # TODO delete warning?
                warn(f'skipping kws:\n{pformat(kws)}\n...because {skip_first_n=}')
                pbar.update()
                if n_to_skip == 0:
                    # wasn't displaying without this
                    pbar.refresh()
                continue

            run_tuned_and_multireponsder_apl_boosted(plot_root, orn_deltas,
                name2weights, kws, try_cache=use_cache, ignore_lr_cache=ignore_lr_cache,
                checks=checks
            )
            pbar.update()

    pbar.close()


if __name__ == '__main__':
    main()

