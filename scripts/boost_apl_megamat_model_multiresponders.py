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
    dict_seq_product, check_model_kws_unique
)


def run_tuned_and_multireponsder_apl_boosted(plot_root: Path, orn_deltas: pd.DataFrame,
    name2weights: ParamDict, kws: ParamDict, *, try_cache: bool = True) -> None:

    # TODO delete. just to regen corr plots w/ different calculation
    kws = dict(kws)
    # TODO work? did i have some other flag to control whether to make plots after
    # loading cache?
    # TODO TODO restore? make CLI flag for this?
    kws['only_return_params'] = False
    #

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
        **kws
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

    use_connectome_APL_weights = kws.get('use_connectome_APL_weights', False)

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

        # TODO TODO TODO handle case where this doesn't exist (non-connectome APL cases)
        # (either add test that using all 1s [for non-scaled] gives us same output, and
        # then construct the input weights that way, or skip this case)
        # TODO TODO or change so we do always pass the connectome APL weights in, but
        # set values to all ones for non-connectome-APL cases in here?
        weight_parquet = plot_root / output_dir / f'{n}.parquet'
        if weight_parquet.exists():
            scaled = read_parquet(weight_parquet)
            # TODO TODO also handle how name2weights will be all None here
            # (or change calling code so they won't be...)
        else:
            assert not use_connectome_APL_weights, \
                f'{weight_parquet} should exist otherwise'

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
        # TODO assert here that normed_but_not_tuning_scaled has mean close to 1
        # now? (now that i've fixed the one_row_per_claw case, but duplicating what
        # fit_mb_model currently does in that case)

        assert pd_allclose(normed_but_not_tuning_scaled * scale, scaled), f'{k=}'

        scaled_series_weight_kws[f'_{n}'] = scaled
        # TODO TODO avoid need for this? or do it internally somewhere?
        # TODO TODO and how should it interact w/ (TBD) flag(s) to control whether
        # tuning happens or not?
        scaled_series_weight_kws[n] = 1.0

        normed_not_scaled_series_weight_kws[f'_{n}'] = normed_but_not_tuning_scaled
        # the scaling factors are expected to be passed in as e.g. `wAPLKC=<float>`,
        # NOT as `wAPLKC_scale=<float>`
        normed_not_scaled_series_weight_kws[n] = scale

    # this is required if we are hardcoding weights (would have to tune otherwise, but
    # would actually just raise an error before that)
    fixed_thr = ret['fixed_thr']

    # TODO TODO loop over a few boost factors?
    boost = 10
    #boost = 5
    #boost = 100
    # TODO TODO also try adding some small offset to all of them, since many might be 0?
    # that's probably doing too much...
    # TODO TODO TODO maybe try something similar in a non-per-claw version tho (so that
    # there would be less 0 entries, since counted per KC instead of per claw)

    spike_counts = read_parquet(plot_root / output_dir / 'spike_counts.parquet')
    n_odors = (spike_counts > 0).T.sum()
    # TODO TODO try scaling proportional to # odors, instead of just a flat scale?
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

    # TODO TODO TODO compare to effects of multiresponder_APL_boost w/
    # _multiresponder_mask defined from this. can i reproduce
    # 2025-07-31_apl_boosted_within_multiresponders_2 slides? and that was pre-tuning,
    # right?
    # TODO TODO + see i can repro those plots (with whichever code paths? both?)
    _wAPLKC = pretune_apl_boost_kws['_wAPLKC'].copy()
    _wAPLKC.loc[multiresponder_index] *= boost
    pretune_apl_boost_kws['_wAPLKC'] = _wAPLKC
    ret2 = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, try_cache=try_cache,
        fixed_thr=fixed_thr, scale_pre_tuning=True,
        # max_iters=200 might work with this 4.5 (oh, nvm, the almost successfull 100
        # iter run started w/ sp_lr_coeff=1.5 i think). worked w/ 1.2 & max_iters=200
        sp_lr_coeff=1.2,
        max_iters=200,
        param_dir_prefix=f'wAPLKC-multiresponder-pretune-{boost:.0f}x_',
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
    ret3 = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, try_cache=try_cache,
        fixed_thr=fixed_thr,
        param_dir_prefix=f'wAPLKC-multiresponder-posttune-{boost:.0f}x_',
        **posttune_apl_boost_kws, **kws
    )
    # TODO TODO actually do anything here? check sparsity change? other plotting?
    # TODO delete
    #breakpoint()


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--use-cache', action='store_true', help='uses cache for '
        'all model outputs, when available. this may mean no models are re-run.'
    )
    parser.add_argument('-s', '--skip-first-n', action='store', default=0, type=int,
        help='skips the first N model parameter combinations (as seen in progress bar)'
    )
    args = parser.parse_args()
    use_cache = args.use_cache
    skip_first_n = args.skip_first_n
    assert skip_first_n > 0

    plot_root = Path('megamat_multiresponder_apl_boost')
    plot_root.mkdir(exist_ok=True)

    # TODO still needed? thought i saw some output of plots being saved (well, not
    # seeing now, when i tried removing)?
    # (here and in other scripts that set it just for that reason)
    al_util.verbose = True

    orn_deltas = megamat_orn_deltas()

    # should be the case that all the weights are the same for all entries within each
    # of these lists
    kw_lists_sharing_weights = [BOUTON_MODEL_KW_LIST]

    # NONCLAW* don't both have use_connectome_apl_weights=True, so they don't both share
    # weights
    nonclaw_connectome_apl = []
    nonclaw_uniform_apl = []
    # TODO TODO TODO maybe don't split apart, so i can use the index of the connectome
    # APL weight version for the all-1s vector for the non-connectome one? otherwise
    # what? pass separately?
    # TODO TODO and if doing above, add a test inside run_... that does another tuning
    # call removing all the weight vectors, and checks outputs are the same (or fixed
    # APL weight call, w/ same scales? either might work?)
    # TODO or enforce that all non-connectome APL have a corresponding connectome
    # APL counterpart (all other params same), and that all the non-connectome ones go
    # after (so we can load the saved weights from the earlier ones)?
    for x in NONCLAW_MODEL_KW_LIST:
        if x.get('use_connectome_APL_weights', False):
            nonclaw_connectome_apl.append(x)
        else:
            nonclaw_uniform_apl.append(x)

    kw_lists_sharing_weights.extend([nonclaw_connectome_apl, nonclaw_uniform_apl])

    # TODO TODO replace this w/ something that checks all wPNKC_params/apl_weight_params
    # EXCEPT use_connectome_APL_weights are the same between all elements in the list?
    for i, xs in enumerate(kw_lists_sharing_weights):
        connectome_apl = {x.get('use_connectome_APL_weights', False) for x in xs}
        assert len(connectome_apl) == 1, (f'at index {i} in kw_lists_sharing_weights, '
            f'{connectome_apl=} had more than one unique value. items in this kw_list '
            'can not share weights'
        )

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
        # TODO delete (after figuring out how to handle weights/name2weights stuff below
        # [and code using it in run_tuned...] in this case)
        #if not kws.get('use_connectome_APL_weights'):
        #    warn('skipping non-connectome APL weights cases for now. need to change '
        #        'downstream code to support in this script.'
        #    )
        #    pbar.update(len(dict_seq_product(kw_list, try_each_with)))
        #    continue
        #assert kws['use_connectome_APL_weights']
        #
        if kws.get('use_connectome_APL_weights'):
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
        else:
            wAPLKC = None
            wKCAPL = None
            wAPLPN = None
            wPNAPL = None

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
                name2weights, kws, try_cache=use_cache
            )
            pbar.update()

    pbar.close()


if __name__ == '__main__':
    main()

