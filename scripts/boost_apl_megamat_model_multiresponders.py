#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from al_analysis.mb_model import (fit_and_plot_mb_model, connectome_wPNKC, ParamDict,
    connectome_APL_weight, read_parquet, megamat_orn_deltas, BOUTON_MODEL_KW_LIST
)


def run_tuned_and_multireponsder_apl_boosted(plot_root: Path, name2weights: ParamDict,
    orn_deltas: pd.DataFrame, kws: ParamDict) -> None:

    # TODO make module-level set of paths instead?
    output_dirs: Set[str] = set()
    # NOTE: using this rather than _fit_mb_model, so that onestep LR cache can be used
    # (if QUICK=1). not available in fit_mb_model.
    # TODO hardcode try_cache=False?
    ret = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)
    output_dir = ret['output_dir']
    output_dirs.add(output_dir)

    scale_param_names = tuple(k for k in ret.keys()
        if k.startswith('w') and k.endswith('_scale')
    )

    scaled_series_weight_kws = dict()
    normed_not_scaled_series_weight_kws = dict()
    for k, scale in ret.items():
        if k not in scale_param_names:
            continue

        assert type(scale) is float, \
            f'expected type float for {k=}. got {type(scaled)=}'

        # e.g. 'wAPLKC_scale' -> 'wAPLKC'
        n = k.split('_')[0]
        scaled = read_parquet(plot_root / output_dir / f'{n}.parquet')
        normed_but_not_tuning_scaled = name2weights[n]
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

    # TODO TODO TODO do another version using normed-but-not-scaled (for pre-tuning
    # scaling?) (or does my implementation require the other ones, for some reason?)

    # TODO TODO TODO plot clustered responses in a way that tracks the same cells from
    # tuning and the calls below (use fixed order, from clustering on tuning) (factor
    # out any fns for this? already have code for it in natmix_data/analysis.py?)

    apl_boost_kws = dict(scaled_series_weight_kws)
    rs = read_parquet(plot_root / output_dir / 'responses.parquet')
    n_odors = (rs > 0).T.sum()
    # TODO TODO try scaling proportional to # odors, instead of just a flat scale?
    multiresponders = n_odors >= 2
    _wAPLKC = apl_boost_kws['_wAPLKC'].copy()
    _wAPLKC.loc[multiresponders[multiresponders].index.droplevel('kc_type')] *= 20
    apl_boost_kws['_wAPLKC'] = _wAPLKC
    ret3 = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, fixed_thr=fixed_thr,
        # TODO TODO format factor into param_dir_prefix
        param_dir_prefix='wAPLKC-multiresponder-20x_', **apl_boost_kws, **kws
    )
    # TODO delete
    breakpoint()


def main():
    plot_root = Path('megamat_multiresponder_apl_boost')
    plot_root.mkdir(exist_ok=True)

    orn_deltas = megamat_orn_deltas()

    wPNKC_params = get_connectome_wPNKC_params(kws)
    # NOTE: one_row_per_claw is not an argument to connectome_wPNKC, nor is
    # use_connectome_APL_weights
    # TODO change how this adds a connectome='hemibrain' we don't have as
    # input? why can't i just let that be handled by default? (any other uses
    # currently depend on that? just check no tests fail after removing [once less
    # tests in general are failing...])
    #assert all(x in kws for x in wPNKC_params)
    wPNKC = connectome_wPNKC(**wPNKC_params)

    assert kws['use_connectome_APL_weights']

    # TODO neither fn below explicitly takes this, so could delete if i end up
    # changing so it's implied by e.g. prat_claws=True (or even if it's the default)
    assert kws['one_row_per_claw']
    # TODO delete
    print(f'{wPNKC_params=}')
    #

    weights = connectome_APL_weights(wPNKC=wPNKC, **wPNKC_params)
    # True as of 2026-04-14 (and maybe not always after), when scaling (in python)
    # each weight vector by `len(weights) / weights.sum()`
    assert all(np.isclose(ws.mean(), 1) for ws in weights), ('still scaling (in python)'
        ' each weight vector to mean of 1?'
    )
    # not yet scaled by tuning in olfsysm, just "normalization" in python
    wAPLKC, wKCAPL, wAPLPN, wPNAPL = weights
    name2weights = {
        'wAPLKC': wAPLKC,
        'wKCAPL': wKCAPL,
        'wAPLPN': wAPLPN,
        'wPNAPL': wPNAPL,
    }

    for kws in tqdm(BOUTON_MODEL_KW_LIST, unit='model-params'):
        run_tuned_and_multireponsder_apl_boosted(plot_root, orn_deltas, name2weights,
            kws
        )


if __name__ == '__main__':
    main()

