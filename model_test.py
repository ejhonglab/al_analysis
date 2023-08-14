#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from drosolf import orns

from al_analysis import matt_data_dir, fit_mb_model


# TODO TODO TODO re: narrow-odors-jupyter/modeling.ipynb:
# - what is diff between "connection-weighted" vs "synapse-weighted" Hemibrain matrix?
# - which (if either) was used to produce hemibrain plot in preprint?
# TODO relevant to disrepancy between matt's / my (via prat) wPNKC matrix?

# Hardcoded from what value this takes in fit_and_plot_mb_model calls in
# al_analysis.py. fit_mb_model should process Hallem odor names such that all of
# these are in there, internally (though at -2 rather than -3...)
remy_odors = [
    '1-5ol @ -3',
    '1-6ol @ -3',
    '1-8ol @ -3',
    '2-but @ -3',
    '2h @ -3',
    '6al @ -3',
    'B-cit @ -3',
    'IaA @ -3',
    'Lin @ -3',
    'aa @ -3',
    'benz @ -3',
    'eb @ -3',
    'ep @ -3',
    'ms @ -3',
    'pa @ -3',
    't2h @ -3',
    'va @ -3',
]

# TODO convert this to unit test(s)?
def main():
    # TODO detect/set this automatically?
    #
    # This commit, from 2022-02-01, added option to block double-draws
    # (and overall changed how rng was used slightly, but probably inconsequential to
    # behavior on average). Just need to revert olfsysm/libolfsysm/src/olfsysm.cpp to
    # get stuff to a state where we can set this True (at least when repo is otherwise
    # in c70b0e7f)
    #olfsysm_is_pre_c70b0e7f = True
    olfsysm_is_pre_c70b0e7f = False

    n_claws = 7
    draw_types = ('hemidraw', 'uniform')
    n_first_to_check = 2

    # TODO also test n_claws=5 in both hemidraw and uniform draw cases?
    for draw_type in draw_types:
        matt_responses_path = matt_data_dir / 'reference' / f'{draw_type}-{n_claws}.npy'

        # Of shape (# KCs, # Hallem odors, # random repeats [=seeds] of wPNKC draw),
        # e.g. (1630, 110, 100)
        matt_responses = np.load(matt_responses_path)

        # Same seed Matt starts at in matt-modeling/docs/independent-draw-reference.html
        seed = 94894 + 1

        assert matt_responses.shape[:2] == (1630, 110)
        n_repeats = matt_responses.shape[-1]

        responses = []
        for i in tqdm(range(min(n_repeats, n_first_to_check)), unit='seed',
            desc=f'{draw_type} ({n_claws=})'):

            # TODO is it actually appreciably faster to only re-run the run_KC_sims step
            # (w/ last param = True), as matt does?
            # TODO regenerate matt's hemidraw / uniform responses w/ the methanoic acid
            # bug fixed?
            curr_responses, _ = fit_mb_model(pn2kc_connections=draw_type,
                n_claws=n_claws, _use_matt_wPNKC=True, seed=(seed + i),
                # TODO regen matt's things with methanoic acid mistake fixed -> compare
                # to that -> delete this flag
                _add_back_methanoic_acid_mistake=True
            )

            # NOTE: this comparison only works w/ olfsysm 0d23530 (and NOT c70b0e7f or
            # later), as the slight changes to the RNG handling introduced in c70b0e7f
            # cause different output for the same seed (but probably don't change the
            # behavior on average?)
            # TODO actually check that (w/ sufficiently large n_repeats), behavior is
            # "appropriately" close on average?
            # TODO factor stuff that requires this olfsysm version into another test
            # that is only run in that context?
            if olfsysm_is_pre_c70b0e7f:
                assert np.array_equal(curr_responses, matt_responses[:, :, i])
            else:
                responses.append(curr_responses)
                # TODO still aggregate responses (to compute mean) and plot mean
                # responses (and/or compare within some tolerance?) to compare
                # qualitatively
                print('can not compare random draw responses exactly, as olfsysm RNG '
                    'has changed!'
                )

        # TODO TODO just plot remy_odors (and do same w/ matt_responses)
        # (do w/ new >=c70b0e7f olfsysm, to show that behavior is qualitatively the same
        # on average)

        # TODO delete (/ modify to qualitatively compare mean behavior in new olfsysm vs
        # in matt's old outputs)
        '''
        import matplotlib.pyplot as plt
        import natmix
        from hong2p.xarray import odor_corr_frame_to_dataarray
        from al_analysis import sort_odors, remy_matshow_kwargs

        # TODO replace w/ just using hardcoded remy_odors order?
        responses = sort_odors(responses.T, add_panel='megamat').T
        responses = responses.droplevel('panel', axis='columns')
        import ipdb; ipdb.set_trace()
        responses.columns.name = 'odor1'
        model_corr_df = responses.corr()
        model_corr = odor_corr_frame_to_dataarray(model_corr_df)
        # TODO TODO subset just to remy odors to qualitatively compare to what matt got
        fig = natmix.plot_corr(model_corr, **remy_matshow_kwargs)

        plt.show()
        import ipdb; ipdb.set_trace()
        '''
        #

    # TODO TODO fix code that generated hemimatrix.npy / delete
    # (to remove effect of hc_data.csv methanoic acid bug that persisted in many copies
    # of this csv) (won't be equal to `wide` until fixed)
    #
    # Still not sure which script of Matt's wrote this (couldn't find by grepping his
    # code on hal), but we can compare it to the same matrix reformatted from
    # responses.csv (which is written in hemimat-modeling.html)
    #hemi = np.load(matt_data_dir / 'reference/hemimatrix.npy')

    # I regenerated this, using Matt's account on hal, by manually running all the
    # relevant code from matt-modeling/docs/hemimat-modeling.html, because it seemed the
    # previous version was affected by the hc_data.csv methanoic acid error.
    # After regenerating it, my outputs computed in this script are now equal.
    df = pd.read_csv(matt_data_dir / 'hemibrain/halfmat/responses.csv')

    # The Categoricals are just to keep order of odors and KC body IDs the same as in
    # input. https://stackoverflow.com/questions/57177605/
    df['ordered_odors'] = pd.Categorical(df.odor, categories=df.odor.unique(),
        ordered=True
    )
    df['ordered_kcs'] = pd.Categorical(df.kc, categories=df.kc.unique(), ordered=True)
    wide = df.pivot(columns='ordered_odors', index='ordered_kcs', values='r')
    del df

    #assert np.array_equal(hemi, wide.values)
    #del hemi

    # TODO rename to run_model? or have a separate fn for that? take `mp` (and `rv` too,
    # or are even fit thresholds in mp?) as input (and return from fit_model?)?
    # TODO modify so i don't need to return gkc_wide here (or at least be more clear
    # about what it is, both in docs and in name)?
    responses, gkc_wide = fit_mb_model(pn2kc_connections='hemibrain',
        _use_matt_wPNKC=True
    )

    # (i might decide to change this index name, inside fit_mb_model...)
    assert gkc_wide.index.name == 'bodyid'
    assert np.array_equal(wide.index, gkc_wide.index)
    assert np.array_equal(responses, wide)
    print("hemibrain (halfmat) responses equal to Matt's")

    # TODO TODO also try orn_deltas having one less odor than hallem or something?
    # or change the names? to make it more clear we aren't getting the other half of the
    # concatenated matrix
    # TODO standard transpose orientation for my data + this, so i don't need to
    # tranpose (as much)? (think i want rows = odors?)
    # TODO TODO TODO also test (+ get working w/) columns='glomerulus', for easier use
    # on my data
    orn_deltas = orns.orns(columns='receptor', add_sfr=False).T

    # TODO TODO also test if input has glomeruli instead of receptors
    r1, _ = fit_mb_model(orn_deltas, tune_on_hallem=True, pn2kc_connections='hemibrain',
        _use_matt_wPNKC=True
    )

    # NOTE: tune_on_hallem would be True by default here anyway
    r2, _ = fit_mb_model(tune_on_hallem=True, pn2kc_connections='hemibrain',
        _use_matt_wPNKC=True
    )

    assert np.array_equal(r1.values, r2.values)
    # (model_kc)
    assert r1.index.equals(r2.index)
    # this won't be true for odors passed through odor2abbrev
    assert (
        ((r1.columns + ' @ -2') == r2.columns).sum() / len(r1.columns) >= 0.5
    ), 'assuming more than half of hallem odors not in odor2abbrev'

    r3, _ = fit_mb_model(tune_on_hallem=True, pn2kc_connections='hemibrain',
        # TODO sim_odors actually need to be a set?
        sim_odors=set(remy_odors), _use_matt_wPNKC=True
    )

    def is_remy_odor_col(c):
        if c.replace(' @ -2', ' @ -3') in remy_odors:
            return True
        return False

    remy_odor_cols = [c for c in r3.columns if is_remy_odor_col(c)]
    assert remy_odor_cols == [c for c in r2.columns if is_remy_odor_col(c)]
    assert r3[remy_odor_cols].equals(r2[remy_odor_cols])

    # TODO delete? or factor to a separate unit test just checking this call to model
    # doesn't fail? replace w/ using matt's input seed(s) and actually comparing to one
    # of his other non-hemibrain draws
    r2, _ = fit_mb_model(pn2kc_connections='hemidraw', _use_matt_wPNKC=True, n_claws=6)


if __name__ == '__main__':
    main()

