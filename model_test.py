#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from drosolf import orns

from al_analysis import matt_data_dir, fit_mb_model


# TODO TODO TODO re: narrow-odors-jupyter/modeling.ipynb:
# - what is diff between "connection-weighted" vs "synapse-weighted" Hemibrain matrix?
# - which (if either) was used to produce hemibrain plot in preprint?

# TODO convert this to unit test(s)?
def main():
    # TODO TODO TODO also load and try to reproduce his hemidraw / uniform draw stuff
    # TODO TODO how to get seeds to be the same tho?

    #import ipdb; ipdb.set_trace()

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
    responses, gkc_wide = fit_mb_model(connectome_wPNKC=True, _use_matt_wPNKC=True)

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
    r1, _ = fit_mb_model(orn_deltas, tune_on_hallem=True, connectome_wPNKC=True,
        _use_matt_wPNKC=True
    )

    # NOTE: tune_on_hallem would be True by default here anyway
    r2, _ = fit_mb_model(tune_on_hallem=True, connectome_wPNKC=True,
        _use_matt_wPNKC=True
    )

    assert np.array_equal(r1.values, r2.values)
    # (model_kc)
    assert r1.index.equals(r2.index)
    # this won't be true for odors passed through odor2abbrev
    assert (
        ((r1.columns + ' @ -2') == r2.columns).sum() / len(r1.columns) >= 0.5
    ), 'assuming more than half of hallem odors not in odor2abbrev'

    # Hardcoded from what value this takes in fit_and_plot_mb_model calls in
    # al_analysis.py. fit_mb_model should process Hallem odor names such that all of
    # these are in there, internally (though at -2 rather than -3...)
    remy_odors = {
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
    }
    r3, _ = fit_mb_model(tune_on_hallem=True, connectome_wPNKC=True,
        sim_odors=remy_odors, _use_matt_wPNKC=True
    )

    def is_remy_odor_col(c):
        if c.replace(' @ -2', ' @ -3') in remy_odors:
            return True
        return False

    remy_odor_cols = [c for c in r3.columns if is_remy_odor_col(c)]
    assert remy_odor_cols == [c for c in r2.columns if is_remy_odor_col(c)]
    assert r3[remy_odor_cols].equals(r2[remy_odor_cols])

    # TODO w/ both calls above, or just w/ orn_deltas left None (as we'd normally call
    # when just trying to check w/ hallem input)?

    # TODO delete? or factor to a separate unit test just checking this call to model
    # doesn't fail? replace w/ using matt's input seed(s) and actually comparing to one
    # of his other non-hemibrain draws
    # TODO TODO TODO TODO why does this seem to produce frac_silent=0??? red flag?
    # (for connectome stuff tuned on hallem, it's 0.443)
    r2, _ = fit_mb_model(connectome_wPNKC=False, _use_matt_wPNKC=True)


if __name__ == '__main__':
    main()

