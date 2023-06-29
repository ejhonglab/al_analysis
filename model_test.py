#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

import olfsysm as osm
import drosolf
from drosolf.orns import orns


# TODO TODO TODO re: narrow-odors-jupyter/modeling.ipynb:
# - what is diff between "connection-weighted" vs "synapse-weighted" Hemibrain matrix?
# - which (if either) was used to produce hemibrain plot in preprint?

def main():
    matt_data_dir = Path('../matt/matt-modeling/data')

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

    #assert np.array_equal(hemi, wide.values)
    #del hemi


    gkc_wide = pd.read_csv(matt_data_dir / 'hemibrain/halfmat/gkc-halfmat-wide.csv')
    gkc_mat = pd.read_csv(matt_data_dir / 'hemibrain/halfmat/gkc-halfmat.csv',
        header=None
    )
    # All other columns are glomerulus names.
    assert gkc_wide.columns[0] == 'bodyid'
    assert np.array_equal(gkc_wide.iloc[:, 1:].values, gkc_mat)

    mp = osm.ModelParams()

    # TODO TODO TODO what was matt using this for in narrow-odors-jupyter/modeling.ipynb
    #mp.kc.ignore_ffapl = True

    mp.kc.thr_type = 'uniform'

    hc_data_csv = str(Path('~/src/olfsysm/hc_data.csv').expanduser())
    osm.load_hc_data(mp, hc_data_csv)

    orn_deltas = orns(add_sfr=False, drop_sfr=False).T
    sfr = orn_deltas['spontaneous firing rate']
    assert orn_deltas.columns[-1] == 'spontaneous firing rate'
    orn_deltas = orn_deltas.iloc[:, :-1]

    # NOTE: as long as we finish all changes to mp.orn.data.[delta|spont] BEFORE
    # initializing RunVars, we should be fine. After that, sizes of these matrices
    # should not change (but content can).
    skip_idx = None
    for i, osm_one_orn_deltas in enumerate(mp.orn.data.delta):

        my_one_orn_deltas = orn_deltas.iloc[i]
        if not np.array_equal(osm_one_orn_deltas, my_one_orn_deltas):
            assert np.array_equal(orn_deltas.iloc[i + 1], osm_one_orn_deltas)
            skip_idx = i
            break

    assert skip_idx is not None

    # TODO TODO merge da4m/l hallem data?
    # TODO do same w/ 33b (adding it into 47a and 85a Hallem data, for DM3 and DM5,
    # respectively)?

    # TODO TODO check this is right (33b/DM3. is that what ann drops?)
    # (33b goes to both DM3 and DM5, even though each of those has another unique
    # receptor)
    # TODO TODO where does ann even say she drops the 8th glomerulus
    # (or where is it in her code?)
    skip_or = orn_deltas.index[skip_idx]
    skip_glom = drosolf.orns.receptor2glomerulus[skip_or]
    print(f'\nDropping Hallem data for Or{skip_or}/{skip_glom} '
        f'(index={skip_idx}), consistent with Kennedy work.\n'
    )
    del skip_or, skip_glom

    shared_idx = np.setdiff1d(np.arange(len(orn_deltas)), [skip_idx])
    sfr = sfr.iloc[shared_idx]
    orn_deltas = orn_deltas.iloc[shared_idx]
    assert np.array_equal(sfr, mp.orn.data.spont[:, 0])
    assert np.array_equal(orn_deltas, mp.orn.data.delta)

    assert sfr.index[0] == '2a'
    assert orn_deltas.index[0] == '2a'

    # TODO try removing .copy()?
    sfr = sfr.iloc[1:].copy()
    orn_deltas = orn_deltas.iloc[1:].copy()
    mp.orn.data.spont = sfr
    mp.orn.data.delta = orn_deltas

    # TODO in narrow-odors-jupyter/modeling.ipynb, why does matt set
    # mp.kc.tune_from = np.arange(110, step=2)
    # (only tuning on every other odor from hallem, it seems)

    # TODO need to remove DA4m (2a) from gkc_wide / gkc_mat first too?  don't see matt
    # doing it in hemimat-modeling... (i don't think i need to.  rv.pn.pn_sims below had
    # receptor-dim length of 22)

    mp.kc.preset_wPNKC = True
    mp.kc.N = len(gkc_mat)

    rv = osm.RunVars(mp)

    rv.kc.wPNKC = gkc_mat

    osm.run_ORN_LN_sims(mp, rv)
    osm.run_PN_sims(mp, rv)
    osm.run_KC_sims(mp, rv, True)

    responses = rv.kc.responses

    assert np.array_equal(wide.index, gkc_wide.bodyid)
    assert np.array_equal(responses, wide)

    # TODO TODO TODO when fitting calcium->spike fn, should i add points from 0 dF/F ->
    # spontaneous spike rate?
    # TODO make sure each glomerulus has its specific spontaneous firing rate reflected
    # in its equation

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

