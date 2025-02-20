#!/usr/bin/env python3

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from drosolf import orns

from hong2p import viz
from al_analysis import (matt_data_dir, fit_mb_model, savefig, diverging_cmap, cmap,
    abbrev_hallem_odor_index, panel2name_order
)
import al_analysis


# TODO TODO TODO re: narrow-odors-jupyter/modeling.ipynb:
# - what is diff between "connection-weighted" vs "synapse-weighted" Hemibrain matrix?
# - which (if either) was used to produce hemibrain plot in preprint?
# TODO relevant to disrepancy between matt's / my (via prat) wPNKC matrix?

al_analysis.plot_fmt = 'pdf'

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

    # TODO maybe move these first three runs + assertions into model_test.py, and break
    # rest of sensitivity analysis stuff into own script?

    # TODO TODO TODO see what values i get for these from fitting (using my data as
    # input) -> grid search around those!
    # TODO TODO TODO [a version] also using my data as input!!! (+ search around those
    # params, using that input data)
    # TODO factor all sensitivity analysis into fns that can be called after model
    # tuning, varying around tuned params (and comparing to tuned outputs, in plots)?
    #
    # got these from uniform tuning, using first call below
    # (hemibrain, _use_matt_wPNKC=True)
    uniform_fixed_thr = 145.97261409492017
    # NOTE: can't exactly reproduce output only specifying wAPLKC like this.
    # also had to specify `wKCAPL = np.ones((1, mp.kc.N)) * 0.002386503067484662`
    # (which is almost exactly 1/wAPLKC). not currently exposed as param tho.
    uniform_wAPLKC = 3.8899999999999992

    # TODO delete
    # no longer actually need this. defining from wAPLKC works.
    uniform_wKCAPL = 0.002386503067484662

    # TODO move to another file / delete
    '''
    # don't want to consider choices of the above 2 that give us sparsities outside this
    # range
    min_sparsity = 0.03
    max_sparsity = 0.25

    # TODO restore False
    #ignore_existing = False
    ignore_existing = True

    # TODO TODO TODO rename / delete / move existing outputs -> re-run searches
    # -> see if relative insensitivity to wAPLKC is still there (and can i fix it?)

    # TODO TODO break sensitivity analysis into its own script, distinct from
    # model_test.py
    parent_output_dir = Path('kc_model_sensitivity')
    # savefig will generally do this for us below
    parent_output_dir.mkdir(exist_ok=True)

    tried_param_cache = parent_output_dir / 'tried.csv'
    if tried_param_cache.exists():
        tried = pd.read_csv(tried_param_cache)
        # TODO assert columns are same as below (refactor col def above conditional)
    else:
        tried = pd.DataFrame(columns=['fixed_thr', 'wAPLKC', 'megamat_sparsity',
            'full_hallem_sparsity'
        ])

    # TODO tqdm
    for fixed_thr, wAPLKC in itertools.product(
            # TODO try to make sure uniform_* params are included in sweep?
            # (just need odd num= kwarg for linspace?)
            #(uniform_fixed_thr,),
            #(uniform_wAPLKC,)

            # TODO test whether these ranges are reasonable
            np.linspace(uniform_fixed_thr / 10, uniform_fixed_thr * 10, num=10),

            # TODO does it change now that i fixed wKCAPL handling? delete comments
            # below?
            #
            # TODO TODO actually get it so wAPLKC changes output sparsity
            # (for a given threshold choice)
            # trying in to [uniform_wAPLKC / 2, uniform_wAPLKC * 4] didn't seem to do
            # that. [u/4, u*10] barely seemed to cause changes either...
            # TODO TODO maybe set to 0 to check it's actually doing something?
            # TODO TODO re-check modelling code, and the values of these after
            # running stuff (all their elements should be from scalar input)
            #
            np.linspace(uniform_wAPLKC / 10, uniform_wAPLKC * 10, num=10)

            # TODO probably delete logspace steps. now that wKCAPL actually working,
            # probably don't need.
            # TODO TODO logarithmic steps?
            # .1 (10**-1) - 10,000 (10**4)
            #np.logspace(-1, 6, num=6)
        ):

        # TODO decide whether whether to skip current params based on tried
        # (rows where both params are >= current, and megamat_sparsity was <
        # min_sparsity, or reverse)

        print(f'{fixed_thr=}')
        print(f'{wAPLKC=}')

        # TODO TODO also skip if directory exists, unless a flag to recomplete?
        output_dir = parent_output_dir / f'thr{fixed_thr:.2f}_wAPLKC{wAPLKC:.2f}'
        if output_dir.exists() and not ignore_existing:
            print(f'{output_dir} already existed. skipping!')
            continue

        # TODO delete
        #print('tried:')
        #print(tried)

        # TODO maybe use full_hallem_sparsity (sparsity110) for these cutoffs instead?
        # sparsity17 is a bit higher in all cases i've seen (often ~2x sparsity110)
        if ((tried.fixed_thr <= fixed_thr) & (tried.wAPLKC <= wAPLKC) &
            (tried.megamat_sparsity < min_sparsity)).any():

            print(f'sparsity17 would be < {min_sparsity=}')
            continue

        elif ((tried.fixed_thr >= fixed_thr) & (tried.wAPLKC >= wAPLKC) &
                (tried.megamat_sparsity > max_sparsity)).any():

            print(f'sparsity17 would be > {max_sparsity=}')
            continue

        # TODO TODO turn off _use_matt_wPNKC for actual fitting (+ probably rebase first
        # to include the allowdd code)
        # TODO flag to disable printing olfsysm output (and use that by default)
        # TODO or save log to each directory (eh...)?
        responses, _, gkc_wide, _ = fit_mb_model(pn2kc_connections='hemibrain',
            _use_matt_wPNKC=True, fixed_thr=fixed_thr, wAPLKC=wAPLKC
        )

        responses = abbrev_hallem_odor_index(responses, axis='columns')

        sparsity110 = responses.mean().mean()

        # TODO refactor to share subsetting to megamat w/ al_analysis stuff
        # TODO assert have all 17? would if fail if one was missing?
        panel_odors = [f'{n} @ -2' for n in panel2name_order['megamat']]
        responses = responses.loc[:, panel_odors]

        responses.columns = responses.columns.map(lambda x: x[:-(len(' @ -2'))])

        # sparsity within Remy's 17 megamat odors
        sparsity17 = responses.mean().mean()
        # TODO TODO actually write this to a file or something?
        # (already in CSV, right? output to separate CSV in each dir?)
        print(f'{sparsity17=}')

        tried = tried.append(
            {
                'fixed_thr': fixed_thr, 'wAPLKC': wAPLKC,
                'megamat_sparsity':  sparsity17, 'full_hallem_sparsity': sparsity110,
            },
            ignore_index=True
        )
        tried = tried.sort_values(['fixed_thr','wAPLKC'])
        tried.to_csv(tried_param_cache, index=False)

        # TODO should stuff have to be out-of-bounds for BOTH 17 odor panel and
        # overall, to be skipped?
        if not (min_sparsity <= sparsity17 <= max_sparsity):
            print('sparsity17 outside bounds!')
            continue

        # TODO subscript 17?
        title = (f'thr={fixed_thr:.2f}, wAPLKC={wAPLKC:.2f}'
            f' (sparsity17={sparsity17:.2f})'
        )

        # TODO fix aspect ratio so we can actually see each of the odors (/ delete)
        # TODO transpose?
        fig, _ = viz.matshow(responses, cmap=cmap)
        # TODO replace w/ regular axes title (maybe pass ax in to matshow?)?
        fig.suptitle(title)
        savefig(fig, output_dir, 'responses')

        fig, _ = viz.matshow(responses.corr(), cmap=diverging_cmap, vmin=-1.0, vmax=1.0)
        fig.suptitle(title)
        savefig(fig, output_dir, 'corr')
        # TODO corr diff plot too (even if B doesn't want to plot it for now)?

        # TODO TODO TODO show line for outputs w/ tuned parameters on sparsity +
        # n_odors_per_cell plots
        # TODO and (maybe later) correlation diffs wrt model w/ tuned params

        # TODO TODO fixed cbar/ylimits for sparsity plots
        # TODO TODO fixed scales for plots in general (corr prob ok already?)
        # TODO and assert no sparsity (/ value) goes outside limits

        sparsity_per_odor = responses.mean(axis='rows')
        sparsity_per_odor = sparsity_per_odor.to_frame('sparsity')

        # TODO delete?
        # TODO .T?
        fig, _ = viz.matshow(sparsity_per_odor, cmap=cmap)
        fig.suptitle(title)
        savefig(fig, output_dir, 'sparsity_per_odor_mat')
        #

        # TODO sort by sparsity?
        #
        # TODO or do i want lineplot instead? (see S1B, which has both, bar for hallem
        # data)
        fig, ax = plt.subplots()
        sns.barplot(sparsity_per_odor.reset_index(), x='odor', y='sparsity',
            color='black'
        )
        # TODO simpler way?
        for x in ax.get_xticklabels():
            x.set_rotation(90)

        # TODO maybe make ylabel 'response rate' instead (more accurate than
        # 'sparsity'...)
        ax.set_title(title)
        savefig(fig, output_dir, 'sparsity_per_odor')

        xlabel = '# odors'
        ylabel = '# cells responding to N odors'

        # TODO TODO say how many total cells (looks like 1630 in halfmat model now?)
        # TODO TODO and/or just say fraction of cells in these next two plots

        # how many odors each cell responds to
        n_odors_per_cell = responses.sum(axis='columns')
        fig, ax = plt.subplots()
        # discrete=True will mainly just set binwidth to 1 (binwidth=1 actually didn't
        # work for me, but this does)
        sns.histplot(ax=ax, data=n_odors_per_cell, discrete=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        savefig(fig, output_dir, 'n_odors_per_cell_hist')

        # (for if i want a line plot of above)
        #
        # this will be ordered w/ silent cells first, cells reponding to 1 odor 2nd, ...
        n_odors_per_cell_counts = n_odors_per_cell.value_counts().sort_index()
        fig, ax = plt.subplots()
        # blue to be consistent w/ hemibrain line in preprint?
        color = 'blue'
        sns.lineplot(n_odors_per_cell_counts, color=color, marker='o',
            markerfacecolor='white', markeredgecolor=color
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        savefig(fig, output_dir, 'n_odors_per_cell')

        responses.to_csv(output_dir / 'responses.csv')
        responses.to_pickle(output_dir / 'responses.p')
        #import ipdb; ipdb.set_trace()

        print()

    # TODO delete
    print('EXITING EARLY!!!')
    import sys; sys.exit()
    #
    '''

    # TODO if i move sensitivity analysis stuff (that varies fixed_thr/wAPLKC) to a
    # separate file, move this there too
    """
    #
    # TODO delete (as long as 2 param call works [which it does], don't need 3 param
    # version)
    '''
    responses, _, gkc_wide, _ = fit_mb_model(pn2kc_connections='hemibrain',
        _use_matt_wPNKC=True, fixed_thr=uniform_fixed_thr, wAPLKC=uniform_wAPLKC,
        wKCAPL=uniform_wKCAPL
    )
    sparsity110 = responses.mean().mean()
    print(f'sparsity: {sparsity110:.4f}')

    # TODO refactor this checking against matt's hemibrain outputs
    # (if i plan to keep all these checks, rather than just 1 assertion on responses,
    # in each of 3 places i'm currently checking against same data...)
    assert gkc_wide.index.name == 'bodyid'
    assert np.array_equal(wide.index, gkc_wide.index)
    assert np.array_equal(responses, wide)
    print("hemibrain (halfmat) responses equal to Matt's (fixing all 3 params to "
        "uniform tuning outputs)\n"
    )
    '''

    # now also working
    responses, _, gkc_wide, _ = fit_mb_model(pn2kc_connections='hemibrain',
        _use_matt_wPNKC=True, fixed_thr=uniform_fixed_thr, wAPLKC=uniform_wAPLKC,
    )
    sparsity110 = responses.mean().mean()
    print(f'sparsity: {sparsity110:.4f}')

    assert gkc_wide.index.name == 'bodyid'
    assert np.array_equal(wide.index, gkc_wide.index)
    assert np.array_equal(responses, wide)
    print("hemibrain (halfmat) responses equal to Matt's (fixing just fixed_thr and "
        "wAPLKC to uniform tuning outputs)\n"
    )
    """

    # TODO rename to run_model? or have a separate fn for that? take `mp` (and `rv` too,
    # or are even fit thresholds in mp?) as input (and return from fit_model?)?
    # TODO modify so i don't need to return gkc_wide here (or at least be more clear
    # about what it is, both in docs and in name)?
    responses, _, gkc_wide, _ = fit_mb_model(tune_on_hallem=True,
        pn2kc_connections='hemibrain', _use_matt_wPNKC=True
    )
    # (i might decide to change this index name, inside fit_mb_model...)
    assert gkc_wide.index.name == 'bodyid'
    assert np.array_equal(wide.index, gkc_wide.index)
    assert np.array_equal(responses, wide)
    print("hemibrain (halfmat) responses equal to Matt's (uniform tuning)\n")

    # TODO TODO also try orn_deltas having one less odor than hallem or something?
    # or change the names? to make it more clear we aren't getting the other half of the
    # concatenated matrix
    # TODO standard transpose orientation for my data + this, so i don't need to
    # tranpose (as much)? (think i want rows = odors?)
    # TODO TODO TODO also test (+ get working w/) columns='glomerulus', for easier use
    # on my data
    orn_deltas = orns.orns(columns='receptor', add_sfr=False).T

    orn_deltas = abbrev_hallem_odor_index(orn_deltas, axis='columns')

    odor_names = list(orn_deltas.columns)
    orn_deltas.columns += ' @ -2'

    # TODO TODO address `not have_DA4m` breakpoint in fit_mb_model (currently commented)
    #
    # TODO TODO also test if input has glomeruli instead of receptors
    # TODO TODO TODO compare output to if we pass orn_deltas (w/ glomeruli?) but set
    # tune_on_hallem=False
    # TODO TODO TODO compare that to passing just the megamat subset
    # TODO TODO TODO and also compare to passing megamat subset, but w/
    # tune_on_hallem=True
    r1, _, _, _ = fit_mb_model(orn_deltas, tune_on_hallem=True,
        pn2kc_connections='hemibrain', _use_matt_wPNKC=True
    )

    assert np.array_equal(r1.values, responses.values)
    # (model_kc)
    assert r1.index.equals(responses.index)
    # TODO fix comment / assert message. i think it's odor2abbrev...
    # this won't be true for odors passed through odoresponsesabbrev
    assert (
        ((r1.columns + ' @ -2') == responses.columns).sum() / len(r1.columns) >= 0.5
    ), 'assuming more than half of hallem odors not in odoresponsesabbrev'

    r3, _, _, _ = fit_mb_model(tune_on_hallem=True, pn2kc_connections='hemibrain',
        # it's not an issue that remy_odors has '@ -3' suffix in sim_odors here.
        # fit_mb_model allows concs in [-3, -1) to match hallem.
        #
        # set would work, but order of odors would not be fixed then.
        sim_odors=sorted(set(remy_odors)), _use_matt_wPNKC=True
    )

    def is_remy_odor_col(c):
        if c.replace(' @ -2', ' @ -3') in remy_odors:
            return True
        return False

    remy_odor_cols = [c for c in r3.columns if is_remy_odor_col(c)]
    assert remy_odor_cols == [c for c in responses.columns if is_remy_odor_col(c)]
    assert r3[remy_odor_cols].equals(responses[remy_odor_cols])

    megamat_odors = panel2name_order['megamat']
    assert len(megamat_odors) == 17

    # TODO delete (if moving earlier works)
    #orn_deltas = abbrev_hallem_odor_index(orn_deltas, axis='columns')
    #assert all(x in orn_deltas.columns for x in megamat_odors)
    #
    assert all(x in odor_names for x in megamat_odors)
    megamat_deltas = orn_deltas[megamat_odors].copy()

    r4, _, _, _ = fit_mb_model(megamat_deltas, tune_on_hallem=True,
        pn2kc_connections='hemibrain', _use_matt_wPNKC=True
    )

    # TODO worth also checking same but in uniform draw case
    # (equiv of either (r3 or r4) vs responses)?
    assert np.array_equal(
        r3.sort_index(axis='columns').values,
        r4.sort_index(axis='columns').values
    )
    # TODO TODO TODO compare to just tuning on megamat odors? matt tunes on all hallem
    # for all preprint stuff, right?

    # TODO TODO add check that sorting wPNKC in some other order doesn't change any
    # outputs qualitatively (e.g. the average correlation across several seeds)
    # (would expect it to change effect of seed on draw, but hopefully nothing else,
    # in only uniform/hemidraw/caron cases [not any fixed wPNKC cases])

    # NOTE: I'm now planning on rebasing some of my changes (originally made relative to
    # 0d23530, the commit before c70b0e7f) after a commit that just reverts c70b0e7.
    #
    # TODO TODO maybe restore what c70b0e7 added, but add options to allow fully
    # disabling the new rng that commit also added (which is probably not consequential,
    # other than making it hard to compare exactly to older results)
    #
    # TODO detect/set this automatically?
    #
    # This commit, from 2022-02-01, added option to block double-draws
    # (and overall changed how rng was used slightly, but probably inconsequential to
    # behavior on average). Just need to revert olfsysm/libolfsysm/src/olfsysm.cpp to
    # get stuff to a state where we can set this True (at least when repo is otherwise
    # in c70b0e7f)
    olfsysm_is_pre_c70b0e7f = True
    #olfsysm_is_pre_c70b0e7f = False

    # TODO TODO TODO wPNKC at least constant (across runs) here? why not in 54 glom case
    # from al_analysis?

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

            # TODO try passing in hallem odors as orn_deltas= (w/ diff column order?)
            # (both w/ and w/o sim_odors, but start w/o?) and see if that alone breaks
            # uniform draw wPNKC consistency?

            # TODO is it actually appreciably faster to only re-run the run_KC_sims step
            # (w/ last param = True), as matt does?
            # TODO regenerate matt's hemidraw / uniform responses w/ the methanoic acid
            # bug fixed?
            curr_responses, _, _, _ = fit_mb_model(pn2kc_connections=draw_type,
                n_claws=n_claws, _use_matt_wPNKC=True, seed=(seed + i),
                # TODO regen matt's things with methanoic acid mistake fixed -> compare
                # to that -> delete this flag
                _add_back_methanoic_acid_mistake=True
            )

            # TODO TODO TODO see how different things actually are w/ c70b0e7f tho.
            # presumably not qualitatively different, and probably not even perceptibly?
            #
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

        # TODO just plot remy_odors (and do same w/ matt_responses)
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


if __name__ == '__main__':
    main()

