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

from hong2p.olf import parse_odor_name
from hong2p.viz import matshow
from hong2p.util import symlink, subset_same_in_all_dicts, shorten_path

from al_analysis import al_util
from al_analysis.al_util import (savefig, ParamDict, warn, read_parquet, to_json,
    read_json
)
from al_analysis.mb_model import (fit_and_plot_mb_model, megamat_orn_deltas,
    dict_seq_product, format_weights, format_model_params, get_thr_and_APL_weights,
    glomerulus_col, save_and_remove_from_param_dict, drop_silent_model_cells,
    load_and_plot_dynamics, update_var2range, MinMaxDict, natmix_orn_deltas,
    sort_rois_by_response_classes, format_response_class, summarize_response_classes,
    assert_fit_and_plot_outputs_equal
)


# TODO TODO also try (everything with?) target sparsity of 0.05-0.06? (instead of 0.1)
# TODO refactor to share w/ BOUTON_MODEL_KW_LIST and TRY_CLAW_MODELS_WITH
# TODO TODO factor out a TRY_[ALL_]MODELS_WITH from TRY_CLAW_MODELS_WITH (to mb_model,
# and use here and elsewhere)
MODEL_TUNE_KWS: List[ParamDict] = dict_seq_product(
    [
        # TODO TODO also try prat_boutons=False
        dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
            # TODO TODO also try w/ this =False?
            use_connectome_APL_weights=True
        )
    ],
    # TODO re-order so claw_dynamics gets put towards end of dir names, not at start
    # claw_dynamics=False is the default, and how the code has been for a long time
    [dict()] + dict_seq_product([dict(claw_dynamics=True)],
        [dict(), dict(allow_net_inh_per_claw=True)],
    ),
    # pn_claw_to_apl=False is the default, and could normally be omitted, but doing
    # it this way produces nicer directory names when using subset_same_in_all_dicts
    # to exclude params
    [dict(pn_claw_to_apl=True), dict(pn_claw_to_apl=False)]
)

OUTPUT_ROOT_NAME: str = 'PNAPL_stepping'
EXTRA_PANELS_DIRNAME: str = 'extra_panels'
spike_counts_parquet: str = 'spike_counts.parquet'

# TODO tuple, to make sure this doesn't get mutated?
STEPS = [100, 20, 1.0, 0.5, 10, .1]

def analyze_outputs(plot_dir: Path, *, plot_dynamics: bool = False,
    corners_only: bool = False, corners_and_tuned: bool = False) -> None:
    # TODO doc

    # TODO put behind verbose flag?
    print(f'analyzing outputs under: {plot_dir}')

    if plot_dir.parent.parent.name == EXTRA_PANELS_DIRNAME:
        panel = plot_dir.parent.name
    else:
        panel = 'megamat'

    kstr = plot_dir.name
    shared_cols = ['wAPLPN', 'wPNAPL', 'sparsity', 'n_silent_cells', 'avg_lts',
        'n_avg_odors_responded_to'
    ]
    # TODO TODO TODO also compute # of cells in diff response classes, using same code
    # natmix_data/analysis.py currently uses for that (after the scaling and
    # everything)?
    natmix_cols = [
        # TODO TODO TODO implement at least these
        '5mix_minus_max_avg_sparsity',
        'binary-mix_minus_max_avg_sparsity',

        '5mix_minus_max_avg_n_spikes',
        'binary-mix_minus_max_avg_n_spikes',

        'avg_5mix_minus_max_per-responding-cell_n_spikes',
        'avg_binary-mix_minus_max_per-responding-cell_n_spikes',

        # TODO TODO also (within responders) compute cell specific mix - max component
        # TODO TODO also count # of mix responders / # component only responders, and
        # summarize those?
    ]
    panel2extra_cols = {
        'megamat': ['hept_pent_corr'],
        'kiwi': natmix_cols,
        'control': natmix_cols,
    }
    cols = shared_cols + panel2extra_cols[panel]
    cols_computed_over_responders_only = ['avg_lts', 'n_avg_odors_responded_to',
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

        # TODO TODO have diverging cmap around tuned value for all of these?
        #'sparsity': (0, 1),
        'sparsity': (0, .2),

        # TODO assert no data exceeds max here (and for all? doing already?)
        'n_avg_odors_responded_to': (0, 6),

        # TODO is this correct? verify
        'avg_lts': (0, 1),
    }
    vals = []
    plot_suffix: str = '.pdf'
    d0_dynamics_plotnames: Optional[Set[str]] = None
    d0_dynamics_plot_dirnames: Set[str] = set()
    n_combos_seen = 0
    var2range: MinMaxDict = dict()

    dir_iter = list(plot_dir.glob('*/'))
    tqdm_kws = dict()
    if corners_only or corners_and_tuned:
        if corners_only:
            total = 4
        elif corners_and_tuned:
            total = 9
        # TODO want to keep disabling tqdm if -c/-C? don't need total then...
        tqdm_kws = dict(total=total, disable=True)

    dir_iter = tqdm(dir_iter, unit='model-dir', **tqdm_kws)

    existing_var2range = None
    if plot_dynamics:
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

    _warned_dirskip = False
    dirnames_analyzed = set()
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
        options = None
        if corners_only:
            options = (min(STEPS), max(STEPS))
        elif corners_and_tuned:
            options = (min(STEPS), 1.0, max(STEPS))

        if options is not None:
            if not (a2p in options and p2a in options):
                if not _warned_dirskip:
                    warn(f'skipping {d.name}, and all other directories not among '
                        'parameter subset selected by -c or -C'
                    )
                    _warned_dirskip = True
                continue

            dirnames_analyzed.add(d.name)
            n_combos_seen += 1

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
                try:
                    assert src.is_file() and not src.is_symlink()
                except AssertionError:
                    # TODO TODO diff error message if written to same run? or some other
                    # way of clarifying? load responses and check no responses?
                    # TODO what fraction of current cases (and -C cases) is this being
                    # hit for?
                    warn(f'{d.name} missing plot {p}. need to regenerated. may have had'
                        ' and error in plot creation'
                    )
                    continue

                curr_plot_link_dir = (plot_dir / p).with_suffix('')
                assert curr_plot_link_dir.is_dir()

                # can't use with_suffix on something w/ d.name as name, or it will strip
                # the last bit of the rightmost float parameter from name (after
                # decimal)
                link = curr_plot_link_dir / f'{d.name}{plot_suffix}'
                if link.exists():
                    assert link.is_symlink()
                    link.unlink()

                # TODO verbose=True (wouldn't currently do what i want)? print
                # something?
                symlink(src, link)

        # TODO also compute + save/load (to json) + use min/max limits for all vars
        # below (should only relevant if doing a higher dimensional sweep, where i'll be
        # plotting a grid of those grids)

        rs = read_parquet(d / 'responses.parquet')
        sp = rs.mean().mean()

        # NOTE: all quantities computed using rs_nosilent (as opposed to responses that
        # still have non-responding cells) should have their column name manually added
        # to cols_computed_over_responders_only above
        rs_nosilent = drop_silent_model_cells(rs)
        n_silent = len(rs) - len(rs_nosilent)
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

        shared_vals = (a2p, p2a, sp, n_silent, L.mean(), avg_n_odors)

        panel_vals = None
        if panel == 'megamat':
            # TODO also load + compute correlations using (scaled?) spike counts (w/
            # noise added?)? and compare those to remy corrs?

            # TODO average over corr w/ 1-6ol too? also eb/ep vs 1-5ol/1-6ol block?
            # TODO + maybe ratio/diff w/ corrs in rest of off diag, if needed (but prob
            # not)
            hept_pent_corr = rs_nosilent.corr().loc['1-5ol @ -3', '2h @ -3']
            # TODO TODO say what remy's value (computed on real KCs) is here, for
            # reference?
            panel_vals = (hept_pent_corr,)

        elif panel in ('kiwi', 'control'):
            # TODO refactor to share any of  this odor processing w/
            # natmix_data/analysis.py?
            odors = list(rs.columns)
            assert not any('pfo' in x or 'solvent' in x for x in odors)

            n_comps = 5
            comps = odors[:n_comps]
            assert not any('+' in x or 'mix' in x for x in comps)

            binary_comps = comps[-2:]
            binary_comp_names = [parse_odor_name(o) for o in binary_comps]

            mixes = odors[n_comps:]
            assert all('+' in x or 'mix' in x for x in mixes)

            # TODO TODO analyze both binary mixes (i.e. also including the "air" mix)?
            binary_mixes = [x for x in mixes if '+' in x]
            assert len(binary_mixes) == 2, f'{binary_mixes=}'
            invial_binary_mixes = [x for x in binary_mixes if '(air mix)' not in x]
            assert len(invial_binary_mixes) == 1, f'{invial_binary_mixes=}'
            binary_mix = invial_binary_mixes[0]

            assert all(n in binary_mix for n in binary_comp_names)

            # name should be 'cmix0' or 'kmix0' (full string ending in ' @ 0', even for
            # diluted ones)
            undiluted_5comp_mixes = [x for x in mixes if '-' not in x and '+' not in x]
            assert len(undiluted_5comp_mixes) == 1, f'{undiluted_5comp_mixes=}'
            undiluted_5comp_mix = undiluted_5comp_mixes[0]

            # TODO if n_responses_for_spike > 1, also send values <= that threshold to 0
            # here? (for spike_counts matrix)

            # TODO TODO also just catplot of these values per odor, for each step w/in
            # model parametrization? (for each of these 2-3, similar to plots for
            # george/yang stuff)
            mean_response_rates_per_odor = rs.mean()
            mix_minus_max_avg_sparsity = (
                mean_response_rates_per_odor[undiluted_5comp_mix] -
                mean_response_rates_per_odor[comps].max()
            )
            binary_mix_minus_max_avg_sparsity = (
                mean_response_rates_per_odor[binary_mix] -
                mean_response_rates_per_odor[binary_comps].max()
            )

            ss = read_parquet(d / spike_counts_parquet)

            avg_n_spikes_per_odor = ss.mean()
            mix_minus_max_avg_n_spikes = (
                avg_n_spikes_per_odor[undiluted_5comp_mix] -
                avg_n_spikes_per_odor[comps].max()
            )
            binary_mix_minus_max_avg_n_spikes = (
                avg_n_spikes_per_odor[binary_mix] -
                avg_n_spikes_per_odor[binary_comps].max()
            )

            ss_nosilent = ss.loc[rs_nosilent.index]
            # TODO TODO TODO make plot of response classes with this

            # TODO plot distribution of these?
            # TODO TODO apply some type of log/logistic scaling before computing these
            # differences (like in natmix_data/analysis.py)?
            percell_mix_minus_max = (
                ss_nosilent[undiluted_5comp_mix] - ss_nosilent[comps].T.max()
            )
            avg_mix_minus_max_percell = percell_mix_minus_max.mean()

            percell_binary_mix_minus_max = (
                ss_nosilent[binary_mix] - ss_nosilent[binary_comps].T.max()
            )
            avg_binary_mix_minus_max_percell = percell_binary_mix_minus_max.mean()

            # TODO use dicts instead of tuples so i can't screw up order wrt order of
            # cols above
            panel_vals = (
                mix_minus_max_avg_sparsity,
                binary_mix_minus_max_avg_sparsity,

                mix_minus_max_avg_n_spikes,
                binary_mix_minus_max_avg_n_spikes,

                avg_mix_minus_max_percell,
                avg_binary_mix_minus_max_percell,
            )
            # TODO TODO TODO what to load / compute from remy's data for comparison?
            # TODO TODO TODO save CSV/parquet from natmix_data/analysis.py, rather than
            # moving all that code over?
            #
            # TODO TODO maybe that's easier with response classes analysis?

            # TODO TODO TODO use similar code to define response classes, w/o actually
            # scaling to remy kc data (just use binarized responses?)
            # TODO TODO and do for at least n_spikes_for_response (just using responses)
            # and one spike above whatever that is (could compute if don't have params,
            # by checking min # spikes in elements considered as responses)
            # TODO TODO TODO condense down into # segmenting vs not?
            # TODO TODO also # mix-only responders?
            classes = sort_rois_by_response_classes(
                ss_nosilent[comps + [undiluted_5comp_mix]].T, 1.0
            )
            breakpoint()

        assert panel_vals is not None, f'{panel=} not recognized'

        # TODO use a dict instead? or something else make it easier to swap out a few
        vals.append(shared_vals + panel_vals)

    if corners_only:
        # TODO change if needed, if i sweep over more than 2 dims (i.e. adding wAPLKC
        # and wKCAPL)
        assert n_combos_seen == 4, f'{n_combos_seen=} != 4'
    elif corners_and_tuned:
        assert n_combos_seen == 9, f'{n_combos_seen=} != 9'

    if corners_only or corners_and_tuned:
        # TODO TODO why this printed 6 times? isn't it just 3 panels?
        print('only analyzed the following directory names, because -c/-C:')
        # TODO use len of this instead of n_combos_seen, and then delete that var?
        pprint(dirnames_analyzed)

    df = pd.DataFrame.from_records(vals, columns=cols)
    df = df.set_index(['wAPLPN', 'wPNAPL'], verify_integrity=True)
    if len(df) == 0:
        raise IOError(f'found no stepped model output subdirectories under {plot_dir}')

    if plot_dynamics:
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
        # that's why check=True path is failing, b/c they are converted to lists on
        # reading
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
        # TODO delete? put behind verbose flag?
        #print(f'analyzing {c}')

        fig, ax = plt.subplots()
        mat = df[c].unstack()
        mat.columns = mat.columns.astype(str)
        mat.index = mat.index.astype(str)

        if c in col2range:
            vmin, vmax = col2range[c]
            assert mat.min().min() >= vmin, f'{c=} {mat.min().min()=} {vmin}'
            assert mat.max().max() <= vmax, f'{c=} {mat.max().max()=} {vmax}'
        else:
            # TODO expand around by some tolerance at least?
            vmin = mat.min().min()
            vmax = mat.max().max()
            warn(f'{c} not in col2range. add there for fixed scale! using {vmin=:.2f} '
                f'and {vmax=:.2f} from data'
            )

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
        if c in cols_computed_over_responders_only:
            title += '\nsilent cells dropped'
        else:
            title += '\nall cells, including silent'
        # TODO include other model params?
        ax.set_title(title)

        savefig(fig, plot_root, f'{kstr}__{c}', bbox_inches='tight')

    # TODO put behind verbose flag?
    print()


def step_pn_apl_weights_around_tuned(plot_dir: Path, orn_deltas: pd.DataFrame,
    kws: ParamDict, *, ignore_existing: bool = False, try_lr_cache: bool = False,
    save_dynamics: bool = False, plot_dynamics: bool = False, tuned_only: bool = False,
    corners_only: bool = False, corners_and_tuned: bool = False,
    scale_pre_tuning: bool = False) -> None:
    # TODO doc
    """Runs `orn_deltas`
    Args:
        orn_deltas: glomerulus (rows) X odors (columns) estimated spike delta DataFrame

        kws: passed to all `fit_and_plot_mb_model` calls, and passed thru
            `format_model_params` to create output subdirectory names

        ignore_existing: if False, will attempt to load cached model outputs (already in
            directories that would be created), rather than re-running models. If True,
            will always re-run models.

        try_lr_cache: passed to `fit_and_plot_mb_model`

        save_dynamics: if True, will save DataArray pickles of all model internal
            dynamic quantities (e.g. membrane potential of KCs over time, to each odor)

        corners_only: if True, only analyzes combinations of min/max step for each
            parameter

        corners_and_tuned: similar to `corners_only`, but adds tuned (scale = 1.0) value
            too, for a total of 9 combos.

        scale_pre_tuning: skips all stepping of weights post-tuning. Post-tuning
            stepping was initially all I had implemented, and will often lead to
            response rates well outside target range. Pre-tuning stepping takes more
            time (when running the model at each of the steps, because it has to tune
            again), but should always produce average response rates within target range
            (if convergence is achieved, and there'd be an error if not).
    """
    output_kws = dict(
        # if return_dynamics is True, fit_and_plot_mb_model will write DataArrays
        # containing dynamics as pickles, before popping them from returned param dict.
        # plot_example_dynamics will make some internal plots using the same data, but
        # then will not return them from fit_mb_model (so they will not be saved).
        return_dynamics=save_dynamics,

        # TODO TODO TODO was broken on claw_dynamics=True stuff (still an issue?)
        # ...
        #    plot_apl_dynamics(plot_dir, dynamics_dict, stim_timing_kws, odor=odor,
        #  File "/home/tom/src/al_analysis/mb_model.py", line 9960, in plot_apl_dynamics
        #    assert (claws_no_inh >= claws).all()
        #
        # TODO delete make_plots=True? just make sure plot_example_dynamics sets
        # that? or do i really want (/have) a flag controlling plots other than
        # plot_example_dynamics (i.e. internal corrs)? rename, if that's what
        # it's for?
        plot_example_dynamics=plot_dynamics,

        make_plots=True, connectome_weight_plots=False
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
    max_iters = 200

    plot_root = plot_dir.parent
    # TODO TODO also step around fixed_thr/wAPLKC from previous tuning, e.g. a
    # similar model w/o the PN<>APL weights?
    # TODO try a hypergrid stepping thr and APL independently too? (w/ same steps
    # for PN>APL and APL>PN weights)
    params = fit_and_plot_mb_model(plot_root, plot_dirname=plot_dir.name,
        orn_deltas=orn_deltas, verbose=True, try_cache=not ignore_existing,
        try_lr_cache=try_lr_cache, **kws, **output_kws,
        return_olfsysm_vars=return_olfsysm_vars, delete_pretime=delete_pretime,
        max_iters=max_iters
    )
    if tuned_only:
        warn('skipping all PN<>APL weight sweeping, because tuned_only=True')
        return

    # TODO TODO check for signs output dirs below are older than tuned dir, and regen if
    # so

    # TODO TODO TODO also implement scaling pre-tuning now

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

    # TODO TODO also try at a few diff wKCAPL/wAPLKC scales? (paper hemibrain was
    # wAPLKC=4.63/wKCAPL=0.00252, for ref)
    # TODO TODO worth trying w/ change in how thr is calculated, so it's not
    # relative to spont in? (how to even do? what's that look like w/ other things
    # same?)
    # TODO worth trying w/ a couple diff sp_factor_pre_APL? (1.5 / 3.0?)
    # TODO TODO these are ultimately sorted before plots, right?
    if corners_only:
        steps = [min(STEPS), max(STEPS)]
    elif corners_and_tuned:
        steps = [min(STEPS), 1.0, max(STEPS)]
    else:
        steps = STEPS

    # TODO should i pass these in? or basically doesn't matter?
    # TODO TODO or should it all be a separate call, where we tune on these (maybe like
    # whatever i currently think makes sense as input to natmix_data/analysis.py)?
    natmix_deltas = natmix_orn_deltas()
    natmix_panel_vals = natmix_deltas.columns.get_level_values('panel')
    kiwi_deltas = natmix_deltas.loc[:, natmix_panel_vals == 'kiwi']
    control_deltas = natmix_deltas.loc[:, natmix_panel_vals == 'control']

    # TODO TODO add CLI arg to select which panels to run (at least for the parameter
    # stepping), and skip others?

    panel2orn_deltas = {
        'megamat': orn_deltas,
        'kiwi': kiwi_deltas,
        'control': control_deltas,
    }
    # TODO provide warning / fail early if we can estimate we won't have enough disk
    # space (if return_dynamics / plot_example_dynamics)?
    extra_panel_dirs = []
    extra_panels_root = plot_root / EXTRA_PANELS_DIRNAME
    if len(set(panel2orn_deltas.keys()) - {'megamat'}) > 0:
        extra_panels_root.mkdir(exist_ok=True)

    for panel, deltas in tqdm(panel2orn_deltas.items(), unit='panel'):
        if panel == 'megamat':
            panel_plot_dir = plot_dir
        else:
            panel_plot_dir = extra_panels_root / panel / plot_dir.name
            panel_plot_dir.mkdir(exist_ok=True, parents=True)
            extra_panel_dirs.append(panel_plot_dir)
            # TODO TODO TODO still save outputs of tuned model among root (so i can more
            # easily analyse w/ natmix_data/analysis.py, for one)
            # TODO or link to all the outputs under the 1/1 (=tuned) dir at least?

        print()
        print(f'panel: {panel}')
        print(f'saving outputs under: {panel_plot_dir}')
        print('stepping wAPLPN & wPNAPL around tuned values:')
        for ap, pa in tqdm(list(product(steps, steps)), unit='param-combo'):
            step = dict(thr_and_apl_kws)
            if not scale_pre_tuning:
                step['wAPLPN'] = wAPLPN_scale * ap
                # TODO (delete?) make sure this one is also in format_model_params
                # output, esp when not derivable from wAPLPN (not using
                # format_model_params for now)
                step['wPNAPL'] = wPNAPL_scale * pa
            else:
                # TODO TODO need to do anything else?
                step['wAPLPN'] = ap
                step['wPNAPL'] = pa

            # TODO TODO actually try per bouton/claw[/KC?] inh dynamics (prob both
            # for KC claws and PN boutons)
            # TODO (delete?) only matter if i also have a per-bouton/claw synaptic
            # depression (or some other kind of saturation?) add that too?

            param_dirname = (
                ('scale-pre-tuning_True_' if scale_pre_tuning else '') +
                # TODO factor ', ' stripping into option for format_weights (/ another
                # fn?) with orig values scaled, could get duplicate plot dir names, b/c
                # some
                # values too small for .3f float format
                # TODO change float formatting in format_weights to fix that
                # TODO delete?
                #format_weights(step['wAPLPN'], 'wAPLPN').strip(', ') + '_' +
                #format_weights(step['wPNAPL'], 'wPNAPL').strip(', ')
                format_weights(ap, 'wAPLPN').strip(', ') + '_' +
                format_weights(pa, 'wPNAPL').strip(', ')
            ).replace('=', '-')
            param_dir = panel_plot_dir / param_dirname

            print(f'{param_dir.name}')

            # TODO (esp when implementing pre-tuning scaling?) add option to ignore LR
            # cache (should be a CLI arg that also applies to above tuning calls too)
            step_params = fit_and_plot_mb_model(panel_plot_dir,
                plot_dirname=param_dir.name, try_cache=not ignore_existing,
                try_lr_cache=try_lr_cache, orn_deltas=deltas,
                scale_pre_tuning=scale_pre_tuning, **step, **kws,
                **output_kws
            )

            # TODO update to work when stepping arbitrary # of parameters
            if ap == 1 and pa == 1:
                if panel == 'megamat':
                    # TODO put behind a checks flag?
                    #
                    # should inspire confidence the same should be (approximately) true
                    # for the extra_panels cases, even though we don't currently have
                    # thet separate tuned outputs there
                    assert_fit_and_plot_outputs_equal(plot_root, params, step_params,
                        plot_root2=panel_plot_dir,
                        # TODO would need to test something else if we also do
                        # pretune scaling as one case of script run w/ no args
                        # (this assumes it's just one or the other)
                        ignore_tuning_params=not scale_pre_tuning
                    )
                    print('megamat tuned values matched those in 1/1 scaled output dir')
                else:
                    stepped_output_dir = panel_plot_dir / step_params['output_dir']
                    assert stepped_output_dir.is_dir(), f'{stepped_output_dir=}'
                    for x in stepped_output_dir.glob('*'):
                        # TODO assert no non-link/dir stuff exists in this dir?
                        if x.is_symlink():
                            # just so we don't need to worry about any existing symlinks
                            # being stale or causing issues w/ symlink creation
                            x.unlink()

                        # also work if link is just a directory to put the links in
                        # (currently, no. raises OSError)?
                        # TODO modify symlink to support that behavior (+test)?
                        #link = panel_plot_dir
                        #
                        link = panel_plot_dir / x.name
                        # TODO no need to check link doesn't already exist (as something
                        # other than a link), right? will currently fail if it's a dir
                        # that exists. what about if iti's a file?
                        #
                        # relative symlinks, as i'd want, and as default
                        symlink(x, link)

            # TODO TODO TODO do panel specific analysis? allow caller to pass in? (maybe
            # dict of panel -> fns to run [taking plot_dir, and saving plots to within
            # the dir?])
            # TODO TODO or better fit to do those things in the analyze_outputs part?
            # TODO TODO whether via external call or hardcoding in here, want to
            # incorporate some of the natmix_data/analysis.py response class analysis
            # into the kiwi/control cases here


def main():
    # TODO TODO add CLI arg(s) to skip a certain # of param-combos and/or panels
    #
    # TODO add -q/--quiet option to hide basically everything but the progress bars
    # TODO but ideally still log all output to a file...
    #
    # RawTextHelpFormatter is to preserve the newlines
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
    parser.add_argument('-r', '--ignore-lr-cache', action='store_true', help='uses no '
        'values in any cache of learning rates (e.g. sp_lr_coeff) (but will still write'
        ' to them in the same circumstances as normal)'
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
    parser.add_argument('-R', '--reverse', action='store_true', help='iterates through '
        'MODEL_TUNE_KWS in reverse order, as an easy way to test multiple cases if '
        'code downstream of tuning is causing earlier cases to fail.'
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
    parser.add_argument('-C', '--corners-and-tuned', action='store_true',
        help='only analyzes the parameters combos where both parameters are either min/'
        'tuned[=1]/max scale. does 5 more combos than -c/--corners-only. mostly for '
        'testing.'
    )
    # TODO is this also happening in initial fit_mb_model calls w/o me requesting it (i
    # think so. ig it's default?)? do i want that? maybe assert this flag isn't passed
    # unless -o/--only-analyze-outputs then?
    # TODO TODO try to make sure this does work w/ --tuned-only tho (maybe still via
    # analyze_outputs?)
    parser.add_argument('-p', '--plot-dynamics', action='store_true',
        help='loads and plots saved dynamics (in the analyze_outputs call, so '
        'works with -o/--only-analyze-outputs)'
    )
    # TODO reword weights->parameters after adding support for stepping more than just
    # weights
    parser.add_argument('-P', '--scale-pre-tuning', action='store_true',
        help='Instead of post-tuning stepping of weights (where response rate will '
        'often be [potentially far] outside target range), scales weights before tuning'
        ' (currently leaving threshold same as in tuned).'
    )
    args = parser.parse_args()
    ignore_existing = args.ignore_existing
    try_lr_cache = not args.ignore_lr_cache
    save_dynamics = args.save_dynamics
    tuned_only = args.tuned_only
    reverse = args.reverse
    only_analyze_outputs = args.only_analyze_outputs
    corners_only = args.corners_only
    corners_and_tuned = args.corners_and_tuned
    plot_dynamics = args.plot_dynamics
    scale_pre_tuning = args.scale_pre_tuning

    assert not (corners_only and corners_and_tuned), 'only pick one of -c or -C'

    if only_analyze_outputs:
        assert not (save_dynamics or tuned_only or ignore_existing), \
            'all of these incompatible with -o/--only-analyze-outputs'

    # TODO TODO is `step_model_pn_apl -C -i -d` really not regenerating dynamics
    # for tuned dirs, or am i tripping? fix if it isn't
    # TODO TODO and why is `-t -d` running the model again? (no -i)

    # TODO is this required to see when we are saving figs (think so)? change so that's
    # not the case (and for saving other things, if necessary)?
    al_util.verbose = True

    # TODO TODO TODO also analyze natmix (ea/eb/binary/kmix, same for control) mix
    # suppression (and plot avg of metric(s) of that across steps, and do same for yangs
    # real and synthetic odors) (here or be factoring some code out of here and then
    # calling in those scripts? prob here)
    # TODO want to refactor at all, so other code can call the stepping (with their own
    # stats / analysis passed in? how?)

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

    plot_dir_list: List[Path] = list()

    model_kws = list(MODEL_TUNE_KWS)
    if reverse:
        model_kws = model_kws[::-1]

    for kws in model_kws:
        # TODO delete / verbose?
        print(f'{kws=}')

        plot_dirname = format_model_params(kws, exclude=same_in_all)
        plot_dir = plot_root / plot_dirname
        assert plot_dir not in plot_dir_list, f'duplicate {plot_dir=}'
        plot_dir_list.append(plot_dir)

        # still want to build plot_dir_list above, as it's used for loop below, even if
        # only analyzing existing outputs
        if only_analyze_outputs:
            continue

        step_pn_apl_weights_around_tuned(plot_dir, orn_deltas, kws,
            ignore_existing=ignore_existing, try_lr_cache=try_lr_cache,
            save_dynamics=save_dynamics, plot_dynamics=plot_dynamics,
            tuned_only=tuned_only, corners_only=corners_only,
            corners_and_tuned=corners_and_tuned, scale_pre_tuning=scale_pre_tuning
        )

    if tuned_only:
        warn('not calling analyze_outputs on any output directory, because '
            '-t/--tuned-only. stepped (subdirectory) outputs may be older than tuned '
            'outputs (out of date).'
        )
        return

    # TODO TODO should scale_pre_tuning also apply to calls below?
    # TODO want to sort those outputs separately somehow, either way?

    # TODO delete / verbose?
    print()

    analyze_kws = dict(
        # if only_analyze_outputs=False, the same plots should have been created in
        # calls above
        plot_dynamics=(plot_dynamics and only_analyze_outputs),
        corners_only=corners_only,
        corners_and_tuned=corners_and_tuned,
    )
    for plot_dir in plot_dir_list:
        # TODO TODO some plots that compare tuned values for same stats across elements
        # of this plot list? maybe w/ and w/o connectome APL? (similar to
        # model_yang_mixtures.py plots)
        # TODO and maybe a version where it's one facet per tuned model, and then a
        # point for each step? at least for the one where it's scaled pre-tuning?
        # or still only display as matrix plots?
        analyze_outputs(plot_dir, **analyze_kws)

        # TODO why not exist? (still an issue? in a fresh install?)
        # (was it just when only_analyze_outputs, but we hadn't actually generated
        # output dirs for any of these panels yet?)
        extra_panels_dir = plot_root / EXTRA_PANELS_DIRNAME
        # TODO warn if this doesn't exist (under some circumstances?)
        if not extra_panels_dir.is_dir():
            continue

        for panel_dir in extra_panels_dir.glob('*'):
            if not panel_dir.is_dir():
                # TODO warn if (certain ones? under certain circumstances?) missing?
                continue

            panel_plot_dir = panel_dir / plot_dir.name
            if not panel_plot_dir.is_dir():
                warn(f'{panel_plot_dir=} was not a directory!')
                continue
            # TODO warn if different set in either of these? or in these vs above?
            analyze_outputs(panel_plot_dir, **analyze_kws)


if __name__ == '__main__':
    main()

