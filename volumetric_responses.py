#!/usr/bin/env python3

import os
from os.path import join, split, exists, splitext, expanduser
from pprint import pprint
from collections import Counter, defaultdict
import warnings
import time
import shutil
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import yaml
from suite2p import run_s2p

from hong2p import util, thor, viz
from hong2p.suite2p import suite2p_params


def yaml_data2pin_lists(yaml_data):
    """
    Pins used as balances can be part of these lists despite not having a corresponding
    odor in 'pins2odors'.
    """
    return [x['pins'] for x in yaml_data['pin_sequence']['pin_groups']]


def yaml_data2odor_lists(yaml_data):
    """Returns a list-of-lists of dictionary representation of odors.

    Each dictionary will have at least the key 'name' and generally also 'log10_conc'.

    The i-th list contains all of the odors presented simultaneously on the i-th odor
    presentation.
    """
    pin_lists = yaml_data2pin_lists(yaml_data)
    # int pin -> dict representing odor (keys 'name', 'log10_conc', etc)
    pins2odors = yaml_data['pins2odors']

    odor_lists = []
    for pin_list in pin_lists:

        odor_list = []
        for p in pin_list:
            if p in pins2odors:
                odor_list.append(pins2odors[p])

        odor_lists.append(odor_list)

    return odor_lists


def remove_consecutive_repeats(odor_lists):
    """Returns a list-of-str without any consecutive repeats and int # of repeats.

    Wanted to also take a list-of-lists-of-dicts, where each dict represents one odor
    and each internal list represents all of the odors on one trial, but the internal
    lists (nor the dicts they contain) would not be hashable, and thus cannot work with
    Counter as-is.

    Assumed that all elements of `odor_lists` are repeated the same number of times, and
    all repeats are consecutive. Actually now as long as any repeats are to full # and
    consecutive, it is ok for a particular odor (e.g. solvent control) to be repeated
    `n_repeats` times in each of several different positions.
    """
    # In Python 3.7+, order should be guaranteed to be equal to order first encountered
    # in odor_lists.
    # TODO modify to also allow counting non-hashable stuff (i.e.  dictionaries), so i
    # can pass my (list of) lists-of-dicts representation directly
    counts = Counter(odor_lists)

    count_values = set(counts.values())
    n_repeats = min(count_values)
    without_consecutive_repeats = odor_lists[::n_repeats]

    # TODO possible to combine these two lines to one?
    # https://stackoverflow.com/questions/25674169
    nested = [[x] * n_repeats for x in without_consecutive_repeats]
    flat = [x for xs in nested for x in xs]
    assert flat == odor_lists, 'variable number or non-consecutive repeats'

    # TODO TODO TODO add something like (<n>) to subsequent n_repeats occurence of the
    # same odor (e.g. solvent control) (OK without as long as we are prefixing filenames
    # with presentation index, but not-OK if we ever wanted to stop that)

    return without_consecutive_repeats, n_repeats


def format_odor(odor_dict, conc=True, name_conc_delim=' @ ', conc_key='log10_conc'):
    """Takes a dict representation of an odor to a pretty str.

    Expected to have at least 'name' key, but will also use 'log10_conc' (or `conc_key`)
    if available, unless `conc=False`.
    """
    ostr = odor_dict['name']

    if conc_key in odor_dict:
        # TODO opt to choose between 'solvent' and no string (w/ no delim below used?)?
        # what do i do in hong2p.util fn now?
        if odor_dict[conc_key] is None:
            return 'solvent'

        if conc:
            ostr += f'{name_conc_delim}{odor_dict[conc_key]}'

    return ostr


def odordict_sort_key(odor_dict):
    name = odor_dict['name']

    # If present, we expect this value to be a non-positive number.
    # Using 0 as default for lack of 'log10_conc' key because that case should indicate
    # some type of pure odor (or something where the concentration is specified in the
    # name / unknown). '5% cleaning ammonia in water' for example, where original
    # concentration of cleaning ammonia is unknown.
    log10_conc = odor_dict.get('log10_conc', 0)

    # 'log10_conc: null' in one of the YAMLs should map to None here.
    if log10_conc is None:
        log10_conc = float('-inf')

    assert type(log10_conc) in (int, float), f'type({log10_conc}) == {type(log10_conc)}'

    return (name, log10_conc)


def sort_odor_list(odor_list):
    """Returns a sorted list of dicts representing odors for one trial.
    """
    return sorted(odor_list, key=odordict_sort_key)


def format_odor_list(odor_list, delim=' + ', **kwargs):
    """Takes list of dicts representing odors for one trial to pretty str.
    """
    odor_strs = [format_odor(x, **kwargs) for x in sort_odor_list(odor_list)]
    odor_strs = [x for x in odor_strs if x != 'solvent']
    if len(odor_strs) > 0:
        return delim.join(odor_strs)
    else:
        return 'solvent'


def format_thorimage_dir(thorimage_dir):
    prefix, thorimage_basename = split(thorimage_dir)
    prefix, fly_num = split(prefix)
    _, date = split(prefix)
    return '/'.join([date, fly_num, thorimage_basename])


# TODO maybe convert to just two column df instead and then use some pandas functions to
# convert that to index?
def odor_lists_to_multiindex(odor_lists, add_repeat_col=True, **format_odor_kwargs):
    # TODO establish order of the two(?) odors, and reorder levels in multiindex as
    # necessary
    unique_lens = {len(x) for x in odor_lists}
    if len(unique_lens) != 1:
        raise NotImplementedError

    # This one would be more straight forward to relax than the above one
    if unique_lens != {2}:
        raise NotImplementedError

    odor1_str_list = []
    odor2_str_list = []

    if add_repeat_col:
        odor_mix_counts = defaultdict(int)
        odor_mix_repeats = []

    for odor_list in odor_lists:
        # NOTE: just assuming these are in the order we want, and in an appropriately
        # consistent order, for now (probably true, tbf)
        odor1, odor2 = odor_list

        odor1_str = format_odor(odor1, **format_odor_kwargs)
        odor1_str_list.append(odor1_str)

        odor2_str = format_odor(odor2, **format_odor_kwargs)
        odor2_str_list.append(odor2_str)

        if add_repeat_col:
            odor_mix = (odor1_str, odor2_str)
            odor_mix_repeats.append(odor_mix_counts[odor_mix])
            odor_mix_counts[odor_mix] += 1

    if not add_repeat_col:
        odor_str_lists = [odor1_str_list, odor2_str_list]
        index = pd.MultiIndex.from_arrays(odor_str_lists)
        index.names = ['odor1', 'odor2']
    else:
        index = pd.MultiIndex.from_arrays([odor1_str_list, odor2_str_list,
            odor_mix_repeats
        ])
        index.names = ['odor1', 'odor2', 'repeat']

    return index


# TODO write tests for this function for the case when n_trial_length_args is empty and
# also has at least one appropriate element
# NOTE: it is important that keyword argument are after *n_trial_length_args, so they
# are interpreted as keyword ONLY arguments.
def split_into_trials(movie_length_array, bounding_frames, *n_trial_length_args,
    after_onset_only=True, squeeze=True, subtract_and_divide_baseline=True,
    exclude_last_baseline_frame=True):

    # TODO TODO if i actually might want to default to exclude_last_baseline_frame=
    # True, b/c of problems w/ my frame assignment / whatever, should i not also
    # consider that last frame for computing stats for a given trial (treating it as if
    # it were first_odor_frame)?

    # TODO probably instead raise a ValueError here w/ message indicating
    # movie_length_array probably needs to be transposed
    assert all([all([i < len(movie_length_array) for i in trial_bounds])
        for trial_bounds in bounding_frames
    ])

    # TODO this should also probably be a ValueError
    assert all([len(x) == len(bounding_frames) for x in n_trial_length_args])

    for (start_frame, odor_onset_frame, end_frame), *trial_args in zip(bounding_frames,
        *n_trial_length_args):

        # Not doing + 1 to odor_onset_frame since we only want to include up to the
        # previous frame in the baseline.
        before_onset = movie_length_array[start_frame:odor_onset_frame]
        after_onset = movie_length_array[odor_onset_frame:(end_frame + 1)]

        if subtract_and_divide_baseline:
            if not exclude_last_baseline_frame:
                baseline = movie_length_array[start_frame:odor_onset_frame]
            else:
                baseline = movie_length_array[start_frame:(odor_onset_frame - 1)]

            # TODO maybe also accept a function to compute baseline / accept appropriate
            # dimensional input [same as mean would be] to subtract directly?
            # TODO test this baselining approach works w/ other dimensional inputs too

            mean_baseline = baseline.mean(axis=0)

            after_onset = (after_onset - mean_baseline) / mean_baseline

        if after_onset_only:
            if squeeze and len(trial_args) == 0:
                yield after_onset
            else:
                yield (after_onset,) + tuple(trial_args)

        # TODO maybe delete this branch? not sure when i'd actually want to use it...
        else:
            if subtract_and_divide_baseline:
                before_onset = (before_onset - mean_baseline) / mean_baseline

            # NOTE: no need to squeeze here because always at least length 2
            yield (before_onset, after_onset) + tuple(trial_args)


# TODO also use this to compute dF/F images in loop below?
# TODO TODO try to make a test case for the above
# TODO maybe default to mean in a fixed window as below? or not matter as much since i'm
# actually using this on data already meaned within an ROI (though i could use on other
# data later)?
def compute_trial_stats(traces, bounding_frames, odor_order_with_repeats=None,
    stat=lambda x: np.max(x, axis=0)):

    if odor_order_with_repeats is not None:
        assert len(bounding_frames) == len(odor_order_with_repeats)

    # TODO return as pandas series if odor_order_with_repeats is passed, with odor
    # index containing that data? test this would also be somewhat natural in 2d/3d case

    trial_stats = []

    for trial_traces in split_into_trials(traces, bounding_frames):
        #print('trial_traces.shape:', trial_traces.shape)

        curr_trial_stats = stat(trial_traces)

        # TODO TODO adapt to also work in case input is a movie
        # TODO TODO also work in 1d input case (i.e. if just data from single ROI was
        # passed)
        # traces.shape[1] == # of ROIs
        assert curr_trial_stats.shape == (traces.shape[1],)

        trial_stats.append(curr_trial_stats)

    trial_stats = np.stack(trial_stats)

    if odor_order_with_repeats is None:
        return trial_stats
    else:
        index = odor_lists_to_multiindex(odor_order_with_repeats)

        trial_stats_df = pd.DataFrame(index=index, data=trial_stats)
        # TODO maybe only do this if a certain dimension if 2d input (time x ROIs) is
        # passed in (but is it possible to name these correctly based on dimension in
        # general, or even for any particular dimension? if not, don't name ever)
        trial_stats_df.columns.name = 'roi'
        return trial_stats_df


def plot_roi_stats_odorpair_grid(single_roi_series, show_repeats=False, ax=None):

    assert single_roi_series.index.names == ['odor1', 'odor2', 'repeat']

    #roi_df = single_roi_series.unstack(level=(0,2))
    roi_df = single_roi_series.unstack(level=0)

    def index_sort_key(level):
        # The assignment below failed for some int dtype levels, even though the boolean
        # mask dictating where assignment should happen must have been all False...
        if level.dtype != np.dtype('O'):
            return level

        sort_key = level.values.copy()

        solvent_elements = sort_key == 'solvent'
        conc_delimiter = '@'
        assert all([conc_delimiter in x for x in sort_key[~ solvent_elements]])

        # TODO share this + fn wrapping this that returns sort key w/ other place that
        # is using sorted(..., key=...)
        def parse_log10_conc(odor_str):
            assert conc_delimiter in odor_str
            parts = odor_str.split(conc_delimiter)
            assert len(parts) == 2
            return float(parts[1].strip())

        conc_keys = [parse_log10_conc(x) for x in sort_key[~ solvent_elements]]
        sort_key[~ solvent_elements] = conc_keys

        # Setting solvent to an unused log10 concentration just under lowest we have,
        # such that it gets sorted into overall order as I want (first, followed by
        # lowest concentration).
        # TODO TODO maybe just use float('-inf') or numpy equivalent here?
        # should give me the sorting i want.
        sort_key[solvent_elements] = min(conc_keys) - 1

        # TODO do i actually need to convert it back to a similar Index like the docs
        # say, or can i leave it as an array?
        return sort_key

    roi_df = roi_df.sort_index(key=index_sort_key, axis=0
        ).sort_index(key=index_sort_key, axis=1)

    title = f'ROI {single_roi_series.name}'
    #title = None

    if show_repeats:
        viz.matshow(roi_df.droplevel('repeat'), ax=ax, title=title,
            group_ticklabels=True,
        )
    else:
        # 'odor2' is the one on the row axis, as one level alongside 'repeat'
        # TODO not sure why sort=False seems to be ignored... bug?
        mean_df = roi_df.groupby('odor2', sort=False).mean()
        mean_df.sort_index(key=index_sort_key, inplace=True)
        viz.matshow(mean_df, ax=ax, title=title)


def plot_roi(roi_stat, ops, ax=None):
    #from scipy.ndimage import binary_closing

    if ax is None:
        ax = plt.gca()

    roi_img = np.zeros((ops['Ly'], ops['Lx'])) * np.nan
    xpix = roi_stat['xpix']
    ypix = roi_stat['ypix']
    print(f'xpix range: [{xpix.min(), xpix.max()}]')
    print(f'ypix range: [{ypix.min(), ypix.max()}]')

    roi_img[ypix, xpix] = roi_stat['lam']

    # TODO in future, might be nice to convert xpix / ypix and only ever make roi_img of
    # the shape of the cropped ROI (change crop_to_nonzero and fn it calls)

    cropped, ((x_min, x_max), (y_min, y_max)) = util.crop_to_nonzero(roi_img, margin=0)

    ax.imshow(cropped)
    ax.axis('off')

    #closed = binary_closing(roi_img_tmp > 0)
    #ax.contour(closed > 0, [0.5])


def load_s2p_pickle(npy_path):
    return np.load(npy_path, allow_pickle=True)


def set_suite2p_iscell_label(s2p_out_dir, roi_num, is_good):
    """
    Args:
        roi_num (int): index of ROI to change label of, according to suite2p
        s2p_out_dir (str): must directly contain a single iscell.npy file
    """
    assert is_good in (True, False)

    iscell_npy = join(s2p_out_dir, 'iscell.npy')
    iscell = load_s2p_pickle(iscell_npy)

    import ipdb; ipdb.set_trace()
    dtype_before = iscell.dtype
    # TODO TODO TODO need to index the correct column (1st of 2)
    iscell[roi_num] = float(is_good)
    assert iscell.dtype == dtype_before

    # TODO only write if diff
    # TODO TODO see how suite2p actually writes this file, in case there is anything
    # special
    import ipdb; ipdb.set_trace()
    # NOTE: if redcell.npy is used, may also need to modify that? see suite2p/io:save.py
    np.save(iscell_npy, iscell)
    import ipdb; ipdb.set_trace()


# TODO maybe wrap call that checks this (and opens suite2p if not) with something that
# also saves an empty hidden file in the directory to mark good if no modifications
# necessary? or unlikely enough? / just prompt before opening suite2p each time?
def is_iscell_modified(s2p_out_dir):
    iscell = load_s2p_pickle(join(s2p_out_dir, 'iscell.npy'))

    # Defined in suite2p/suite2p/classification/classifier.py, in kwarg to `run`.
    # `run` is called in classify.py in the same directory, which doesn't pass this
    # kwarg (so it should always be this default value).
    p_threshold = 0.5

    # TODO warn if iscell[:, 0] is all ones? + fail if all zeros?
    # TODO does this work?
    # TODO should equality be included in comparison to p_threshold?
    return not np.array_equal(
        iscell[:, 0].astype(np.bool_),
        iscell[:, 1] >= p_threshold
    )


# TODO kwarg to allow passing trial stat fn in that includes frame rate info as closure,
# for picking frames in a certain time window after onset and computing mean?
def suite2p_trace_plots(thorimage_dir, bounding_frames, odor_order_with_repeats):

    s2p_output_dir = join(thorimage_dir, 'suite2p')
    combined_dir = join(s2p_output_dir, 'combined')
    traces_path = join(combined_dir, 'F.npy')
    if not exists(traces_path):
        print(f'{traces_path} did not exist! skipping suite2p_trace_plots!')
        return

    traces = load_s2p_pickle(traces_path)
    # TODO TODO TODO are traces output by suite2p already delta F / F, or just F?
    # (seems like just F, though not i'm pretty sure there were some options for using
    # some percentile as a baseline, so need to check again)

    # TODO regarding single entries in this array (each a dict):
    # - what is 'soma_crop'? it's a boolean array but what's it mean?
    # - what is 'med'? (len 2 list, but is it a centroid or what?)
    # - where are the weights for the ROI? (expecting something of same length as xpix
    #   and ypix)? it's not 'lam', is it? and if it's 'lam', should i normalized it
    #   before using? why isn't it already normalized?
    stat = load_s2p_pickle(join(combined_dir, 'stat.npy'))

    if not is_iscell_modified(combined_dir):
        # TODO TODO TODO use modified suite2p to open gui **with appropriate data**
        print('\nIS CELL **NOT** MODIFIED\n')
    # TODO delete
    else:
        print('\nIS CELL MODIFIED!!!\n')
    return
    #

    iscell = load_s2p_pickle(join(combined_dir, 'iscell.npy'))

    ops = load_s2p_pickle(join(combined_dir, 'ops.npy')).item()

    good_rois = iscell[:, 0].astype(np.bool_)
    # TODO note, one/both of these might need to change to account for merged ROIs...
    print('# ROIs', len(good_rois))
    print('# good ROIs:', good_rois.sum())

    # Transposing because original is of shape (# ROIs, # timepoints in movie),
    # but compute_trial_stats expects first dimension to be of size # timepoints in
    # movie (so it can be used directly on movie as well).
    traces = traces.T

    trial_stats = compute_trial_stats(traces, bounding_frames, odor_order_with_repeats)

    # TODO move this before compute_trial_stats, and just check that it being a
    # dataframe (specifically w/ numbered axes), doesn't break anything inside that fn
    # (earlier so i can factor merged + iscell/not-iscell handling into its own function
    # / part of a load function, but so i can preserve ROI IDs as in suite2p using
    # pandas labels)
    trial_stats = trial_stats.iloc[:, good_rois]

    # TODO check numbering is consistent w/ suite2p numbering in case where there is
    # some merging (currently handling will treat it incorrectly)
    # TODO also plot roi / outline of roi on corresponding [mean?] plane / maybe with
    # other planes for context?

    for roi in trial_stats.columns:
        fig, axs = plt.subplots(nrows=2, ncols=1)

        #fig.suptitle(f'ROI {roi}')

        roi1_series = trial_stats.loc[:, roi]
        plot_roi_stats_odorpair_grid(roi1_series, ax=axs[0])

        roi_stat = stat[roi]
        plot_roi(roi_stat, ops, ax=axs[1])

    # TODO delete
    plt.show()
    import ipdb; ipdb.set_trace()
    #

    # TODO TODO [option to] use non-weighted footprints (as well as footprints that have
    # had the binary closing operation applied before uniform weighting)

    # TODO [option to] exclude stuff that doesn't have any 0's in iscell / warn [/ open
    # gui for labelling?]

    # TODO TODO TODO savefigs (would require some refactoring to reuse the other one...)


failed_suite2p_dirs = []
def run_suite2p(thorimage_dir, overwrite=False):
    # verbose=True is also the default
    # TODO expose if_exists kwarg as kwarg here?
    util.thor2tiff(thorimage_dir, if_exists='ignore', verbose=True)

    suite2p_dir = join(thorimage_dir, 'suite2p')

    if exists(suite2p_dir):
        if overwrite:
            shutil.rmtree(suite2p_dir)
            os.mkdir(suite2p_dir)
        else:
            return

    # TODO TODO TODO also maybe load ops from any existing suite2p subdirectory of the
    # raw data directory, rather than the ops_user.npy contents, if the former exists

    ops_dir = expanduser('~/.suite2p/ops')
    # NOTE: {ops_dir}/ops_user.npy is what gets written w/ the gui button to save new
    # defaults
    ops_file = join(ops_dir, 'ops_user.npy')
    # TODO print the path of the ops*.npy file we ultimately load ops from
    ops = load_s2p_pickle(ops_file).item()

    # TODO TODO perhaps try having threshold_scaling depend on plane, and use run_plane
    # instead? (decrease on lower planes, where it's harder to find stuff generally)

    # TODO maybe use suite2p's options for ignoring flyback frames to ignore depths
    # beyond those i've now settled on?

    data_specific_ops = suite2p_params(thorimage_dir)
    for k, v in data_specific_ops.items():
        assert k in ops
        ops[k] = v

    db = {
        'data_path': [thorimage_dir],
    }

    # TODO probably build up a list of directories for which suite2p was run for the
    # first time, and print at end, so the ROIs can be manually
    # inspected/categorized/merged

    # TODO TODO maybe check if corresponding suite2p directory exists before running?

    # TODO actually care about ops_end?
    # TODO TODO may want / need to put in a try/except
    try:
        ops_end = run_s2p(ops=ops, db=db)
    except Exception as e:
        traceback.print_exc()
        failed_suite2p_dirs.append(thorimage_dir)
        # TODO should i do this or should i just use the other plane data?
        # or have some necessary steps not run by the point the likely error is
        # encountered (usually a ValueError about ROIs not being found, and usually in
        # one of the deeper planes), even for the earlier planes?
        #print(f'Removing suite2p created {suite2p_dir} because run_s2p failed')
        #shutil.rmtree(suite2p_dir)


def main():
    skip_if_experiment_plot_dir_exists = False

    do_suite2p = True

    #overwrite_suite2p = True
    overwrite_suite2p = False
    #delete_suite2p_dir_on_err = False

    plot_fmt = 'svg'
    # If True, the directory name containing (date, fly, thorimage_dir) information will
    # also be in the prefix for each of the plots saved within that directory (harder to
    # lose track in image viewers / after copying, but more verbose).
    prefix_plot_fnames = False
    save_figs = False

    # Since some of the pilot experiments had 6 planes (but top 5 should be the
    # same as in experiments w/ only 5 total), and that last plane largely doesn't
    # have anything measurably happening. All steps should have been 12um, so picking
    # top n will yield a consistent total height of the volume.
    n_top_z_to_analyze = 5

    dff_vmin = 0
    dff_vmax = 3.0

    ax_fontsize = 7

    analyze_glomeruli_diagnostics = False

    stimfile_root = util.stimfile_root()

    # TODO TODO maybe refactor so i store the current set of things we might care to
    # analyze in like a csv file or somthing? or could i make it cheap to enumerate from
    # raw odor metadata? might be nice so i could do diff operations on this set
    # (convert to tiffs for suite2p, use here, etc) without having to either duplicate
    # the definition or cram all the stuff i might want to do behind one script (maybe
    # that's fine if i organize it well, and mainly just have a short script dispatch to
    # other functions?)
    # TODO TODO could even make such a function in hong2p.util, maybe as a similar
    # `<x>2paired_thor_dirs(...)`
    # TODO could include support for alternately doing everything after a date / within
    # a date range or something, like i might have had a parameter for in some other
    # analysis script

    experiment_keys = [
        ('2021-03-07', 1),
        ('2021-03-07', 2),
        ('2021-03-08', 1),
        ('2021-03-08', 2),
        ('2021-04-28', 1),
        ('2021-04-29', 1),
        ('2021-05-03', 1),

        # NOTE: ethyl hex. + 1-hexanol here was some of the first data I was testing
        # suite2p volumetric analysis with.
        ('2021-05-05', 1),

        # TODO probably delete butanal + acetone here (cause possible conc mixup for i
        # think butanal)
        ('2021-05-10', 1),

        ('2021-05-11', 1),
        ('2021-05-18', 1),

        ('2021-05-24', 1),
        ('2021-05-24', 2),

        ('2021-05-25', 1),
        ('2021-05-25', 2),

        # NOTE: no useful data for either fly on 2021-06-07
        ('2021-06-08', 1),
        ('2021-06-08', 2),
    ]

    keys_and_paired_dirs = util.date_fly_list2paired_thor_dirs(experiment_keys,
        verbose=True, ignore_strs=('anat',)
    )
    fly_processing_time_data = []
    for (date, fly_num), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:

        if (not analyze_glomeruli_diagnostics and
            'glomeruli_diagnostics' in thorimage_dir):
            continue

        fly_before = time.time()

        experiment_id = format_thorimage_dir(thorimage_dir)
        experiment_basedir = util.to_filename(experiment_id, period=False)

        # Created below after we decide whether to skip a given experiment based on the
        # experiment type, etc.
        experiment_dir = join(plot_fmt, experiment_basedir)

        # TODO maybe check if empty and don't skip if so?
        if skip_if_experiment_plot_dir_exists and exists(experiment_dir):
            print(f'skipping analysis for this experiment because {experiment_dir} '
                'exists\n'
            )
            continue

        if do_suite2p:
            run_suite2p(thorimage_dir, overwrite=overwrite_suite2p)

        def suptitle(title, fig=None):
            if fig is None:
                fig = plt.gcf()

            fig.suptitle(f'{experiment_id}\n{title}')

        def savefig(fig, desc):
            basename = util.to_filename(desc) + plot_fmt

            if prefix_plot_fnames:
                fname_prefix = experiment_basedir + '_'
                basename = fname_prefix + basename

            if save_figs:
                fig.savefig(join(experiment_dir, basename))

        single_plane_fps, xy, z, c, n_flyback, _, xml = thor.load_thorimage_metadata(
            thorimage_dir, return_xml=True
        )

        notes = thor.get_thorimage_notes_xml(xml)
        parts = notes.split()
        yaml_path = None
        for p in parts:
            if p.endswith('.yaml'):
                if yaml_path is not None:
                    raise ValueError('encountered multiple *.yaml substrings!')

                yaml_path = p
                # TODO change data + delete this hack
                if '""' in yaml_path:
                    date_str = '_'.join(yaml_path.split('_')[:2])
                    print('date_str:', date_str)
                    yaml_path = yaml_path.replace('""', date_str)

                # Since paths copied/pasted within Windows may have '\' as a file
                # separator character.
                yaml_path = yaml_path.replace('\\', '/')

                if not exists(join(stimfile_root, yaml_path)):
                    prefix, ext = splitext(yaml_path)
                    yaml_dir = '_'.join(prefix.split('_')[:3])
                    subdir_path = join(stimfile_root, yaml_dir, yaml_path)
                    if exists(subdir_path):
                        yaml_path = subdir_path

                print('yaml_path:', yaml_path)
                yaml_path = join(stimfile_root, yaml_path)
                assert exists(yaml_path), f'{yaml_path}'

        assert yaml_path is not None
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        odor_lists = yaml_data2odor_lists(data)

        os.makedirs(experiment_dir, exist_ok=True)

        # NOTE: converting to list-of-str FIRST, so that each element will be
        # hashable, and thus can be counted inside `remove_consecutive_repeats`
        odor_order_with_repeats = [format_odor_list(x) for x in odor_lists]
        try:
            odor_order, n_repeats = remove_consecutive_repeats(odor_order_with_repeats)
        except AssertionError:
            print('REMOVE_CONSECUTIVE_REPEATS FAILED')
            continue

        before = time.time()

        thorsync_df = thor.load_thorsync_hdf5(thorsync_dir)

        load_hdf5_s = time.time() - before

        bounding_frames = thor.assign_frames_to_odor_presentations(thorsync_df,
            thorimage_dir
        )
        assert len(bounding_frames) == len(odor_order_with_repeats)

        if do_suite2p:
            suite2p_trace_plots(thorimage_dir, bounding_frames, odor_lists)

        # TODO TODO TODO delete
        #continue
        #

        volumes_per_second = single_plane_fps / (z + n_flyback)

        before = time.time()

        movie = thor.read_movie(thorimage_dir)

        read_movie_s = time.time() - before

        # TODO TODO TODO maybe make a plot like this, but use the actual frame times
        # (via thor.get_frame_times) + actual odor onset / offset times, and see if
        # that lines up any better?
        '''
        avg = util.full_frame_avg_trace(movie)
        ffavg_fig, ffavg_ax = plt.subplots()
        ffavg_ax.plot(avg)
        for _, first_odor_frame, _ in bounding_frames:
            # TODO need to specify ymin/ymax to existing ymin/ymax?
            ffavg_ax.axvline(first_odor_frame)

        ffavg_ax.set_xlabel('Frame Number')
        savefig(ffavg_fig, 'ffavg')
        #plt.show()
        '''

        if z > n_top_z_to_analyze:
            warnings.warn(f'{thorimage_dir}: only analyzing top {n_top_z_to_analyze} '
                'slices'
            )
            movie = movie[:, :n_top_z_to_analyze, :, :]
            assert movie.shape[1] == n_top_z_to_analyze
            z = n_top_z_to_analyze

        '''
        anat_baseline = movie.mean(axis=0)
        baseline_fig, baseline_axs = plt.subplots(1, z, squeeze=False)
        for d in range(z):
            ax = baseline_axs[0, d]

            ax.imshow(anat_baseline[d], vmin=0, vmax=9000)
            ax.set_axis_off()
            """
            print(anat_baseline[d].min())
            print(anat_baseline[d].mean())
            print(anat_baseline[d].max())
            print()
            """
        suptitle('average of whole movie', baseline_fig)
        savefig(baseline_fig, 'avg')
        '''

        # TODO (optionally) tqdm this + inner loop together
        # (or is read_movie and reading hdf5 actually dominating time now?)
        for i, o in enumerate(odor_order):

            plot_desc = o
            #if 'glomeruli_diagnostics' in thorimage_dir:
            # TODO TODO either:
            # - always use 2 digits (leading 0)
            # - only do for glomeruli_diagnostics (where one digit should be fine)
            # - pick # of digits from len(odor_order)
            plot_desc = f'{i + 1}_{plot_desc}'

            def dff_imshow(ax, dff_img):
                im = ax.imshow(dff_img, vmin=dff_vmin, vmax=dff_vmax)

                # TODO TODO figure out how do what this does EXCEPT i want to leave the
                # xlabel / ylabel (just each single str)
                ax.set_axis_off()

                # part but not all of what i want above
                #ax.set_xticklabels([])
                #ax.set_yticklabels([])

                return im

            trial_heatmap_fig, trial_heatmap_axs = plt.subplots(nrows=n_repeats,
                ncols=z, squeeze=False
            )

            # Will be of shape (1, z), since squeeze=False
            mean_heatmap_fig, mean_heatmap_axs = plt.subplots(ncols=z, squeeze=False)

            trial_mean_dffs = []

            for n in range(n_repeats):
                # This works because the repeats of any particular odor were all
                # consecutive in all of these experiments.
                presentation_index = (i * n_repeats) + n

                start_frame, first_odor_frame, end_frame = bounding_frames[
                    presentation_index
                ]

                # TODO delete
                '''
                if min(movie.shape) > 1 and i == (len(odor_order) - 1) and n == 0:
                    plt.close('all')

                    # TODO maybe set off two volumes being compared
                    avmin = movie.min()
                    avmax = movie.max()

                    # this is two frames before first_odor_frame
                    # i.e. np.array_equal(movie[start_frame:(first_odor_frame + 1)][-1],
                    # fof) == True
                    curr_baseline_last_vol = movie[start_frame:(first_odor_frame - 1)][-1]
                    viz.image_grid(curr_baseline_last_vol, vmin=avmin, vmax=avmax)
                    plt.suptitle('current last baseline frame')

                    fbof = movie[first_odor_frame - 1]
                    viz.image_grid(fbof)
                    plt.suptitle('frame before odor')

                    fof = movie[first_odor_frame]
                    viz.image_grid(fof)
                    plt.suptitle('first odor frame')

                    plt.show()
                    import ipdb; ipdb.set_trace()
                    continue
                '''
                #

                # NOTE: was previously skipping the frame right before the odor frame
                # too, but I think this was mainly out of fear the assignment of the
                # first odor frame might have been off by one, but I haven't really
                # seen examples of this, after looking through a lot of examples.
                # Possible examples where the frame before first_odor_frame also has
                # some response (looking at first repeat of last odor pair in
                # volumetric stuff only):
                # - 2021-03-08/1/acetone_and_butanal
                # - 2021-03-08/1/acetone_and_butanal_redo
                # - 2021-03-08/1/1hexanol_and_ethyl_hexanoate
                # - 2021-04-28/1/butanal_and_acetone
                # (and didn't check stuff past that for the moment)
                # conclusion: need to keep ignoring the last frame until i can fix this
                # issue
                baseline = movie[start_frame:(first_odor_frame - 1)].mean(axis=0)
                #baseline = movie[start_frame:first_odor_frame].mean(axis=0)

                '''
                print('baseline.min():', baseline.min())
                print('baseline.mean():', baseline.mean())
                print('baseline.max():', baseline.max())
                '''
                # hack to make df/f values more reasonable
                # TODO still an issue?
                #baseline = baseline + 10.

                # TODO TODO why is baseline.max() always the same???
                # (is it still?)

                dff = (movie[first_odor_frame:end_frame] - baseline) / baseline

                # TODO TODO change to using seconds and rounding to nearest[higher/lower
                # multiple?] from there
                #response_volumes = 1
                response_volumes = 2

                # TODO off by one at start? (still relevant?)
                mean_dff = dff[:response_volumes].mean(axis=0)

                trial_mean_dffs.append(mean_dff)

                '''
                max_dff = dff.max(axis=0)
                print(max_dff.min())
                print(max_dff.mean())
                print(max_dff.max())
                print()
                '''

                # TODO TODO TODO factor to hong2p.thor
                # TODO actually possible for it to be non-int in Experiment.xml?
                zstep_um = int(round(float(xml.find('ZStage').attrib['stepSizeUM'])))
                #

                for d in range(z):
                    ax = trial_heatmap_axs[n, d]

                    # TODO offset to left so it doesn't overlap and re-enable
                    if d == 0:
                        # won't work until i fix set_axis_off thing in dff_imshow above
                        ax.set_ylabel(f'Trial {n + 1}', fontsize=ax_fontsize,
                            rotation='horizontal'
                        )

                    if n == 0 and z > 1:
                        curr_z = -zstep_um * d
                        ax.set_title(f'{curr_z} $\\mu$m', fontsize=ax_fontsize)

                    im = dff_imshow(ax, mean_dff[d])

            # (end loop over repeats of one odor)

            # TODO TODO see link in hong2p.viz.image_grid for ways to eliminate
            # whitespace + refactor some of this into that viz module
            hspace = 0
            wspace = 0.014

            trial_heatmap_fig.subplots_adjust(hspace=hspace, wspace=wspace)

            viz.add_colorbar(trial_heatmap_fig, im)

            suptitle(o, trial_heatmap_fig)
            # TODO TODO TODO embed number signifying order of this odor w/in overall
            # presentation order in at least the glomeruli diagistics case too (to help
            # in troubleshooting contamination) (in other files too)
            savefig(trial_heatmap_fig, plot_desc + '_trials')


            avg_mean_dff = np.mean(trial_mean_dffs, axis=0)

            # TODO TODO TODO refactor this or at least the labelling portion within
            for d in range(z):
                ax = mean_heatmap_axs[0, d]

                #if d == 0:
                #    ax.set_ylabel(f'Mean of {n_repeats} trials', fontsize=ax_fontsize,
                #        rotation='horizontal'
                #    )

                if z > 1:
                    curr_z = -zstep_um * d
                    ax.set_title(f'{curr_z} $\\mu$m', fontsize=ax_fontsize)

                im = dff_imshow(ax, avg_mean_dff[d])
            #

            mean_heatmap_fig.subplots_adjust(wspace=wspace)
            viz.add_colorbar(mean_heatmap_fig, im)

            suptitle(o, mean_heatmap_fig)
            savefig(mean_heatmap_fig, plot_desc)

        fly_total_s = time.time() - fly_before
        fly_processing_time_data.append((load_hdf5_s, read_movie_s, fly_total_s))

        print()

    if do_suite2p:
        print('failed_suite2p_dirs:')
        pprint(failed_suite2p_dirs)

    # TODO check whether this still can cause a crash if there are a bunch of windows
    # open / fail gracefully there
    #plt.show()

    #print('processing time data:')
    #pprint(fly_processing_time_data)
    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

