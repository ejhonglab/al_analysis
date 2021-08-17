#!/usr/bin/env python3

import argparse
import os
from os.path import join, split, exists, splitext, expanduser
from pprint import pprint
from collections import Counter, defaultdict
import warnings
import time
import shutil
import traceback
import subprocess
import pickle
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import yaml
from suite2p import run_s2p
import ijroi
# TODO find a nicer perceptually linear colormap from here?
#import colorcet as cc

from hong2p import util, thor, viz
from hong2p import suite2p as s2p
from hong2p.suite2p import LabelsNotModifiedError, LabelsNotSelectiveError
from hong2p.util import shorten_path


SAVE_FIGS = True
PLOT_FMT = 'svg'
# TODO also use this cmap for trial df/f images (or at least not viridis?)
CMAP = 'plasma'

analysis_intermediates_root = util.analysis_intermediates_root()

odor2abbrev = {
    'methyl salicylate': 'MS',
    'hexyl hexanoate': 'HH',
    'furfural': 'FUR',
    '1-hexanol': 'HEX',
    '1-octen-3-ol': 'OCT',
    '2-heptanone': '2H',
    'acetone': 'ACE',
    'butanal': 'BUT',
    'ethyl acetate': 'EA',
    'ethyl butyrate': 'EB',
    'ethyl hexanoate': 'EH',
    'hexyl acetate': 'HA',
}
odors_without_abbrev = set()

# TODO replace similar fn (if still exists?) already in hong2p? or use the hong2p one?
# (just want to prefer the "fast" data root)
def get_analysis_dir(date, fly_num, thorimage_basedir):
    return join(analysis_intermediates_root,
        util.get_fly_dir(date, fly_num), thorimage_basedir
    )


def savefig(fig, experiment_fig_dir, desc, close=True):
    # If True, the directory name containing (date, fly, thorimage_dir) information will
    # also be in the prefix for each of the plots saved within that directory (harder to
    # lose track in image viewers / after copying, but more verbose).
    prefix_plot_fnames = False
    basename = util.to_filename(desc) + PLOT_FMT

    if prefix_plot_fnames:
        experiment_basedir = split(experiment_fig_dir)[0]
        fname_prefix = experiment_basedir + '_'
        basename = fname_prefix + basename

    if SAVE_FIGS:
        fig.savefig(join(experiment_fig_dir, basename))

    if close:
        plt.close(fig)


FAIL_INDICATOR_PREFIX = 'FAILING_'

# TODO maybe take error object / traceback and save stack trace actually?
def make_fail_indicator_file(analysis_dir, suffix, err=None):
    """Makes empty file (e.g. FAILING_suite2p) in analysis_dir, to mark step as failing
    """
    path = Path(join(analysis_dir, f'{FAIL_INDICATOR_PREFIX}{suffix}'))
    if err is None:
        path.touch()
    else:
        err_str = ''.join(traceback.format_exception(type(err), err, err.__traceback__))
        path.write_text(err_str)


def _list_fail_indicators(analysis_dir):
    return glob.glob(join(analysis_dir, FAIL_INDICATOR_PREFIX + '*'))


def last_fail_suffixes(analysis_dir):
    """Returns if an analysis_dir has any fail indicators and str with their suffixes
    """
    suffixes = [
        split(x)[1][len(FAIL_INDICATOR_PREFIX):]
        for x in _list_fail_indicators(analysis_dir)
    ]
    if len(suffixes) == 0:
        return False, None
    else:
        return True, ' AND '.join(suffixes)


def clear_fail_indicators(analysis_dir):
    """Deletes any fail indicator files in analysis_dir
    """
    for f in _list_fail_indicators(analysis_dir):
        os.remove(f)


def stimulus_yaml_from_thorimage_xml(xml):
    """Returns absolute path to stimulus YAML file from note field in ThorImage XML.

    XML just contains a manually-entered path relative to where the olfactometer code
    that generated it was run, but assuming it was copied to the appropriate location,
    this absolute path should exist.
    """
    stimfile_root = util.stimfile_root()

    notes = thor.get_thorimage_notes_xml(xml)

    yaml_path = None
    parts = notes.split()
    for p in parts:
        p = p.strip()
        if p.endswith('.yaml'):
            if yaml_path is not None:
                raise ValueError('encountered multiple *.yaml substrings!')

            yaml_path = p

    assert yaml_path is not None

    # TODO change data + delete this hack
    if '""' in yaml_path:
        date_str = '_'.join(yaml_path.split('_')[:2])
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

    yaml_path = join(stimfile_root, yaml_path)
    assert exists(yaml_path), f'{yaml_path}'

    return yaml_path


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


def thorimage_xml2yaml_info_and_odor_lists(xml):
    """Returns yaml_path, yaml_data, odor_lists
    """
    yaml_path = stimulus_yaml_from_thorimage_xml(xml)

    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    odor_lists = yaml_data2odor_lists(yaml_data)

    return yaml_path, yaml_data, odor_lists


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

    # TODO add something like (<n>) to subsequent n_repeats occurence of the same odor
    # (e.g. solvent control) (OK without as long as we are prefixing filenames with
    # presentation index, but not-OK if we ever wanted to stop that)

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


def index2single_odor_name(index, name_conc_delim='@'):
    odors = {x.split(name_conc_delim)[0].strip() for x in index}
    odors = {x for x in odors if x != 'solvent'}
    assert len(odors) == 1
    return odors.pop()


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
    """Returns a sorted list of dicts representing odors for one trial

    Name takes priority over concentration, so with the same set of odor names in each
    trial's odor_list, this should produce a consistent ordering (and same indexes can
    be used assuming equal length of all)
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


# TODO did i already implement this logic somewhere in this file? use this code if so
def n_odors_per_trial(odor_lists):
    """
    Assumes same number of odors on each trial
    """
    len_set = {len(x) for x in odor_lists}
    assert len(len_set) == 1
    return len_set.pop()


def odor_lists2names_and_conc_ranges(odor_lists):
    """
    Gets a hashable representation of the odors and each of their concentration ranges.
    Ex: ( ('butanal', (-5, -4, -3)), ('acetone', (-5, -4, -3)) )

    Tuples of concentrations measured will be sorted in ascending order.

    What is returned doesn't have any information about order of presentation, and
    experiments are only equivalent if that didn't change, nor other decisions about the
    trial structure (typically these things have stayed pretty constant though)
    """
    # So that we don't return both:
    # ( ('a', (-2, -1)), ('b', (-2, -1)) ) and...
    # ( ('b', (-2, -1)), ('a', (-2, -1)) )
    odor_lists = [sort_odor_list(ol) for ol in odor_lists]

    def name_i(i):
        name_set = {x[i]['name'] for x in odor_lists}
        assert len(name_set) == 1, ('assuming odor_lists w/ same odor (names, not concs'
            ') at each trial'
        )
        return name_set.pop()

    def conc_range_i(i):
        concs_i_including_solvent = {x[i]['log10_conc'] for x in odor_lists}
        concs_i = sorted([x for x in concs_i_including_solvent if x is not None])
        return tuple(concs_i)

    n = n_odors_per_trial(odor_lists)
    names_and_conc_ranges = tuple((name_i(i), conc_range_i(i)) for i in range(n))
    return names_and_conc_ranges


def is_pairgrid(odor_lists):
    # TODO TODO reimplement in a way that actually checks there are all pairwise
    # concentrations, rather than just assuming so if there are 2 odors w/ 3 non-zero
    # concs each
    names_and_conc_ranges = odor_lists2names_and_conc_ranges(odor_lists)

    return len(names_and_conc_ranges) == 2 and (
        all([len(cs) == 3 for _, cs in names_and_conc_ranges])
    )


def is_reverse_order(odor_lists):
    o1_list = [o1 for o1, _ in odor_lists if o1['log10_conc'] is not None]

    def get_conc(o):
        return o['log10_conc']

    return get_conc(o1_list[0]) > get_conc(o1_list[-1])


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

        # TODO refactor
        if odor1['name'] in odor2abbrev:
            odor1['name'] = odor2abbrev[odor1['name']]

        if odor2['name'] in odor2abbrev:
            odor2['name'] = odor2abbrev[odor2['name']]
        # TODO also do solvent->PFO

        #

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


def separate_names_and_concs_tuples(names_and_concs_tuple):
    """
    Takes input like ( ('butanal', (-5, -4, -3)), ('acetone', (-5, -4, -3)) ) to
    output like ('acetone', 'butanal'), ((-5, -4, -3), (-5, -4, -3))
    """
    names = tuple(n for n, _ in names_and_concs_tuple)
    concs = tuple(cs for _, cs in names_and_concs_tuple)
    return names, concs


def odor_names2final_concs(**paired_thor_dirs_kwargs):
    """Returns dict of names tuple -> concentrations tuple

    Loops over same directories as main analysis
    """
    keys_and_paired_dirs = util.paired_thor_dirs(verbose=False,
        **paired_thor_dirs_kwargs
    )

    names2final_concs = dict()
    for (_, _), (thorimage_dir, _) in keys_and_paired_dirs:

        xml = thor.get_thorimage_xmlroot(thorimage_dir)
        ti_time = thor.get_thorimage_time_xml(xml)

        yaml_path, yaml_data, odor_lists = \
            thorimage_xml2yaml_info_and_odor_lists(xml)

        try:
            names_and_concs_tuple = odor_lists2names_and_conc_ranges(odor_lists)

        # TODO change error from assertionerror if gonna catch it...
        except AssertionError:
            continue

        if not is_pairgrid(odor_lists):
            continue

        names, curr_concs = separate_names_and_concs_tuples(names_and_concs_tuple)
        names2final_concs.update({names: curr_concs})

    return names2final_concs


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
        # previous frame in the baseline. This slicing seems to work the same for both
        # numpy arrays and pandas DataFrames of the same shape (at least for stuff where
        # the rows are the default RangeIndex, which is all I tested)
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
    """
    Args:
        odor_order_with_repeats: if passed, will be passed to odor_lists_to_multiindex
    """

    if odor_order_with_repeats is not None:
        assert len(bounding_frames) == len(odor_order_with_repeats)

    # TODO return as pandas series if odor_order_with_repeats is passed, with odor
    # index containing that data? test this would also be somewhat natural in 2d/3d case

    trial_stats = []

    for trial_traces in split_into_trials(traces, bounding_frames):
        curr_trial_stats = stat(trial_traces)

        # TODO TODO adapt to also work in case input is a movie
        # TODO TODO also work in 1d input case (i.e. if just data from single ROI was
        # passed)
        # traces.shape[1] == # of ROIs
        assert curr_trial_stats.shape == (traces.shape[1],)

        trial_stats.append(curr_trial_stats)

    trial_stats = np.stack(trial_stats)

    if odor_order_with_repeats is None:
        index = None
    else:
        index = odor_lists_to_multiindex(odor_order_with_repeats)

    trial_stats_df = pd.DataFrame(index=index, data=trial_stats)
    trial_stats_df.index.name = 'trial'

    # TODO maybe implement somthing that also works w/ xarrays? maybe make my own
    # function that dispatches to the appropriate concatenation function accordingly?

    # Since np.stack (probably as well as other similar numpy functions) converts pandas
    # stuff to numpy, and pd.concat doesn't work with numpy arrays.
    if hasattr(traces, 'columns'):
        trial_stats_df.columns = traces.columns
    else:
        # TODO maybe only do this if a certain dimension if 2d input (time x ROIs) is
        # passed in (but is it possible to name these correctly based on dimension in
        # general, or even for any particular dimension? if not, don't name ever)
        trial_stats_df.columns.name = 'roi'

    return trial_stats_df


def plot_roi_stats_odorpair_grid(single_roi_series, show_repeats=False, ax=None):

    assert single_roi_series.index.names == ['odor1', 'odor2', 'repeat']

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

    # TODO delete
    # TODO try some colormaps from colorcet too
    #for cmap in ['plasma', 'magma', 'cividis', 'inferno', 'gray']:
    #    fig, ax = plt.subplots()
    #

    common_matshow_kwargs = dict(
        ax=ax, title=title, transpose_sort_key=index2single_odor_name, cmap=CMAP,
        # not working
        #vmin=-0.2, vmax=2.0
    )

    if show_repeats:
        cax = viz.matshow(roi_df.droplevel('repeat'), group_ticklabels=True,
            **common_matshow_kwargs
        )
    else:
        # 'odor2' is the one on the row axis, as one level alongside 'repeat'
        # TODO not sure why sort=False seems to be ignored... bug?
        mean_df = roi_df.groupby('odor2', sort=False).mean()
        mean_df.sort_index(key=index_sort_key, inplace=True)
        cax = viz.matshow(mean_df, **common_matshow_kwargs)

    return cax


s2p_not_run = []
iscell_not_modified = []
iscell_not_selective = []
no_merges = []
# TODO kwarg to allow passing trial stat fn in that includes frame rate info as closure,
# for picking frames in a certain time window after onset and computing mean?
# TODO TODO TODO to the extent that i need to convert suite2p rois to my own and do my
# own trace extraction, maybe just modify my fork of suite2p to save sufficient
# information in combined view stat.npy to invert the tiling? unless i really can find a
# reliable way to invert that step...
def suite2p_trace_plots(analysis_dir, bounding_frames, odor_order_with_repeats,
    plot_dir=None):
    """
    Returns whether plots were generated.
    """
    try:
        traces, roi_stats, ops, merges = s2p.load_s2p_combined_outputs(analysis_dir)

    except (IOError, LabelsNotModifiedError, LabelsNotSelectiveError) as e:

        if isinstance(e, IOError):
            s2p_not_run.append(analysis_dir)

        elif isinstance(e, LabelsNotModifiedError):
            iscell_not_modified.append(analysis_dir)

        elif isinstance(e, LabelsNotSelectiveError):
            iscell_not_selective.append(analysis_dir)

        print(f'NOT generating suite2p plots because {e}')
        return False

    if len(merges) == 0:
        no_merges.append(analysis_dir)

    verbose = True
    traces, rois = s2p.remerge_suite2p_merged(traces, roi_stats, ops, merges,
        verbose=False
    )

    # TODO delete
    '''
    plt.plot(traces, linestyle='None', marker='.')
    plt.gca().set_prop_cycle(None)
    plt.plot(traces, alpha=0.3)
    ax = plt.gca()
    ax.set_xlabel('Frame')
    ax.set_ylabel('F')
    short_id = shorten_path(analysis_dir)
    ax.set_title(short_id)
    plt.show()
    import ipdb; ipdb.set_trace()
    '''
    #

    trial_stats = compute_trial_stats(traces, bounding_frames, odor_order_with_repeats)

    # TODO delete
    '''
    trial_stats_from_arr = compute_trial_stats(traces, bounding_frames,
        odor_order_with_repeats
    )
    '''
    #

    # TODO check numbering is consistent w/ suite2p numbering in case where there is
    # some merging (currently handling will treat it incorrectly)
    # TODO also plot roi / outline of roi on corresponding [mean?] plane / maybe with
    # other planes for context?
    if SAVE_FIGS and plot_dir is not None:
        roi_dir = join(plot_dir, 'roi')
        os.makedirs(roi_dir, exist_ok=True)

    include_rois = True

    for roi in trial_stats.columns:
        if include_rois:
            fig, axs = plt.subplots(nrows=2, ncols=1)
            ax = axs[0]
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # TODO more globally appropriate title? (w/ fly and other metadata. maybe number
        # flies within each odor pair / sequentially across days, and just use that as a
        # short ID?)

        #fig.suptitle(f'ROI {roi}')

        roi1_series = trial_stats.loc[:, roi]
        cax = plot_roi_stats_odorpair_grid(roi1_series, ax=ax)

        # TODO TODO TODO (assuming colors are saved / randomization is seeded and easily
        # reproducible in suite2p) copy suite2p coloring for plotting ROIs (or at least
        # make my own color scheme consistent across all plots w/in experiment)
        # TODO TODO TODO separate unified ROI plot (like combined view, but maybe all in
        # one row for consistency with my other plots) with same color scheme
        # TODO TODO have scale in single ROI plots and ~"combined" view be the same
        # (each plot pixel same physical size)
        if include_rois:
            roi_stat = roi_stats[roi]
            s2p.plot_roi(roi_stat, ops, ax=axs[1])

        # TODO TODO make colorbar ~size of matrix + move closer + show less decimals in
        # ticks on it
        viz.add_colorbar(fig, cax)

        fig.tight_layout()

        if plot_dir is not None:
            savefig(fig, roi_dir, str(roi))

    # TODO TODO [option to] use non-weighted footprints (as well as footprints that have
    # had the binary closing operation applied before uniform weighting)

    # TODO [option to] exclude stuff that doesn't have any 0's in iscell / warn [/ open
    # gui for labelling?]

    print('generated plots based on suite2p traces')
    return True


# TODO maybe refactor (part of?) this to hong2p.suite2p
failed_suite2p_dirs = []
def run_suite2p(thorimage_dir, analysis_dir, overwrite=False):

    # TODO expose if_exists kwarg as run_suite2p  kwarg?
    # This will create a TIFF <analysis_dir>/raw.tif, if it doesn't already exist
    util.thor2tiff(thorimage_dir, output_dir=analysis_dir, if_exists='ignore',
        verbose=False
    )

    suite2p_dir = s2p.get_suite2p_dir(analysis_dir)
    if exists(suite2p_dir):
        # TODO TODO but maybe actually just return here? because if it failed with the
        # user-level options the first time, won't it just fail this run too?  (or are
        # there another random, small, fixeable errors?)
        '''
        # Since we currently depend on the contents of this directory existing for
        # analysis, and it's generated as one of the last steps in what suite2p does, so
        # many errors will cause this directory to not get generated.
        if not exists(s2p.get_suite2p_combined_dir(analysis_dir)):
            print('suite2p directory existed, but combined subdirectory did not. '
                're-running suite2p!'
            )
            overwrite = True
        '''

        if overwrite:
            shutil.rmtree(suite2p_dir)
            os.mkdir(suite2p_dir)
        else:
            return

    ops_dir = expanduser('~/.suite2p/ops')
    # NOTE: {ops_dir}/ops_user.npy is what gets written w/ the gui button to save new
    # defaults
    ops_file = join(ops_dir, 'ops_user.npy')
    # TODO print the path of the ops*.npy file we ultimately load ops from
    ops = s2p.load_s2p_ops(ops_file)

    # TODO TODO perhaps try having threshold_scaling depend on plane, and use run_plane
    # instead? (decrease on lower planes, where it's harder to find stuff generally)

    # TODO maybe use suite2p's options for ignoring flyback frames to ignore depths
    # beyond those i've now settled on?

    data_specific_ops = s2p.suite2p_params(thorimage_dir)
    for k, v in data_specific_ops.items():
        assert k in ops
        ops[k] = v

    # Only available in my fork of suite2p
    ops['exclude_filenames'] = (
        'ChanA_Preview.tif',
        'ChanB_Preview.tif',
        'lastpair_avg_mean_dff.tif',
    )

    db = {
        #'data_path': [thorimage_dir],
        # TODO TODO TODO update suite2p to take substrs / globs to ignore input files on
        # (at least for TIFFs specifically, to ignore other TIFFs in input dir)
        'data_path': [analysis_dir],
    }

    # TODO probably build up a list of directories for which suite2p was run for the
    # first time, and print at end, so the ROIs can be manually
    # inspected/categorized/merged

    success = False
    try:
        # TODO actually care about ops_end?
        ops_end = run_s2p(ops=ops, db=db)
        success = True

    except Exception as err:
        traceback.print_exc()
        failed_suite2p_dirs.append(analysis_dir)

        make_fail_indicator_file(analysis_dir, 'suite2p', err)

        # TODO should i do this or should i just use the other plane data?
        # or have some necessary steps not run by the point the likely error is
        # encountered (usually a ValueError about ROIs not being found, and usually in
        # one of the deeper planes), even for the earlier planes?
        #print(f'Removing suite2p created {suite2p_dir} because run_s2p failed')
        #shutil.rmtree(suite2p_dir)

    if success:
        # NOTE: this is not changing the iscell.npy files in the plane<x> folders,
        # as I'm trying to avoid dealing with those folders at all as much as possible
        combined_dir = s2p.get_suite2p_combined_dir(analysis_dir)
        s2p.mark_all_suite2p_rois_good(combined_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true',
        help='only *complete* main loop 1 time, for faster testing'
    )
    args = parser.parse_args()

    quick_test_only = args.test

    # TODO try to make this more granular, so the suite2p / non-suite2p analysis can be
    # skipped separately (latter should be generated on first run, but former will at
    # least require some manual steps [and the initial automated suite2p run may also
    # fail])
    # TODO TODO also make more granular in the sense that we don't necessarily want to
    # run suite2p on glomerli diagnostics stuff, etc
    # TODO TODO probably make trial (+ avg across trial) dff image plots in their own
    # directory, so it's easier to just check if that directory exists
    # (or maybe check if any svg are in exp dir root? maybe assume plots are made if we
    # have a suite2p directory?)
    skip_if_experiment_plot_dir_exists = False

    # TODO TODO probably make another category or two for data marked as failed (in the
    # breakdown of data by pairs * concs at the end)
    retry_previously_failed = False

    # Whether to only analyze experiments sampling 2 odors at all pairwise
    # concentrations (the main type of experiment for this project)
    analyze_pairgrids_only = True

    # If there are multiple experiments with the same odors, only the data from the most
    # recent concentrations for those odors will be analyzed.
    final_concentrations_only = True

    analyze_reverse_order = False

    # Will be set False if analyze_pairgrids_only=True
    analyze_glomeruli_diagnostics = False

    # Whether to analyze any single plane data that is found under the enumerated
    # directories.
    analyze_2d_tests = False

    non_suite2p_analysis = True
    ignore_existing_nonsuite2p_outputs = False

    # Whether to run the suite2p pipeline (generates outputs among raw data, in
    # 'suite2p' subdirectories)
    do_suite2p = True

    # Will just skip if already exists if False
    overwrite_suite2p = False

    analyze_suite2p_outputs = True
    # TODO TODO add flags to only analyze stuff considered "done" in breakdown at end
    # (and probably also that has some merges at least)

    # Since some of the pilot experiments had 6 planes (but top 5 should be the
    # same as in experiments w/ only 5 total), and that last plane largely doesn't
    # have anything measurably happening. All steps should have been 12um, so picking
    # top n will yield a consistent total height of the volume.
    n_top_z_to_analyze = 5

    ignore_bounding_frame_cache = False

    # TODO shorten any remaining absolute paths if this is True, so we can diff outputs
    # across installs w/ data in diff paths
    print_full_paths = False

    dff_vmin = 0
    dff_vmax = 3.0

    ax_fontsize = 7

    if analyze_pairgrids_only:
        analyze_glomeruli_diagnostics = False

    # TODO TODO TODO skip all data that doesn't have final concentrations of ethyl
    # hexanoate (i went down) (and in general do this for anything where concentration
    # changed)

    # TODO skip just the butanal and acetone experiment here, which seems bad
    #('2021-05-05', 1),

    # TODO TODO TODO probably delete butanal + acetone here (cause possible conc
    # mixup for i think butanal) (unless i can clarify what mixup might have been
    # from notes + it seems clear enough from data it didn't happen)
    #('2021-05-10', 1),

    # TODO TODO TODO fix how 2021-07-21/1/EH_and_1H seems to have the stimulus file
    # for acetone+butanal (from previous exp w/ same fly) in notes

    # Using this in addition to ignore_prepairing in call below, because that changes
    # the directories considered for Thor[Image/Sync] pairing, and would cause that
    # process to fail in some of these cases.
    bad_thorimage_dirs = [
        # skipping for now just because responses in df/f images seem weak. compare to
        # others tho.
        '2021-03-08/2',

        # Just has glomeruli diagnostics. No real data.
        '2021-05-11/1',

        # Both flies only have glomeruli diagnostics.
        '2021-06-07',

        '2021-06-24/1/msl_and_hh',

        # Frame <-> time assignment is currently failing for all the real data from this
        # day.
        #'2021-06-24/2',

        # TODO was it the 1 or the 2 fly that had the problem where every plane seemed
        # the same?
        #'2021-07-21/TODO',

    ]

    bad_suite2p_analysis_dirs = [
        # skipping for now cause suite2p output looks weird (for both recordings)
        '2021-03-07/2',

        # Just a few glomeruli visible in (at least the last odor) trials
        '2021-05-24/1/1oct3ol_and_2h',

        # TODO TODO revisit this one
        '2021-05-24/2/ea_and_etb',

        # TODO why?
        # eb_and_ea recording looks bad (mainly just one glomerulus? lots of stuff seems
        # to come in 4s at times rather than 3s [well really just one group of 4], and
        # generally not much signal)
        '2021-05-25/1/eb_and_ea',

        # TODO TODO TODO add other stuff that was bad
    ]
    # TODO refactor so this is not necessary if possible
    # This will get populated w/ full paths matching path fragments in variable above,
    # to filter out these paths in printing out status of suite2p analyses at the end.
    full_bad_suite2p_analysis_dirs = []

    # TODO maybe factor this to / use existing stuff in hong2p in place of this
    os.makedirs(analysis_intermediates_root, exist_ok=True)

    common_paired_thor_dirs_kwargs = dict(
        start_date='2021-03-07', ignore=bad_thorimage_dirs, ignore_prepairing=('anat',)
    )

    if final_concentrations_only:
        names2final_concs = odor_names2final_concs(**common_paired_thor_dirs_kwargs)

    # TODO is there currently anything preventing suite2p_trace_plots from trying to run
    # on non-pair stuff apart from the fact that i haven't labeled outputs of the auto
    # suite2p runs that may or may not have happened on them?
    # (if not, probably add such logic)

    experiment_plot_dirs = []
    experiment_analysis_dirs = []

    # TODO TODO TODO note that 2021-07-(21,27,28) contain reverse-order experiments.
    # indicate this fact in the plots for these experiments!!!

    keys_and_paired_dirs = util.paired_thor_dirs(verbose=True, print_skips=False,
        print_fast=False, print_full_paths=print_full_paths,
        **common_paired_thor_dirs_kwargs
    )

    main_start_s = time.time()
    exp_processing_time_data = []

    names_and_concs2analysis_dirs = defaultdict(list)
    failed_assigning_frames_to_odors = []

    seen_stimulus_yamls2thorimage_dirs = defaultdict(list)

    for (date, fly_num), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:

        if 'glomeruli' in thorimage_dir and 'diag' in thorimage_dir:
            if not analyze_glomeruli_diagnostics:
                print('skipping because experiment is just glomeruli diagnostics\n')
                continue

            is_glomeruli_diagnostics = True
        else:
            is_glomeruli_diagnostics = False

        analysis_dir = get_analysis_dir(date, fly_num, split(thorimage_dir)[1])
        os.makedirs(analysis_dir, exist_ok=True)
        experiment_analysis_dirs.append(analysis_dir)

        if retry_previously_failed:
            clear_fail_indicators(analysis_dir)
        else:
            has_failed, suffixes_str = last_fail_suffixes(analysis_dir)
            if has_failed:
                print(f'skipping because previously failed {suffixes_str}\n')
                continue

        exp_start = time.time()

        single_plane_fps, xy, z, c, n_flyback, _, xml = thor.load_thorimage_metadata(
            thorimage_dir, return_xml=True
        )

        if not analyze_2d_tests and z == 1:
            print('skipping because experiment is single plane\n')
            continue

        experiment_id = shorten_path(thorimage_dir)
        experiment_basedir = util.to_filename(experiment_id, period=False)

        # Created below after we decide whether to skip a given experiment based on the
        # experiment type, etc.
        # TODO rename to experiment_plot_dir or something
        plot_dir = join(PLOT_FMT, experiment_basedir)

        # TODO refactor scandir thing to not_empty or something
        if (skip_if_experiment_plot_dir_exists and exists(plot_dir)
            and any(os.scandir(plot_dir))):

            print(f'skipping because {plot_dir} exists\n')
            continue

        def suptitle(title, fig=None):
            if fig is None:
                fig = plt.gcf()

            fig.suptitle(f'{experiment_id}\n{title}')

        def exp_savefig(fig, desc, **kwargs):
            savefig(fig, plot_dir, desc, **kwargs)

        if SAVE_FIGS:
            os.makedirs(plot_dir, exist_ok=True)

            # (to remove empty directories at end)
            experiment_plot_dirs.append(plot_dir)

        yaml_path, yaml_data, odor_lists = thorimage_xml2yaml_info_and_odor_lists(xml)
        print('yaml_path:', shorten_path(yaml_path, n_parts=2))

        # TODO also exclude stuff where stimuli were not pairs. maybe just try/except
        # the code extracting stimulus info in here? or separate fn, run first, to
        # detect *if* we are dealing w/ pair-grid data?
        if not is_glomeruli_diagnostics:
            # So that we can count how many flies we have for each odor pair (and
            # concentration range, in case we varied that at one point)
            names_and_concs_tuple = odor_lists2names_and_conc_ranges(odor_lists)

            if final_concentrations_only:
                names, curr_concs = separate_names_and_concs_tuples(
                    names_and_concs_tuple
                )

                if (names in names2final_concs and
                    names2final_concs[names] != curr_concs):

                    print('skipping because not using final concentrations for '
                        f'{names}\n'
                    )
                    continue

        pulse_s = float(int(yaml_data['settings']['timing']['pulse_us']) / 1e6)
        if pulse_s < 3:
            print(f'skipping because odor pulses were {pulse_s} (<3s) long (old)\n')
            continue

        if is_pairgrid(odor_lists):
            if not analyze_reverse_order and is_reverse_order(odor_lists):
                print('skipping because a reverse order experiment\n')
                continue
        else:
            if analyze_pairgrids_only:
                print('skipping because not a pair grid experiment\n')
                continue

        if yaml_path in seen_stimulus_yamls2thorimage_dirs:
            short_yaml_path = shorten_path(yaml_path, n_parts=2)
            raise ValueError(f'stimulus yaml {short_yaml_path} already seen in:\n'
                f'{seen_stimulus_yamls2thorimage_dirs[yaml_path][0]}/Experiment.xml'
            )
        seen_stimulus_yamls2thorimage_dirs[yaml_path].append(thorimage_dir)

        if not is_glomeruli_diagnostics:
            names_and_concs2analysis_dirs[names_and_concs_tuple].append(analysis_dir)

        # NOTE: converting to list-of-str FIRST, so that each element will be
        # hashable, and thus can be counted inside `remove_consecutive_repeats`
        odor_order_with_repeats = [format_odor_list(x) for x in odor_lists]
        odor_order, n_repeats = remove_consecutive_repeats(odor_order_with_repeats)

        # TODO use that list comprehension way of one lining this? equiv for sets?
        name_lists = [[o['name'] for o in os] for os in odor_lists]
        for ns in name_lists:
            for n in ns:
                if n not in odor2abbrev:
                    odors_without_abbrev.add(n)
        del name_lists

        before = time.time()

        bounding_frame_yaml_cache = join(analysis_dir, 'trial_bounding_frames.yaml')

        if ignore_bounding_frame_cache or not exists(bounding_frame_yaml_cache):
            # TODO TODO don't bother doing this if we only have suite2p analysis left to
            # do, and the required output directory doesn't exist / ROIs haven't been
            # manually filtered / etc
            try:
                bounding_frames = thor.assign_frames_to_odor_presentations(thorsync_dir,
                    thorimage_dir
                )
                assert len(bounding_frames) == len(odor_order_with_repeats)

                # TODO TODO move inside assign_frames_to_odor_presentations
                # Converting numpy int types to python int types, and tuples to lists,
                # for (much) nicer YAML output.
                bounding_frames = [ [int(x) for x in xs] for xs in bounding_frames]

                # TODO use yaml instead. save under analysis_dir
                with open(bounding_frame_yaml_cache, 'w') as f:
                    yaml.dump(bounding_frames, f)

            # Currently seems to reliably happen iff we somehow accidentally also image
            # with the red channel (which was allowed despite those channels having gain
            # 0 in the few cases so far)
            except AssertionError as err:
                traceback.print_exc()
                print()

                failed_assigning_frames_to_odors.append(thorimage_dir)

                make_fail_indicator_file(analysis_dir, 'assign_frames', err)

                continue

        else:
            with open(bounding_frame_yaml_cache, 'r') as f:
                bounding_frames = yaml.safe_load(f)

        # (loading the HDF5 should be the main time cost in the above fn)
        load_hdf5_s = time.time() - before

        if do_suite2p:
            run_suite2p(thorimage_dir, analysis_dir, overwrite=overwrite_suite2p)

        if analyze_suite2p_outputs:
            if not any([b in thorimage_dir for b in bad_suite2p_analysis_dirs]):
                suite2p_trace_plots(analysis_dir, bounding_frames, odor_lists,
                    plot_dir=plot_dir
                )
            else:
                full_bad_suite2p_analysis_dirs.append(analysis_dir)
                print('not making suite2p plots because outputs marked bad\n')

        if not non_suite2p_analysis:
            print()

            if quick_test_only:
                break

            continue

        if not ignore_existing_nonsuite2p_outputs:
            # Assuming that if analysis_dir has *any* plots directly inside of it, it
            # has all of what we want from non_suite2p_analysis (including any
            # intermediates that would go in analysis_dir).
            # Set ignore_existing_nonsuite2p_outputs=False to do this analysis
            # regardless, regenerating any overlapping plots.
            if len(glob.glob(join(plot_dir, f'*.{PLOT_FMT}'))) > 0:
                print('skipping non-suite2p analysis because plot dir contains '
                    f'{PLOT_FMT}\n'
                )

                if quick_test_only:
                    break

                continue

        before = time.time()

        movie = thor.read_movie(thorimage_dir)

        read_movie_s = time.time() - before

        # TODO TODO maybe make a plot like this, but use the actual frame times
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
        exp_savefig(ffavg_fig, 'ffavg')
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
        exp_savefig(baseline_fig, 'avg')
        '''

        for i, o in enumerate(odor_order):

            # TODO either:
            # - always use 2 digits (leading 0)
            # - pick # of digits from len(odor_order)
            plot_desc = f'{i + 1}_{o}'

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
                # TODO TODO maybe just add like 1e4 or something?
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

                # TODO factor to hong2p.thor
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
            close = i < len(odor_order) - 1
            # TODO need to make sure figures from earlier iterations are closed
            exp_savefig(trial_heatmap_fig, plot_desc + '_trials', close=close)

            avg_mean_dff = np.mean(trial_mean_dffs, axis=0)

            # TODO replace LHS of "and" w/ volumetric only flag if i add one
            if min(movie.shape) > 1 and i == (len(odor_order) - 1):
                # TODO factor out + check this is consistent w/ write_tiff. might wanna
                # just modify write_tiff so i can specify it's missing the T not Z
                # dimension (to the extent it matters...), which the docstring currently
                # says it doesn't support (or just explictly add singleton dimension
                # before, in here?)
                avg_mean_dff_tiff = join(analysis_dir, 'lastpair_avg_mean_dff.tif')

                # This expand_dims operation doesn't seem to have added a label to
                # slider in FIJI, but maybe FIJI still cares, and maybe the ROI manager
                # will generate labels differently? As long as I can load the ROIs w/
                # the metadata I need it shouldn't matter...
                avg_dff_for_tiff = np.expand_dims(avg_mean_dff, axis=0
                    ).astype(np.float32)

                util.write_tiff(avg_mean_dff_tiff, avg_dff_for_tiff, strict_dtype=False)

                # TODO put all ijroi stuff behind a flag like do_suite2p

                # TODO TODO auto open this roi in imagej (and maybe also open roi
                # manager), for labelling (unless ROI file exists)
                # TODO maybe also a flag to load each with existing ROI files (if
                # exists), for checking / modification

                # TODO compare tiff data to matplotlib plots that should have same data,
                # just for sanity checking that my tiff writing is working correctly

                # TODO TODO load <analysis_dir>/RoiSet.zip if exists and use as fn
                # that processes suite2p output if exists
                # TODO maybe refactor that fn to separate plotting from suite2p/ijroi
                # data source?
                '''
                ijroiset_filename = join(analysis_dir, 'RoiSet.zip')
                #print('ijrois:', ijroiset_filename)

                # TODO refactor all this ijroi loading / mask creation [+ probably trace
                # extraction too]
                name_and_roi_list = ijroi.read_roi_zip(ijroiset_filename)

                masks = util.ijrois2masks(name_and_roi_list, movie.shape[-3:],
                    as_xarray=True
                )

                # TODO also try merging via correlation/overlap thresholds?
                masks = util.merge_ijroi_masks(masks, check_no_overlap=True)

                # TODO TODO merge w/ bool masks converted from suite2p ROIS,
                # extract traces for all, and make plots derived from traces, as
                # suite2p_trace_plots currently has
                # TODO TODO perhaps also option to just analyze masks from ijrois and
                # not merge w/ suite2p stuff?

                #import ipdb; ipdb.set_trace()
                '''

            # TODO refactor this or at least the labelling portion within
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
            exp_savefig(mean_heatmap_fig, plot_desc)

        plt.close('all')

        exp_total_s = time.time() - exp_start
        exp_processing_time_data.append((load_hdf5_s, read_movie_s, exp_total_s))

        print()

        if quick_test_only:
            break


    def earliest_analysis_dir_date(analysis_dirs):
        return min(d.split(os.sep)[-3] for d in analysis_dirs)

    # TODO exclude non-pairgrid stuff from here
    print('\nOdor pair counts for all data considered (including stuff where suite2p '
        'plots not generated for various reasons):'
    )
    for names_and_concs, analysis_dirs in sorted(names_and_concs2analysis_dirs.items(),
        key=lambda x: earliest_analysis_dir_date(x[1])):

        names_and_concs_strs = []
        for name, concs in names_and_concs:
            conc_range_str = ','.join([str(c) for c in concs])
            #print(f'{name} @ {conc_range_str}')
            names_and_concs_strs.append(f'{name} @ {conc_range_str}')

        print(f'({len(analysis_dirs)}) ' + ' mixed with '.join(names_and_concs_strs))

        # TODO maybe come up with a local-file-format (part of a YAML in the raw data
        # dir?) type indicator that (current?) suite2p output just looks bad and doesn't
        # just need labelling, to avoid spending more effort on bad data?
        # (and include that as a status here probably)

        s2p_statuses = (
            'not run',
            'ROIs need manual labelling (or output looks bad)',
            'ROIs may need merging (done if not)',
            'marked bad',
            'done',
        )
        not_done_dirs = set()
        for s2p_status in s2p_statuses:

            if s2p_status == s2p_statuses[0]:
                status_dirs = [x for x in analysis_dirs if x in s2p_not_run]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[1]:
                status_dirs = [
                    x for x in analysis_dirs
                    if (x in iscell_not_modified) or (x in iscell_not_selective)
                ]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[2]:
                status_dirs = [
                    x for x in analysis_dirs if x in no_merges
                ]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[3]:
                # maybe don't show these ones?
                status_dirs = [
                    x for x in analysis_dirs if x in full_bad_suite2p_analysis_dirs
                ]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[4]:
                status_dirs = [x for x in analysis_dirs if x not in not_done_dirs]

            else:
                assert False

            show_empty_statuses = False
            if show_empty_statuses or len(status_dirs) > 0:
                print(f' - {s2p_status} ({len(status_dirs)})')
                for analysis_dir in status_dirs:
                    short_id = shorten_path(analysis_dir)
                    print(f'   - {short_id}')

        print()
    print()

    # TODO also check that all loaded data is using same stimulus program
    # (already skipping stuff with odor pulse < 3s tho)

    total_s = time.time() - main_start_s
    print(f'Took {total_s:.0f}s\n')

    # TODO probably just remove any empty directories [matching pattern?] at same level?
    # with some flag set, maybe?
    # Remove directories that were/would-have-been created to save plots/intermediates
    # for an experiment, but are empty.
    # TODO TODO try to do during loop, so Ctrl-C / other interruption is less likely to
    # leave empty directories lying around
    for d in experiment_plot_dirs + experiment_analysis_dirs:
        if not any(os.scandir(d)):
            os.rmdir(d)


    # Only actually shortens if print_full_paths=False
    def shorten_and_pprint(paths):
        if not print_full_paths:
            paths = [shorten_path(p) for p in paths]

        pprint(paths)


    def print_nonempty_path_list(name, paths, alt_msg=None):
        if len(paths) == 0:
            if alt_msg is not None:
                print(alt_msg)

            return

        print(f'{name} ({len(paths)}):')
        shorten_and_pprint(paths)
        print()


    print_nonempty_path_list(
        'failed_assigning_frames_to_odors:', failed_assigning_frames_to_odors
    )

    if do_suite2p:
        print_nonempty_path_list(
            'suite2p failed:', failed_suite2p_dirs
        )

    if analyze_suite2p_outputs:
        print_nonempty_path_list(
            'suite2p needs to be run on the following data:', s2p_not_run,
            alt_msg='suite2p has been run on all currently included data'
        )

        # NOTE: only possible if suite2p for these was run outside of this pipeline, as
        # `run_suite2p` in this file currently marks all ROIs as "good" (as cells, as
        # far as suite2p is concerned) immediately after a successful suite2p run.
        print_nonempty_path_list(
            'suite2p outputs with ROI labels not modified:', iscell_not_modified
        )

        print_nonempty_path_list(
            'suite2p outputs where no ROIs were marked bad:', iscell_not_selective
        )
        print_nonempty_path_list(
            'suite2p outputs where no ROIs were merged:', no_merges
        )

    if len(odors_without_abbrev) > 0:
        print('odors_without_abbrev:')
        pprint(odors_without_abbrev)


if __name__ == '__main__':
    main()

