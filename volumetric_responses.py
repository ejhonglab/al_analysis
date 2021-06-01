#!/usr/bin/env python3

import os
from os.path import join, split, exists, splitext
from pprint import pprint
from collections import Counter
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import yaml

from hong2p import util, thor, viz


def yaml_data2pin_lists(yaml_data):
    """
    Pins used as balances can be part of these lists despite not having a corresponding
    odor in 'pins2odors'.
    """
    return [x['pins'] for x in yaml_data['pin_sequence']['pin_groups']]


def yaml_data2odor_lists(yaml_data):
    """Returns a list-of-lists of dictionary representation of odors.


    Each dictionary will have at least the keys 'name' and 'log10_conc'.

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


def format_odor_list(odor_list, delim=' + ', **kwargs):
    """Takes list of dicts representing odors for one trial to pretty str.
    """
    odor_strs = [format_odor(x, **kwargs) for x in odor_list]
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


def main():
    skip_if_experiment_plot_dir_exists = True
    #skip_if_experiment_plot_dir_exists = False

    plot_fmt = 'svg'
    # If True, the directory name containing (date, fly, thorimage_dir) information will
    # also be in the prefix for each of the plots saved within that directory (harder to
    # lose track in image viewers / after copying, but more verbose).
    prefix_plot_fnames = False

    # Since some of the pilot experiments had 6 planes (but top 5 should be the
    # same as in experiments w/ only 5 total), and that last plane largely doesn't
    # have anything measurably happening. All steps should have been 12um, so picking
    # top n will yield a consistent total heightof the volume.
    n_top_z_to_analyze = 5

    dff_vmin = 0
    dff_vmax = 3.0

    ax_fontsize = 7

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
        ('2021-05-05', 1),

        # TODO probably delete butanal + acetone here (cause possible conc mixup for i
        # think butanal)
        ('2021-05-10', 1),

        ('2021-05-11', 1),
        ('2021-05-18', 1),
    ]

    keys_and_paired_dirs = util.date_fly_list2paired_thor_dirs(experiment_keys,
        verbose=True
    )
    fly_processing_time_data = []
    for (date, fly_num), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:
        fly_before = time.time()

        experiment_id = format_thorimage_dir(thorimage_dir)
        experiment_basedir = util.to_filename(experiment_id, period=False)

        # Created below after we decide whether to skip a given experiment based on the
        # experiment type, etc.
        experiment_dir = join(plot_fmt, experiment_basedir)

        if skip_if_experiment_plot_dir_exists and exists(experiment_dir):
            print(f'skipping analysis for this experiment because {experiment_dir} '
                'exists\n'
            )
            continue

        def suptitle(title, fig=None):
            if fig is None:
                fig = plt.gcf()

            fig.suptitle(f'{experiment_id}\n{title}')

        def savefig(fig, desc):
            basename = util.to_filename(desc) + plot_fmt

            if prefix_plot_fnames:
                fname_prefix = experiment_basedir + '_'
                basename = fname_prefix + basename

            fig.savefig(join(experiment_dir, basename))

        single_plane_fps, xy, z, c, n_flyback, _, xml = thor.load_thorimage_metadata(
            thorimage_dir, return_xml=True
        )

        notes = thor.get_thorimage_notes_xml(xml)
        parts = notes.split()
        yaml_path = None
        for p in parts:
            if p.endswith('.yaml'):
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

                break

        assert yaml_path is not None
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        odor_lists = yaml_data2odor_lists(data)

        os.makedirs(experiment_dir, exist_ok=True)

        # NOTE: converting to list-of-str FIRST, so that each element will be
        # hashable, and thus can be counted inside `remove_consecutive_repeats`
        odor_strs = [format_odor_list(x) for x in odor_lists]
        try:
            odor_order, n_repeats = remove_consecutive_repeats(odor_strs)
        except AssertionError:
            print('REMOVE_CONSECUTIVE_REPEATS FAILED')
            continue

        print('n_repeats:', n_repeats)
        print('odor_order:')
        pprint(odor_order)

        before = time.time()

        thorsync_df = thor.load_thorsync_hdf5(thorsync_dir)

        load_hdf5_s = time.time() - before

        bounding_frames = thor.assign_frames_to_odor_presentations(thorsync_df,
            thorimage_dir
        )

        assert len(bounding_frames) == len(odor_lists)

        # this fps if just the xy fps i believe, so need to also include number of z
        # slices (including flyback)
        volumes_per_second = single_plane_fps / (z + n_flyback)

        print('single_plane_fps:', single_plane_fps)
        print('volumes_per_second:', volumes_per_second)
        print('1 / volumes_per_second:', 1 / volumes_per_second)

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

                baseline = movie[start_frame:(first_odor_frame - 1)].mean(axis=0)

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


    # TODO check whether this still can cause a crash if there are a bunch of windows
    # open / fail gracefully there
    #plt.show()

    #print('processing time data:')
    #pprint(fly_processing_time_data)
    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

