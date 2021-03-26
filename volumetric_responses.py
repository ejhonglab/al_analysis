#!/usr/bin/env python3

from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import tifffile

import hong2p.util as u


def main():
    # TODO TODO TODO is the weird behavior of this script, including the weird
    # min / max values, partially cause i perhaps have some bug in my
    # hong2p/thor2tiff command line tool? maybe in recent changes to write_tiff?
    data_dir = '/home/tom/mb_team/raw_data'
    #thorimage_dir = join(data_dir, '2021-02-23/1/fn_001')
    thorimage_dir = join(data_dir, '2021-03-07/1/t2h_single_plane')
    #tiff_fname = /3_left.tif')

    movie = u.read_movie(thorimage_dir)

    single_plane_fps, xy, z, c, n_flyback, _ = u.load_thorimage_metadata(
        thorimage_dir
    )

    pre_odor_s = 5.0
    odor_s = 1.0
    post_odor_s = 14.0

    trial_s = pre_odor_s + odor_s + post_odor_s
    # this fps if just the xy fps i believe, so need to also include number of z
    # slices (including flyback)
    volumes_per_second = single_plane_fps / (z + n_flyback)
    # frames means volumes here
    frames_per_trial = int(round(trial_s * volumes_per_second))

    print('single_plane_fps:', single_plane_fps)
    print('volumes_per_second:', volumes_per_second)
    print('1 / volumes_per_second:', 1 / volumes_per_second)
    # frames means volumes here
    print('frames_per_trial:', frames_per_trial)

    '''
    odor_order = [
        '5% cleaning ammonia in water',
        'geranyl acetate',
        'methyl acetate',
        'phenylacetaldehyde',
        '2,3-butanedione',
        'trans-2-hexenal',
        'methyl salicylate',
        '2-butanone',
    ]
    '''
    odor_order = [
        'trans-2-hexenal',
    ]

    n_repeats = 3
    n_trials = n_repeats * len(odor_order)

    print('n_trials:', n_trials)

    # TODO TODO TODO why are these different? try w/ thorsync
    print('n_trials * frames_per_trial:', n_trials * frames_per_trial)
    print('n_frames:', len(movie))

    #'''
    anat_baseline = movie.mean(axis=0)
    baseline_fig, baseline_axs = plt.subplots(1, z)
    for d in range(z):
        # TODO fix hack that added support for 2d case
        try:
            ax = baseline_axs[d]
        # 'AxesSubplot' object does not support indexing
        except TypeError:
            ax = baseline_axs

        ax.imshow(anat_baseline[d], vmin=0, vmax=9000)
        ax.set_axis_off()
        """
        print(anat_baseline[d].min())
        print(anat_baseline[d].mean())
        print(anat_baseline[d].max())
        print()
        """
    baseline_fig.suptitle('average of whole movie')
    #plt.show()
    #'''

    for i, o in enumerate(odor_order):
        vmin = 0
        #vmax = 3.0
        vmax = 5.0

        #max_fig, max_axs = plt.subplots(n_repeats, z)
        mean_fig, mean_axs = plt.subplots(n_repeats, z)

        for n in range(n_repeats):
            # frames means volumes here
            start_frame = (i * n) * frames_per_trial
            end_frame = start_frame + frames_per_trial

            # at ~2.22... volumes per second, this should just fit within
            # pre_odor_s
            # TODO try 1 though, or try offset by -1 frames, in case that
            # difference between 215 frames and the 216 expected could get us in
            # trouble here
            baseline_volumes = 2

            baseline = movie[start_frame:start_frame + baseline_volumes
                ].mean(axis=0)

            '''
            print('baseline.min():', baseline.min())
            print('baseline.mean():', baseline.mean())
            print('baseline.max():', baseline.max())
            '''
            # hack to make df/f values more reasonable
            #baseline = baseline + 10.

            # TODO TODO why is baseline.max() always the same???

            # TODO try in window. maybe less noise.
            dff = ((movie[start_frame + baseline_volumes:end_frame] - baseline)
                / baseline
            )

            response_volumes = 1
            #response_volumes = 2
            # TODO off by one at start?
            mean_dff = dff[:response_volumes].mean(axis=0)

            max_dff = dff.max(axis=0)
            '''
            print(max_dff.min())
            print(max_dff.mean())
            print(max_dff.max())
            print()
            '''

            '''
            vmin = 0
            vmax = 3.0

            fig, axs = plt.subplots(1, z)
            '''
            for d in range(z):
                '''
                ax = max_axs[n, d]
                im = ax.imshow(max_dff[d], vmin=vmin, vmax=vmax)
                '''

                try:
                    ax = mean_axs[n, d]
                # TODO fix hack
                except IndexError:
                    ax = mean_axs[n]

                im = ax.imshow(mean_dff[d], vmin=vmin, vmax=vmax)
                ax.set_axis_off()

                # include microns
                #ax.set_title('')

            #cb_ax = max_fig.add_axes([0.83, 0.1, 0.02, 0.8])
            #cbar = max_fig.colorbar(im, cax=cb_ax)

            #plt.suptitle(f'{o} trial {n}')
            #plt.show()

        #max_fig.suptitle(f'{o} (max)')

        mean_fig.suptitle(f'{o}')
        #plt.show()

    plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

