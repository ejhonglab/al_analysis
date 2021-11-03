#!/usr/bin/env python3

from os.path import join, split
from pprint import pprint
import glob

import numpy as np
import pandas as pd
import xarray as xr
import ijroi

from hong2p import util, thor
from hong2p import suite2p as s2p
from suite2p.io.utils import get_plane_ops_and_folder_lists
from suite2p.io import compute_dydx


def main():
    #thorimage_dir = '/home/tom/2p_data/raw_data/2021-05-05/1/ehex_and_1-6ol'
    #analysis_dir = thorimage_dir
    thorimage_dir = '/mnt/d1/2p_data/raw_data/2021-04-28/1/butanal_and_acetone'
    analysis_dir = thorimage_dir.replace('raw_data', 'analysis_intermediates')

    #
    '''
    #roi_path = join(analysis_dir, '07171-00119-00079.roi')

    for roi_path in glob.glob(join(analysis_dir, '*.roi')):

        print('roi_path:', split(roi_path)[1])

        with open(roi_path, 'rb') as f:
            roi = ijroi.read_roi(f, points_only=False)

        print(roi.name)
        print(roi.c)
        print(roi.z)
        print(roi.t)
        print()

    import ipdb; ipdb.set_trace()
    '''
    #

    '''
    # default=True (inside load_s2p[_combined]_outputs)
    good_only = True
    # default=False
    merge_inputs = True
    # default=False
    merge_outputs = False

    traces, roi_stats, ops, merges = s2p.load_s2p_combined_outputs(analysis_dir,
        good_only=good_only, merge_inputs=merge_inputs, merge_outputs=merge_outputs
    )
    iscell = s2p.load_iscell(s2p.get_suite2p_combined_dir(analysis_dir))

    # u = unmerged
    unmerged_combined_dir = join(analysis_dir, 'suite2p_unmerged', 'combined')
    utraces, uroi_stats, uops, umerges = s2p.load_s2p_outputs(unmerged_combined_dir,
        good_only=good_only, merge_inputs=merge_inputs, merge_outputs=merge_outputs
    )
    uiscell = s2p.load_iscell(unmerged_combined_dir)

    assert np.array_equal(uiscell, iscell[:len(uiscell)])

    if merge_inputs and merge_outputs:
        assert (
            sum([len(s.get('imerge', [])) == 0 for s in roi_stats.values()]) ==
            sum([len(s.get('imerge', [])) == 0 for s in uroi_stats.values()])
        )

        assert (
            sum([len(s.get('imerge', [])) > 0 for s in roi_stats.values()]) ==
            traces.shape[1] - utraces.shape[1]
        )

        # 29 should be the index of the first ROI created via merging
        assert np.array_equal(
            utraces.loc[:, roi_stats[29]['imerge']].mean(axis=1), traces.loc[:, 29]
        )

    import ipdb; ipdb.set_trace()
    '''

    '''
    roipath = join(analysis_dir, 'RoiSet.zip')

    movie = thor.read_movie(thorimage_dir)

    #name_and_roi_list = ijroi.read_roi_zip(roipath)
    name_and_roi_list = ijroi.read_roi_zip(roipath, points_only=False)
    #import ipdb; ipdb.set_trace()

    masks = util.ijrois2masks(name_and_roi_list, movie.shape[-3:],
        as_xarray=True
    )
    import ipdb; ipdb.set_trace()
    # TODO TODO TODO at least add flag to make this also do what remerge_suite2p_merged
    # is doing (i.e. picking the plane with the strongest signal and not using the
    # components of the masks in other planes). probably just refactor merge handling to
    # allow ijroi and suite2p handling to use the same logic for this.
    merged = util.merge_ijroi_masks(masks, check_no_overlap=True)
    '''

    #merged = util.ijroi_masks(analysis_dir, thorimage_dir)
    #import ipdb; ipdb.set_trace()

    traces, roi_stats, ops, merges = s2p.load_s2p_combined_outputs(analysis_dir)

    # TODO TODO compare to stuff loaded from individual plane folders + incorporate into
    # unit tests of this invert_* fn. need to figure out how to invert ROI numbering too
    # though...
    #inverted = {
    #    i: s2p.invert_combined_view_offset(ops, r) for i, r in roi_stats.items()
    #}

    # TODO TODO TODO fix how shape of these are currently coming out as shape of
    # combined view, not shape of a single plane (in x, y). maybe by integrating the
    # invert_combined_view_offset step into these fns?
    #rois = {i: s2p.suite2p_roi_stat2roi(r, ops) for i, r in roi_stats.items()}
    #rois = s2p.suite2p_stat2rois(roi_stats, ops)

    traces, rois = s2p.remerge_suite2p_merged(traces, roi_stats, ops, merges,
        verbose=True
    )

    # TODO make a unit test in hong2p comparing these outputs to those computed as above
    # (after refactoring that in to the hong2p suite2p module + probably splitting so
    # there is also a fn to just compute the offset, and another to apply it)
    '''
    s2p_dir = join(analysis_dir, 'suite2p')
    plane_ops_list, plane_folders = get_plane_ops_and_folder_lists(s2p_dir)
    dy, dx = compute_dydx(plane_ops_list)
    '''

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

