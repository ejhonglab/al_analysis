#!/usr/bin/env python3

from pathlib import Path
from pprint import pprint
from typing import Dict, Any

import numpy as np
from suite2p.io import BinaryFile
# TODO how to summarize what this is actually doing?
from suite2p.extraction.masks import create_masks
from tqdm import tqdm

from hong2p.roi import extract_traces
from hong2p.suite2p import load, load_ops, load_stat, load_traces, load_s2p_masks


def main():
    matrix = Path('/mnt/matrix')
    remy_data_dir = (
        matrix / 'Remy-Data/projects/odor_space_collab/processed_data'
    )

    # TODO copy some of this test data locally? the tiff is ~20GB tho...
    # TODO parse Experiment.xml from here to get expected x and y at least?
    #
    example_kc_soma_fly_dir = remy_data_dir / '2022-10-10/1/megamat0'

    # (just going to load the per-plane .bin files for now, which is also what my
    # analysis uses. check with her there's no reason to use either of the others)
    #
    # TODO and did she compute the TIFF from the suite2p movie? maybe i should check it
    # against raw? but at least how i run suite2p, range of output movie is still useful
    # for me doing dF/F stuff... her TIFFs have similar range and dtype to mine?
    # (same question for whichever output i end up using)
    #
    # TODO load this tiff (in chunks? extracting w/in each? it's 19.0 GB on disk)?
    # this TIFF (which I can load via ImageJ -> import -> TIFF virtual stack) is of
    # shape z=16, t=9076, x/y=256
    #tiff_path = example_kc_soma_fly_dir / 'Image_001_001.tif'
    # TODO or load the (38.1 GB on disk) h5 file under s2p_root?
    # Image_001_001_els__d1_256_d2_256_d3_16_order_F_frames_9076_.h5

    # each s2p_root / 'suite2p/plane<x>' dir also has both an ops.npy and a stat.npy, as
    # well as F.npy). s2p_root has an ops.npy (that probably doesn't have anything the
    # ops in the subdirs doesn't have), but nothing else.
    s2p_root = example_kc_soma_fly_dir / 'source_extraction_s2p/suite2p'

    # TODO delete? or probably use.
    #
    # i think the one in s2p_combined probably has the outputs i want
    # (actually, probably not, at least for simple test. may need it later to align
    # against iscell labels, but also seems like i might need to invert the combined
    # tiling to get proper coordinates of ROIs)
    s2p_combined = s2p_root / 'combined'

    # TODO delete?
    combined_ops = load_ops(s2p_combined)
    #

    # TODO delete? traces are separate, and those are what i want anyway, right?
    # anything in stat besides ROIs?
    # TODO check that i was right that combined stat has ROIs offset for display in
    # 2d? cause if not (if ROIs equiv to those i'm loading from plane dirs), could just
    # load from combined... still need to load movie from plane dirs tho? prob no real
    # advantage...
    # TODO maybe use load_s2p_masks instead of load_stat (or delete)
    combined_stat = load_stat(s2p_combined)
    #

    # TODO what is Fall.mat, and does remy actually use it? would it be useful for me?
    # is it only in combined dir?

    # TODO try to see if same as from plane dirs (maybe delete after)
    combined_F = load_traces(s2p_combined)
    #

    print(f'analyzing suite2p outputs under:\n{s2p_root}')

    plane_dirs = sorted(s2p_root.glob('plane*/'),
        key=lambda x: int(x.name[len('plane'):])
    )

    # xml says
    # <ZStage name="ThorZPiezo" steps="16" ...
    # which is consistent w/ 16 plane dirs
    nplanes = len(plane_dirs)
    print(f'{nplanes=}')
    print()

    last_idx = 0
    for plane_dir in tqdm(plane_dirs, desc='re-analyzing suite2p plane dirs',
        unit='plane'):

        print(f'{plane_dir.name}/')
        # could assert all > 0
        masks = load_s2p_masks(plane_dir)
        n_rois, ny, nx = masks.shape

        binary_plane_movie_path = plane_dir / 'data.bin'
        with BinaryFile(ny, nx, binary_plane_movie_path) as bf:
            # "nImg x Ly x Lx", meaning T x frame height x frame width
            plane_movie = bf.data

        # type(plane_movie)=<class 'numpy.ndarray'>
        # plane_movie.dtype=dtype('int16')
        #
        # (for first plane)
        # plane_movie.max()=12568
        # plane_movie.min()=-501
        n_frames, y2, x2 = plane_movie.shape
        assert ny == y2 and nx == x2

        # TODO TODO anything i can do to make this faster? time this? (option to) tqdm
        # inside this fn?
        # TODO print time this takes, so i can start trying to improve it
        #
        # "footprints: should be of shape ([Z,] Y, X, #-ROIs)"
        # masks.shape is currently (Y, X, #-ROIs), so need to move first to last
        #
        # transposing outputs to get into same (ROIs, T) shape as F
        traces = extract_traces(plane_movie, np.moveaxis(masks, 0, -1), _sum=True,
            verbose=True
        ).T
        # TODO delete. (_sum=True calculation is correct, and fn should that should be
        # default behavior, perhaps moving ROI normalizing in there, if i haven't
        # already)
        #mtraces = extract_traces(plane_movie, np.moveaxis(masks, 0, -1), verbose=True).T

        F = load_traces(plane_dir)

        # TODO delete
        # what explains drastically different scales between these
        # calculations?
        # (that i had not yet normalized weights like s2p does. i am doing that now
        # [and using _sum=True to extract_traces, which i'll probably make new default
        # behavior, deleting that flag. should now assert input footprints are already
        # normalized, or do itself (maybe only if a flag set)], and seems to match)
        #
        # ipdb> F.min()
        # 0.0
        # ipdb> F.mean()
        # 656.58856
        # ipdb> F.max()
        # 6770.4116
        #
        # ipdb> mtraces.mean()
        # 0.23007914442832478
        # ipdb> mtraces.max()
        # 3.697587728500366
        # ipdb> traces.min()
        # 0.0
        #
        # ipdb> traces.mean()
        # 15078.466809254693
        # ipdb> traces.max()
        # 242325.109375
        #
        assert np.allclose(F, traces)
        # ok cool i can repro now (using _sum=True and normalized weights)
        # TODO TODO check my own (boolean) calc couldn't stand to be changed
        # slightly (or see if it's equiv to normalizing and using my new extraction
        # code. matter if it's not exactly? dF/F still the same, regardless of absolute
        # scale, i assume? test this [comparing against _sum=True w/ normalized boolean
        # mask inputs?)

        assert F.shape == (n_rois, n_frames)

        assert np.array_equal(combined_F[last_idx:(last_idx + len(F))], F)
        last_idx += len(F)

        print()

    # TODO (if i want to also load + use odor metadata and compute corrs, but probably
    # not otherwise) load probably iscell_soma.npy or maybe iscell_good_xid0.npy [maybe
    # both?] to subset ROIs


if __name__ == '__main__':
    main()

