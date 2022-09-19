#!/usr/bin/env python3

from os.path import join

import numpy as np
#import cv2
from scipy.ndimage import binary_closing
import matplotlib.pyplot as plt

from hong2p.viz import plot_closed_contours


def load(npy_path):
    return np.load(npy_path, allow_pickle=True)


def main():
    s2p_output_dir = '/home/tom/2p_data/raw_data/2021-05-05/1/ehex_and_1-6ol/suite2p'
    combined_dir = join(s2p_output_dir, 'combined')
    plane0_dir = join(s2p_output_dir, 'plane0')

    traces = load(join(combined_dir, 'F.npy'))

    # TODO regarding single entries in this array (each a dict):
    # - what is 'soma_crop'? it's a boolean array but what's it mean?
    # - what is 'med'? (len 2 list, but is it a centroid or what?)
    # - where are the weights for the ROI? (expecting something of same length as xpix
    #   and ypix)? it's not 'lam', is it? and if it's 'lam', should i normalized it
    #   before using? why isn't it already normalized?
    stat = load(join(combined_dir, 'stat.npy'))

    iscell = load(join(combined_dir, 'iscell.npy'))
    ops = load(join(combined_dir, 'ops.npy')).item()

    kernel = np.ones((3,3), np.uint8)

    #fig = plt.figure()
    # NOTE: these Lx and Ly do indeed reflect dimensions of the tiled images, not a
    # single frame from a single movie
    roi_img = np.zeros((ops['Ly'], ops['Lx'])) * np.nan
    for s in stat:
        xpix = s['xpix']
        ypix = s['ypix']
        print(f'xpix range: [{xpix.min(), xpix.max()}]')
        print(f'ypix range: [{ypix.min(), ypix.max()}]')
        # TODO TODO if this ends up being useful, fix to allow embedding in specified
        # image rather than necessarily padding as it's currently doing
        # (assuming input is a tightly cropped ROI, as output by CNMF)

        roi_img_tmp = np.zeros((ops['Ly'], ops['Lx'])) * np.nan
        roi_img_tmp[ypix, xpix] = s['lam']

        #dilated = cv2.dilate(roi_img_tmp, kernel, iterations=1)
        #plot_closed_contours(dilated)

        plt.figure()
        ax = plt.gca()
        plt.imshow(roi_img_tmp)

        # TODO TODO TODO some function to modify xpix/ypix/lam to accomodate new pixels
        # how to reweight? just assign min? normalize at some point? was it somehow
        # normalized before / otherwise how were particular pixel weights chosen?
        closed = binary_closing(roi_img_tmp > 0)
        plt.figure()
        plt.imshow(closed)

        #plt.figure()
        #plot_closed_contours(closed)
        ax.contour(closed > 0, [0.5])

        #plot_closed_contours(roi_img_tmp)
        plt.show()
        import ipdb; ipdb.set_trace()

        # TODO TODO TODO figure out how to get original coordinates in movie for planes
        # beyond the first one in the combined view. reverse engineer the tiling / see
        # if there is other metadata saved in combined view to invert the tiling.
        # TODO TODO if i can get everything working off of the combined view, maybe
        # delete the single plane data to prevent people from loading the
        # non-manually-corrected stuff?

        roi_img[ypix, xpix] = s['lam']

    plt.figure()
    plt.imshow(roi_img)
    plt.show()

    # TODO do coordinates change w/ plane in combined view, or can i just work with the
    # combined data w/o modifications?

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

