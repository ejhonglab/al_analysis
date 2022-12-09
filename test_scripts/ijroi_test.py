#!/usr/bin/env python3

import ijroi

from hong2p.roi import ijroi_masks


def main():
    test_dir = ('/mnt/d1/2p_data/analysis_intermediates/2021-04-28/1/'
        'butanal_and_acetone/'
    )

    roi_fname = test_dir + 'test_z_only_pos3.roi'
    with open(roi_fname, 'rb') as f:
        roi = ijroi.read_roi(f, points_only=False)

    #print(roi)

    thorimage_dir = test_dir.replace('analysis_intermediates', 'raw_data')

    # Defined on a TIFF created by projecting across the time dimension, so dims ZYX
    on_3d_test_fname = test_dir + 'rois_defined_on_3dtiff_test.zip'
    m3 = ijroi_masks(on_3d_test_fname, thorimage_dir)

    on_4d_test = test_dir + 'RoiSet.zip'
    m4 = ijroi_masks(on_4d_test, thorimage_dir)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

