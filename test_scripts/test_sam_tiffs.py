#!/usr/bin/env python3

from pathlib import Path

from hong2p import util

from al_analysis import analysis2thorimage_dir


def main():
    tensor_path = Path('/mnt/tensor')
    sam_dir = tensor_path / 'Sam/Hong2P_Data/analysis_intermediates'

    ok_dir = sam_dir / '2025-01-03/1/peb_gcamp6f_8-9days_fly1_ALr_acv_solvent_002'
    problem_dir = sam_dir / '2025-01-21/1/test_001'

    ok_tif = ok_dir / 'flipped.tif'
    problem_tif = problem_dir / 'flipped.tif'

    ok_raw_dir = analysis2thorimage_dir(ok_dir)

    # copied from original location b/c large and taking too long to load
    #problem_raw_dir = analysis2thorimage_dir(problem_dir)
    problem_raw_dir = Path('.') / '2025-01-21/1/test_001'

    output_dir = Path('.')

    for raw_dir in (ok_raw_dir, problem_raw_dir):
        print(f'{raw_dir=}')

        # TODO add check_round_trip=True?
        # process getting killed in round trip test, b/c using ~38 GB (at time of being
        # killed, as can be seen w/ `dmesg -T | egrep -i 'killed process'`)
        #
        # TODO TODO also get to work w/ discard_channel_b=False (currently
        # NotImplementedError in thor.read_movie)
        util.thor2tiff(raw_dir, output_dir=output_dir, if_exists='overwrite',
            discard_channel_b=True, check_round_trip=True, _debug=True
        )

        print()

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

