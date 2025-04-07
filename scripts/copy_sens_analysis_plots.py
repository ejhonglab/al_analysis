#!/usr/bin/env python3
#
# run from sensitivity analysis root (w/ direct children dirs for thr/wAPLKC param
# choices)

from pathlib import Path
# copy2 also attempts to preserve all metadata (mtime / permissions / etc)
from shutil import copy2


def main():
    for d in Path('.').glob('thr*_wAPLKC*'):
        print(d)
        assert d.is_dir()

        plots_names_to_copy = ['corr.pdf', 'combined_odors-per-cell_and_sparsity.pdf']
        for name in plots_names_to_copy:
            plot = d / name
            assert plot.exists()

            # e.g. 'corr'
            plot_dir = Path(plot.stem)
            plot_dir.mkdir(exist_ok=True)

            # will overwrite target if it happens to already exist
            copy2(plot, plot_dir / f'{d.name}_{name}')


if __name__ == '__main__':
    main()

