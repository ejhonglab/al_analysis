#!/usr/bin/env python3
#
# run from sensitivity analysis root (w/ direct children dirs for thr/wAPLKC param
# choices)

from pathlib import Path
# copy2 also attempts to preserve all metadata (mtime / permissions / etc)
from shutil import copy2


def main():
    corr_plot_dir = Path('corrs')

    for d in Path('.').glob('thr*_wAPLKC*'):
        print(d)
        assert d.is_dir()

        corr_plot = d / 'corr.pdf'
        assert corr_plot.exists()

        copy2(corr_plot, corr_plot_dir / f'{d.name}_corr.pdf')


if __name__ == '__main__':
    main()

