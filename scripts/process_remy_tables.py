#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Dict

import numpy as np
import pandas as pd
# seem to have this w/o needing to install anything else
#from openpyxl import load_workbook
from tqdm import tqdm

from hong2p.chem import cid2chemfig_tex, draw_molecule

from al_analysis import al_util
from al_analysis.al_util import plot_n_per_odor_pair, sort_odors, read_json, to_json


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--ignore-cache', action='store_true', help='will ignore '
        'CID -> chemfig TeX cache'
    )
    args = parser.parse_args()
    ignore_cache = args.ignore_cache

    al_util.verbose = True

    output_dir = Path('.')
    chemfig_tex_output_dir = output_dir / 'chemfig'
    chemfig_tex_output_dir.mkdir(exist_ok=True)

    # TODO TODO delete. for testing changes to N plotting fns
    from al_analysis.al_util import (load_natmix_dff, plot_n_per_odor_and_glom, savefig,
        plot_n, plot_n_per_odor_pair, count_n_per_odor_and_glom
    )
    orn_df = load_natmix_dff()
    # counts currently will have trials if i don't do this
    # (or if i don't do similar post-processing of counts to collapse across trials)
    orn_means = orn_df.groupby(level=['panel','odor1','odor2'], sort=False).mean()
    gcounts2 = count_n_per_odor_and_glom(orn_means)
    fig, gcounts = plot_n_per_odor_and_glom(orn_means)
    savefig(fig, output_dir, 'gcounts')

    # TODO (done? delete) implement (in a way that doesn't break above)
    #fig, pcounts = plot_n_per_odor_pair(orn_means)
    #savefig(fig, output_dir, 'pcounts')
    #breakpoint()
    #

    tables_dir = Path(
        '~/Documents/2026-07-28_thesis_figures/remy-paper/Tables'
    ).expanduser()
    n_stim_dir = tables_dir / 'Stims-per-Fly'
    n_stim_excel = n_stim_dir / 'stims_per_fly.xlsx'

    #odor_excel = tables_dir / 'Odor-Panels.xlsx'
    odor_excel = tables_dir / 'Odor-Panels_fixed.xlsx'
    #odor_ws = load_workbook(odor_excel)

    chemfig_pdfdir = Path('.') / 'chemfig_pdf'
    chemfig_pdfdir.mkdir(exist_ok=True)

    for sheet_name in ['megamat17 KC soma', 'validation2 KC soma']:
        if 'megamat' in sheet_name:
            panel = 'megamat17'
        else:
            assert 'validation' in sheet_name
            panel = 'validation2'
        print(f'{panel=}')

        df = pd.read_excel(n_stim_excel, sheet_name=sheet_name, skiprows=4)
        cols_to_drop = ['# flies', '# flies presented', '# flies in ms (post-QC)']
        df = df.drop(columns=[x for x in cols_to_drop if x in df.columns]).set_index(
            'stimulus', verify_integrity=True
        )
        to_drop = ['# stims seen', '# stims presented', '# in ms (pooled, post-QC)']
        df = df.drop(index=[x for x in to_drop if x in df.index])

        # should only be relevant for validation2 panel.
        # drops 'MethOH @ -2.0', 'geos @ -5.0', 'pfo @ 0.0'
        df = df.drop(index=df.index[df.index.str.contains('not in ms grid')])

        # TODO delete. doesn't seem i can tell which rows are struckthrough or not
        # (even though it's first 17 in validation2 that aren't...)
        odf = pd.read_excel(odor_excel, sheet_name=panel, skiprows=1)
        if panel == 'validation2':
            # just assuming it's first 17 odors here. would have to use load_workbook
            # otherwise, like:
            # https://stackoverflow.com/questions/76306226/pandas-read-excel-and-flag-rows-with-striketrough
            odf = odf.iloc[:17]

        odf = odf.set_index('abbrev', verify_integrity=True)

        # validation panel should also have this many odors, at least after subsetting
        # to correct set of odors... (which i may need to manually do first)
        n_odors = 17
        assert len(odf) == n_odors

        # TODO TODO TODO factor some of below into a al_util.count_n_per_odor_pair fn
        assert not df.columns.duplicated().any()
        # no regex option here to worry about
        # all columns should be flies in format <YYYY-DD-MM>/<fly_num>
        assert (df.columns.str.count('/') == 1).all()

        # note that some validation2 odors will just end with (pooled) and say the conc
        # instead of # trials (assumes 3 trials, as all validation2 data should be)
        odor_parts = df.index.str.split(' @ ')
        if panel == 'megamat17':
            assert (odor_parts.str.len() == 2).all()
            names = odor_parts.map(lambda x: x[0])

            concs = odor_parts.map(lambda x: float(x[-1]))
            # only true for megamat
            assert set(concs) == {-3.0}
        else:
            assert odor_parts[odor_parts.str.len() == 1].map(
                lambda x: x[0].endswith(' (pooled)')
            ).all()
            # TODO just define names this way in both cases? should work for all
            # actually, maybe not exactly...
            names = df.index.str.split(' ').map(lambda x: x[0])

        assert not names.duplicated().any()
        assert (names.str.len() > 0).all()

        df.index = names

        if panel == 'validation2':
            assert odf.index.isin(names).all()
            # 17 < 22
            assert n_odors < len(names)
            assert names.isin(odf.index).sum() == n_odors
            # TODO say which we are dropping
            df = df[df.index.isin(odf.index)]

        cid2chemfig = dict()
        cids = odf.CID
        assert np.issubdtype(cids.dtype, int)
        for i, cid in enumerate(tqdm(cids, unit='CID',
            desc='converting CID -> chemfig TeX')):

            cser = odf.iloc[i].squeeze()
            assert cser.CID == cid
            # NOTE: cser.name gives us index, which is abbrev, and NOT what I want
            print(f'{cser["name"]} (CID={cid}):\n')

            # TODO delete
            # TODO keep?
            # TODO TODO also compare w/ rdkit?
            # TODO TODO pass output_dir to save to their own dir?
            #
            # NOTE: .tex chemfig structures will currently be in
            # <chemfig_pdfdir>/chemfig, as currently implemented
            # (except the ones saved separately in the separate (duplicate)
            # cid2chemfig_tex call this script does)
            #print(f'{cid=}')
            draw_molecule(cid, output_dir=chemfig_pdfdir, verbose=True,
                  ignore_cache=ignore_cache
            )

            chemfig_tex_str = cid2chemfig_tex(cid)
            print(chemfig_tex_str)

            cid2chemfig[cid] = chemfig_tex_str

            # TODO TODO compare rendered chemdraw to rdkit (can it make something
            # vector)? particularly for stereochem (e.g. longifolene)
            # TODO TODO also invoke latex (or some python module that can do it?)
            # to render each chemdraw structure to a pdf?

        # TODO TODO should i save the tex as values directly or as print(values)
        # output?
        for cid, chemfig_str in cid2chemfig.items():
            (chemfig_tex_output_dir / f'{cid}.tex').write_text(chemfig_str)

        assert len(df) == n_odors

        # TODO all of this data is used right? including st uff w/ just 2 trials?
        ser = df.stack()
        assert ser.notna().all()
        assert len(ser) == df.notna().sum().sum()

        if panel == 'validation2':
            # 3          197
            # -2.0        11
            # ✗ -2.0       7
            # ✗ 3          6
            # -2.0         5
            # -1.5         5
            # -3.0         5
            # ✗ -1.75      4
            # ✗ -3.0       2
            # -1.75        1

            # TODO i assume .str.contains('✗') only returns NaN for non-str input?
            to_zero = ser.str.contains('✗').fillna(False)
            assert to_zero.any()
            ser[to_zero] = 0

            # ipdb> ser.value_counts()
            # 3        197
            # 0         19
            # -2.0      11
            # -2.0       5
            # -1.5       5
            # -3.0       5
            # -1.75      1

            # TODO should i be replacing 0 w/ NaN to get same effect as below?
            # (before unstacking) (don't think it matters)
            # TODO TODO move some of this to counting fn?
            ser[ser != 0] = 1
            n0_before = (ser == 0).sum()
            has_odor = ser.unstack()
            assert (has_odor == 0).sum().sum() == n0_before

            has_odor = has_odor.fillna(0)
            assert has_odor.max().max() == 1
            restacked = has_odor.stack()
            assert has_odor.size == len(restacked)
            # (stack should never give you NaN anyway)
            assert restacked.notna().all()
            assert set(restacked) == {0, 1}

        elif panel == 'megamat17':
            # print(ser.value_counts())
            # 3.0    208
            # 2.0     19
            # 4.0      1
            has_odor = df.notna().astype(int)
            assert not (ser == 0).any()
            ser[:] = 1
            has_odor2 = ser.unstack().fillna(0).astype(int)
            assert has_odor.equals(has_odor2)

        n_flies_per_pair = has_odor.dot(has_odor.T)

        n_per_pair2 = pd.DataFrame(index=has_odor.index, columns=has_odor.index, data=0)
        flies_with_pair = defaultdict(list)
        for fly in has_odor.columns:
            # TODO TODO also track which flies each pair has, for sanity checking
            # (matrix w/ same indices but list elements, and add to each? or nested
            # [default]dicts?)
            fly_ser = has_odor[fly]
            fly_counts = fly_ser.to_frame().dot(fly_ser.to_frame().T).stack()
            assert len(fly_counts) == n_odors**2
            assert not fly_counts.index.duplicated().any()

            unique_fly_counts = set(fly_counts)
            # a fly might have all pairs
            assert unique_fly_counts <= {0, 1}
            assert max(unique_fly_counts) == 1

            for (o1, o2), has_pair in fly_counts.items():
                assert fly_counts.loc[o2, o1] == has_pair
                n_per_pair2.loc[o1, o2] += has_pair

                if has_pair:
                    assert fly not in flies_with_pair[(o1, o2)]
                    flies_with_pair[(o1, o2)].append(fly)

        assert n_flies_per_pair.equals(n_per_pair2)

        min_n = n_flies_per_pair.min().min()
        # since this is true, len(flies_with_pair) should also be n_odors**2
        assert min_n > 0
        assert len(flies_with_pair) == n_odors**2

        max_n = n_flies_per_pair.max().max()
        print(f'n={min_n}-{max_n}')

        flies_with_pair_df = pd.DataFrame(index=has_odor.index, columns=has_odor.index)
        # will be all NaN to start, since no data passed
        assert flies_with_pair_df.isna().all().all()
        for (o1, o2), flies in flies_with_pair.items():
            flies_with_pair_df.loc[o1, o2] = ','.join(flies)

        assert flies_with_pair_df.notna().any().any()
        assert (flies_with_pair_df.stack().str.len() > 0).all()

        # megamat:
        # ipdb> n_flies_per_pair.stack().sort_values().head(20)
        # stimulus  stimulus
        # aa        6al         4
        # benz      aa          4
        # t2h       aa          4
        # Lin       aa          4
        # 2h        aa          4
        # aa        2h          4
        #           Lin         4
        #           benz        4
        # eb        aa          4
        # ep        aa          4
        # aa        eb          4
        # 6al       aa          4
        # aa        ms          4
        # ms        aa          4
        # B-cit     aa          4
        # aa        t2h         4
        #           ep          4
        #           B-cit       4
        # 1-6ol     aa          5
        # aa        1-6ol       5

        # validation:
        # ipdb> n_flies_per_pair.stack().sort_values().head(23)
        # stimulus  stimulus
        # euc       ger         2
        #           geos        2
        # geos      euc         2
        # ger       euc         2
        # geos      1p3one      3
        # 1p3one    geos        3
        # geos      -bCar       3
        # 2-mib     euc         3
        # -aPine    geos        3
        # euc       bbenz       3
        # ger       geos        3
        # euc       2-mib       3
        # mchav     geos        3
        # -bCar     geos        3
        # long      geos        3
        # geos      bbenz       3
        #           ger         3
        #           long        3
        #           mchav       3
        # bbenz     geos        3
        # geos      -aPine      3
        # bbenz     euc         3
        # ger       sab         4

        # TODO TODO TODO report these in a more sensible way, but getting triangular
        # first, e.g. like:
        # https://stackoverflow.com/questions/34417685/melt-the-upper-triangular-matrix-of-a-pandas-dataframe
        # ipdb> n_flies_per_pair.stack().value_counts()
        # 12    55
        # 4     44
        # 6     34
        # 11    23
        # 10    21
        # 3     18
        # 7     17
        # 14    16
        # 9     15
        # 8     11
        # 13     9
        # 20     9
        # 18     7
        # 5      6
        # 2      4

        # TODO plot triangular w/ similar N colormap to my own plots? seems
        # like it would be hard to condense otherwise... (and would probably best be an
        # actual supplemental file, not included in latex/pdf)

        # TODO move this to plot_n_per_odor_pair? (prob need before sort_odors tho...)
        # (dupe to plot_n_per_odor_pair? after checking no existing odor levels)
        n_flies_per_pair.index.name = 'odor'
        n_flies_per_pair.columns.name = 'odor'

        sort_panel = {
            'megamat17': 'megamat',
            'validation2': 'validation2',
        }[panel]
        n_flies_per_pair = sort_odors(n_flies_per_pair, panel=sort_panel)

        # TODO make sure these plots have odors in their canonical order
        # (defined in any of the sheets i'm working with? or load from elsewhere?
        # panel2name_order [current value for validation2 correct?]?)
        #
        # yticklabels like this so it doesn't assume there's an roi level
        fig = plot_n_per_odor_pair(n_flies_per_pair)
        savefig(fig, output_dir, f'pcounts_{panel}')

        # TODO report mean off diagonal?

        # TODO are there really validation2 pairs that only have n=2? which?

        print()

    breakpoint()


if __name__ == '__main__':
    main()

