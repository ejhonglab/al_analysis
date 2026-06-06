#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pformat, pprint
from itertools import combinations
import shutil
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm

from hong2p.util import pd_allclose, addlevel, add_group_id, reindex
from hong2p import olf
from hong2p.olf import parse_odor_name, solvent_str, component_delim
from natmix import drop_mix_dilutions

from al_analysis import al_util
from al_analysis.al_util import (savefig, plot_responses, read_parquet, to_csv,
    to_parquet, data_root, fly_cols, warn
)
from al_analysis.mb_model import (megamat_orn_deltas, fit_and_plot_mb_model,
    megamat_orn_deltas, natmix_orn_deltas, get_thr_and_APL_weights, format_model_params,
    get_odor_fname_suffix, KC_ID, dict_seq_product, abbrev_model_id,
    calc_mix_suppression, get_diff_col, diff_col2desc, FULL_MODEL_KW_LIST,
    NoCachedModelOutputsError, logistic, summarize_response_classes,
    add_missing_cells_to_nonresponders, format_response_class,
    plot_response_class_summary
)


# TODO use pre-existing const kw list vars in mb_model for that, and to replace some of
# what i have here now
#
# passing the CLI arg -f will use FULL_MODEL_KW_LIST (currently len 137) instead of this
test_with_connectome_vs_uniform_apl = [
    dict(weight_divisor=20),
    dict(one_row_per_claw=True, prat_claws=True),
    dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True),
]
SHORT_MODEL_TUNE_KWS = [
    # comparison for all other model cases, to see to what extent changes to PN>KC
    # weight matrix (and potentially other changes) matter
    dict(pn2kc_connections='uniform', n_claws=7),
] + dict_seq_product(test_with_connectome_vs_uniform_apl,
    [dict(), dict(use_connectome_APL_weights=True)]
)

def yang2tom_odor_index(df: pd.DataFrame) -> pd.MultiIndex:
    """Returns new column index for `df`, with levels 'odor' and 'repeat'
    """
    # requiring ')' before '.' b/c don't want to match '.' instead parens like
    # 'farn(-2.5)'
    # TODO just strip last two chars if 2nd-from-last char is '.'? check equiv?
    parts = df.columns.str.split('\)\.')
    parts_len = parts.str.len()

    # so that PFO and other cases can be handled the same way below, to calculate repeat
    # number
    parts = parts.map(lambda x:
        # TODO delete. need more general handling, since we also have stuff like
        # 'Bmix28.1', 'banana.1', etc
        #x[0].split('.') if any(x[0].startswith(y) for y in ('PFO','air')) else x
        # TODO this work? or need to check for absense of ')' / '('
        # TODO assert '(' not in x[0] for all where len(x) == 1?
        x[0].split('.') if (len(x) == 1 and '(' not in x[0]) else x
    )

    # TODO TODO TODO some flies might only have 3-4 trials. check NaNs. maybe drop only
    # to only first # common trials (or average across all later)
    # (i think yang was saying it might only have been for the subdirectory data)
    assert (parts_len <= 2).all()
    # 1,2,...5 (repeat=0 ommitted. that's the case where parts is len 1)
    repeat_strs = parts.map(lambda x: '0' if len(x) == 1 else x[1])
    # cast to int should fail if not possible for all elements
    repeat = repeat_strs.astype(int)
    assert repeat.astype(str).equals(repeat_strs)
    odors = parts.map(lambda x: x[0]).str.strip(')')

    def yang2tom_odor_str(x: str) -> str:
        replace_dict = {
            '(': ' @ ',
            ')': '',
            '+': component_delim,
        }
        for k, v in replace_dict.items():
            x = x.replace(k, v)

        # TODO or return solvent_str?
        if x == 'PFO':
            return 'pfo'

        return x

    # TODO assert same # of duplicates as before? (and 1:1 w/ what we had before?)
    tom_odors = odors.map(yang2tom_odor_str)

    both = pd.concat([
            tom_odors.rename('tom').to_frame(index=False),
            odors.rename('yang').to_frame(index=False)
        ], axis='columns'
    )
    # ipdb> both.drop_duplicates(subset='tom')
    #                       tom              yang
    # 0                 2h @ -8             2h(-8
    # 6                 2h @ -7             2h(-7
    # 12              farn @ -4           farn(-4
    # 18              farn @ -3           farn(-3
    # 24                ma @ -8             ma(-8
    # 30                ma @ -7             ma(-7
    # 36    2h @ -7 + farn @ -3    2h(-7)+farn(-3
    # 42      2h @ -7 + ma @ -7      2h(-7)+ma(-7
    # 48    ma @ -7 + farn @ -3    ma(-7)+farn(-3
    # 54                    pfo               PFO
    # 57            farn @ -2.5         farn(-2.5
    # 63  2h @ -7 + farn @ -2.5  2h(-7)+farn(-2.5
    # 69  ma @ -7 + farn @ -2.5  ma(-7)+farn(-2.5
    #
    # so the string processing shouldn't have changed the information in the strings.
    assert both.drop_duplicates(subset='tom').index.equals(
        both.drop_duplicates(subset='yang').index
    )

    odors = tom_odors.rename('odor')
    odor_index = pd.MultiIndex.from_arrays([odors, repeat.rename('repeat')])
    assert odor_index.get_level_values('odor').equals(odors)

    return odor_index


# TODO TODO add flag to only return wildtype (or at least no TNT (i.e. excluding
# TNT_nolabel)) (+ to parent load_... fn)?
def preprocess_yang_data(df: pd.DataFrame, *, verbose: bool = True,
    drop_expt: bool = True, drop_exp_type: bool = True, _check_farn: bool = False,
    expected_nonmix_odors: Optional[Set[str]] = None) -> pd.DataFrame:
    # TODO doc

    have_exp_type = False
    if 'exp_type' in df.columns:
        index_cols = ['exp_type', 'brain', 'roi']
        have_exp_type = True
    else:
        index_cols = ['brain', 'roi']

    assert not df.loc[:, index_cols].isna().any().any()

    parts = df.brain.str.split('_')
    assert (parts.str.len() == 2).all()
    # TODO explicitly specify format? seems fine w/o...
    # (Yang uses YYYYMMDD format)
    df['date'] = pd.to_datetime(parts.map(lambda x: x[0]))

    rest = parts.map(lambda x: x[1])
    prefix_and_recording_num = rest.str.split('fun')
    assert (prefix_and_recording_num.str.len() == 2).all()
    recording_num = prefix_and_recording_num.map(lambda x: int(x[-1]))
    assert recording_num.min() == 1
    if recording_num.max() > 1:
        # TODO TODO make sure these are getting dropped too if needed. add separate col
        # w/ recording # if we have any that are >1?
        # (check handling in one_recording_per_fly code)
        df['recording_num'] = recording_num
    # TODO delete. subdir has some 'fun2' suffices
    #assert rest.str.endswith('fun1').all()

    # TODO also convert to int? assert all positive? assert no dupes (apart from side)
    # within day? assert all consecutive w/in day (don't really care...)?
    df['fly_num'] = rest.map(lambda x: int(x[0]))
    assert not df.fly_num.isna().any()

    df['sex'] = rest.map(lambda x: x[1])
    # TODO drop the one 'M' or did it only have TNT_label data anyway?
    assert {'F'} <= set(df.sex.unique()) <= {'F', 'M'}

    df['side'] = rest.map(lambda x: x[2])
    assert set(df.side.unique()) == {'L', 'R'}

    if have_exp_type:
        recording_cols = ['exp_type'] + fly_cols + ['side']
    else:
        recording_cols = fly_cols + ['side']

    if 'recording_num' in df.columns:
        recording_cols += ['recording_num']

    assert (df.groupby(fly_cols).sex.nunique() == 1).all()

    # TODO maintain sex? warn if not all the same?
    to_drop = ['sex']
    if drop_expt:
        # TODO rename this col expt then?
        to_drop += ['brain']
    else:
        recording_cols += ['brain']

    df = df.drop(columns=to_drop)

    # TODO do this (and reset_index() again?) before str processing currently above?
    # or just do (something like this) after str processing, using some of those cols in
    # place of some of these?
    df = df.set_index(recording_cols + ['roi'], verify_integrity=True)

    orig_n_odors = len(df.columns)

    # TODO should i lump data from all of exp_type=['WT', 'TNTin_nolabel',
    # 'TNTin_label'] cases (everything but 'TNT_label') together?
    # that's what i'm doing here now:
    # TODO add flag to just use WT. might be safer. compare (not sure spread looks any
    # better there, esp w/o tossing an outlier. from looking at first side of each at
    # least)
    if have_exp_type:
        exp_type_to_drop = 'TNT_label'
        exp_types = df.index.get_level_values('exp_type')
        exp_types_to_keep = exp_types != exp_type_to_drop
        if verbose and (~ exp_types_to_keep).any():
            # "...will lump all data from exp_types=['WT', 'TNTin_nolabel',
            # 'TNTin_label'] together"
            warn(f'dropping exp_type={exp_type_to_drop}. will lump all data from '
                f'exp_types={list(exp_types[exp_types_to_keep].unique())} together.'
            )

        df = df[exp_types_to_keep].copy()
        if drop_exp_type:
            df = df.droplevel('exp_type')

        recording_cols = [x for x in recording_cols if x != 'exp_type']

    # TODO (delete) so were flies only numbered within experimental conditions per day?
    # or was this a mistake? oh, ig one side was labelled and the other wasn't.
    # TODO may still want to assert we don't have certain combos of exp_type
    # within a fly, but we shouldn't anyway
    # MultiIndex([(  'TNTin_label', '2025-03-21', 3, 'L'),
    #             ('TNTin_nolabel', '2025-03-21', 3, 'R')],
    # TODO what is dropping these two flies (after comparison w/ one_recording_per_fly
    # above)? is it just that these two are only TNT_label flies? (probably, but assert
    # that?)
    # ipdb> one_recording_per_fly.droplevel('side').difference(df.index.droplevel('roi'
    #  ).drop_duplicates())
    # MultiIndex([('2025-02-14', 4),
    #             ('2025-02-19', 3)],
    #            names=['date', 'fly_num'])
    assert not df.index.duplicated().any()

    # TODO if refactoring, add flag to not drop any side (/recordings) (might want to
    # drop duplicate recordings, but keep multiple sides [at least, that's how yang
    # operates i believe])?
    # (and make sure rest of fn actually supports it. maybe add to some internal var
    # like fly_cols, but that optionally includes 'side' too?)
    #
    # one fly has two exp_type values (b/c one side labelled and one is not)
    # TODO should exp_type be in this or not (now that i've added
    # drop_exp_type[=False])?
    non_recording_cols = ['roi']
    recording_index = df.index.droplevel(non_recording_cols)
    # TODO use .last() instead? might select the later recordings yang seems to think
    # are often more stable (but later). flag to select between the two?
    one_recording_per_fly = pd.MultiIndex.from_frame(recording_index.drop_duplicates(
        ).to_frame(index=False).groupby(fly_cols).first().reset_index()
    )
    # TODO doesn't break anything in drop_exp_type=True (old) case, right?
    one_recording_per_fly = one_recording_per_fly.reorder_levels([
        x for x in df.index.names if x not in non_recording_cols
    ])
    #
    assert not one_recording_per_fly.droplevel('side').duplicated().any()
    recording_to_keep_mask = recording_index.isin(one_recording_per_fly)
    if verbose and (~ recording_to_keep_mask).any():
        expts_dropped = df[~recording_to_keep_mask].index.droplevel('roi'
            ).drop_duplicates()

        warn('dropping the following experiments, for which each fly has another '
            'recording (probably on the other side) not dropped:\n'
            f'{expts_dropped.to_frame().to_string(index=False)}'
        )
    # NOTE: probably important that this dropping happens after dropping based on
    # exp_type. In case we accidentally select only the TNT_label case for a fly that
    # also has TNT_nolabel.
    non_fly_cols = [x for x in df.index.names if x not in fly_cols]
    flies_before = df.index.droplevel(non_fly_cols).drop_duplicates().sort_values()

    df = df[recording_to_keep_mask].copy()
    assert df.index.droplevel(non_recording_cols).drop_duplicates().sort_values(
        ).equals(one_recording_per_fly.sort_values())

    flies_after = df.index.droplevel(non_fly_cols).drop_duplicates().sort_values()
    assert flies_before.equals(flies_after)

    # if this 1st assertion fails, code above is not working correctly
    assert (
        len(df.index.droplevel('roi').drop_duplicates()) == len(flies_after)
    )
    to_drop = set(non_fly_cols) - {'roi'}
    if not drop_expt:
        assert len(df.index.get_level_values('brain').unique()) == len(flies_after)
        to_drop -= {'brain'}

    if not drop_exp_type:
        assert len(df.index.droplevel([
            x for x in df.index.names if x not in fly_cols + ['exp_type']]
            ).drop_duplicates()
        ) == len(flies_after)
        to_drop -= {'exp_type'}

    assert {'side'} <= to_drop <= {'side', 'recording_num'}, f'{to_drop=}'

    df = df.droplevel(list(to_drop))
    assert not df.index.duplicated().any()
    recording_cols = [x for x in recording_cols if x not in to_drop]

    odor_index = yang2tom_odor_index(df)
    odors = odor_index.get_level_values('odor')
    # TODO replace some of above odor index construction w/ this? factor into one fn?
    #
    # TODO TODO this not working? why farn still after ma in ma+farn?
    # TODO TODO oh, maybe strs aren't even being derived from this? probably should be
    # tho...
    odor_lists = odors.map(lambda x: olf.parse_odor_list(x))

    odor_index = olf.odor_lists_to_multiindex(odor_lists)
    assert odor_index.names == ['odor1', 'odor2', 'repeat']

    mix_rows = (odor_index.droplevel('repeat').to_frame() != solvent_str).T.all()
    assert (mix_rows | (odor_index.get_level_values('odor2') == solvent_str)).all()
    mix_index = mix_rows[mix_rows].index
    mix_components = set(mix_index.get_level_values('odor1').unique())
    mix_components |= set(mix_index.get_level_values('odor2').unique())
    assert solvent_str not in mix_components
    # TODO assert all mixes are only preset in a single odor1/2 order?
    # (should be guaranteed by my olf fns?)

    # NOTE: mismatches between her main concs and diags:
    # - 2h @ -6 in diag, -8 here
    # - farn @ -2 in diag, -2.5/-3 here
    # - ma @ -7 in diag, and here (-8 in diag ever? why was i thinking one was at -8?)
    # TODO how similar is her component data from one vs the other conc? she have both
    # in any flies (if so, how many?)?
    # TODO do all flies only have one pair or the other? or some flies have both?
    #
    # ipdb> pp [x for x in odor_index.get_level_values('odor').unique()]
    # ['2h @ -8',
    #  '2h @ -7',
    #  'farn @ -4',
    #  'farn @ -3',
    #  'ma @ -8',
    #  'ma @ -7',
    #  '2h @ -7 + farn @ -3',
    #  '2h @ -7 + ma @ -7',
    #  'ma @ -7 + farn @ -3',
    #  'pfo',
    #  'farn @ -2.5',
    #  '2h @ -7 + farn @ -2.5',
    #  'ma @ -7 + farn @ -2.5']
    mix_or_comp_mask = (
        odors.isin(mix_components) | odors.str.contains(component_delim, regex=False)
    )
    nonmix_odors = set(odors[~mix_or_comp_mask].unique())
    if verbose and len(nonmix_odors) > 0:
        warn('dropping the following odors, which are not components of any mixture:\n'
            f'{nonmix_odors}'
        )

    if expected_nonmix_odors is not None:
        assert expected_nonmix_odors == nonmix_odors, (
            f'{expected_nonmix_odors=}\n{nonmix_odors=}'
        )
    else:
        if verbose:
            warn('set expected_nonmix_odors=<set-of-odor-strs> to enable assertion')

    odor_index = olf.add_mix_str_index_level(odor_index)
    assert odor_index.names == ['odor1', 'odor2', 'repeat', 'odor']
    df.columns = odor_index.droplevel(['odor1', 'odor2']).reorder_levels(
        ['odor', 'repeat']
    )
    assert len(odors) == len(df.columns) == orig_n_odors
    df = df.loc[:, mix_or_comp_mask].copy()
    odors = df.columns.get_level_values('odor')
    mix_mask = odors.str.contains(component_delim, regex=False)
    comps = odors[~mix_mask]

    metadata = df.groupby('odor', axis='columns').mean().stack(
        ).droplevel(['roi']).index.to_frame(index=False)
    recordings_per_odor = metadata.drop_duplicates(['odor'] + recording_cols)
    assert metadata.drop_duplicates(['odor'] + fly_cols)[fly_cols].equals(
        recordings_per_odor[fly_cols]
    )
    if verbose:
        print(f'components: {sorted(comps.unique())}')

        recordings_per_mix = recordings_per_odor[recordings_per_odor.odor.str.contains(
            component_delim, regex=False)].odor.value_counts().sort_index().rename(
            'count').rename_axis('mix').to_frame()

        print('mixes (-> # flies):\n' + recordings_per_mix.to_string())
        print('mixes (ignoring concs): ' + str(sorted(
            odors[mix_mask].unique().map(olf.strip_concs_from_odor_str).unique()
        )))

    # TODO use something like this in main to make summary plots (to merge this
    # data w/ model KC data using different concentrations) (doing that. just need to
    # warn now and can delete this). would need to separately warn there.
    without_concs = comps.map(olf.strip_concs_from_odor_str)

    # TODO diag farn conc is -2, so maybe just keep -2.5 actually? what fraction of
    # data is that?
    # TODO still decide between 'farn @ -3' and '@ -2.5'? or average both together
    # and rename to one (leaning towards that, but will handle outside of this fn)?
    # (diag is -2. see counts of each reported below, for her two concs)
    # TODO TODO refactor to share this code w/ get_yang... fn, to warn near end, where i
    # actually use this stuff (and where i actually want to strip concs)
    # (or just move this code there? any point to having it here?)
    odors_at_multiple_concs = pd.DataFrame({
        'name': without_concs, 'odor': comps
    }).drop_duplicates().groupby('name').filter(lambda x: len(x) > 1)
    if len(odors_at_multiple_concs) > 0:
        # TODO actually implement dropping/merging of these odors at mulitiple concs
        # (no, delete)? (or maybe only when actually making summary plots including
        # both, so i can warn about concentration differences there, when comparing
        # model [based on my diags] and KC stuff)
        if verbose:
            warn('the following odors are at multiple concentrations:\n'
                f'{odors_at_multiple_concs.odor.sort_values().to_string(index=False)}'
            )
        # oh, it looks like all the flies have 'farn @ -3' (at least those remaining
        # here), and some of them also have 'farn @ -2.5':
        # ipdb> df.loc[:, odors.isin(odors_at_multiple_concs.odor)].groupby('odor',
        #    axis='columns').mean().isna().sum()
        # odor
        # farn @ -2.5    11810
        # farn @ -3          0
        #
        # TODO just keep 'farn @ -3' then? to not end up averaging across flies?

        recordings_per_multipleconc_odor = recordings_per_odor[
            recordings_per_odor.odor.isin(odors_at_multiple_concs.odor)
        ].reset_index(drop=True)

        # TODO delete
        recordings_per_odor2 = df.loc[:, odors.isin(odors_at_multiple_concs.odor)
            ].groupby('odor', axis='columns').mean().stack().droplevel(['roi']
            ).index.to_frame(index=False).drop_duplicates(['odor'] + fly_cols
            ).reset_index(drop=True)
        assert recordings_per_odor2.equals(recordings_per_multipleconc_odor)
        #
        n_recordings_per_odor = recordings_per_odor.odor.value_counts().sort_index()
        if verbose:
            warn('# recordings per odor (for odors at multiple concs):\n' +
                n_recordings_per_odor.to_string(name=False)
            )

        # TODO move outside this fn? would be hard to refactor tho... put behind some
        # flag / extra kwarg indicating this is the check to make?
        if _check_farn:
            recordings_per_odor = recordings_per_odor.set_index(['odor'] + fly_cols,
                verify_integrity=True
            )
            farn3_flies = recordings_per_odor.loc['farn @ -3'].index.sort_values()
            farn2p5_flies = recordings_per_odor.loc['farn @ -2.5'].index.sort_values()
            assert len(farn2p5_flies.difference(farn3_flies)) == 0

            all_flies = recordings_per_odor.droplevel('odor').index.drop_duplicates(
                ).sort_values()
            assert all_flies.equals(farn3_flies)

    isna = df.isna()
    assert not isna.all().any()
    assert not isna.T.all().any()
    # TODO also check for cols all NaN w/in fly (e.g. to detect flies that only have 3-4
    # instead of 6 trials)

    fly2n_repeats = df.groupby(fly_cols, group_keys=False).apply(lambda x:
        x.dropna(how='all', axis='columns').columns.get_level_values('repeat').max()
    )
    unique_n_repeats = fly2n_repeats.unique()
    if verbose and len(unique_n_repeats) > 1:
        warn('not all flies had same # of repeats!\n# repeats -> # flies with that many'
            f':\n{fly2n_repeats.value_counts().to_string()}'
        )
    # TODO (delete) maybe dropping that single subdir fly w/ only 2 trials. don't think
    # i want to drop the other flies in subdir down from 3-4 to only 2 trials...
    # TODO and/or drop single parent dir fly w/ 3 instead of 5 repeats?
    # TODO could maybe do that as well as dropping all 4->3?
    # TODO did any other recording for any of these flies have more repeats? also
    # select based on that, when picking recording?

    # assuming this pre-sorting is necessary to get consistent fly_ids in next step
    df = df.sort_index(level=fly_cols, kind='stable')
    index_df = df.index.to_frame(index=False)

    # TODO also delete date/fly_num cols (nah)? or concat them into one str ID
    # rather than this integer one?
    # TODO also do this for natmix data i'm also comparing some stuff against
    index_df = add_group_id(index_df, fly_cols, 'fly_id')
    # TODO add recording_id too? esp if i'm including e.g. multiple sides for one
    # fly
    df.index = pd.MultiIndex.from_frame(index_df)

    put_at_end = ['roi']
    if have_exp_type and not drop_exp_type:
        put_at_end.append('exp_type')

    assert all(x in df.index.names for x in put_at_end)
    df = df.reorder_levels(
        [x for x in df.index.names if x not in put_at_end] + put_at_end
    )
    # TODO also sort_index() before returning?

    if verbose:
        print()

    return df


def load_yang_kc_data(*, drop_exp_type: bool = True, verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns concatenated response amplitude and binarized response data.
    """
    # TODO (delete) is all this data (both yang_dir and yang_subdir contents) stuff i
    # have run through the model for her (no, concs differ somewhat, at lesat for
    # yang_dir contents)? or can i not generate appropriate input for some of them
    # (maybe for subdir? do i or sam have right concs anywhere?)

    # should contain binary diagnostic odor mix data
    yang_dir = data_root / 'from_yang/20260415_YZtransfer'

    # only file in yang_dir with responses is concatenated data in:
    # 20250129_0326_KC_odormix_resp_good.csv
    mean_csv = yang_dir / '20250129_0326_KC_odormix_resp_good.csv'
    # col 0 is just ROI number (seemingly concatenated across recordings). don't see any
    # other fly/recording IDs in this output.
    df = pd.read_csv(mean_csv, index_col=0)
    # (delete) always pick L/R for all? average / compute across the two within each?
    # yang have a record of which came first, or which was better, for each fly?
    # (first yes, it's in filename / brain ID, but better not really. she things later
    # is generally more stable, but I might prefer earlier typically. shouldn't really
    # matter)
    #
    # (delete) do all flies have both hemispheres (not quite)?
    # ipdb> df.loc[:, df.dtypes == np.dtype('O')].drop_duplicates()
    #            brain       exp_type
    # 20250129_1FLfun1             WT
    # 20250129_1FRfun1             WT
    # 20250129_2FLfun1             WT
    # 20250129_2FRfun1             WT
    # 20250320_1FLfun1             WT
    # 20250320_1FRfun1             WT
    # 20250320_2FLfun1             WT
    # 20250214_2FRfun1  TNTin_nolabel
    # 20250321_3FRfun1  TNTin_nolabel
    # 20250221_1FRfun1    TNTin_label
    # 20250224_2FLfun1    TNTin_label
    # 20250224_3FLfun1    TNTin_label
    # 20250321_3FLfun1    TNTin_label
    # 20250324_1FRfun1    TNTin_label
    # 20250324_2FLfun1    TNTin_label
    # 20250214_3MLfun1    TNTin_label
    # 20250214_4FLfun1      TNT_label
    # 20250219_3FLfun1      TNT_label
    # 20250219_3FRfun1      TNT_label
    df.index.name = 'roi'
    df = df.reset_index()
    df = preprocess_yang_data(df, drop_exp_type=drop_exp_type, drop_expt=False,
        # TODO have pfo in there by default?
        expected_nonmix_odors={'pfo', '2h @ -8', 'farn @ -4', 'ma @ -8'},
        _check_farn=True, verbose=verbose
    )

    # should be able to calculate:
    # All_resprate_respthresh1_trialfr0p5_wmixpred.csv from per fly files like:
    # 20250129_1FLfun1_resp_binary_respthresh1_trialfr0p5_wmixpred.csv
    #
    # same for both yang_dir and subdir? no, for subdir it's:
    # _responsive_binlabel_df_respthresh1_trialfr0.5.csv
    binarized_suffix = '_resp_binary_respthresh1_trialfr0p5_wmixpred.csv'
    fly_bin_dfs = []
    for resp_csv in sorted(yang_dir.glob(f'*{binarized_suffix}')):
        # NOTE: these also have columns like:
        # - 'experiment' (i assume this should all match prefix of file)
        # - '2h(-7)+ma(-7)_predicted' (which can all be dropped?)
        fly_bin_df = pd.read_csv(resp_csv, index_col=0)
        fly_bin_df.index.name = 'roi'

        expts = fly_bin_df['experiment'].unique()
        assert len(expts) == 1
        expt = expts[0]
        recording_id = resp_csv.name[:-len(binarized_suffix)]
        assert expt == recording_id
        fly_bin_df = fly_bin_df.drop(columns='experiment')

        # yang says this is just union of whether the cell responded to the components.
        # don't need that.
        fly_bin_df = fly_bin_df.drop(columns=[
            x for x in fly_bin_df.columns if x.endswith('_predicted')
        ])
        fly_bin_df = addlevel(fly_bin_df, 'brain', recording_id)
        fly_bin_dfs.append(fly_bin_df.astype(float))

    bdf = pd.concat(fly_bin_dfs, verify_integrity=True)

    df_recordings = set(df.index.get_level_values('brain'))
    df = df.droplevel('brain')

    bdf = bdf[bdf.index.get_level_values('brain').isin(df_recordings)].copy()
    bdf = preprocess_yang_data(bdf.reset_index(), drop_exp_type=drop_exp_type,
        verbose=False, expected_nonmix_odors={'pfo', '2h @ -8', 'farn @ -4', 'ma @ -8'},
        _check_farn=True
    )
    # seems like maybe neither way already sorted? at least on rows?
    df = df.sort_index()
    bdf = bdf.sort_index()
    # do seem to need this too, at least on one
    df = df.sort_index(axis='columns')
    bdf = bdf.sort_index(axis='columns')
    if drop_exp_type:
        assert df.index.equals(bdf.index)
    else:
        assert df.index.droplevel('exp_type').equals(bdf.index)
        # to add exp_type level (the last level) to bdf.index
        bdf.index = df.index.copy()

    # TODO refactor to share w/ assertions below?
    assert df.columns.get_level_values('odor').unique().equals(
        bdf.columns.get_level_values('odor')
    )
    assert (bdf.columns.get_level_values('repeat') == 0).all()
    assert df.columns.get_level_values('repeat').max() > 0
    #

    # should contain ma+2h and ea+eb data. all wildtype (exp_type='WT' from above)
    yang_subdir = yang_dir / '20260116_20260210'

    # should be able to calculate all (e.g.):
    # 20260121_2FLfun1_resp_rate_respthresh1trialfr0p5_wmixpred.csv
    # (which just contains mean response rates per odor) from (e.g.):
    # 20260121_2FLfun1_responsive_binlabel_df_respthresh1_trialfr0.5.csv
    binarized_suffix = '_responsive_binlabel_df_respthresh1_trialfr0.5.csv'

    # the 3rd and final file for each fly should have responses:
    # 20260116_1FLfun1_els_moco_s2p_mean_curated_Fc_zscored_full_cell_resp_df_odor_sorted.csv
    #
    # all flies in this subfolder should be "good", and can thus be concatenated
    # together to create comparable input as loaded from the single "good" CSV in
    # yang_dir
    response_csv_suffix = (
        '_els_moco_s2p_mean_curated_Fc_zscored_full_cell_resp_df_odor_sorted.csv'
    )
    fly_resp_csvs = sorted(yang_subdir.glob(f'*{response_csv_suffix}'))
    # the '*.csv' glob is really just to exclude stuff like '.~lock.<filename>.csv#'
    # when opening CSVs in OpenOffice
    all_subdir_files = [x for x in yang_subdir.glob('*.csv') if x.is_file()]
    assert len(fly_resp_csvs) * 3 == len(all_subdir_files), \
        'expected 3 files for each fly, and nothing else'

    fly_dfs = []
    fly_bin_dfs = []
    for resp_csv in fly_resp_csvs:
        # TODO note some/all of these have odor='air'. make sure that's also being
        # dropped (should be)
        fly_df = pd.read_csv(resp_csv, index_col=0)
        fly_df.index.name = 'roi'
        # all columns should be odor
        # TODO assert all columns same across CSVs? how else to concat? just let there
        # be NaNs for some odor X repeat combos, and handle inside preproces fn
        # probably
        recording_id = resp_csv.name[:-len(response_csv_suffix)]

        # TODO TODO oh, so these are collapsed over trials? want that? have some
        # version where it's not?
        fly_bin_df = pd.read_csv(yang_subdir / f'{recording_id}{binarized_suffix}',
            index_col=0
        )
        fly_bin_df.index.name = 'roi'

        odor_index1 = yang2tom_odor_index(fly_df)
        odors1 = odor_index1.get_level_values('odor').unique()

        odor_index2 = yang2tom_odor_index(fly_bin_df)
        odors2 = odor_index2.get_level_values('odor').unique()
        assert len(set(odors2)) == len(fly_bin_df.columns)
        assert (odor_index2.get_level_values('repeat') == 0).all()

        assert odors1.equals(odors2)
        max_n_trials = odor_index1.get_level_values('repeat').max() + 1
        # NOTE: only one fly had only 2 trials. drop this fly? rest had 3 or 4.
        # recording_id='20260116_1FLfun1'
        # max_n_trials=2
        # recording_id='20260121_2FLfun1'
        # max_n_trials=4
        # recording_id='20260121_2FLfun2'
        # max_n_trials=3
        # recording_id='20260121_2FRfun1'
        # max_n_trials=3
        # recording_id='20260202_1FLfun1'
        # max_n_trials=3
        # recording_id='20260202_1FRfun1'
        # max_n_trials=3
        # recording_id='20260203_2FLfun1'
        # max_n_trials=3
        # recording_id='20260203_2FRfun1'
        # max_n_trials=3
        # recording_id='20260210_2FRfun1'
        # max_n_trials=4
        #
        # TODO delete
        #print(f'{recording_id=}')
        #print(f'{max_n_trials=}')
        #

        fly_df = addlevel(fly_df, 'brain', recording_id)
        fly_df = addlevel(fly_df, 'exp_type', 'WT')
        fly_dfs.append(fly_df)

        fly_bin_df = addlevel(fly_bin_df, 'brain', recording_id)
        fly_bin_df = addlevel(fly_bin_df, 'exp_type', 'WT')
        # TODO casting to float to cause a bit less of a headache when concat later
        # introduces NaN. currently get a FutureWarning on first attempt of using the
        # output of that concatenation:
        # FutureWarning: In a future version, object-dtype columns with all-bool values
        # will not be included in reductions with bool_only=True. Explicitly cast to
        # bool dtype instead.
        fly_bin_dfs.append(fly_bin_df.astype(float))

    # TODO assert set of columns (union across all input dfs) is same as output columns
    # of concat?
    df2 = pd.concat(fly_dfs, verify_integrity=True)
    bdf2 = pd.concat(fly_bin_dfs, verify_integrity=True)

    # TODO (delete) is it only 2 repeats for each of these? or varying number?
    # (2 for one fly, 3-4 for rest)
    df2 = preprocess_yang_data(df2.reset_index(), drop_exp_type=drop_exp_type,
        expected_nonmix_odors={'pfo', 'Iaa @ -2', 'banana', 'Bmix28', 'air'},
        verbose=verbose
    )
    bdf2 = preprocess_yang_data(bdf2.reset_index(), drop_exp_type=drop_exp_type,
        verbose=False,
        expected_nonmix_odors={'pfo', 'Iaa @ -2', 'banana', 'Bmix28', 'air'}
    )

    assert df2.index.equals(bdf2.index)
    assert df2.columns.get_level_values('odor').unique().equals(
        bdf2.columns.get_level_values('odor')
    )
    assert (bdf2.columns.get_level_values('repeat') == 0).all()
    assert df2.columns.get_level_values('repeat').max() > 0

    # so despite there being '2h @ -7 + ma @ -7' (+ component) data in both datasets,
    # none of the flies overlap between the datasets
    # TODO drop it from the subdir dataset? or also plot it there?
    # otherwise, subdir dataset has (at two concs each) ea+eb and 2h+1o3ol, though
    # probably don't have any ideal ORN data to compare it too, since all lower than
    # concs I've used other than in the (poorly done / analyzed) ramp experiments
    assert len(df.index.droplevel('roi').drop_duplicates().intersection(
        df2.index.droplevel('roi').drop_duplicates()
    )) == 0

    # TODO better names than df and df2?
    df = addlevel(df, 'panel', 'diag-binaries')
    bdf = addlevel(bdf, 'panel', 'diag-binaries')

    df2 = addlevel(df2, 'panel', 'natmix-top2-dilute')
    bdf2 = addlevel(bdf2, 'panel', 'natmix-top2-dilute')

    df = pd.concat([df, df2], verify_integrity=True)
    bdf = pd.concat([bdf, bdf2], verify_integrity=True)

    # averaging over repeats
    # NOTE: need level= (in current pandas) or else sort=False doesn't behave
    df = df.groupby(level='odor', axis='columns', sort=False).mean()

    assert (bdf.columns.get_level_values('repeat') == 0).all()
    bdf = bdf.droplevel('repeat', axis='columns')

    assert df.columns.equals(bdf.columns)
    assert df.index.equals(bdf.index)
    assert df.isna().equals(bdf.isna())

    return df, bdf


def main():
    # TODO add CLI flag to save [+plot?] dynamics (so i can use outputs as input
    # for natmix_data/analysis.py at least?)?
    # TODO add flag to ignore cached outputs (and also to include/exclude based on model
    # param dir names? refactor to share all w/ some of my other scripts?)?
    parser = ArgumentParser()
    # TODO refactor to share this (and -e/-x) w/ step_model_pn_apl, and
    # natmix_data/analysis.py?
    parser.add_argument('model_output_dirnames', nargs='?', help='comma separated '
        'list of substrings matching model output directory names \n(subdirectories of '
        '<model_output_root>/<panel> directories). \nsee also -e and -x.'
    )
    parser.add_argument('-c', '--use-cache', action='store_true', help='uses cache for '
        'all model outputs, when available. this may mean no models are re-run.'
    )
    parser.add_argument('-o', '--only-analyze-cache', action='store_true', help='will '
        'not run any models. will only load cached model outputs and analyze those.'
    )
    parser.add_argument('-f', '--full-model-params', action='store_true', help='will '
        'run models for each set of parameters in `FULL_MODEL_KW_LIST`, instead of the '
        'much shorter `SHORT_MODEL_TUNE_KWS`'
    )
    parser.add_argument('-e', '--exclude-substrings', action='store', help='comma '
        'separated list of substrings to EXCLUDE model output directory names \n'
        'containing them. complementary to model_output_dirnames.'
    )
    parser.add_argument('-x', '--exact-model-dirnames', action='store_true',
        help='model_output_dirnames is interpreted as a list of exact directory names,'
        '\nrather than a list of substrings contained within some model output \n'
        'directory names.'
    )
    parser.add_argument('-d', '--save-dynamics', action='store_true', help='will write '
        'NetCDF files with many model dynamic variables (change plot root, since can '
        'take a lot of disk space)'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='print more')
    args = parser.parse_args()
    model_output_dirnames = args.model_output_dirnames
    use_cache = args.use_cache
    # TODO TODO summarize how many (+ which) we actually are analyzing at end of loop?
    only_analyze_cache = args.only_analyze_cache
    full_model_params = args.full_model_params
    exclude_substrings = args.exclude_substrings
    exact_model_dirnames = args.exact_model_dirnames
    save_dynamics = args.save_dynamics
    verbose = args.verbose
    quiet = not verbose
    # TODO set in fit_and_plot_mb_model instead? (currently errs if try_cache=False
    # [default] and cache_only=True [not default])
    if only_analyze_cache:
        use_cache = True

    if model_output_dirnames is not None:
        model_output_dirnames = model_output_dirnames.split(',')

    if exclude_substrings is not None:
        exclude_substrings = exclude_substrings.split(',')

    if not full_model_params:
        model_tune_kws = SHORT_MODEL_TUNE_KWS
    else:
        model_tune_kws = FULL_MODEL_KW_LIST
        warn(f'running models on all {len(FULL_MODEL_KW_LIST)} elements in '
            'FULL_MODEL_KW_LIST, because -f/--full-model-params passed!'
        )

    if save_dynamics:
        plot_root = Path('/mnt/d0/yang_mix_outputs').resolve()
        warn(f'writing to {plot_root=} instead of usual under current directory, b/c '
            'saving (potentially large) dynamics outputs'
        )
    else:
        plot_root = Path('yang_mix_outputs').resolve()

    plot_root.mkdir(exist_ok=True)

    # directories under this used to just be under plot_root, but especially when
    # running on FULL_MODEL_KW_LIST, it was getting pretty cluttered
    tune_root = plot_root / 'megamat-tuned'
    tune_root.mkdir(exist_ok=True)

    # otherwise we currently won't see the names of plots being saved printed in blue
    al_util.verbose = True

    # TODO CLI flag to set drop_exp_type=False/True? (and same flag to control
    # similar debug info for Remy KC experiments, if i include any of that)
    # TODO add CLI flag to skip processing KC data?
    # TODO TODO check drop_expt_type=False doesn't break anything (+ fix if not)
    # (seems ok, but still want to compare plots visually w/ and w/o)
    yang_df, yang_bin_df = load_yang_kc_data(drop_exp_type=False, verbose=verbose)

    model_strs = [format_model_params(x) for x in model_tune_kws]
    model_str2abbrev = {m: abbrev_model_id(m) for m in model_strs}
    if verbose:
        # TODO delete? now that i'm mostly not using the abbrevs anyway (for now, at
        # least, after mostly using model_pnkc_class now)
        print('model ID -> abbrev:')
        pprint(model_str2abbrev)
        print()

    df = megamat_orn_deltas(drop_diags=False)

    # no other panels in output of fn above
    #
    # indexing differently for `mdf`, since some of code using it below expects 'panel'
    # column level to be there still, but `df.loc[:, 'megamat']` would drop that level
    mdf = df.loc[:, df.columns.get_level_values('panel') == 'megamat']
    diags = df.loc[:, df.columns.get_level_values('panel') == 'glomeruli_diagnostics']

    natmix_df = natmix_orn_deltas()

    tune_df = mdf
    del mdf

    series_list = []
    # make binary mixtures of synthetic diagnostics (150Hz spike delta to each of the
    # two glomeruli, for all combinations of them)
    gloms_to_mix = ['DM4', 'VM5d', 'DC3']
    for x in gloms_to_mix:
        ser = pd.Series(index=df.index.copy(), name=f'{x}-300 @ 0', data=0.0)
        ser.loc[x] = 300.0
        ser.name = ('syn-diag-binaries', ser.name)
        series_list.append(ser)

    glom_combos = list(combinations(gloms_to_mix, 2))
    for x, y in glom_combos:
        # TODO also sort x and y alphabetically here?
        # (less important prob, since presumably won't be comparing this directly to any
        # real data, at least not without transforming to odor names that best
        # approximate these?)
        mser = pd.Series(index=df.index.copy(), name=f'{x}-150/{y}-150 @ 0', data=0.0)
        mser.loc[x] = 150.0
        mser.loc[y] = 150.0
        mser.name = ('syn-diag-binaries', mser.name)
        series_list.append(mser)

    # make binary mixtures by combining the real diagnostic data for each of these
    # three odors (should target same glomeruli as above). again all pairwise combos.
    diag_subset = diags.loc[:, diags.columns.get_level_values('odor').isin(
        ('2h @ -6', 'farn @ -2', 'ma @ -7')
    )]
    odor2glom = {
        '2h @ -6': 'VM5d',
        'farn @ -2': 'DC3',
        'ma @ -7': 'DM4',
    }
    assert diag_subset.index.equals(df.index)

    new_panels = ('diag-binaries_mean', 'diag-binaries_max', 'diag-binaries_max-rest0')
    for panel in new_panels:
        # replacing 'glomeruli_diagnostics' panel w/ each of these new ones, so
        # single components will appear in all panel specific plots
        comp_df = addlevel(diag_subset.droplevel('panel', axis='columns'), 'panel',
            panel, axis='columns'
        )
        if panel == 'diag-binaries_max-rest0':
            comp_df.loc[~comp_df.index.isin(odor2glom.values())] = 0

        # technically not just a list of series anymore... but should still work
        series_list.append(comp_df)

    # TODO also have this control whether concs appear (in a consistent manner) for
    # components (+rename)
    leave_concs_in_mix = False
    for x, y in combinations(diag_subset.columns.get_level_values('odor'), 2):
        # just to sort alphabetically on component names, to be consistent w/ other code
        # (mainly code loading and processing odors in yang's data, which uses olf fns
        # that do that)
        x, y = sorted([x, y])
        # hack so some of the string processing code inside modelling doesn't err
        # ideally we'd just have mix_name=f'{x} + {y}'
        # TODO that hack actually necessary? test?
        if leave_concs_in_mix:
            mix_name = f'{x.replace(" @ ", "")} + {y.replace(" @ ", "")} @ 0'
        else:
            # TODO want space on either side of +?
            mix_name = f'{parse_odor_name(x)}+{parse_odor_name(y)} @ 0'

        d1 = diag_subset.loc[:, (slice(None), x)].squeeze()
        d2 = diag_subset.loc[:, (slice(None), y)].squeeze()
        both = pd.concat([d1, d2], axis='columns', verify_integrity=True)
        both.columns.names = ['panel', 'odor']

        mean_ser = both.mean(axis='columns')
        max_ser = both.max(axis='columns')

        g1 = odor2glom[x]
        g2 = odor2glom[y]
        max_zerod_ser = max_ser.copy()
        max_zerod_ser[~max_zerod_ser.index.isin((g1, g2))] = 0

        mean_ser.name = 'mean'
        max_ser.name = 'max'
        max_zerod_ser.name = 'max, non-cognate gloms 0d'
        to_plot = pd.concat([
                both.droplevel('panel', axis='columns'),
                mean_ser, max_ser, max_zerod_ser
            ], axis='columns', verify_integrity=True
        )
        mix_suffix = get_odor_fname_suffix(f'{x}_and_{y}')
        plot_responses(to_plot, plot_root, f'diags_vs_constructed-mixtures{mix_suffix}',
            cbar_label='est. ORN firing rate delta (Hz)'
        )

        # first element of each of these should be in `new_panels` defined before this
        # loop (otherwise single components will not be added correctly)
        mean_ser.name = ('diag-binaries_mean', mix_name)
        series_list.append(mean_ser)
        max_ser.name = ('diag-binaries_max', mix_name)
        series_list.append(max_ser)
        max_zerod_ser.name = ('diag-binaries_max-rest0', mix_name)
        series_list.append(max_zerod_ser)

    test_df = pd.concat(series_list, axis='columns', verify_integrity=True)
    test_df.columns.names = ['panel', 'odor']
    assert not test_df.isna().any().any()

    # TODO TODO delete! this is dropping the panels:
    # 'syn-diag-binaries', 'diag-binaries_mean', 'diag-binaries_max-rest0'
    print('remove dropping of other synthetic panels!')
    test_df = test_df.loc[:,
        test_df.columns.get_level_values('panel') == 'diag-binaries_max'
    ]
    #

    # TODO TODO how different are thr and APL scale params if tuning on natmix_df
    # instead of megamat_df?
    # TODO TODO what about if we tune on one of the diag-subset dfs?

    # TODO want to do anything about his other than fillna(0)? prob not
    # ipdb> natmix_df.index.difference(test_df.index)
    # Index(['DA1', 'DA4l', 'DA4m', 'V', 'VA1d', 'VA1v'], ...
    # ipdb> test_df.index.difference(natmix_df.index)
    # Index(['VA4'], dtype='object', name='glomerulus')
    test_df = pd.concat([test_df, natmix_df], axis='columns', verify_integrity=True)
    test_df = test_df.fillna(0.0)

    test_df = test_df.sort_index().sort_index(axis='columns')
    del df

    panels = list(test_df.columns.get_level_values('panel').unique())

    # saw 0.212 on some kiwi/control stuff (tuned on megamat)
    # now .2228 on something
    #response_rate_plot_max = 0.23
    # need at least this for comparison to real KC data
    # TODO TODO switch between maxes depending on whether we are comparing to real KC
    # data or not ?
    response_rate_plot_max = 0.32

    # TODO delete
    #dfs = []
    # metadata in index, and column for each of:
    # responded,num_spikes,logistic_scaled_num_spikes
    model_roi_odor_dfs = []
    tuned_dfs = []
    failing_kws_and_tracebacks = []
    seen_model_strs = set()
    failed_model_strs = []
    # only populated if model_output_dirs (positiona) / -e / -x CLI args used to
    # restrict which model param combos we analyze
    skipped_model_strs = []
    # only populated if -o/--only-analyze-cache
    model_strs_without_cache = []
    for kws in tqdm(model_tune_kws, unit='model (on all panels)'):
        # TODO only print this if verbose? or if we don't end up skipping this dir?
        #print(f'{kws=}')

        # TODO rename plot_dirname? explicitly pass it as dirname? don't think it
        # should be necessary
        #
        # asserting that this matches created dirname right after fit_and_plot_mb_model
        # call below
        model_str = format_model_params(kws)
        assert model_str not in seen_model_strs, f'already saw {model_str=}'
        seen_model_strs.add(model_str)

        # TODO keep a list of what we skipped?
        skip = False
        # TODO should exact also apply here?
        # TODO err if model_output_dirnames is None/[] and exact...=True?
        if (exclude_substrings is not None and
            any(s in model_str for s in exclude_substrings)):
            warn(f'skipping {model_str} because it contained an exclude substring')
            skip = True

        if model_output_dirnames is not None:
            if (not exact_model_dirnames and
                not any(s in model_str for s in model_output_dirnames)):

                warn(f'skipping {model_str} because it did not match any substring '
                    'in model_output_dirnames'
                )
                skip = True

            elif (exact_model_dirnames and
                not any(s == model_str for s in model_output_dirnames)):
                warn(f'skipping {model_str} because it did not match any element of '
                    'model_output_dirnames'
                )
                skip = True

        if skip:
            skipped_model_strs.append(model_str)
            continue

        if not full_model_params or verbose:
            print(model_str)

        # TODO delete eventually (after everyone using model updates and runs this code,
        # or manually moves stuff)
        old_tune_model_dir = plot_root / model_str
        new_tune_model_dir = tune_root / model_str
        if old_tune_model_dir.exists():
            assert old_tune_model_dir.is_dir()
            assert not new_tune_model_dir.exists()
            warn(f'moving megamat-tuned dir {model_str} from old location under '
                f'{plot_root.name} to under {plot_root.name}/{tune_root.name}'
            )
            shutil.move(old_tune_model_dir, new_tune_model_dir)
            assert not old_tune_model_dir.is_dir()
            assert new_tune_model_dir.is_dir()
        #

        # TODO delete
        # TODO TODO add CLI flag for t his?
        # TODO TODO may need to delete empty/incomplete dirs for this to work...
        #if new_tune_model_dir.exists():
        #    warn('SKIPPING EXISTING TUNE_DIR')
        #    continue
        #

        # TODO why is this seemingly not using LR cache in home? it's still tuning
        # on megamat, so it should be the same, no? (true?)
        try:
            # TODO TODO put all these in a megamat/megamat_tuned/tuned dir. getting a
            # bit cluttered now
            tuned_params = fit_and_plot_mb_model(tune_root, orn_deltas=tune_df,
                try_cache=use_cache, cache_only=only_analyze_cache, quiet=quiet,
                return_dynamics=save_dynamics,
                # TODO change max_iters to 200?
                response_rate_plot_max=response_rate_plot_max, max_iters=100, **kws
            )
        except NoCachedModelOutputsError as err:
            if new_tune_model_dir.exists():
                # TODO maybe w/ flag to do/not do this?
                warn(f'deleting {new_tune_model_dir} that did not have minimum cache '
                    'requirements (probably error during creation)'
                )
                shutil.rmtree(new_tune_model_dir)

            warn(f'{err}\n...and -o/--only-analyze-cache set. skipping!')
            model_strs_without_cache.append(model_str)
            continue

        except AssertionError as err:
            if new_tune_model_dir.exists():
                # TODO maybe w/ flag to do/not do this?
                warn(f'deleting {new_tune_model_dir} because encountered error during '
                    'call to populate it'
                )
                shutil.rmtree(new_tune_model_dir)

            msg = traceback.format_exc()
            # TODO print these at end (+ fix assertion issue and remove this, ideally)
            failing_kws_and_tracebacks.append((kws, msg))
            warn(f'modelling call failed with the following message:\n{msg}\n'
                'skipping for now!\n'
            )
            failed_model_strs.append(model_str)
            continue

        assert tuned_params['output_dir'] == model_str, (
            f"{tuned_params['output_dir']=} != {model_str=}"
        )

        thr_and_apl_kws = get_thr_and_APL_weights(tuned_params, kws)
        # TODO TODO is it a mistake that i'm getting a vector for wAPLKC for some
        # of these? i.e. prat-claws_True. model at least working properly?
        # just a formatting mistake?
        if verbose:
            # TODO say more, like that we tuned on megamat and where those outputs are?
            print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')

        tuned_model_output_dir = tune_root / tuned_params['output_dir']
        trs = read_parquet(tuned_model_output_dir / 'responses.parquet')
        tss = read_parquet(tuned_model_output_dir / 'spike_counts.parquet')

        wPNKC = read_parquet(tuned_model_output_dir / 'wPNKC.parquet')
        raw_wPNKC = wPNKC.copy()
        # TODO only store wPNKC (+ later save) for a unique set of wPNKC params?
        # (most model params don't affect wPNKC. way too cluttered when saving under
        # root, esp when full_model_params=True)
        save_kc_glom_counts: bool = False
        if save_kc_glom_counts:
            if kws.get('one_row_per_claw', False):
                wPNKC = wPNKC.groupby(KC_ID).sum()

                if kws.get('prat_boutons', False):
                    wPNKC = wPNKC.droplevel(
                        [x for x in wPNKC.columns.names if x != 'glomerulus'],
                        axis='columns'
                    )
                    wPNKC = wPNKC.groupby('glomerulus', axis='columns').sum()

            kc_glom_combo_counts = wPNKC[gloms_to_mix].value_counts().sort_index()
            kc_glom_combo_counts.name = 'n_kcs'

            to_csv(kc_glom_combo_counts,
                plot_root / f'kc-glom-combo-counts_{model_str}.csv'
            )
            to_parquet(kc_glom_combo_counts,
                plot_root / f'kc-glom-combo-counts_{model_str}.parquet'
            )

        # TODO TODO plot binarized version, just counting # KCs getting any amount of
        # input from each combo? or some kind of dist of total amount of input per
        # combo? (separate line for those that only get input from one?)
        # TODO best way to plot this? for uniform model, can get value_counts like:
        # ipdb> wPNKC[gloms_to_mix].value_counts().sort_index()
        # DM4  VM5d  DC3
        # 0.0  0.0   0.0    1223
        #            1.0     178
        #            2.0       9
        #            3.0       1
        #      1.0   0.0     153
        #            1.0      15
        #            2.0       3
        #      2.0   0.0       7
        # 1.0  0.0   0.0     186
        #            1.0      17
        #      1.0   0.0      17
        #            1.0       1
        #      2.0   0.0       2
        # 2.0  0.0   0.0      14
        #            1.0       1
        #      1.0   0.0       1
        # dtype: int64

        mean_num_spikes = addlevel(tss.mean(), 'model', model_str)
        mean_num_spikes.name = 'mean_num_spikes'

        mean_response_rate = addlevel(trs.mean(), 'model', model_str)
        mean_response_rate.name = 'mean_response_rate'

        stats = pd.concat([mean_num_spikes, mean_response_rate], axis='columns',
            verify_integrity=True
        )
        tuned_dfs.append(stats)

        # TODO one tqdm that updates for each iteration of this too? copy another script
        # i have that does taht?
        for panel in panels:
            panel_dir = plot_root / panel
            panel_dir.mkdir(exist_ok=True)
            panel_df = test_df.loc[:,test_df.columns.get_level_values('panel') == panel]

            try:
                params = fit_and_plot_mb_model(panel_dir, orn_deltas=panel_df,
                    try_cache=use_cache, cache_only=only_analyze_cache, quiet=quiet,
                    return_dynamics=save_dynamics,
                    response_rate_plot_max=response_rate_plot_max, **kws,
                    **thr_and_apl_kws
                )
            # TODO TODO err if any of these are missing cached output, but some aren't
            # (including tuned above)?
            except NoCachedModelOutputsError as err:
                # TODO TODO just err here? or at least if any one of these works
                # but the others don't? do i want to generate this data for any panels i
                # can? report these failures separately? (don't want to just append
                # model_str... need panel too)
                model_strs_without_cache.append(f'{panel}/{model_str}')

                warn(f'when running pre-tuned model on {panel=}:\n{err}\n...and '
                    '-o/--only-analyze-cache set. skipping!'
                )
                continue

            model_output_dir = panel_dir / params['output_dir']
            rs = read_parquet(model_output_dir / 'responses.parquet')
            ss = read_parquet(model_output_dir / 'spike_counts.parquet')

            wPNKC2 = read_parquet(model_output_dir / 'wPNKC.parquet')
            assert raw_wPNKC.equals(wPNKC2), 'wPNKC should not change across tuned/not'

            def add_metadata(data):
                data = addlevel(data, 'model', model_str)
                return addlevel(data, 'panel', panel)

            mean_num_spikes = add_metadata(ss.mean())
            mean_num_spikes.name = 'mean_num_spikes'

            mean_response_rate = add_metadata(rs.mean())
            mean_response_rate.name = 'mean_response_rate'

            assert len(ss.squeeze().shape) == 2
            assert not ss.isna().any().any()
            assert ss.columns.name == 'odor'
            size_before = ss.size
            ss = ss.stack().rename('num_spikes')
            assert len(ss) == size_before
            assert isinstance(ss, pd.Series)
            assert not ss.isna().any()
            ss = add_metadata(ss)

            assert len(rs.squeeze().shape) == 2
            assert not rs.isna().any().any()
            assert rs.columns.name == 'odor'
            size_before = rs.size
            rs = rs.stack().rename('responded')
            assert len(rs) == size_before
            assert isinstance(rs, pd.Series)
            assert not rs.isna().any()
            rs = add_metadata(rs)

            # from some output of:
            # ```
            # ./analysis.py -r /mnt/d0/PNAPL_stepping/extra_panels/ pn-claw-to-apl_True,pn-claw-to-apl_False -x -m
            # ```
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=1.69, x0=2.10, L=2.98
            # final cost: 0.210
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.08
            # n_spikes=1: 0.40
            # n_spikes=2: 1.36
            # n_spikes=3: 2.45
            # n_spikes=4: 2.87
            # n_spikes=5: 2.96
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=1.47, x0=2.28, L=2.94
            # final cost: 0.213
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.10
            # n_spikes=1: 0.39
            # n_spikes=2: 1.18
            # n_spikes=3: 2.19
            # n_spikes=4: 2.72
            # n_spikes=5: 2.89
            #
            # from some output of: (on old, outdated outputs committed in natmix_data/)
            # ```
            # ./analysis.py -m
            # ```
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.10, x0=1.78, L=2.97
            # final cost: 0.220
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.07
            # n_spikes=1: 0.48
            # n_spikes=2: 1.82
            # n_spikes=3: 2.76
            # n_spikes=4: 2.94
            # n_spikes=5: 2.96
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.16, x0=1.78, L=3.07
            # final cost: 0.219
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.06
            # n_spikes=1: 0.48
            # n_spikes=2: 1.89
            # n_spikes=3: 2.86
            # n_spikes=4: 3.04
            # n_spikes=5: 3.07
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.01, x0=1.81, L=2.78
            # final cost: 0.217
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.07
            # n_spikes=1: 0.46
            # n_spikes=2: 1.66
            # n_spikes=3: 2.55
            # n_spikes=4: 2.74
            # n_spikes=5: 2.77
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.15, x0=1.79, L=3.13
            # final cost: 0.219
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.07
            # n_spikes=1: 0.49
            # n_spikes=2: 1.92
            # n_spikes=3: 2.92
            # n_spikes=4: 3.11
            # n_spikes=5: 3.13
            #
            # these seem like reasonable enough (round) defaults, from looking at
            # some fit outputs i was getting (in comments above)
            logistic_scaled_num_spikes = logistic(ss, k=2.0, x0=2.0, L=3.0)
            logistic_scaled_num_spikes = logistic_scaled_num_spikes.rename(
                'logistic_scaled_num_spikes'
            )

            model_roi_odor_sers = [ss, rs, logistic_scaled_num_spikes]
            m0_index = model_roi_odor_sers[0].index
            assert all(x.index.equals(m0_index)
                for x in model_roi_odor_sers[1:]
            )
            model_roi_odor_df = pd.concat(model_roi_odor_sers, axis='columns',
                verify_integrity=True
            )
            assert model_roi_odor_df.index.equals(m0_index)
            assert len(model_roi_odor_df.columns) == len(model_roi_odor_sers)
            model_roi_odor_dfs.append(model_roi_odor_df)

            # TODO delete
            #stats = pd.concat([mean_num_spikes, mean_response_rate], axis='columns',
            #    verify_integrity=True
            #)
            #dfs.append(stats)

        if not full_model_params or verbose:
            print()


    stat_names = model_roi_odor_dfs[0].columns
    for x in model_roi_odor_dfs[1:]:
        assert x.columns.equals(stat_names)
    stat_names = list(stat_names)

    def model_pnkc_class(model_str: str, *, prefix: str = '',
        # TODO it's actually a Counter (not dict), so use Mapping?
        pnkc2n_models: Optional[Dict[str, int]] = None) -> str:
        # TODO rename nonclaw?
        """Groups models into 'uniform'/'nonclaw'/'claw'/'bouton'
        """
        uniform_substr = 'pn2kc_uniform'
        wd20_substr = 'weight-divisor_20'
        claw_substr = 'prat-claws_True'
        bouton_substr = 'prat-boutons_True'

        if uniform_substr in model_str:
            assert not any(
                x in model_str for x in [wd20_substr, claw_substr, bouton_substr]
            )
            pnkc = 'uniform'

        elif wd20_substr in model_str:
            assert not any(x in model_str for x in [claw_substr, bouton_substr])
            pnkc = 'nonclaw'

        elif bouton_substr in model_str:
            # this one actually should always be present if bouton_substr is, for now
            assert claw_substr in model_str
            pnkc = 'bouton'

        else:
            assert claw_substr in model_str, f'{model_str=} did not match any classes'
            pnkc = 'claw'

        suffix = ''
        if full_model_params:
            if pnkc2n_models is not None:
                n_variants = pnkc2n_models[pnkc]
                if n_variants == 1:
                    suffix = f' ({n_variants} variant)'
                else:
                    suffix = f' ({n_variants} variants)'

        return f'{prefix}{pnkc}{suffix}'

    # TODO TODO summarize what is common among things that failed (and also that is not
    # shared w/ any of the stuff that succeeded?)?

    if full_model_params:
        print('concatenating model dfs...', end='', flush=True)

    # need the .reset_index() since KC_TYPE not present for uniform model (but is for
    # all others), so can't concat based on indices, which produces bad output when not
    # all have same level names
    model_roi_odor_df = pd.concat([x.reset_index() for x in model_roi_odor_dfs])
    assert not model_roi_odor_df[['panel', 'model', 'odor', 'kc_id']].duplicated(
        ).any()

    unique_model_ids = model_roi_odor_df.model.unique()
    if full_model_params:
        print('done', flush=True)

        warn(f'{len(unique_model_ids)}/{len(FULL_MODEL_KW_LIST)} model params '
            'successfully run'
        )

    def param_str_parts(model_str: str) -> frozenset:
        parts = model_str.split('__')
        assert len(parts) == len(set(parts))
        return frozenset(parts)

    def compare_missing_to_present(missing_strs: List[str], desc: str = 'missing'
        ) -> None:

        printed_header = False
        working_only_intersection2missing_dirs = defaultdict(list)
        working_only_intersection2working_dirs = defaultdict(list)
        # TODO delete if not necessary
        #missing_only_intersection2missing_dirs = defaultdict(list)
        #missing_only_intersection2working_dirs = defaultdict(list)
        #
        missing_parts_list = []
        for model_str in missing_strs:
            missing_parts = param_str_parts(model_str)
            missing_parts_list.append(missing_parts)

            largest_overlap = frozenset()
            # in case there are multiple distinct sets w/ same amount of overlap.
            # not sure if that will/could happen.
            n_overlapping2overlap_set = defaultdict(set)
            n_overlapping2working_model_strs = defaultdict(list)
            n_overlapping2working_parts = defaultdict(list)

            for working_model_str in unique_model_ids:
                assert working_model_str != model_str, 'model both working and not...'

                working_parts = param_str_parts(working_model_str)

                overlap = missing_parts & working_parts
                if len(overlap) >= len(largest_overlap):
                    largest_overlap = overlap

                if len(overlap) > 0:
                    n_overlapping2overlap_set[len(overlap)].add(overlap)
                    # TODO union/intersection across all these later? or
                    # union/intersection of differnces from missing_parts, in both
                    # directions?
                    n_overlapping2working_parts[len(overlap)].append(
                        working_parts
                    )
                    n_overlapping2working_model_strs[len(overlap)].append(
                        working_model_str
                    )

            if len(largest_overlap) == 0:
                continue

            # converting from frozenset to set just for nicer printing
            largest_overlaps = [
                set(x) for x in n_overlapping2overlap_set[len(largest_overlap)]
            ]
            # not sure there will ever be any cases when this len is NOT 1.
            # could assert, but don't really require it.
            if len(largest_overlaps) == 1:
                largest_overlaps = largest_overlaps[0]
                largest_overlaps_str = str(largest_overlaps)
            else:
                largest_overlaps_str = pformat(largest_overlaps)
            del largest_overlaps

            if not printed_header:
                # TODO update for accuracy/delete
                #print('missing model ID -> largest param set overlap(s) across working '
                #    'model IDs:'
                #)
                printed_header = True

            working_parts_list = n_overlapping2working_parts[len(largest_overlap)]
            # TODO pick some subset of these?
            working_only_union = set()
            missing_only_union = set()
            working_only_intersection = None
            missing_only_intersection = None
            for working_parts in working_parts_list:
                missing_only = missing_parts - working_parts
                working_only = working_parts - missing_parts
                working_only_union |= set(working_only)
                missing_only_union |= set(missing_only)

                if working_only_intersection is None:
                    working_only_intersection = set(working_only)
                else:
                    working_only_intersection &= set(working_only)

                if missing_only_intersection is None:
                    missing_only_intersection = set(missing_only)
                else:
                    missing_only_intersection &= set(missing_only)

            # TODO TODO why are these always empty? bug? or do i just not have any
            # parameters that always fail?
            assert len(missing_only_intersection) == 0, f'{missing_only_intersection=}'
            assert len(missing_only_union) == 0, f'{missing_only_union=}'

            # NOTE: this is NOT largest_overlapS, which was intermediate for printing
            working_model_strs = n_overlapping2working_model_strs[len(largest_overlap)]

            working_only_intersection2missing_dirs[frozenset(working_only_intersection)
                ].append(model_str)
            working_only_intersection2working_dirs[frozenset(working_only_intersection)
                ].extend(working_model_strs)

            # TODO delete if not necessary
            #missing_only_intersection2missing_dirs[frozenset(missing_only_intersection)
            #    ].append(model_str)
            #missing_only_intersection2working_dirs[frozenset(missing_only_intersection)
            #    ].extend(working_model_strs)
            #

            # TODO TODO try grouping by working/missing_only_intersection, and listing
            # working and non-working dirs for each of those? how many distinct such
            # sets?
            if verbose:
                # TODO hide this even if verbose? too much...
                print(f'{model_str}:\nlargest overlap(s): {largest_overlaps_str}')
                print('working models with this overlap:\n'
                    f'{pformat(working_model_strs)}'
                )
                # TODO delete (/pick subset)
                #print()
                #print(f'{working_only_union=}')
                #print(f'{working_only_intersection=}')
                ##print(f'{missing_only_union=}')
                ##print(f'{missing_only_intersection=}')
                #
                print()

        # TODO delete

        if verbose:
            print()

        # TODO TODO does summing len(missing_dirs) across all these give us all the
        # missing dirs (and they are unique across lists, right?)
        seen_missing_dirs = set()
        for working_only, missing_dirs in working_only_intersection2missing_dirs.items():
            assert not any(x in seen_missing_dirs for x in missing_dirs)
            assert len(missing_dirs) == len(set(missing_dirs))
            seen_missing_dirs.update(missing_dirs)
            if not verbose:
                continue

            # TODO reword "some working versions"?
            print(f'params shared by some working versions: {set(working_only)}')
            # TODO TODO print just the common subset of below instead? would reveal
            # prat_boutons is the common denominator for connectome APL case...
            print(f'missing model directories:')
            for d in missing_dirs:
                print(d)
            # TODO are there any failed directories that have some of the
            # working_only? (prob only care if they have *all*?)
            # TODO and are there any working directories that have any params
            # unique to the failing side?
            print()

        assert len(seen_missing_dirs) + len(unique_model_ids) == len(model_tune_kws)
        # TODO delete/use
        #working_parts_list = [param_str_parts(x) for x in unique_model_ids]
        # TODO TODO also print any missing dirs that contain these params (will there be
        # any?)

        # TODO delete
        #print()
        #print('missing_only_intersection2missing_dirs:')
        #pprint(dict(missing_only_intersection2missing_dirs))
        #

        # TODO maybe something simple (like set diff) makes sense within each
        # model_pnkc_class?


    # only populated if model_output_dirs (positiona) / -e / -x CLI args used to
    # restrict which model param combos we analyze
    if len(skipped_model_strs) > 0:
        warn(f'skipped_model_strs (b/c CLI args):\n{pformat(skipped_model_strs)}\n')

    if len(failed_model_strs) > 0:
        warn('failed_model_strs (AssertionError during this run):\n'
            f'{pformat(failed_model_strs)}\n'
        )
        compare_missing_to_present(failed_model_strs, 'failed')

    if len(model_strs_without_cache) > 0:
        warn('model_strs_without_cache (only relevant b/c -o/--only-analyze-cache):\n'
            f'{pformat(model_strs_without_cache)}\n'
        )
        compare_missing_to_present(model_strs_without_cache, 'without cache')

    pnkc2n_models = Counter([model_pnkc_class(x, prefix='') for x in unique_model_ids])

    # TODO number model w/in model_clas?
    model_roi_odor_df['model_pnkc_class'] = model_roi_odor_df.model.map(
        lambda x: model_pnkc_class(x, pnkc2n_models=pnkc2n_models)
    )
    model_roi_odor_df.model = model_roi_odor_df.model.apply(abbrev_model_id)

    # TODO need to do this in advance for mixes (or process here to strip conc,
    # leaving in format suitable for my conc stripping fn until here?)
    # i think there was concern the modelling code might still not handle odor strs
    # formatted as my code typically expects for mixtures? is that still true? fix that?
    # as a result, i'm processing diag-binaries_* mixes above to look like:
    # 'ma-7 + farn-2 @ 0' here, instead of 'ma @ -7 + farn @ -2'
    # (currently will just do in advance for mixtures, and continue processing like
    # above)
    model_roi_odor_df.odor = model_roi_odor_df.odor.map(parse_odor_name)

    # TODO TODO print # of spikes -> values, for first few # of spikes, and print max
    # before and after (after should approach L, right?)

    source_col = 'source'
    model_roi_odor_df = model_roi_odor_df.rename(columns={
        'model': source_col,
        'kc_id': 'roi',
    })

    # just since old df didn't have it. could keep?
    model_roi_odor_df = model_roi_odor_df.drop(columns='kc_type')
    # would need to subset to exclude KC_TYPE if we were not dropping above
    assert not model_roi_odor_df.isna().any().any()

    id_cols = ['panel', source_col, 'model_pnkc_class', 'odor', 'roi']
    assert set(id_cols) | set(stat_names) == set(model_roi_odor_df.columns)
    tidy = pd.melt(model_roi_odor_df, id_vars=id_cols, value_vars=stat_names,
        var_name='stat'
    )
    assert tidy['value'].size == model_roi_odor_df[stat_names].size
    assert not tidy.isna().any().any()
    model_roi_odor_df = tidy
    del tidy

    # TODO TODO also handle this from per-roi stuff, like all other outputs now
    # assuming only one panel ('megamat'), so it's ok this doesn't have a panel column
    tdf = pd.concat(tuned_dfs, verify_integrity=True)
    tdf = tdf.reset_index()
    tdf['model_pnkc_class'] = tdf.model.map(
        lambda x: model_pnkc_class(x, pnkc2n_models=pnkc2n_models)
    )
    tdf.model = tdf.model.apply(abbrev_model_id)
    tdf.odor = tdf.odor.map(parse_odor_name)
    tdf = pd.melt(tdf, id_vars=['model', 'model_pnkc_class', 'odor'],
        value_vars=['mean_num_spikes', 'mean_response_rate'], var_name='stat'
    )
    tdf = tdf.rename(columns={'model': source_col})

    tdf['connectome_apl'] = tdf[source_col].str.contains('_connectome-APL')
    tdf[source_col] = tdf[source_col].str.replace('_connectome-APL', '')
    # TODO (delete?) print tdf (/ use to set / check ylim below)

    # TODO also sort components in fixed order? matter (prob not?)?
    def odor_sort_fn(x):
        # TODO why are we also including '/'? are all mixtures not using '+'? provide
        # example of what uses '/' at least (or switch all to using '+'?)
        v1 = 1 * (x.str.contains('+', regex=False)) | (x.str.contains('/', regex=False))
        # to put the cmix0/kmix 0 at end
        v2 = 2 * x.str.contains('mix0', regex=False)
        return v1 + v2

    model_roi_odor_df = model_roi_odor_df[~(
        model_roi_odor_df.odor.str.contains('mix-') |
        model_roi_odor_df.odor.str.contains('(air mix)', regex=False
    ))].copy()

    # NOTE: sorting column index of yang_[bin_]df here did not fix odor order in
    # combined model + KC plots, as subsequent processing seemed to screw up order.
    # now doing similar sort_values call in get_yang_panel_means fn.
    model_roi_odor_df = model_roi_odor_df.sort_values(by='odor', kind='stable',
        key=odor_sort_fn
    )
    model_roi_odor_df = model_roi_odor_df.reset_index(drop=True)

    model_roi_odor_df['connectome_apl'] = model_roi_odor_df[source_col].str.contains(
        '_connectome-APL'
    )
    model_roi_odor_df[source_col] = model_roi_odor_df[source_col].str.replace(
        '_connectome-APL', ''
    )

    NORM_PREFIX: str = 'normalized_'

    # TODO TODO put NORM_[TO_FLYMEAN_MAX|PER_FLY] in fnames?

    # If False, the max individual fly (trial-averaged) response (across all KC data,
    # and all odors), will be sent to 1. If True, the max fly-averaged (after trial
    # averaging) odor response will be sent to 1, so some individual fly points could
    # exceed 1.
    # TODO TODO implement+try True (/ delete)
    NORM_TO_FLYMEAN_MAX: bool = True

    # If False, normalize (max->1) within each of the two datasets (across all odors):
    # - all models
    # - all individual points within all KC data, no matter the fly or exp_type
    #
    # If True, models are handled the same, but now each KC fly will have one odor with
    # each stat max=1.
    NORM_PER_FLY: bool = False
    # TODO delete? make sense? enough flies have all / most odors?
    #NORM_PER_FLY: bool = True

    # TODO happy with these?
    if not NORM_PER_FLY:
        NORM_DESC = 'within models and within KCs'
    else:
        NORM_DESC = 'within models and per KC fly'

    # TODO rename KC_RESPONSE_COL or something?
    resp_col: str = 'mean_Fc_zscore'

    compare_normalized = {
        # TODO separate mapping indicating what we should call this shared
        # thing? or maybe i should keep both names, esp if i want to compare
        # multiple normalized response strength metrics (i.e. adding a logistic
        # scaled version of # spikes)
        'mean_num_spikes': resp_col,
        'mean_logistic_scaled_num_spikes': resp_col,
    }

    unique_model_pnkc_classes = model_roi_odor_df['model_pnkc_class'].unique()
    assert 'KCs' not in unique_model_pnkc_classes, \
        f'{unique_model_pnkc_classes=}'
    # TODO (still an issue?) is this screwed up in the megamat case for some reason?
    # why do keys still have *_connectome-APL duplicates there but not for
    # subsequent calls?
    source_palette = dict(zip(
        unique_model_pnkc_classes,
        sns.color_palette(n_colors=len(unique_model_pnkc_classes))
    ))

    # TODO delete?
    # TODO refactor to share w/ al_util? (also used in al_analysis?)
    #light_grey = (.8, .8, .8)

    # TODO TODO maybe want something between 0.0 and 0.8? 0.2-3?
    # (yea (0,0,0) [=black] doesn't work in the twinx so much)
    #kc_color = (0.3, 0.3, 0.3)
    # TODO TODO TODO this work when other things are tuples? what's equivalent tuple for
    # this?
    kc_color = 'm'
    # TODO need to check if we actually load KC data? prob doesn't matter...
    source_palette['KCs'] = kc_color

    assert 'roi' in id_cols
    assert 'connectome_apl' not in id_cols
    id_cols = [x for x in id_cols if x != 'roi'] + ['connectome_apl', 'roi']

    df = model_roi_odor_df.groupby([x for x in id_cols if x != 'roi'] + ['stat'],
        sort=False).value.mean().reset_index()

    df.stat = df.stat.map({
        'logistic_scaled_num_spikes': 'mean_logistic_scaled_num_spikes',
        'num_spikes': 'mean_num_spikes',
        'responded': 'mean_response_rate',
    })

    # above doesn't screw up odor sorting
    assert df.sort_values(by='odor', kind='stable', key=odor_sort_fn).equals(df)

    COL_ORDER = [
        'mean_logistic_scaled_num_spikes',
        'mean_num_spikes',
        'mean_response_rate',
    ]
    mean_stat_names = list(df.stat.unique())
    assert set(COL_ORDER) == set(mean_stat_names)
    assert len(set(mean_stat_names)) == len(mean_stat_names)

    # this is only taken a param of pointplot (which catplot should pass it to),
    # but pointplot only defines it per hue level, so would prob need to either
    # duplicate colors (and have same len list of colors and markers) or two calls?
    #
    # TODO delete?
    uniform_apl_marker = '.'
    # can't get this to reproduce my open circles w/ stripplot (if only i could access
    # fillstyle...)
    #uniform_apl_marker = 'o'
    connectome_apl_marker = '+'

    uniform_apl_linestyle = '-'
    connectome_apl_linestyle = '--'

    marker_kws = dict(
        # default markeredgewidth seems ~2
        # TODO TODO set even lower markeredgewidth/markersize if full_model_params?
        # (would want to copy this dict to model_marker_kws, since this is
        # currently also used for KC data)
        # TODO TODO as long as i'm using strip plot and not pointplot, i don't think i
        # need this
        linestyle='none',
        # TODO take out of here? use model_markersize (always the largest one?)
        markersize=8.0
    )

    kc_err_alpha = 0.4
    kc_point_alpha = 0.7

    #  will be set to model_alpha_for_legend=0.7 in add_fixed_legend
    model_alpha = 0.5
    # TODO TODO have this apply to markers too, or also set edgewidth and/or size of
    # those differently in these two cases?
    # TODO what linewidth do i want by default?
    model_linewidth = 1
    model_markersize = 8.0

    # TODO delete?
    model_markeredgewidth = 1.0

    if full_model_params:
        # TODO TODO still use default alpha (and/or plot last) for uniform, in this
        # case, since there's only one point liek that and it's hard to find
        # TODO dodge/jitter more? other changes?
        # will be set to model_alpha_for_legend=0.7 in add_fixed_legend
        model_alpha = 0.15
        # TODO use this for everything? (markers included? comparable to
        # markeredgewidth?)
        model_linewidth = 0.5
        model_markersize = 6.0

        # TODO delete?
        model_markeredgewidth = 0.5

    # do need linewidth=1.5 (or something nonzero, to see anything for '+' marker.
    # edgecolor='face' is not enough). do need edgecolor='face' or else seaborn puts
    # black/grey border. also need to make points not filled.
    # TODO also have linewidth here depend on full_model_params?
    # i prefer edgecolor='none' to edgecolor='face' (w/ marker='.' at least)
    model_marker_kws = dict(size=model_markersize, linewidth=1.5, edgecolor='none',
        # does nothing (with marker='.' at least. same w/ ='o')
        #facecolor='none'
        # TypeError: stripplot() got multiple values for argument 'data'
        #data=dict(fillstyle='none')
        # doesn't produce open circles like i want
        #facecolor='none'
        # doesn't work (could pass in data=dict() tho?)
        #fillstyle='none'
    )

    def add_fixed_legend(g: sns.axisgrid.FacetGrid, df: pd.DataFrame,
        lines: bool = True) -> None:

        legend_data = dict(g._legend_data)

        # sort into a particular order? (should already be in order i want, with
        # uniform first and more complex models moving down)
        label_order = list(unique_model_pnkc_classes)

        model_alpha_for_legend: float = 0.7
        for k, artist in legend_data.items():
            # TODO worth changing calls creating `g` inputs s.t. 'KCs' is never in
            # legend already? prob not?
            if k == 'KCs':
                continue

            assert k in label_order, f'{k=} not in model keys ({label_order})'

            assert_msg = ('expected either separate alpha w/ RGB, or get_alpha()=None '
                'with RGBA color'
            )
            color = artist.get_color()
            if artist.get_alpha() is not None:
                assert len(color) == 3, assert_msg
                artist.set_alpha(model_alpha_for_legend)
            else:
                assert len(color) == 4, assert_msg
                rgb = color[:-1]
                artist.set_color(rgb + (model_alpha_for_legend,))

        # TODO is this only an issue because mean_response_rate doesn't have a
        # separate twin ax to plot real KC data on? or would it be an issue
        # regardless? either way, this is probably the easiest fix.
        if 'exp_type' in df.columns:
            unique_exp_types = df.exp_type.dropna().unique()
            legend_data = {
                k: v for k, v in legend_data.items() if k not in unique_exp_types
            }

        # how to get multiple "titles" with different section in legend:
        # https://stackoverflow.com/questions/24787041
        # other (more complicated answer) might even allow nice left alignment, but
        # going with simpler solution using title_proxy Rectangle for now.
        title_proxy = Rectangle((0,0), 0, 0, color='w')
        # TODO (delete. i think issue was just that there *are* no uniform-APL 'boutons'
        # cases) try to replace all existing (PN>KC) (so != 'KCs') legend entries w/
        # the circle artists of the same hue, so it doesn't matter which we plot first
        # (currently only the bouton case is getting + as marker. it does have some
        # non-connectome-APL cases right? or no?)
        # TODO add a suffix to 'boutons' clarifying it only has connectome-APL variant,
        # if leaving marker as-is there?

        pnkc_title = 'model PN>KC connectivity:'
        legend_data[pnkc_title] = title_proxy
        label_order = [pnkc_title] + label_order

        line_kws = dict(linestyle='-', marker='none')

        empty_line = ''
        legend_data[empty_line] = title_proxy
        # TODO can i repeat an entry in label_order, or will i need to define something
        # like empty_line2 = ' '?
        label_order.append(empty_line)

        # since they are handled w/ a separate plot call that just plots KCs, and we
        # set legend=False on that
        if 'KCs' in df[source_col].unique() and 'KCs' not in legend_data:
            # TODO pass in kc_alpha (but use this as default)?
            legend_data['KCs'] = Line2D([0], [0], color=kc_color, alpha=kc_err_alpha,
                **line_kws
            )
            # TODO also do one for KC points? (if any. add flag?)

        if 'KCs' in legend_data:
            label_order = ['KCs', empty_line] + label_order

        replace = {
            'bouton': 'bouton (no uniform APL)',
        }
        if not lines:
            replace.update({
                'KCs': 'KCs (95% CI on mean)',
            })

        legend_data = {
            (replace[k] if k in replace else k): v for k, v in legend_data.items()
        }
        label_order = [replace[k] if k in replace else k for k in label_order]

        apl_artist_kws = dict(color='k', alpha=0.5)
        if lines:
            apl_artist_kws.update(line_kws)
        else:
            apl_artist_kws.update(marker_kws)

        # assumed the markers we want are either lines (solid vs dashed) or point
        # circles/X's markers
        if lines:
            uniform_apl_kws = dict(linestyle=uniform_apl_linestyle)
            connectome_apl_kws = dict(linestyle=connectome_apl_linestyle)
        else:
            uniform_apl_kws = dict(marker=uniform_apl_marker, markeredgewidth=0.0)
            connectome_apl_kws = dict(marker=connectome_apl_marker, markeredgewidth=1.0)

        apl_title = 'model APL (within each hue):'
        legend_data[apl_title] = title_proxy
        label_order.append(apl_title)

        # TODO (delete) automatically switch between *_apl_marker and *_apl_linestyle,
        # based on type of other legend artists?
        #
        # adding an extra space at end here so that it doesn't conflict with 'uniform'
        # PN>KC entry from earlier
        uniform_apl = 'uniform '
        assert uniform_apl not in legend_data, 'would overwrite previous PN>KC uniform'
        legend_data[uniform_apl] = Line2D([0], [0],
            **{**apl_artist_kws, **uniform_apl_kws}
        )
        connectome_apl = 'connectome'
        legend_data[connectome_apl] = Line2D([0], [0],
            **{**apl_artist_kws, **connectome_apl_kws}
        )
        label_order += [uniform_apl, connectome_apl]
        assert set(label_order) == set(legend_data.keys()), \
            f'{label_order=} {legend_data.keys()=}'

        g.add_legend(legend_data=legend_data, label_order=label_order,
            # TODO just always have it be source_col?
            #title='model variant' if source_col == 'model' else 'source'
            title=''
        )


    mean_prefix = 'mean_'

    def plot_panel_stats_across_models(df: pd.DataFrame, panel: str, suffix: str = ''
        ) -> None:

        stat2ymax = {
            'mean_response_rate': response_rate_plot_max,

            'mean_num_spikes': 0.4,

            'mean_logistic_scaled_num_spikes': 0.3,
        }

        if 'model_stat' in df.columns:
            max_model_stats = df[['model_stat','value']].groupby('model_stat').max(
                ).squeeze()

            for k, v in max_model_stats.items():
                if k.startswith(NORM_PREFIX):
                    continue

                # TODO warn/err if not startswith mean_prefix?
                assert k.startswith(mean_prefix)
                v2 = stat2ymax[k]
                if v > v2:
                    # TODO TODO TODO is this even triggering? why is scale still broken?
                    warn(f'model max {k} ({v:.1f}) exceeded hardcoded stat2ymax value '
                        f'({v2:.1f}). replacing value in stat2ymax!'
                    )
                    stat2ymax[k] = v

        def plot_fn(data, *cols, **kwargs):
            is_normalized = False
            if 'is_normalized' in data.columns:
                is_normalized = data.is_normalized.unique()
                assert len(is_normalized) == 1, f'{is_normalized=}'
                is_normalized = is_normalized[0]

            ax = plt.gca()
            plotted_kc_data = False
            # real KC data will only have connectome_apl=False
            for connectome_apl in (True, False):
                # doing it this way, rather than initial groupby, so that point markers
                # get plotted last, so all i have to do is add a separate colorless '+'
                # marker to legend to handle that (was mix of '.' and '+' in legend,
                # since uniform doesn't have connectome APL case.
                gdf = data[data.connectome_apl == connectome_apl]
                marker = (
                    uniform_apl_marker if not connectome_apl else connectome_apl_marker
                )

                # TODO just move the kc stuff before / after the loop?
                ylabel = None
                from_kcs = gdf[source_col] == 'KCs'
                # intentionally plotting KC stuff first, since it's busier and i want
                # model stuff to be plotted on top, to stand out better
                if from_kcs.any():
                    assert not plotted_kc_data, ('should only be present for '
                        'connectome_apl=False case'
                    )
                    plotted_kc_data = True

                    kc_stats = data['kc_stat'].dropna().unique()
                    assert len(kc_stats) == 1
                    kc_stat = kc_stats[0]

                    model_stats = data['model_stat'].dropna().unique()
                    assert len(model_stats) == 1
                    model_stat = model_stats[0]

                    # model_stat != kc_stat to exclude case where BOTH are not
                    # normalized i.e. top right stat=mean_response_rate for both.
                    # TODO may still want to try handling that case here, esp if i don't
                    # end up also normalizing mean_response_rate (scales pretty
                    # different)
                    if not is_normalized and (model_stat != kc_stat):
                        kc_ax = ax.twinx()
                        kc_ax.spines['top'].set_visible(False)
                        # TODO leave this one, but change color (+alpha?) to kc_color
                        #kc_ax.spines['right'].set_visible(False)
                        kc_ax.set_ylabel(f'KC {kc_stat}', color=kc_color)
                        kc_ax.tick_params(axis='y', color=kc_color)
                        for text in kc_ax.yaxis.get_ticklabels():
                            text.set_color(kc_color)

                        assert model_stat != 'response_strength', ('should not have '
                            'name of normalized stat ("response_strength") here'
                        )

                        model_ymax = None
                        try:
                            model_ymax = stat2ymax[model_stat]
                        except KeyError:
                            warn(f'{model_stat=} missing from stat2ymax. could not '
                                'align unnormalized model vs KC response strength.\n'
                                f'{gdf[~from_kcs].value.max()=}'
                            )

                        if model_ymax is not None:
                            # TODO TODO TODO why is this so screwed up again in -f case?
                            # kiwi numspikes and logistic numspikes barely have any KC
                            # CIs on the axes....
                            # TODO TODO TODO is it that the other ax changes scale
                            # later?
                            # TODO TODO TODO calculate model stat max in here, rather
                            # than use stat2ymax dict?
                            model_mean = gdf[~from_kcs].value.mean()
                            scale_relative_to_ymax = model_mean / model_ymax
                            # taking mean w/in odor first, then mean, so we don't weight
                            # odors w/ more flies more. shouldn't matter too much.
                            kc_mean = gdf[from_kcs].groupby('odor').value.mean().mean()
                            kc_ymax = kc_mean / scale_relative_to_ymax
                            assert (gdf[from_kcs].value >= 0).all(), 'set diff ymin'
                            kc_ax.set_ylim([0.0, kc_ymax])
                    else:
                        kc_ax = ax

                    kwargs_without_color = {
                        k: v for k, v in kwargs.items() if k != 'color'
                    }

                    err_kws = dict(linewidth=1.5, alpha=kc_err_alpha)
                    # TODO just leave color to palette again?
                    sns.pointplot(gdf[from_kcs], *cols, color=kc_color,
                        # need dodge=False here as long as we only have one hue level
                        # here, or else will get ZeroDivisionError
                        dodge=False, marker='none', linestyle='none',
                        alpha=kc_point_alpha, seed=1, err_kws=err_kws,
                        # legend=False here hides legend inside this plot, but then
                        # there is also no marker next to 'KCs' in outside legend
                        # (will try to add handling for that outside)
                        # TODO decrease capsize from 0.3?
                        capsize=0.3, ax=kc_ax, legend=False, **kwargs_without_color
                    )

                    # TODO restore true?
                    plot_individual_flies = False
                    if plot_individual_flies:
                        # TODO also use kc_color here?
                        stripplot_kws = dict(color='k', legend=False, alpha=0.4,
                            size=2.0
                        )
                        _debug_exp_type = False
                        # TODO rename this flag to be more generic, and then also use
                        # hue to show '5comp' vs 'binary' in remy kiwi/control cases?
                        if _debug_exp_type and 'exp_type' in gdf.columns:
                            # TODO also try fly_id? different plot fn, w/ lines
                            # connecting points for one fly, across odors? multiple
                            # calls, each w/ diff marker (instead of hue)?
                            assert 'palette' not in kwargs
                            unique_exp_types = gdf[from_kcs].exp_type.unique()
                            exp_type_palette = dict(zip(
                                sorted(unique_exp_types),
                                sns.color_palette('hls', len(unique_exp_types))
                            ))
                            stripplot_kws = dict(hue='exp_type',
                                palette=exp_type_palette, alpha=0.6, size=2.0
                            )

                        # TODO set jitter=False?
                        sns.stripplot(gdf[from_kcs], *cols, ax=kc_ax, **stripplot_kws,
                            **kwargs_without_color
                        )
                    else:
                        if verbose:
                            warn('not plotting individual flies, b/c '
                                f'{plot_individual_flies=}'
                            )
                    # TODO TODO say somewhere what the errorbars is 95% CI (on mean) by
                    # default

                # can't use float dodge here like in some other places unfortunately.
                # TODO ig i could make figure wider?
                # jitter=1.0 is too much. 0.3 too much too, esp w/ aspect=1.1
                sns.stripplot(gdf[~from_kcs], *cols, hue='model_pnkc_class',
                    jitter=0.13, dodge=False, palette=source_palette, marker=marker,
                    alpha=model_alpha, ax=ax, **model_marker_kws, **kwargs
                )


        def plot_stats(df: pd.DataFrame, extra_suffix: str = '') -> None:
            from_model = df[source_col] != 'KCs'
            kc_response_stat = None
            if from_model.all():
                df = df[~ df.stat.str.startswith(NORM_PREFIX)].copy()
                col_order = [x for x in COL_ORDER if x in df.stat.unique()]
                facet_kws = dict()
            else:
                df = df.copy()
                model_stats = set(df[from_model].stat.unique())
                kc_stats = set(df[~from_model].stat.unique())

                model_normed_response_stats = {
                    x for x in model_stats if x.startswith(NORM_PREFIX)
                }
                kc_normed_response_stats = {
                    x for x in kc_stats if x.startswith(NORM_PREFIX)
                }
                # TODO TODO how to handle case where we actually do have multiple
                # of these, e.g. {'normalized_mean_logistic_scaled_num_spikes',
                # 'normalized_mean_num_spikes'}???
                # TODO TODO just make two calls, w/ diff subsets? (yea, prob for now)
                #
                # TODO also filter out stuff that endswith mean_response_rate, or do
                # anything special there? (if assertion we only have one of each here
                # fails because i add a normed mean_response_rate stat, can handle
                # then)
                assert len(model_normed_response_stats) == 1, \
                    f'{model_normed_response_stats=}'

                assert len(kc_normed_response_stats) == 1, \
                    f'{kc_normed_response_stats=}'

                model_normed_response_stat = model_normed_response_stats.pop()
                kc_normed_response_stat = kc_normed_response_stats.pop()

                # TODO also need to relax this if using a normalized mean_response_rate
                # column
                assert model_normed_response_stat != kc_normed_response_stat, \
                    f'{model_normed_response_stat=} == {kc_normed_response_stat}'

                model_response_stat = model_normed_response_stat[len(NORM_PREFIX):]
                assert model_response_stat in model_stats
                assert model_stats == {'mean_response_rate',
                    model_normed_response_stat, model_response_stat
                }

                kc_response_stat = kc_normed_response_stat[len(NORM_PREFIX):]
                assert kc_response_stat in kc_stats
                # TODO may need to update if i include logistic scaled stuf in same plot
                assert kc_stats == {'mean_response_rate',
                    kc_normed_response_stat, kc_response_stat
                }

                df['is_normalized'] = df.stat.str.startswith(NORM_PREFIX)

                # saving original stat names, so we can access these later (instead of
                # just 'response_strength', after replacing all below
                df.loc[from_model, 'model_stat'] = df.stat
                df.loc[~from_model, 'kc_stat'] = df.stat

                shared_norm_stat_name = 'response_strength'
                df['stat'] = df.stat.replace({
                    # want same stat name across rows (i.e. whether normalized or not),
                    # so stat can be col_order and then row can be is_normalized
                    model_response_stat: shared_norm_stat_name,
                    kc_response_stat: shared_norm_stat_name,
                    model_normed_response_stat: shared_norm_stat_name,
                    kc_normed_response_stat: shared_norm_stat_name,
                })

                if kc_response_stat.startswith(mean_prefix):
                    kc_response_stat = kc_response_stat[len(mean_prefix):]

                if model_response_stat.startswith(mean_prefix):
                    model_response_stat = model_response_stat[len(mean_prefix):]

                col_order = [shared_norm_stat_name, 'mean_response_rate']
                row_order = [False, True]
                facet_kws = dict(row='is_normalized', row_order=row_order)

            assert set(df.stat.unique()) == set(col_order), \
                f'{set(df.stat.unique())=} != {set(col_order)=}'

            g = sns.FacetGrid(data=df, col='stat', col_order=col_order, sharey=False,
                aspect=1.1, **facet_kws
            )
            g.map_dataframe(plot_fn, x='odor', y='value')

            add_fixed_legend(g, df, lines=False)

            assert g.col_names == col_order, f'{g.col_names=} != {col_order=}'

            # if only columns, axes.shape[0] should be 1 (w/ number of cols in shape[1])
            assert len(g.axes.shape) == 2, f'{g.axes.shape=}'

            for (i, j, hue), gdf in g.facet_data():
                ax = g.axes[i, j]
                # hue is index into g.hue_names. ig i'm not actually managing hue
                # through the FacetGrid, at least for this one? so i can just assert
                # this
                assert g.hue_names is None and hue == 0, ('hue was not previously '
                    'managed by FacetGrid. would need to update code'
                )

                if len(gdf) == 0:
                    # should be normalized mean_response_rate only
                    assert g.row_names[i] == True
                    assert g.col_names[j] == 'mean_response_rate'
                    continue

                is_normed = False
                if 'is_normalized' in gdf.columns:
                    is_normed = gdf.is_normalized
                    if is_normed.all():
                        is_normed = True
                    else:
                        assert not is_normed.any()
                        is_normed = False

                stat_col = 'stat'
                assert not gdf[source_col].isna().any()
                from_kcs = gdf[source_col] == 'KCs'

                if from_kcs.any():
                    assert 'model_stat' in gdf.columns
                    stat_col = 'model_stat'

                # TODO need to subset to ~from_kcs?
                # dropna should only be dropping model_stat rows for source == 'KCs'
                model_stats = gdf[stat_col].dropna().unique()
                assert len(model_stats) == 1, f'{model_stats=}'
                model_stat = model_stats[0]

                # this should always contain the unnormalized value
                assert model_stat != 'response_strength', f'{model_stat=}'

                sser = gdf[~from_kcs].value
                # TODO true?
                assert len(sser) > 0

                # how much is needed to make sure at least KC confidence
                # intervals are fully shown? adjust based on that automatically?
                # 0.2 currently seems to barely work in all the cases i care about
                # (kiwi/control/diag-binaries_max)
                margin = 0.2

                ymin = 0
                if not is_normed:
                    try:
                        ymax = stat2ymax[model_stat]
                    except KeyError as err:
                        ymax = sser.max()
                        # (err just contains name of missing key)
                        warn(f'stat2ymax missing model_stat={err}\nsetting {ymax=:.3f}')
                else:
                    # only other place that uses stat2ymax does not deal with any
                    # normalized data, so we can hardcode this here, and exclude this
                    # from stat2ymax.
                    #
                    # just adding a tiny bit of margin on top of 1.0, to clip points
                    # less mainly. all normalized (typically max=1.0) data currently
                    # takes the name 'response_strength'
                    ymax = 1.0 + margin

                try:
                    assert sser.min() >= ymin, f'{model_stat=} {sser.min()=} < {ymin=}'
                except AssertionError as err:
                    ymin = sser.min()
                    warn(f'{err}\nsetting {ymin=:.3f}')

                try:
                    assert sser.max() <= ymax, f'{model_stat=} {sser.max()=} > {ymax=}'
                except AssertionError as err:
                    ymax = sser.max() + margin
                    warn(f'{err}\nsetting {ymax=:.3f}')

                ax.set_ylim([ymin, ymax])
                if is_normed:
                    assert g.row_names[i] == True
                else:
                    assert g.row_names == [] or g.row_names[i] == False

                if j == 0 or i == 0:
                    if model_stat.startswith(NORM_PREFIX):
                        # TODO delete
                        #ylabel = f'normalized\n{model_stat[len(NORM_PREFIX):]}'

                        # TODO like this better than in suptitle?
                        ylabel = f'normalized\n({NORM_DESC})'
                        # TODO delete?
                        #ylabel = 'normalized'
                    else:
                        # TODO delete
                        #ylabel = model_stat
                        ylabel = 'raw'

                    ax.set_ylabel(ylabel)

                if i == 0:
                    title = model_stat
                    if title.startswith(mean_prefix):
                        title = title[len(mean_prefix):]
                    # TODO assert it doesn't start with NORM_PREFIX?

                    if (panel in natmix_panels and 'response_rate' in model_stat and
                        not from_model.all()):

                        assert kc_response_stat is not None
                        title += ('\nKC threshold: '
                            f'{kc_response_stat}>={NATMIX_KC_THRESH:.1f}'
                        )

                    # TODO what's good here? 7 too small. may need ~10?
                    ax.set_title(title, fontsize=10)
                else:
                    # TODO or None?
                    ax.set_title('')

            ncols = g.axes.shape[1]
            assert len(g.axes.shape) == 2 and ncols > 1, f'{g.axes.shape=}'
            nrows = g.axes.shape[0]
            labels = None
            if g.row_names != []:
                assert nrows > 1
                # seems that xticklabels not currently defined for anything but last
                # row, but still not showing up for last row for some reason
                for i, col_name in zip(range(ncols), g.col_names):
                    ax = g.axes[-1, i]
                    # TODO check other cols if labels are defined (not just [-1,i])?
                    if labels is None:
                        labels = ax.get_xticklabels()
                    else:
                        l2 = ax.get_xticklabels()
                        # == doesn't seem to return True when i'd want for Text objects
                        # reprs are like "Text(0, 0, '2h')", and include both position
                        # of label and text
                        assert all(repr(x) == repr(y) for x, y in zip(labels, l2))
                        assert all(
                            x.get_text() == y.get_text() for x, y in zip(labels, l2)
                        )

                    if col_name == 'mean_response_rate':
                        ax.remove()

                # trying to make more space for right y-label (for twinx overlay on top
                # left axes)
                # .05 is less than default. .15 too it seems
                # TODO g.fig is same as g.figure, right? (was using latter before here)
                # (docs say figure should be preferred moving forward)
                # TODO could use move_legend, but don't really care
                g.fig.subplots_adjust(wspace=0.75, hspace=0.4)
            else:
                assert nrows == 1

            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)

            # labels=None here still produces the labels I want for nrows=1 case
            g.set_xticklabels(labels=labels, rotation=90)

            suptitle = panel
            if (~ from_model).any():
                assert df.is_normalized.any()

            g.fig.suptitle(suptitle, y=1.04)
            # normalize_fname=False to not convert '__' -> '_'
            savefig(g, plot_root, f'{panel}{suffix}{extra_suffix}',
                normalize_fname=False
            )

        if 'connectome_apl' in df.columns:
            df = df.copy()
            df.connectome_apl = df.connectome_apl.fillna(False)

        # TODO TODO try to combine logistic scaled into same plot...
        is_model = df[source_col] != 'KCs'
        logistic_scaled = df.stat.str.contains('logistic_scaled')
        raw_num_spikes = df.stat.str.contains('num_spikes') & ~logistic_scaled

        if (~ is_model).any():
            plot_stats(df[~is_model | ~raw_num_spikes], '_vs_kcs_logistic-scaled')
            plot_stats(df[~is_model | ~logistic_scaled], '_vs_kcs')

        if logistic_scaled.any():
            plot_stats(df[is_model & ~raw_num_spikes], '_logistic-scaled')

        plot_stats(df[is_model & ~logistic_scaled])
        # TODO is this just missing from megamat tdf or what?

        # TODO or just compare all pairs of normalized things (across model vs
        # fly)? will only be one pair initially anyway


    yang_panel = yang_df.index.get_level_values('panel')
    assert yang_panel.equals(yang_bin_df.index.get_level_values('panel'))
    def get_yang_panel_ser(panel: str, df: Optional[pd.DataFrame] = None) -> pd.Series:
        if df is None:
            df = yang_df

        flyroi_ser = df.loc[yang_panel == panel].dropna(how='all', axis='columns'
            ).stack()
        # after .stack(), Series index should be `fly_cols + ['roi' ,'odor]`, and there
        # should be no NaN (but could be varying # of ROIs per fly, and potentially
        # different odors for different flies)
        assert not flyroi_ser.isna().any()
        return flyroi_ser


    def strip_concs(odors: Union[pd.Index, pd.Series]) -> Union[pd.Index, pd.Series]:
        return odors.map(olf.strip_concs_from_odor_str
            ).str.replace(component_delim, '+', regex=False)


    def process_yang_odors(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # TODO also use that leave_concs_in_mix flag? (which currently doesn't support
        # leaving concs in components anyway...)
        without_concs = strip_concs(df.odor)

        nonvalue_cols = None
        if without_concs.nunique() < df.odor.nunique():
            # TODO include more detailed info, similar to some stuff currently
            # output in load/preprocess yang data fns?
            warn('averaging over multiple concs for some odors in KC data! see load/'
                'preprocess output above for more details.'
            )
            nonvalue_cols = [x for x in df.columns if x != 'value']
            assert not df[nonvalue_cols].duplicated().any()

        df['odor'] = without_concs
        # TODO assert not None?
        if nonvalue_cols is not None:
            assert df[nonvalue_cols].duplicated().any()
            df = df.groupby(nonvalue_cols, sort=False).value.mean(
                ).reset_index()

        # necessary for plots to still have odors in order i want
        df = df.sort_values(by='odor', kind='stable', key=odor_sort_fn)
        # TODO delete? shouldn't matter
        df = df.reset_index(drop=True)
        #
        return df


    # TODO factor to module level? (would then also have to pass in dataframes from
    # load_* fn)
    def get_yang_panel_means(panel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        flyroi_ser = get_yang_panel_ser(panel)
        assert isinstance(flyroi_ser, pd.Series)

        # averaging over ROIs, within each (fly, odor)
        group_cols = ['panel', 'odor'] + fly_cols
        assert all(x in flyroi_ser.index.names for x in group_cols)
        # this will be the case iff load_yang_kc_data had drop_exp_type=False arg
        if 'exp_type' in flyroi_ser.index.names:
            group_cols += ['exp_type']

        # could define group_cols in one step like this, but might want to have the
        # columns listed above in the earlier order positions that should enforce
        group_cols += [x for x in flyroi_ser.index.names
            if x not in group_cols and x != 'roi'
        ]
        # we are just intending to average over ROIs, within unique combos of all other
        # index variables
        assert set(flyroi_ser.index.names) - {'roi'} == set(group_cols)

        # we have already computed mean (of per-trial, within-window Fc_zscore mean
        # from 2s after onset of a 2s odor pulse to 8s after onset (same as Remy's
        # maybe? / similar?)
        # TODO why did yang not start averaging within odor pulse? really that slow?
        flyavg_ser = flyroi_ser.groupby(group_cols).mean().rename(resp_col)
        assert isinstance(flyavg_ser, pd.Series)

        flyroi_bin_ser = get_yang_panel_ser(panel, yang_bin_df)
        # (delete) does yang's analysis pipeline already (effectively?) exclude or
        # not-pull-out non-responding ROIs? some estimate of what fraction of ROIs fall
        # in that category if so? (i.e. how many double counts does she already have, if
        # she doesn't attempt to resolve them. then denominator should just be total #
        # expected KCs). she believes analysis should pull out non-responders, there
        # should be no bias favoring responders (in double counting or identification),
        # and that there has been no similar filtering at this point.
        flyavg_bin_ser = flyroi_bin_ser.groupby(group_cols).mean().rename(
            'mean_response_rate'
        )
        assert isinstance(flyavg_bin_ser, pd.Series)
        assert flyavg_ser.index.equals(flyavg_bin_ser.index)

        flyavg_df = pd.concat([flyavg_ser, flyavg_bin_ser], axis='columns',
            verify_integrity=True
        )
        assert flyavg_df.index.equals(flyavg_ser.index)
        # now the existing index will be duplicated, once for each existing column name
        # (which will be the values of the new 'stat' column, with the values in new
        # 'value' columns, similar to model data in `df` outside)
        flyavg_df = flyavg_df.melt(ignore_index=False, var_name='stat')
        flyavg_df = flyavg_df.reset_index()

        # TODO TODO factor out odor processing, to share w/ preprocessing for mix supp
        # calc too? (and check whether i can do it before above processing, to same
        # effect? if so, i could move it into get_yang_panel_ser)
        # (prob can't do in get_yang_panel_ser, b/c i think some of the code above is
        # currently changing order of odors, so need to at least sort them after? or
        # would at least need to change code above to preserve order, if possible)
        flyavg_df = process_yang_odors(flyavg_df)

        # TODO normalize yang's stuff outside here, so we can do it across panels?
        # even want to do it across panels? i don't think so...
        return flyavg_df


    # TODO move to module level?
    def mix_supp_list2flystat_df(mix_supp_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        mix_supp = pd.concat(mix_supp_dfs, verify_integrity=True).reset_index()
        diff_col = get_diff_col(mix_supp)

        assert 'stat' not in mix_supp.columns
        # TODO problem that it isn't previxed w/ 'mean_'
        # (will probably have to change handling of model ones anyway, for same reason,
        # if theres an issue)
        mix_supp['stat'] = 'Fc_zscore'

        assert 'value' not in mix_supp.columns
        # TODO delete
        #return mix_supp.rename(columns={diff_col: 'value'})
        return mix_supp

    MIX_SUPP_IN_RESPONDERS_ONLY: bool = True

    # TODO also do something with the other panel in yang_dfs? (natmix-top2-dilute)
    # (i think those concs are too low to be able to compare to any orn data we really
    # have currently... since the ramp experiments i did were not of good quality)
    # TODO could at least lump the data for that one mix that overlaps between that
    # panel and diag-binaries into diag-binaries (would probably want to do when
    # initially loading yang data, and maybe also exclude from natmix-top2-dilute at the
    # same time)
    # TODO rename flyodor->flyavg_odor?
    # TODO TODO might i also want to store fly odor stats that are also per ROI?
    # TODO TODO TODO want to do any preprocess for that (storing stuff behind a new
    # panel2kc_flyroi_odor_stats, or just calc directly from yang_df/mdf?)
    panel2kc_flyodor_stats: Dict[str, pd.DataFrame] = dict()
    # TODO rename fly->flyroi?
    panel2kc_fly_stats: Dict[str, pd.DataFrame] = dict()
    # stats like max intensity for each (ROI, odor), to make e.g. distributions of odor
    # intensity for mixes vs components
    panel2kc_flyroi_odor_stats: Dict[str, pd.DataFrame] = dict()
    for panel in ('diag-binaries',):
        yang_flyavg_df = get_yang_panel_means(panel)
        panel2kc_flyodor_stats[panel] = yang_flyavg_df

        flyroi_ser = get_yang_panel_ser(panel)
        # TODO make sure this doesn't cause issues w/ calculating mix
        # suppression (it will make farn -2.5 and -3, both comps and in mixes, look the
        # same, potentially w/o ROIs having same meaning across them? see _check_farn
        # code)
        # TODO maybe drop to only farn -3 instead? seems all flies should have that
        # (from _check_farn code) (compare output from current method to dropping farn
        # -2.5)
        flyroi_df = process_yang_odors(flyroi_ser.rename('value').reset_index())

        # TODO TODO TODO is this consistent w/ how i'll handle model diag mixes tho
        # (prob not?)? also computed across all components/diag mixes, or just within
        # each?
        # TODO TODO TODO instead, store responder mask alongside (in?) whatever i'll use
        # to calculate mix_suppression, and then calculate which "non-responders" to
        # drop down there?
        panel_responded = yang_bin_df.loc[panel].dropna(how='all', axis='columns')
        panel_responders = panel_responded.T.any()
        panel_responders = addlevel(panel_responders, 'panel', panel)

        panel_responded.columns = strip_concs(panel_responded.columns)
        # TODO need to groupby farn / farn-mixes and take union of whether it
        # responded to any now? i assume so. doing it now
        # TODO change mix_supp_list2... below to not care if 'stat' already in df, and
        # then don't copy here?
        panel_responded = panel_responded.stack()
        assert not panel_responded.isna().any()
        assert set(panel_responded.unique()) == {0, 1}
        panel_responded = panel_responded.astype(bool)
        # seems it's already sorted anyway
        panel_responded = panel_responded.groupby(level=panel_responded.index.names,
            sort=False).any()

        panel_responded = panel_responded.rename('responded')

        flyroi_odor_stats = flyroi_df.copy()
        flyroi_odor_stats['stat'] = resp_col

        flyroi_odor_stats = flyroi_odor_stats.set_index(panel_responded.index.names)
        flyroi_odor_stats['responded'] = panel_responded.loc[flyroi_odor_stats.index]
        flyroi_odor_stats = flyroi_odor_stats.reset_index()
        panel2kc_flyroi_odor_stats[panel] = flyroi_odor_stats

        # TODO TODO does a responder rate this low make sense???
        # ipdb> yang_bin_df.loc[panel].dropna(how='all', axis='columns').T.any().sum()
        # 4851
        # ipdb> yang_bin_df.loc[panel].dropna(how='all', axis='columns').T.any(
        #   ).sum() / 20442
        # 0.2373055474024068

        mixes = [x for x in flyroi_df.odor.unique() if '+' in x]
        # ['2h+farn', '2h+ma', 'farn+ma']
        mix_supp_dfs = []
        for mix in mixes:
            c1, c2 = mix.split('+')
            mix_df = flyroi_df[flyroi_df.odor.isin((c1, c2, mix))]
            assert set(mix_df.odor.unique()) == {c1, c2, mix}

            mix_df = mix_df.set_index([x for x in mix_df.columns if x != 'value'],
                verify_integrity=True).squeeze().unstack('odor')
            assert not mix_df.isna().any().any()

            mix_df = mix_df.sort_index(level='odor', key=odor_sort_fn, axis='columns',
                kind='stable').T

            assert panel_responders.index.equals(mix_df.columns)
            # TODO TODO TODO should we be recalculating responders within just this mix
            # and components, to be consistent w/ model handling below? or change that
            # to match this
            if MIX_SUPP_IN_RESPONDERS_ONLY:
                mix_df = mix_df.loc[:, panel_responders]

            # TODO TODO also group into response classes here, before calculating
            # mix suppression? (changing that code to work w/ 2 components in process)

            mix_supp = calc_mix_suppression(mix_df)
            mix_supp['mix'] = mix
            mix_supp = mix_supp.set_index('mix', append=True)
            mix_supp_dfs.append(mix_supp)

        mix_supp = mix_supp_list2flystat_df(mix_supp_dfs)
        panel2kc_fly_stats[panel] = mix_supp

    # contents:
    # remy_kiwi-control_5comp_Fc_zscore.csv
    # remy_kiwi-control_5comp_Fc_zscore.parquet
    # remy_kiwi-control_5comp_fly2n-total-rois.csv
    # remy_kiwi-control_5comp_fly2n-total-rois.parquet
    # remy_kiwi-control_binary_Fc_zscore.csv
    # remy_kiwi-control_binary_Fc_zscore.parquet
    # remy_kiwi-control_binary_fly2n-total-rois.parquet
    # remy_kiwi-control_binary_fly2n-total-rois.csv
    remy_dir = data_root / 'internal/remy_natmix_kcs'
    assert remy_dir.is_dir()

    # TODO factor out this loading?
    prefix = 'remy_kiwi-control_'
    response_suffix = '_Fc_zscore.parquet'
    fly2n_total_rois_suffix = '_fly2n-total-rois.parquet'
    # TODO TODO also load fly2n_total_rois to help calculate total response rates
    # (matter? are those actually mostly non-responders?)
    # TODO TODO use fns from natmix_data/analysis.py to help w/ that? refactor them
    # around? (have moved most to mb_model by now)
    mdf = read_parquet(remy_dir / f'{prefix}5comp{response_suffix}')
    # TODO TODO (delete? done?) also use this as denominator for sparsity (and at least
    # do that for the binary mix one too, if not for any response class stuff)
    fly2n_total_rois_5comp = read_parquet(
        remy_dir / f'{prefix}5comp{fly2n_total_rois_suffix}'
    )
    mdf = drop_mix_dilutions(mdf)

    # TODO TODO sanity check this isn't all the same exact data for ea/eb (or is it
    # pinned to that mix or something? why spread so small on activation strength for
    # that particular [top] concentration step?)
    bdf = read_parquet(remy_dir / f'{prefix}binary{response_suffix}')
    fly2n_total_rois_binary = read_parquet(
        remy_dir / f'{prefix}binary{fly2n_total_rois_suffix}'
    )

    def preprocess_natmix_kc_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename_axis(index={'odor1': 'odor'}).rename_axis(
            columns={'cells': 'roi'}
        )

        dfs = []
        for panel in df.index.get_level_values('panel').unique():
            pdf = addlevel(df.loc[panel].dropna(how='all', axis='columns'), 'panel',
                panel, axis='columns'
            )
            dfs.append(pdf)

        # TODO could assert # NaN + # non-NaN hasn't changed (seems that nothing changed
        # except moving outer level of row index to outer level of col index though,
        # from inspection)
        return pd.concat(dfs, verify_integrity=True)

    # TODO better names for these than mdf/bdf
    mdf = preprocess_natmix_kc_df(mdf)
    bdf = preprocess_natmix_kc_df(bdf)

    # both should be {'control', kiwi'}
    natmix_panels = set(mdf.columns.get_level_values('panel').unique())
    assert natmix_panels == set(bdf.columns.get_level_values('panel').unique())

    # TODO refactor to share processing of odor names that has happened to
    # natmix_df contents (that's precomputed, right? what computed it?)
    # (eh, whatever. i've already done what i need here)
    mdf = mdf.rename({'cmix @ 0': 'cmix0 @ 0', 'kmix @ 0': 'kmix0 @ 0'}, level='odor')

    # only the highest concs (the concs that are also the components in the 5-component
    # mixtures) of the binary ramp experiment should be in both natmix_df and bdf
    binary_comps = natmix_df.columns[
        natmix_df.columns.get_level_values('odor').isin(bdf.index)
    ]
    binary_comps_and_mixes = []
    # get list of only the top-concentration binary mixture and its components, to
    # subset bdf to that
    for panel, odf in binary_comps.to_frame(index=False).groupby('panel'):
        panel_comps_and_mix = []
        assert len(odf) == 2, 'expected 2 top-components per panel'
        assert odf.odor.nunique() == 2
        sorted_comps = sorted(odf.odor)
        panel_comps_and_mix.extend(sorted_comps)

        mix_str = component_delim.join(sorted_comps)
        assert mix_str in bdf.index, f'{mix_str=} not in index:\n{bdf.index}'
        panel_comps_and_mix.append(mix_str)

        mix_str_rev = component_delim.join(sorted_comps[::-1])
        assert mix_str_rev not in bdf.index, (f'did not expect {mix_str_rev=} in index:'
            f'\n{bdf.index}'
        )
        binary_comps_and_mixes.extend(panel_comps_and_mix)

    bdf = bdf.loc[binary_comps_and_mixes]

    # TODO TODO group ROIs into response classes (both here and for model cells),
    # before computing mix suppression, so i can also plot average mix suppression for
    # each [alongside fraction of overall population]?
    # (and also do for diag-binaries above)

    # TODO refactor to share across the two (binary/5comp) cases (and w/ diag stuff?)?
    flyroi_binary_ser = bdf.T.stack().rename(resp_col)
    assert isinstance(flyroi_binary_ser, pd.Series)
    assert not flyroi_binary_ser.isna().any()
    assert len(flyroi_binary_ser) == bdf.notna().sum().sum()
    flyroi_binary_ser = addlevel(flyroi_binary_ser, 'mix', 'binary')
    #
    flyroi_5comp_ser = mdf.T.stack().rename(resp_col)
    assert isinstance(flyroi_5comp_ser, pd.Series)
    assert not flyroi_5comp_ser.isna().any()
    assert len(flyroi_5comp_ser) == mdf.notna().sum().sum()
    # TODO try both averaging over mix level and plotting each separately, when
    # comparing to model stuff below (components should overlap)
    flyroi_5comp_ser = addlevel(flyroi_5comp_ser, 'mix', '5comp')

    natmix_stat_ser = pd.concat([flyroi_5comp_ser, flyroi_binary_ser],
        verify_integrity=True
    )

    nonroi_levels = [x for x in natmix_stat_ser.index.names if x != 'roi']
    # TODO TODO pick a threshold that gives us same average response rate as model
    # data? (or at least say what that would be, and what average response rate is with
    # whatever threshold we choose, and what it is on the model data)?
    #
    # TODO TODO retune on just kiwi/control, to get higher response rate? do for
    # all panels? would prob also be fine to tune on kiwi/control separately?
    #
    # ipdb> df[df.panel.isin(natmix_panels) & (df.stat == 'mean_response_rate')
    #   ].value.mean()
    # 0.08937741802973936
    #
    # ipdb> df[df.panel.isin(natmix_panels) & (df.stat == 'mean_response_rate')
    #   ].groupby('source').value.mean()
    # source
    # prat-claws                   0.089080
    # prat-claws_connectome-APL    0.103431
    # uniform                      0.083425
    # wd20                         0.084480
    # wd20_connectome-APL          0.086472
    #
    # TODO TODO so maybe i should use higher threshold in natmix_data/analysis.py?
    # i think i use .8 there. yang uses 1.0, but also requires > (>=?) 50% of trials,
    # so probably more stringent than my approach
    # ipdb> (natmix_stat_ser >= 0.8).groupby('odor').mean().mean()
    # 0.1652843599867842
    # ipdb> (natmix_stat_ser >= 1.0).groupby('odor').mean().mean()
    # 0.13511062134858848
    # ipdb> (natmix_stat_ser >= 1.25).groupby('odor').mean().mean()
    # 0.10669566477411459
    # ipdb> (natmix_stat_ser >= 1.4).groupby('odor').mean().mean()
    # 0.09302150916533318
    # ipdb> (natmix_stat_ser >= 1.5).groupby('odor').mean().mean()
    # 0.08485341122613663
    #
    # TODO TODO include this in suptitle of plots that use it
    # TODO TODO include in fname too? plot for a few thresh values?
    NATMIX_KC_THRESH: float = 1.5
    # TODO also make some plots showing effect of this cutoff (like the ones in
    # natmix_data/analysis.py, where i send subthreshold values to 0 in a plot of
    # clustered responses)

    # TODO delete. doesn't include total # ROIs like below
    #natmix_flyavg_bin_ser = (natmix_stat_ser >= NATMIX_KC_THRESH).groupby(
    #    nonroi_levels).mean().rename('mean_response_rate')

    # otherwise could not index fly2n_total_roi data sources without panel (which they
    # don't have)
    assert len(natmix_stat_ser.index.to_frame(index=False)[['mix'] + fly_cols
        ].drop_duplicates()) == len(
        natmix_stat_ser.index.to_frame(index=False)[['mix', 'panel'] + fly_cols
        ].drop_duplicates()
    )

    nrois_5comp = addlevel(fly2n_total_rois_5comp, 'mix', '5comp')
    nrois_binary = addlevel(fly2n_total_rois_binary, 'mix', 'binary')
    natmix_nrois = pd.concat([nrois_5comp, nrois_binary], verify_integrity=True)
    natmix_n_responding = (natmix_stat_ser >= NATMIX_KC_THRESH).groupby(nonroi_levels
        ).sum()
    # TODO did i not need to reorder levels before and after this division? matter?
    natmix_flyavg_bin_ser = (natmix_n_responding / natmix_nrois).dropna()

    natmix_flyavg_ser = natmix_stat_ser.groupby(level=nonroi_levels).mean()

    natmix_flyavg_bin_ser = natmix_flyavg_bin_ser.reorder_levels(
        natmix_flyavg_ser.index.names
    ).sort_index().rename('mean_response_rate')
    natmix_flyavg_ser = natmix_flyavg_ser.sort_index()

    # TODO refactor to share w/ processing of yang data above?
    natmix_stat_df = pd.concat([natmix_flyavg_ser, natmix_flyavg_bin_ser],
        axis='columns', verify_integrity=True
    )
    # TODO TODO failing now (after adding n_total_rois) fix!
    assert natmix_stat_df.index.equals(natmix_flyavg_ser.index)
    # now the existing index will be duplicated, once for each existing column name
    # (which will be the values of the new 'stat' column, with the values in new
    # 'value' columns, similar to model data in `df` outside)
    natmix_stat_df = natmix_stat_df.melt(ignore_index=False, var_name='stat')
    natmix_stat_df = natmix_stat_df.reset_index()
    #

    # TODO also add fly_id to this at some point

    natmix_stat_df['odor'] = strip_concs(natmix_stat_df.odor)
    natmix_stat_df = natmix_stat_df.sort_values(by='odor', kind='stable',
        key=odor_sort_fn
    )
    natmix_stat_df = natmix_stat_df.reset_index(drop=True)

    for panel in natmix_panels:
        panel2kc_flyodor_stats[panel] = natmix_stat_df[natmix_stat_df.panel == panel]

        panel_mdf = mdf.loc[:, mdf.columns.get_level_values('panel') == panel
            ].dropna(how='all')
        assert not panel_mdf.isna().any().any()
        panel_mdf = addlevel(panel_mdf, 'mix', '5comp', axis='columns')

        panel_bdf = bdf.loc[:, bdf.columns.get_level_values('panel') == panel
            ].dropna(how='all')
        assert not panel_bdf.isna().any().any()
        panel_bdf = addlevel(panel_bdf, 'mix', 'binary', axis='columns')

        if MIX_SUPP_IN_RESPONDERS_ONLY:
            panel_mdf = panel_mdf.loc[:, (panel_mdf >= NATMIX_KC_THRESH).any()]
            panel_bdf = panel_bdf.loc[:, (panel_bdf >= NATMIX_KC_THRESH).any()]

        mix_supp_5comp = calc_mix_suppression(panel_mdf)
        mix_supp_binary = calc_mix_suppression(panel_bdf)

        flyroi_odor_stats = pd.concat(
            [x.T.stack('odor').rename(resp_col) for x in [panel_mdf, panel_bdf]],
            verify_integrity=True
        ).reset_index()

        flyroi_odor_stats.odor = strip_concs(flyroi_odor_stats.odor)
        flyroi_odor_stats = flyroi_odor_stats.sort_values(by='odor', kind='stable',
            key=odor_sort_fn
        )

        assert resp_col in flyroi_odor_stats.columns
        assert 'value' not in flyroi_odor_stats.columns
        flyroi_odor_stats = flyroi_odor_stats.rename(columns={resp_col: 'value'})
        assert 'stat' not in flyroi_odor_stats.columns
        flyroi_odor_stats['stat'] = resp_col

        flyroi_odor_stats['responded'] = flyroi_odor_stats.value >= NATMIX_KC_THRESH

        panel2kc_flyroi_odor_stats[panel] = flyroi_odor_stats

        panel_mix_supp = mix_supp_list2flystat_df([mix_supp_5comp, mix_supp_binary])
        # TODO rename panel2kc_mix_supp? (no other stats in here)
        panel2kc_fly_stats[panel] = panel_mix_supp

    # these should be the same across all model panels
    unique_stats = set(df.stat.unique())

    suffix = ''
    for panel in panel2kc_flyodor_stats:
        kc_df = panel2kc_flyodor_stats[panel].copy()
        # TODO assert only one panel in kc_df? not sure i care? (and might even want to
        # pass both kiwi/control in that case, to normalize across both?) prob wouldn't
        # want to compute same thing twice tho...

        unique_kc_stats = set(kc_df.stat.unique())
        shared_stats = unique_stats & unique_kc_stats
        assert all(
            k in unique_stats and k not in shared_stats
            for k in compare_normalized.keys()
        )
        assert all(
            v in unique_kc_stats and v not in shared_stats
            for v in compare_normalized.values()
        )
        # TODO (delete. should always have mean_response_rate now) how to handle when
        # mean_response_rate is NOT in both (and thus not in shared_stats here), as
        # currently for natmix data? just fix that by thresholding it to get a binarized
        # version (have done that now)?
        expected_model_stats = set(compare_normalized.keys()) | shared_stats
        assert df.stat.isin(expected_model_stats).all(), \
            f'{df.stat.unique()=} {expected_model_stats=}'

        expected_kc_stats = set(compare_normalized.values()) | shared_stats
        assert kc_df.stat.isin(expected_kc_stats).all(), \
            f'{kc_df.stat.unique()=} {expected_kc_stats=}'

        kc_df['source'] = 'KCs'

        normed_dfs = []
        # TODO TODO add flag to also add normalized versions for any w/ name matching
        # exactly? (i.e. mean_response_rate)?
        # TODO TODO TODO also try retuning on her odors, with her mean response rate
        # (and how much do fixed_thr and wAPLKC_scale differ between that and tuning on
        # megamat [and for each model, including bouton versions]?)
        for x in compare_normalized.values():
            raw_df = kc_df[kc_df.stat == x]

            # TODO need to handle negative values?
            vmin = raw_df.value.min()
            if vmin < 0:
                warn(f'{vmin=} < 0 for KC data in {panel=}. issue for normalization?')

            normed_df = raw_df.copy()
            # TODO also try subtracting min? meh
            # TODO TODO try NORM_PER_FLY again?
            if not NORM_PER_FLY:
                normed_df['value'] = normed_df.value / normed_df.value.max()
            else:
                # if i decide to include stuff defined across odors in these KC
                # dfs (i.e. mix suppression), may need to change handling here
                # (probably will handle that stuff separately though)
                #
                # oh, so all flies do have all odors actually? at least, before any
                # attempt to lump flies from the other dataset (with one of these 3
                # mixtures, and its components) in with this data
                assert (
                    len(raw_df.groupby(fly_cols).odor.unique().map(tuple).unique()) == 1
                ), 'not all flies had all odors'

                cols_before = list(normed_df.columns)
                normed_df = normed_df.set_index(fly_cols + ['odor'],
                    verify_integrity=True
                )
                normed_df['value'] /= normed_df.groupby(fly_cols).value.max()
                normed_df = normed_df.reset_index()[cols_before].copy()
                assert np.allclose(normed_df.groupby(fly_cols).value.max(), 1)

            if NORM_TO_FLYMEAN_MAX:
                normed_df['value'] /= normed_df.groupby('odor').value.mean().max()

            normed_df['stat'] = f'{NORM_PREFIX}{x}'
            normed_dfs.append(normed_df)

        kc_df = pd.concat([kc_df] + normed_dfs, ignore_index=True)
        panel2kc_flyodor_stats[panel] = kc_df

    model_response_strengths = model_roi_odor_df.pivot(columns='stat', values='value',
        index=[x for x in model_roi_odor_df.columns if x not in ('stat','value')]
    )
    assert not model_response_strengths.isna().any().any()
    assert set(model_response_strengths['responded'].unique()) == {0, 1}
    model_response_strengths['responded'] = model_response_strengths.responded.astype(
        bool
    )
    # TODO rename all of above too, to avoid confusion (or rename var w/ same name in
    # loop below)
    all_model_response_strengths = model_response_strengths.reset_index()

    natmix_panel_class_frac_list = []

    comps_to_drop = [
        'fur', 'ms', 'va', 'EtOH', 'IAol', 'IaA'
    ]
    # TODO TODO is the change from 2026-05-10 outputs to those on 2026-05-20 just
    # the change in tuning convergence? or what else? is it only the wd20 and prat-claws
    # stuff moving? ignore LR cache and regen?
    # NOTE: seems like it might *just* be the wd20 case moving around (neither uniform
    # nor prat-claws [nor either APL case for latter] seem to have changed)
    # TODO TODO if so, use smaller sp_acc for everything, to minimize tuning
    # related noise? otherwise, what is the difference?
    for panel in panels:
        pdf = df[df.panel == panel].copy()

        kc_panel = None
        if panel.startswith('diag-binaries_'):
            kc_panel = 'diag-binaries'

        elif panel in panel2kc_flyodor_stats:
            kc_panel = panel

        if kc_panel is not None:
            assert kc_panel in panel2kc_flyodor_stats
            assert kc_panel in panel2kc_fly_stats
            assert kc_panel in panel2kc_flyroi_odor_stats
            # TODO and assert they are all not-None?
            for i, x in enumerate([
                    panel2kc_fly_stats, panel2kc_fly_stats, panel2kc_flyroi_odor_stats
                ]):
                assert panel2kc_fly_stats[kc_panel] is not None, \
                    f'{panel=} {kc_panel=} {i=}'
        else:
            assert panel not in panel2kc_flyodor_stats
            assert panel not in panel2kc_fly_stats
            assert panel not in panel2kc_flyroi_odor_stats

        for_mix_supp = model_roi_odor_df[
            (model_roi_odor_df.panel == panel) & (model_roi_odor_df.stat != 'responded')
        ].pivot(
            index='odor', values='value',
            # TODO delete
            # TODO assert these are all remaining columns?
            #columns=['panel', 'stat', 'source', 'model_pnkc_class', 'connectome_apl',
            #    'roi'
            #]
            columns=['source', 'connectome_apl', 'model_pnkc_class', 'roi', 'stat']
        )
        # TODO delete? refactor (do before pivot?)?
        responded = model_roi_odor_df[
            (model_roi_odor_df.panel == panel) & (model_roi_odor_df.stat == 'responded')
        ].pivot(
            index='odor', values='value',
            # TODO delete?
            # TODO assert these are all remaining columns?
            #columns=['panel', 'stat', 'source', 'model_pnkc_class', 'connectome_apl',
            #    'roi'
            #]
            columns=['source', 'connectome_apl', 'model_pnkc_class', 'roi']
        )

        # TODO put in title/fname too
        if MIX_SUPP_IN_RESPONDERS_ONLY:
            # TODO TODO calculate w/in each mix instead, rather than across panel?
            # (or whatever, as long as it's consistent across KC / model data)
            # (test in particular for diag binaries?)
            # TODO TODO is this responded based dropping same as what i would get w/ my
            # current WIP approach for response strengths?
            for_mix_supp = for_mix_supp.loc[:, responded.any()]

        # TODO sort ROIs into response classes first? (maybe after grouping by mix, esp
        # in diag-binaries case?)
        model_mix_supp = None
        model_mix_resps = None
        if panel in natmix_panels or kc_panel == 'diag-binaries':
            binary_mix_mask = for_mix_supp.index.str.contains('\+')

            for_binary_list = []
            for binary_mix in for_mix_supp.index[binary_mix_mask]:
                ca, cb = binary_mix.split('+')
                assert all((for_mix_supp.index == x).sum() == 1 for x in (ca, cb))

                binary_and_comp_mask = for_mix_supp.index.isin((ca, cb, binary_mix))
                for_binary = for_mix_supp[binary_and_comp_mask]

                # TODO TODO also do this step here? or just count all that responded
                # anywhere in panel?
                # TODO TODO TODO also have to change KC dropping now, to be consistent
                # w/ this? check!
                if MIX_SUPP_IN_RESPONDERS_ONLY:
                    for_binary = for_binary.loc[:,
                        responded[binary_and_comp_mask].any()
                    ]

                if kc_panel == 'diag-binaries':
                    for_binary = addlevel(for_binary, 'mix', binary_mix, axis='columns')

                for_binary_list.append(for_binary)

            if panel in natmix_panels:
                assert len(for_binary_list) == 1
                for_mix_supp_binary = for_binary_list[0]
                assert 'mix' not in for_mix_supp_binary.columns.names
                # TODO use name of mix instead? (would also have to change KC handling)
                for_mix_supp_binary = addlevel(for_mix_supp_binary, 'mix', 'binary',
                    axis='columns'
                )
                mix_supp_binary = calc_mix_suppression(for_mix_supp_binary)
            else:
                assert len(for_binary_list) > 1
                mix_supp_binary = pd.concat(
                    [calc_mix_suppression(x) for x in for_binary_list],
                    verify_integrity=True
                )

            full_mix_mask = for_mix_supp.index.str.contains('mix')
            if kc_panel == 'diag-binaries':
                assert full_mix_mask.sum() == 0
                model_mix_supp = mix_supp_binary
            else:
                assert full_mix_mask.sum() == 1
                full_mix = for_mix_supp.index[full_mix_mask][0]

                for_mix_supp_5comp = for_mix_supp[~binary_mix_mask]

                # TODO delete? or keep calc up here? check against below?
                # can i replace w/ part of for_mix_supp_5comp? (would have to change
                # indexing so that i drop all nonresponding (KC, odor) pairs, which
                # isn't that easy in current format)
                # TODO TODO also implement this for binary mixture case?
                # TODO TODO TODO where is the kiwi data. why just control?
                panel_response_strengths = all_model_response_strengths[
                    (all_model_response_strengths.panel == panel) &
                    ~all_model_response_strengths.odor.str.contains('\+')
                ]
                # TODO put behind flag?
                # TODO leave the responded column to be able to make this decision
                # later?
                panel_response_strengths = panel_response_strengths[
                    panel_response_strengths.responded
                ]
                # TODO TODO also do this step here? or just count all that responded
                # anywhere in panel?
                # TODO TODO TODO also have to change KC dropping now, to be consistent
                # w/ this? check!
                if MIX_SUPP_IN_RESPONDERS_ONLY:
                    for_mix_supp_5comp = for_mix_supp_5comp.loc[:,
                        responded[~binary_mix_mask].any()
                    ]
                    # TODO delete?
                    l1 = len(panel_response_strengths[
                        # adding 'odor' is all it takes for length to be same as that
                        # of the full dataframe
                        ['source','connectome_apl','model_pnkc_class','roi']
                    ].drop_duplicates())
                    l2 = int(len(for_mix_supp_5comp.columns) / 2)
                    assert l1 == l2, f'{l1=} {l2=}'
                    #

                # NOTE: the below also changes l1 above (unique combos of those columns
                # that dont include odor... make sense?)
                # TODO TODO note that top component differs depending on metric. going
                # to just hardcode which component i want to compare to for now
                # (1o3ol for control, and probably eb for kiwi. need to check there)
                #
                # ipdb> panel_response_strengths.groupby('odor')[[
                #   'logistic_scaled_num_spikes', 'num_spikes']].mean()
                # stat   logistic_scaled_num_spikes  num_spikes
                # odor
                # 1o3ol                    0.815560    1.579604
                # 2h                       0.751807    1.574113
                # cmix0                    0.587520    1.209181
                # fur                      0.721496    1.465144
                # ms                       0.872448    1.795571
                # va                       1.006529    2.084291
                panel2top_component = {
                    'control': '1o3ol',
                    # ea is actually the highest on average:
                    # TODO maybe i should go back whatever is highest in real KCs tho?
                    # unless it's iaa...
                    # ipdb> panel_response_strengths.groupby('odor'
                    # )[['logistic_scaled_num_spikes','num_spikes']].mean()
                    # stat   logistic_scaled_num_spikes  num_spikes
                    # odor
                    # EtOH                     0.846472    1.547826
                    # IAol                     0.887554    1.754902
                    # IaA                      1.148865    3.132588
                    # ea                       1.637313    7.843023
                    # eb                       1.152635    3.088710
                    # kmix0                    0.785062    1.594416
                    #
                    'kiwi': 'ea',
                }
                top_component = panel2top_component[panel]
                panel_response_strengths = panel_response_strengths[
                    # TODO assert both present alone?
                    panel_response_strengths.odor.isin((top_component, full_mix))
                ]

                # TODO use name of mix instead?
                for_mix_supp_5comp = addlevel(for_mix_supp_5comp, 'mix', '5comp',
                    axis='columns'
                )
                mix_supp_5comp = calc_mix_suppression(for_mix_supp_5comp)

                model_mix_supp = pd.concat([mix_supp_binary, mix_supp_5comp],
                    verify_integrity=True
                )

                panel_response_strengths['mix'] = '5comp'
                model_mix_resps = panel_response_strengths.copy()

            model_mix_supp = model_mix_supp.reset_index()

        # TODO TODO fit gaussian to non-responders in yang's labelled non-responders, as
        # well as in my thresholded Remy data (for a few thresholds?)
        # TODO eventually incorporate as preprocessing step in fn that fits scaling fn
        # to match spike count distribution to real data?

        if kc_panel is not None:
            # NOTE: don't need to rename panel in this dataframe (no matter if it's diff
            # from current `panel` in loop, because plotting fn does not actually use
            # panel in input data, just the input panel str for naming stuff)
            # TODO rename flyodor_stats?
            kc_df = panel2kc_flyodor_stats[kc_panel]

            s1 = set(pdf.odor.unique())
            s2 = set(kc_df.odor.unique())
            # TODO maybe for some other panels (besides diag-binaries, where there is an
            # exact match) i will just want to check s2 is a subset of s1?
            # TODO (delete) why does pdf have a mix of stripped concs and not now?
            # (was probably a bug that was since fixed. was only for one directory that
            # i deleted)
            # > s1
            # {'2h-6 + farn-2', 'ma-7 + farn-2', 'ma-7 + 2h-6', 'farn', 'ma', '2h+ma',
            # 'farn+ma', '2h', '2h+farn'}
            # > s2
            # {'farn', 'ma', '2h+ma', 'farn+ma', '2h', '2h+farn'}
            assert s1 == s2, f'{panel=} {s1 - s2=}'

            # TODO make sure odors are sorted in same order, so that we don't
            # screw up xticklabel order for plots that include KC data (and will that
            # fix it? how did i sort odors for model stuff again? do here or when
            # loading yang data?)
            # (they should be now, but assert that here?)

            fly_stats = panel2kc_fly_stats[kc_panel]
            assert model_mix_supp is not None, ('model mix suppression not calculated '
                f'for {panel=} {kc_panel=}'
            )
            mix_supp = pd.concat([model_mix_supp, fly_stats], ignore_index=True)

            # TODO set these in code populating panel2kc_fly_stats above? any other
            # panel2* data currently missing this for KCs? add both even earlier in KC
            # data?
            mix_supp[source_col] = mix_supp[source_col].fillna('KCs')
            mix_supp['model_pnkc_class'] = mix_supp['model_pnkc_class'].fillna('KCs')
            mix_supp.connectome_apl = mix_supp.connectome_apl.fillna(False)
            #

            facet_kws = dict()
            if 'mix' in mix_supp.columns and mix_supp.mix.nunique() > 1:
                facet_kws = dict(row='mix')

            # TODO delete. should be filled earlier now
            # TODO TODO TODO well it's not, and neither is 'source' (why???)
            #mix_supp.model_pnkc_class = mix_supp.model_pnkc_class.fillna('KCs')

            diff_col = get_diff_col(mix_supp)

            from_kcs = mix_supp.source == 'KCs'
            kc_mix_supp = mix_supp[from_kcs]
            model_mix_supp = mix_supp[~from_kcs]

            # TODO in each facet title, say how many nonresponders were dropped
            # (or in suptitle/legend for KCs?) would probably need a CSV for models...

            model_stat_order = ['logistic_scaled_num_spikes', 'num_spikes']

            # TODO delete? trying a version where we plot KC data on all facet, just
            # according to 'mix'
            #g = sns.FacetGrid(data=mix_supp, col='stat', hue='model_pnkc_class',
            # TODO also want to try looking at these only w/in mix responding KCs?
            g = sns.FacetGrid(data=model_mix_supp, col='stat',
                col_order=model_stat_order, hue='model_pnkc_class',
                # TODO restore sharey=False? just want to sanity check KC data is
                # aligned correctly. actually, i think it makes the plot easier to read
                # w/ it =True.
                palette=source_palette, sharey=True, **facet_kws
            )

            # TODO TODO TODO should everything be z-scored before computing mix - max?
            # or how to make more comparable (at least, in terms of the expected offset
            # from 0)? i suppose even just making logistic ceiling higher would do that?
            # TODO TODO TODO at least plot input distributions for each of this (hist
            # for spike counts) and maybe kde/hist for logistic scaled spike counts
            def plot_one_dist_per_model(data, *args, **kwargs):
                # make below conditional if this fails
                assert 'connectome_apl' in data.columns

                assert not data.mix.isna().any()
                mix = data.mix.unique()
                assert len(mix) == 1
                mix = mix[0]
                kdf = kc_mix_supp[kc_mix_supp.mix == mix]
                assert len(kdf) > 0
                sns.kdeplot(data=kdf, color=kc_color, label='KCs',
                    **{k: v for k, v in kwargs.items() if k not in ('color', 'label')}
                )

                # some connectome_apl=False comes second, which is hopefully what makes
                # it into legend (so i don't have the dotted lines there, just the hue)
                data = data.sort_values(by='connectome_apl', ascending=False)

                for gn, gdf in data.groupby(['source','connectome_apl'], sort=False):
                    source, connectome_apl = gn
                    # TODO TODO fix legend so it shows both linestyles (and uses that to
                    # show connectome vs uniform APL, w/o color, like i do w/ markers
                    # for other legend. refactor?)
                    linestyle = '--' if connectome_apl else '-'
                    # TODO only label on first one or something legend screwed up
                    # otherwise?
                    sns.kdeplot(data=gdf, *args, linewidth=model_linewidth,
                        linestyle=linestyle, **kwargs
                    )

            g.map_dataframe(plot_one_dist_per_model, x=diff_col, alpha=model_alpha,
                common_norm=False
            )

            # TODO try wider for some/all data?
            if kc_panel != 'diag-binaries':
                g.set(xlim=(-6, 4))
            else:
                g.set(xlim=(-6, 6))

            suptitle = f'{panel}\ndistribution of "mixture suppression" across KCs'
            if MIX_SUPP_IN_RESPONDERS_ONLY:
                suptitle += '\nsilent cells dropped (both model and real KCs)'
            else:
                suptitle += '\nall cells included'
            g.fig.suptitle(suptitle, y=1.10)
            # TODO TODO TODO also one plot of means of these?

            # TODO work?
            # TODO TODO try to change above plotting so solid lines get plotted second,
            # so that we don't have dashed lines in legend for nonclaw/claw
            add_fixed_legend(g, mix_supp, lines=True)

            g.set_titles('{col_name}')
            # TODO put odor (components + mix? just mix name?) in this actually (instead
            # of ylabel), and show for both rows?
            g.set_xlabels(diff_col2desc(diff_col))

            assert len(g.row_names) > 1
            assert g.axes.shape[0] == len(g.row_names)
            for (i, j, hue), gdf in g.facet_data():
                if hue != 0:
                    continue

                ax = g.axes[i, j]
                ylabel = ''
                if j == 0:
                    ylabel = f'{g.row_names[i]} mix\ndensity across KCs'

                if i != 0:
                    ax.set_title('')

                ax.set_ylabel(ylabel)

            savefig(g, plot_root, f'{diff_col}_dists_{panel}')

            # TODO TODO TODO compare mix response amplitude across KCs and models? or
            # how do i want to handle that? (already sufficiently captured between the
            # mean response amplitude + mean sparsity plot?)
            # TODO TODO TODO maybe just add distributions of response amplitude for each
            # odor, across all KCs? could have one hue (line) per odor, and one facet
            # per KCs and  for each model variant? or just have two linestyles (one for
            # all components, one for 5component mix, and then use hue for KCs vs model
            # variants? and normalize each KDE so component one doesn't swamp mix
            # responders)
            if model_mix_resps is None:
                warn('model_mix_resps not currently defined for binary mixture case!')
            else:
                flyroi_odor_stats = panel2kc_flyroi_odor_stats[kc_panel]

                kc_responded = flyroi_odor_stats[flyroi_odor_stats.mix == '5comp'
                    ].pivot(
                    columns=fly_cols + ['roi'], index='odor', values='responded'
                )
                kc_responded = kc_responded.sort_index(kind='stable', key=odor_sort_fn)

                n_filtered_kcs = kc_responded.columns.to_frame().groupby(level=fly_cols
                    ).size()
                assert n_filtered_kcs.sum() == len(kc_responded.columns)
                n_total_kcs = fly2n_total_rois_5comp.loc[n_filtered_kcs.index]
                assert (n_total_kcs > n_filtered_kcs).all()

                # TODO need to groupby fly? ifl prob not. compare w/ model below
                kc_class_sizes = summarize_response_classes(kc_responded)
                assert kc_class_sizes.sum() == len(kc_responded.columns)

                # no need to do this for model stuff, since that data has not had the
                # silent cells dropped here
                kc_class_sizes = add_missing_cells_to_nonresponders(kc_class_sizes,
                    n_total_kcs
                )
                assert kc_class_sizes.sum().equals(n_total_kcs)

                # just for current response class defining code to work (below)
                model_responded = responded.sort_index(kind='stable', key=odor_sort_fn)

                # TODO also calculate something like these w/ just binary mix and
                # components?
                # TODO can i use some other var defined with current binary mix already?
                bmix = model_responded.index[model_responded.index.str.contains('\+')]
                model_responded = model_responded.drop(index=bmix)
                assert model_responded.index.equals(kc_responded.index)
                model_responded = model_responded.astype(bool)

                # any point grouping by source first? yes, currently source column will
                # not be preserved otherwise. fly_cols seemed to be handled
                # automatically however.
                # TODO modify summarize_resposne_classes to automatically group on other
                # column index levels (and summarize for each), or take a kwarg to
                # specify which levels to treat like fly_cols?)
                # TODO TODO TODO are there any other places i'm mistakenly assuming
                # source still encodes APL connectome/uniform? was initially thinking
                # that here, and not including connectome_apl in group levels
                model_id_cols = ['source', 'connectome_apl', 'model_pnkc_class']
                model_class_sizes = model_responded.groupby(level=model_id_cols,
                    axis='columns', sort=False).apply(lambda x:
                    summarize_response_classes(x, verbose=False).to_frame()
                )
                assert model_class_sizes.columns.names == model_id_cols + [None]
                assert (
                    model_class_sizes.columns.get_level_values(-1) == 'n_rois'
                ).all()
                model_class_sizes = model_class_sizes.droplevel(level=-1,axis='columns')
                model_class_sizes = model_class_sizes.fillna(0)
                assert np.allclose(model_class_sizes, model_class_sizes.astype(int))
                model_class_sizes = model_class_sizes.astype(int)
                assert model_class_sizes.sum().equals(
                    model_responded.columns.to_frame().groupby(level=model_id_cols,
                        sort=False).size()
                )

                assert model_class_sizes.index.names == kc_class_sizes.index.names
                model_class_sizes = model_class_sizes.sort_index()
                assert kc_class_sizes.equals(kc_class_sizes.sort_index())
                if not model_class_sizes.index.equals(kc_class_sizes.index):
                    warn(f'{panel=} response classes only in either model or KC:\n'
                        f'{model_class_sizes.index.difference(kc_class_sizes.index)=}\n'
                        f'{kc_class_sizes.index.difference(model_class_sizes.index)=}'
                    )

                shared_index = kc_class_sizes.index.union(model_class_sizes.index)
                # TODO also check it has all consecutive categories? prob don't care
                # that much...
                assert shared_index.equals(shared_index.sort_values())

                kc_class_sizes = reindex(kc_class_sizes, shared_index, fill_value=0)
                model_class_sizes = reindex(model_class_sizes, shared_index,
                    fill_value=0
                )

                # TODO already code in natmix_data/analysis.py for this?  refactor to
                # share?
                group_cols = ['mix_resp', 'n_comps']
                def agg_within_mixresp_and_ncomps(df: pd.DataFrame) -> pd.Series:
                    assert df.index.names == group_cols + ['max_comp_idx']
                    ser = df.groupby(level=group_cols).sum().T.stack(group_cols)
                    ser = ser.rename('frac_response_class')
                    return ser

                kc_class_fracs = kc_class_sizes / kc_class_sizes.sum()
                assert np.isclose(kc_class_fracs.sum(), 1).all()
                assert kc_class_fracs.sum().index.names == fly_cols

                model_class_fracs = model_class_sizes / model_class_sizes.sum()

                kc_class_fracs = agg_within_mixresp_and_ncomps(kc_class_fracs)
                model_class_fracs = agg_within_mixresp_and_ncomps(model_class_fracs)
                # TODO expand both indices to full posibility of response classes (up to
                # max observed), and fill any missing values w/ 0. or do at end?
                panel_class_fracs = pd.concat(
                    [x.reset_index() for x in [kc_class_fracs, model_class_fracs]],
                    ignore_index=True
                )
                panel_class_fracs.connectome_apl = \
                    panel_class_fracs.connectome_apl.fillna(False)

                panel_class_fracs.source = panel_class_fracs.source.fillna('KCs')
                panel_class_fracs.model_pnkc_class = \
                    panel_class_fracs.model_pnkc_class.fillna('KCs')

                panel_class_fracs = panel_class_fracs.set_index(
                    model_id_cols + fly_cols + group_cols,
                    verify_integrity=True
                ).squeeze()
                panel_class_fracs = addlevel(panel_class_fracs, 'panel', panel)
                natmix_panel_class_frac_list.append(panel_class_fracs)


                flyroi_odor_stats = flyroi_odor_stats[
                    flyroi_odor_stats.responded &
                    # this is subsetting down to just full mix + (hardcoded) "top"
                    # component, consistent w/ what i'm currently doing on model data
                    # above
                    flyroi_odor_stats.odor.isin(model_mix_resps.odor.unique())
                ]
                flyroi_odor_stats = flyroi_odor_stats.drop(columns='responded')

                # TODO add flag to control this
                kc_min = flyroi_odor_stats.value.min()
                assert not np.isclose(kc_min, 0)
                flyroi_odor_stats.value -= kc_min
                assert np.isclose(flyroi_odor_stats.value.min(), 0)
                flyroi_odor_stats.stat = flyroi_odor_stats.stat.str.replace(
                    mean_prefix, ''
                )
                flyroi_odor_stats.stat += ' - threshold'

                # currently we are filtering to only have responding (KC, odor) pairs
                # above
                assert model_mix_resps.responded.all()
                model_mix_resps = model_mix_resps.drop(columns='responded')

                model_stats = [x for x in model_mix_resps.columns if 'num_spikes' in x]
                model_mix_resps = model_mix_resps.melt(
                    value_vars=model_stats, var_name='stat',
                    id_vars=[x for x in model_mix_resps.columns if x not in model_stats]
                )

                # TODO TODO also want to compare normalized/not, as like other
                # facetgrid? prob can't use same plotting fn, b/c do think i might want
                # displots of some kind here, prob kde

                # TODO delete this concatenation, if i'm just going to plot model vs KC
                # stuff separately below?
                response_strengths = pd.concat(
                    [flyroi_odor_stats, model_mix_resps], ignore_index=True
                )
                # TODO refactor to share w/ other filling
                response_strengths['connectome_apl'] = \
                    response_strengths.connectome_apl.fillna(False)
                response_strengths['source'] = response_strengths.source.fillna('KCs')
                response_strengths['model_pnkc_class'] = \
                    response_strengths.model_pnkc_class.fillna('KCs')
                #
                # TODO TODO TODO zscore these? or how to align? based on min/max?
                # (since we are limiting all KC data to responses, might make sense to
                # have min->0)

                # TODO rename either this or the var before the loop
                # (currently renaming one before loop at last second to all_*)
                model_response_strengths = response_strengths[
                    response_strengths.source != 'KCs'
                ]
                kc_response_strengths = response_strengths[
                    response_strengths.source == 'KCs'
                ]

                # TODO TODO flag to plot responders only vs not? (+ include in
                # fnames/etc)
                # TODO TODO want distribution for each odor, or just for top component
                # vs mix? (currently just picking a top component for each, and only
                # analyzing 5comp case at all)

                # TODO TODO col='mix', and loop over stat, making multiple plots w/ diff
                # suffixes? (for now i'm only analyzing the 5comp mixes tho)
                g = sns.FacetGrid(data=model_response_strengths, row='odor', col='stat',
                    col_order=model_stat_order, hue='model_pnkc_class',
                    # TODO try sharey=True?
                    # TODO TODO sharex=True (b wanted, i think) (at least w/in stat?)
                    palette=source_palette, sharey=False, sharex=False
                )

                # TODO TODO also plot raw distributions of response strengths for
                # each odor, one per facet (row=stat, col=odor, KCs on separate row)
                def plot_response_strength_dist_per_model(data, *args, **kwargs):
                    # make below conditional if this fails
                    assert 'connectome_apl' in data.columns

                    ax = plt.gca()
                    kc_ax = ax.twiny()
                    assert not data.odor.isna().any()
                    odor = data.odor.unique()
                    assert len(odor) == 1
                    odor = odor[0]
                    kdf = kc_response_strengths[kc_response_strengths.odor == odor]
                    # this will either be mean_Fc_zscore or 'mean_Fc_zscore - threshold'
                    kc_stats = kdf.stat.unique()
                    assert len(kc_stats) == 1
                    kc_stat = kc_stats[0]
                    assert len(kdf) > 0
                    sns.kdeplot(data=kdf, ax=kc_ax, color=kc_color, label='KCs', **{
                        k: v for k, v in kwargs.items() if k not in ('color', 'label')
                    })
                    kc_ax.set_xlabel(f'KC {kc_stat}', color=kc_color,
                        alpha=kc_err_alpha
                    )
                    # TODO refactor to share w/ twinx mb_model APL Vm plotting?
                    # alpha not supported here
                    kc_ax.tick_params(axis='x', color=kc_color)
                    for text in kc_ax.xaxis.get_ticklabels():
                        # alpha also not supported here
                        text.set_color(kc_color)
                    #

                    # TODO set color of top spine / ticks, like in APL Vm plotting
                    # elsewhere?
                    kc_ax.spines['right'].set_visible(False)
                    kmin = kdf.value.min()
                    kmax = kdf.value.max()
                    kc_ax.set_xlim([kmin - 0.5, kmax + 0.5])

                    # some connectome_apl=False comes second, which is hopefully what
                    # makes it into legend (so i don't have the dotted lines there, just
                    # the hue)
                    data = data.sort_values(by='connectome_apl', ascending=False)

                    for gn, gdf in data.groupby(['source','connectome_apl'],sort=False):
                        source, connectome_apl = gn
                        linestyle = '--' if connectome_apl else '-'
                        sns.kdeplot(data=gdf, ax=ax, *args, linewidth=model_linewidth,
                            linestyle=linestyle, **kwargs
                        )

                g.map_dataframe(plot_response_strength_dist_per_model, x='value',
                    # TODO why does cut=True not seem to have line clipped at 0 for
                    # min subtracted Fc_zscore??
                    alpha=model_alpha, common_norm=False, log_scale=(False, True),
                    cut=True
                )

                # TODO what do i want here? even want same across stats? prob not?
                # TODO have max depend on stat?
                #g.set(xlim=(-0.1, 6))
                # need anything slightly above 1? yes (should be <10 tho)
                g.set(ylim=(1e-5, 5))
                g.fig.subplots_adjust(hspace=0.6)

                suptitle = (f'{panel}\nactivation strengths across KCs\n'
                    'responder (KC, odor) pairs only'
                )
                # TODO change name of flag used here? have one to control it in both
                # cases, or two flags?
                # TODO TODO TODO + have whatever flag (if i have one here) control last
                # part of message above, not below (which was duped from other plot)
                # TODO delete
                #if MIX_SUPP_IN_RESPONDERS_ONLY:
                #    suptitle += '\nsilent cells dropped (both model and real KCs)'
                #else:
                #    suptitle += '\nall cells included'
                g.fig.suptitle(suptitle, y=1.10)

                # TODO work?
                add_fixed_legend(g, response_strengths, lines=True)

                g.set_titles('')

                assert len(g.row_names) > 1
                assert g.axes.shape[0] == len(g.row_names)
                for (i, j, hue), gdf in g.facet_data():
                    if hue != 0:
                        continue

                    ax = g.axes[i, j]
                    ylabel = ''
                    if j == 0:
                        ylabel = f'{g.row_names[i]}\ndensity across KCs'

                    if i != 0:
                        ax.set_title('')

                    ax.set_ylabel(ylabel)
                    ax.tick_params(labelbottom=True)
                    ax.set_xlabel(g.col_names[j])

                    mmin = 0
                    mmax = gdf.value.max()
                    ax.set_xlim([mmin - 0.5, mmax + 0.5])

                # TODO maybe save one version logscaled and one not?
                savefig(g, plot_root, f'response-strength_dists_{panel}')

            # TODO TODO TODO regen class mean/count + diagnostic plots for a few
            # particular model variants? (and use same threshold as here!, or at least
            # as one variant!)

            normed_dfs = []
            for x in compare_normalized.keys():
                raw_df = pdf[pdf.stat == x]

                # TODO need to handle negative values?
                vmin = raw_df.value.min()
                if vmin < 0:
                    warn(f'{vmin=} < 0 for model data in {panel=}. issue for '
                        'normalization?'
                    )
                #

                # TODO can i refactor to share this w/ above or not really?
                # TODO if i'm going to normalize per-fly above, should i normalize per
                # model here? ifl not... idk
                normed_df = raw_df.copy()
                # TODO also try subtracting min? meh
                normed_df['value'] = normed_df.value / normed_df.value.max()
                normed_df['stat'] = f'{NORM_PREFIX}{x}'
                normed_dfs.append(normed_df)

            pdf = pd.concat([pdf] + normed_dfs, ignore_index=True)

            pdf = pd.concat([pdf, kc_df], ignore_index=True)

            # TODO still want to (try?) normalizing mean_response_rate for each, even
            # though in theory those could be directly comparable?

        plot_panel_stats_across_models(pdf, panel, suffix)

        # TODO TODO also make a version with just 5component and not binary mixes?
        if panel in natmix_panels:
            pdf_nocomps = pdf[~pdf.odor.isin(comps_to_drop)]
            plot_panel_stats_across_models(pdf_nocomps, panel, f'{suffix}_nocomps')

    # contains both model and KC 5comp kiwi/control data
    class_fracs = pd.concat(natmix_panel_class_frac_list, verify_integrity=True)

    # TODO TODO second log-yscale version of this?
    plot_response_class_summary(class_fracs, plot_root, hue='model_pnkc_class',
        palette=source_palette, model_marker_kws=model_marker_kws, alpha=model_alpha,
        facet_kws=dict(height=4, aspect=1.2), jitter=0.3
    )

    plot_panel_stats_across_models(tdf, 'megamat', suffix)


if __name__ == '__main__':
    main()

