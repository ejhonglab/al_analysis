#!/usr/bin/env python3
"""Fetch Yang's APL16 KC data (APL>TNT and TNT3c control) from olfdb and compute RDMs.

Downloads olfdb (hong-data) datasets 19 (TNT: APL output blocked) and 20 (TNT3c:
control) as long CSVs plus the original per-recording files, then computes, per
recording (date, fly, hemisphere):
- trial-mean response of each KC to each odor
- fraction of KCs responding to each odor (trial-mean response > threshold)
- odor x odor RDMs over all KCs of the recording: 1 - Pearson r and 1 - cosine
  similarity, with and without the solvent control (pfo)

and their means over recordings and over flies, and compares the recording-mean 16-odor
RDMs against the two matrices the report analyzed (the `ctrl` / `tnt` columns of
apl_pair_analysis/all120_pairs.csv), to find which recipe reproduces them.

Everything is written under <shared>/data/yang_apl16_kc_olfdb/ (README.md there
describes each output). Nothing is re-downloaded if the long CSVs are already present.

Usage:
    ./venv/bin/python yang_apl16_kc_data.py [--shared-dir DIR] [--olfdb URL]
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent
DEFAULT_SHARED = Path('/local/tensor/Tom/claude_kc_remapping')
DEFAULT_OLFDB = 'http://hal:8080'

# olfdb dataset id -> genotype label used in Yang's files. TNT3c is the control with APL
# intact (TNT inactive); TNT has APL output blocked.
DATASETS = {19: 'TNT', 20: 'TNT3c'}
SOLVENT = 'pfo'
# as in Yang's / the earlier Claude analysis: a KC responds to an odor if its trial-mean
# response is > 1 (units as uploaded, see README)
RESPONSE_THRESH = 1.0
RECORDING_KEYS = ['date', 'fly_num', 'hemibrain']


def fetch(url: str, dest: Path) -> None:
    """Downloads url to dest (via urllib), making parent dirs. Skips if dest exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'downloading {url} -> {dest}')
    with urlopen(url, timeout=600) as r, open(dest, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def download(olfdb: str, out_dir: Path) -> dict:
    """Downloads metadata, long CSV and original files of each dataset into out_dir/<genotype>/.

    Returns:
        dict mapping genotype -> path of its long CSV.
    """
    paths = {}
    for ds_id, geno in DATASETS.items():
        d = out_dir / geno
        fetch(f'{olfdb}/api/datasets/{ds_id}', d / f'dataset_{ds_id}_metadata.json')
        fetch(f'{olfdb}/api/download?ids={ds_id}&format=long',
            d / f'dataset_{ds_id}_long.csv')
        fetch(f'{olfdb}/api/download?ids={ds_id}&format=files',
            d / f'dataset_{ds_id}_original_files.zip')
        paths[geno] = d / f'dataset_{ds_id}_long.csv'
    return paths


def odor_abbrev(odor_str: str) -> str:
    """'1-5ol @ -3' -> '1-5ol'."""
    return odor_str.split(' @ ')[0]


def load_trialmeans(long_csv: Path, genotype: str) -> pd.DataFrame:
    """Loads a long CSV and returns trial-mean responses, one row per (recording, KC, odor).

    Columns: genotype, date, fly_num, hemibrain, roi, odor, log10_conc, n_trials, value.
    """
    df = pd.read_csv(long_csv, dtype={'date': str, 'hemibrain': str})
    assert set(df.columns) == {'date', 'fly_num', 'hemibrain', 'roi', 'panel', 'odor',
        'repeat', 'value'}, df.columns
    assert (df.panel == 'APL16').all()
    df['log10_conc'] = df.odor.map(lambda s: float(s.split(' @ ')[1]))
    df['odor'] = df.odor.map(odor_abbrev)
    g = df.groupby(RECORDING_KEYS + ['roi', 'odor', 'log10_conc'], sort=False)['value']
    tm = g.agg(value='mean', n_trials='size').reset_index()
    assert (tm.n_trials == 3).all(), tm.n_trials.value_counts()
    tm.insert(0, 'genotype', genotype)
    return tm


def rdm_long(mat: pd.DataFrame, odors: list) -> pd.DataFrame:
    """Pairwise 1 - Pearson and 1 - cosine between odor rows of a (odor x KC) matrix.

    Returns one row per unordered odor pair (odor_a < odor_b in `odors` order).
    """
    x = mat.loc[odors].to_numpy(dtype=float)
    # pearson
    xc = x - x.mean(axis=1, keepdims=True)
    pear = (xc @ xc.T) / np.sqrt(np.outer((xc**2).sum(1), (xc**2).sum(1)))
    cos = (x @ x.T) / np.sqrt(np.outer((x**2).sum(1), (x**2).sum(1)))
    rows = []
    for i in range(len(odors)):
        for j in range(i + 1, len(odors)):
            rows.append((odors[i], odors[j], 1 - pear[i, j], 1 - cos[i, j]))
    return pd.DataFrame(rows, columns=['odor_a', 'odor_b', 'one_minus_pearson',
        'one_minus_cosine'])


def square(long_df: pd.DataFrame, col: str, odors: list) -> pd.DataFrame:
    """Symmetric odor x odor matrix (zero diagonal) from a long pair table."""
    m = pd.DataFrame(0.0, index=odors, columns=odors)
    for a, b, v in long_df[['odor_a', 'odor_b', col]].itertuples(index=False):
        m.loc[a, b] = m.loc[b, a] = v
    return m


def report_matrices() -> dict:
    """The 16-odor control / TNT corr-distance matrices the report analyzed, as pair tables."""
    p = HERE / 'apl_pair_analysis' / 'all120_pairs.csv'
    df = pd.read_csv(p, index_col=[0, 1])
    df.index.names = ['odor_a', 'odor_b']
    return {'TNT3c': df['ctrl'], 'TNT': df['tnt']}


def compare(pairs_mean: pd.DataFrame, ref: pd.Series, col: str) -> str:
    """One line comparing a recipe's mean pair distances to the report's for one genotype."""
    s = pairs_mean.set_index(['odor_a', 'odor_b'])[col]
    # the report's pairs may be in either order
    idx = []
    for a, b in ref.index:
        idx.append((a, b) if (a, b) in s.index else (b, a))
    s = s.loc[idx].to_numpy()
    r = ref.to_numpy()
    return (f'{col:>18s}: max |diff| {np.abs(s - r).max():.4f}, mean diff {np.mean(s - r):+.4f}, '
        f'pearson {pearsonr(s, r)[0]:.4f}, spearman {spearmanr(s, r)[0]:.4f}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--shared-dir', type=Path, default=DEFAULT_SHARED)
    ap.add_argument('--olfdb', default=DEFAULT_OLFDB)
    args = ap.parse_args()

    out_dir = args.shared_dir / 'data' / 'yang_apl16_kc_olfdb'
    comp_dir = out_dir / 'computed'
    comp_dir.mkdir(parents=True, exist_ok=True)
    long_csvs = download(args.olfdb, out_dir)

    tms = pd.concat([load_trialmeans(p, g) for g, p in long_csvs.items()], ignore_index=True)
    odors_all = list(dict.fromkeys(tms.odor))
    odors16 = [o for o in odors_all if o != SOLVENT]
    assert len(odors16) == 16, odors16
    conc = tms.drop_duplicates('odor').set_index('odor').log10_conc
    print('odors:', ', '.join(f'{o} @ {conc[o]:g}' for o in odors_all))

    tms.to_parquet(comp_dir / 'trialmean_responses_long.parquet', index=False)
    tms.to_csv(comp_dir / 'trialmean_responses_long.csv.gz', index=False)

    rec_keys = ['genotype'] + RECORDING_KEYS
    recordings, pair_rows, frac_rows = [], [], []
    for rec, sub in tms.groupby(rec_keys, sort=True):
        mat = sub.pivot(index='odor', columns='roi', values='value')
        assert not mat.isna().any().any(), rec
        recordings.append(dict(zip(rec_keys, rec), n_kcs=mat.shape[1]))
        for odors, tag in ((odors16, 'no_pfo'), (odors_all, 'with_pfo')):
            pr = rdm_long(mat, odors)
            pr.insert(0, 'odor_set', tag)
            for k, v in zip(rec_keys, rec):
                pr.insert(0, k, v)
            pair_rows.append(pr)
        responding = (mat > RESPONSE_THRESH)
        fr = responding.mean(axis=1).rename('frac_responding').reset_index()
        fr['n_responding'] = responding.sum(axis=1).to_numpy()
        fr['n_kcs'] = mat.shape[1]
        # KCs responding to at least one non-solvent odor
        fr['frac_kcs_responding_any_odor'] = responding.loc[odors16].any(axis=0).mean()
        for k, v in zip(rec_keys, rec):
            fr.insert(0, k, v)
        frac_rows.append(fr)

    recs = pd.DataFrame(recordings)
    recs.to_csv(comp_dir / 'recordings.csv', index=False)
    pairs = pd.concat(pair_rows, ignore_index=True)
    pairs.to_csv(comp_dir / 'rdm_per_recording_long.csv', index=False)
    fracs = pd.concat(frac_rows, ignore_index=True)
    fracs.to_csv(comp_dir / 'response_fraction_per_recording.csv', index=False)

    print('\nrecordings per genotype:')
    print(recs.groupby('genotype').agg(n_recordings=('n_kcs', 'size'),
        n_flies=('fly_num', lambda s: len(set(zip(recs.loc[s.index, 'date'], s)))),
        n_kcs=('n_kcs', 'sum')).to_string())

    # means: over recordings (each recording weight 1) and over flies (hemispheres of a
    # fly averaged first)
    vals = ['one_minus_pearson', 'one_minus_cosine']
    by_fly = pairs.groupby(['genotype', 'odor_set', 'date', 'fly_num', 'odor_a', 'odor_b'],
        sort=False)[vals].mean().reset_index()
    means = {
        'recording_mean': pairs.groupby(['genotype', 'odor_set', 'odor_a', 'odor_b'],
            sort=False)[vals].mean().reset_index(),
        'fly_mean': by_fly.groupby(['genotype', 'odor_set', 'odor_a', 'odor_b'],
            sort=False)[vals].mean().reset_index(),
    }
    for how, m in means.items():
        m.to_csv(comp_dir / f'rdm_{how}_long.csv', index=False)
        for (geno, oset), sub in m.groupby(['genotype', 'odor_set']):
            odors = odors16 if oset == 'no_pfo' else odors_all
            for col in vals:
                sq = square(sub, col, odors)
                sq.to_csv(comp_dir / f'rdm_{how}_{geno}_{oset}_{col}.csv')

    frac_mean = fracs.groupby(['genotype', 'odor'], sort=False).agg(
        frac_responding_mean=('frac_responding', 'mean'),
        frac_responding_sem=('frac_responding', 'sem'),
        n_recordings=('frac_responding', 'size')).reset_index()
    frac_mean.to_csv(comp_dir / 'response_fraction_recording_mean.csv', index=False)
    print(f'\nmean fraction of KCs responding (> {RESPONSE_THRESH:g}), over odors except '
        'pfo, mean over recordings:')
    print(frac_mean[frac_mean.odor != SOLVENT].groupby('genotype').frac_responding_mean
        .mean().round(4).to_string())

    # which recipe reproduces the report's matrices?
    ref = report_matrices()
    lines = ['Comparison of recipes to the matrices in the report (all120_pairs.csv ctrl/tnt),'
        ' 16 odors (pfo excluded), 120 pairs:']
    for how, m in means.items():
        for geno in ('TNT3c', 'TNT'):
            sub = m[(m.genotype == geno) & (m.odor_set == 'no_pfo')]
            lines.append(f'\n{how}, {geno} (report mean {ref[geno].mean():.4f}):')
            for col in vals:
                lines.append('  ' + compare(sub, ref[geno], col)
                    + f', recipe mean {sub[col].mean():.4f}')
    txt = '\n'.join(lines)
    print('\n' + txt)
    (comp_dir / 'comparison_to_report_matrices.txt').write_text(txt + '\n')

    meta = {
        'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'script': str(Path(__file__).relative_to(HERE.parents[2])),
        'al_analysis_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE,
            capture_output=True, text=True).stdout.strip(),
        'olfdb': args.olfdb,
        'datasets': DATASETS,
        'response_thresh': RESPONSE_THRESH,
        'odors': {o: conc[o] for o in odors_all},
    }
    (comp_dir / 'provenance.json').write_text(json.dumps(meta, indent=1) + '\n')


if __name__ == '__main__':
    main()
