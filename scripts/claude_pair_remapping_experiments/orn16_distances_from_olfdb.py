#!/usr/bin/env python3
"""Rebuild the 16-odor ORN odor-odor distance matrix (1 - Pearson r across glomeruli) from
olfdb, over the 36 glomeruli the model's ORN input uses.

The matrix the program has used as its 16-odor ORN reference (apl_pair_analysis/
orn16_consensus_corrdist.csv, `kc_remapping_harness.orn16_consensus_pairs()`) was computed by
the original report's analysis from al_analysis/data/sent_to_vlad/2024-06-21/consensus_mean.csv
over the union of the megamat and validation2 panels' glomeruli, with 0 in place of glomeruli a
panel did not image (DL4 and VM4 for the megamat odors; VA4 and VL1 for the validation2 odors),
so unmeasured glomeruli counted as zero responses (unresolved_issues.md). This script recomputes
it from Tom's ORN data in olfdb (dataset 1, megamat panel, 9 flies; dataset 4, validation2 panel,
6 flies; http://hal:8080) over the 36 glomeruli both panels imaged, the same set as the model's
input (`kc_remapping_harness.apl16_orn_deltas()`):

- pairs within a panel (66 megamat x megamat, 6 validation2 x validation2): the mean over flies of
  each fly's 1 - r, with al_analysis's `al_util.mean_of_fly_corrs` (trial means, one correlation per
  fly across its ROIs, averaged over flies ignoring NaN);
- pairs across panels (48): no fly was imaged with both panels, so a per-fly correlation does not
  exist; these use 1 - r between the two odors' fly-mean glomerulus vectors (trial means, then the
  mean over flies), labelled in the pairs table.
A fly-mean-vector version of all 120 pairs is written alongside for comparison.

Outputs (shared directory, data/model_outputs/inputs/; the existing ORN input
orn_deltas_apl16*.csv files there are not touched):
  orn16_corrdist_olfdb_36glom.csv        16 x 16, the matrix described above (diagonal 0)
  orn16_corrdist_olfdb_36glom_pairs.csv  120 pairs: both versions, the method per pair, the old value
  orn16_corrdist_olfdb_36glom.json       provenance (olfdb dataset ids and sha256, glomeruli, odors,
                                         al_analysis commit)
Usage: ./venv/bin/python orn16_distances_from_olfdb.py
"""
from __future__ import annotations

import io
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from urllib.request import urlopen

import numpy as np
import pandas as pd

import kc_remapping_harness as H

OLFDB = 'http://hal:8080'
DATASETS = {'megamat': 1, 'validation2': 4}
OUT = H.shared_dir() / 'data' / 'model_outputs' / 'inputs'
STEM = 'orn16_corrdist_olfdb_36glom'


def fetch(ds_id: int) -> tuple:
    with urlopen(f'{OLFDB}/api/download?ids={ds_id}&format=long') as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    with urlopen(f'{OLFDB}/api/datasets/{ds_id}') as r:
        meta = json.loads(r.read())
    return df, meta


def panel_trials(df: pd.DataFrame, panel: str, odors, gloms) -> pd.DataFrame:
    """Trial-level responses of `odors` (abbreviations, at the model input's concentrations):
    rows (odor1, repeat), columns (date, fly_num, roi) restricted to `gloms`."""
    want = {f'{o} @ {H.ORN_INPUT_CONC[o]:g}' for o in odors}
    d = df[(df.panel == panel) & df.odor1.isin(want) & df.roi.isin(gloms)]
    missing = want - set(d.odor1)
    assert not missing, (panel, missing)
    return d.pivot_table(index=['odor1', 'repeat'], columns=['date', 'fly_num', 'roi'], values='value')


def main() -> None:
    from al_analysis.al_util import mean_of_fly_corrs
    gloms = list(H.apl16_orn_deltas().index)
    assert len(gloms) == 36
    trials, metas = {}, {}
    panel_odors = {'megamat': list(H.MEGAMAT12), 'validation2': list(H.VALIDATION2_4)}
    for panel, ds in DATASETS.items():
        df, metas[panel] = fetch(ds)
        absent = set(gloms) - set(df[df.panel == panel].roi)
        assert not absent, (panel, absent)
        trials[panel] = panel_trials(df, panel, panel_odors[panel], gloms)
    abbrev = lambda s: s.split(' @ ')[0]

    # within-panel pairs: mean over flies of per-fly correlations (al_analysis's function)
    within = {}
    n_flies = {}
    for panel, t in trials.items():
        corr = mean_of_fly_corrs(t, square=True)
        corr.index = [abbrev(o) for o in corr.index]; corr.columns = [abbrev(o) for o in corr.columns]
        n_flies[panel] = t.columns.droplevel('roi').nunique()
        for a, b in combinations(panel_odors[panel], 2):
            within[(a, b)] = 1.0 - float(corr.loc[a, b])

    # fly-mean glomerulus vectors (trial mean per fly, then mean over flies), all 16 odors
    vec = {}
    for panel, t in trials.items():
        fm = t.groupby(level='odor1').mean().T.groupby(level='roi').mean()  # roi x odor
        for c in fm.columns:
            vec[abbrev(c)] = fm[c].reindex(gloms)
    V = pd.DataFrame(vec)[list(H.APL16_ODORS)]
    assert not V.isna().any().any()

    old = H.orn16_consensus_pairs()
    get = lambda s, a, b: s.get((a, b), s.get((b, a), np.nan))
    rows = []
    for a, b in combinations(H.APL16_ODORS, 2):
        pa = 'megamat' if a in H.MEGAMAT12 else 'validation2'
        pb = 'megamat' if b in H.MEGAMAT12 else 'validation2'
        flymean = 1.0 - float(np.corrcoef(V[a], V[b])[0, 1])
        if pa == pb:
            main_v, method = within[(a, b)], 'mean over flies of per-fly 1 - r (al_util.mean_of_fly_corrs)'
        else:
            main_v, method = flymean, '1 - r of fly-mean vectors (no fly imaged with both panels)'
        rows.append(dict(odor_a=a, odor_b=b, panel_a=pa, panel_b=pb, distance=main_v, method=method,
            distance_flymean_vectors=flymean, distance_old_zero_filled=get(old, a, b)))
    pairs = pd.DataFrame(rows)
    sq = pd.DataFrame(0.0, index=H.APL16_ODORS, columns=H.APL16_ODORS)
    for r in pairs.itertuples():
        sq.loc[r.odor_a, r.odor_b] = sq.loc[r.odor_b, r.odor_a] = r.distance
    OUT.mkdir(parents=True, exist_ok=True)
    sq.to_csv(OUT / f'{STEM}.csv')
    pairs.to_csv(OUT / f'{STEM}_pairs.csv', index=False)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='orn16_distances_from_olfdb.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfdb=OLFDB, datasets={p: dict(id=DATASETS[p], title=metas[p].get('title'), data_sha256=metas[p].get('data_sha256'), n_flies=int(n_flies[p]))
            for p in DATASETS},
        glomeruli=gloms, odors={o: f'{o} @ {H.ORN_INPUT_CONC[o]:g}' for o in H.APL16_ODORS},
        within_panel='al_analysis.al_util.mean_of_fly_corrs (trial means; per-fly Pearson r across that fly\'s ROIs among the 36; mean over flies ignoring NaN); distance = 1 - mean r',
        across_panel='1 - Pearson r between fly-mean glomerulus vectors (36 glomeruli); no fly imaged with both panels',
        approx_conc_note=H.APPROX_CONC_NOTE)
    (OUT / f'{STEM}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')

    # comparison with the old matrix and the KC data
    pd.set_option('display.width', 200)
    pairs['kind'] = np.where(pairs.panel_a == pairs.panel_b, pairs.panel_a + ' x ' + pairs.panel_b, 'megamat x validation2')
    s = pairs.assign(d_new_old=(pairs.distance - pairs.distance_old_zero_filled).abs(),
        d_within_flymean=(pairs.distance - pairs.distance_flymean_vectors).abs())
    print(s.groupby('kind')[['d_new_old', 'd_within_flymean']].agg(['mean', 'max']).round(4).to_string())
    new = pairs.set_index(['odor_a', 'odor_b']).distance
    fmv = pairs.set_index(['odor_a', 'odor_b']).distance_flymean_vectors
    for g, lab in (('TNT3c', 'KCs, control flies'), ('TNT', 'KCs, APL output blocked (TNT)')):
        kc = H.yang_mean_rdm(g, 'cosine')
        out = []
        for name, o in (('old (zero-filled)', old), ('new', new), ('new, fly-mean vectors only', fmv)):
            a, b = H.align_pairs(kc, o)
            out.append(f'{name} ρ={H.spearman(a, b):.3f}')
        print(f'{lab} vs ORNs: ' + '; '.join(out))
    print('wrote', OUT / f'{STEM}.csv')


if __name__ == '__main__':
    main()
