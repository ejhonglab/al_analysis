#!/usr/bin/env python3
"""The trial-to-trial noise floor of the KC odor-odor distances, and noise-corrected distances.

The report's KC distances are 1 - cosine similarity between two odors' trial-mean response
vectors over all KCs of a recording (Yang's z-scored fluorescence), averaged over recordings.
Two noisy vectors are never parallel, so noise inflates every distance toward 1, and a
noise-free model should not be expected to reach the measured mean distance. From Yang's
per-trial data (3 trials per odor) this script computes, per recording and genotype:

- the single-trial floor: for each odor, 1 - cosine between two different trials of that
  odor (mean over trial pairs), i.e. the distance noise alone produces between repeats;
- the trial-mean floor: the same for the 3-trial means the report uses, from the noise
  algebra below (the expected 1 - cosine between two independent trial means of one odor);
- noise-corrected odor-odor distances: with x̄_o the trial mean of odor o, n_o its trials and
  s_o² the trial-to-trial variance summed over KCs, the signal power is |a_o|² = |x̄_o|² - s_o²/n_o
  (independent noise adds s²/n to the squared norm of a mean and nothing to the dot product
  between different odors), so the corrected cosine is x̄_o·x̄_p / sqrt(|a_o|² |a_p|²) and the
  corrected distance 1 minus that: the distance a noise-free recording would show.

Each is computed for the graded data (the report's) and for the model-like transform
(sub-threshold values, trial value <= 1, set to 0 per trial; the mean of thresholded trials
is then the "trial mean", which differs slightly from thresholding the trial mean). Distances
are averaged over recordings as in the report; the on-vs-off statistics (ρ, isotonic R², the
block) are recomputed on the corrected matrices to check they stand.

Outputs: <shared>/results/noise_floor/noise_floor_summary.csv (one row per genotype x
transform x quantity), corrected_pairs_long.csv (per pair: raw, corrected, both transforms),
per_odor_floor.csv (per odor and recording), and provenance.json.
Usage: ./venv/bin/python noise_floor_trial_distances.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import kc_remapping_harness as H

DATASETS = {'TNT': 19, 'TNT3c': 20}
THRESH = 1.0


def load_trials(genotype: str) -> pd.DataFrame:
    p = H.yang_data_dir() / genotype / f'dataset_{DATASETS[genotype]}_long.csv'
    d = pd.read_csv(p)
    d['abbrev'] = d['odor'].map(H.odor_abbrev)
    d = d[d.abbrev.isin(H.APL16_ODORS)]
    return d


def recording_arrays(d: pd.DataFrame):
    """{(date, fly_num, hemibrain): {odor: trials x KCs array}} with a common KC order."""
    out = {}
    for rec, sub in d.groupby(['date', 'fly_num', 'hemibrain'], sort=True):
        rois = np.array(sorted(sub.roi.unique()))
        by_odor = {}
        for o, so in sub.groupby('abbrev'):
            w = so.pivot(index='repeat', columns='roi', values='value').reindex(columns=rois)
            assert not w.isna().any().any(), (rec, o)
            by_odor[o] = w.to_numpy(dtype=float)
        out[(str(rec[0]), int(rec[1]), str(rec[2]))] = by_odor
    return out


def transform(x: np.ndarray, how: str) -> np.ndarray:
    if how == 'graded':
        return x
    if how == 'thresholded':
        return np.where(x > THRESH, x, 0.0)
    raise ValueError(how)


def analyse(by_odor: dict, how: str):
    """One recording: per-odor floors and the raw / corrected pair distances."""
    odors = [o for o in H.APL16_ODORS if o in by_odor]
    xbar, sig2, n_tr, floor1 = {}, {}, {}, {}
    for o in odors:
        X = transform(by_odor[o], how)
        n = X.shape[0]
        m = X.mean(axis=0)
        s2 = X.var(axis=0, ddof=1).sum() if n > 1 else np.nan   # trial-to-trial noise power, one trial
        xbar[o], sig2[o], n_tr[o] = m, s2, n
        cs = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = X[i], X[j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                cs.append(1 - a @ b / (na * nb) if na > 0 and nb > 0 else np.nan)
        floor1[o] = float(np.nanmean(cs)) if cs else np.nan
    per_odor = []
    for o in odors:
        norm2 = float(xbar[o] @ xbar[o]); noise_mean = sig2[o] / n_tr[o]
        sig_pow = max(norm2 - noise_mean, 1e-12)
        per_odor.append(dict(odor=o, n_trials=n_tr[o], single_trial_floor=floor1[o],
            trial_mean_floor=float(1 - sig_pow / norm2),          # E[1 - cos] between two independent trial means
            reliability=float(sig_pow / norm2),                    # signal share of the trial mean's power
            signal_power=sig_pow, noise_power_per_trial=float(sig2[o]), norm2_trial_mean=norm2))
    per_odor = pd.DataFrame(per_odor).set_index('odor')
    rows = []
    for i, a in enumerate(odors):
        for b in odors[i + 1:]:
            dot = float(xbar[a] @ xbar[b])
            raw = 1 - dot / np.sqrt((xbar[a] @ xbar[a]) * (xbar[b] @ xbar[b]))
            corr = 1 - dot / np.sqrt(per_odor.loc[a, 'signal_power'] * per_odor.loc[b, 'signal_power'])
            rows.append(dict(odor_a=a, odor_b=b, raw=float(raw), corrected=float(corr)))
    return per_odor, pd.DataFrame(rows).set_index(['odor_a', 'odor_b'])


def main() -> None:
    res_dir = H.shared_dir() / 'results' / 'noise_floor'; res_dir.mkdir(exist_ok=True)
    summary, floors, pairs_long = [], [], []
    mats = {}
    for g in ('TNT3c', 'TNT'):
        recs = recording_arrays(load_trials(g))
        for how in ('graded', 'thresholded'):
            per_rec_pairs, per_rec_floor = [], []
            for rec, by_odor in recs.items():
                po, pr = analyse(by_odor, how)
                po = po.reset_index(); po.insert(0, 'recording', '_'.join(map(str, rec))); po.insert(0, 'transform', how); po.insert(0, 'genotype', g)
                per_rec_floor.append(po); per_rec_pairs.append(pr)
            fl = pd.concat(per_rec_floor); floors.append(fl)
            raw = pd.concat([p['raw'] for p in per_rec_pairs], axis=1).mean(axis=1)
            cor = pd.concat([p['corrected'] for p in per_rec_pairs], axis=1).mean(axis=1)
            mats[(g, how, 'raw')] = raw; mats[(g, how, 'corrected')] = cor
            pl = pd.DataFrame({'raw': raw, 'corrected': cor}); pl.insert(0, 'transform', how); pl.insert(0, 'genotype', g)
            pairs_long.append(pl.reset_index())
            # floors: mean over odors within a recording, then over recordings
            f1 = fl.groupby('recording').single_trial_floor.mean().mean()
            fm = fl.groupby('recording').trial_mean_floor.mean().mean()
            rel = fl.groupby('recording').reliability.mean().mean()
            summary += [
                dict(genotype=g, transform=how, quantity='mean pair distance, trial means (as the report)', value=float(raw.mean())),
                dict(genotype=g, transform=how, quantity='mean pair distance, noise-corrected', value=float(cor.mean())),
                dict(genotype=g, transform=how, quantity='same-odor distance between single trials (floor, single trial)', value=float(f1)),
                dict(genotype=g, transform=how, quantity='same-odor distance between independent trial means (floor, trial mean)', value=float(fm)),
                dict(genotype=g, transform=how, quantity='signal share of the trial-mean power (reliability)', value=float(rel)),
                dict(genotype=g, transform=how, quantity='ρ on odor-odor distances, corrected vs raw', value=float(H.spearman(cor, raw))),
                dict(genotype=g, transform=how, quantity='smallest corrected pair distance', value=float(cor.min())),
                dict(genotype=g, transform=how, quantity='largest corrected pair distance', value=float(cor.max())),
            ]
    for how in ('graded', 'thresholded'):
        for kind in ('raw', 'corrected'):
            st, _ = H.compare_on_off(mats[('TNT3c', how, kind)], mats[('TNT', how, kind)])
            summary += [dict(genotype='both', transform=how, quantity=f'{kind}: block (fruity x small-ester isotonic residual)', value=st['block_fruity_small_iso']),
                dict(genotype='both', transform=how, quantity=f'{kind}: isotonic R² on vs off', value=st['iso_r2']),
                dict(genotype='both', transform=how, quantity=f'{kind}: ρ on odor-odor distances between the two KC matrices', value=st['spearman_on_off'])]
    summ = pd.DataFrame(summary); summ.to_csv(res_dir / 'noise_floor_summary.csv', index=False)
    pd.concat(pairs_long).to_csv(res_dir / 'corrected_pairs_long.csv', index=False)
    pd.concat(floors).to_csv(res_dir / 'per_odor_floor.csv', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 90)
    print(summ.pivot_table(index=['transform', 'quantity'], columns='genotype', values='value', sort=False).round(3).to_string())
    fl = pd.concat(floors)
    print('\nper-odor single-trial floor (graded, mean over recordings):')
    print(fl[fl['transform'] == 'graded'].groupby(['genotype', 'odor']).single_trial_floor.mean().unstack(0).reindex(H.APL16_ODORS).round(3).to_string())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='noise_floor_trial_distances.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs={g: str(H.yang_data_dir() / g / f'dataset_{i}_long.csv') for g, i in DATASETS.items()}, threshold=THRESH,
        method='signal power |xbar|^2 - s^2/n per odor; corrected cosine = xbar_a.xbar_b / sqrt(signal_a signal_b); floors from single-trial pairs and from the same algebra')
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    print('->', res_dir)


if __name__ == '__main__':
    main()
