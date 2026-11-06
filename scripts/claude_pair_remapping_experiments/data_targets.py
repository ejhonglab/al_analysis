#!/usr/bin/env python3
"""The data-target registry: every KC- and ORN-data value the reports quote, computed in one
place with its definition and inputs.

Writes results/data_targets/data_targets.json in the shared directory: {name: {value, ci (optional
[lo, hi], 90 % over recordings), definition, function, inputs ({path: sha256}), commit,
computed_at}}. Report templates reference entries as `{{data:<name>|spec}}` (report_values.py);
figure scripts and loss functions should read their reference values from here instead of
literals. KC distances are one minus the cosine similarity of per-recording trial-mean response
vectors, averaged over recordings (the report's matrices); ORN distances are
`kc_remapping_harness.orn16_consensus_pairs()` (1 - Pearson r, 36 glomeruli, olfdb).
Usage: ./venv/bin/python data_targets.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

OUT = H.shared_dir() / 'results' / 'data_targets' / 'data_targets.json'
YANG = H.yang_data_dir() / 'computed'
INPUTS = {
    'kc_rdms': YANG / 'rdm_per_recording_long.csv',
    'kc_fractions': YANG / 'response_fraction_per_recording.csv',
    'kc_trialmeans': YANG / 'trialmean_responses_long.parquet',
    'orn': H.shared_dir() / 'data' / 'model_outputs' / 'inputs' / 'orn16_corrdist_olfdb_36glom.csv',
    'ref66': H.APL_PAIR_DIR / 'shared12_pairs.csv',
    'e7_blocks': H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_blocks_graded.csv',
}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build() -> dict:
    """Computes every registry entry; returns {name: entry}."""
    from E35_divisive_apl import suppression_stats
    from summary_background_figure import rate_ratios
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    sha = {k: _sha(p) for k, p in INPUTS.items()}
    reg = {}

    def add(name, value, definition, function, inputs, ci=None):
        e = dict(value=float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value,
            definition=definition, function=function, inputs={str(INPUTS[k]): sha[k] for k in inputs}, commit=commit, computed_at=now)
        if ci is not None:
            ci = list(ci.values) if hasattr(ci, 'values') else list(ci)
            e['ci'] = [float(ci[0]), float(ci[1])]
        reg[name] = e

    orn = H.orn16_consensus_pairs()
    recs = {g: H.yang_recording_rdms(g, 'cosine') for g in ('TNT3c', 'TNT')}
    kc = {g: recs[g].mean() for g in recs}
    orn, kc['TNT3c'] = H.align_pairs(orn, kc['TNT3c']); _, kc['TNT'] = H.align_pairs(kc['TNT3c'], kc['TNT'])
    idx = kc['TNT3c'].index
    within = pd.Series([H.ODOR_GROUPS.get(a, a) == H.ODOR_GROUPS.get(b, b) for a, b in idx], index=idx)
    block = pd.Series([H.block_label(a, b) == H.FRUITY_SMALL_BLOCK for a, b in idx], index=idx)
    mats = {'orn': (orn, 'ORNs (1 - Pearson r across 36 glomeruli)', ['orn']),
        'ctrl': (kc['TNT3c'], 'KCs, control flies (1 - cosine, mean over 11 recordings)', ['kc_rdms']),
        'tnt': (kc['TNT'], 'KCs, APL output blocked (TNT) (1 - cosine, mean over 8 recordings)', ['kc_rdms'])}
    for name, (m, lab, inp) in mats.items():
        fn = 'data_targets.build (pair distances)'
        add(f'{name}_mean_dist', m.mean(), f'{lab}: mean pair distance over the 120 pairs', fn, inp)
        add(f'{name}_min_dist', m.min(), f'{lab}: smallest pair distance', fn, inp)
        add(f'{name}_max_dist', m.max(), f'{lab}: largest pair distance', fn, inp)
        add(f'{name}_n_gt08', int((m > 0.8).sum()), f'{lab}: number of pairs more than 0.8 apart', fn, inp)
        add(f'{name}_within_mean', m[within].mean(), f'{lab}: mean distance of pairs within a chemical group', fn, inp)
        add(f'{name}_fruity_small_mean', m[block].mean(), f'{lab}: mean distance of the 9 fruity x small-ester pairs', fn, inp)
    add('rho_ctrl_orn', H.spearman(kc['TNT3c'], orn), 'ρ on odor-odor distances, KCs in control flies vs ORNs (120 pairs)', 'harness.spearman', ['kc_rdms', 'orn'])
    add('rho_tnt_orn', H.spearman(kc['TNT'], orn), 'ρ on odor-odor distances, KCs with APL output blocked (TNT) vs ORNs', 'harness.spearman', ['kc_rdms', 'orn'])

    def on_off(means):
        st, _ = H.compare_on_off(means['TNT3c'], means['TNT'])
        return {k: st[k] for k in ('spearman_on_off', 'iso_r2', 'block_fruity_small_iso', 'lin_r2', 'block_fruity_small_lin', 'block_other_between_iso')}
    boot = H.bootstrap_recordings(on_off, recs, n_boot=1000, seed=0)
    fn = 'harness.compare_on_off; 90 % interval: harness.bootstrap_recordings (1000 resamples of recordings, seed 0)'
    add('rho_ctrl_tnt', boot.loc['spearman_on_off', 'observed'], 'ρ on odor-odor distances between the two KC matrices', fn, ['kc_rdms'],
        ci=boot.loc['spearman_on_off', ['lo', 'hi']])
    add('iso_r2', boot.loc['iso_r2', 'observed'], 'isotonic R² of the control matrix as a monotone function of the APL output blocked (TNT) matrix', fn, ['kc_rdms'],
        ci=boot.loc['iso_r2', ['lo', 'hi']])
    add('lin_r2', boot.loc['lin_r2', 'observed'], 'linear R² of the control matrix on the APL output blocked (TNT) matrix', fn, ['kc_rdms'], ci=boot.loc['lin_r2', ['lo', 'hi']])
    add('block', boot.loc['block_fruity_small_iso', 'observed'], 'fruity x small-ester block: mean isotonic residual of the 9 pairs (control on APL output blocked)', fn,
        ['kc_rdms'], ci=boot.loc['block_fruity_small_iso', ['lo', 'hi']])
    add('block_linear', boot.loc['block_fruity_small_lin', 'observed'], 'the block with a linear instead of the isotonic fit', fn, ['kc_rdms'],
        ci=boot.loc['block_fruity_small_lin', ['lo', 'hi']])
    add('block_other_between', boot.loc['block_other_between_iso', 'observed'], 'mean isotonic residual of the other between-group pairs', fn, ['kc_rdms'],
        ci=boot.loc['block_other_between_iso', ['lo', 'hi']])

    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        sh = H.split_half_reliability(recs[g], n_splits=200, seed=0)
        add(f'{name}_split_half_rho', sh['split_half'], f'ρ between the mean matrices of random halves of the {"control" if g == "TNT3c" else "APL output blocked (TNT)"} recordings (200 splits)',
            'harness.split_half_reliability', ['kc_rdms'], ci=(sh['split_half_lo'], sh['split_half_hi']))
        add(f'{name}_ceiling', sh['ceiling'], 'measurement ceiling on odor-odor distances: sqrt of the Spearman-Brown-corrected split-half ρ '
            '(expected ρ of a noise-free model with the observed mean matrix)', 'harness.split_half_reliability', ['kc_rdms'])

    fr = {g: H.yang_response_fractions(g)[H.APL16_ODORS] for g in ('TNT3c', 'TNT')}
    rng = np.random.default_rng(0)
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        add(f'{name}_frac', fr[g].mean(axis=1).mean(), 'fraction of KCs responding (trial mean > 1), mean over the 16 odors and the recordings', 'harness.yang_response_fractions', ['kc_fractions'])
        vals = []
        n = len(fr[g])
        for _ in range(200):
            i = rng.permutation(n)
            vals.append(H.spearman(fr[g].iloc[i[:n // 2]].mean(), fr[g].iloc[i[n // 2:]].mean()))
        r = float(np.nanmean(vals)); full = 2 * r / (1 + r)
        add(f'{name}_rate_rho_split_half', r, 'ρ between the per-odor response fractions of random halves of the recordings (200 splits)', 'data_targets.build', ['kc_fractions'])
        add(f'{name}_rate_rho_ceiling', float(np.sqrt(full)), 'response rate ρ ceiling: sqrt of the Spearman-Brown-corrected split-half ρ of the per-odor response fractions', 'data_targets.build', ['kc_fractions'])
    rr = rate_ratios()
    add('rate_ratio_fraction', rr['rate_ratio_fraction'], 'response rate with APL output blocked (TNT) relative to control flies: ratio of the mean fractions of KCs responding', 'summary_background_figure.rate_ratios', ['kc_fractions'], ci=rr['rate_ratio_fraction_ci90'])
    add('rate_ratio_mean_response', rr['rate_ratio_mean_response'], 'the same ratio for the mean response size (trial-mean responses with negative values set to 0)', 'summary_background_figure.rate_ratios', ['kc_trialmeans'], ci=rr['rate_ratio_mean_response_ci90'])

    e7 = pd.read_csv(INPUTS['e7_blocks'], header=[0, 1], index_col=0)
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        for blk, lab in (('block_fruity_small', 'block'), ('other_between', 'other'), ('block_minus_other', 'block_minus_other')):
            row = e7.loc[f'{g}_{blk}']
            add(f'{name}_excess_{lab}', row[('excess_dist', 'observed')], f'excess shared responders (graded density null, observed minus null distance), {lab} pairs; E7_confound_controls.py',
                'read from E7_confound_controls/B_blocks_graded.csv', ['e7_blocks'], ci=(row[('excess_dist', 'lo')], row[('excess_dist', 'hi')]))

    ref = H.reference_pairs()
    for col, name in (('kc_ctrl', 'ctrl'), ('kc_tnt', 'tnt')):
        st = H.compare_to_references(ref[col], ref)
        add(f'{name}_struct_wt', st['spearman_resid_vs_remap_wt'], 'structure beyond the ORN input vs wild-type somata: ρ of the 66-pair residual around the linear ORN prediction with the same residual of Remy\'s wild-type KC somata', 'harness.compare_to_references', ['ref66'])
        add(f'{name}_vcf', st['spearman_resid_vs_vcf_hyp'], 'natural-source alignment: ρ of the 66-pair residual around the linear ORN prediction with the hyperbolic natural-source distances', 'harness.compare_to_references', ['ref66'])

    sup = suppression_stats(fr['TNT3c'].mean(), fr['TNT'].mean(), {'TNT3c': fr['TNT3c'].mean(), 'TNT': fr['TNT'].mean()})
    boots = []
    for _ in range(2000):
        c = fr['TNT3c'].iloc[rng.integers(0, len(fr['TNT3c']), len(fr['TNT3c']))].mean()
        t = fr['TNT'].iloc[rng.integers(0, len(fr['TNT']), len(fr['TNT']))].mean()
        boots.append(np.log(t / c))
    add('supp_spread', sup['suppression_sd_log'], 'spread of suppression: sd across the 16 odors of log(APL output blocked (TNT) / control response fraction)', 'E35_divisive_apl.suppression_stats', ['kc_fractions'])
    add('supp_noise', float(pd.DataFrame(boots).std().mean()), 'bootstrap noise of the per-odor log ratio (mean over odors of its sd over 2000 resamples of recordings)', 'data_targets.build', ['kc_fractions'])
    add('supp_rho_on', sup['spearman_suppression_vs_on_frac'], 'rank correlation of the per-odor suppression with the control flies\' response fraction', 'E35_divisive_apl.suppression_stats', ['kc_fractions'])
    for k in ('esters', 'other', 'weak'):
        add(f'supp_{k}', sup[f'suppression_{k}'], f'APL output blocked (TNT) / control response fraction, {k} odors (group means)', 'E35_divisive_apl.suppression_stats', ['kc_fractions'])
    add('n_recordings_ctrl', len(recs['TNT3c']), 'control recordings', 'harness.yang_recording_rdms', ['kc_rdms'])
    add('n_recordings_tnt', len(recs['TNT']), 'APL output blocked (TNT) recordings', 'harness.yang_recording_rdms', ['kc_rdms'])
    return reg


def main() -> None:
    reg = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + '\n')
    for k, e in reg.items():
        ci = f"  [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]" if 'ci' in e else ''
        print(f'{k:28s} {e["value"]:.4f}{ci}')
    print('wrote', OUT, len(reg), 'entries')


if __name__ == '__main__':
    main()
