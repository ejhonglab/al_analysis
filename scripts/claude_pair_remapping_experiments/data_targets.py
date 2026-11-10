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
Usage: ./venv/bin/python data_targets.py [--add-new]
(--add-new: compute every entry but write only those not yet in the registry, keeping the existing entries as
they are, and list any existing entry whose recomputed value or interval differs)
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
    'kc_trials_ctrl': H.yang_data_dir() / 'TNT3c' / 'dataset_20_long.csv',
    'kc_trials_tnt': H.yang_data_dir() / 'TNT' / 'dataset_19_long.csv',
    # how Yang's responses are made, checked on her suite2p outputs (yang_response_normalization.py, viki)
    'yang_norm_recordings': H.shared_dir() / 'notes' / 'yang_response_normalization' / 'recordings.csv',
    'yang_norm_kc_scales': H.shared_dir() / 'notes' / 'yang_response_normalization' / 'kc_scales.csv',
}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def apl_effect_resamples(recs: dict, seed: int = 0) -> tuple:
    """The per-pair APL effect of per-recording distances `recs` ({'TNT3c', 'TNT'}: rows recordings,
    columns pairs), with 1000 resamples of recordings (each genotype with replacement: the resampled
    effects' means, their RMS differences from the measured effect) and 500 random split halves (ρ
    between the halves' effects), in this order from one generator seeded `seed`."""
    effect = H.apl_effect(recs['TNT3c'].mean(), recs['TNT'].mean())
    rng_e = np.random.default_rng(seed)
    boot_mean, boot_rmse = [], []
    for _ in range(1000):
        m = H.resample_recording_means(recs, rng_e)
        e_b = H.apl_effect(m['TNT3c'], m['TNT']).reindex(effect.index)
        boot_mean.append(float(e_b.mean()))
        boot_rmse.append(float(np.sqrt(np.mean((e_b - effect) ** 2))))
    sh_rho = []
    for _ in range(500):
        halves = [{}, {}]
        for g, df in recs.items():
            idx = rng_e.permutation(len(df))
            halves[0][g] = df.iloc[idx[:len(df) // 2]].mean()
            halves[1][g] = df.iloc[idx[len(df) // 2:]].mean()
        a = H.apl_effect(halves[0]['TNT3c'], halves[0]['TNT'])
        b = H.apl_effect(halves[1]['TNT3c'], halves[1]['TNT']).reindex(a.index)
        sh_rho.append(H.spearman(a, b))
    return effect, boot_mean, boot_rmse, sh_rho


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
    # the same mean distances with the data transformed as the models' spike counts are
    # (sub-threshold responses, trial mean <= 1, set to 0; a model KC that does not respond
    # contributes 0, so this is the comparable target for spike-count cosine distances)
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        rec_mats = H.yang_recording_matrices(g)
        thr_rdms = H.recording_rdms_from_matrices({k: H.transform_matrix(m, values='thresholded') for k, m in rec_mats.items()})
        add(f'{name}_mean_dist_thresholded', thr_rdms.mean(axis=0).mean(),
            f'KCs, {"control flies" if g == "TNT3c" else "APL output blocked (TNT)"}: mean pair distance over the 120 pairs with '
            'sub-threshold responses (trial mean <= 1) set to 0 before the cosine distance, as a model\'s spike counts are 0 for '
            'non-responding KCs (1 - cosine per recording, mean over recordings)', 'harness.transform_matrix(values=thresholded) + recording_rdms_from_matrices', ['kc_trialmeans'])
    # the trial-to-trial noise floor and the noise-corrected distances (noise_floor_trial_distances.py):
    # noise inflates every cosine distance toward 1, so the corrected mean is the level a noise-free
    # model should match
    from noise_floor_trial_distances import analyse as nf_analyse, load_trials as nf_load, recording_arrays as nf_recs
    corrected_means = {}
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        lab = 'control flies' if g == 'TNT3c' else 'APL output blocked (TNT)'
        floors, pairs = [], []
        for rec, by_odor in nf_recs(nf_load(g)).items():
            po, pr = nf_analyse(by_odor, 'graded')
            floors.append(po); pairs.append(pr['corrected'])
        corrected = pd.concat(pairs, axis=1).mean(axis=1)
        corrected_means[g] = corrected
        add(f'{name}_mean_dist_noise_corrected', corrected.mean(),
            f'KCs, {lab}: mean pair distance over the 120 pairs after correcting each recording\'s cosine for trial-to-trial noise '
            '(signal power |mean|^2 - s^2/n per odor; cosine of the trial means divided by the signal norms), mean over recordings: '
            'the distance a noise-free recording would show', 'noise_floor_trial_distances.analyse', [f'kc_trials_{name}'])
        add(f'{name}_trial_mean_floor', float(np.mean([f.trial_mean_floor.mean() for f in floors])),
            f'KCs, {lab}: expected 1 - cosine between two independent 3-trial means of the same odor (noise alone), mean over odors and recordings',
            'noise_floor_trial_distances.analyse', [f'kc_trials_{name}'])
        add(f'{name}_single_trial_floor', float(np.mean([f.single_trial_floor.mean() for f in floors])),
            f'KCs, {lab}: 1 - cosine between two single trials of the same odor, mean over trial pairs, odors and recordings',
            'noise_floor_trial_distances.analyse', [f'kc_trials_{name}'])
    # the block on the noise-corrected matrices: what a noise-free model should make, if the
    # correction holds for its noise (the trial-to-trial variability experiment, E43, found it
    # recovers a model's deterministic block for two of its three noise models)
    c_ctrl, c_tnt = H.align_pairs(corrected_means['TNT3c'], corrected_means['TNT'])
    st_c, _ = H.compare_on_off(c_ctrl, c_tnt)
    add('block_corrected', st_c['block_fruity_small_iso'],
        'fruity x small-ester block on the noise-corrected matrices: mean isotonic residual of the 9 pairs (control on APL output '
        'blocked), each matrix the mean over recordings of its trial-noise-corrected cosine distances',
        'noise_floor_trial_distances.analyse + harness.compare_on_off', ['kc_trials_ctrl', 'kc_trials_tnt'])
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

    # the per-pair APL effect (control minus APL output blocked (TNT) distance; Tom's main target
    # since 2026-11-09 17:55): its mean and spread, the loss a model with the true effect would
    # still show against the measured one (bootstrap over recordings), and the ceiling on ρ
    # across pairs (Spearman-Brown-corrected split halves)
    effect, boot_mean, boot_rmse, sh_rho = apl_effect_resamples(recs)
    efn = 'harness.apl_effect; 90 % interval and floor: harness.resample_recording_means (1000 resamples of recordings, seed 0)'
    add('apl_effect_mean', effect.mean(), 'mean over the 120 pairs of the per-pair APL effect: the control flies\' distance minus the distance '
        'with APL output blocked (TNT), cosine, each the mean over recordings', efn, ['kc_rdms'], ci=np.percentile(boot_mean, [5, 95]))
    add('apl_effect_sd', effect.std(), 'standard deviation over the 120 pairs of the per-pair APL effect (control minus APL output blocked (TNT))',
        'harness.apl_effect', ['kc_rdms'])
    add('apl_effect_min', effect.min(), 'smallest per-pair APL effect (control minus APL output blocked (TNT))', 'harness.apl_effect', ['kc_rdms'])
    add('apl_effect_max', effect.max(), 'largest per-pair APL effect (control minus APL output blocked (TNT))', 'harness.apl_effect', ['kc_rdms'])
    add('apl_effect_rmse_floor', float(np.mean(boot_rmse)), 'floor of the APL-effect loss: root mean squared difference over the 120 pairs '
        'between the per-pair APL effect of a resampled data set (recordings drawn with replacement within each genotype) and the measured '
        'one, mean over resamples: the loss a model with the true per-pair effect would show against the measured effect', efn, ['kc_rdms'],
        ci=np.percentile(boot_rmse, [5, 95]))
    r_sh = float(np.nanmean(sh_rho)); full = 2 * r_sh / (1 + r_sh)
    add('apl_effect_rho_split_half', r_sh, 'ρ on the per-pair APL effect between random halves of the recordings (each genotype split in two; 500 splits)',
        'data_targets.build', ['kc_rdms'], ci=np.percentile(sh_rho, [5, 95]))
    add('apl_effect_rho_ceiling', float(np.sqrt(full)), 'ceiling on ρ on the per-pair APL effect: sqrt of the Spearman-Brown-corrected split-half ρ '
        '(the expected ρ of a noise-free model with the true per-pair effect against the measured one)', 'data_targets.build', ['kc_rdms'])
    # the same for the data read like-for-like with a model readout (the compressive readout
    # experiment, E65): binarized at trial mean > 1 (the target of a binarized model readout) and
    # thresholded (trial means <= 1 set to 0, as a non-responding model KC contributes 0); and the
    # mean of the noise-corrected effect (the effect a noise-free recording would show)
    for key, lab, rec_v, inp in (('binary', 'binarized responses (trial mean > 1)', {g: H.yang_recording_rdms(g, 'cosine', binarize=1.0) for g in recs},
            ['kc_trialmeans']),
        ('thresholded', 'sub-threshold responses (trial mean <= 1) set to 0', {g: H.yang_recording_rdms_values(g, 'thresholded') for g in recs},
            ['kc_trialmeans'])):
        e_v, bm_v, br_v, sh_v = apl_effect_resamples(rec_v)
        fn_v = f'harness.yang_apl_effect (data from {lab}); interval and floor as apl_effect_rmse_floor'
        add(f'apl_effect_{key}_mean', e_v.mean(), f'mean over the 120 pairs of the per-pair APL effect computed from {lab}', fn_v, inp,
            ci=np.percentile(bm_v, [5, 95]))
        add(f'apl_effect_{key}_rmse_floor', float(np.mean(br_v)), f'floor of the APL-effect loss against the effect computed from {lab} '
            '(as apl_effect_rmse_floor)', fn_v, inp, ci=np.percentile(br_v, [5, 95]))
        r_v = float(np.nanmean(sh_v))
        add(f'apl_effect_{key}_rho_ceiling', float(np.sqrt(2 * r_v / (1 + r_v))), f'ceiling on ρ on the per-pair APL effect computed from {lab} '
            '(as apl_effect_rho_ceiling)', fn_v, inp)
    add('apl_effect_noise_corrected_mean', H.yang_apl_effect(values='noise_corrected').mean(), 'mean over the 120 pairs of the per-pair APL effect '
        'computed from the noise-corrected distances (each recording\'s cosine corrected for trial-to-trial noise, mean over recordings)',
        'harness.yang_apl_effect(values=noise_corrected)', ['kc_trials_ctrl', 'kc_trials_tnt'])
    # the low end of the readout (E68): the data's measurement noise (trial-to-trial sd of the
    # non-responding entries), the noise floor, the APL effect with negative values or values below
    # the floor set to 0, and the shares of entries per amplitude band the low end is calibrated on
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        add(f'{name}_trial_noise_sd', H.kc_trial_noise_sd(g), f'KCs, {"control flies" if g == "TNT3c" else "APL output blocked (TNT)"}: '
            'trial-to-trial sd of the non-responding entries (trial mean <= 1 z), the root of the mean over those entries of their variance '
            'across the three trials, pooled over recordings', 'harness.kc_trial_noise_sd', [f'kc_trials_{name}'])
    floor_z = H.kc_noise_floor()
    add('kc_noise_floor', floor_z, 'noise floor of the KC data\'s "floor" reading: twice the sd of a trial mean\'s noise, ctrl_trial_noise_sd / sqrt(3) '
        '(the control flies\' noise, used for both genotypes)', 'harness.kc_noise_floor', ['kc_trials_ctrl'])
    for key, lab, inp in (('nonneg', 'trial means with negative values set to 0', ['kc_trialmeans']),
            ('floor', f'trial means at or below the noise floor ({floor_z:.3f} z) set to 0', ['kc_trialmeans', 'kc_trials_ctrl'])):
        e_v, bm_v, br_v, sh_v = apl_effect_resamples({g: H.yang_recording_rdms_values(g, key) for g in recs})
        fn_v = f'harness.yang_apl_effect(values={key}); interval and floor as apl_effect_rmse_floor'
        add(f'apl_effect_{key}_mean', e_v.mean(), f'mean over the 120 pairs of the per-pair APL effect computed from {lab}', fn_v, inp,
            ci=np.percentile(bm_v, [5, 95]))
        add(f'apl_effect_{key}_rmse_floor', float(np.mean(br_v)), f'floor of the APL-effect loss against the effect computed from {lab} '
            '(as apl_effect_rmse_floor)', fn_v, inp, ci=np.percentile(br_v, [5, 95]))
        r_v = float(np.nanmean(sh_v))
        add(f'apl_effect_{key}_rho_ceiling', float(np.sqrt(2 * r_v / (1 + r_v))), f'ceiling on ρ on the per-pair APL effect computed from {lab} '
            '(as apl_effect_rho_ceiling)', fn_v, inp)
    from E68_readout_low_end import trialmean_band_shares
    bands = {'share_05_15': 'with a trial mean above 0.5 and at most 1.5 z', 'share_05_1': 'with a trial mean above 0.5 and at most 1 z',
        'share_1_15': 'with a trial mean above 1 and at most 1.5 z', 'share_025_1': 'with a trial mean above 0.25 and at most 1 z',
        'share_above_4': 'with a trial mean above 4 z', 'share_below_m025': 'with a trial mean below -0.25 z', 'share_negative': 'with a negative trial mean'}
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        lab = 'control flies' if g == 'TNT3c' else 'APL output blocked (TNT)'
        sh = trialmean_band_shares(g)
        for k, desc in bands.items():
            add(f'{name}_{k.replace("share_", "share_z_")}', sh[k], f'KCs, {lab}: share of the KC x odor entries (16 odors) {desc}, mean over recordings',
                'E68_readout_low_end.trialmean_band_shares', ['kc_trialmeans'])
        add(f'{name}_resp_amp_mean_z', sh['resp_amp_mean'], f'KCs, {lab}: mean trial mean of the responders (above 1 z), pooled over odors per '
            'recording, mean over recordings', 'E68_readout_low_end.trialmean_band_shares', ['kc_trialmeans'])

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
    from E26_claw_power import WEAK
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        w = float(fr[g].mean()[WEAK].mean())
        add(f'{name}_weak_frac', w, 'fraction of KCs responding (trial mean > 1) to the six weak-input odors (ms, paa, euc, -aPine, B-myr, 1-8ol), mean over those odors and the recordings', 'data_targets.build', ['kc_fractions'])
        add(f'{name}_weak_share', w / float(fr[g].mean(axis=1).mean()), "the six weak-input odors' response fraction relative to the mean over the 16 odors", 'data_targets.build', ['kc_fractions'])

    e7 = pd.read_csv(INPUTS['e7_blocks'], header=[0, 1], index_col=0)
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        for blk, lab in (('block_fruity_small', 'block'), ('other_between', 'other'), ('block_minus_other', 'block_minus_other')):
            row = e7.loc[f'{g}_{blk}']
            add(f'{name}_excess_{lab}', row[('excess_dist', 'observed')], f'excess shared responders (graded density null, observed minus null distance), {lab} pairs; E7_confound_controls.py',
                'read from E7_confound_controls/B_blocks_graded.csv', ['e7_blocks'], ci=(row[('excess_dist', 'lo')], row[('excess_dist', 'hi')]))

    ref = H.reference_pairs()

    def vcf_alignment(means):
        # natural-source alignment of each genotype's 66-pair residual (the recording means
        # reproduce shared12_pairs.csv's kc_ctrl / kc_tnt to 1e-8), and control minus TNT
        out = {g: H.compare_to_references(means[g].reindex(ref.index), ref)['spearman_resid_vs_vcf_hyp'] for g in ('TNT3c', 'TNT')}
        return {'ctrl': out['TNT3c'], 'tnt': out['TNT'], 'diff': out['TNT3c'] - out['TNT']}
    vboot = H.bootstrap_recordings(vcf_alignment, recs, n_boot=1000, seed=0)
    vfn = 'harness.compare_to_references on the recording means of the 66 pairs; 90 % interval: harness.bootstrap_recordings (1000 resamples of recordings, seed 0)'
    for col, name in (('kc_ctrl', 'ctrl'), ('kc_tnt', 'tnt')):
        st = H.compare_to_references(ref[col], ref)
        add(f'{name}_struct_wt', st['spearman_resid_vs_remap_wt'], 'structure beyond the ORN input vs wild-type somata: ρ of the 66-pair residual around the linear ORN prediction with the same residual of Remy\'s wild-type KC somata', 'harness.compare_to_references', ['ref66'])
        add(f'{name}_vcf', st['spearman_resid_vs_vcf_hyp'], 'natural-source alignment: ρ of the 66-pair residual around the linear ORN prediction with the hyperbolic natural-source distances', vfn,
            ['ref66', 'kc_rdms'], ci=vboot.loc[name, ['lo', 'hi']])
    add('vcf_apl_effect', vboot.loc['diff', 'observed'], 'natural-source alignment, control flies minus APL output blocked (TNT): what APL output adds to it', vfn,
        ['ref66', 'kc_rdms'], ci=vboot.loc['diff', ['lo', 'hi']])

    for col, name, lab in (('kc_ctrl', 'ctrl', 'control flies'), ('kc_tnt', 'tnt', 'APL output blocked (TNT)')):
        _, fit = H.linear_resid(ref[col], ref['orn'])
        for k, desc in (('slope', 'slope b'), ('intercept', 'intercept a'), ('r2', 'R²')):
            add(f'{name}_orn_fit_{k}', fit[k], f'{desc} of the least-squares line KC distance = a + b × mean ORN distance over the 66 megamat pairs, KCs in {lab}',
                'harness.linear_resid (as in compare_to_references)', ['ref66'])
    add('n_ref_pairs', len(ref), 'odor pairs shared by the APL16 panel and the megamat references (12 odors)', 'harness.reference_pairs', ['ref66'])
    add('n_vcf_filled', int(ref['vcf_raw'].isna().sum()), 'of those pairs, how many have no raw natural-source dissimilarity (embedding-filled)', 'harness.reference_pairs', ['ref66'])

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
    # how Yang's KC distances agree with Remy's wild-type megamat KC data on the 12 shared odors
    # (66 pairs; yang_remy_kc_agreement.py). Remy's distances are correlation distances (1 - Pearson
    # across KC somata or claws), Yang's cosine distances, so the comparisons are rank-based
    # (Pearson r of the two distance sets is given as well).
    from yang_remy_kc_agreement import agreement
    ag = agreement(ref, n_boot=2000, seed=0)
    agfn = 'yang_remy_kc_agreement.agreement'
    remy_sets = {'soma': "Remy's wild-type KC somata (kc_remy_combo, 18 flies; correlation distance)",
        'soma_2022': "Remy's 2022 wild-type KC somata (4 flies; correlation distance)",
        'claw': "Remy's wild-type KC claws (4 flies; correlation distance)"}
    for x, geno in (('ctrl', 'control flies'), ('tnt', 'flies with APL output blocked (TNT)')):
        for r, desc in remy_sets.items():
            add(f'rho_{x}_remy_{r}', ag[f'rho_{x}_remy_{r}'], f'ρ on odor-odor distances, KCs in {geno} (Yang, cosine distance) vs {desc}, the 66 pairs of the 12 shared odors', agfn, ['kc_rdms', 'ref66'],
                ci=[ag['rho_%s_remy_soma_lo' % x], ag['rho_%s_remy_soma_hi' % x]] if r == 'soma' else None)
            add(f'pearson_{x}_remy_{r}', ag[f'pearson_{x}_remy_{r}'], f'Pearson r between the same two sets of 66 distances', agfn, ['kc_rdms', 'ref66'])
        add(f'rho_{x}_orn_66', ag[f'rho_{x}_orn_66'], f'ρ on odor-odor distances, KCs in {geno} vs ORNs, on the 66 shared pairs', agfn, ['kc_rdms', 'ref66'])
        add(f'{x}_ceiling_66', ag[f'{x}_ceiling_66'], f'measurement ceiling on odor-odor distances of the {geno} matrix on the 66 shared pairs (sqrt of the Spearman-Brown-corrected split-half ρ, 200 splits)', 'harness.split_half_reliability', ['kc_rdms', 'ref66'])
        add(f'{x}_split_half_66', ag[f'{x}_split_half_66'], f'ρ between the mean matrices of random halves of the {geno} recordings, 66 shared pairs (200 splits)', 'harness.split_half_reliability', ['kc_rdms', 'ref66'])
        add(f'{x}_rec_rho_remy_soma', ag[f'{x}_rec_rho_remy_soma'], f"mean over {geno} recordings of one recording's ρ on odor-odor distances with Remy's wild-type KC somata (66 pairs)", agfn, ['kc_rdms', 'ref66'])
        add(f'{x}_rec_rho_loo', ag[f'{x}_rec_rho_loo'], f"mean over {geno} recordings of one recording's ρ on odor-odor distances with the mean of the other recordings of its genotype (66 pairs)", agfn, ['kc_rdms', 'ref66'])
        add(f'{x}_mean_66', ag[f'{x}_mean_66'], f'KCs, {geno}: mean cosine distance over the 66 shared pairs', agfn, ['kc_rdms', 'ref66'])
        add(f'{x}_block_mean_66', ag[f'{x}_block_mean_66'], f'KCs, {geno}: mean cosine distance of the 9 fruity x small-ester pairs', agfn, ['kc_rdms', 'ref66'])
        add(f'{x}_within_mean_66', ag[f'{x}_within_mean_66'], f'KCs, {geno}: mean cosine distance of the within-group pairs among the 12 shared odors', agfn, ['kc_rdms', 'ref66'])
    add('remy_soma_mean_66', ag['remy_soma_mean_66'], "Remy's wild-type KC somata: mean correlation distance over the 66 shared pairs", agfn, ['ref66'])
    add('remy_soma_block_mean_66', ag['remy_soma_block_mean_66'], "Remy's wild-type KC somata: mean correlation distance of the 9 fruity x small-ester pairs", agfn, ['ref66'])
    add('remy_soma_within_mean_66', ag['remy_soma_within_mean_66'], "Remy's wild-type KC somata: mean correlation distance of the within-group pairs", agfn, ['ref66'])
    add('rho_remy_soma_orn', ag['rho_remy_soma_orn'], "ρ on odor-odor distances, Remy's wild-type KC somata vs ORNs (66 pairs)", agfn, ['ref66'])
    add('rho_remy_soma_claw', ag['rho_remy_soma_claw'], "ρ on odor-odor distances, Remy's wild-type KC somata vs Remy's KC claws (66 pairs)", agfn, ['ref66'])
    add('rho_remy_soma_soma_2022', ag['rho_remy_soma_soma_2022'], "ρ on odor-odor distances, Remy's 18-fly KC somata vs the 2022 4-fly somata (66 pairs)", agfn, ['ref66'])
    add('rho_ctrl_tnt_66', ag['rho_ctrl_tnt_66'], 'ρ on odor-odor distances between the two Yang KC matrices on the 66 shared pairs', agfn, ['kc_rdms', 'ref66'])
    # how concentrated each odor's response vector is, and how many odors each KC responds to (the compressive
    # readout experiment's part A, E65): the calibration targets of its readouts and of the APL-effect fits (E67)
    from E65_spike_count_readout import part_a
    _, amp, _ = part_a()
    afn = 'E65_spike_count_readout.part_a (per recording over the 16 odors, mean over recordings)'
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        lab = 'control flies' if g == 'TNT3c' else 'APL output blocked (TNT)'
        a = amp.loc[g]
        add(f'{name}_participation_ratio_resp', a['participation_ratio_resp'], f'KCs, {lab}: participation ratio of each odor\'s response '
            'vector across KCs, (sum x^2)^2 / (n sum x^4), on responders only (trial means <= 1 set to 0), mean over odors and recordings', afn, ['kc_trialmeans'])
        add(f'{name}_participation_ratio_all', a['participation_ratio_all'], f'KCs, {lab}: the same participation ratio on all values', afn, ['kc_trialmeans'])
        add(f'{name}_top1_share_resp', a['top1_share_resp'], f'KCs, {lab}: share of each odor\'s squared response vector carried by the top 1 % '
            'of KCs, responders only', afn, ['kc_trialmeans'])
        add(f'{name}_resp_amp_cv', a['resp_amp_cv'], f'KCs, {lab}: coefficient of variation of the responders\' trial-mean amplitudes '
            '(z > 1), pooled over odors per recording, mean over recordings', afn, ['kc_trialmeans'])
        for b, blab in (('0', 'none'), ('1', 'one'), ('2_3', 'two or three'), ('4_7', 'four to seven'), ('8plus', 'eight or more')):
            add(f'{name}_odors_per_kc_{b}', a[f'odors_{b}'], f'KCs, {lab}: fraction of KCs responding (trial mean > 1) to {blab} of the 16 '
                'odors, mean over recordings', afn, ['kc_trialmeans'])
    # the realism band on odors per KC (Tom, 2026-11-09): how far a single recorded fly's distribution of the number of
    # odors each KC responds to is from the other flies' mean (total variation distance over the five bins); a model
    # whose distance from the flies' mean exceeds the most atypical fly's is outside the band
    opk_bins = lambda n: np.array([np.mean(n == 0), np.mean(n == 1), np.mean((n >= 2) & (n <= 3)), np.mean((n >= 4) & (n <= 7)),
        np.mean(n >= 8)])
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        lab = 'control flies' if g == 'TNT3c' else 'APL output blocked (TNT)'
        per = {}
        for k, m in H.yang_recording_matrices(g).items():
            m = H.abbrev_columns(m)[H.APL16_ODORS]
            per[k] = opk_bins((m.to_numpy(dtype=float) > 1).sum(axis=1))
        P = pd.DataFrame(per).T
        loo = [0.5 * float(np.abs(P.loc[k].to_numpy() - P.drop(index=k).mean().to_numpy()).sum()) for k in P.index]
        ofn = 'data_targets.build (per recording, responders at trial mean > 1; leave-one-out over recordings)'
        add(f'{name}_odors_per_kc_tv_max', max(loo), f'KCs, {lab}: the largest total variation distance between one recording\'s '
            'distribution of the number of odors each KC responds to (0, 1, 2-3, 4-7, 8 or more of the 16) and the mean of the other '
            'recordings: the realism band on odors per KC', ofn, ['kc_trialmeans'])
        add(f'{name}_odors_per_kc_tv_median', float(np.median(loo)), f'KCs, {lab}: the median of the same leave-one-out distances', ofn,
            ['kc_trialmeans'])
    # the recordings' per-KC normalization (E73, part A): each KC's mean of squared trial values over the whole
    # recording sits under an edge at about 1 / q, q the share of the recording inside the response windows, as a
    # z-score over the whole recording requires; q from the edge (the top 1 % of each recording's KCs by mean
    # square), from the single-odor responders with the largest trial means and from the noise budget (each KC's
    # own trial noise against its mean square), per recording, over the 19 APL16 recordings
    from E73_per_kc_normalization import edge_stats, load_panel_trials
    zrows = []
    for key, r in load_panel_trials('APL16').items():
        summ, _ = edge_stats(r['X'], r['stim'])
        zrows.append(dict(genotype=r['genotype'], **summ))
    zr = pd.DataFrame(zrows)
    zfn = 'E73_per_kc_normalization.edge_stats (per recording, 51 trials per KC: 16 odors and the solvent, 3 trials each)'
    for g, name in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        lab = 'control flies' if g == 'TNT3c' else 'APL output blocked (TNT)'
        d = zr[zr.genotype == g]
        add(f'{name}_zcell_edge', d.ms_max.mean(), f'KCs, {lab}: the largest per-KC mean of squared trial values (every trial of the '
            'recording, the solvent\'s included), mean over recordings: the edge a z-score over the whole recording puts at about 1 / q',
            zfn, [f'kc_trials_{name}'], ci=np.percentile(d.ms_max, [5, 95]))
        add(f'{name}_zcell_edge_gap', d.edge_gap.mean(), f'KCs, {lab}: the largest per-KC mean square minus its 99.9th percentile, mean '
            'over recordings (how sharp the edge is)', zfn, [f'kc_trials_{name}'], ci=np.percentile(d.edge_gap, [5, 95]))
    for est, desc in (('q_edge', 'from the edge: per KC, the q its variance budget implies (1 = q (ms - t^2) - q^2 mean^2 + t^2 / r0^2, '
            't the KC\'s own trial noise, r0 a silent KC\'s), median over the top 1 % of the recording\'s KCs by mean square'),
            ('q_single', 'from the single-odor responders: the same per-KC q, median over the 10 KCs responding to one odor with the '
             'largest trial means'),
            ('q_noise', 'from the noise budget: least squares of the KCs\' own trial-noise variance on their mean square, '
             't^2 = r0^2 - r0^2 q (ms - t^2), q = -slope / intercept')):
        v = zr[est].dropna()
        add('zcell_q' if est == 'q_edge' else f'zcell_{est}', float(np.median(v)), f'KCs (both genotypes): the share q of the '
            f'recording inside the response windows under a z-score over the whole recording, {desc}; median over the 19 APL16 '
            'recordings, interval the 5th to 95th percentiles over recordings', zfn, ['kc_trials_ctrl', 'kc_trials_tnt'],
            ci=np.percentile(v, [5, 95]))
    # the recordings' structure as Yang's pipeline makes them (add_to_reports/yang_response_definition.md): the share of
    # each recording inside the 51 response windows (18 volumes, 2-8 s after onset), and each KC's whole-trace SD (its z
    # unit) over its baseline-noise SD, by the number of odors it responds to
    yr = pd.read_csv(INPUTS['yang_norm_recordings'])
    add('zcell_q_window', yr.frac_window_volumes.mean(), 'the share of an APL16 recording inside its 51 response windows (18 volumes each, '
        '2-8 s after onset, of about 7,194 volumes), mean over the 19 recordings', 'yang_response_normalization.py (recordings.csv)',
        ['yang_norm_recordings'])
    ks = pd.read_csv(INPUTS['yang_norm_kc_scales'])
    for b in list(range(11)) + ['11plus']:
        m = ks.n_odors_z1 >= 11 if b == '11plus' else ks.n_odors_z1 == b
        add(f'kc_scale_ratio_{b}', ks.loc[m, 'scale_ratio'].median(), 'KCs (both genotypes) responding (trial mean > 1 z) to '
            f'{"11 or more" if b == "11plus" else b} of the 16 odors: the median of the whole-trace SD of Fc (the z unit) over its SD within '
            'the pre-odor windows (each window\'s mean removed)', 'yang_response_normalization.py (kc_scales.csv)', ['yang_norm_kc_scales'])
    # who carries the APL effect (E74): the per-pair effect split by the number of odors each KC responds to in each
    # condition (0, 1, 2-3, 4-7, 8 or more), with 90 % intervals over resamples of recordings, the floor of the
    # class-resolved loss and the single-recording-pair yardstick
    from E74_apl_effect_by_odors_per_kc import registry_entries as e74_entries
    for name, value, definition, ci in e74_entries():
        add(name, value, definition, 'E74_apl_effect_by_odors_per_kc.data_tables (2000 resamples of recordings, seed 0)',
            ['kc_trialmeans'], ci=ci)
    add('n_recordings_ctrl', len(recs['TNT3c']), 'control recordings', 'harness.yang_recording_rdms', ['kc_rdms'])
    add('n_recordings_tnt', len(recs['TNT']), 'APL output blocked (TNT) recordings', 'harness.yang_recording_rdms', ['kc_rdms'])
    return reg


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--add-new', action='store_true', help='write only the entries not yet in the registry; the existing '
        'entries are kept as they are, and any whose recomputed value or interval differs is listed')
    args = ap.parse_args()
    reg = build()
    if args.add_new and OUT.exists():
        old = json.loads(OUT.read_text())
        differ = [k for k in old if k in reg and (reg[k]['value'] != old[k]['value'] or reg[k].get('ci') != old[k].get('ci'))]
        new = [k for k in reg if k not in old]
        for k in differ:
            print(f'kept (recomputed value differs): {k} {old[k]["value"]!r} (now {reg[k]["value"]!r})')
        print(f'adding {len(new)} entries: {", ".join(new)}')
        reg = dict(old, **{k: reg[k] for k in new})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + '\n')
    for k, e in reg.items():
        ci = f"  [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]" if 'ci' in e else ''
        print(f'{k:28s} {e["value"]:.4f}{ci}')
    print('wrote', OUT, len(reg), 'entries')


if __name__ == '__main__':
    main()
