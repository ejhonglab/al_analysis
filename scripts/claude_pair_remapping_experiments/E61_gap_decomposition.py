#!/usr/bin/env python3
"""E61: where the gap in rho on odor-odor distances sits.

A per-pair and per-odor decomposition of the models' disagreement with the two KC matrices (APL on against the
control flies, APL off against APL output blocked (TNT)), from saved pair distances, with no simulations. rho on
odor-odor distances is Spearman's rho over the 120 pairs, 1 - 6 sum(d^2) / (n (n^2 - 1)) with d a pair's rank in
the model minus its rank in the data (ranks over the 120 pairs; ties by the average rank, so the identity holds up
to ties), so each pair has a rank residual d and a contribution 6 d^2 / (n (n^2 - 1)) to 1 - rho.

Models: the three reference models at 6.8 % (the eight glomeruli at 1.5 at 40 and 70 Hz, E44; the eight at 1.25 at
70 Hz, E45), the first joint fit (E10's k 2, the best APL-on rho), the structure-loss fit on a re-drawn wiring (E30's
within-type claw-permutation draw 2, the best APL-off rho), the bouton baseline with connectome APL weights and the
best centrifugal-input variant (E9), and the ORN input itself (1 - Pearson across the 40 responding glomeruli, the
matrix the fits are scored against as structure beyond the input).

(A) Per pair: the rank residual with APL on and off (16 x 16 matrices, ordered lists) and the contribution to 1 - rho;
    the data's per-recording spread of each pair's distance beside it.
(B) Per odor: the mean absolute rank residual of the odor's 15 pairs and the leave-one-odor-out rho (the odor's pairs
    dropped; its gain over rho is the odor's cost), both conditions, for every model and for the ORN input, so pairs the
    model and the input both miss (an input problem) are separated from pairs only the model misses (a mechanism
    problem); shares of 1 - rho by odor group and by pair class (the fruity x small-ester block pairs, within-group,
    between-group; pairs of the weak-input odors ms, paa, euc, -aPine, B-myr, 1-8ol).
(C) The on-off relation: the model's pair-by-pair APL effect (APL-off minus APL-on distance) against the data's, which
    pairs the model's APL moves the wrong way, and the block pairs among them.
(D) The contrasting models: the correlation of the residual vectors between models (do they miss the same pairs?).

Outputs: <shared>/results/E61_gap_decomposition/ (pair_residuals.csv, odor_summary.csv, class_summary.csv,
apl_effect.csv, residual_correlations.csv, worst_pairs.csv, comparison_table.md, E61_rank_residuals.png,
E61_per_odor.png, E61_apl_effect.png, provenance.json).
Usage: ./venv/bin/python E61_gap_decomposition.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
import noise_floor_trial_distances as NF

EXPERIMENT = 'E61_gap_decomposition'
RES = H.shared_dir() / 'results'
ODORS = list(H.APL16_ODORS)
GROUPS = dict(H.ODOR_GROUPS)
FRUITY, SMALL = ('2h', 'IaA', 'pa'), ('2-but', 'eb', 'ep')
WEAK = ('ms', 'paa', 'euc', '-aPine', 'B-myr', '1-8ol')
DATA_OF = {'on': 'TNT3c', 'off': 'TNT'}
DATA_LABEL = {'on': 'control flies', 'off': 'APL output blocked (TNT)'}
# (label, results folder, tag in model_pair_distances_long.csv); None folder: the ORN input
MODELS = [
    ('the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
    ('eight at 1.5, 70 Hz, 6.8 %', 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3'),
    ('eight at 1.25, 70 Hz, 6.8 %', 'E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3'),
    ('the first joint fit (k 2; the best APL-on rho)', 'E10_joint_fit', 'k2'),
    ('the structure-loss fit on a re-drawn wiring (the best APL-off rho)', 'E30_wiring_replication', 'wiring-claw_permutation_within_type-seed2_k2'),
    ('the bouton baseline (connectome APL weights)', 'E9_centrifugal_input', 'bouton_none'),
    ('the centrifugal input (the best APL-off rho among its variants)', 'E9_centrifugal_input', 'BEST_OFF'),
    ('the ORN input', None, None),
]
SHORT = {m[0]: s for m, s in zip(MODELS, ('reference 40 Hz', 'eight at 1.5, 70 Hz', 'eight at 1.25, 70 Hz', 'first joint fit',
    'best APL-off fit (re-drawn wiring)', 'bouton baseline', 'centrifugal input', 'ORN input'))}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_class(a: str, b: str) -> str:
    if (a in FRUITY and b in SMALL) or (a in SMALL and b in FRUITY):
        return 'block'
    return 'within group' if GROUPS[a] == GROUPS[b] else 'between groups'


def load_model_pairs(folder: str, tag: str, inputs: dict) -> dict:
    """{'on': Series, 'off': Series} of a model's saved pair distances (index (odor_a, odor_b) in the data's order)."""
    f = RES / folder / 'model_pair_distances_long.csv'
    inputs[str(f)] = sha256(f)
    d = pd.read_csv(f)
    out = {}
    for cond, names in (('on', ('on',)), ('off', ('acute_off', 'off'))):
        s = d[(d['tag'] == tag) & d['condition'].isin(names)].set_index(['odor_a', 'odor_b'])['distance']
        if len(s) != 120:
            raise ValueError(f'{folder}:{tag} {cond}: {len(s)} pairs')
        out[cond] = s
    return out


def best_off_tag(folder: str, data_off: pd.Series, inputs: dict) -> str:
    f = RES / folder / 'model_pair_distances_long.csv'
    d = pd.read_csv(f)
    best, best_rho = None, -2
    for tag in d['tag'].unique():
        s = d[(d['tag'] == tag) & d['condition'].isin(('acute_off', 'off'))].set_index(['odor_a', 'odor_b'])['distance']
        if len(s) != 120:
            continue
        r = H.spearman(s.reindex(data_off.index), data_off)
        if r > best_rho:
            best, best_rho = tag, r
    return best


def data_spread(genotype: str, inputs: dict) -> pd.DataFrame:
    """Per pair: the mean and SD across recordings of the three-trial-mean cosine distance, and the mean rank
    of the pair across recordings' own rankings (the data's own per-pair uncertainty)."""
    p = H.yang_data_dir() / genotype / f'dataset_{NF.DATASETS[genotype]}_long.csv'
    inputs[str(p)] = sha256(p)
    recs = NF.recording_arrays(NF.load_trials(genotype))
    raws = [NF.analyse(by_odor, 'graded')[1]['raw'] for by_odor in recs.values()]
    M = pd.concat(raws, axis=1)
    ranks = M.apply(lambda col: pd.Series(sstats.rankdata(col.to_numpy()), index=col.index))
    return pd.DataFrame(dict(data_mean=M.mean(axis=1), data_sd=M.std(axis=1, ddof=1), data_rank_sd=ranks.std(axis=1, ddof=1),
        n_recordings=M.shape[1]))


def residuals(model: pd.Series, data: pd.Series) -> pd.DataFrame:
    """Per pair: ranks, the rank residual (model minus data) and the contribution to 1 - rho."""
    data = data.reindex(model.index)
    rm = sstats.rankdata(model.to_numpy())
    rd = sstats.rankdata(data.to_numpy())
    n = len(model)
    d = rm - rd
    return pd.DataFrame(dict(model_distance=model.to_numpy(), data_distance=data.to_numpy(), rank_model=rm, rank_data=rd,
        rank_resid=d, contribution=6 * d ** 2 / (n * (n ** 2 - 1))), index=model.index)


def loo_rho(model: pd.Series, data: pd.Series) -> pd.Series:
    """Per odor: rho over the pairs not involving the odor, minus rho over all pairs."""
    data = data.reindex(model.index)
    full = H.spearman(model, data)
    out = {}
    for o in ODORS:
        keep = [p for p in model.index if o not in p]
        out[o] = H.spearman(model[keep], data[keep]) - full
    return pd.Series(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    out = args.out or H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    data = {c: H.yang_mean_rdm(DATA_OF[c], 'cosine') for c in ('on', 'off')}
    spread = {c: data_spread(DATA_OF[c], inputs) for c in ('on', 'off')}
    orn = H.orn16_consensus_pairs()
    models = {}
    for label, folder, tag in MODELS:
        if folder is None:
            models[label] = {'on': orn, 'off': orn, 'tag': 'orn16_consensus_pairs', 'folder': None}
            continue
        if tag == 'BEST_OFF':
            tag = best_off_tag(folder, data['off'], inputs)
        pr = load_model_pairs(folder, tag, inputs)
        pr.update(tag=tag, folder=folder)
        models[label] = pr
    rows, odor_rows, cls_rows, apl_rows, rho_rows = [], [], [], [], []
    resid_vectors = {}
    orn_res = {c: residuals(orn, data[c]) for c in ('on', 'off')}
    for label, pr in models.items():
        rh = {}
        for c in ('on', 'off'):
            r = residuals(pr[c], data[c])
            rho = H.spearman(pr[c], data[c])
            rh[c] = rho
            resid_vectors[(label, c)] = r['rank_resid']
            loo = loo_rho(pr[c], data[c])
            sp = spread[c].reindex(r.index)
            orr = orn_res[c].reindex(r.index)
            for (a, b), rr in r.iterrows():
                rows.append(dict(model=label, short=SHORT[label], tag=pr['tag'], condition=c, odor_a=a, odor_b=b, pair=f'{a} / {b}',
                    pair_class=pair_class(a, b), weak_pair=(a in WEAK or b in WEAK), **rr.to_dict(),
                    orn_rank_resid=float(orr.loc[(a, b), 'rank_resid']), data_sd=float(sp.loc[(a, b), 'data_sd']),
                    data_rank_sd=float(sp.loc[(a, b), 'data_rank_sd'])))
            ab = r['rank_resid'].abs()
            for o in ODORS:
                idx = [p for p in r.index if o in p]
                big = ab[idx] >= 30
                big_orn = orr.loc[idx, 'rank_resid'].abs() >= 30
                odor_rows.append(dict(model=label, short=SHORT[label], condition=c, odor=o, group=GROUPS[o], weak=o in WEAK,
                    mean_abs_resid=float(ab[idx].mean()), mean_abs_resid_orn=float(orr.loc[idx, 'rank_resid'].abs().mean()),
                    contribution_share=float(r.loc[idx, 'contribution'].sum() / (1 - rho)) if rho < 1 else np.nan,
                    loo_delta_rho=float(loo[o]), n_big_misses=int(big.sum()), n_big_misses_also_orn=int((big & big_orn).sum()),
                    mean_signed_resid=float(r.loc[idx, 'rank_resid'].mean())))
            r2 = r.copy()
            r2['pair_class'] = [pair_class(a, b) for a, b in r2.index]
            r2['weak_pair'] = [(a in WEAK or b in WEAK) for a, b in r2.index]
            for key, col in (('pair_class', 'pair_class'), ('weak_pair', 'weak_pair')):
                for val, g in r2.groupby(col):
                    cls_rows.append(dict(model=label, short=SHORT[label], condition=c, classification=key, value=str(val), n_pairs=len(g),
                        share_of_gap=float(g['contribution'].sum() / (1 - rho)) if rho < 1 else np.nan,
                        mean_abs_resid=float(g['rank_resid'].abs().mean()), rho=rho))
        rho_rows.append(dict(model=label, short=SHORT[label], tag=pr['tag'], rho_on=rh['on'], rho_off=rh['off']))
        # (C) the APL effect per pair
        eff_m = (pr['off'].reindex(pr['on'].index) - pr['on'])
        eff_d = (data['off'].reindex(pr['on'].index) - data['on'].reindex(pr['on'].index))
        for (a, b) in pr['on'].index:
            apl_rows.append(dict(model=label, short=SHORT[label], odor_a=a, odor_b=b, pair=f'{a} / {b}', pair_class=pair_class(a, b),
                effect_model=float(eff_m[(a, b)]), effect_data=float(eff_d[(a, b)]),
                wrong_way=bool(np.sign(eff_m[(a, b)]) != np.sign(eff_d[(a, b)]) and abs(eff_d[(a, b)]) > 0.02)))
    pairs = pd.DataFrame(rows)
    odors = pd.DataFrame(odor_rows)
    classes = pd.DataFrame(cls_rows)
    apl = pd.DataFrame(apl_rows)
    rhos = pd.DataFrame(rho_rows)
    labels = [m[0] for m in MODELS]
    corr_rows = []
    for c in ('on', 'off'):
        for la in labels:
            for lb in labels:
                corr_rows.append(dict(condition=c, model_a=SHORT[la], model_b=SHORT[lb],
                    spearman=float(sstats.spearmanr(resid_vectors[(la, c)], resid_vectors[(lb, c)].reindex(resid_vectors[(la, c)].index)).statistic)))
    corr = pd.DataFrame(corr_rows)
    worst = pd.concat([g.reindex(g['rank_resid'].abs().sort_values(ascending=False).index).head(12).assign(rank_in_list=range(1, 13))
        for _, g in pairs.groupby(['model', 'condition'])], ignore_index=True)
    pairs.to_csv(out / 'pair_residuals.csv', index=False)
    odors.to_csv(out / 'odor_summary.csv', index=False)
    classes.to_csv(out / 'class_summary.csv', index=False)
    apl.to_csv(out / 'apl_effect.csv', index=False)
    corr.to_csv(out / 'residual_correlations.csv', index=False)
    worst.to_csv(out / 'worst_pairs.csv', index=False)
    rhos.to_csv(out / 'rhos.csv', index=False)
    write_table(out, rhos, odors, classes, apl, corr, worst)
    make_figures(out, pairs, odors, apl, resid_vectors)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    (out / 'provenance.json').write_text(json.dumps(dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E61_gap_decomposition.py', host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=commit, models={l: dict(folder=m['folder'], tag=m['tag']) for l, m in models.items()},
        data=dict(on=DATA_OF['on'], off=DATA_OF['off'], measure='cosine, mean over recordings (H.yang_mean_rdm)'),
        orn='H.orn16_consensus_pairs (1 - Pearson across the 40 responding glomeruli)', weak_input_odors=list(WEAK),
        big_miss='|rank residual| >= 30 of 120', inputs=inputs, approx_conc_note=H.APPROX_CONC_NOTE), indent=1) + '\n')
    print(f'-> {out}', flush=True)


def fmt(x, spec='+.0f'):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def write_table(out: Path, rhos, odors, classes, apl, corr, worst) -> None:
    L = ['| model | ρ on odor-odor distances, APL on | APL off |', '|---|---:|---:|']
    for _, r in rhos.iterrows():
        L.append(f"| {r['model']} | {r['rho_on']:.3f} | {r['rho_off']:.3f} |")
    L += ['| KC data (ceiling) | 0.98 | 0.97 |', '']
    ref = 'the reference model (eight at 1.5, 40 Hz, 6.8 %)'
    L += ['Per odor, the reference model and the ORN input: mean absolute rank residual of the odor\'s 15 pairs (of 120 ranks), the gain in ρ from '
          'dropping the odor\'s pairs, and the share of 1 - ρ its pairs carry:', '',
          '| odor | group | APL on: mean abs residual, model / ORN input | gain in ρ without the odor, model / ORN | share of the gap, model | APL off: mean abs residual, model / ORN input | gain in ρ without the odor, model / ORN | share of the gap, model |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    o_ref = odors[odors.model == ref].set_index(['condition', 'odor'])
    o_orn = odors[odors.model == 'the ORN input'].set_index(['condition', 'odor'])
    for o in ODORS:
        cells = [o, GROUPS[o]]
        for c in ('on', 'off'):
            a, b = o_ref.loc[(c, o)], o_orn.loc[(c, o)]
            cells += [f"{a['mean_abs_resid']:.0f} / {b['mean_abs_resid']:.0f}", f"{a['loo_delta_rho']:+.3f} / {b['loo_delta_rho']:+.3f}", f"{100 * a['contribution_share']:.0f} %"]
        L.append('| ' + ' | '.join(cells) + ' |')
    L += ['', 'Shares of 1 - ρ by pair class (the reference model; the ORN input in brackets):', '',
          '| class | pairs | APL on: share of the gap | mean abs residual | APL off: share of the gap | mean abs residual |', '|---|---:|---:|---:|---:|---:|']
    cr = classes[classes.model == ref].set_index(['condition', 'classification', 'value'])
    co = classes[classes.model == 'the ORN input'].set_index(['condition', 'classification', 'value'])
    for key, vals in (('pair_class', ('block', 'within group', 'between groups')), ('weak_pair', ('True', 'False'))):
        for v in vals:
            name = v if key == 'pair_class' else ('pairs of a weak-input odor' if v == 'True' else 'pairs of no weak-input odor')
            a_on, a_off = cr.loc[('on', key, v)], cr.loc[('off', key, v)]
            b_on, b_off = co.loc[('on', key, v)], co.loc[('off', key, v)]
            L.append(f"| {name} | {int(a_on['n_pairs'])} | {100 * a_on['share_of_gap']:.0f} % ({100 * b_on['share_of_gap']:.0f} %) | {a_on['mean_abs_resid']:.0f} ({b_on['mean_abs_resid']:.0f}) | "
                     f"{100 * a_off['share_of_gap']:.0f} % ({100 * b_off['share_of_gap']:.0f} %) | {a_off['mean_abs_resid']:.0f} ({b_off['mean_abs_resid']:.0f}) |")
    L += ['', 'The twelve worst pairs of the reference model (rank residual: model rank minus data rank, of 120; the ORN input\'s residual for the same pair; the data\'s per-recording SD of the pair\'s rank):', '']
    for c in ('on', 'off'):
        L += [f'APL {c}:', '', '| pair | class | rank residual | ORN input\'s | data rank SD | model distance | data distance |', '|---|---|---:|---:|---:|---:|---:|']
        w = worst[(worst.model == ref) & (worst.condition == c)]
        for _, r in w.iterrows():
            L.append(f"| {r['pair']} | {r['pair_class']} | {r['rank_resid']:+.0f} | {r['orn_rank_resid']:+.0f} | {r['data_rank_sd']:.0f} | {r['model_distance']:.3f} | {r['data_distance']:.3f} |")
        L.append('')
    L += ['The APL effect per pair (APL-off minus APL-on distance), model against data: Spearman ρ across the 120 pairs, and the pairs the model moves the wrong way where the data move by more than 0.02:', '',
          '| model | ρ of the APL effect with the data\'s | pairs moved the wrong way | of which block pairs |', '|---|---:|---:|---:|']
    for label in [m[0] for m in MODELS if m[1] is not None]:
        g = apl[apl.model == label]
        rho_e = sstats.spearmanr(g['effect_model'], g['effect_data']).statistic
        L.append(f"| {label} | {rho_e:.2f} | {int(g['wrong_way'].sum())} | {int((g['wrong_way'] & (g['pair_class'] == 'block')).sum())} |")
    L += ['', 'Do the models miss the same pairs? Spearman ρ between the residual vectors (APL on / APL off):', '']
    shorts = [SHORT[m[0]] for m in MODELS]
    L += ['| | ' + ' | '.join(shorts) + ' |', '|---|' + '---:|' * len(shorts)]
    for sa in shorts:
        cells = []
        for sb in shorts:
            von = corr[(corr.condition == 'on') & (corr.model_a == sa) & (corr.model_b == sb)]['spearman'].iloc[0]
            voff = corr[(corr.condition == 'off') & (corr.model_a == sa) & (corr.model_b == sb)]['spearman'].iloc[0]
            cells.append(f'{von:.2f} / {voff:.2f}')
        L.append(f'| {sa} | ' + ' | '.join(cells) + ' |')
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def make_figures(out: Path, pairs, odors, apl, resid_vectors) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = H.oklab_diverging_cmap()
    show = [('the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'the reference model'), ('the first joint fit (k 2; the best APL-on rho)', 'the first joint fit'),
            ('the structure-loss fit on a re-drawn wiring (the best APL-off rho)', 'the best APL-off fit'), ('the ORN input', 'the ORN input')]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    for i, c in enumerate(('on', 'off')):
        for j, (label, title) in enumerate(show):
            v = resid_vectors[(label, c)]
            M = pd.DataFrame(np.nan, index=ODORS, columns=ODORS)
            for (a, b), x in v.items():
                M.loc[a, b] = x
                M.loc[b, a] = x
            ax = axes[i, j]
            im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=-80, vmax=80)
            ax.set_xticks(range(16)); ax.set_yticks(range(16))
            ax.set_xticklabels(ODORS, rotation=90, fontsize=7); ax.set_yticklabels(ODORS, fontsize=7)
            ax.set_title(f"{title}, APL {c}\n(rank residual against the {DATA_LABEL[c]}; ρ {H.spearman(pairs[(pairs.model == label) & (pairs.condition == c)].set_index(['odor_a', 'odor_b'])['model_distance'], pairs[(pairs.model == label) & (pairs.condition == c)].set_index(['odor_a', 'odor_b'])['data_distance']):.2f})", fontsize=8.5)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='rank residual (model rank minus data rank, of 120 pairs)')
    fig.suptitle('Where the gap in ρ on odor-odor distances sits: each pair\'s rank in the model minus its rank in the KC data', fontsize=11)
    fig.savefig(out / 'E61_rank_residuals.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    ref = 'the reference model (eight at 1.5, 40 Hz, 6.8 %)'
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    x = np.arange(16)
    for i, c in enumerate(('on', 'off')):
        om = odors[(odors.model == ref) & (odors.condition == c)].set_index('odor').reindex(ODORS)
        oo = odors[(odors.model == 'the ORN input') & (odors.condition == c)].set_index('odor').reindex(ODORS)
        ax = axes[i, 0]
        ax.bar(x - 0.2, om['mean_abs_resid'], 0.4, label='the reference model', color='#2b6cb0')
        ax.bar(x + 0.2, oo['mean_abs_resid'], 0.4, label='the ORN input', color='#999999')
        ax.set_xticks(x); ax.set_xticklabels(ODORS, rotation=90)
        ax.set_ylabel("mean absolute rank residual of the odor's 15 pairs (of 120)")
        ax.set_title(f'APL {c}: against the {DATA_LABEL[c]}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        ax = axes[i, 1]
        ax.bar(x - 0.2, om['loo_delta_rho'], 0.4, label='the reference model', color='#2b6cb0')
        ax.bar(x + 0.2, oo['loo_delta_rho'], 0.4, label='the ORN input', color='#999999')
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(ODORS, rotation=90)
        ax.set_ylabel("gain in ρ when the odor's pairs are dropped")
        ax.set_title(f'APL {c}: which odors cost the most ρ')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle("Per odor: how much of the reference model's gap each odor carries, and whether the ORN input carries it too", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'E61_per_odor.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    g = apl[apl.model == ref]
    colors = {'block': '#d9480f', 'within group': '#2f9e44', 'between groups': '#2b6cb0'}
    for cls, gg in g.groupby('pair_class'):
        ax.scatter(gg['effect_data'], gg['effect_model'], s=26, color=colors[cls], label=f'{cls} ({len(gg)} pairs)', zorder=3)
    lim = (min(g['effect_data'].min(), g['effect_model'].min()) - 0.02, max(g['effect_data'].max(), g['effect_model'].max()) + 0.02)
    ax.plot(lim, lim, ':', color='0.5', lw=1); ax.axhline(0, color='0.7', lw=0.8); ax.axvline(0, color='0.7', lw=0.8)
    rho_e = sstats.spearmanr(g['effect_model'], g['effect_data']).statistic
    ax.set(xlabel="the data's APL effect per pair (APL output blocked (TNT) minus control flies, cosine distance)",
        ylabel="the reference model's APL effect per pair (APL off minus APL on)", title=f"The APL effect pair by pair, model against data (ρ {rho_e:.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'E61_apl_effect.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
