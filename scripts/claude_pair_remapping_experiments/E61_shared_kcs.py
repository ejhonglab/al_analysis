#!/usr/bin/env python3
"""E61 addendum: the KCs behind the paa / B-myr / 2-but collapse.

The per-pair decomposition of the gap in rho on odor-odor distances (E61) found paa, B-myr and 2-but far too
close to each other in every model (cosine distance 0.14-0.19 where the control flies have 0.89-0.93), and the
per-odor rate oracle (E62) found that putting each odor's response fraction on the data's barely moves rho. This
asks, from the archived responses and wiring of the reference model (the eight glomeruli at 1.5, 40 Hz per-claw
threshold, 6.8 %) and of the eight at 1.25 at 70 Hz, with APL on and with APL off, and with no simulations:

(A) which KCs respond to paa, B-myr and 2-but, how many respond to two or all three of them, against the
    expectation if the responders were independent sets, and the same for the other weak-input odors (euc,
    -aPine, ms, 1-8ol); beside it the control flies' and the APL-output-blocked flies' responder overlaps
    (observed over expected, from the weak-odor responder analysis, E23);
(B) which glomeruli the shared responders' claws carry, against all KCs (the fraction of KCs in the set with at
    least one claw from the glomerulus over the fraction among all KCs), with each glomerulus's PN rate to the
    odors (the APL-off sharing diagnostic's per-glomerulus rates, E21) and its gain;
(C) how many of a responder's claws are driven above the per-claw knee by each odor (PN rate times the
    glomerulus gain above the knee), which glomeruli those claws come from, and for the shared responders
    whether the same glomerulus drives them for both odors;
(D) the ORN input's own picture: each focus odor's strongest glomeruli and the ones they share.

Outputs: <shared>/results/E61_gap_decomposition/shared_kcs/ (overlap.csv, triple.csv, glomerulus_enrichment.csv,
driven_claws.csv, driven_claws_shared.csv, orn_top_glomeruli.csv, comparison_table.md, E61_shared_kcs.png,
provenance.json). Usage: ./venv/bin/python E61_shared_kcs.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

OUT = H.shared_dir() / 'results' / 'E61_gap_decomposition' / 'shared_kcs'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RES = H.shared_dir() / 'results'
GAIN_GLOMS = ['VA7m', 'DL1', 'D', 'DM1', 'DP1m', 'VC1', 'DC2', 'DM5']
# (label, short, archive folder, tag, per-claw knee in Hz, gain on the eight glomeruli)
MODELS = [
    ('the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'reference 40 Hz', 'E44_operating_point',
        'thr40_gain1.5_rate0.068_k2_cap3', 40.0, 1.5),
    ('eight at 1.25, 70 Hz, 6.8 %', 'eight at 1.25, 70 Hz', 'E45_gain_at_ctrl_frac',
        'thr70_gain1.25_rate0.068_k2_cap3', 70.0, 1.25),
]
CONDS = {'on': 'on', 'off': 'acute_off'}
GENOTYPE_OF = {'on': 'TNT3c', 'off': 'TNT'}
COND_LABEL = {'on': 'APL on (control flies)', 'off': 'APL off (APL output blocked (TNT))'}
FOCUS = ['paa', 'B-myr', '2-but', 'euc', '-aPine', 'ms', '1-8ol']
TRIPLE = ['paa', 'B-myr', '2-but']
SETS = [('paa',), ('B-myr',), ('2-but',), ('paa', 'B-myr'), ('paa', '2-but'), ('B-myr', '2-but'), ('paa', 'B-myr', '2-but')]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_responses(folder: str, tag: str, cond: str) -> pd.DataFrame:
    """KC x odor responses (0 / 1) of an archived variant, odors abbreviated, index kc_id."""
    r = pd.read_parquet(ARCHIVE / folder / f'{tag}_{CONDS[cond]}_responses.parquet')
    r = H.abbrev_columns(r)
    r.index = r.index.get_level_values('kc_id')
    return (r[H.APL16_ODORS] > 0)


def claw_table() -> pd.DataFrame:
    """One row per claw of the bouton base: kc_id, glomerulus."""
    w = pd.read_parquet(ARCHIVE / 'inputs' / 'wPNKC_bouton_base.parquet')
    vals = w.to_numpy()
    assert ((vals != 0).sum(axis=1) == 1).all(), 'one bouton per claw expected'
    glom = w.columns.get_level_values('glomerulus').to_numpy()[vals.argmax(axis=1)]
    return pd.DataFrame(dict(kc_id=w.index.get_level_values('kc_id').to_numpy(), glomerulus=glom))


def set_name(odors) -> str:
    return ' & '.join(odors)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    claws = claw_table()
    gloms = sorted(claws.glomerulus.unique())
    kc_gloms = claws.groupby('kc_id').glomerulus.apply(list)
    pn_path = RES / 'E21_apl_off_sharing' / 'glomerulus_rates_per_odor.csv'
    pn = pd.read_csv(pn_path, index_col=0)
    pn.columns = [H.odor_abbrev(c) for c in pn.columns]
    overlap_path = RES / 'E23_weak_odor_responders' / 'overlap_profiles.csv'
    data_ov = pd.read_csv(overlap_path)
    data_ratio = {}
    for r in data_ov.itertuples():
        data_ratio[(r.genotype, r.odor, r.other)] = (r.ratio, r.observed, r.expected)
        data_ratio.setdefault((r.genotype, r.other, r.odor), (r.ratio, r.observed, r.expected))
    orn = H.abbrev_columns(H.apl16_orn_deltas().droplevel('panel', axis=1))
    inputs = {str(p.relative_to(H.shared_dir())): sha256(p) for p in [pn_path, overlap_path, ARCHIVE / 'inputs' / 'wPNKC_bouton_base.parquet']}

    overlap_rows, triple_rows, enrich_rows, driven_rows, shared_rows = [], [], [], [], []
    for label, short, folder, tag, knee, gain in MODELS:
        gain_of = pd.Series({g: (gain if g in GAIN_GLOMS else 1.0) for g in gloms})
        drive = pn.reindex(gloms).fillna(0.0).mul(gain_of, axis=0)  # glomerulus x odor PN rate times the gain
        for cond in CONDS:
            p = ARCHIVE / folder / f'{tag}_{CONDS[cond]}_responses.parquet'
            inputs[str(p.relative_to(H.shared_dir()))] = sha256(p)
            resp = load_responses(folder, tag, cond)
            kcs = resp.index[resp.index.isin(kc_gloms.index)]
            resp = resp.loc[kcs]
            N = len(resp)
            n = resp.sum()
            # (A) pairwise overlaps among the focus odors, and the triple
            for a, b in combinations(FOCUS, 2):
                shared = int((resp[a] & resp[b]).sum())
                exp = n[a] * n[b] / N
                g = GENOTYPE_OF[cond]
                dr = data_ratio.get((g, a, b), (np.nan, np.nan, np.nan))
                overlap_rows.append(dict(model=label, short=short, condition=cond, odor_a=a, odor_b=b, pair=f'{a} / {b}',
                    n_a=int(n[a]), n_b=int(n[b]), n_kcs=N, shared=shared, expected=exp, ratio=shared / exp if exp else np.nan,
                    jaccard=shared / max(int((resp[a] | resp[b]).sum()), 1), frac_of_smaller=shared / max(min(n[a], n[b]), 1),
                    data_genotype=g, data_ratio=dr[0], data_observed=dr[1], data_expected=dr[2]))
            allthree = resp[TRIPLE].all(axis=1)
            anytwo = (resp[TRIPLE].sum(axis=1) >= 2)
            exp3 = n[TRIPLE].prod() / N ** 2
            triple_rows.append(dict(model=label, short=short, condition=cond, n_paa=int(n['paa']), n_bmyr=int(n['B-myr']),
                n_2but=int(n['2-but']), n_kcs=N, all_three=int(allthree.sum()), expected_all_three=exp3,
                two_or_more=int(anytwo.sum()), union=int(resp[TRIPLE].any(axis=1).sum()),
                frac_paa_also_bmyr=float((resp['paa'] & resp['B-myr']).sum() / max(n['paa'], 1)),
                frac_bmyr_also_paa=float((resp['paa'] & resp['B-myr']).sum() / max(n['B-myr'], 1)),
                frac_2but_also_paa_or_bmyr=float((resp['2-but'] & (resp['paa'] | resp['B-myr'])).sum() / max(n['2-but'], 1))))
            # (B) glomerulus enrichment of the sets' claws
            has = pd.DataFrame({g: kc_gloms.loc[kcs].apply(lambda L, g=g: g in L) for g in gloms}, index=kcs)
            n_claws = pd.DataFrame({g: kc_gloms.loc[kcs].apply(lambda L, g=g: L.count(g)) for g in gloms}, index=kcs)
            frac_all = has.mean()
            for odors in SETS:
                members = resp[list(odors)].all(axis=1)
                m = int(members.sum())
                if m == 0:
                    continue
                frac_set = has[members].mean()
                for g in gloms:
                    enrich_rows.append(dict(model=label, short=short, condition=cond, set=set_name(odors), n_set=m, glomerulus=g,
                        frac_set=float(frac_set[g]), frac_all=float(frac_all[g]),
                        enrichment=float(frac_set[g] / frac_all[g]) if frac_all[g] > 0 else np.nan,
                        mean_claws_set=float(n_claws.loc[members, g].mean()), gain=float(gain_of[g]),
                        pn_rate_min_over_set=float(pn.reindex([g]).fillna(0.0)[list(odors)].min(axis=1).iloc[0]),
                        drive_min_over_set=float(drive.loc[g, list(odors)].min()),
                        driven_min_over_set=bool(drive.loc[g, list(odors)].min() > knee)))
            # (C) driven claws: per focus odor, the responders' claws above the knee
            for o in FOCUS:
                driven_gloms = set(drive.index[drive[o] > knee])
                counts = kc_gloms.loc[kcs].apply(lambda L: sum(1 for g in L if g in driven_gloms))
                resp_o = resp[o]
                for grp, mask in (('responders', resp_o), ('non-responders', ~resp_o)):
                    c = counts[mask]
                    if len(c) == 0:
                        continue
                    top = pd.Series([g for L in kc_gloms.loc[kcs[mask]] for g in L if g in driven_gloms]).value_counts()
                    driven_rows.append(dict(model=label, short=short, condition=cond, odor=o, group=grp, n=int(mask.sum()),
                        driven_glomeruli=','.join(sorted(driven_gloms, key=lambda g: -drive.loc[g, o])), n_driven_glomeruli=len(driven_gloms),
                        mean_driven_claws=float(c.mean()), frac_0=float((c == 0).mean()), frac_1=float((c == 1).mean()),
                        frac_2=float((c == 2).mean()), frac_3plus=float((c >= 3).mean()),
                        top_driven_glomeruli=','.join(f'{g}:{top[g] / max(int(mask.sum()), 1):.2f}' for g in top.index[:5])))
            # the shared responders: which glomerulus drives them for each odor of the pair
            for a, b in combinations(TRIPLE, 2):
                members = kcs[resp[a] & resp[b]]
                da, db = set(drive.index[drive[a] > knee]), set(drive.index[drive[b] > knee])
                both = da & db
                same = 0
                only_shared_gloms = 0
                for kc in members:
                    L = kc_gloms.loc[kc]
                    ga = {g for g in L if g in da}
                    gb = {g for g in L if g in db}
                    if ga and gb and (ga & gb):
                        same += 1
                    if ga and ga <= both and gb and gb <= both:
                        only_shared_gloms += 1
                shared_rows.append(dict(model=label, short=short, condition=cond, pair=f'{a} / {b}', n_shared=len(members),
                    glomeruli_driven_by_both=','.join(sorted(both, key=lambda g: -min(drive.loc[g, a], drive.loc[g, b]))),
                    frac_same_glomerulus_drives_both=same / max(len(members), 1),
                    frac_all_driven_claws_from_shared_glomeruli=only_shared_gloms / max(len(members), 1),
                    top_glomeruli_of_shared=','.join(f'{g}:{v:.2f}' for g, v in has.loc[members].mean().sort_values(ascending=False).head(5).items())))

    overlap = pd.DataFrame(overlap_rows); overlap.to_csv(out / 'overlap.csv', index=False)
    triple = pd.DataFrame(triple_rows); triple.to_csv(out / 'triple.csv', index=False)
    enrich = pd.DataFrame(enrich_rows); enrich.to_csv(out / 'glomerulus_enrichment.csv', index=False)
    driven = pd.DataFrame(driven_rows); driven.to_csv(out / 'driven_claws.csv', index=False)
    shared = pd.DataFrame(shared_rows); shared.to_csv(out / 'driven_claws_shared.csv', index=False)
    # (D) the ORN input's top glomeruli per focus odor
    orn_rows = []
    for o in FOCUS:
        s = orn[o].sort_values(ascending=False)
        for rank, (g, v) in enumerate(s.head(6).items(), 1):
            orn_rows.append(dict(odor=o, rank=rank, glomerulus=g, orn_delta=float(v), gained=g in GAIN_GLOMS,
                pn_rate_base=float(pn[o].get(g, np.nan))))
    orn_top = pd.DataFrame(orn_rows); orn_top.to_csv(out / 'orn_top_glomeruli.csv', index=False)

    write_table(out, overlap, triple, enrich, driven, shared, orn_top)
    make_figure(out, overlap, enrich)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    (out / 'provenance.json').write_text(json.dumps(dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E61_shared_kcs.py', host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=commit, models=[dict(label=l, folder=f, tag=t, knee_hz=k, gain=g) for l, _, f, t, k, g in MODELS],
        gain_glomeruli=GAIN_GLOMS, focus_odors=FOCUS, driven='PN rate (E21 glomerulus_rates_per_odor.csv, base model, 0-0.5 s) x gain > knee',
        data_overlap='E23 overlap_profiles.csv (per-recording z > 1 responders; observed / expected n_a n_b / N)', inputs=inputs,
        approx_conc_note=H.APPROX_CONC_NOTE), indent=1) + '\n')
    print(f'-> {out}', flush=True)


def write_table(out: Path, overlap, triple, enrich, driven, shared, orn_top) -> None:
    L = []
    L.append('Responder overlaps among the focus odors (KCs responding to both; observed over the expectation n_a n_b / N if '
        'the two responder sets were independent; the flies\' value from the weak-odor responder analysis (E23), control flies '
        'for APL on and APL output blocked (TNT) for APL off):\n')
    for (m, c), g in overlap.groupby(['short', 'condition'], sort=False):
        L.append(f'{m}, {COND_LABEL[c]}:\n')
        L.append('| pair | responders a / b (of N) | shared | expected | model ratio | flies\' ratio | Jaccard | share of the smaller set |')
        L.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for r in g.itertuples():
            L.append(f'| {r.pair} | {r.n_a} / {r.n_b} ({r.n_kcs}) | {r.shared} | {r.expected:.1f} | {r.ratio:.2f} | '
                f'{"" if np.isnan(r.data_ratio) else f"{r.data_ratio:.2f}"} | {r.jaccard:.2f} | {r.frac_of_smaller:.2f} |')
        L.append('')
    L.append('paa, B-myr and 2-but together:\n')
    L.append('| model | condition | responders paa / B-myr / 2-but | to all three | expected if independent | to two or more | to any | of paa\'s also B-myr | of B-myr\'s also paa | of 2-but\'s also paa or B-myr |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in triple.itertuples():
        L.append(f'| {r.short} | {r.condition} | {r.n_paa} / {r.n_bmyr} / {r.n_2but} | {r.all_three} | {r.expected_all_three:.1f} | {r.two_or_more} | '
            f'{r.union} | {r.frac_paa_also_bmyr:.2f} | {r.frac_bmyr_also_paa:.2f} | {r.frac_2but_also_paa_or_bmyr:.2f} |')
    L.append('')
    L.append('Glomeruli of the sets\' claws (the fraction of the set\'s KCs with at least one claw from the glomerulus, over the fraction among '
        'all KCs; the five most enriched glomeruli carried by at least a third of the set; the glomerulus\'s PN rate (base model) to the '
        'set\'s odors, the smallest over the set, with the gain applied in brackets, and whether that exceeds the per-claw knee):\n')
    for (m, c), g in enrich.groupby(['short', 'condition'], sort=False):
        L.append(f'{m}, {COND_LABEL[c]}:\n')
        L.append('| set | KCs | glomerulus: fraction of the set / of all KCs (enrichment), PN rate to the set\'s odors x gain |')
        L.append('|---|---:|---|')
        for s, gg in g.groupby('set', sort=False):
            top = gg[gg.frac_set >= 1 / 3].sort_values('enrichment', ascending=False).head(5)
            cells = '; '.join(f"{r.glomerulus}: {r.frac_set:.2f} / {r.frac_all:.2f} ({r.enrichment:.1f}x), "
                f"{r.pn_rate_min_over_set:.0f} Hz" + (f" (x{r.gain:g} = {r.drive_min_over_set:.0f})" if r.gain != 1 else '')
                + (' above the knee' if r.driven_min_over_set else '') for r in top.itertuples())
            L.append(f'| {s} | {int(gg.n_set.iloc[0])} | {cells} |')
        L.append('')
    L.append('Driven claws (claws whose glomerulus\'s PN rate times the gain exceeds the per-claw knee for the odor) of the odor\'s '
        'responders and non-responders:\n')
    L.append('| model | condition | odor | driven glomeruli (by drive) | group | KCs | mean driven claws | none / one / two / three or more | top driven glomeruli (fraction of the group carrying one) |')
    L.append('|---|---|---|---|---|---:|---:|---:|---|')
    for r in driven.itertuples():
        L.append(f'| {r.short} | {r.condition} | {r.odor} | {r.driven_glomeruli} | {r.group} | {r.n} | {r.mean_driven_claws:.2f} | '
            f'{r.frac_0:.2f} / {r.frac_1:.2f} / {r.frac_2:.2f} / {r.frac_3plus:.2f} | {r.top_driven_glomeruli} |')
    L.append('')
    L.append('The shared responders of each pair: the glomeruli both odors drive above the knee, the fraction of the shared KCs for which '
        'one glomerulus drives them for both odors, the fraction whose driven claws for both odors all come from the shared glomeruli, '
        'and the glomeruli they carry most often:\n')
    L.append('| model | condition | pair | shared KCs | glomeruli driven by both | same glomerulus drives both | all driven claws from shared glomeruli | glomeruli carried (fraction) |')
    L.append('|---|---|---|---:|---|---:|---:|---|')
    for r in shared.itertuples():
        L.append(f'| {r.short} | {r.condition} | {r.pair} | {r.n_shared} | {r.glomeruli_driven_by_both} | {r.frac_same_glomerulus_drives_both:.2f} | '
            f'{r.frac_all_driven_claws_from_shared_glomeruli:.2f} | {r.top_glomeruli_of_shared} |')
    L.append('')
    L.append('The ORN input: each focus odor\'s six strongest glomeruli (delta spike rate, Hz; * = one of the eight gained glomeruli), with the '
        'base model\'s PN rate:\n')
    L.append('| odor | glomeruli, strongest first (ORN delta; PN rate) |')
    L.append('|---|---|')
    for o, g in orn_top.groupby('odor', sort=False):
        L.append(f'| {o} | ' + '; '.join(f"{r.glomerulus}{'*' if r.gained else ''} ({r.orn_delta:.0f}; {r.pn_rate_base:.0f})" for r in g.itertuples()) + ' |')
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def make_figure(out: Path, overlap: pd.DataFrame, enrich: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = overlap[(overlap.short == 'reference 40 Hz')]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5), gridspec_kw=dict(width_ratios=[1, 1, 1.5], hspace=0.55, wspace=0.3))
    for row, cond in enumerate(['on', 'off']):
        g = ref[ref.condition == cond]
        for col, (val, title) in enumerate([('ratio', 'the reference model'), ('data_ratio', 'the flies')]):
            M = pd.DataFrame(np.nan, index=FOCUS, columns=FOCUS)
            for r in g.itertuples():
                M.loc[r.odor_a, r.odor_b] = M.loc[r.odor_b, r.odor_a] = getattr(r, val)
            ax = axes[row, col]
            im = ax.imshow(np.log2(M.to_numpy(dtype=float)), cmap='RdBu_r', vmin=-3, vmax=3)
            ax.set_xticks(range(len(FOCUS))); ax.set_xticklabels(FOCUS, rotation=90); ax.set_yticks(range(len(FOCUS))); ax.set_yticklabels(FOCUS)
            for i in range(len(FOCUS)):
                for j in range(len(FOCUS)):
                    v = M.iat[i, j]
                    if i != j and not np.isnan(v):
                        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color='white' if abs(np.log2(v)) > 1.8 else 'black')
            who = 'control flies' if cond == 'on' else 'APL output blocked (TNT)'
            ax.set_title(f"{title}, {'APL on' if cond == 'on' else 'APL off'}" if col == 0 else f'{who}' + '\n(responder overlap, observed / expected)', fontsize=10)
            if col == 0:
                ax.set_title(f"{title}, {'APL on' if cond == 'on' else 'APL off'}\n(responder overlap, observed / expected)", fontsize=10)
        cb = fig.colorbar(im, ax=axes[row, :2].tolist(), fraction=0.025, pad=0.02)
        cb.set_label('observed / expected (log2 scale)')
        # enrichment bars for the triple's shared responders
        e = enrich[(enrich.short == 'reference 40 Hz') & (enrich.condition == cond) & (enrich.set == 'paa & B-myr & 2-but')]
        if len(e) == 0:
            e = enrich[(enrich.short == 'reference 40 Hz') & (enrich.condition == cond) & (enrich.set == 'paa & B-myr')]
        e = e.sort_values('frac_set', ascending=False).head(10)
        ax = axes[row, 2]
        x = np.arange(len(e))
        ax.bar(x - 0.2, e.frac_set, 0.4, label=f"KCs responding to {e.set.iloc[0] if len(e) else ''} (n = {int(e.n_set.iloc[0]) if len(e) else 0})")
        ax.bar(x + 0.2, e.frac_all, 0.4, color='grey', label='all KCs')
        ax.set_xticks(x); ax.set_xticklabels([f"{g}{'*' if g in GAIN_GLOMS else ''}" for g in e.glomerulus], rotation=45)
        ax.set_ylabel('fraction of KCs with a claw from the glomerulus')
        ax.set_title(f"glomeruli of the shared responders' claws, {'APL on' if cond == 'on' else 'APL off'}\n(* = gained glomerulus)", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle('The KCs behind the paa / B-myr / 2-but collapse: the reference model against the flies', fontsize=12)
    fig.savefig(out / 'E61_shared_kcs.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
