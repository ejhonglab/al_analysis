#!/usr/bin/env python3
"""E7, model side: RDM-recipe sensitivity of model-vs-data statistics, and the density null
applied to model outputs.

M1 (needs only `results/E1_apl_knockout/model_pair_distances_long.csv`, written by hal's
E1 run): every E1 model (base x rate x condition) in each of the six model RDM variants is
scored against the data in each of six matching data RDM variants (graded / binary values x
cosine / Pearson x all / responsive cells), for both genotypes: Spearman with the data,
Spearman of the deviations from the linear ORN prediction (the report's "model deviations
agree 0.60 with the data" statistic), and the 66-pair deviation agreement with Remy's
wild-type somata. The "like with like" cell (model recipe = data recipe) is flagged, so
the recipe dependence hal found in E1 (point 4) is laid out in full.

M2 (needs per-KC model outputs under `<shared>/data/E1_model_outputs/`, parquet files
named `<tag>_<condition>_responses.parquet` / `_spike_counts.parquet`, KCs x odors):
density (within-odor KC-identity shuffle) null for the model, as for the data in
`E7_confound_controls.py`: expected vs observed cosine similarity, excess similarity, and
the model's excess vs the data's at similar response rates. Skipped if the files are absent.

Outputs to the shared results/E7_confound_controls/ with prefix M.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H


EXP = 'E7_confound_controls'
GENOS = ('TNT3c', 'TNT')
CTRL, TNT = GENOS

# data RDM variant -> (transform kwargs, measure); names chosen to pair with the model's
DATA_VARIANTS: Dict[str, Tuple[dict, str]] = {
    'graded_cosine_all': (dict(values='graded', cells='all'), 'cosine'),
    'graded_pearson_all': (dict(values='graded', cells='all'), 'pearson'),
    'graded_pearson_responsive': (dict(values='graded', cells='responsive'), 'pearson'),
    'binary_cosine_all': (dict(values='binary', cells='all'), 'cosine'),
    'binary_pearson_all': (dict(values='binary', cells='all'), 'pearson'),
    'binary_pearson_responsive': (dict(values='binary', cells='responsive'), 'pearson'),
}
# model RDM variant (harness MODEL_RDM_VARIANTS) -> the data variant with the same recipe
LIKE_WITH_LIKE: Dict[str, str] = {
    'spike_counts_cosine': 'graded_cosine_all',
    'spike_counts_pearson': 'graded_pearson_all',
    'spike_counts_pearson_nosilent': 'graded_pearson_responsive',
    'responses_cosine': 'binary_cosine_all',
    'responses_pearson': 'binary_pearson_all',
    'responses_pearson_nosilent': 'binary_pearson_responsive',
}
CONDITIONS = ('on', 'acute_off')
# which data genotype each model condition is meant to reproduce
CONDITION_GENO = {'on': CTRL, 'acute_off': TNT, 'homeo_full_off': TNT, 'homeo_data_off': TNT}


def data_variant_means(mats: Dict[str, Dict[tuple, pd.DataFrame]]
    ) -> Dict[str, Dict[str, pd.Series]]:
    """{data variant: {genotype: recording-mean pair distances}}."""
    out = {}
    for name, (kws, measure) in DATA_VARIANTS.items():
        out[name] = {g: H.recording_rdms_from_matrices(mats[g], measure=measure,
            thresh=1.0, **kws).mean() for g in GENOS}
    return out


def load_model_pairs(path: Path) -> Dict[Tuple[str, str, str], pd.Series]:
    """{(tag, condition, variant): canonical pair Series} from E1's long table."""
    df = pd.read_csv(path)
    out = {}
    for key, g in df.groupby(['tag', 'condition', 'variant']):
        s = pd.Series(g.distance.to_numpy(), index=pd.MultiIndex.from_arrays(
            [g.odor_a, g.odor_b], names=['odor_a', 'odor_b']))
        out[key] = H.canonical_pairs(s)
    return out


def deviation_agreement(model: pd.Series, data: pd.Series, orn66: pd.Series) -> float:
    """Spearman between model and data deviations from the linear ORN prediction, on the
    66 megamat pairs (ORN = the report's tidy-table ORN distances)."""
    m = model.loc[model.index.intersection(orn66.index)]
    d = data.loc[m.index]
    rm, _ = H.linear_resid(m, orn66.loc[m.index])
    rd, _ = H.linear_resid(d, orn66.loc[m.index])
    return H.spearman(rm, rd)


def m1(out: Path, model_pairs_csv: Path, mats, orn, ref) -> pd.DataFrame:
    models = load_model_pairs(model_pairs_csv)
    dv = data_variant_means(mats)
    long = []
    for name, per_g in dv.items():
        for g, s in per_g.items():
            for (a, b), v in s.items():
                long.append(dict(genotype=g, variant=name, odor_a=a, odor_b=b, distance=v))
    pd.DataFrame(long).to_csv(out / 'M_data_rdm_variants_long.csv', index=False)

    rows = []
    for (tag, cond, mvar), m in models.items():
        if cond not in CONDITIONS:
            continue
        refc = H.compare_to_references(m, ref)
        for dvar, per_g in dv.items():
            for g in GENOS:
                d = per_g[g]
                c = H.compare_model_to_data(m, d, orn=orn)
                rows.append(dict(tag=tag, condition=cond, model_variant=mvar,
                    data_variant=dvar, genotype=g,
                    like_with_like=LIKE_WITH_LIKE[mvar] == dvar,
                    intended_genotype=CONDITION_GENO[cond] == g,
                    spearman=c['spearman'], spearman_dev=c['spearman_dev'],
                    mean_model=c['mean_model'], mean_data=c['mean_data'],
                    dev_sd_model=c['dev_sd_model'], dev_sd_data=c['dev_sd_data'],
                    dev66_vs_data=deviation_agreement(m, d, ref['orn']),
                    dev66_vs_wt=refc['spearman_resid_vs_remap_wt'],
                    spearman66_vcf_hyp=refc['spearman_vcf_hyp'],
                    resid66_vs_vcf_hyp=refc['spearman_resid_vs_vcf_hyp'],
                ))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'M1_recipe_sensitivity_long.csv', index=False)

    # summary 1: deviation agreement with WT (model-only statistic) per model variant
    wt = (tab.drop_duplicates(['tag', 'condition', 'model_variant'])
        .pivot_table(index=['tag', 'condition'], columns='model_variant', values='dev66_vs_wt'))
    wt = wt[list(LIKE_WITH_LIKE)]
    wt.round(3).to_csv(out / 'M1_dev66_vs_wt_by_model_variant.csv')
    print('\nM1a. model deviations (from linear ORN prediction, 66 megamat pairs) vs Remy WT '
        'somata deviations, by model RDM variant:')
    print(wt.round(2).to_string())

    # summary 2: like-with-like vs intended genotype, all stats
    lwl = tab[tab.like_with_like & tab.intended_genotype]
    piv = lwl.pivot_table(index=['tag', 'condition'], columns='model_variant',
        values=['spearman', 'spearman_dev', 'dev66_vs_data'])
    piv.round(3).to_csv(out / 'M1_like_with_like.csv')
    for stat in ('spearman', 'spearman_dev', 'dev66_vs_data'):
        print(f'\nM1b. like-with-like, {stat} (model on vs TNT3c, acute_off vs TNT):')
        print(piv[stat][list(LIKE_WITH_LIKE)].round(2).to_string())

    # summary 3: full model-variant x data-variant grid of spearman_dev for two models
    for tag in ('claw-dynamics_sp0.05', 'hemibrain-wd20_sp0.05'):
        sub = tab[(tab.tag == tag) & (tab.condition == 'on') & (tab.genotype == CTRL)]
        grid = sub.pivot_table(index='model_variant', columns='data_variant',
            values='spearman_dev').loc[list(LIKE_WITH_LIKE), list(DATA_VARIANTS)]
        grid.round(3).to_csv(out / f'M1_grid_spearman_dev_{tag}_on_vs_TNT3c.csv')
        print(f'\nM1c. {tag} (APL on) vs TNT3c: Spearman of deviations from ORN, model '
            'recipe (rows) x data recipe (columns):')
        print(grid.round(2).to_string())
    return tab


def plot_m1(out: Path, tab: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7})
    tags = ['claw-dynamics_sp0.05', 'claw_sp0.05', 'hemibrain-wd20_sp0.05']
    fig, axs = plt.subplots(2, len(tags), figsize=(4.2 * len(tags), 6.4))
    for j, tag in enumerate(tags):
        for i, (cond, g) in enumerate((('on', CTRL), ('acute_off', TNT))):
            ax = axs[i, j]
            sub = tab[(tab.tag == tag) & (tab.condition == cond) & (tab.genotype == g)]
            grid = sub.pivot_table(index='model_variant', columns='data_variant',
                values='spearman_dev').loc[list(LIKE_WITH_LIKE), list(DATA_VARIANTS)]
            im = ax.imshow(grid.to_numpy(), vmin=-0.6, vmax=0.6, cmap='RdBu_r')
            for (r, c), v in np.ndenumerate(grid.to_numpy()):
                ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(v) > 0.35 else 'black')
            for k, mv in enumerate(grid.index):
                c = list(DATA_VARIANTS).index(LIKE_WITH_LIKE[mv])
                ax.add_patch(plt.Rectangle((c - 0.5, k - 0.5), 1, 1, fill=False, lw=1.6,
                    edgecolor='black'))
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels(grid.columns, rotation=60, ha='right')
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels(grid.index if j == 0 else [])
            ax.set_title(f'{tag}, {cond} vs {g}')
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02,
        label='Spearman: model vs data deviations from the ORN prediction (120 pairs)')
    fig.suptitle('E7 model side: recipe dependence of the deviation statistic '
        '(boxes = same recipe on both sides)', y=0.995)
    fig.savefig(out / 'M1_recipe_grid.png', bbox_inches='tight', dpi=150)
    fig.savefig(out / 'M1_recipe_grid.pdf', bbox_inches='tight')
    plt.close(fig)


def m2(out: Path, model_dir: Path, mats, orn) -> Optional[pd.DataFrame]:
    files = sorted(model_dir.glob('*_responses.parquet')) if model_dir.is_dir() else []
    if not files:
        print(f'M2 skipped: no *_responses.parquet under {model_dir}')
        return None
    data_binary = {g: H.recording_rdms_from_matrices(mats[g], values='binary').mean()
        for g in GENOS}
    data_null = {g: H.recording_density_null_rdms(mats[g], values='binary').mean()
        for g in GENOS}
    data_frac = {g: float(H.recording_response_fractions(mats[g], 1.0).mean().mean())
        for g in GENOS}
    rows, long = [], []
    for f in files:
        key = f.name[:-len('_responses.parquet')]
        resp = pd.read_parquet(f)
        if resp.shape[0] < resp.shape[1]:
            resp = resp.T
        resp = H.abbrev_columns(resp.astype(float))
        obs = H.pair_distances(resp, 'cosine')
        null = 1 - H.density_null_similarity(resp)
        exc = obs - null
        frac = float(resp.mean().mean())
        row = dict(model=key, response_fraction=frac, mean_obs=float(obs.mean()),
            mean_null=float(null.mean()), mean_excess=float(exc.mean()))
        bm = H.block_means(exc)
        row['excess_fruity_small'] = float(bm.get(H.FRUITY_SMALL_BLOCK, np.nan))
        row['excess_other_between'] = float(bm.drop(['within', H.FRUITY_SMALL_BLOCK],
            errors='ignore').mean())
        row['excess_within'] = float(bm.get('within', np.nan))
        for g in GENOS:
            dexc = data_binary[g] - data_null[g]
            c = H.compare_model_to_data(exc, dexc, orn=orn)
            row[f'excess_spearman_vs_{g}'] = c['spearman']
            row[f'excess_spearman_dev_vs_{g}'] = c['spearman_dev']
        rows.append(row)
        for (a, b), v in exc.items():
            long.append(dict(model=key, odor_a=a, odor_b=b, observed=obs[(a, b)],
                null=null[(a, b)], excess=v))
    for g in GENOS:
        dexc = data_binary[g] - data_null[g]
        bm = H.block_means(dexc)
        rows.append(dict(model=f'data_{g}', response_fraction=data_frac[g],
            mean_obs=float(data_binary[g].mean()), mean_null=float(data_null[g].mean()),
            mean_excess=float(dexc.mean()), excess_fruity_small=float(bm[H.FRUITY_SMALL_BLOCK]),
            excess_other_between=float(bm.drop(['within', H.FRUITY_SMALL_BLOCK]).mean()),
            excess_within=float(bm['within'])))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'M2_density_null_models.csv', index=False)
    pd.DataFrame(long).to_csv(out / 'M2_density_null_model_pairs_long.csv', index=False)
    print('\nM2. density null on model binarized responses (cosine) vs data (binary > 1):')
    print(tab.round(3).to_string())
    return tab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--model-pairs', type=Path, default=None,
        help='E1 model_pair_distances_long.csv (default: shared results/E1_apl_knockout/)')
    ap.add_argument('--model-outputs', type=Path, default=None,
        help='directory of per-KC model outputs (default: shared data/E1_model_outputs/)')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    out = H.results_dir(EXP)
    shared = H.shared_dir()
    model_pairs_csv = args.model_pairs or (shared / 'results' / 'E1_apl_knockout' /
        'model_pair_distances_long.csv')
    model_dir = args.model_outputs or (shared / 'data' / 'E1_model_outputs')

    orn = H.orn16_consensus_pairs()
    ref = H.reference_pairs()
    mats = {g: H.yang_recording_matrices(g) for g in GENOS}

    tab = m1(out, model_pairs_csv, mats, orn, ref)
    plot_m1(out, tab)
    m2(out, model_dir, mats, orn)
    print(H.APPROX_CONC_NOTE)


if __name__ == '__main__':
    main()
