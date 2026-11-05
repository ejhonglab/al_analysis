"""Analysis of APL>TNT vs control (TNT3c) KC odor-odor correlation-distance matrices,
compared against the lab's megamat17 distances in other spaces (tidy_abbrev_dists.tsv)
and the 16-odor ORN consensus means.

Run with: python3 -I apl_analysis.py
"""
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

np.random.seed(0)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 200)

REPO = '/home/claude/repos/al_analysis'
TIDY = f'{REPO}/scripts/corr_minus_orn_vs_vcf_repro/data/tidy_abbrev_dists.tsv'
CONSENSUS = f'{REPO}/al_analysis/data/sent_to_vlad/2024-06-21/consensus_mean.csv'

ctrl = pd.read_csv('TNT3c_dist_apl16.csv', index_col=0)
tnt = pd.read_csv('TNT_dist_apl16.csv', index_col=0)
assert list(ctrl.index) == list(ctrl.columns) == list(tnt.index) == list(tnt.columns)
odors16 = list(ctrl.index)
print('16 odors:', odors16)

def pairs_from_square(df):
    """Upper-triangle pairs -> Series indexed by (a, b) with a before b in df order."""
    names = list(df.index)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[(names[i], names[j])] = df.iloc[i, j]
    return pd.Series(out)

c = pairs_from_square(ctrl)
t = pairs_from_square(tnt)
assert np.allclose(ctrl.values, ctrl.values.T) and np.allclose(tnt.values, tnt.values.T)

print('\n=== A. TNT (APL output blocked) vs TNT3c (control), 120 pairs ===')
print(f'mean corr distance: control {c.mean():.3f} (corr {1-c.mean():.3f}), '
      f'TNT {t.mean():.3f} (corr {1-t.mean():.3f})')
print(f'median: control {c.median():.3f}, TNT {t.median():.3f}')
print(f'min/max control {c.min():.3f}/{c.max():.3f}; TNT {t.min():.3f}/{t.max():.3f}')
rho, p = stats.spearmanr(c, t)
r, pr = stats.pearsonr(c, t)
print(f'Spearman(control, TNT) = {rho:.3f} (p={p:.1e}); Pearson r = {r:.3f}')
# Fisher-z view of correlations
zc = np.arctanh(np.clip(1 - c, -0.999, 0.999))
zt = np.arctanh(np.clip(1 - t, -0.999, 0.999))
print(f'Pearson of Fisher-z correlations: {stats.pearsonr(zc, zt)[0]:.3f}')

# linear fit control ~ TNT; residual = pair-specific APL effect beyond global trend
slope, intercept, rfit, _, _ = stats.linregress(t, c)
pred = intercept + slope * t
resid = c - pred
print(f'OLS control = {intercept:.3f} + {slope:.3f}*TNT; R^2 = {rfit**2:.3f}; '
      f'resid SD = {resid.std():.3f}')
# Also fit on correlations (1-d) with Fisher z
slope_z, int_z, rz, _, _ = stats.linregress(zt, zc)
print(f'Fisher-z OLS z_ctrl = {int_z:.3f} + {slope_z:.3f}*z_TNT; R^2 = {rz**2:.3f}')

delta = c - t  # how much APL decorrelates each pair (control more decorrelated => positive)
print(f'\nAPL effect (control - TNT) per pair: mean {delta.mean():.3f}, SD {delta.std():.3f}, '
      f'min {delta.min():.3f}, max {delta.max():.3f}')
df = pd.DataFrame({'ctrl': c, 'tnt': t, 'delta': delta, 'resid': resid})
print('\nPairs where APL decorrelates LEAST relative to trend (resid most negative):')
print(df.sort_values('resid').head(12).round(3))
print('\nPairs where APL decorrelates MOST relative to trend (resid most positive):')
print(df.sort_values('resid').tail(12).round(3))

# Per-odor mean effect
per_odor = {}
for o in odors16:
    sel = [k for k in df.index if o in k]
    per_odor[o] = dict(mean_ctrl=df.loc[sel, 'ctrl'].mean(), mean_tnt=df.loc[sel, 'tnt'].mean(),
                       mean_delta=df.loc[sel, 'delta'].mean(), mean_resid=df.loc[sel, 'resid'].mean())
per_odor = pd.DataFrame(per_odor).T.sort_values('mean_resid')
print('\nPer-odor means (sorted by mean residual):')
print(per_odor.round(3))

# Is the APL effect proportional to baseline (i.e. delta vs control)?
print(f'\nSpearman(delta, control dist) = {stats.spearmanr(delta, c)[0]:.3f}; '
      f'Spearman(delta, TNT dist) = {stats.spearmanr(delta, t)[0]:.3f}')

# Clustering comparison
def clusters(df_sq, k):
    Z = linkage(squareform(df_sq.values, checks=False), method='average')
    return fcluster(Z, k, criterion='maxclust')
for k in (4, 5, 6, 7):
    ari = adjusted_rand_score(clusters(ctrl, k), clusters(tnt, k))
    print(f'ARI(control vs TNT clustering, k={k}) = {ari:.2f}')
for k in (5, 6):
    print(f'k={k} control clusters:', dict(zip(odors16, clusters(ctrl, k))))
    print(f'k={k} TNT clusters:    ', dict(zip(odors16, clusters(tnt, k))))

print('\n=== B. Comparison with megamat17 spaces (12 shared odors, 66 pairs) ===')
tidy = pd.read_csv(TIDY, sep='\t', index_col=[0, 1])
tidy.index.names = ['a', 'b']
vcf_seed_cols = [f'vcf5q_ND030_D03_seed{i:02d}' for i in range(1, 11)]
tidy['vcf_hyp_D3'] = tidy[vcf_seed_cols].mean(axis=1)
cols = {
    'chem_ecfp': 'chem', 'orn_remy': 'orn', 'pn_dendrites-correlation': 'pn_dend',
    'new_pn_boutons_dff': 'pn_bout', 'new_kc_claws_dff': 'kc_claw',
    'kc_remy_combo': 'kc_soma', 'new_kc_soma_nls_dff': 'kc_soma_new',
    'uniform_tom': 'model_uniform', 'hemibrain_tom': 'model_hemibrain',
    'hemibrain_wd20_tom': 'model_hemibrain_wd20', 'vcf5q_dat': 'vcf_raw',
    'vcf_hyp_D3': 'vcf_hyp',
}
shared = [o for o in odors16 if o in tidy.index.get_level_values(0)]
print('shared odors:', shared, len(shared))
rows = []
for i in range(len(shared)):
    for j in range(i + 1, len(shared)):
        a, b = shared[i], shared[j]
        if (a, b) in tidy.index:
            row = tidy.loc[(a, b), list(cols)]
        else:
            row = tidy.loc[(b, a), list(cols)]
        row = row.rename(cols)
        key = (a, b) if (a, b) in c.index else (b, a)
        row['kc_ctrl'] = c[key]
        row['kc_tnt'] = t[key]
        row.name = (a, b)
        rows.append(row)
sub = pd.DataFrame(rows).astype(float)
print(sub.shape)
sub.to_csv('shared12_pairs.csv')

def sp(x, y):
    return stats.spearmanr(sub[x], sub[y])[0]

def pe(x, y):
    return stats.pearsonr(sub[x], sub[y])[0]

print('\nSanity: control (TNT3c) vs WT KC soma megamat (kc_remy_combo): '
      f'Spearman {sp("kc_ctrl", "kc_soma"):.3f}, Pearson {pe("kc_ctrl", "kc_soma"):.3f}')
print('        TNT vs WT KC soma: '
      f'Spearman {sp("kc_tnt", "kc_soma"):.3f}, Pearson {pe("kc_tnt", "kc_soma"):.3f}')
print(f'        control vs kc_soma_new: Spearman {sp("kc_ctrl", "kc_soma_new"):.3f}')
print(f'mean dist on 66 pairs: orn {sub.orn.mean():.3f}, pn_bout {sub.pn_bout.mean():.3f}, '
      f'kc_claw {sub.kc_claw.mean():.3f}, kc_soma(WT) {sub.kc_soma.mean():.3f}, '
      f'ctrl {sub.kc_ctrl.mean():.3f}, tnt {sub.kc_tnt.mean():.3f}')

print('\nSpearman rho of each KC dataset with upstream stages / models (66 pairs):')
targets = ['orn', 'pn_dend', 'pn_bout', 'kc_claw', 'kc_soma', 'model_uniform',
           'model_hemibrain_wd20', 'chem', 'vcf_raw', 'vcf_hyp']
tab = pd.DataFrame({k: {tg: sp(k, tg) for tg in targets}
                    for k in ['kc_tnt', 'kc_ctrl', 'kc_soma', 'kc_claw', 'pn_bout', 'orn']})
print(tab.round(3))
print('\nPearson r version:')
tabp = pd.DataFrame({k: {tg: pe(k, tg) for tg in targets}
                     for k in ['kc_tnt', 'kc_ctrl', 'kc_soma', 'kc_claw', 'pn_bout', 'orn']})
print(tabp.round(3))

# Bootstrap over pairs for key differences
def boot_diff(fn, x1, x2, y, n=5000):
    idx = np.arange(len(sub))
    out = []
    for _ in range(n):
        s = np.random.choice(idx, len(idx), replace=True)
        d = sub.iloc[s]
        out.append(fn(d[x1], d[y])[0] - fn(d[x2], d[y])[0])
    out = np.array(out)
    return out.mean(), np.percentile(out, [5, 95])

for y in ['orn', 'kc_claw', 'pn_bout', 'chem', 'vcf_hyp', 'vcf_raw', 'model_hemibrain_wd20']:
    m, ci = boot_diff(stats.spearmanr, 'kc_tnt', 'kc_ctrl', y)
    mp, cip = boot_diff(stats.pearsonr, 'kc_tnt', 'kc_ctrl', y)
    print(f'rho(TNT,{y}) - rho(ctrl,{y}): Spearman {m:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]; '
          f'Pearson {mp:+.3f} [{cip[0]:+.3f},{cip[1]:+.3f}]')

# Remapping vector (KC soma WT minus linear prediction from ORN) vs APL effect vector
def resid_on(y, x):
    s_, i_, *_ = stats.linregress(sub[x], sub[y])
    return sub[y] - (i_ + s_ * sub[x])

remap_wt = resid_on('kc_soma', 'orn')
remap_ctrl = resid_on('kc_ctrl', 'orn')
remap_tnt = resid_on('kc_tnt', 'orn')
apl_eff = resid_on('kc_ctrl', 'kc_tnt')      # pair-specific APL effect (beyond global scaling)
apl_raw = sub.kc_ctrl - sub.kc_tnt
print('\nResidual structure (66 pairs):')
print(f'SD of ORN->KC residual: WT {remap_wt.std():.3f}, ctrl {remap_ctrl.std():.3f}, TNT {remap_tnt.std():.3f}')
print(f'Spearman(remap_ctrl, remap_tnt) = {stats.spearmanr(remap_ctrl, remap_tnt)[0]:.3f} '
      f'(do the same pairs deviate from ORN with and without APL?)')
print(f'Spearman(remap_wt, remap_tnt) = {stats.spearmanr(remap_wt, remap_tnt)[0]:.3f}; '
      f'Spearman(remap_wt, remap_ctrl) = {stats.spearmanr(remap_wt, remap_ctrl)[0]:.3f}')
print(f'Spearman(APL pair-specific effect, WT remap) = {stats.spearmanr(apl_eff, remap_wt)[0]:.3f} '
      f'(p={stats.spearmanr(apl_eff, remap_wt)[1]:.2g})')
print(f'Spearman(APL raw effect ctrl-TNT, WT KC-ORN raw) = '
      f'{stats.spearmanr(apl_raw, sub.kc_soma - sub.orn)[0]:.3f}')
print(f'Spearman(APL pair-specific effect, VCF hyp) = {stats.spearmanr(apl_eff, sub.vcf_hyp)[0]:.3f} '
      f'(p={stats.spearmanr(apl_eff, sub.vcf_hyp)[1]:.2g}); vs vcf_raw '
      f'{stats.spearmanr(apl_eff, sub.vcf_raw)[0]:.3f}; vs chem {stats.spearmanr(apl_eff, sub.chem)[0]:.3f}')
print(f'Spearman(ORN->KC residual, VCF hyp): WT {stats.spearmanr(remap_wt, sub.vcf_hyp)[0]:.3f}, '
      f'ctrl {stats.spearmanr(remap_ctrl, sub.vcf_hyp)[0]:.3f}, TNT {stats.spearmanr(remap_tnt, sub.vcf_hyp)[0]:.3f}')
print(f'Pearson(ORN->KC residual, VCF hyp): WT {stats.pearsonr(remap_wt, sub.vcf_hyp)[0]:.3f}, '
      f'ctrl {stats.pearsonr(remap_ctrl, sub.vcf_hyp)[0]:.3f}, TNT {stats.pearsonr(remap_tnt, sub.vcf_hyp)[0]:.3f}')

# How much of the remapping (KC vs ORN rank disagreement) survives APL block?
print('\nRank-order agreement with ORN: ctrl %.3f, TNT %.3f, WT KC %.3f, claws %.3f, boutons %.3f, '
      'uniform model %.3f, hemibrain wd20 %.3f' % (
          sp('kc_ctrl', 'orn'), sp('kc_tnt', 'orn'), sp('kc_soma', 'orn'), sp('kc_claw', 'orn'),
          sp('pn_bout', 'orn'), sp('model_uniform', 'orn'), sp('model_hemibrain_wd20', 'orn')))

sub['remap_wt'] = remap_wt
sub['remap_ctrl'] = remap_ctrl
sub['remap_tnt'] = remap_tnt
sub['apl_eff'] = apl_eff
print('\nTop pairs by WT remapping (KC more similar than ORN predicts; most negative resid):')
print(sub.sort_values('remap_wt')[['orn', 'kc_soma', 'kc_ctrl', 'kc_tnt', 'remap_wt', 'remap_tnt', 'apl_eff', 'vcf_hyp']].head(10).round(3))
print('\nTop pairs by WT remapping (KC less similar than ORN predicts; most positive resid):')
print(sub.sort_values('remap_wt')[['orn', 'kc_soma', 'kc_ctrl', 'kc_tnt', 'remap_wt', 'remap_tnt', 'apl_eff', 'vcf_hyp']].tail(10).round(3))

print('\n=== C. 16-odor ORN consensus matrix vs TNT / control (120 pairs) ===')
cons = pd.read_csv(CONSENSUS)
cons['abbrev'] = cons['odor1'].str.split(' @ ').str[0]
cons['conc'] = cons['odor1'].str.split(' @ ').str[1].astype(float)
want = {'2h': 'megamat', 'IaA': 'megamat', 'pa': 'megamat', '2-but': 'megamat', 'eb': 'megamat',
        'ep': 'megamat', '6al': 'megamat', 't2h': 'megamat', '1-8ol': 'megamat', '1-5ol': 'megamat',
        '1-6ol': 'megamat', 'ms': 'megamat', 'PAA': 'validation2', '-aPine': 'validation2',
        'euc': 'validation2', 'B-myr': 'validation2'}
glom_cols = [x for x in cons.columns if x not in ('panel', 'odor1', 'abbrev', 'conc')]
rows = {}
for ab, panel in want.items():
    r_ = cons[(cons.abbrev == ab) & (cons.panel == panel)]
    assert len(r_) == 1, (ab, panel, len(r_))
    rows['paa' if ab == 'PAA' else ab] = r_.iloc[0][glom_cols].astype(float)
orn16 = pd.DataFrame(rows).T.loc[odors16]
print('ORN consensus concentrations used:',
      {ab: float(cons[(cons.abbrev == ab) & (cons.panel == p)].conc.iloc[0]) for ab, p in want.items()})
# drop glomeruli that are all-zero / NaN across these odors
orn16 = orn16.loc[:, (orn16.fillna(0) != 0).any(axis=0)]
print('glomeruli retained:', orn16.shape[1], 'NaNs:', int(orn16.isna().sum().sum()))
orn16 = orn16.fillna(0)
orn_corr = np.corrcoef(orn16.values)
orn_dist16 = pd.DataFrame(1 - orn_corr, index=odors16, columns=odors16)
o16 = pairs_from_square(orn_dist16)
# compare 12-odor subset of consensus-derived ORN dists with orn_remy from tidy
o12 = o16[[k for k in o16.index if k[0] in shared and k[1] in shared]]
o12 = o12.reindex([k if k in o12.index else (k[1], k[0]) for k in sub.index])
print(f'consensus-ORN (12 odors) vs tidy orn_remy: Spearman {stats.spearmanr(o12.values, sub.orn.values)[0]:.3f}, '
      f'Pearson {stats.pearsonr(o12.values, sub.orn.values)[0]:.3f}')
print(f'120 pairs: Spearman(ORN16, control) = {stats.spearmanr(o16, c)[0]:.3f}; '
      f'Spearman(ORN16, TNT) = {stats.spearmanr(o16, t)[0]:.3f}')
print(f'120 pairs: Pearson(ORN16, control) = {stats.pearsonr(o16, c)[0]:.3f}; '
      f'Pearson(ORN16, TNT) = {stats.pearsonr(o16, t)[0]:.3f}')
# bootstrap difference
idx = np.arange(120)
diffs = []
for _ in range(5000):
    s = np.random.choice(idx, 120, replace=True)
    diffs.append(stats.spearmanr(o16.values[s], t.values[s])[0] - stats.spearmanr(o16.values[s], c.values[s])[0])
diffs = np.array(diffs)
print(f'rho(ORN16,TNT) - rho(ORN16,ctrl): {diffs.mean():+.3f} 90% CI [{np.percentile(diffs,5):+.3f}, {np.percentile(diffs,95):+.3f}]')
# residual structure on 120 pairs
def resid_series(y, x):
    s_, i_, *_ = stats.linregress(x, y)
    return y - (i_ + s_ * x)
rm_c = resid_series(c, o16)
rm_t = resid_series(t, o16)
ae = resid_series(c, t)
print(f'120 pairs: Spearman(ORN->ctrl resid, ORN->TNT resid) = {stats.spearmanr(rm_c, rm_t)[0]:.3f}; '
      f'Spearman(APL pair-specific effect, ORN->ctrl resid) = {stats.spearmanr(ae, rm_c)[0]:.3f}; '
      f'Spearman(APL pair-specific effect, ORN->TNT resid) = {stats.spearmanr(ae, rm_t)[0]:.3f}')
big = pd.DataFrame({'orn16': o16, 'ctrl': c, 'tnt': t, 'resid_ctrl_vs_orn': rm_c,
                    'resid_tnt_vs_orn': rm_t, 'apl_eff': ae})
big.to_csv('all120_pairs.csv')
print('\nPairs with largest convergence in KCs (ctrl) relative to ORN16 (most negative residual):')
print(big.sort_values('resid_ctrl_vs_orn').head(12).round(3))
print('\nPairs with largest divergence in KCs (ctrl) relative to ORN16:')
print(big.sort_values('resid_ctrl_vs_orn').tail(12).round(3))
for k in (4, 5, 6):
    print(f'ARI k={k}: ORN16 vs ctrl {adjusted_rand_score(clusters(orn_dist16, k), clusters(ctrl, k)):.2f}; '
          f'ORN16 vs TNT {adjusted_rand_score(clusters(orn_dist16, k), clusters(tnt, k)):.2f}; '
          f'ctrl vs TNT {adjusted_rand_score(clusters(ctrl, k), clusters(tnt, k)):.2f}')
print('ORN16 clusters k=6:', dict(zip(odors16, clusters(orn_dist16, 6))))
orn_dist16.round(3).to_csv('orn16_consensus_corrdist.csv')
