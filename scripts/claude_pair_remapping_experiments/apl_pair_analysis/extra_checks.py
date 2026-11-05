import numpy as np, pandas as pd
from scipy import stats
sub = pd.read_csv('shared12_pairs.csv', index_col=[0, 1])
big = pd.read_csv('all120_pairs.csv', index_col=[0, 1])
nm = pd.read_csv('null_model_residuals.csv', index_col=[0, 1])

def resid(y, x):
    s, i, *_ = stats.linregress(x, y)
    return y - (i + s * x)

# model deviations from ORN vs data deviations from ORN (66 pairs)
r_wt = resid(sub.kc_soma, sub.orn)
r_hb = resid(sub.model_hemibrain_wd20, sub.orn)
r_un = resid(sub.model_uniform, sub.orn)
r_ct = resid(sub.kc_ctrl, sub.orn)
r_tn = resid(sub.kc_tnt, sub.orn)
print('Spearman(model-vs-ORN residual, data-vs-ORN residual), 66 pairs:')
for nm_, r in [('hemibrain_wd20', r_hb), ('uniform', r_un)]:
    print(f'  {nm_}: vs WT {stats.spearmanr(r, r_wt)[0]:.3f}, vs ctrl {stats.spearmanr(r, r_ct)[0]:.3f}, vs TNT {stats.spearmanr(r, r_tn)[0]:.3f}')
print(f'SD of model-vs-ORN residual: hemibrain {r_hb.std():.3f}, uniform {r_un.std():.3f}; data WT {r_wt.std():.3f}')
print(f'Spearman(model, WT KC): uniform {stats.spearmanr(sub.model_uniform, sub.kc_soma)[0]:.3f}, hemibrain {stats.spearmanr(sub.model_hemibrain_wd20, sub.kc_soma)[0]:.3f}')
print(f'mean dist: uniform {sub.model_uniform.mean():.3f}, hemibrain {sub.model_hemibrain_wd20.mean():.3f}, WT KC {sub.kc_soma.mean():.3f}')

# residual variance fractions
print(f'\n12 odors: resid SD vs ORN ctrl {r_ct.std():.3f}, TNT {r_tn.std():.3f}; APL share of ctrl residual variance '
      f'{1 - (r_tn.var()/r_ct.var()):.2f}')
rc = resid(big.ctrl, big.orn16); rt = resid(big.tnt, big.orn16)
print(f'16 odors: resid SD vs ORN ctrl {rc.std():.3f}, TNT {rt.std():.3f}; APL share {1 - rt.var()/rc.var():.2f}')
# closing of the gap to a feedforward (rank-preserving) model
for lab, a, b in [('16 odors', stats.spearmanr(big.orn16, big.ctrl)[0], stats.spearmanr(big.orn16, big.tnt)[0]),
                  ('12 odors', stats.spearmanr(sub.orn, sub.kc_ctrl)[0], stats.spearmanr(sub.orn, sub.kc_tnt)[0])]:
    print(f'{lab}: rho(ORN,ctrl)={a:.3f}, rho(ORN,TNT)={b:.3f}, gap to 0.99 closed by APL block: {(b-a)/(0.99-a):.2f}')

# per-odor APL effect (isotonic residual) vs ORN drive
cons = pd.read_csv('/home/claude/repos/al_analysis/al_analysis/data/sent_to_vlad/2024-06-21/consensus_mean.csv')
cons['abbrev'] = cons['odor1'].str.split(' @ ').str[0].replace({'PAA': 'paa'})
want = {'2h': 'megamat', 'IaA': 'megamat', 'pa': 'megamat', '2-but': 'megamat', 'eb': 'megamat', 'ep': 'megamat',
        '6al': 'megamat', 't2h': 'megamat', '1-8ol': 'megamat', '1-5ol': 'megamat', '1-6ol': 'megamat', 'ms': 'megamat',
        'paa': 'validation2', '-aPine': 'validation2', 'euc': 'validation2', 'B-myr': 'validation2'}
gl = [c for c in cons.columns if c not in ('panel', 'odor1', 'abbrev')]
drive = {}
for ab, p in want.items():
    r = cons[(cons.abbrev == ab) & (cons.panel == p)].iloc[0][gl].astype(float).fillna(0)
    drive[ab] = dict(mean=r.mean(), n_strong=(r > 0.5).sum(), top=r.max())
drive = pd.DataFrame(drive).T
per = {}
for o in drive.index:
    sel = [k for k in nm.index if o in k]
    per[o] = dict(iso=nm.loc[sel, 'res_iso'].mean(), gauss=nm.loc[sel, 'res_gauss'].mean(), lin=nm.loc[sel, 'res_lin'].mean())
per = pd.DataFrame(per).T.join(drive)
print('\nPer-odor APL pair-specific effect vs ORN drive:')
print(per.sort_values('iso').round(3))
for col in ('mean', 'n_strong', 'top'):
    print(f'Spearman(per-odor isotonic resid, ORN {col}) = {stats.spearmanr(per.iso, per[col])[0]:.2f} (p={stats.spearmanr(per.iso, per[col])[1]:.2g}); '
          f'gauss: {stats.spearmanr(per.gauss, per[col])[0]:.2f}')
# ORN overlap between the two ester groups vs others
orn16 = pd.read_csv('orn16_consensus_corrdist.csv', index_col=0)
print('\nORN16 corr distance, fruity x small_ester block mean:',
      round(orn16.loc[['2h', 'IaA', 'pa'], ['2-but', 'eb', 'ep']].values.mean(), 3),
      '; fruity x alc:', round(orn16.loc[['2h', 'IaA', 'pa'], ['1-8ol', '1-5ol', '1-6ol']].values.mean(), 3),
      '; alc x terp:', round(orn16.loc[['1-8ol', '1-5ol', '1-6ol'], ['-aPine', 'euc', 'B-myr']].values.mean(), 3),
      '; ald x terp:', round(orn16.loc[['6al', 't2h'], ['-aPine', 'euc', 'B-myr']].values.mean(), 3))
# isotonic residual for selected pairs
for k in [('1-5ol', '1-6ol'), ('eb', 'ep'), ('6al', 't2h'), ('-aPine', 'euc'), ('IaA', 'pa'), ('2h', 'IaA'), ('2h', 'pa')]:
    print(k, nm.loc[k].round(3).to_dict())
