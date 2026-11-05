"""Does a non-specific (monotone / uniform-sparsening) transform explain control from TNT?
Run with python3 -I null_models.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

ctrl = pd.read_csv('TNT3c_dist_apl16.csv', index_col=0)
tnt = pd.read_csv('TNT_dist_apl16.csv', index_col=0)
names = list(ctrl.index)
iu = np.triu_indices(len(names), 1)
c = ctrl.values[iu]
t = tnt.values[iu]
pairs = [(names[i], names[j]) for i, j in zip(*iu)]

# 1) linear
sl, ic, *_ = stats.linregress(t, c)
res_lin = c - (ic + sl * t)
# 2) isotonic (any monotone map)
iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(t, c)
res_iso = c - iso.predict(t)
print(f'linear: R^2 {1 - res_lin.var()/c.var():.3f}, resid SD {res_lin.std():.4f}')
print(f'isotonic: R^2 {1 - res_iso.var()/c.var():.3f}, resid SD {res_iso.std():.4f}')
coef = np.polyfit(t, c, 2)
res_q = c - np.polyval(coef, t)
print(f'quadratic: R^2 {1 - res_q.var()/c.var():.3f}, resid SD {res_q.std():.4f}; coefs {coef}')

# 3) Uniform-sparsening null: both datasets are thresholded versions of the same latent
# bivariate-Gaussian drive with correlation rho_in; response prob p_tnt (APL off) and
# p_ctrl (APL on). phi(rho_in, p) = (P11 - p^2)/(p(1-p)), P11 = P(X>z, Y>z), z = Phi^-1(1-p).
xg = np.linspace(-8, 8, 4001)
dx = xg[1] - xg[0]
pdf = norm.pdf(xg)

def phi_table(p, rhos):
    z = norm.ppf(1 - p)
    mask = xg > z
    out = np.empty_like(rhos)
    for k, r in enumerate(rhos):
        s = np.sqrt(max(1 - r * r, 1e-12))
        tail = 1 - norm.cdf((z - r * xg[mask]) / s)
        p11 = np.sum(pdf[mask] * tail) * dx
        out[k] = (p11 - p * p) / (p * (1 - p))
    return out

rhos = np.linspace(-0.95, 0.999, 300)
corr_c = 1 - c
corr_t = 1 - t
best = (np.inf, None, None)
for p_t in np.arange(0.12, 0.61, 0.01):
    tab_t = phi_table(p_t, rhos)
    rho_in = np.interp(corr_t, tab_t, rhos)   # tab_t monotone increasing in rho
    for p_c in np.arange(0.03, p_t, 0.01):
        tab_c = phi_table(p_c, rhos)
        pred = np.interp(rho_in, rhos, tab_c)
        mse = np.mean((corr_c - pred) ** 2)
        if mse < best[0]:
            best = (mse, p_t, p_c, pred.copy())
mse, p_tnt, p_ctrl, pred_c = best
res_g = corr_c - pred_c
print(f'thresholded-Gaussian uniform-sparsening null: fitted p_TNT={p_tnt:.2f}, p_ctrl={p_ctrl:.2f}; '
      f'R^2 {1 - res_g.var()/corr_c.var():.3f}, resid SD {res_g.std():.4f}; '
      f'mean corr actual ctrl {corr_c.mean():.3f}, predicted {pred_c.mean():.3f}')
# also: what does the model predict for shape? (print mapping at a few corr_t values)
for ct in (0.1, 0.3, 0.5, 0.7, 0.8):
    rin = np.interp(ct, phi_table(p_tnt, rhos), rhos)
    print(f'  corr_TNT={ct:.1f} -> latent rho_in={rin:.2f} -> predicted corr_ctrl={np.interp(rin, rhos, phi_table(p_ctrl, rhos)):.3f}')

df = pd.DataFrame({'ctrl': c, 'tnt': t, 'res_lin': res_lin, 'res_iso': res_iso, 'res_gauss': -res_g},
                  index=pd.MultiIndex.from_tuples(pairs))
print('\nSpearman between residual definitions: lin-iso %.2f, lin-gauss %.2f, iso-gauss %.2f' % (
    stats.spearmanr(res_lin, res_iso)[0], stats.spearmanr(res_lin, -res_g)[0], stats.spearmanr(res_iso, -res_g)[0]))
print('\nMost negative Gaussian-null residuals (APL spares these pairs beyond uniform sparsening):')
print(df.sort_values('res_gauss').head(10).round(3))
print('\nMost positive Gaussian-null residuals (APL decorrelates these pairs beyond uniform sparsening):')
print(df.sort_values('res_gauss').tail(14).round(3))
df.to_csv('null_model_residuals.csv')

groups = {'2h': 'fruity', 'IaA': 'fruity', 'pa': 'fruity', '2-but': 'small_ester', 'eb': 'small_ester',
          'ep': 'small_ester', '6al': 'ald', 't2h': 'ald', '1-8ol': 'alc', '1-5ol': 'alc', '1-6ol': 'alc',
          'ms': 'ms', 'paa': 'paa', '-aPine': 'terp', 'euc': 'terp', 'B-myr': 'terp'}
df['g1'] = [groups[a] for a, b in pairs]
df['g2'] = [groups[b] for a, b in pairs]
df['within'] = df.g1 == df.g2
print('\nResiduals by within/between chemical group:')
print(df.groupby('within')[['res_lin', 'res_iso', 'res_gauss']].agg(['mean', 'std', 'count']).round(3))
blk = df[~df.within].copy()
blk['pair'] = ['-'.join(sorted([a, b])) for a, b in zip(blk.g1, blk.g2)]
print('\nBetween-group block mean residuals (sorted by Gaussian-null residual):')
print(blk.groupby('pair')[['res_gauss', 'res_iso', 'ctrl', 'tnt']].mean().sort_values('res_gauss').round(3))
# t-test of block means vs zero for the fruity-small_ester block
fs = blk[blk.pair == 'fruity-small_ester']
print(f'\nfruity-small_ester block: n={len(fs)}, mean Gaussian-null resid {fs.res_gauss.mean():.3f} '
      f'(SD {fs.res_gauss.std():.3f}), other between-group pairs mean {blk[blk.pair != "fruity-small_ester"].res_gauss.mean():.3f}; '
      f'Mann-Whitney p={stats.mannwhitneyu(fs.res_gauss, blk[blk.pair != "fruity-small_ester"].res_gauss).pvalue:.2g}')
