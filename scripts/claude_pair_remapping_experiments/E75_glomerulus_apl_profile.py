#!/usr/bin/env python3
"""E75: where APL acts, by glomerulus. The profile of APL inhibition over glomeruli that the per-pair APL effect asks for,
at the cell, the claw and the bouton, on the breadth-2 APL-effect fit (C2, E67) across wirings and against the connectomes.

The error the APL-effect fit on re-drawn wirings (E72) left is common to every wiring
(results/E72_wiring_population/common_error_pairs.csv): the model's per-pair APL effect is too small among 2h, IaA, pa,
eb, ep and ms and too large among 1-8ol, 1-5ol, 1-6ol and 6al. A wiring draw keeps each glomerulus's claws per KC type,
so a rule written per glomerulus is the one wiring-independent way to say which KCs APL acts on.

A profile is a score s_g per glomerulus applied at one site:
- 'cell': every claw's APL->KC weight of a KC times exp(sum over g of s_g x its number of claws from g) (E15's / E17's
  log-linear form), on top of the breadth weighting;
- 'claw': the APL->KC weight of each claw from g times exp(s_g);
- 'bouton': the APL->bouton weight of each bouton of g times exp(s_g).
The factor is divided by its mean (over claws, or over boutons), so the fitted scales keep their meaning.

Parts (CLI `--part`):
- A: run by `E43_model_trial_noise.py` (committed code; runner outputs/E75_glomerulus_apl_profile/run_viki_e75_partA.py);
  `--part A-score` scores it as E72's part E (`E72_wiring_population.part_e_score` on this experiment's partB/).
- B: no simulations: every archived variant's pattern error against E72's common error (`archive_vs_common`).
- C: the sensitivity map on C2 at its fitted parameters on the real wiring (`sensitivity_map`): known sets, then one
  glomerulus at a time at each site, the APL->KC scale bisected to C2's number of APL-on responses.
- D1: the linear profile from the map, cross-validated over odors and over recording halves, with the gate (`linear_profile`).
- D2: the chosen profiles simulated (`simulate_profiles`).
- F: the profile against the connectomes' APL densities and the glomeruli's drive (`anatomy`).
Fits with a profile (part D3) go through `E10_joint_fit.py --apl-glom-profile FILE`.

Outputs: results/E75_glomerulus_apl_profile/ (archive_vs_common_error.csv, sensitivity_map.csv and sensitivity_pairs.csv,
profiles.csv, profile_cv.csv, anatomy.csv, figures, provenance); simulated outputs under outputs/E75_glomerulus_apl_profile/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E75_glomerulus_apl_profile'
SITES = ('cell', 'claw', 'bouton')
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
E15_EIGHT = ('DM2', 'VM2', 'DM1', 'DM6', 'VM7d', 'DM4', 'VM3', 'VA2')
C2_FOUR = ('DM3', 'DM2', 'DM5', 'DM6')
ESTER_GROUP = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL_GROUP = ('1-8ol', '1-5ol', '1-6ol', '6al')


def claw_glomeruli(W: pd.DataFrame) -> pd.Series:
    """Each claw's glomerulus (from the claw x bouton wiring `W`, one bouton per claw), indexed as W's rows."""
    M = W.to_numpy() != 0
    if not (M.sum(axis=1) == 1).all():
        raise ValueError('expected exactly one bouton per claw')
    g = W.columns.get_level_values('glomerulus').to_numpy()[M.argmax(axis=1)]
    return pd.Series(g, index=W.index, name='glomerulus')


def profile_factors(site: str, scores: Dict[str, float], W: pd.DataFrame, wAPLKC: pd.Series, wAPLPN: pd.Series) -> pd.Series:
    """The multiplicative factor a profile puts on the APL->KC weights (site 'cell' or 'claw', indexed as `wAPLKC`) or on
    the APL->bouton weights (site 'bouton', indexed as `wAPLPN`), divided by its mean. All scores 0 gives exactly 1."""
    s = {str(k): float(v) for k, v in scores.items() if float(v) != 0.0}
    if site == 'bouton':
        g = wAPLPN.index.get_level_values('glomerulus')
        f = np.exp(np.array([s.get(x, 0.0) for x in g]))
        idx = wAPLPN.index
    elif site in ('claw', 'cell'):
        cg = claw_glomeruli(W).reindex(wAPLKC.index)
        if cg.isna().any():
            raise ValueError('claws of wAPLKC missing from W')
        per_claw = np.array([s.get(x, 0.0) for x in cg.to_numpy()])
        if site == 'claw':
            f = np.exp(per_claw)
        else:
            kc = wAPLKC.index.get_level_values(H.KC_ID_LEVEL)
            per_kc = pd.Series(per_claw, index=kc).groupby(level=0).sum()
            f = np.exp(per_kc.reindex(kc).to_numpy())
        idx = wAPLKC.index
    else:
        raise ValueError(site)
    if not s:
        return pd.Series(np.ones(len(idx)), index=idx)
    return pd.Series(f / f.mean(), index=idx)


def apply_profile(site: str, scores: Dict[str, float], W: pd.DataFrame, wAPLKC: pd.Series, wAPLPN: pd.Series
                  ) -> Tuple[pd.Series, pd.Series]:
    """(APL->KC weights, APL->bouton weights) with the profile applied at `site`; the other set unchanged."""
    f = profile_factors(site, scores, W, wAPLKC, wAPLPN)
    if site == 'bouton':
        return wAPLKC, wAPLPN * f
    return wAPLKC * f, wAPLPN


def pair_mask(pairs: pd.Index, group: Sequence[str]) -> np.ndarray:
    """Pairs with both odors in `group`."""
    a = pairs.get_level_values(0)
    b = pairs.get_level_values(1)
    return np.asarray(a.isin(group) & b.isin(group))


def common_error(readout: str = 'cal') -> pd.Series:
    """E72's per-pair error common to the eight wirings (model minus data, means removed), by (odor_a, odor_b)."""
    name = 'common_error_pairs.csv' if readout == 'cal' else 'common_error_pairs_raw.csv'
    d = pd.read_csv(H.results_dir('E72_wiring_population') / name)
    return d.set_index(['odor_a', 'odor_b'])['common'].sort_index()


def error_vs_common(err: pd.Series, readout: str) -> Dict[str, float]:
    """A pattern error (model minus data, means removed, by pair) against the common error: correlation, projection
    (1 = as much of it as the eight wirings' mean has), the ester and alcohol groups' mean errors, its rms."""
    c = common_error(readout)
    e = err.reindex(c.index)
    if e.isna().any():
        e = err.copy()
        e.index = pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in e.index])
        c2 = c.copy()
        c2.index = pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in c2.index])
        e = e.reindex(c2.index)
        c = c2
    ev, cv = e.to_numpy(dtype=float), c.to_numpy(dtype=float)
    ok = np.isfinite(ev)
    return dict(corr_common=float(np.corrcoef(ev[ok], cv[ok])[0, 1]), proj_common=float(ev[ok] @ cv[ok] / (cv[ok] @ cv[ok])),
                err_ester=float(np.nanmean(ev[pair_mask(c.index, ESTER_GROUP)])),
                err_alcohol=float(np.nanmean(ev[pair_mask(c.index, ALCOHOL_GROUP)])),
                err_rms=float(np.sqrt(np.nanmean(ev ** 2))))


def archive_vs_common(out: Path) -> pd.DataFrame:
    """Part B: for every archived variant with both conditions, its per-pair pattern error on raw counts (and under the
    calibrated readout where spike counts are archived) against the common error: correlation, projection, the ester
    and alcohol groups' errors. Writes archive_vs_common_error.csv."""
    import E72_wiring_population as E72
    data_raw = E72.data_for('raw')
    rows = []
    for folder in sorted(H.shared_dir().joinpath('results').glob('E*')):
        lp = folder / 'model_pair_distances_long.csv'
        if not lp.exists():
            continue
        d = pd.read_csv(lp)
        if not {'tag', 'condition', 'odor_a', 'odor_b', 'distance'} <= set(d.columns):
            continue
        d = d.drop_duplicates(['tag', 'condition', 'odor_a', 'odor_b'], keep='last')
        for tag, g in d.groupby('tag'):
            conds = set(g.condition)
            if not {'on', 'acute_off'} <= conds:
                continue
            on = g[g.condition == 'on'].set_index(['odor_a', 'odor_b'])['distance']
            off = g[g.condition == 'acute_off'].set_index(['odor_a', 'odor_b'])['distance']
            try:
                parts = H.apl_effect_pair_parts(on, off, data_raw)
            except Exception as e:
                print(f'{folder.name}:{tag}: {e}')
                continue
            row = dict(folder=folder.name, tag=tag, n_pairs=len(parts),
                       **{f'raw_{k}': v for k, v in error_vs_common(parts['error'], 'raw').items()})
            o = E72.load_fit(folder.name, tag)
            if o is not None:
                try:
                    dist = E72.distances(o, 'cal')
                    row.update({f'cal_{k}': v for k, v in error_vs_common(E72.pattern_error(dist, 'cal'), 'cal').items()})
                except Exception as e:
                    print(f'{folder.name}:{tag} (calibrated): {e}')
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'archive_vs_common_error.csv', index=False)
    print(f'{len(res)} variants; {int(res.filter(like="cal_").notna().any(axis=1).sum()) if len(res) else 0} with spike counts')
    return res


def split_scores(site: str, scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """{site: {glomerulus: score}} of a profile: one site's scores as they are, or a multi-site profile ('a+b') whose
    scores are keyed 'site:glomerulus'."""
    if '+' not in site:
        return {site: dict(scores)}
    out: Dict[str, Dict[str, float]] = {s: {} for s in site.split('+')}
    for k, v in scores.items():
        s, g = k.split(':', 1)
        out[s][g] = float(v)
    return out


class ProfileModel:
    """C2 (the breadth-2 APL-effect fit, E67) at its fitted parameters on the real wiring, as a fixed-parameter simulator
    whose APL->KC and APL->bouton weights a profile changes in place (the simulator's own unscaled weight arrays, checked
    against the weights it was built with). `run_on(x)` runs APL on with the APL->KC, KC->APL and bouton->APL scales times x
    (the fits' tie; APL->bouton and the somatic threshold unchanged); the APL-off run does not depend on APL's weights and
    is C2's own."""

    def __init__(self, out_dir: Path):
        import E10_joint_fit as J
        import E72_wiring_population as E72
        from al_analysis.mb_model import make_fixed_param_kc_simulator
        self.orn = H.apl16_orn_deltas()
        self.p = E72.fitted_params(*C2)
        _, wnorm, W, breadth = J.bouton_base(self.orn, out_dir, None)
        self.W = W
        self.wAPLKC = J.breadth_weights(wnorm['wAPLKC'], breadth, 2.0)
        self.wAPLPN = wnorm['wAPLPN']
        kws = dict(J.base_kws(None), _wAPLKC=self.wAPLKC, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=self.wAPLPN,
                   _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=self.p['wKCKC_scale'], kc_kc_tau=J.TAU)
        self.sim, _ = make_fixed_param_kc_simulator(self.orn, initial_params=dict(self.p), **kws)
        cells = dict(zip(self.sim.__code__.co_freevars, self.sim.__closure__))
        self._kc = cells['wAPLKC_unscaled'].cell_contents
        self._pn = cells['wAPLPN_unscaled'].cell_contents
        self._kc0, self._pn0 = self._kc.copy(), self._pn.copy()
        if not (np.array_equal(self._kc0.ravel(), self.wAPLKC.to_numpy()) and np.array_equal(self._pn0.ravel(), self.wAPLPN.to_numpy())):
            raise RuntimeError("the simulator's unscaled APL weights are not the ones it was built with")
        saved = E72.load_fit(*C2)
        r_on, s_on = self.sim(dict(self.p))
        r_off, s_off = self.sim(dict(self.p, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
        for (r, s), (rs, ss), c in (((r_on, s_on), saved['on'], 'on'), ((r_off, s_off), saved['off'], 'off')):
            n = int((s.to_numpy(dtype=float) != ss.to_numpy(dtype=float)).sum())
            if n or not s.index.equals(ss.index):
                raise RuntimeError(f"C2's {c} run differs from its saved outputs in {n} counts")
        self.base = {'on': (r_on, s_on), 'off': (r_off, s_off)}
        self.target_n = int((r_on.to_numpy(dtype=float) > 0.5).sum())
        self.size = int(r_on.size)
        self.site, self.scores = None, {}

    def set_profile(self, site: Optional[str], scores: Dict[str, float]) -> None:
        """Set the APL weights to the base's times a profile (`site` one of SITES, or several joined by '+', whose scores
        are keyed 'site:glomerulus'); None or no scores restores the base's weights."""
        self._kc[...] = self._kc0
        self._pn[...] = self._pn0
        self.site, self.scores = site, dict(scores)
        if site is None or not scores:
            return
        fkc = np.ones(self._kc0.size)
        fpn = np.ones(self._pn0.size)
        for st, sc in split_scores(site, scores).items():
            f = profile_factors(st, sc, self.W, self.wAPLKC, self.wAPLPN).to_numpy()
            if st == 'bouton':
                fpn = fpn * f
            else:
                fkc = fkc * f
        self._pn[...] = self._pn0 * fpn.reshape(self._pn0.shape)
        self._kc[...] = self._kc0 * fkc.reshape(self._kc0.shape)

    def run_on(self, x: float):
        q = dict(self.p, wAPLKC_scale=self.p['wAPLKC_scale'] * x, wKCAPL_scale=self.p['wKCAPL_scale'] * x,
                 wPNAPL_scale=self.p['wPNAPL_scale'] * x)
        return self.sim(q)


def driven_glomeruli(model: 'ProfileModel') -> List[str]:
    """The glomeruli with ORN input in the panel (in the wiring and in the ORN matrix)."""
    g = set(model.W.columns.get_level_values('glomerulus'))
    og = model.orn.index.get_level_values('glomerulus') if 'glomerulus' in model.orn.index.names else model.orn.index
    return sorted(x for x in og if x in g)


def point_specs(stage: str, model: 'ProfileModel', out: Path) -> List[dict]:
    """The points of a stage: 'known' (E15's eight and C2's four glomeruli at factors 2 and 4, at each site), 'map' (each
    driven glomerulus at factor 2 at each site), 'neg' (factor 1/2 for the eight per site whose factor-2 point changed the
    per-pair APL effect most, by rms under the readout; needs the map)."""
    pts = []
    if stage == 'known':
        for site in SITES:
            for name, gl in (('E15_eight', E15_EIGHT), ('C2_four', C2_FOUR)):
                for f in (2, 4):
                    pts.append(dict(point_id=f'{site}:{name}:x{f}', stage=stage, site=site, scores={g: float(np.log(f)) for g in gl}))
    elif stage == 'map':
        for site in SITES:
            for g in driven_glomeruli(model):
                pts.append(dict(point_id=f'{site}:{g}:x2', stage=stage, site=site, scores={g: float(np.log(2))}))
    elif stage == 'neg':
        m = pd.read_csv(out / 'sensitivity_map.csv')
        m = m[m.stage == 'map']
        for site in SITES:
            top = m[m.site == site].sort_values('cal_change_rms', ascending=False).head(8)
            for pid in top.point_id:
                g = pid.split(':')[1]
                pts.append(dict(point_id=f'{site}:{g}:x0.5', stage=stage, site=site, scores={g: float(-np.log(2))}))
    else:
        raise ValueError(stage)
    return pts


_CACHE: Dict[str, object] = {}


def score_point(o: Dict[str, tuple], base_eff: Dict[str, pd.Series]) -> Tuple[dict, List[dict]]:
    """A point's statistics (E72's per-readout effect statistics, the common-error comparison, the response statistics of
    E67's full_row, E74's class split under the readout) and its per-pair APL effect rows (raw and calibrated)."""
    import E72_wiring_population as E72
    import E67_apl_effect_fits as E67
    import E74_apl_effect_by_odors_per_kc as E74
    row, pairs = {}, []
    for ro in ('raw', 'cal', 'binary'):
        d = E72.distances(o, ro)
        row.update({f'{ro}_{k}': v for k, v in E72.effect_stats(d, ro).items()})
        if ro in ('raw', 'cal'):
            row.update({f'{ro}_{k}': v for k, v in error_vs_common(E72.pattern_error(d, ro), ro).items()})
            eff = H.apl_effect(d['on'], d['off'])
            ch = eff - base_eff[ro].reindex(eff.index)
            row[f'{ro}_change_rms'] = float(np.sqrt(np.nanmean(ch.to_numpy(dtype=float) ** 2)))
            for (a, b), v in eff.items():
                pairs.append(dict(readout=ro, odor_a=a, odor_b=b, effect=float(v)))
    fr = E67.full_row(o)
    row.update({k: fr[k] for k in ('rate_on', 'rate_off', 'ratio', 'rate_rho_on', 'rate_rho_off', 'opk_on_dist', 'opk_off_dist',
                                   'opk_off_0', 'opk_off_8plus', 'cal_block', 'raw_block', 'realism_flag', 'K') if k in fr})
    on, off = (H.abbrev_columns(o[c][1].astype(float))[E74.ODORS].to_numpy(dtype=float) for c in ('on', 'off'))
    blk = E74.read_cal(on, off)
    if 'data' not in _CACHE:
        _CACHE['data'] = E74.data_effect_array()
    st = E74._variant_stats(blk, _CACHE['data'])
    row.update({f'class_{k}': v for k, v in st.items() if k.startswith(('eff_', 'class_rmse'))})
    return row, pairs


def sensitivity_map(out: Path, points: Optional[Sequence[dict]] = None, stage: str = 'known') -> pd.DataFrame:
    """Part C: each point (site, scores) simulated on C2 at its fitted parameters on the real wiring, the APL->KC scale
    (with KC->APL and bouton->APL tied) bisected until the number of APL-on responses equals C2's; per point the per-pair
    APL effect (raw, calibrated with K re-calibrated, binarized), the loss and its parts, the projection on the common
    error, the group errors, the response rates and ρ, odors per KC, the class split. Appends to sensitivity_map.csv and
    sensitivity_pairs.csv (resumable)."""
    import json
    import E69_kc_firing_ceiling as E69
    import E72_wiring_population as E72
    model = ProfileModel(H.outputs_dir(EXPERIMENT))
    base_eff = {ro: H.apl_effect(*[E72.distances(model.base, ro)[c] for c in ('on', 'off')]) for ro in ('raw', 'cal')}
    mp, pp = out / 'sensitivity_map.csv', out / 'sensitivity_pairs.csv'
    done = set(pd.read_csv(mp).point_id) if mp.exists() else set()
    if 'base' not in done:
        row, pairs = score_point(model.base, base_eff)
        _append(mp, [dict(point_id='base', stage='base', site='', scores='{}', factor=1.0, n_responses=model.target_n,
                          at_bound=False, n_sims=0, **row)])
        _append(pp, [dict(point_id='base', **r) for r in pairs])
    pts = list(points) if points is not None else point_specs(stage, model, out)
    for pt in pts:
        if pt['point_id'] in done:
            continue
        model.set_profile(pt['site'], pt['scores'])
        f, (r, s), hist, at_bound = E69.bisect_factor(model.run_on, model.target_n, model.size, (0.1, 10.0), 1.25,
                                                      f"{pt['point_id']} APL scale")
        o = {'on': (r, s), 'off': model.base['off']}
        row, pairs = score_point(o, base_eff)
        _append(mp, [dict(point_id=pt['point_id'], stage=pt['stage'], site=pt['site'], scores=json.dumps(pt['scores']), factor=f,
                          n_responses=int((r.to_numpy(dtype=float) > 0.5).sum()), at_bound=at_bound, n_sims=len(hist), **row)])
        _append(pp, [dict(point_id=pt['point_id'], **r) for r in pairs])
        print(f"{pt['point_id']}: APL x{f:.4f}; readout loss {row['cal_loss']:.4f} (pattern {row['cal_pattern']:.4f}), projection on the "
              f"common error {row['cal_proj_common']:.3f}, ester / alcohol error {row['cal_err_ester']:+.3f} / {row['cal_err_alcohol']:+.3f}, "
              f"change rms {row['cal_change_rms']:.4f}", flush=True)
    model.set_profile(None, {})
    return pd.read_csv(mp)


def _append(path: Path, rows: List[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, mode='a', header=not path.exists())


LAMBDAS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
S_MAX = float(np.log(4))


def _canon(idx: pd.Index) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in idx], names=['odor_a', 'odor_b'])


def map_matrix(out: Path, site: str, readout: str = 'cal') -> pd.DataFrame:
    """The map of a site: per pair (rows, canonical order) and glomerulus (columns), the change of the per-pair APL effect
    per unit score (the factor-2 point's change from C2's, divided by ln 2), under `readout`."""
    m = pd.read_csv(out / 'sensitivity_map.csv')
    pr = pd.read_csv(out / 'sensitivity_pairs.csv')
    pr = pr[pr.readout == readout]
    base = pr[pr.point_id == 'base'].set_index(['odor_a', 'odor_b'])['effect']
    base.index = _canon(base.index)
    cols = {}
    for _, r in m[(m.stage == 'map') & (m.site == site)].iterrows():
        g = r.point_id.split(':')[1]
        e = pr[pr.point_id == r.point_id].set_index(['odor_a', 'odor_b'])['effect']
        e.index = _canon(e.index)
        cols[g] = (e - base.reindex(e.index)) / np.log(2)
    M = pd.DataFrame(cols).sort_index()
    return M


def population_effect(readout: str = 'cal') -> pd.Series:
    """The per-pair APL effect of E72's mean over its eight wirings (the population whose error is the common error)."""
    import E72_wiring_population as E72
    fits = [E72.load_fit(f, t) for _, f, t in E72.new_fit_members()]
    md = E72.mean_distances([E72.distances(o, readout) for o in fits])
    e = H.apl_effect(md['on'], md['off'])
    e.index = _canon(e.index)
    return e.sort_index()


def _centered(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def fit_scores(M: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    """Scores s minimizing |c + Mc s|^2 + lam * n |s|^2 with |s_g| <= ln 4 (Mc: the map's columns with their means over
    pairs removed, since the pattern error removes the means; c the target pattern error)."""
    from scipy.optimize import lsq_linear
    Mc = _centered(M)
    n, k = Mc.shape
    A = np.vstack([Mc, np.sqrt(lam * n) * np.eye(k)]) if lam > 0 else Mc
    b = np.concatenate([-c, np.zeros(k)]) if lam > 0 else -c
    return lsq_linear(A, b, bounds=(-S_MAX, S_MAX)).x


def residual(M: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The predicted pattern error after the profile: c plus the centered predicted change (re-centered)."""
    r = c + _centered(M) @ s
    return r - r.mean()


def _loo_odor(M: pd.DataFrame, c: pd.Series, lam: float) -> Tuple[float, float]:
    """Leave one odor out: for each odor, scores fitted on the pairs without it, the predicted residual on its pairs.
    (fraction of the held-out pairs' squared common error removed, summed over the 16 folds; the same for no profile = 0)."""
    odors = sorted(set(M.index.get_level_values(0)) | set(M.index.get_level_values(1)))
    num = den = 0.0
    for o in odors:
        held = np.array([o in pr for pr in M.index])
        s = fit_scores(M.to_numpy()[~held], c.to_numpy()[~held] - c.to_numpy()[~held].mean(), lam)
        pred = c.to_numpy() + _centered(M.to_numpy()) @ s
        pred = pred - pred.mean()
        num += float((pred[held] ** 2).sum())
        den += float((c.to_numpy()[held] ** 2).sum())
    return 1.0 - num / den, den


def _inner_lambda(M: pd.DataFrame, c: pd.Series) -> float:
    """The ridge strength with the best leave-one-odor-out reduction (chosen on the data the profile is built from)."""
    best = max(LAMBDAS, key=lambda lam: _loo_odor(M, c, lam)[0])
    return best


def _split_halves(n_splits: int = 100, seed: int = 0) -> List[pd.Series]:
    """The data's per-pair APL effect from random halves of the recordings (control flies and APL output blocked (TNT)
    each split in two), as pairs of (half A, half B) effects."""
    rng = np.random.default_rng(seed)
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    out = []
    for _ in range(n_splits):
        halves = []
        ic, it = rng.permutation(len(ctrl)), rng.permutation(len(tnt))
        for sel_c, sel_t in ((ic[:len(ic) // 2], it[:len(it) // 2]), (ic[len(ic) // 2:], it[len(it) // 2:])):
            e = ctrl.iloc[sel_c].mean() - tnt.iloc[sel_t].mean()
            e.index = _canon(e.index)
            halves.append(e.sort_index())
        out.append(tuple(halves))
    return out


def linear_profile(out: Path) -> pd.DataFrame:
    """Part D1: per site, the ridge-penalized score vector whose linear prediction (the map's per-pair changes) cancels
    the common error, |s_g| <= ln 4; a two-site and a sparse version; leave-one-odor-out and split-half cross-validation;
    the gate. Writes profiles.csv and profile_cv.csv."""
    import hashlib
    import json
    c = common_error('cal')
    c.index = _canon(c.index)
    c = c.sort_index()
    pop = population_effect('cal')
    maps = {site: map_matrix(out, site).reindex(c.index) for site in SITES}
    variants = {site: maps[site] for site in SITES}
    two = sorted(SITES, key=lambda s: _loo_odor(maps[s], c, _inner_lambda(maps[s], c))[0], reverse=True)[:2]
    variants['+'.join(two)] = pd.concat({s: maps[s] for s in two}, axis=1)
    splits = _split_halves()
    prof_rows, cv_rows = [], []
    for name, M in variants.items():
        lam = _inner_lambda(M, c)
        s = fit_scores(M.to_numpy(), c.to_numpy(), lam)
        loo, _ = _loo_odor(M, c, lam)
        # split halves: the population's error against half A's effect, the profile built on it, scored against half B's
        wins, red = 0, []
        for a, b in splits:
            ca = (pop - pop.mean()) - (a.reindex(pop.index) - a.reindex(pop.index).mean())
            cb = (pop - pop.mean()) - (b.reindex(pop.index) - b.reindex(pop.index).mean())
            sa = fit_scores(M.reindex(pop.index).to_numpy(), ca.to_numpy(), lam)
            rb = residual(M.reindex(pop.index).to_numpy(), cb.to_numpy(), sa)
            r0 = float((cb.to_numpy() ** 2).sum())
            wins += float((rb ** 2).sum()) < r0
            red.append(1.0 - float((rb ** 2).sum()) / r0)
        full = residual(M.to_numpy(), c.to_numpy(), s)
        # (two-site profiles key their scores site:glomerulus)
        scores = {(f'{k[0]}:{k[1]}' if isinstance(k, tuple) else str(k)): float(v) for k, v in zip(M.columns, s)}
        sha = hashlib.sha256(json.dumps(scores, sort_keys=True).encode()).hexdigest()
        prof_rows.append(dict(profile=name, variant='ridge', lam=lam, scores=json.dumps(scores), sha256=sha,
                              in_sample_reduction=1.0 - float((full ** 2).sum()) / float((c.to_numpy() ** 2).sum()),
                              n_at_bound=int((np.abs(s) >= S_MAX - 1e-9).sum())))
        gate = (loo >= 0.15) and (wins / len(splits) >= 0.9)
        cv_rows.append(dict(profile=name, variant='ridge', lam=lam, loo_odor_reduction=loo, split_win_fraction=wins / len(splits),
                            split_reduction_median=float(np.median(red)), gate=gate))
        print(f'{name}: ridge {lam:g}; in-sample reduction {prof_rows[-1]["in_sample_reduction"]:.3f}, leave-one-odor-out {loo:.3f}, '
              f'split halves won {wins}/{len(splits)} (median reduction {np.median(red):.3f}); gate {"passed" if gate else "failed"}', flush=True)
    # the sparse version, per site: forward selection of at most eight glomeruli by leave-one-odor-out reduction
    for site in SITES:
        M = maps[site]
        chosen: List[str] = []
        best_loo = 0.0
        for _ in range(8):
            cand = [(g, _loo_odor(M[chosen + [g]], c, 0.0)[0]) for g in M.columns if g not in chosen]
            g, v = max(cand, key=lambda x: x[1])
            if v <= best_loo + 1e-3:
                break
            chosen.append(g)
            best_loo = v
        if not chosen:
            continue
        s = fit_scores(M[chosen].to_numpy(), c.to_numpy(), 0.0)
        full = residual(M[chosen].to_numpy(), c.to_numpy(), s)
        scores = {g: float(v) for g, v in zip(chosen, s)}
        prof_rows.append(dict(profile=site, variant='sparse', lam=0.0, scores=json.dumps(scores),
                              sha256=hashlib.sha256(json.dumps(scores, sort_keys=True).encode()).hexdigest(),
                              in_sample_reduction=1.0 - float((full ** 2).sum()) / float((c.to_numpy() ** 2).sum()),
                              n_at_bound=int((np.abs(s) >= S_MAX - 1e-9).sum())))
        cv_rows.append(dict(profile=site, variant='sparse', lam=0.0, loo_odor_reduction=best_loo, split_win_fraction=np.nan,
                            split_reduction_median=np.nan, gate=np.nan))
        print(f'{site}: sparse {chosen}; leave-one-odor-out {best_loo:.3f}', flush=True)
    pr, cv = pd.DataFrame(prof_rows), pd.DataFrame(cv_rows)
    pr.to_csv(out / 'profiles.csv', index=False)
    cv.to_csv(out / 'profile_cv.csv', index=False)
    return cv


def _map_for(out: Path, site: str) -> pd.DataFrame:
    """The map of a profile's site(s), columns keyed as the profile's scores (glomerulus, or 'site:glomerulus')."""
    if '+' not in site:
        return map_matrix(out, site)
    return pd.concat({s: map_matrix(out, s) for s in site.split('+')}, axis=1).pipe(
        lambda M: M.set_axis([f'{a}:{b}' for a, b in M.columns], axis=1))


def simulate_profiles(out: Path) -> pd.DataFrame:
    """Part D2: the chosen profiles at 0.5, 1 and 1.5 times their strength, the leave-one-odor-out profiles on their
    held-out pairs, permuted controls; scored as part C. Runs only if the gate passed at some site. Writes
    profile_sims.csv (per point: the simulated change against the linear prediction, the predicted reduction of the common
    error if the change transferred, the real wiring's own loss and pattern part)."""
    import json
    cv = pd.read_csv(out / 'profile_cv.csv')
    prof = pd.read_csv(out / 'profiles.csv')
    passed = cv[(cv.variant == 'ridge') & (cv.gate.astype(str) == 'True')]
    if passed.empty:
        print('the gate failed at every site: outcome (c); D2 does not run', flush=True)
        return pd.DataFrame()
    c = common_error('cal')
    c.index = _canon(c.index)
    c = c.sort_index()
    chosen = []
    best = passed.sort_values('loo_odor_reduction', ascending=False).iloc[0]
    chosen.append(prof[(prof.profile == best.profile) & (prof.variant == 'ridge')].iloc[0])
    two = prof[(prof.variant == 'ridge') & prof.profile.str.contains(r'\+', regex=True)]
    if len(two) and two.iloc[0].profile in set(passed.profile):
        chosen.append(two.iloc[0])
    sp = cv[cv.variant == 'sparse'].sort_values('loo_odor_reduction', ascending=False)
    if len(sp):
        chosen.append(prof[(prof.profile == sp.iloc[0].profile) & (prof.variant == 'sparse')].iloc[0])
    pts = []
    for r in chosen:
        sc = json.loads(r.scores)
        for k in (0.5, 1.0, 1.5):
            pts.append(dict(point_id=f'D2:{r.profile}:{r.variant}:x{k:g}', stage='D2', site=r.profile,
                            scores={g: v * k for g, v in sc.items()}))
    # leave one odor out, at the best ridge profile's site and strength
    M = _map_for(out, best.profile).reindex(c.index)
    lam = float(best.lam)
    odors = sorted(set(c.index.get_level_values(0)) | set(c.index.get_level_values(1)))
    for o in odors:
        held = np.array([o in pr for pr in c.index])
        s = fit_scores(M.to_numpy()[~held], c.to_numpy()[~held] - c.to_numpy()[~held].mean(), lam)
        pts.append(dict(point_id=f'D2loo:{best.profile}:{o}', stage='D2loo', site=best.profile,
                        scores={g: float(v) for g, v in zip(M.columns, s)}))
    # permuted controls: the best profile's scores permuted across glomeruli
    sc = json.loads(chosen[0].scores)
    keys = list(sc)
    for seed in (1, 2):
        perm = np.random.default_rng(seed).permutation(len(keys))
        pts.append(dict(point_id=f'D2perm:{best.profile}:{seed}', stage='D2perm', site=best.profile,
                        scores={keys[i]: sc[keys[j]] for i, j in enumerate(perm)}))
    sensitivity_map(out, points=pts)
    return score_profile_sims(out)


def score_profile_sims(out: Path) -> pd.DataFrame:
    """Each part-D2 point against the linear prediction and the common error (profile_sims.csv)."""
    import json
    m = pd.read_csv(out / 'sensitivity_map.csv')
    pr = pd.read_csv(out / 'sensitivity_pairs.csv')
    pr = pr[pr.readout == 'cal']
    base = pr[pr.point_id == 'base'].set_index(['odor_a', 'odor_b'])['effect']
    base.index = _canon(base.index)
    c = common_error('cal')
    c.index = _canon(c.index)
    c = c.sort_index()
    base_row = m[m.point_id == 'base'].iloc[0]
    rows = []
    for _, r in m[m.stage.astype(str).str.startswith('D2')].iterrows():
        e = pr[pr.point_id == r.point_id].set_index(['odor_a', 'odor_b'])['effect']
        e.index = _canon(e.index)
        ch = (e - base.reindex(e.index)).reindex(c.index)
        sc = json.loads(r.scores)
        M = _map_for(out, r.site).reindex(c.index)
        pred = M[list(sc)].to_numpy() @ np.array([sc[g] for g in sc])
        chc = ch.to_numpy() - ch.to_numpy().mean()
        res = c.to_numpy() + chc
        res = res - res.mean()
        row = dict(point_id=r.point_id, stage=r.stage, site=r.site, factor=r.factor,
                   corr_sim_pred=float(np.corrcoef(ch.to_numpy(), pred)[0, 1]),
                   transfer_reduction=1.0 - float((res ** 2).sum()) / float((c.to_numpy() ** 2).sum()),
                   cal_loss=r.cal_loss, cal_pattern=r.cal_pattern, cal_rho=r.cal_rho, raw_loss=r.raw_loss,
                   d_cal_loss=r.cal_loss - base_row.cal_loss, d_cal_pattern=r.cal_pattern - base_row.cal_pattern,
                   rate_rho_on=r.rate_rho_on, realism_flag=r.realism_flag)
        if str(r.stage) == 'D2loo':
            o = r.point_id.rsplit(':', 1)[1]
            held = np.array([o in pp for pp in c.index])
            row['heldout_reduction'] = 1.0 - float((res[held] ** 2).sum()) / float((c.to_numpy()[held] ** 2).sum())
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'profile_sims.csv', index=False)
    if len(res):
        print(res.round(4).to_string(), flush=True)
    return res


STRONG_TNT = (('2024-07-17', 3, 'L'), ('2024-07-25', 1, 'L'))


def _target(effect: pd.Series, pop: pd.Series) -> pd.Series:
    """The population's pattern error against a data effect (both means removed), canonical pair order."""
    e = effect.copy()
    e.index = _canon(e.index)
    e = e.reindex(pop.index)
    return (pop - pop.mean()) - (e - e.mean())


def six_target(pop: pd.Series) -> pd.Series:
    """The population's error against the six typical recordings with APL output blocked (TNT) (the eight without
    2024-07-17 fly 3 L and 2024-07-25 fly 1 L) and all eleven control recordings."""
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    keep = [r for r in tnt.index if tuple((str(r[0]), int(r[1]), str(r[2]))) not in STRONG_TNT]
    if len(keep) != len(tnt) - 2:
        raise RuntimeError(f'expected to drop 2 of {len(tnt)} TNT recordings, kept {len(keep)}')
    return _target(ctrl.mean() - tnt.loc[keep].mean(), pop)


def sim_change(out: Path, point_id: str, index: pd.Index) -> pd.Series:
    """A simulated point's change of the per-pair APL effect (calibrated readout) from C2's, canonical pair order."""
    pr = pd.read_csv(out / 'sensitivity_pairs.csv')
    pr = pr[pr.readout == 'cal']
    base = pr[pr.point_id == 'base'].set_index(['odor_a', 'odor_b'])['effect']
    e = pr[pr.point_id == point_id].set_index(['odor_a', 'odor_b'])['effect']
    base.index, e.index = _canon(base.index), _canon(e.index)
    return (e - base.reindex(e.index)).reindex(index)


def reduction(target: pd.Series, change: pd.Series, mask: Optional[np.ndarray] = None) -> float:
    """The share of `target`'s squared sum removed when `change` (its mean removed) is added (re-centered); over the pairs
    of `mask` if given."""
    ch = change.to_numpy(dtype=float)
    r = target.to_numpy(dtype=float) + (ch - ch.mean())
    r = r - r.mean()
    tg = target.to_numpy(dtype=float)
    if mask is None:
        mask = np.ones(len(r), dtype=bool)
    return 1.0 - float((r[mask] ** 2).sum()) / float((tg[mask] ** 2).sum())


def diagnostic_d2(out: Path) -> dict:
    """Part D2 as a staged diagnostic of the cell site after the failed gate (planned by a Fable subagent, 2026-11-10
    17:08; E75's Status): stage 1, the cell ridge profile at 0.5, 1, 1.5 and 2 times its scores, two permutations at the
    best strength k*, and the sparse cell profile at 0.5 and 1 (description only); stage 2, the sixteen leave-one-odor-out
    cell profiles at k*, pooled on their held-out pairs; stage 3, the first 20 of D1's splits that put the two strong TNT
    recordings in different halves, the profile built on half A at k*, scored against half B. A stage runs only if the
    one before passed. Writes d2_stages.json and profile_sims.csv."""
    import json
    c = common_error('cal')
    c.index = _canon(c.index)
    c = c.sort_index()
    pop = population_effect('cal')
    c6 = six_target(pop)
    prof = pd.read_csv(out / 'profiles.csv')
    ridge = prof[(prof.profile == 'cell') & (prof.variant == 'ridge')].iloc[0]
    sparse = prof[(prof.profile == 'cell') & (prof.variant == 'sparse')].iloc[0]
    sc = json.loads(ridge.scores)
    lam = float(ridge.lam)
    m0 = pd.read_csv(out / 'sensitivity_map.csv')
    base_loss = float(m0[m0.point_id == 'base'].cal_loss.iloc[0])
    res = dict(profile_sha256=ridge.sha256, lam=lam, base_cal_loss=base_loss)
    # stage 1
    ks = (0.5, 1.0, 1.5, 2.0)
    pts = [dict(point_id=f'D2s1:cell:ridge:x{k:g}', stage='D2s1', site='cell', scores={g: v * k for g, v in sc.items()}) for k in ks]
    ssc = json.loads(sparse.scores)
    pts += [dict(point_id=f'D2s1:cell:sparse:x{k:g}', stage='D2s1', site='cell', scores={g: v * k for g, v in ssc.items()}) for k in (0.5, 1.0)]
    sensitivity_map(out, points=pts)
    m = pd.read_csv(out / 'sensitivity_map.csv').set_index('point_id')
    tr = {k: reduction(c, sim_change(out, f'D2s1:cell:ridge:x{k:g}', c.index)) for k in ks}
    kstar = max(tr, key=tr.get)
    keys = list(sc)
    perm_pts = []
    for seed in (1, 2):
        perm = np.random.default_rng(seed).permutation(len(keys))
        perm_pts.append(dict(point_id=f'D2s1:cell:perm{seed}:x{kstar:g}', stage='D2s1', site='cell',
                             scores={keys[i]: sc[keys[j]] * kstar for i, j in enumerate(perm)}))
    sensitivity_map(out, points=perm_pts)
    m = pd.read_csv(out / 'sensitivity_map.csv').set_index('point_id')
    pid = f'D2s1:cell:ridge:x{kstar:g}'
    ch = sim_change(out, pid, c.index)
    s1 = dict(transfer_reduction={f'x{k:g}': v for k, v in tr.items()}, kstar=kstar, kstar_transfer=tr[kstar],
              kstar_six=reduction(c6, ch), kstar_loss=float(m.loc[pid, 'cal_loss']), kstar_loss_gain=base_loss - float(m.loc[pid, 'cal_loss']),
              kstar_corr_sim_pred=float(np.corrcoef(ch.to_numpy(), _centered(map_matrix(out, 'cell').reindex(c.index)[keys].to_numpy())
                                                    @ np.array([sc[g] * kstar for g in keys]) + 0)[0, 1]),
              perms={p['point_id']: dict(transfer=reduction(c, sim_change(out, p['point_id'], c.index)),
                                         loss_gain=base_loss - float(m.loc[p['point_id'], 'cal_loss'])) for p in perm_pts},
              sparse={f'x{k:g}': dict(transfer=reduction(c, sim_change(out, f'D2s1:cell:sparse:x{k:g}', c.index)),
                                      loss=float(m.loc[f'D2s1:cell:sparse:x{k:g}', 'cal_loss'])) for k in (0.5, 1.0)},
              realism={pid: str(m.loc[pid, 'realism_flag'])})
    s1['passed'] = bool(s1['kstar_transfer'] >= 0.15 and s1['kstar_six'] >= 0.5 * s1['kstar_transfer'] and s1['kstar_loss_gain'] >= 0.002
                        and all(v['transfer'] < 0.5 * s1['kstar_transfer'] and v['loss_gain'] < 0.5 * s1['kstar_loss_gain']
                                for v in s1['perms'].values()))
    res['stage1'] = s1
    _write_d2(out, res)
    print(json.dumps(res['stage1'], indent=1, default=float), flush=True)
    if not s1['passed']:
        res['outcome'] = 'stage 1 failed: outcome (c) final'
        _write_d2(out, res)
        return res
    # stage 2: the leave-one-odor-out profiles at k*
    M = map_matrix(out, 'cell').reindex(c.index)
    odors = sorted(set(c.index.get_level_values(0)) | set(c.index.get_level_values(1)))
    pts, masks = [], {}
    for o in odors:
        held = np.array([o in pr for pr in c.index])
        s = fit_scores(M.to_numpy()[~held], c.to_numpy()[~held] - c.to_numpy()[~held].mean(), lam)
        pts.append(dict(point_id=f'D2s2:cell:loo:{o}:x{kstar:g}', stage='D2s2', site='cell',
                        scores={g: float(v) * kstar for g, v in zip(M.columns, s)}))
        masks[o] = held
    sensitivity_map(out, points=pts)
    num = den = 0.0
    folds = {}
    for o in odors:
        chg = sim_change(out, f'D2s2:cell:loo:{o}:x{kstar:g}', c.index).to_numpy(dtype=float)
        r = c.to_numpy() + (chg - chg.mean())
        r = r - r.mean()
        held = masks[o]
        num += float((r[held] ** 2).sum())
        den += float((c.to_numpy()[held] ** 2).sum())
        folds[o] = 1.0 - float((r[held] ** 2).sum()) / float((c.to_numpy()[held] ** 2).sum())
    s2 = dict(pooled_heldout_reduction=1.0 - num / den, folds=folds)
    s2['passed'] = bool(s2['pooled_heldout_reduction'] >= 0.15)
    res['stage2'] = s2
    _write_d2(out, res)
    print(json.dumps(dict(pooled=s2['pooled_heldout_reduction'], passed=s2['passed']), default=float), flush=True)
    if not s2['passed']:
        res['outcome'] = 'stage 2 failed: outcome (c) final'
        _write_d2(out, res)
        return res
    # stage 3: D1's splits (seed 0, D1's order) with the two strong TNT recordings in different halves, the first 20
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    strong = [i for i, r in enumerate(tnt.index) if tuple((str(r[0]), int(r[1]), str(r[2]))) in STRONG_TNT]
    rng = np.random.default_rng(0)
    chosen = []
    for k in range(100):
        ic, it = rng.permutation(len(ctrl)), rng.permutation(len(tnt))
        hA = set(it[:len(it) // 2])
        if (strong[0] in hA) == (strong[1] in hA):
            continue
        a = ctrl.iloc[ic[:len(ic) // 2]].mean() - tnt.iloc[it[:len(it) // 2]].mean()
        b = ctrl.iloc[ic[len(ic) // 2:]].mean() - tnt.iloc[it[len(it) // 2:]].mean()
        chosen.append((k, _target(a, pop), _target(b, pop)))
        if len(chosen) == 20:
            break
    pts = []
    Mp = map_matrix(out, 'cell').reindex(pop.index)
    for k, ca, cb in chosen:
        sa = fit_scores(Mp.to_numpy(), ca.to_numpy(), lam)
        pts.append(dict(point_id=f'D2s3:cell:split{k}:x{kstar:g}', stage='D2s3', site='cell',
                        scores={g: float(v) * kstar for g, v in zip(Mp.columns, sa)}))
    sensitivity_map(out, points=pts)
    wins, per = 0, {}
    for (k, ca, cb), pt in zip(chosen, pts):
        rd = reduction(cb, sim_change(out, pt['point_id'], pop.index))
        per[k] = rd
        wins += rd > 0
    s3 = dict(n=len(chosen), wins=wins, reductions=per)
    s3['passed'] = bool(wins >= 18)
    res['stage3'] = s3
    res['outcome'] = 'all three stages passed: D3 reopens' if s3['passed'] else 'stage 3 failed: outcome (c) final'
    _write_d2(out, res)
    print(json.dumps(dict(wins=wins, n=len(chosen), passed=s3['passed']), default=float), flush=True)
    return res


def _write_d2(out: Path, res: dict) -> None:
    import json
    (out / 'd2_stages.json').write_text(json.dumps(res, indent=1, default=float) + '\n')


def anatomy(out: Path) -> pd.DataFrame:
    """Part F: the scores against the hemibrain's and the male CNS's per-glomerulus APL densities and their ratio, each
    glomerulus's drive, E17's co-drive score, E15's eight, the gain sets. Writes anatomy.csv."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['A-score', 'B', 'C', 'D1', 'D2', 'D2-diag', 'F'])
    ap.add_argument('--stage', default='known', choices=['known', 'map', 'neg'], help="part C's stage")
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'A-score':
        import E72_wiring_population as E72
        E72.part_e_score(out)
    elif args.part == 'C':
        sensitivity_map(out, stage=args.stage)
    elif args.part == 'D2-diag':
        diagnostic_d2(out)
    else:
        {'B': archive_vs_common, 'D1': linear_profile, 'D2': simulate_profiles, 'F': anatomy}[args.part](out)


if __name__ == '__main__':
    main()
