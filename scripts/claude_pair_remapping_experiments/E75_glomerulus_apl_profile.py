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
        self._kc[...] = self._kc0
        self._pn[...] = self._pn0
        self.site, self.scores = site, dict(scores)
        if site is None or not scores:
            return
        f = profile_factors(site, scores, self.W, self.wAPLKC, self.wAPLPN).to_numpy()
        if site == 'bouton':
            self._pn[...] = self._pn0 * f.reshape(self._pn0.shape)
        else:
            self._kc[...] = self._kc0 * f.reshape(self._kc0.shape)

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


def linear_profile(out: Path) -> pd.DataFrame:
    """Part D1: per site, the ridge-penalized score vector whose linear prediction (the map's per-pair changes) cancels
    the common error, |s_g| <= ln 4; a two-site and a sparse version; leave-one-odor-out and split-half cross-validation;
    the gate. Writes profiles.csv and profile_cv.csv."""
    raise NotImplementedError


def simulate_profiles(out: Path) -> pd.DataFrame:
    """Part D2: the chosen profiles at 0.5, 1 and 1.5 times their strength, the leave-one-odor-out profiles on their
    held-out pairs, permuted controls; scored as part C."""
    raise NotImplementedError


def anatomy(out: Path) -> pd.DataFrame:
    """Part F: the scores against the hemibrain's and the male CNS's per-glomerulus APL densities and their ratio, each
    glomerulus's drive, E17's co-drive score, E15's eight, the gain sets. Writes anatomy.csv."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['A-score', 'B', 'C', 'D1', 'D2', 'F'])
    ap.add_argument('--stage', default='known', choices=['known', 'map', 'neg'], help="part C's stage")
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'A-score':
        import E72_wiring_population as E72
        E72.part_e_score(out)
    elif args.part == 'C':
        sensitivity_map(out, stage=args.stage)
    else:
        {'B': archive_vs_common, 'D1': linear_profile, 'D2': simulate_profiles, 'F': anatomy}[args.part](out)


if __name__ == '__main__':
    main()
