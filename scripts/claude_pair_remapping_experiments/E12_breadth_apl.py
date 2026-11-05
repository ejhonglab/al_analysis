#!/usr/bin/env python3
"""E12: APL->KC weight proportional to KC breadth, in the bouton base (shared-responder removal).

E7 (atlas) showed that the data's one robust pair-specific APL effect, the fruity x
small-ester block, is APL removing a population of KCs that respond to both odor groups:
with APL blocked those pairs share responders far beyond what density predicts, with APL
intact they are at the level of other between-group pairs. Every APL variant so far (global,
spatial units, type-specific, bouton-level) suppresses proportionally and leaves the shared
responders in. Here each KC's APL->KC weight (per claw, connectome weights) is multiplied
by ((breadth + 1) / mean(breadth + 1))^k, breadth = number of odors the KC responds to in
the APL-off bouton model (E3's tuned parameters), k in {0, 0.5, 1, 2} and -1 as a control;
the KC<->APL scales are then rescaled so the number of responses equals the k = 0 model's.
A graded-breadth variant uses the KC's total spike count instead.

Readouts (as E4, plus): the fruity x small-ester block, E7's density-null excess (observed
minus expected cosine distance under within-odor KC-identity shuffles) for the block pairs
vs other between-group pairs, with APL on and off, isotonic R2, deviations vs control.

Outputs: <shared>/results/E12_breadth_apl/ and ./outputs/E12_breadth_apl/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E12_breadth_apl.py [-k 0 0.5 1 2 -1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, match_kc_apl_scale, per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus

EXPERIMENT = 'E12_breadth_apl'
E3_DIR = H.HERE / 'outputs' / 'E3_drive_dependent_gain'


def e3_bouton_params() -> Dict[str, float]:
    """E3's tuned bouton-base parameters (row bouton_pnapl_x1 of its stats)."""
    s = pd.read_csv(H.results_dir('E3_drive_dependent_gain') / 'stats_variants.csv')
    r = s[s.tag == 'bouton_pnapl_x1'].iloc[0]
    return {k: float(r[k]) for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale',
        'wAPLPN_scale', 'wPNAPL_scale')}


def density_excess(spike_counts: pd.DataFrame) -> Dict[str, float]:
    """E7's excess distance (observed - density-null cosine distance), by pair block."""
    obs = H.pair_distances(spike_counts, 'cosine')
    null = 1 - H.density_null_similarity(spike_counts)
    null = null.reindex(obs.index)
    excess = obs - null
    blocks = H.block_means(excess)
    between = [b for b in blocks.index if b not in ('within', H.FRUITY_SMALL_BLOCK)]
    return dict(excess_block=float(blocks.get(H.FRUITY_SMALL_BLOCK, np.nan)),
        excess_other_between=float(blocks.loc[between].mean()),
        excess_within=float(blocks.get('within', np.nan)), excess_all=float(excess.mean()),
        null_dist_mean=float(null.mean()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-k', '--exponents', nargs='+', type=float, default=[0.0, 0.5, 1.0, 2.0, -1.0])
    ap.add_argument('--breadth', nargs='+', choices=['n_odors', 'spikes'], default=['n_odors', 'spikes'])
    args = ap.parse_args()
    from al_analysis.mb_model import make_fixed_param_kc_simulator

    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.outputs_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {
        'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')},
    }
    data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    stats_path = res_dir / 'stats_variants.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}

    initial = e3_bouton_params()
    # the bouton base at E3's tuned parameters (no tuning): its wiring and its final APL
    # weight Series, divided by the tuned scales to get the normalized weights that
    # hardcoded `_w*` overrides must be given (they are used unnormalized; passing these
    # back reproduces the base exactly)
    from al_analysis.mb_model import fit_mb_model
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'],
        wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
        wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **BOUTON,
        make_plots=False, silent=True
    )
    wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
    wAPLKC0, wKCAPL0 = wnorm['wAPLKC'], wnorm['wKCAPL']
    claw_kc = wAPLKC0.index.get_level_values(H.KC_ID_LEVEL)
    # breadth from E3's APL-off bouton model
    off_resp = pd.read_parquet(E3_DIR / 'bouton_pnapl_x1_acute_off_responses.parquet')
    off_spk = pd.read_parquet(E3_DIR / 'bouton_pnapl_x1_acute_off_spike_counts.parquet')
    kc_ids = off_resp.index.get_level_values(H.KC_ID_LEVEL)
    breadths = {
        'n_odors': pd.Series(off_resp.astype(float).sum(axis='columns').to_numpy(), index=kc_ids),
        'spikes': pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(), index=kc_ids),
    }
    n_target = None
    k0_resp = out_dir / 'k0_on_responses.parquet'
    if 'k0' in done and k0_resp.exists():
        # response-count target from the saved k = 0 run (resumed run)
        n_target = int((pd.read_parquet(k0_resp).astype(float) != 0).to_numpy().sum())
    pairs = {}

    for bname in args.breadth:
        b = breadths[bname]
        for k in args.exponents:
            if k == 0.0 and bname != args.breadth[0]:
                continue   # k = 0 is the same model for both breadth definitions
            tag = f'{bname}_k{k:g}' if k != 0.0 else 'k0'
            if tag in done:
                continue
            t0 = time.time()
            try:
                bp1 = (b + 1.0)
                factor_kc = (bp1 / bp1.mean()) ** k
                f_claw = factor_kc.reindex(claw_kc).fillna(1.0).to_numpy()
                wAPLKC = wAPLKC0 * f_claw
                sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **BOUTON,
                    _wAPLKC=wAPLKC, _wKCAPL=wKCAPL0, _wAPLPN=wnorm['wAPLPN'],
                    _wPNAPL=wnorm['wPNAPL'], make_plots=False, silent=True
                )
                r0, s0 = sim(dict(initial))
                if n_target is None:
                    if k != 0.0:
                        raise RuntimeError('run k=0 first (sets the response-count target)')
                    n_target = int((r0.astype(float) != 0).to_numpy().sum())
                    f, r_on, s_on = 1.0, r0, s0
                else:
                    f, r_on, s_on = match_kc_apl_scale(sim, dict(initial), n_target)
                p_on = dict(initial, wAPLKC_scale=initial['wAPLKC_scale'] * f,
                    wKCAPL_scale=initial['wKCAPL_scale'] * f)
                r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                del sim
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=tag, tag=tag, breadth=bname, k=k, error=repr(e)[:300]))
                pd.DataFrame(rows).to_csv(stats_path, index=False)
                continue
            for cond, (r, s) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            meta = dict(variant=tag, base='bouton', seed=None, target_rate=0.05, tag=tag,
                breadth=bname, k=k, kc_apl_factor=f, n_kcs=len(r_on),
                n_glomeruli=wPNKC.shape[1], rate_on=rate(r_on), rate_off=rate(r_off),
                seconds=time.time() - t0,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
                wAPLKC_cv=float(np.std(wAPLKC[wAPLKC > 0]) / np.mean(wAPLKC[wAPLKC > 0])),
                breadth_spearman_factor=float(pd.Series(f_claw).corr(pd.Series(wAPLKC0.to_numpy()), method='spearman')),
            )
            row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs,
                drive, data, ref
            )
            row.update(per_odor_extras(r_on, r_off, on[H.PRIMARY_MODEL_RDM],
                off[H.PRIMARY_MODEL_RDM], drive, data_rates
            ))
            for cond, s in (('on', s_on), ('off', s_off)):
                row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
            rows.append(row)
            pairs[(tag, 'on')] = on[H.PRIMARY_MODEL_RDM]
            pairs[(tag, 'acute_off')] = off[H.PRIMARY_MODEL_RDM]
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            print(f"{tag}: f {f:.3f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; "
                f"block {row[H.PRIMARY_MODEL_RDM + '__block_fruity_small_iso']:+.3f}; excess on "
                f"block {row['on_excess_block']:+.3f} / other {row['on_excess_other_between']:+.3f}, "
                f"off block {row['off_excess_block']:+.3f} / other {row['off_excess_other_between']:+.3f}; "
                f"vs ctrl {row[H.PRIMARY_MODEL_RDM + '__on_spearman']:.3f}; {meta['seconds']:.0f}s",
                flush=True
            )

    # the data's excess, for reference (recording means of per-recording excess)
    try:
        tm = H.yang_trialmeans()
        drow = {}
        for g in ('TNT3c', 'TNT'):
            vals = []
            for rec, sub in tm[tm.genotype == g].groupby(['date', 'fly_num', 'hemibrain']):
                mat = sub.pivot(index='roi', columns='odor', values='value')
                mat = mat[[o for o in mat.columns if o != H.SOLVENT]]
                vals.append(density_excess(mat))
            dmean = pd.DataFrame(vals).mean()
            drow.update({f'data_{g}_{kk}': float(v) for kk, v in dmean.items()})
        (res_dir / 'data_density_excess.json').write_text(json.dumps(drow, indent=1) + '\n')
        print('data excess:', {k: round(v, 3) for k, v in drow.items() if 'block' in k or 'other' in k})
    except Exception as e:
        print('data excess failed:', repr(e)[:200])

    if pairs:
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E12_breadth_apl.py', host=subprocess.run(['hostname'], capture_output=True,
            text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        base=BOUTON, initial_params=initial, exponents=args.exponents, breadth=args.breadth,
        approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
