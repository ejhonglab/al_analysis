#!/usr/bin/env python3
"""E13: bouton base with a claw-coincidence requirement (the two knobs that moved anything).

E3 found bouton-level APL (APL inhibits PN boutons) reproduces the odor-level APL effect;
E5 found a claw-coincidence requirement (`n_claws_active_to_spike`) is the only KC-level
change that moves overall decorrelation to the data's level and removes the model's
non-specific excess shared responders. Here both are combined: the bouton base (E3) tuned
with APL on at 5 % with `n_claws_active_to_spike` in {2, 3}, APL on and acute off (both
APL output weights zero), scored as E12 (E4 statistics, per-odor drive / density readouts,
E7's density-null excess by pair block).

Outputs: <shared>/results/E13_bouton_coincidence/ and ./outputs/E13_bouton_coincidence/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E13_bouton_coincidence.py [-n 2 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess

EXPERIMENT = 'E13_bouton_coincidence'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-n', '--n-claws', nargs='+', type=int, default=[2, 3])
    args = ap.parse_args()

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
    pairs = {}

    for n in [None] + list(args.n_claws):
        tag = 'bouton_base' if n is None else f'bouton_coincidence-{n}'
        if tag in done:
            continue
        kws = dict(BOUTON) if n is None else dict(BOUTON, n_claws_active_to_spike=n)
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag,
                with_pn_contrast=False
            )
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, n_claws_active_to_spike=n, error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            continue
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag,
            n_claws_active_to_spike=n, n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1],
            rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
            **extra, **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale',
                'wAPLPN_scale', 'wPNAPL_scale') if k in params},
        )
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        row.update(per_odor_extras(on_out[0], off_out[0], on[H.PRIMARY_MODEL_RDM],
            off[H.PRIMARY_MODEL_RDM], drive, data_rates
        ))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        rows.append(row)
        pairs[(tag, 'on')] = on[H.PRIMARY_MODEL_RDM]
        pairs[(tag, 'acute_off')] = off[H.PRIMARY_MODEL_RDM]
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        print(f"{tag}: rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; mean dist on "
            f"{row[H.PRIMARY_MODEL_RDM + '__on_mean_model']:.3f}; block "
            f"{row[H.PRIMARY_MODEL_RDM + '__block_fruity_small_iso']:+.3f}; iso vs drive "
            f"{row['spearman_per_odor_iso_vs_drive']:+.2f}; excess on block {row['on_excess_block']:+.3f} "
            f"/ other {row['on_excess_other_between']:+.3f}; vs ctrl "
            f"{row[H.PRIMARY_MODEL_RDM + '__on_spearman']:.3f}; dev vs ctrl "
            f"{row[H.PRIMARY_MODEL_RDM + '__on_spearman_dev']:+.2f}; {meta['seconds']:.0f}s", flush=True
        )

    if pairs:
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E13_bouton_coincidence.py', host=subprocess.run(['hostname'],
            capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        base=BOUTON, n_claws=args.n_claws, approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
