#!/usr/bin/env python3
"""Fills the gaps in the archive's params.csv files from sources that leave no room for guessing.

The archive (<shared>/data/model_outputs/E<n>_*/) holds one params.csv per experiment,
built from the experiments' stats tables. Columns differ between experiments and many cells
are empty although the value is recorded elsewhere: in the experiment's provenance.json
(base kwargs, sweep scalars), in the variant tables the scripts define, or in the tag
itself (the scripts build tags from the parameter values, e.g. `thr70_k2_kckc100`). This
script fills only empty cells, only from those exact sources, and logs every fill:

- rows with neither a tag nor a variant (statistics rows that the archive's table merge
  swept in) are moved to params_unidentified_rows.csv in the same folder;
- `variant` from the base name or the tag when missing;
- `base` (name) from the provenance's `bases` keys (longest prefix of the tag / variant),
  the single `base` dict (named by its kwargs), or the script's variant table;
- `base_kwargs_json` plus one column per base kwarg (one_row_per_claw, prat_claws, ...)
  from the provenance / script dicts;
- parameter columns from the tag's tokens, with an explicit token -> column map per
  experiment (TOKEN_MAPS; the same token means different things in different scripts);
- provenance scalars that apply to every row (kc_kc_tau, claw_thr, claw_pow_ref, ...);
- aliases of one quantity under two names (fixed_thr / fixed_thr_mean when the threshold is
  not a vector, kc_kc_inh / wKCKC_scale, E2 coupling / apl_unit_coupling, E3 wAPLKC /
  wAPLKC_scale, which E3's table built from params['wAPLKC_scale']).

Backups go to <archive>/params_backup/<experiment>.params.csv.<timestamp>; the fills to
<archive>/params_migration_log.csv (experiment, tag, column, old, new, source) and a copy
per experiment. Dry run by default; `--apply` writes.

    ./venv/bin/python migrate_model_output_params.py            # report what would change
    ./venv/bin/python migrate_model_output_params.py --apply    # back up, write, log
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
BASE_KW_COLS = ['one_row_per_claw', 'prat_claws', 'use_connectome_APL_weights', 'n_spikes_for_response',
    'prat_boutons', 'pn_claw_to_apl', 'claw_dynamics', 'pn2kc_connections', 'weight_divisor', 'n_claws',
    '_drop_glom_with_plus', 'homeostatic_thrs', 'equalize_kc_type_sparsity', 'target_sparsity_factor_pre_APL',
    'dist_weight', 'apl_units']

# token (regex on one '_'-separated piece of the tag) -> column, per experiment. Only cells
# that are empty get filled. 'N' is a number.
NUM = r'(-?\d+(?:\.\d+)?)'
COMMON = [(rf'^sp{NUM}$', 'target_rate', float), (rf'^{NUM}pct$', 'target_rate', lambda v: float(v) / 100),
    (rf'^seed{NUM}$', 'seed', int)]
TOKEN_MAPS = {
    'E2': [(rf'^coupling{NUM}$', 'apl_unit_coupling', float), (r'^(fixed|matched|single)$', 'apl_units_mode', str)],
    'E3': [(rf'^inhadd{NUM}$', 'pn_inhadd', float)],
    'E8': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^k{NUM}$', 'k', float), (rf'^kckc{NUM}$', 'kc_kc_inh', float),
        (rf'^tau{NUM}$', 'kc_kc_tau', float), (r'^(connectome|uniform)$', 'wkckc', str), (rf'^breadth{NUM}$', 'k', float)],
    'E10': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^k{NUM}$', 'k', float)],
    'E11': [(rf'^coinc{NUM}$', 'n_claws_active_to_spike', int), (rf'^units{NUM}$', 'apl_unit_coupling', float),
        (rf'^x{NUM}$', 'coinput_factor', float)],
    'E12': [(rf'^k{NUM}$', 'k', float), (r'^(spikes|n_odors)$', 'breadth', str)],
    'E13': [(rf'^coincidence-{NUM}$', 'n_claws_active_to_spike', int)],
    'E14': [(rf'^f{NUM}$', 'factor', float)],
    'E15': [(rf'^k{NUM}$', 'k', float), (rf'^m{NUM}$', 'm', float)],
    'E16': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^tanh{NUM}$', 'claw_sat', float), (rf'^hard{NUM}$', 'claw_sat', float)],
    'E17': [(rf'^k{NUM}$', 'k', float), (rf'^m{NUM}$', 'm', float)],
    'E18': [(rf'^p{NUM}$', 'bouton_ves_p', float), (rf'^tau{NUM}$', 'bouton_tau_r', float)],
    'E19': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^k{NUM}$', 'k', float), (rf'^kckc{NUM}$', 'kc_kc_inh', float)],
    'E20': [(rf'^a{NUM}$', 'alpha', float), (rf'^thr{NUM}$', 'claw_thr', float), (rf'^k{NUM}$', 'k', float),
        (rf'^kckc{NUM}$', 'kc_kc_inh', float)],
    'E22': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^w{NUM}$', 'claw_thr_width', float)],
    'E24': [(rf'^f{NUM}$', 'factor', float)],
    'E25': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^k{NUM}$', 'k', float)],
    'E26': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^pow{NUM}$', 'claw_pow', float)],
    'E27': [(rf'^g{NUM}$', 'gain', float), (r'^(oracle|random|none)$', 'kind', str), (rf'^s{NUM}$', 'seed', int)],
    'E28': [(rf'^thr{NUM}$', 'claw_thr', float), (rf'^k{NUM}$', 'k', float), (rf'^kckc{NUM}$', 'kc_kc_inh', float),
        (rf'^g{NUM}$', 'gain', float)],
    'E32': [(rf'^inhadd{NUM}$', 'pn_inhadd', float)],
}
# E16's cap form is the token's word, the cap its number
FORM_TOKENS = {'E16': [(rf'^(tanh|hard){NUM}$', 'form')]}
# claw_thr 0 in a tag means no per-claw threshold: leave the cell empty (the default)
ZERO_MEANS_DEFAULT = {'claw_thr', 'claw_sat'}
# provenance scalars that apply to every row of the experiment (column, provenance key)
PROV_SCALARS = {
    'E10': [('kc_kc_tau', 'tau')], 'E19': [('kc_kc_tau', 'tau')], 'E25': [('kc_kc_tau', 'tau')], 'E28': [('kc_kc_tau', 'tau')],
    'E22': [('claw_thr', 'thr')], 'E24': [('claw_thr', 'claw_thr')], 'E27': [('claw_thr', 'claw_thr')],
    'E32': [('claw_thr', 'claw_thr')], 'E26': [('claw_pow_ref', 'claw_pow_ref')],
}
# E8: 'plain' = the base APL weights (no breadth weighting, exponent 0); 'breadth3' = exponent 3
CONFIG_TO_K = {'E8': {'plain': 0.0, 'breadth3': 3.0}}
# scripts whose response-rate target (target_sparsity passed to fit_mb_model) is a module constant and
# appears nowhere in the archive: (script, constant); E17 tunes to E15's constant (E17's TARGET = E15.TARGET)
SCRIPT_TARGET = {'E11': ('E11_coincidence_claw_inhibition.py', 'TARGET'), 'E14': ('E14_oracle_coresponders.py', 'TARGET'),
    'E15': ('E15_breadth_shared_apl.py', 'TARGET'), 'E17': ('E15_breadth_shared_apl.py', 'TARGET')}
# scripts run with the default of a --target option that neither the provenance nor the run log records
# (E6: `--target`, default 0.05; its README and run.log show no other value)
SCRIPT_DEFAULT_TARGET = {'E6': ('E6_readout_window.py', '--target')}
RESULTS = H.shared_dir() / 'results'


def script_constant(script: str, name: str) -> Optional[float]:
    """A numeric module-level constant `NAME = <number>` of a script in this folder, or None."""
    p = Path(__file__).with_name(script)
    if not p.exists():
        return None
    m = re.search(rf'^{re.escape(name)}\s*=\s*({NUM})\s*$', p.read_text(), re.M)
    return float(m.group(1)) if m else None


def script_option_default(script: str, option: str) -> Optional[float]:
    """The numeric default of an argparse option of a script in this folder, or None."""
    p = Path(__file__).with_name(script)
    if not p.exists():
        return None
    m = re.search(rf"add_argument\('{re.escape(option)}'[^)]*default=({NUM})", p.read_text())
    return float(m.group(1)) if m else None


def numeric(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def fit_rate_target(results_name: str, archive_dir: Path, tag: str) -> Tuple[Optional[float], str]:
    """The response-rate target a joint fit (`E10_joint_fit.py` row `tag`) was fitted at: its per-fit
    provenance file's `rate_target` (archive copy, else the results folder's), else its results fits.csv
    row's `rate_target`; (None, '') when neither records it."""
    for d in (archive_dir, RESULTS / results_name):
        p = d / f'provenance_{tag}.json'
        if p.exists():
            v = numeric(json.load(open(p)).get('rate_target'))
            if v is not None:
                return v, f'{p.name} rate_target'
    f = RESULTS / results_name / 'fits.csv'
    if f.exists():
        fits = pd.read_csv(f, low_memory=False)
        if 'tag' in fits.columns and 'rate_target' in fits.columns:
            r = fits[fits['tag'].astype(str) == tag]
            if len(r) == 1:
                v = numeric(r['rate_target'].iloc[0])
                if v is not None:
                    return v, f'results/{results_name}/fits.csv rate_target'
    return None, ''


def is_na(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() in ('', 'nan'))


def base_name_from_kwargs(kw: dict) -> str:
    if kw.get('prat_boutons'):
        return 'bouton'
    if kw.get('prat_claws'):
        return 'claw-dynamics' if kw.get('claw_dynamics') else 'claw'
    if kw.get('pn2kc_connections') == 'hemibrain' and 'weight_divisor' in kw:
        return f"hemibrain-wd{kw['weight_divisor']}"
    return kw.get('pn2kc_connections', 'base')


def script_variants(exp: str):
    """Variant / base kwargs tables defined in the experiment's script (as of HEAD)."""
    try:
        if exp == 'E4':
            import E4_substrate as E
            return {k: dict(v[0]) for k, v in E.VARIANTS.items()}, {}
        if exp == 'E5':
            import E5_kc_heterogeneity as E
            bases = {k: dict(v) for k, v in getattr(E, 'BASES', {}).items()}
            return {k: (dict(v) if isinstance(v, dict) else {}) for k, v in E.VARIANTS.items()}, bases
        if exp == 'E11':
            import E11_coincidence_claw_inhibition as E
            return {}, {E.BASE: dict(E.BASES[E.BASE])}
        if exp == 'E14':
            import E14_oracle_coresponders as E
            return {}, {k: dict(v) for k, v in E.BASES14.items()}
        if exp == 'E17':
            import E17_panel_general_apl as E
            return {}, {E.BASE: dict(E.BASE_KWS)}
    except Exception as e:  # scripts import the harness and data; failures are logged, not fatal
        print(f'  ({exp}: could not import its script for variant kwargs: {e!r})')
    return {}, {}


class Migration:
    def __init__(self, exp_dir: Path):
        self.dir = exp_dir
        self.exp = exp_dir.name.split('_')[0]
        self.params_path = exp_dir / 'params.csv'
        self.df = pd.read_csv(self.params_path, dtype={'tag': str, 'variant': str, 'base': str}, low_memory=False)
        prov_path = exp_dir / 'provenance.json'
        self.prov = json.load(open(prov_path)) if prov_path.exists() else {}
        self.log = []
        self.unidentified = None
        self.stats = None

    def fill(self, idx, col: str, value, source: str):
        if col not in self.df.columns:
            self.df[col] = np.nan
            if isinstance(value, str):
                self.df[col] = self.df[col].astype(object)
        old = self.df.at[idx, col]
        if not is_na(old) or value is None:
            return False
        if isinstance(value, (str, bool, np.bool_)) and self.df[col].dtype != object:
            # pandas 3 refuses a bool / str in a numeric column: widen it first
            self.df[col] = self.df[col].astype(object)
        self.df.at[idx, col] = value
        self.log.append(dict(experiment=self.dir.name, tag=self.df.at[idx, 'tag'] if 'tag' in self.df.columns else '',
            column=col, old='' if is_na(old) else old, new=value, source=source))
        return True

    # --- sources -----------------------------------------------------------------
    def bases(self) -> dict:
        """name -> kwargs from the provenance (bases / base / variants) and the script."""
        out = {}
        b = self.prov.get('bases')
        if isinstance(b, dict):
            for k, v in b.items():
                kw = v[0] if isinstance(v, list) and v and isinstance(v[0], dict) else v
                if isinstance(kw, dict):
                    out[k] = dict(kw)
        if isinstance(self.prov.get('base'), dict):
            kw = dict(self.prov['base']); out.setdefault(base_name_from_kwargs(kw), kw)
        if isinstance(self.prov.get('base_kws'), dict):
            kw = dict(self.prov['base_kws'])
            name = 'claw-dynamics_coinc3' if kw.get('n_claws_active_to_spike') == 3 and kw.get('claw_dynamics') else base_name_from_kwargs(kw)
            out.setdefault(name, kw)
        v = self.prov.get('variants')
        if isinstance(v, dict):
            for k, kw in v.items():
                if isinstance(kw, dict):
                    out.setdefault(k, dict(kw))
        sv, sb = script_variants(self.exp)
        for k, kw in list(sb.items()) + list(sv.items()):
            out.setdefault(k, kw)
        return out

    def file_stems(self) -> set:
        """Variant names the archived KC x odor matrices are stored under."""
        stems = set()
        for p in self.dir.glob('*.parquet'):
            m = re.match(r'(.+)_(on|off|acute_off|homeo_full_off|homeo_data_off)_(spike_counts|responses)\.parquet$', p.name)
            if m:
                stems.add(m.group(1))
        return stems

    def run(self):
        df = self.df
        stems = self.file_stems()
        # a model row has a tag; a row with only a variant counts when the variant names an
        # archived output (the variant-keyed tables), so per-statistic rows swept in from
        # other tables are set aside
        ident = pd.Series(False, index=df.index)
        if 'tag' in df.columns:
            ident |= df['tag'].notna() & (df['tag'].astype(str).str.strip() != '')
        if 'variant' in df.columns:
            has_v = df['variant'].notna() & (df['variant'].astype(str).str.strip() != '')
            if 'tag' not in df.columns or not stems:
                ident |= has_v
            else:
                ident |= has_v & df['variant'].astype(str).isin(stems)
        if ident.sum() == 0:
            # an experiment-wide target still applies to every row (E6: both bases run at --target's default)
            target = None
            if self.exp in SCRIPT_TARGET:
                target = (script_constant(*SCRIPT_TARGET[self.exp]), f'{SCRIPT_TARGET[self.exp][0]} {SCRIPT_TARGET[self.exp][1]}')
            elif self.exp in SCRIPT_DEFAULT_TARGET:
                script, opt = SCRIPT_DEFAULT_TARGET[self.exp]
                target = (script_option_default(script, opt), f'{script} {opt} default (no other value recorded)')
            if target is not None and target[0] is not None:
                for idx in df.index:
                    self.fill(idx, 'target_rate', target[0], target[1] + '; experiment-wide, rows not otherwise identified')
                print(f'{self.dir.name}: no identified rows; target_rate only')
                return bool(self.log)
            print(f'{self.dir.name}: no identified rows; left untouched')
            return False
        if (~ident).any():
            self.unidentified = df[~ident].copy()
            self.df = df = df[ident].copy()
        if 'tag' not in df.columns:
            df['tag'] = df['variant']
            self.log.append(dict(experiment=self.dir.name, tag='*', column='tag', old='', new='variant', source='tag := variant (no tag column)'))
        bases = self.bases()
        names_by_len = sorted(bases, key=len, reverse=True)
        variants_prov = self.prov.get('variants') if isinstance(self.prov.get('variants'), dict) else {}
        for idx in df.index:
            tag = str(df.at[idx, 'tag']) if not is_na(df.at[idx, 'tag']) else ''
            variant = str(df.at[idx, 'variant']) if 'variant' in df.columns and not is_na(df.at[idx, 'variant']) else ''
            key = tag or variant
            # base name: existing column, else longest base name that prefixes the tag / variant
            base = df.at[idx, 'base'] if 'base' in df.columns and not is_na(df.at[idx, 'base']) else None
            if base is None:
                for n in names_by_len:
                    if key == n or key.startswith(n + '_') or variant == n or variant.startswith(n + '_'):
                        base = n; self.fill(idx, 'base', n, 'provenance bases / variants: longest name prefixing the tag'); break
                if base is None and len(bases) == 1:
                    base = next(iter(bases)); self.fill(idx, 'base', base, 'provenance base (the experiment\'s only base)')
            if not variant and 'variant' in df.columns:
                self.fill(idx, 'variant', base or tag, 'variant := base name' if base else 'variant := tag')
            # base kwargs
            kw = dict(bases.get(base, {})) if base else {}
            if self.exp == 'E4' and variant in bases:
                kw = dict(bases[variant])
            if kw:
                self.fill(idx, 'base_kwargs_json', json.dumps(kw, sort_keys=True), 'provenance bases / base / variants or the script\'s variant table')
                for k, v in kw.items():
                    if k in BASE_KW_COLS:
                        self.fill(idx, k, v if not isinstance(v, (list, dict)) else json.dumps(v), f'base kwargs of {base or variant}')
            # provenance variant tuples (E11: [n_claws_active_to_spike, apl_unit_coupling, coinput_factor]; E14: [kc_set, factor])
            if self.exp == 'E11' and variant in variants_prov and isinstance(variants_prov[variant], list):
                for col, v in zip(('n_claws_active_to_spike', 'apl_unit_coupling', 'coinput_factor'), variants_prov[variant]):
                    if v is not None:
                        self.fill(idx, col, v, 'provenance variants tuple')
            if self.exp == 'E14' and variant in variants_prov and isinstance(variants_prov[variant], list):
                for col, v in zip(('kc_set', 'factor'), variants_prov[variant]):
                    if v is not None:
                        self.fill(idx, col, v, 'provenance variants tuple')
            # tag tokens
            for tok in tag.split('_'):
                for pat, col, conv in COMMON + TOKEN_MAPS.get(self.exp, []):
                    m = re.match(pat, tok)
                    if m:
                        val = conv(m.group(1))
                        if col in ZERO_MEANS_DEFAULT and isinstance(val, (int, float)) and val == 0:
                            continue
                        self.fill(idx, col, val, f'tag token {tok!r}')
                for pat, col in FORM_TOKENS.get(self.exp, []):
                    m = re.match(pat, tok)
                    if m:
                        self.fill(idx, col, m.group(1), f'tag token {tok!r}')
            # config words -> breadth exponent (E8)
            if self.exp in CONFIG_TO_K and 'config' in df.columns and not is_na(df.at[idx, 'config']):
                cfg = str(df.at[idx, 'config'])
                if cfg in CONFIG_TO_K[self.exp]:
                    self.fill(idx, 'k', CONFIG_TO_K[self.exp][cfg], f'config {cfg!r} (E8: plain = no breadth weighting, breadth3 = exponent 3)')
            # provenance scalars
            for col, pkey in PROV_SCALARS.get(self.exp, []):
                v = self.prov.get(pkey)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.fill(idx, col, float(v), f'provenance {pkey}')
            # aliases
            if 'fixed_thr_mean' in df.columns and 'fixed_thr' in df.columns:
                vec = df.at[idx, 'fixed_thr_is_vector'] if 'fixed_thr_is_vector' in df.columns else None
                if not (isinstance(vec, (bool, np.bool_)) and vec) and str(vec) != 'True':
                    if is_na(df.at[idx, 'fixed_thr']) and not is_na(df.at[idx, 'fixed_thr_mean']):
                        self.fill(idx, 'fixed_thr', float(df.at[idx, 'fixed_thr_mean']), 'alias fixed_thr_mean (scalar threshold)')
                    elif is_na(df.at[idx, 'fixed_thr_mean']) and not is_na(df.at[idx, 'fixed_thr']):
                        self.fill(idx, 'fixed_thr_mean', float(df.at[idx, 'fixed_thr']), 'alias fixed_thr')
            for a, b in (('kc_kc_inh', 'wKCKC_scale'), ('wKCKC_scale', 'kc_kc_inh')):
                if a in df.columns and b in df.columns and is_na(df.at[idx, a]) and not is_na(df.at[idx, b]):
                    self.fill(idx, a, float(df.at[idx, b]), f'alias {b}')
            if 'target' in df.columns and not is_na(df.at[idx, 'target']) and pd.api.types.is_numeric_dtype(df['target']):
                # (E9's 'target' is the centrifugal target name, not a rate: skipped by the dtype test)
                self.fill(idx, 'target_rate', float(df.at[idx, 'target']), 'alias target')
            if self.exp == 'E2' and 'coupling' in df.columns and not is_na(df.at[idx, 'coupling']):
                self.fill(idx, 'apl_unit_coupling', float(df.at[idx, 'coupling']), 'alias coupling (E2 column = fit_mb_model apl_unit_coupling)')
            if self.exp == 'E3' and 'wAPLKC' in df.columns and not is_na(df.at[idx, 'wAPLKC']):
                self.fill(idx, 'wAPLKC_scale', float(df.at[idx, 'wAPLKC']), "alias wAPLKC (E3's table stored params['wAPLKC_scale'] under this name)")
            self.fill_target_rate(idx, tag)
        return True

    def fill_target_rate(self, idx, tag: str):
        """target_rate (the APL-on response fraction the model was tuned or fitted to) from the source that
        records it: a joint fit's own provenance or fits.csv row; for a model run at another experiment's
        fitted parameters with the threshold not re-tuned, that fit's target; E43's noise runs, the target of
        the model they add noise to (provenance model_info); else the script's constant or option default."""
        df = self.df
        if 'target_rate' in df.columns and not is_na(df.at[idx, 'target_rate']):
            return
        if not tag:
            return
        v, src = fit_rate_target(self.dir.name, self.dir, tag)
        if v is not None:
            self.fill(idx, 'target_rate', v, src); return
        if self.stats is None:
            sp = RESULTS / self.dir.name / 'stats_variants.csv'
            self.stats = pd.read_csv(sp, low_memory=False) if sp.exists() else pd.DataFrame()
        st = self.stats
        if len(st) and {'tag', 'ref_fit', 'ref_experiment'} <= set(st.columns):
            r = st[st['tag'].astype(str) == tag]
            if len(r) == 1 and not is_na(r['ref_fit'].iloc[0]) and not is_na(r['ref_experiment'].iloc[0]):
                f = numeric(r['thr_factor_vs_fit'].iloc[0]) if 'thr_factor_vs_fit' in r.columns else None
                if f == 1.0 and not ('target_rate' in r.columns and numeric(r['target_rate'].iloc[0]) is not None):
                    ref_exp, ref_fit = str(r['ref_experiment'].iloc[0]), str(r['ref_fit'].iloc[0])
                    v, src = fit_rate_target(ref_exp, ARCHIVE / ref_exp, ref_fit)
                    if v is not None:
                        self.fill(idx, 'target_rate', v, f'the fit it runs at, threshold not re-tuned: {ref_exp} {ref_fit}, {src}'); return
        if self.exp == 'E43':
            info = self.prov.get('model_info', {}).get(tag.split('__')[0], {})
            source = str(info.get('source', ''))
            m = re.match(r'^results/(E\d+_\w+)/fits\.csv row (\S+)', source)
            if m:
                v, src = fit_rate_target(m.group(1), ARCHIVE / m.group(1), m.group(2))
                if v is not None:
                    self.fill(idx, 'target_rate', v, f'the model the noise is added to (provenance model_info: {source}), {src}'); return
            m = re.match(r'^(E\d+) stats row (\S+)', source)
            if m:
                arch = next(iter(sorted(ARCHIVE.glob(f'{m.group(1)}_*'))), None)
                if arch is not None and (arch / 'params.csv').exists():
                    q = pd.read_csv(arch / 'params.csv', low_memory=False)
                    r = q[q['tag'].astype(str) == m.group(2)] if 'tag' in q.columns and 'target_rate' in q.columns else q.iloc[0:0]
                    if len(r) == 1 and numeric(r['target_rate'].iloc[0]) is not None:
                        self.fill(idx, 'target_rate', float(r['target_rate'].iloc[0]),
                            f'the model the noise is added to (provenance model_info: {source}), {arch.name} params.csv target_rate'); return
        if self.exp in SCRIPT_TARGET:
            script, name = SCRIPT_TARGET[self.exp]
            v = script_constant(script, name)
            if v is not None:
                self.fill(idx, 'target_rate', v, f'{script} {name} (target_sparsity passed to fit_mb_model)'); return
        if self.exp in SCRIPT_DEFAULT_TARGET:
            script, opt = SCRIPT_DEFAULT_TARGET[self.exp]
            v = script_option_default(script, opt)
            if v is not None:
                self.fill(idx, 'target_rate', v, f'{script} {opt} default (no other value recorded)')

    def write(self, stamp: str):
        backup = ARCHIVE / 'params_backup'; backup.mkdir(exist_ok=True)
        shutil.copy2(self.params_path, backup / f'{self.dir.name}.params.csv.{stamp}')
        if self.unidentified is not None and len(self.unidentified):
            up = self.dir / 'params_unidentified_rows.csv'
            old = pd.read_csv(up) if up.exists() else None
            pd.concat([old, self.unidentified], ignore_index=True).to_csv(up, index=False) if old is not None else self.unidentified.to_csv(up, index=False)
        self.df.to_csv(self.params_path, index=False)
        if self.log:
            lg = pd.DataFrame(self.log)
            lp = self.dir / 'params_migration_log.csv'
            if lp.exists():
                lg = pd.concat([pd.read_csv(lp), lg], ignore_index=True)
            lg.to_csv(lp, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('experiments', nargs='*', help='archive folder names (default: all)')
    ap.add_argument('--apply', action='store_true', help='write (default: dry run)')
    args = ap.parse_args()
    dirs = [ARCHIVE / e for e in args.experiments] if args.experiments else sorted(
        (p for p in ARCHIVE.glob('E*') if (p / 'params.csv').exists()), key=lambda p: int(p.name.split('_')[0][1:]))
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    all_logs = []
    for d in dirs:
        m = Migration(d)
        if not m.run():
            continue
        n_un = 0 if m.unidentified is None else len(m.unidentified)
        by_col = pd.Series([l['column'] for l in m.log]).value_counts().to_dict() if m.log else {}
        print(f'{d.name}: {len(m.df)} rows kept, {n_un} unidentified rows set aside, {len(m.log)} cells filled: {by_col}')
        all_logs += m.log
        if args.apply:
            m.write(stamp)
    if args.apply and all_logs:
        lp = ARCHIVE / 'params_migration_log.csv'
        lg = pd.DataFrame(all_logs); lg.insert(0, 'migration', stamp)
        if lp.exists():
            lg = pd.concat([pd.read_csv(lp), lg], ignore_index=True)
        lg.to_csv(lp, index=False)
        print(f'applied; backups in {ARCHIVE / "params_backup"}, log {lp} ({len(all_logs)} fills)')
    elif not args.apply:
        print(f'dry run: {len(all_logs)} cells would be filled; re-run with --apply')


if __name__ == '__main__':
    main()
