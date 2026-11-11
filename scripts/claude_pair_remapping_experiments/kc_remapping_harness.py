#!/usr/bin/env python3
"""Shared harness for the ORN->KC remapping / APL experiments (E1-E10).

Loads the data targets (Yang's APL16 KC data: TNT = APL output blocked, TNT3c = control;
per recording, from the shared directory), the model input (16 APL16 odors from the
megamat and validation2 ORN spike-delta estimates), and the reference pair distances the
Nov 3 report used, and computes the report's statistics for any pair of odor x odor
distance matrices (APL on vs off, model vs data), with bootstrap-over-recordings error
bars.

Conventions (see experiment_priorities.md in the shared directory):
- The data matrices are 1 - cosine similarity over all KCs of a recording (trial-mean
  responses), averaged over recordings. `pair_distances` computes the same for model
  outputs (and 1 - Pearson, for comparison).
- Pairs are indexed by (odor_a, odor_b) tuples in `APL16_ODORS` order (`PAIRS16`).
- Model ORN input for paa, euc and -aPine is at the lab's validation2 concentrations, not
  Yang's (`APPROX_CONC_ODORS`, `APPROX_CONC_NOTE`): say so wherever results involving
  those odors are reported.

Run `python kc_remapping_harness.py` to reproduce the report's data-vs-data statistics
(and print bootstrap CIs over recordings) as a self-test.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

# Every model run from these experiments must use a clean olfsysm build of a known commit
# (CLAUDE.md, "Model provenance"): importing the harness makes `fit_mb_model` refuse
# anything else, regardless of AL_ANALYSIS_ALLOW_DIRTY_OLFSYSM. `model_provenance()` is
# what the scripts put in their provenance.json.
from al_analysis import mb_model as _mb_model
_mb_model.STRICT_CLEAN_OLFSYSM = True
model_provenance = _mb_model.model_provenance


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
APL_PAIR_DIR = HERE / 'apl_pair_analysis'
CONSENSUS_CSV = REPO / 'al_analysis/data/sent_to_vlad/2024-06-21/consensus_mean.csv'

# the report's odor order (= Yang's original matrix files)
APL16_ODORS: List[str] = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', '6al', 't2h', '1-8ol',
    '1-5ol', '1-6ol', 'ms', 'paa', '-aPine', 'euc', 'B-myr'
]
MEGAMAT12: List[str] = APL16_ODORS[:12]
VALIDATION2_4: List[str] = APL16_ODORS[12:]
SOLVENT = 'pfo'
# index level names of fit_mb_model outputs
KC_ID_LEVEL = 'kc_id'
KC_TYPE_LEVEL = 'kc_type'

# chemical groups used for the report's block analysis (null_models.py)
ODOR_GROUPS: Dict[str, str] = {'2h': 'fruity', 'IaA': 'fruity', 'pa': 'fruity',
    '2-but': 'small_ester', 'eb': 'small_ester', 'ep': 'small_ester', '6al': 'ald',
    't2h': 'ald', '1-8ol': 'alc', '1-5ol': 'alc', '1-6ol': 'alc', 'ms': 'ms', 'paa': 'paa',
    '-aPine': 'terp', 'euc': 'terp', 'B-myr': 'terp',
}
FRUITY_SMALL_BLOCK = 'fruity-small_ester'

# log10 v/v concentrations: Yang's APL16 panel vs the ORN data available as model input
YANG_CONC: Dict[str, float] = {**{o: -3.0 for o in MEGAMAT12}, 'paa': -3.0, '-aPine': -2.0,
    'euc': -2.0, 'B-myr': -2.0
}
ORN_INPUT_CONC: Dict[str, float] = {**{o: -3.0 for o in MEGAMAT12}, 'paa': -4.0,
    '-aPine': -1.5, 'euc': -1.75, 'B-myr': -2.0
}
APPROX_CONC_ODORS: Dict[str, Tuple[float, float]] = {o: (YANG_CONC[o], ORN_INPUT_CONC[o])
    for o in APL16_ODORS if YANG_CONC[o] != ORN_INPUT_CONC[o]
}
APPROX_CONC_NOTE: str = ('Model ORN input for ' + ', '.join(
    f'{o} ({y:g} in data vs {m:g} in ORN input)' for o, (y, m) in APPROX_CONC_ODORS.items())
    + ' is within 1 log10 unit of, but not at, the concentration Yang used (allowed by '
    "Tom's rule); results involving these odors are approximate."
)

PAIRS16: List[Tuple[str, str]] = [(a, b) for i, a in enumerate(APL16_ODORS)
    for b in APL16_ODORS[i + 1:]
]
MEGAMAT_PAIRS66: List[Tuple[str, str]] = [p for p in PAIRS16
    if p[0] in MEGAMAT12 and p[1] in MEGAMAT12
]

# fixed colors for conditions (same hues as Yang's earlier Claude analysis)
CONDITION_COLORS: Dict[str, str] = {'control': '#2a78d6', 'TNT': '#c8553d',
    'APL on': '#2a78d6', 'APL off': '#c8553d', 'model': '#555555', 'data': '#999999',
}


# ---------------------------------------------------------------------------
# paths
def olfsysm_dir() -> Path:
    """Returns the olfsysm checkout the installed olfsysm was built from.

    ~/src/olfsysm on Tom's hosts, ~/src/kc_queue/olfsysm on lab members' machines.
    """
    return _mb_model.olfsysm_checkout_path()


def shared_dir() -> Path:
    """The shared directory on tensor (Tom/claude_kc_remapping).

    Set KC_REMAPPING_SHARED to override. Raises FileNotFoundError if none found.
    """
    cands = [os.environ.get('KC_REMAPPING_SHARED')] + [
        f'/{root}/tensor/Tom/claude_kc_remapping' for root in ('local', 'mnt', 'media')
    ]
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c)
    raise FileNotFoundError('shared directory not found; set KC_REMAPPING_SHARED')


def results_dir(experiment: str) -> Path:
    """<shared>/results/<experiment>/ (compact outputs), created if needed."""
    d = shared_dir() / 'results' / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir(experiment: str) -> Path:
    """./outputs/<experiment>/ (large local outputs, gitignored), created if needed."""
    d = HERE / 'outputs' / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_targets_path() -> Path:
    """The data-target registry, <shared>/results/data_targets/data_targets.json (written by
    data_targets.py)."""
    return shared_dir() / 'results' / 'data_targets' / 'data_targets.json'


_DATA_TARGETS: Dict[str, dict] = {}


def data_target(name: str) -> float:
    """Value of one entry of the data-target registry (e.g. 'ctrl_ceiling', 'block').

    Figure scripts and loss functions read KC / ORN data values with this rather than from
    literals. The registry is read once per process; run data_targets.py to rebuild it.

    Raises:
        KeyError: `name` is not in the registry.
    """
    if not _DATA_TARGETS:
        _DATA_TARGETS.update(json.loads(data_targets_path().read_text()))
    if name not in _DATA_TARGETS:
        raise KeyError(f'{name!r} is not in the data-target registry ({data_targets_path()})')
    return float(_DATA_TARGETS[name]['value'])


def data_target_ci(name: str) -> Tuple[float, float]:
    """The 90 % interval (lo, hi) of a data-target registry entry that has one (`data_target`).

    Raises:
        KeyError: `name` is not in the registry, or has no interval.
    """
    data_target(name)
    ci = _DATA_TARGETS[name].get('ci')
    if ci is None:
        raise KeyError(f'{name!r} has no interval in the data-target registry')
    return float(ci[0]), float(ci[1])


# ---------------------------------------------------------------------------
# odors and pairs
def odor_abbrev(odor) -> str:
    """'1-5ol @ -3' -> '1-5ol' (last element if a tuple, e.g. (panel, odor))."""
    if isinstance(odor, tuple):
        odor = odor[-1]
    return str(odor).split(' @ ')[0]


def abbrev_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Returns copy with odor columns renamed to abbreviations (must stay unique)."""
    out = df.copy()
    out.columns = pd.Index([odor_abbrev(c) for c in df.columns], name='odor')
    assert not out.columns.duplicated().any(), out.columns[out.columns.duplicated()]
    return out


def odor_order(odors: Sequence[str]) -> List[str]:
    """Orders odors as in APL16_ODORS, then others alphabetically."""
    rank = {o: i for i, o in enumerate(APL16_ODORS)}
    return sorted(odors, key=lambda o: (rank.get(o, len(rank)), o))


def pair_list(odors: Sequence[str]) -> List[Tuple[str, str]]:
    """Unordered pairs of odors, each (a, b) with a before b in `odor_order`."""
    odors = odor_order(odors)
    return [(a, b) for i, a in enumerate(odors) for b in odors[i + 1:]]


def pairs_from_square(sq: pd.DataFrame) -> pd.Series:
    """Upper triangle of a symmetric odor x odor DataFrame, as a pair Series."""
    sq = abbrev_columns(sq)
    sq.index = sq.columns
    pairs = pair_list(sq.columns)
    return pd.Series([sq.loc[a, b] for a, b in pairs],
        index=pd.MultiIndex.from_tuples(pairs, names=['odor_a', 'odor_b'])
    )


def square_from_pairs(pairs: pd.Series) -> pd.DataFrame:
    """Symmetric odor x odor DataFrame (zero diagonal) from a pair Series."""
    odors = odor_order(set(pairs.index.get_level_values(0)) |
        set(pairs.index.get_level_values(1))
    )
    sq = pd.DataFrame(0.0, index=odors, columns=odors)
    for (a, b), v in pairs.items():
        sq.loc[a, b] = sq.loc[b, a] = v
    return sq


def canonical_pairs(pairs: pd.Series) -> pd.Series:
    """Reindexes a pair Series (keys (a, b) in any order) to `pair_list` order."""
    d = {}
    for (a, b), v in pairs.items():
        key = (a, b) if (a, b) in set(pair_list([a, b])) else (b, a)
        d[key] = v
    odors = odor_order(set(k for p in d for k in p))
    pl = pair_list(odors)
    return pd.Series([d[p] for p in pl],
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b'])
    )


def align_pairs(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Subsets two canonical pair Series to their common pairs (same order), dropping NaN."""
    common = a.index.intersection(b.index)
    a2, b2 = a.loc[common], b.loc[common]
    ok = a2.notna() & b2.notna()
    return a2[ok], b2[ok]


def subset_pairs(pairs: pd.Series, odors: Sequence[str]) -> pd.Series:
    """Pairs with both odors in `odors`."""
    keep = [p for p in pairs.index if p[0] in odors and p[1] in odors]
    return pairs.loc[keep]


# ---------------------------------------------------------------------------
# model input
def apl16_orn_deltas() -> pd.DataFrame:
    """ORN spike-delta input for the 16 APL16 odors: glomeruli x (panel, odor).

    The 12 megamat odors from `megamat_orn_deltas` and paa, -aPine, euc, B-myr from
    `validation2_orn_deltas`, restricted to the 36 glomeruli both have (megamat alone
    has VA4 and VL1 too; validation2 alone has DL4 and VM4). Concentrations are the lab's
    (see APPROX_CONC_ODORS for the three that differ from Yang's).
    """
    from al_analysis.mb_model import megamat_orn_deltas, validation2_orn_deltas

    m = megamat_orn_deltas()
    v = validation2_orn_deltas()
    gl = sorted(set(m.index) & set(v.index))

    def pick(df: pd.DataFrame, panel: str, odors: Sequence[str]) -> pd.DataFrame:
        cols = [c for c in df.columns if c[0] == panel and odor_abbrev(c) in odors]
        cols = sorted(cols, key=lambda c: APL16_ODORS.index(odor_abbrev(c)))
        return df.loc[gl, cols]

    orn = pd.concat([pick(m, 'megamat', MEGAMAT12), pick(v, 'validation2', VALIDATION2_4)],
        axis='columns'
    )
    assert [odor_abbrev(c) for c in orn.columns] == APL16_ODORS
    for c in orn.columns:
        assert float(c[1].split(' @ ')[1]) == ORN_INPUT_CONC[odor_abbrev(c)], c
    assert orn.shape[0] == 36 and not orn.isna().any().any()
    return orn


def orn_input_drive(orn_deltas: pd.DataFrame) -> pd.Series:
    """Per-odor mean spike delta across input glomeruli (a measure of total ORN drive)."""
    return abbrev_columns(orn_deltas).mean(axis='index')


def orn_input_pairs(orn_deltas: pd.DataFrame) -> pd.Series:
    """1 - Pearson between odors across input glomeruli."""
    return pair_distances(orn_deltas, measure='pearson')


# ---------------------------------------------------------------------------
# data: Yang's APL16 KC data (from the shared directory), the report's tables
def yang_data_dir() -> Path:
    return shared_dir() / 'data' / 'yang_apl16_kc_olfdb'


_MEASURE_COL = {'cosine': 'one_minus_cosine', 'pearson': 'one_minus_pearson'}


def _recording_rdm_long_csv(binarize: Optional[float]) -> Path:
    """Path of the per-recording long RDM table (computed and cached if `binarize`)."""
    comp = yang_data_dir() / 'computed'
    if binarize is None:
        return comp / 'rdm_per_recording_long.csv'
    path = comp / f'rdm_per_recording_binarized_gt{binarize:g}_long.csv'
    if path.exists():
        return path
    tm = yang_trialmeans()
    rows = []
    keys = ['genotype', 'date', 'fly_num', 'hemibrain']
    for rec, sub in tm.groupby(keys, sort=True):
        mat = sub.pivot(index='roi', columns='odor', values='value')
        mat = (mat > binarize).astype(float)
        for odor_set, odors in (('no_pfo', [o for o in mat.columns if o != SOLVENT]),
                ('with_pfo', list(mat.columns))):
            cos = pair_distances(mat[odors], 'cosine')
            pear = pair_distances(mat[odors], 'pearson')
            d = pd.DataFrame({'one_minus_pearson': pear, 'one_minus_cosine': cos})
            d.index.names = ['odor_a', 'odor_b']
            d = d.reset_index()
            d.insert(0, 'odor_set', odor_set)
            for k, v in reversed(list(zip(keys, rec))):
                d.insert(0, k, v)
            rows.append(d)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def yang_recording_rdms(genotype: str, measure: str = 'cosine', with_pfo: bool = False,
    binarize: Optional[float] = None) -> pd.DataFrame:
    """Per-recording pair distances: rows (date, fly_num, hemibrain), columns pairs.

    Args:
        genotype: 'TNT3c' (control) or 'TNT' (APL output blocked)
        measure: 'cosine' (1 - cosine similarity; what the report's matrices are) or
            'pearson' (1 - Pearson r), over all KCs of the recording
        with_pfo: include the solvent control as a 17th odor
        binarize: if passed, distances are computed from binarized trial-mean responses
            (response > this threshold; Yang uses 1), the fairer target for binarized
            model `responses`. Cached under computed/.
    """
    df = pd.read_csv(_recording_rdm_long_csv(binarize), dtype={'date': str, 'hemibrain': str})
    df = df[(df.genotype == genotype) &
        (df.odor_set == ('with_pfo' if with_pfo else 'no_pfo'))
    ]
    col = _MEASURE_COL[measure]
    wide = df.pivot_table(index=['date', 'fly_num', 'hemibrain'],
        columns=['odor_a', 'odor_b'], values=col
    )
    # to canonical pair order
    rows = {rec: canonical_pairs(wide.loc[rec]) for rec in wide.index}
    out = pd.DataFrame(rows).T
    out.index = wide.index
    return out


def yang_mean_rdm(genotype: str, measure: str = 'cosine', how: str = 'recording',
    with_pfo: bool = False, binarize: Optional[float] = None) -> pd.Series:
    """Mean pair distances over recordings ('recording') or flies ('fly').

    With how='recording', measure='cosine', binarize=None, this equals the report's
    matrices. See `yang_recording_rdms` for `binarize`.
    """
    recs = yang_recording_rdms(genotype, measure=measure, with_pfo=with_pfo,
        binarize=binarize
    )
    if how == 'recording':
        return recs.mean()
    elif how == 'fly':
        return recs.groupby(level=['date', 'fly_num']).mean().mean()
    raise ValueError(how)


def yang_response_fractions(genotype: str, thresh: float = 1.0) -> pd.DataFrame:
    """Per-recording fraction of KCs with trial-mean response > `thresh` to each odor.

    Rows recordings (date, fly_num, hemibrain), columns odors. Yang's values are z-scored
    fluorescence; she has used thresholds of 0.75 and 1 (see notes/yang-summary.md).
    """
    if thresh == 1.0:
        df = pd.read_csv(yang_data_dir() / 'computed' /
            'response_fraction_per_recording.csv', dtype={'date': str, 'hemibrain': str}
        )
        df = df[df.genotype == genotype]
        wide = df.pivot_table(index=['date', 'fly_num', 'hemibrain'], columns='odor',
            values='frac_responding'
        )
    else:
        tm = yang_trialmeans()
        tm = tm[tm.genotype == genotype].copy()
        tm['responding'] = (tm.value > thresh).astype(float)
        wide = tm.pivot_table(index=['date', 'fly_num', 'hemibrain'], columns='odor',
            values='responding', aggfunc='mean'
        )
    return wide[odor_order(wide.columns)]


def yang_trialmeans() -> pd.DataFrame:
    """Trial-mean response of each KC to each odor, long (all recordings, both genotypes)."""
    return pd.read_parquet(yang_data_dir() / 'computed' / 'trialmean_responses_long.parquet')


def report_matrices() -> Dict[str, pd.Series]:
    """The two matrices the report analyzed, as pair Series: {'TNT3c': ..., 'TNT': ...}."""
    return {g: pairs_from_square(pd.read_csv(APL_PAIR_DIR / f'{g}_dist_apl16.csv',
        index_col=0)) for g in ('TNT3c', 'TNT')
    }


def orn16_consensus_pairs() -> pd.Series:
    """The 16-odor ORN odor-odor distances: 1 - Pearson r across the 36 glomeruli of the
    model's ORN input, rebuilt from olfdb (datasets 1 and 4) by orn16_distances_from_olfdb.py
    (shared directory, data/model_outputs/inputs/orn16_corrdist_olfdb_36glom.csv). Within-panel
    pairs are the mean over flies of per-fly 1 - r (al_util.mean_of_fly_corrs); cross-panel pairs
    (no fly imaged both panels) use the fly-mean vectors.

    Since 2026-11-06; before, this returned the report's matrix
    (`orn16_consensus_pairs_zero_filled()`), which counted glomeruli a panel did not image as
    zero responses (unresolved_issues.md).
    """
    return pairs_from_square(pd.read_csv(shared_dir() / 'data' / 'model_outputs' / 'inputs' /
        'orn16_corrdist_olfdb_36glom.csv', index_col=0
    ))


def orn16_consensus_pairs_zero_filled() -> pd.Series:
    """The original report's 16-odor ORN matrix (1 - r of consensus glomerular mean dF/F over
    the union of both panels' glomeruli, unimaged glomeruli as 0). Kept for comparison only."""
    return pairs_from_square(pd.read_csv(APL_PAIR_DIR / 'orn16_consensus_corrdist.csv',
        index_col=0
    ))


def consensus_orn_drive() -> pd.Series:
    """Per-odor mean consensus dF/F across glomeruli (the report's 'ORN drive')."""
    cons = pd.read_csv(CONSENSUS_CSV)
    cons['abbrev'] = cons['odor1'].str.split(' @ ').str[0].replace({'PAA': 'paa'})
    panel = {**{o: 'megamat' for o in MEGAMAT12}, **{o: 'validation2' for o in VALIDATION2_4}}
    gl = [c for c in cons.columns if c not in ('panel', 'odor1', 'abbrev')]
    drive = {}
    for o in APL16_ODORS:
        r = cons[(cons.abbrev == o) & (cons.panel == panel[o])]
        assert len(r) == 1, (o, len(r))
        drive[o] = r.iloc[0][gl].astype(float).fillna(0).mean()
    return pd.Series(drive, name='orn_drive')


def reference_pairs() -> pd.DataFrame:
    """The report's 66-pair table of megamat12 distances in other spaces (shared12_pairs.csv).

    Columns: chem, orn, pn_dend, pn_bout, kc_claw, kc_soma (Remy's wild-type megamat
    KC somata), kc_soma_new, model_uniform, model_hemibrain, model_hemibrain_wd20,
    vcf_raw, vcf_hyp (hyperbolic VCF embedding, mean of 10 seeds), kc_ctrl, kc_tnt.
    """
    df = pd.read_csv(APL_PAIR_DIR / 'shared12_pairs.csv', index_col=[0, 1])
    df.index.names = ['odor_a', 'odor_b']
    keyed = {p: df.loc[p] for p in df.index}
    canon = canonical_pairs(pd.Series({p: 0.0 for p in keyed}))
    out = pd.DataFrame([keyed[p] if p in keyed else keyed[(p[1], p[0])]
        for p in canon.index], index=canon.index
    )
    return out.astype(float)


# ---------------------------------------------------------------------------
# model outputs -> pair distances
def pair_distances(mat: pd.DataFrame, measure: str = 'cosine', drop_silent: bool = False
    ) -> pd.Series:
    """Odor x odor distances from a (units x odors) matrix, across units.

    Args:
        mat: rows units (KCs, glomeruli, ...), columns odors (any strings `odor_abbrev`
            can parse). E.g. `spike_counts` or `responses` from `fit_mb_model`.
        measure: 'cosine' -> 1 - cosine similarity (the data recipe); 'pearson' -> 1 - r
        drop_silent: drop units that are 0 for every odor first (irrelevant for cosine;
            matters for Pearson)

    Returns pair Series (NaN where an odor has no response at all).
    """
    m = abbrev_columns(mat).astype(float)
    if drop_silent:
        m = m.loc[(m != 0).any(axis='columns')]
    odors = odor_order(m.columns)
    X = m[odors].to_numpy().T
    with np.errstate(invalid='ignore', divide='ignore'):
        if measure == 'cosine':
            norms = np.linalg.norm(X, axis=1)
            S = (X @ X.T) / np.outer(norms, norms)
        elif measure == 'pearson':
            S = np.corrcoef(X)
        else:
            raise ValueError(measure)
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([1 - S[idx[a], idx[b]] for a, b in pl],
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b'])
    )


MODEL_RDM_VARIANTS: Dict[str, Tuple[str, str, bool]] = {
    # name -> (source, measure, drop_silent)
    'spike_counts_cosine': ('spike_counts', 'cosine', False),
    'spike_counts_pearson': ('spike_counts', 'pearson', False),
    'spike_counts_pearson_nosilent': ('spike_counts', 'pearson', True),
    'responses_cosine': ('responses', 'cosine', False),
    'responses_pearson': ('responses', 'pearson', False),
    'responses_pearson_nosilent': ('responses', 'pearson', True),
}
PRIMARY_MODEL_RDM = 'spike_counts_cosine'


def model_rdms(responses: pd.DataFrame, spike_counts: pd.DataFrame) -> Dict[str, pd.Series]:
    """All `MODEL_RDM_VARIANTS` pair Series for one model output."""
    src = {'responses': responses.astype(float), 'spike_counts': spike_counts.astype(float)}
    return {name: pair_distances(src[s], measure=meas, drop_silent=ds)
        for name, (s, meas, ds) in MODEL_RDM_VARIANTS.items()
    }


def response_fractions(responses: pd.DataFrame) -> pd.Series:
    """Fraction of model KCs responding to each odor (columns -> abbreviations)."""
    return abbrev_columns(responses.astype(float)).mean(axis='index')


# ---------------------------------------------------------------------------
# plotting
def oklab_diverging_cmap(n: int = 256, l_end: float = 0.36, c_end: float = 0.15,
    hue_neg: float = 262.0, hue_pos: float = 27.0):
    """olfdb's (hong-data's) default blue-white-red colormap, as a matplotlib colormap.

    Built in Oklab so that lightness is linear in |value|: white at the center, darkening
    at the same rate towards a blue (negative) and a red (positive) pole of equal lightness
    (port of `oklabDiverging` in /opt/olfdb/app/static/colormaps.js, same constants; chroma
    shrunk into the sRGB gamut the same way). Use with a symmetric norm (white at 0).
    """
    from matplotlib.colors import ListedColormap

    def oklab_to_linear_srgb(L, a, b):
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l, m, s_cubed = l_ ** 3, m_ ** 3, s_ ** 3
        return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s_cubed,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s_cubed,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s_cubed)

    def to_srgb8(c):
        c = min(1.0, max(0.0, c))
        v = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(v * 255)

    def oklch_to_rgb8(L, C, hue):
        h = np.deg2rad(hue)
        for _ in range(24):
            rgb = oklab_to_linear_srgb(L, C * np.cos(h), C * np.sin(h))
            if all(-1e-4 <= x <= 1.0001 for x in rgb):
                return [to_srgb8(x) for x in rgb]
            C *= 0.92
        return [to_srgb8(x) for x in oklab_to_linear_srgb(L, 0.0, 0.0)]

    half = (n - 1) / 2
    colors = []
    for i in range(n):
        x = (i - half) / half
        t = abs(x)
        rgb = oklch_to_rgb8(1 - t * (1 - l_end), c_end * t, hue_neg if x < 0 else hue_pos)
        colors.append([v / 255 for v in rgb])
    return ListedColormap(colors, name='oklab_div')


# ---------------------------------------------------------------------------
# statistics
def linear_resid(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """Residual of y around its OLS fit on x, plus fit summary (slope, intercept, r2)."""
    y, x = align_pairs(y, x)
    slope, intercept, r, _, _ = stats.linregress(x.to_numpy(), y.to_numpy())
    resid = y - (intercept + slope * x)
    return resid, dict(slope=float(slope), intercept=float(intercept), r2=float(r**2),
        resid_sd=float(resid.std())
    )


def isotonic_resid(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """Residual of y around the best increasing (isotonic) fit on x, plus r2."""
    y, x = align_pairs(y, x)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    pred = iso.fit(x.to_numpy(), y.to_numpy()).predict(x.to_numpy())
    resid = y - pred
    return resid, dict(r2=float(1 - resid.var() / y.var()), resid_sd=float(resid.std()))


def spearman(a: pd.Series, b: pd.Series) -> float:
    a, b = align_pairs(a, b)
    return float(stats.spearmanr(a, b)[0]) if len(a) > 2 else np.nan


def pearson(a: pd.Series, b: pd.Series) -> float:
    a, b = align_pairs(a, b)
    return float(stats.pearsonr(a, b)[0]) if len(a) > 2 else np.nan


def block_label(a: str, b: str) -> str:
    g1, g2 = ODOR_GROUPS.get(a, a), ODOR_GROUPS.get(b, b)
    return 'within' if g1 == g2 else '-'.join(sorted([g1, g2]))


def block_means(pairs: pd.Series) -> pd.Series:
    """Mean of a pair Series within each between-group block ('within' = same group)."""
    labels = pd.Series([block_label(a, b) for a, b in pairs.index], index=pairs.index)
    return pairs.groupby(labels).mean()


def per_odor_means(pairs: pd.Series) -> pd.Series:
    """Mean over the pairs containing each odor."""
    odors = odor_order(set(pairs.index.get_level_values(0)) |
        set(pairs.index.get_level_values(1))
    )
    return pd.Series({o: pairs[[o in p for p in pairs.index]].mean() for o in odors})


def compare_on_off(on: pd.Series, off: pd.Series, *, orn: Optional[pd.Series] = None,
    drive: Optional[pd.Series] = None, gap_ceiling: float = 0.99
    ) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    """The report's statistics relating an APL-on (control) to an APL-off (TNT) matrix.

    Args:
        on, off: pair distance Series (1 - similarity) with APL intact / blocked
        orn: ORN pair distances (e.g. `orn16_consensus_pairs()`), for the
            ORN-agreement statistics
        drive: per-odor ORN drive (e.g. `consensus_orn_drive()`), for the per-odor
            APL effect vs drive correlation

    Returns:
        stats: flat dict. Keys include mean_on / mean_off, spearman_on_off,
            lin_r2 / iso_r2 (variance of `on` explained by a linear / any monotone
            function of `off`), lin_resid_sd / iso_resid_sd, block_fruity_small_iso
            (mean isotonic residual of the fruity-ester x small-ester block; the data
            give about +0.055), with `orn`: spearman_orn_on / _off, gap_closed
            (fraction of the gap from spearman_orn_on to `gap_ceiling` closed by APL
            block), apl_share_resid_var (1 - var(resid_off) / var(resid_on) around the
            linear ORN prediction), spearman_resid_orn_on_off, and with `drive`:
            spearman_per_odor_iso_vs_drive.
        tables: 'pairs' (per-pair on, off, delta, lin_resid, iso_resid, [orn,
            resid_orn_on, resid_orn_off, apl_eff]), 'blocks', 'per_odor'.
    """
    on, off = align_pairs(on, off)
    out: Dict[str, float] = dict(n_pairs=len(on), mean_on=float(on.mean()),
        mean_off=float(off.mean()), mean_sim_on=float(1 - on.mean()),
        mean_sim_off=float(1 - off.mean()), spearman_on_off=spearman(on, off),
        pearson_on_off=pearson(on, off),
    )
    lin_res, lin = linear_resid(on, off)
    iso_res, iso = isotonic_resid(on, off)
    out.update(lin_r2=lin['r2'], lin_resid_sd=lin['resid_sd'], lin_slope=lin['slope'],
        lin_intercept=lin['intercept'], iso_r2=iso['r2'], iso_resid_sd=iso['resid_sd'],
    )
    pairs = pd.DataFrame({'on': on, 'off': off, 'delta': on - off, 'lin_resid': lin_res,
        'iso_resid': iso_res
    })
    labels = pd.Series([block_label(a, b) for a, b in on.index], index=on.index)
    blocks = pd.DataFrame({'iso_resid': block_means(iso_res),
        'lin_resid': block_means(lin_res), 'on': block_means(on), 'off': block_means(off),
        'n_pairs': labels.value_counts(),
    })
    out['block_fruity_small_iso'] = float(blocks.iso_resid.get(FRUITY_SMALL_BLOCK, np.nan))
    out['block_fruity_small_lin'] = float(blocks.lin_resid.get(FRUITY_SMALL_BLOCK, np.nan))
    between = [b for b in blocks.index if b not in ('within', FRUITY_SMALL_BLOCK)]
    out['block_other_between_iso'] = float(blocks.loc[between, 'iso_resid'].mean())
    per_odor = pd.DataFrame({'on': per_odor_means(on), 'off': per_odor_means(off),
        'delta': per_odor_means(on - off), 'iso_resid': per_odor_means(iso_res),
        'lin_resid': per_odor_means(lin_res),
    })
    if orn is not None:
        orn_a = orn.loc[orn.index.intersection(on.index)]
        pairs['orn'] = orn_a
        out['spearman_orn_on'] = spearman(orn_a, on)
        out['spearman_orn_off'] = spearman(orn_a, off)
        out['pearson_orn_on'] = pearson(orn_a, on)
        out['pearson_orn_off'] = pearson(orn_a, off)
        r_on, f_on = linear_resid(on, orn_a)
        r_off, f_off = linear_resid(off, orn_a)
        pairs['resid_orn_on'] = r_on
        pairs['resid_orn_off'] = r_off
        pairs['apl_eff'] = lin_res
        out['resid_orn_sd_on'] = f_on['resid_sd']
        out['resid_orn_sd_off'] = f_off['resid_sd']
        out['apl_share_resid_var'] = float(1 - r_off.var() / r_on.var())
        out['gap_closed'] = float((out['spearman_orn_off'] - out['spearman_orn_on']) /
            (gap_ceiling - out['spearman_orn_on'])
        )
        out['spearman_resid_orn_on_off'] = spearman(r_on, r_off)
        out['spearman_apl_eff_vs_resid_orn_on'] = spearman(lin_res, r_on)
    if drive is not None:
        d = drive.reindex(per_odor.index)
        per_odor['drive'] = d
        ok = d.notna()
        out['spearman_per_odor_iso_vs_drive'] = float(stats.spearmanr(per_odor.iso_resid[ok],
            d[ok])[0]
        )
        out['spearman_per_odor_delta_vs_drive'] = float(stats.spearmanr(per_odor.delta[ok],
            d[ok])[0]
        )
    return out, {'pairs': pairs, 'blocks': blocks, 'per_odor': per_odor}


REFERENCE_COLS: List[str] = ['orn', 'pn_dend', 'pn_bout', 'kc_claw', 'kc_soma', 'kc_soma_new',
    'model_uniform', 'model_hemibrain_wd20', 'chem', 'vcf_raw', 'vcf_hyp'
]


def compare_to_references(pairs: pd.Series, ref: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
    """Spearman of a pair Series with each megamat12 reference space (66 pairs), and the
    residual-vs-VCF statistics of the report.

    Keys: spearman_<ref> for `REFERENCE_COLS`; resid_orn_sd; spearman_resid_vs_vcf_hyp,
    spearman_resid_vs_vcf_raw, spearman_resid_vs_chem; spearman_resid_vs_remap_wt
    (deviations from the ORN prediction vs those of Remy's wild-type KC somata),
    spearman_resid_vs_remap_ctrl / _tnt (vs Yang's control / TNT deviations); mean_66.
    """
    if ref is None:
        ref = reference_pairs()
    p = pairs.loc[pairs.index.intersection(ref.index)]
    ref = ref.loc[p.index]
    out = {'n_pairs': len(p), 'mean_66': float(p.mean())}
    for c in REFERENCE_COLS:
        out[f'spearman_{c}'] = spearman(p, ref[c])
    resid, fit = linear_resid(p, ref['orn'])
    out['resid_orn_sd'] = fit['resid_sd']
    for c in ('vcf_hyp', 'vcf_raw', 'chem'):
        out[f'spearman_resid_vs_{c}'] = spearman(resid, ref[c])
    for name, col in (('wt', 'kc_soma'), ('ctrl', 'kc_ctrl'), ('tnt', 'kc_tnt')):
        r_ref, _ = linear_resid(ref[col], ref['orn'])
        out[f'spearman_resid_vs_remap_{name}'] = spearman(resid, r_ref)
    return out


def compare_model_to_data(model: pd.Series, data: pd.Series, *,
    orn: Optional[pd.Series] = None) -> Dict[str, float]:
    """How well a model pair-distance Series matches a data one (same condition).

    Keys: n_pairs, spearman, pearson, mean_model, mean_data, rmse, and with `orn`:
    spearman_dev (Spearman between model and data deviations from the linear ORN
    prediction) and pearson_dev, dev_sd_model, dev_sd_data.
    """
    m, d = align_pairs(model, data)
    out = dict(n_pairs=len(m), spearman=spearman(m, d), pearson=pearson(m, d),
        mean_model=float(m.mean()), mean_data=float(d.mean()),
        rmse=float(np.sqrt(np.mean((m - d)**2))),
    )
    if orn is not None:
        o = orn.loc[orn.index.intersection(m.index)]
        rm, fm = linear_resid(m, o)
        rd, fd = linear_resid(d, o)
        out.update(spearman_dev=spearman(rm, rd), pearson_dev=pearson(rm, rd),
            dev_sd_model=fm['resid_sd'], dev_sd_data=fd['resid_sd'],
        )
    return out


def apl_effect(on: pd.Series, off: pd.Series) -> pd.Series:
    """Per-pair APL effect on odor-odor distances: the APL-on distance minus the APL-off
    distance of each pair (canonical pair order; pairs missing from either side dropped)."""
    a, b = align_pairs(canonical_pairs(on), canonical_pairs(off))
    return (a - b).rename('apl_effect')


_YANG_APL_EFFECT: Dict[tuple, pd.Series] = {}


YANG_EFFECT_VALUES = ('graded', 'thresholded', 'noise_corrected', 'nonneg', 'floor')


def yang_recording_rdms_values(genotype: str, values: str = 'graded') -> pd.DataFrame:
    """Per-recording cosine pair distances (rows recordings, columns pairs) of the KC data
    transformed as `values` says: 'graded' (as `yang_recording_rdms`), 'thresholded' (trial
    means <= 1 set to 0 before the distance, as a model KC that does not respond contributes 0),
    'noise_corrected' (each recording's cosine corrected for trial-to-trial noise,
    `noise_floor_trial_distances.analyse`: the distance a noise-free recording would show),
    'nonneg' (negative trial means set to 0) or 'floor' (trial means at or below the noise-defined
    floor `kc_noise_floor()` set to 0: twice the sd of a trial mean's noise, from the control
    flies' trial-to-trial sd of the non-responding entries, the same floor for both genotypes)."""
    if values == 'graded':
        return yang_recording_rdms(genotype, 'cosine')
    if values in ('thresholded', 'nonneg', 'floor'):
        thresh = {'thresholded': KC_RESPONSE_Z, 'nonneg': 0.0}.get(values)
        if thresh is None:
            thresh = kc_noise_floor()
        return recording_rdms_from_matrices({k: transform_matrix(m, values='thresholded', thresh=thresh)
            for k, m in yang_recording_matrices(genotype).items()})
    if values == 'noise_corrected':
        # a script module of this folder that imports the harness, hence imported here
        from noise_floor_trial_distances import analyse, load_trials, recording_arrays
        rows = {}
        for rec, by_odor in recording_arrays(load_trials(genotype)).items():
            _, pr = analyse(by_odor, 'graded')
            rows[rec] = canonical_pairs(pr['corrected'])
        return _stack_recordings(rows)
    raise ValueError(f'unknown values {values!r} (one of {YANG_EFFECT_VALUES})')


def yang_apl_effect(measure: str = 'cosine', binarize: Optional[float] = None,
    values: str = 'graded') -> pd.Series:
    """The KC data's per-pair APL effect: the control flies' (TNT3c) distance minus the
    distance with APL output blocked (TNT), each the mean over recordings (`yang_mean_rdm`).
    `binarize` (e.g. 1, Yang's responder threshold on z-scored trial means) computes it from
    binarized responses instead, the like-for-like target of a binarized model readout
    (`readout(..., 'binary')`). `values` 'thresholded', 'noise_corrected', 'nonneg' or 'floor'
    (cosine only, without `binarize`) computes it from the data transformed as
    `yang_recording_rdms_values` says.
    Cached per (measure, binarize, values)."""
    key = (measure, binarize, values)
    if key not in _YANG_APL_EFFECT:
        if values == 'graded':
            _YANG_APL_EFFECT[key] = apl_effect(yang_mean_rdm('TNT3c', measure, binarize=binarize),
                yang_mean_rdm('TNT', measure, binarize=binarize))
        else:
            if measure != 'cosine' or binarize is not None:
                raise ValueError('values other than graded are computed for cosine distances without binarize')
            _YANG_APL_EFFECT[key] = apl_effect(yang_recording_rdms_values('TNT3c', values).mean(),
                yang_recording_rdms_values('TNT', values).mean())
    return _YANG_APL_EFFECT[key]


def apl_effect_stats(on: pd.Series, off: pd.Series, data: Optional[pd.Series] = None
    ) -> Dict[str, float]:
    """How well a model's per-pair APL effect (APL-on minus APL-off distance) matches the
    KC data's, over the pairs both have (the 120 pairs of the APL16 panel).

    Args:
        on, off: the model's pair distances with APL on and with APL off (acute), cosine
            distance on spike counts as everywhere else.
        data: the data's per-pair APL effect; default `yang_apl_effect()` (control flies
            minus APL output blocked (TNT), cosine, mean over recordings).

    Keys (`apl_effect_` prefix): `rmse`, the APL-effect loss (root mean squared difference
    between the model's and the data's per-pair effect; lower is better); `rmse_centered`, the
    same with each side's mean removed (the pattern part: rmse^2 = mean_error^2 +
    rmse_centered^2); `mean` (the model's mean effect), `mean_error` (model minus data),
    `spearman` and `pearson` (the per-pair pattern), `n_pairs`.
    """
    data = yang_apl_effect() if data is None else data
    m, d = align_pairs(apl_effect(on, off), data)
    err = m - d
    mean_error = float(err.mean())
    return dict(apl_effect_rmse=float(np.sqrt(np.mean(err ** 2))),
        apl_effect_rmse_centered=float(np.sqrt(np.mean((err - mean_error) ** 2))),
        apl_effect_mean=float(m.mean()), apl_effect_mean_error=mean_error,
        apl_effect_spearman=spearman(m, d), apl_effect_pearson=pearson(m, d),
        apl_effect_n_pairs=int(len(m)))


def pair_design(pairs: pd.Index, odors: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """The additive design over odors of a set of odor pairs: one row per pair, one column per odor,
    1 for the pair's two odors (so a pair's value is modelled as the sum of its odors' shares).
    Returns (matrix, odor order); `odors` defaults to the odors in `pairs`, in order of appearance."""
    if odors is None:
        odors = list(dict.fromkeys([o for p in pairs for o in p]))
    col = {o: i for i, o in enumerate(odors)}
    X = np.zeros((len(pairs), len(odors)))
    for r, (a, b) in enumerate(pairs):
        X[r, col[a]] = 1.0
        X[r, col[b]] = 1.0
    return X, odors


def odor_shares(effect: pd.Series) -> Tuple[pd.Series, float]:
    """Each odor's share of a per-pair quantity under the additive model `value(a, b) = s_a + s_b`
    (least squares over the pairs), and the fraction of the quantity's variance over pairs the
    additive model explains."""
    X, odors = pair_design(effect.index)
    y = effect.to_numpy(dtype=float)
    s, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ s
    var = float(np.var(y))
    return pd.Series(s, index=pd.Index(odors, name='odor'), name='share'), (float(1 - np.var(y - fit) / var) if var > 0 else np.nan)


def apl_effect_pair_parts(on: pd.Series, off: pd.Series, data: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per pair, the parts of the APL-effect loss (`apl_effect_decomposition`): columns `model` and
    `data` (each side's per-pair effect), `error` (model minus data, each with its mean removed: the
    pattern part's error), `per_odor` (its projection onto the additive space over odors) and
    `pair_specific` (the residual; `error = per_odor + pair_specific`). Index the pairs both sides
    have; `data` defaults to `yang_apl_effect()`."""
    data = yang_apl_effect() if data is None else data
    m, d = align_pairs(apl_effect(on, off), data)
    err = (m - m.mean()) - (d - d.mean())
    X, _ = pair_design(m.index)
    coef, *_ = np.linalg.lstsq(X, err.to_numpy(dtype=float), rcond=None)
    proj = X @ coef
    return pd.DataFrame({'model': m, 'data': d, 'error': err, 'per_odor': proj,
        'pair_specific': err.to_numpy(dtype=float) - proj}, index=m.index)


CURVE_EPS = 1e-6


def power_curve(s_off: np.ndarray, s_on: np.ndarray) -> Tuple[float, float]:
    """Least-squares fit of `s_on = a * s_off ** b` over pairs (similarities, one minus cosine distance; values below
    CURVE_EPS enter as CURVE_EPS), a and b positive (E77). Returns (a, b)."""
    from scipy.optimize import curve_fit
    x = np.maximum(np.asarray(s_off, dtype=float), CURVE_EPS)
    y = np.asarray(s_on, dtype=float)
    (a, b), _ = curve_fit(lambda x, a, b: a * x ** b, x, y, p0=(1.0, 1.0), bounds=([1e-9, 1e-9], [np.inf, np.inf]), maxfev=20000)
    return float(a), float(b)


def apply_power_curve(s_off, curve) -> np.ndarray:
    """`a * s_off ** b` for `curve` (a, b), or per pair for an (n, 2) array of curves."""
    c = np.asarray(curve, dtype=float)
    x = np.maximum(np.asarray(s_off, dtype=float), CURVE_EPS)
    if c.ndim == 1:
        return c[0] * x ** c[1]
    return c[:, 0] * x ** c[:, 1]


def apl_effect_inherited_split(m_on: pd.Series, m_off: pd.Series, d_on: pd.Series, d_off: pd.Series,
                               data_curve=None, model_curve=None) -> pd.DataFrame:
    """A model's per-pair APL-effect error split into what it inherits from its APL-off distances, its on-against-off
    curve and a pair-specific part (E77). With similarities s = 1 - distance and monotone curves s_on ~ f(s_off),
    f(x) = a x^b fitted to the data (f_d) and to the model (f_m), and g_d(x) = x - f_d(x):

        error = (m_on - m_off) - (d_on - d_off)
              = [g_d(s_off_m) - g_d(s_off_d)]          inherited: the data's own curve applied to both APL-off sets
              + [f_d(s_off_m) - f_m(s_off_m)]          curve: the model's curve against the data's
              + [r_d - r_m]                             pair-specific: each side's residual about its own curve,

    with r = s_on - f(s_off), exactly (to rounding). The `_c` columns are the same terms with their means over pairs
    removed (the pattern part's error is the sum of those).

    Args:
        m_on, m_off, d_on, d_off: pair distances (any pair order; aligned in canonical order).
        data_curve, model_curve: (a, b), or one (a, b) per pair as an (n, 2) array in the canonical pair order (e.g. a
            data curve fitted leaving each pair's odors out); fitted in sample when None.

    Returns:
        DataFrame by pair: error, inherited, curve, pair_specific and their `_c` versions; `attrs` holds the two curves
        when fitted here.
    """
    mo, mf = align_pairs(canonical_pairs(m_on), canonical_pairs(m_off))
    do, df_ = align_pairs(canonical_pairs(d_on), canonical_pairs(d_off))
    mo, do = align_pairs(mo, do)
    mf, df_ = mf.reindex(mo.index), df_.reindex(mo.index)
    som, sfm, sod, sfd = (1 - x.to_numpy(dtype=float) for x in (mo, mf, do, df_))
    attrs = {}
    if data_curve is None:
        data_curve = power_curve(sfd, sod)
        attrs['data_curve'] = data_curve
    if model_curve is None:
        model_curve = power_curve(sfm, som)
        attrs['model_curve'] = model_curve
    fd_m, fd_d = apply_power_curve(sfm, data_curve), apply_power_curve(sfd, data_curve)
    fm_m = apply_power_curve(sfm, model_curve)
    inherited = (sfm - fd_m) - (sfd - fd_d)
    curve = fd_m - fm_m
    pair_specific = (sod - fd_d) - (som - fm_m)
    error = (mo.to_numpy(dtype=float) - mf.to_numpy(dtype=float)) - (do.to_numpy(dtype=float) - df_.to_numpy(dtype=float))
    out = pd.DataFrame({'error': error, 'inherited': inherited, 'curve': curve, 'pair_specific': pair_specific}, index=mo.index)
    for c in list(out.columns):
        out[f'{c}_c'] = out[c] - out[c].mean()
    out.attrs.update(attrs)
    return out


def inherited_shares(split: pd.DataFrame, mask=None) -> Dict[str, float]:
    """Each term's share of the squared pattern error (its projection on the centered error, over `mask`'s pairs if
    given; the three add to 1) and the pattern part's rms (E77)."""
    e = split['error_c'].to_numpy(dtype=float)
    m = np.ones(len(e), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    den = float(e[m] @ e[m])
    return {**{f'share_{k}': float(split[f'{k}_c'].to_numpy(dtype=float)[m] @ e[m]) / den for k in ('inherited', 'curve', 'pair_specific')},
            'pattern_rms': float(np.sqrt(np.mean(e[m] ** 2)))}


def apl_effect_decomposition(on: pd.Series, off: pd.Series, data: Optional[pd.Series] = None
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
    """The APL-effect loss (`apl_effect_stats`) split into three parts whose squares add:
    `rmse**2 = mean_error**2 + per_odor**2 + pair_specific**2`. The pattern part's error (model
    minus data, each with its mean removed) is projected onto the additive space over odors (a
    pair's value the sum of its two odors' shares, `odor_shares`): the projection's root mean
    square is the per-odor part, the residual's the pair-specific part.

    Args:
        on, off: the model's pair distances with APL on and APL off (any readout, e.g.
            `readout_pair_distances`).
        data: the data's per-pair effect; default `yang_apl_effect()`.

    Returns (dict with the `apl_effect_stats` keys plus `apl_effect_per_odor`,
    `apl_effect_pair_specific`, `apl_effect_share_pearson` (the model's per-odor shares against the
    data's), `apl_effect_model_additive_r2` and `apl_effect_data_additive_r2` (variance over pairs
    each side's additive model explains); DataFrame indexed by odor with the model's and the data's
    shares and their difference).
    """
    data = yang_apl_effect() if data is None else data
    stats = apl_effect_stats(on, off, data)
    parts = apl_effect_pair_parts(on, off, data)
    sm, r2m = odor_shares(parts['model'])
    sd, r2d = odor_shares(parts['data'])
    shares = pd.DataFrame({'model_share': sm, 'data_share': sd.reindex(sm.index)})
    shares['share_error'] = shares.model_share - shares.data_share
    stats.update(apl_effect_per_odor=float(np.sqrt(np.mean(parts['per_odor'] ** 2))),
        apl_effect_pair_specific=float(np.sqrt(np.mean(parts['pair_specific'] ** 2))),
        apl_effect_share_pearson=float(np.corrcoef(shares.model_share, shares.data_share)[0, 1]),
        apl_effect_model_additive_r2=r2m, apl_effect_data_additive_r2=r2d)
    return stats, shares


# ---------------------------------------------------------------------------
# readouts of the model's spike counts (E65): the model's side of the per-pair APL effect is a
# cosine distance on raw spike counts, the data's on z-scored GCaMP fluorescence, which
# compresses large responses; these map a (KCs x odors) spike-count matrix onto a compressed
# signal before the distances are taken
READOUT_KINDS = ('raw', 'sqrt', 'power', 'saturating', 'log1p', 'binary', 'rank_match')


def readout(counts: pd.DataFrame, kind: str = 'raw', param=None, floor: float = 0.0) -> pd.DataFrame:
    """A readout of a model's spike counts (rows KCs, columns odors), elementwise.

    Args:
        counts: spike counts, e.g. the archived `<tag>_<condition>_spike_counts.parquet`.
        kind: 'raw' (the counts; the target as defined); 'sqrt' (n ** 0.5); 'power'
            (n ** param, 0 < param <= 1); 'saturating' (n / (n + param), half-saturation at
            `param` spikes); 'log1p' (log(1 + n / param)); 'binary' (1 where n >= param,
            default 2, i.e. the model's `responses`); 'rank_match' (each count mapped through
            the lookup `param` from `rank_match_map`, a monotone count -> amplitude map).
        param: the readout's parameter (see `kind`); None for 'raw' and 'sqrt'.
        floor: 'saturating' only: the one-spike floor b of the readout with a low end (E68),
            (n + b) / (n + b + param) for n >= 1 and 0 for n = 0, so b raises the weight of the
            weakest responses (b = -1 reads one spike as 0). The default 0 is the plain
            saturating readout, computed exactly as before.

    Every readout keeps 0 at 0 and is monotone in the count, so which KCs respond, and the
    response rates, are the same under every readout.
    """
    x = counts.astype(float)
    if floor and kind != 'saturating':
        raise ValueError(f'floor is defined for the saturating readout only, not {kind!r}')
    if kind == 'raw':
        return x
    if kind == 'sqrt':
        return x ** 0.5
    if kind == 'power':
        return x ** float(param)
    if kind == 'saturating':
        if not floor:
            return x / (x + float(param))
        b = float(floor)
        if b < -1:
            raise ValueError(f'floor must be >= -1 (one spike reads (1 + b) / (1 + b + K)), not {b}')
        with np.errstate(invalid='ignore', divide='ignore'):
            return ((x + b) / (x + b + float(param))).where(x > 0, 0.0)
    if kind == 'log1p':
        return np.log1p(x / float(param))
    if kind == 'binary':
        return (x >= (2.0 if param is None else float(param))).astype(float)
    if kind == 'rank_match':
        lut = param
        vals = np.interp(x.to_numpy(), lut.index.to_numpy(dtype=float), lut.to_numpy(dtype=float))
        return pd.DataFrame(np.where(x.to_numpy() > 0, vals, 0.0), index=x.index, columns=x.columns)
    raise ValueError(f'unknown readout {kind!r} (one of {READOUT_KINDS})')


def readout_pair_distances(counts: pd.DataFrame, kind: str = 'raw', param=None, floor: float = 0.0
    ) -> pd.Series:
    """Cosine odor-odor distances across KCs after `readout` (`floor` as there): the model's
    side of the per-pair APL effect under that readout ('raw' gives the `spike_counts_cosine`
    distances)."""
    return pair_distances(readout(counts, kind, param, floor=floor), measure='cosine')


def apl_effect_stats_readout(on_counts: pd.DataFrame, off_counts: pd.DataFrame, kind: str = 'raw',
    param=None, data: Optional[pd.Series] = None) -> Dict[str, float]:
    """`apl_effect_stats` of a model whose APL-on and acute APL-off spike counts are read out
    the same way (`readout`, one `kind` and `param` for both conditions, so the APL effect stays
    a prediction). `data` defaults to the KC data's effect, or for kind 'binary' to the effect
    of the binarized data (`yang_apl_effect(binarize=1.0)`). The keys of `apl_effect_stats`,
    plus `mean_dist_on` and `mean_dist_off` (the mean pair distance in each condition)."""
    if data is None:
        data = yang_apl_effect(binarize=1.0) if kind == 'binary' else yang_apl_effect()
    on = readout_pair_distances(on_counts, kind, param)
    off = readout_pair_distances(off_counts, kind, param)
    out = apl_effect_stats(on, off, data)
    out.update(mean_dist_on=float(on.mean()), mean_dist_off=float(off.mean()))
    return out


def amplitude_stats(mat: pd.DataFrame) -> pd.DataFrame:
    """How concentrated each odor's response vector is across units (rows units, columns odors;
    a model's counts or readout, or a recording's z-scored trial means), per odor:
    `top1_share` / `top5_share`, the share of the odor's squared response vector (the quantity a
    cosine distance weighs) carried by the top 1 % / 5 % of units; `participation_ratio`,
    (sum x^2)^2 / (n sum x^4), the effective share of units carrying it (1 when all units carry
    equal squared amplitude, 1 / n when one unit carries all); `n_units`."""
    x = mat.astype(float).to_numpy()
    n = x.shape[0]
    sq = x ** 2
    tot = sq.sum(axis=0)
    srt = -np.sort(-sq, axis=0)
    k1, k5 = max(1, int(np.ceil(0.01 * n))), max(1, int(np.ceil(0.05 * n)))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = pd.DataFrame(dict(top1_share=srt[:k1].sum(axis=0) / tot, top5_share=srt[:k5].sum(axis=0) / tot,
            participation_ratio=tot ** 2 / (n * (sq ** 2).sum(axis=0)), n_units=n), index=mat.columns)
    return out


def responder_cv(values: np.ndarray) -> float:
    """Coefficient of variation (sd / mean) of the responders' amplitudes, pooled over odors."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std() / v.mean()) if len(v) and v.mean() > 0 else np.nan


def readout_stat(on_counts: pd.DataFrame, kind: str, param, stat: str = 'participation_ratio',
    responders_min: Optional[float] = None, floor: float = 0.0) -> float:
    """A concentration statistic of the readout of `on_counts`: an `amplitude_stats` column
    (mean over odors) or 'responder_cv' (`responder_cv` of the readout among responders).
    `responders_min` (e.g. 2, the model's response threshold) zeroes the readout of the
    non-responding entries first, like-for-like with the data's statistics on responders only
    (trial means <= 1 set to 0); 'responder_cv' always uses responders (default 2 spikes).
    `floor`: the saturating readout's one-spike floor (`readout`; default 0, none)."""
    r = readout(on_counts, kind, param, floor=floor)
    if stat == 'responder_cv':
        thr = 2.0 if responders_min is None else responders_min
        return responder_cv(r.to_numpy()[on_counts.to_numpy() >= thr])
    if responders_min is not None:
        r = r.where(on_counts >= responders_min, 0.0)
    return float(amplitude_stats(r)[stat].mean())


def calibrate_readout(on_counts: pd.DataFrame, kind: str, target: float, stat: str = 'participation_ratio',
    bounds: Tuple[float, float] = (1e-3, 1e3), iters: int = 60, responders_min: Optional[float] = None,
    floor: float = 0.0) -> float:
    """The parameter of a one-parameter readout ('power', 'saturating' or 'log1p') at which the
    APL-on `readout_stat` (`stat`, `responders_min`) equals `target`, e.g. the control flies'
    participation ratio: bisection on a log scale. The parameter is fixed on the APL-on counts
    alone and then used unchanged for APL off, so the APL effect is not fitted. Returns the bound
    nearest to the target when it is not bracketed (for 'power' the upper bound is 1, raw counts).
    `floor`: the saturating readout's one-spike floor, held fixed (`readout`; default 0, none);
    `calibrate_low_end` calibrates it jointly with the half-saturation."""
    if kind == 'power':
        bounds = (max(bounds[0], 1e-3), 1.0)
    f = lambda p: readout_stat(on_counts, kind, p, stat, responders_min, floor)
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    flo, fhi = f(np.exp(lo)) - target, f(np.exp(hi)) - target
    if flo * fhi > 0:
        return float(np.exp(lo) if abs(flo) < abs(fhi) else np.exp(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(np.exp(mid)) - target
        if fm * flo <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float(np.exp(0.5 * (lo + hi)))


def rank_match_map(on_counts: pd.DataFrame, reference: np.ndarray) -> pd.Series:
    """A monotone count -> amplitude lookup for `readout(..., 'rank_match', lut)`: each positive
    count of the APL-on matrix is placed at its mid-rank quantile among the positive counts and
    given the amplitude at that quantile of `reference` (e.g. the control flies' trial-mean
    amplitudes among responders); counts above the APL-on maximum keep the top amplitude (np.interp
    clamps), 0 stays 0. Fixed on APL on, used unchanged for APL off."""
    c = on_counts.astype(float).to_numpy().ravel()
    c = c[c > 0]
    vals, cnt = np.unique(c, return_counts=True)
    q = (np.cumsum(cnt) - 0.5 * cnt) / cnt.sum()
    ref = np.sort(np.asarray(reference, dtype=float))
    amp = np.quantile(ref, q)
    return pd.Series(amp, index=vals, name='amplitude')


# ---------------------------------------------------------------------------
# the flies' recording protocol (E82): the flies' response is the mean calcium signal 2.0 to 8.1 s
# after the odor's onset minus the 5 s before it (a 2 s pulse, nuclear GCaMP6s), so a model spike
# at time s counts by the area inside that window of the indicator kernel it starts,
# (1 - exp(-(t - s) / rise)) exp(-(t - s) / decay) for t > s.

def kernel_window_weights(times, rise: float, decay: float, window: Tuple[float, float] = (2.0, 8.1),
    ref_time: Optional[float] = None) -> np.ndarray:
    """Each spike's weight under the flies' protocol: the area inside `window` of the kernel
    (1 - exp(-u / rise)) exp(-u / decay) (u the time since the spike), divided by the area of
    a spike at `ref_time`.

    Args:
        times: spike times in seconds from the odor's onset (array-like).
        rise, decay: the kernel's time constants in seconds; rise 0 (an instant rise) and decay
            np.inf (no decay) are the limits, so rise 0 with decay inf is a step and every spike
            before the window has weight exactly 1 (with `ref_time` before the window too).
        window: the response window (start, end) in seconds from onset; the flies' is 2.0 to 8.1.
        ref_time: the spike time of weight 1; default the window's start, which for the flies' 2 s
            pulse is the pulse's end (so the weights are in units of a spike at the end of the
            flies' pulse, whatever the model's pulse). Must be before the window's end.

    A spike inside the window counts by the part of its kernel after it, one after the window by
    0. The baseline (the 5 s before onset) holds no kernel of a spike after onset, so it is not
    subtracted; spikes before onset get the same window area with no baseline correction. With a
    finite rise a spike shortly before the window's start can weigh a little more than one at it
    (whose kernel is still rising when the window opens): at most 1.8 % for rise 1 s, decay 5 s.
    Returns float weights, one per spike.
    """
    t = np.asarray(times, dtype=float)
    w0, w1 = float(window[0]), float(window[1])
    rise, decay = float(rise), float(decay)
    if not w1 > w0:
        raise ValueError(f'window must have end > start, not {window}')
    if rise < 0 or not decay > 0:
        raise ValueError(f'rise must be >= 0 and decay > 0, not {rise}, {decay}')
    ref = w0 if ref_time is None else float(ref_time)
    if not ref < w1:
        raise ValueError(f'ref_time ({ref}) must be before the window end ({w1})')

    def area(s: np.ndarray) -> np.ndarray:
        # u from lo to hi: the window in time since the spike, clipped at the spike
        lo, hi = np.maximum(w0 - s, 0.0), np.maximum(w1 - s, 0.0)
        if np.isinf(decay):
            # (the window's length after the spike, computed so that it is the same float for every spike before
            # the window: w1 - w0 exactly)
            first = np.maximum(w1, s) - np.maximum(w0, s)
            tau = rise
        else:
            # decay * (exp(-lo / decay) - exp(-hi / decay)), accurate for long decays
            first = -decay * np.exp(-lo / decay) * np.expm1(-(hi - lo) / decay)
            tau = rise * decay / (rise + decay)
        if tau == 0:
            return first
        return first - tau * (np.exp(-lo / tau) - np.exp(-hi / tau))

    a_ref = float(area(np.array([ref]))[0])
    if not a_ref > 0:
        raise ValueError(f'a spike at ref_time {ref} has no kernel area inside the window {window}')
    return area(t) / a_ref


def kernel_weighted_counts(spike_times: pd.DataFrame, index: pd.Index, columns: pd.Index, rise: float, decay: float,
    window: Tuple[float, float] = (2.0, 8.1), ref_time: Optional[float] = None, stim_onset: float = 0.0
    ) -> pd.DataFrame:
    """A model's spike counts read through the flies' protocol (E82): KCs x odors sums of each
    spike's `kernel_window_weights`.

    Args:
        spike_times: one row per spike, as mb_model's `simulate(..., spike_times=True)` returns
            and E80_pulse.py saves (`<tag>_<condition>_spike_times.parquet`): `kc` and `odor` (the
            positions of the spike's KC and odor in the spike-count matrix), `time_s` (seconds in
            olfsysm's time, where the odor's onset is 0); other columns are ignored.
        index, columns: the spike-count matrix's KC index and odor columns (the rows of `kc` and
            `odor` refer to them by position).
        rise, decay, window, ref_time: the kernel and the window (`kernel_window_weights`).
        stim_onset: the odor's onset in `time_s` units (the window and `ref_time` are relative to
            it).

    Returns a float DataFrame (`index` x `columns`); with every weight 1 (e.g. rise 0 and decay
    inf with every spike before the window) it equals the spike counts exactly. The responses (two
    or more spikes over the whole simulation) are not affected: only the amplitudes are.
    """
    n_kc, n_odor = len(index), len(columns)
    kc = spike_times['kc'].to_numpy(dtype=np.int64)
    od = spike_times['odor'].to_numpy(dtype=np.int64)
    if len(kc) and (kc.min() < 0 or kc.max() >= n_kc or od.min() < 0 or od.max() >= n_odor):
        raise ValueError(f'spike positions outside a {n_kc} x {n_odor} matrix')
    w = kernel_window_weights(spike_times['time_s'].to_numpy(dtype=float) - float(stim_onset), rise, decay,
        window=window, ref_time=ref_time)
    a = np.zeros((n_kc, n_odor), dtype=float)
    np.add.at(a, (kc, od), w)
    return pd.DataFrame(a, index=index, columns=columns)


# ---------------------------------------------------------------------------
# the flies' traces (E84): per-trial excerpts of a KC's signal around each odor onset, the response
# in any window after the onset (Yang's is the mean over volumes round(2 v) .. round(8 v) - 1 after the
# onset volume minus the mean over the round(5 v) volumes before it, v the volume rate), and the
# indicator kernel (1 - exp(-u / rise)) exp(-u / decay) fitted to mean traces under a drive (an onset
# burst, a steady drive over the pulse, or a histogram of firing).

def window_volumes(window: Tuple[float, float], volume_rate: float) -> Tuple[int, int]:
    """A window (start, end) in seconds from the odor's onset as volumes after the onset volume: (round(start
    * volume_rate), round(end * volume_rate)), the end exclusive, as Yang's pipeline makes its 2-8 s window
    (at the APL16 recordings' 2.964 volumes / s: (2, 8) and (2.0, 8.1) are both volumes 6 to 23)."""
    a, b = int(round(float(window[0]) * volume_rate)), int(round(float(window[1]) * volume_rate))
    if not b > a:
        raise ValueError(f'window {window} has no volume at {volume_rate} volumes / s')
    return a, b


def trace_excerpts(traces: np.ndarray, onsets: Sequence[int], n_before: int, n_after: int) -> np.ndarray:
    """Per-onset excerpts of traces with each excerpt's pre-onset mean subtracted.

    Args:
        traces: (units, volumes).
        onsets: the onset volume of each trial.
        n_before: volumes before the onset (the baseline: its mean is subtracted).
        n_after: volumes from the onset volume on (the onset volume is the first of them).

    Returns (trials, units, n_before + n_after) float64; sample j is volume onset - n_before + j.
    """
    X = np.asarray(traces, dtype=np.float64)
    on = np.asarray(onsets, dtype=np.int64)
    n_before, n_after = int(n_before), int(n_after)
    if n_before < 1 or n_after < 1:
        raise ValueError('n_before and n_after must be at least 1')
    if len(on) and (on.min() - n_before < 0 or on.max() + n_after > X.shape[1]):
        raise ValueError(f'excerpts of {n_before} + {n_after} volumes run off a trace of {X.shape[1]} volumes')
    out = np.empty((len(on), X.shape[0], n_before + n_after))
    for i, n in enumerate(on):
        seg = X[:, n - n_before:n + n_after]
        out[i] = seg - seg[:, :n_before].mean(axis=1, keepdims=True)
    return out


def windowed_responses(excerpts: np.ndarray, windows: Sequence[Tuple[float, float]], *, first_volume: int,
    volume_rate: Optional[float] = None) -> np.ndarray:
    """Mean of baseline-subtracted excerpts (`trace_excerpts`) over each window.

    Args:
        excerpts: (..., samples), sample j being volume `first_volume` + j relative to the onset volume.
        windows: (start, end) pairs, in volumes after the onset volume (end exclusive), or in seconds when
            `volume_rate` is given (`window_volumes`).
        first_volume: the volume of sample 0 relative to the onset volume (-n_before).

    Returns (..., len(windows)). With the window (round(2 v), round(8 v)) on excerpts of a z-scored trace this is
    Yang's per-trial response.
    """
    E = np.asarray(excerpts)
    n = E.shape[-1]
    out = []
    for w in windows:
        a, b = window_volumes(w, volume_rate) if volume_rate is not None else (int(w[0]), int(w[1]))
        i, j = a - int(first_volume), b - int(first_volume)
        if i < 0 or j > n or not j > i:
            raise ValueError(f'window {w} (volumes {a} to {b - 1}) is outside the excerpts '
                f'(volumes {first_volume} to {first_volume + n - 1})')
        out.append(E[..., i:j].mean(axis=-1, dtype=np.float64))
    return np.stack(out, axis=-1)


def indicator_kernel(u, rise: float, decay: float) -> np.ndarray:
    """The indicator kernel (1 - exp(-u / rise)) exp(-u / decay) at times u since a spike (0 for u <= 0; rise 0
    is an instant rise, decay np.inf no decay)."""
    u = np.asarray(u, dtype=float)
    pos = u > 0
    up = np.where(pos, u, 0.0)
    k = np.ones_like(up) if rise == 0 else -np.expm1(-up / rise)
    if not np.isinf(decay):
        k = k * np.exp(-up / decay)
    return np.where(pos, k, 0.0)


def indicator_kernel_integral(u, rise: float, decay: float) -> np.ndarray:
    """The kernel's integral from 0 to u (0 for u <= 0)."""
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    first = u if np.isinf(decay) else -decay * np.expm1(-u / decay)
    tau = rise if np.isinf(decay) else (rise * decay / (rise + decay) if rise > 0 else 0.0)
    return first if tau == 0 else first + tau * np.expm1(-u / tau)


DRIVES: Tuple[str, ...] = ('onset', 'steady', 'histogram')


def _drive_bins(drive: str, pulse: float, edges, weights) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(edges, weights summing to 1) of a drive, None for the onset impulse."""
    if drive == 'onset':
        return None
    if drive == 'steady':
        if not pulse > 0:
            raise ValueError(f'pulse must be > 0, not {pulse}')
        return np.array([0.0, float(pulse)]), np.array([1.0])
    if drive == 'histogram':
        e = np.asarray(edges, dtype=float)
        w = np.asarray(weights, dtype=float)
        if e.ndim != 1 or len(e) != len(w) + 1 or np.any(np.diff(e) <= 0):
            raise ValueError('a histogram drive needs increasing edges, one more than its weights')
        if np.any(w < 0) or not w.sum() > 0:
            raise ValueError('a histogram drive needs non-negative weights with a positive sum')
        return e, w / w.sum()
    raise ValueError(f'drive must be one of {DRIVES}, not {drive!r}')


def kernel_signal(t, rise: float, decay: float, drive: str = 'onset', *, pulse: float = 2.0,
    edges: Optional[Sequence[float]] = None, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """The kernel convolved with a drive of total weight 1, at times `t` (s from the odor's onset).

    Args:
        drive: 'onset' (all firing at the onset, an impulse at 0), 'steady' (uniform firing over 0 to `pulse`)
            or 'histogram' (firing uniform within each bin [edges[i], edges[i + 1]) with total weight
            weights[i]; the weights are divided by their sum).

    Returns float array shaped like `t`.
    """
    t = np.asarray(t, dtype=float)
    bins = _drive_bins(drive, pulse, edges, weights)
    if bins is None:
        return indicator_kernel(t, rise, decay)
    e, w = bins
    keep = w > 0
    lo, hi, w = e[:-1][keep], e[1:][keep], w[keep]
    tt = t.reshape(-1, 1)
    K = (indicator_kernel_integral(tt - lo, rise, decay) - indicator_kernel_integral(tt - hi, rise, decay)) / (hi - lo)
    return (K @ w).reshape(t.shape)


def fit_indicator_kernel(t, y, drive: str = 'onset', *, pulse: float = 2.0, edges=None, weights=None,
    rise_bounds: Tuple[float, float] = (0.05, 10.0), decay_bounds: Tuple[float, float] = (0.5, 200.0),
    fixed: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """Least-squares fit of amplitude * `kernel_signal(t, rise, decay, drive, ...)` to a mean trace `y`.

    The amplitude is solved linearly for each (rise, decay) (and kept >= 0); rise and decay are fitted on a log
    scale within their bounds from several starts. With `fixed` = (rise, decay) only the amplitude is fitted
    (a kernel fitted on one trace tested on another).

    Returns dict: rise, decay, amplitude, rms (root mean squared residual, in y's units), n, rise_at_bound,
    decay_at_bound (within 1 % of a bound), drive.
    """
    from scipy.optimize import least_squares
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 4:
        raise ValueError('need at least 4 finite points')

    def amp_res(rise, decay):
        s = kernel_signal(t, rise, decay, drive, pulse=pulse, edges=edges, weights=weights)
        ss = float(s @ s)
        a = max(float(s @ y) / ss, 0.0) if ss > 0 else 0.0
        return a, y - a * s

    if fixed is not None:
        r, d = float(fixed[0]), float(fixed[1])
        a, res = amp_res(r, d)
        return dict(rise=r, decay=d, amplitude=a, rms=float(np.sqrt(np.mean(res ** 2))), n=int(len(t)),
            rise_at_bound=False, decay_at_bound=False, drive=drive)
    lb = np.log([rise_bounds[0], decay_bounds[0]])
    ub = np.log([rise_bounds[1], decay_bounds[1]])

    def fun(p):
        return amp_res(np.exp(p[0]), np.exp(p[1]))[1]

    best = None
    for r0 in (0.2, 0.7, 2.0):
        for d0 in (2.0, 8.0, 30.0):
            x0 = np.clip(np.log([r0, d0]), lb + 1e-9, ub - 1e-9)
            sol = least_squares(fun, x0, bounds=(lb, ub), x_scale=1.0, xtol=1e-10, ftol=1e-12, gtol=1e-12)
            if best is None or sol.cost < best.cost:
                best = sol
    r, d = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    a, res = amp_res(r, d)

    def at(v, b):
        return bool(v <= b[0] * 1.01 or v >= b[1] / 1.01)
    return dict(rise=r, decay=d, amplitude=a, rms=float(np.sqrt(np.mean(res ** 2))), n=int(len(t)),
        rise_at_bound=at(r, rise_bounds), decay_at_bound=at(d, decay_bounds), drive=drive)


# ---------------------------------------------------------------------------
# the low end of the readout (E68): measurement noise below the data's response criterion, the
# saturating readout's one-spike floor (`readout(..., floor=b)`), the conversion of readout units
# to the data's z units, and the split of a cosine into responder / sub-threshold terms. The
# data's criterion is a trial mean (of 3 trials) above 1 z; its measurement noise is the
# trial-to-trial sd of the non-responding entries (`kc_trial_noise_sd`).
KC_RESPONSE_Z = 1.0
LOW_END_BAND = (0.5, 1.0)


def expected_pair_distances_noisy(mat: pd.DataFrame, noise_sd: float, n_trials: int = 1) -> pd.Series:
    """Cosine distances of a (units x odors) matrix, in expectation over independent zero-mean
    measurement noise of sd `noise_sd` per entry and trial, after averaging `n_trials` trials.

    Closed form: independent noise adds N s^2 / n_trials to each odor's squared length (N units,
    s = `noise_sd`) and nothing, in expectation, to the products between different odors, so the
    distance is 1 - a.b / sqrt((|a|^2 + N s^2 / n)(|b|^2 + N s^2 / n)) (the ratio of the
    expectations, which the sampled mean approaches for many units). `noise_sd` 0 gives
    `pair_distances(mat, 'cosine')`.
    """
    m = abbrev_columns(mat).astype(float)
    odors = odor_order(m.columns)
    X = m[odors].to_numpy().T
    n_units = X.shape[1]
    sq = (X ** 2).sum(axis=1) + n_units * float(noise_sd) ** 2 / n_trials
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (X @ X.T) / np.sqrt(np.outer(sq, sq))
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([1 - S[idx[a], idx[b]] for a, b in pl],
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))


def sample_pseudo_recordings(signal, noise_sd: float, *, n_recordings: int = 10, n_trials: int = 3,
    seed: int = 0, return_trials: bool = False) -> np.ndarray:
    """Pseudo-recordings of a model's readout through measurement noise: independent Gaussian noise
    of sd `noise_sd` added to every entry of every trial (seeded), as the data's trial-to-trial
    noise of the non-responding entries.

    Args:
        signal: the noise-free signal, in z units: a (units x odors) DataFrame or array (the same
            signal on every trial), or an array (n_recordings, n_trials, units, odors) of per-trial
            signals (e.g. jittered trials read out per trial), whose first two dimensions then set
            `n_recordings` and `n_trials`.
        noise_sd: noise sd per entry and trial (z).
        return_trials: return the noisy trials instead of their means.

    Returns (n_recordings, units, odors) noisy trial means, or (n_recordings, n_trials, units,
    odors) noisy trials with `return_trials`.
    """
    sig = signal.to_numpy(dtype=float) if isinstance(signal, pd.DataFrame) else np.asarray(signal, dtype=float)
    if sig.ndim == 2:
        sig = np.broadcast_to(sig, (n_recordings, n_trials) + sig.shape)
    elif sig.ndim != 4:
        raise ValueError(f'signal must be (units, odors) or (recordings, trials, units, odors), not {sig.shape}')
    rng = np.random.default_rng(seed)
    trials = sig + rng.normal(0.0, float(noise_sd), size=sig.shape)
    return trials if return_trials else trials.mean(axis=1)


def noise_corrected_pair_distances(trials: np.ndarray, odors: Sequence[str],
    min_reliability: Optional[float] = None) -> pd.Series:
    """The noise-corrected cosine distances of one recording from its trials (n_trials, units,
    odors), as `noise_floor_trial_distances.analyse(..., 'graded')` computes them for the data:
    each odor's signal power is the squared length of its trial mean minus the trial-to-trial
    variance summed over units divided by the number of trials (floored at 1e-12), and the
    corrected distance is 1 - (dot product of the trial means) / sqrt(signal power a x signal
    power b). Returns a canonical pair Series.

    `min_reliability`: pairs with an odor whose signal power is less than this share of its trial
    mean's squared length are NaN (the correction divides by a near-zero signal there; a model odor
    with almost no responders). The KC data's shares are all above 0.5, so 0.1 leaves them intact.
    """
    X = np.asarray(trials, dtype=float)
    n = X.shape[0]
    xbar = X.mean(axis=0)                                   # units x odors
    s2 = X.var(axis=0, ddof=1).sum(axis=0)                  # per odor, summed over units
    norm2 = (xbar ** 2).sum(axis=0)
    sig = np.maximum(norm2 - s2 / n, 1e-12)
    dots = xbar.T @ xbar
    D = 1 - dots / np.sqrt(np.outer(sig, sig))
    if min_reliability is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            bad = ~(sig / norm2 >= min_reliability)
        D[bad, :] = np.nan
        D[:, bad] = np.nan
    odors = list(odors)
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([D[idx[a], idx[b]] for a, b in pl],
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))


def low_end_stats(trial_signal: np.ndarray, noise_sd: float, *, band: Tuple[float, float] = LOW_END_BAND,
    thresh: float = KC_RESPONSE_Z, weights: Optional[np.ndarray] = None, n_units: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
    """The control-fly statistics the low end is calibrated on, in expectation over Gaussian
    measurement noise (closed form, so a calibration is smooth and needs no sampling): the
    response rate (share of entries whose trial mean exceeds `thresh`), the share of entries
    whose trial mean lies in `band` (lo, hi], and the responders-only participation ratio
    ((sum x^2)^2 / (n sum x^4) over the entries above `thresh`, others 0; the ratio of the
    expectations of the two sums, per group), each the mean over groups (odors, or recordings x
    odors).

    Args:
        trial_signal: (groups, entries) noise-free trial means in z (an entry's mean over trials of
            its readout times the conversion c).
        noise_sd: sd of the noise on a trial mean (the per-trial sd / sqrt(n_trials)); 0 for none.
        weights: (groups, entries) multiplicities of the entries (identical entries grouped);
            default 1.
        n_units: (groups,) number of units per group (the participation ratio's n and the shares'
            denominator); default the summed weights.

    Returns dict with 'rate', 'band_share', 'participation_ratio_resp'.
    """
    mu = np.atleast_2d(np.asarray(trial_signal, dtype=float))
    w = np.ones_like(mu) if weights is None else np.atleast_2d(np.asarray(weights, dtype=float))
    n = w.sum(axis=1) if n_units is None else np.asarray(n_units, dtype=float)
    lo, hi = band
    if noise_sd > 0:
        from scipy.special import ndtr
        sd = float(noise_sd)
        a = (thresh - mu) / sd
        tail = ndtr(-a)
        p_resp = tail
        p_band = ndtr((hi - mu) / sd) - ndtr((lo - mu) / sd)
        # partial moments of the standard normal above a: E[Z^j 1{Z > a}], j = 0..4
        phi = np.exp(-0.5 * a ** 2) / np.sqrt(2 * np.pi)
        m = [tail, phi, a * phi + tail, (a ** 2 + 2) * phi, (a ** 3 + 3 * a) * phi + 3 * tail]
        # E[(mu + sd Z)^k 1{Z > a}] for k = 2, 4, binomially from the partial moments
        e2 = mu ** 2 * m[0] + 2 * mu * sd * m[1] + sd ** 2 * m[2]
        e4 = (mu ** 4 * m[0] + 4 * mu ** 3 * sd * m[1] + 6 * mu ** 2 * sd ** 2 * m[2] + 4 * mu * sd ** 3 * m[3]
            + sd ** 4 * m[4])
    else:
        resp = mu > thresh
        p_resp = resp.astype(float)
        p_band = ((mu > lo) & (mu <= hi)).astype(float)
        e2, e4 = np.where(resp, mu ** 2, 0.0), np.where(resp, mu ** 4, 0.0)
    s2, s4 = (w * e2).sum(axis=1), (w * e4).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        pr = np.where(s4 > 0, s2 ** 2 / (n * s4), np.nan)
    return dict(rate=float(np.mean((w * p_resp).sum(axis=1) / n)), band_share=float(np.mean((w * p_band).sum(axis=1) / n)),
        participation_ratio_resp=float(np.nanmean(pr)))


def _unique_groups(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(groups x entries) values -> (values, weights, n_units), identical values of a group merged
    (zero weights pad the groups to one width)."""
    groups = [np.unique(row, return_counts=True) for row in mu]
    width = max(len(c) for _, c in groups)
    vals, w = np.zeros((len(groups), width)), np.zeros((len(groups), width))
    for g, (u, c) in enumerate(groups):
        vals[g, :len(c)], w[g, :len(c)] = u, c
    return vals, w, np.array([mu.shape[1]] * len(groups), dtype=float)


def _solve_c(grouped, target_rate: float, sd_mean: float, thresh: float = KC_RESPONSE_Z,
    bounds: Tuple[float, float] = (1e-3, 1e4)) -> float:
    """The conversion c at which the expected response rate of c x `grouped` (values, weights,
    n_units) through noise of sd `sd_mean` on a trial mean equals `target_rate`."""
    from scipy.optimize import brentq
    vals, w, n = grouped
    f = lambda lc: low_end_stats(np.exp(lc) * vals, sd_mean, thresh=thresh, weights=w, n_units=n)['rate'] - target_rate
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float(np.exp(lo) if abs(flo) < abs(fhi) else np.exp(hi))
    return float(np.exp(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)))


def z_scale(signal_on, target_rate: float, noise_sd: float, *, n_trials: int = 3, thresh: float = KC_RESPONSE_Z,
    bounds: Tuple[float, float] = (1e-3, 1e4)) -> float:
    """The conversion c of readout units to the data's z units: the factor at which the APL-on
    response rate under the data's criterion (trial mean of `n_trials` trials above `thresh` z,
    through independent Gaussian measurement noise of sd `noise_sd` per trial; `low_end_stats`)
    equals `target_rate` (e.g. `data_target('ctrl_frac')`, the control flies' 6.8 %).

    Args:
        signal_on: the APL-on readout (units x odors DataFrame or array), or per-trial readouts
            (n_recordings, n_trials, units, odors), whose trial means are taken first.

    Root of the (monotone) expected rate in log c (Brent); returns the nearer bound when the target
    is not bracketed.
    """
    sig = signal_on.to_numpy(dtype=float) if isinstance(signal_on, pd.DataFrame) else np.asarray(signal_on, dtype=float)
    if sig.ndim == 4:
        n_trials = sig.shape[1]
        sig = sig.mean(axis=1)                              # recordings x units x odors
        mu = np.concatenate([sig[r].T for r in range(sig.shape[0])])
    else:
        mu = sig.T                                          # odors x units
    return _solve_c(_unique_groups(mu), target_rate, float(noise_sd) / np.sqrt(n_trials), thresh, bounds)


def calibrate_low_end(on_counts, *, noise_sd: float, target_rate: float, target_band: float,
    target_pr: float, n_trials: int = 3, fit_floor: bool = True, K0: float = 10.0, floor0: float = 0.0,
    K_bounds: Tuple[float, float] = (1e-3, 1e3), floor_bounds: Tuple[float, float] = (-1.0, 50.0),
    c_bounds: Tuple[float, float] = (1e-2, 1e3), tol: float = 1e-5, max_iter: int = 50,
    band: Tuple[float, float] = LOW_END_BAND, thresh: float = KC_RESPONSE_Z) -> Dict[str, object]:
    """The saturating readout with a low end, calibrated jointly on the APL-on counts alone, so the
    APL-off amplitudes and the APL effect stay predictions: the half-saturation K, the one-spike
    floor b (`readout(..., 'saturating', K, floor=b)`) and the conversion c of readout units to z,
    through measurement noise of sd `noise_sd` per trial (`low_end_stats`, closed form), such that
    the APL-on response rate under the data's criterion equals `target_rate` (sets c), the share of
    entries with a trial mean in `band` equals `target_band` (sets b) and the responders-only
    participation ratio equals `target_pr` (sets K). Iterated to a fixed point: each round solves b
    (c re-solved at every b) and then K (c re-solved at every K) until K and b change by less than
    `tol` (relative for K, absolute for b). The band share need not be monotone in b (with c
    re-solved), so b is the root nearest to `floor0` among the sign changes on a grid over
    `floor_bounds`; K is the root of the (monotone) participation ratio in log K.

    Args:
        on_counts: APL-on spike counts, (units x odors) DataFrame or array, or per-trial counts
            (n_recordings, n_trials, units, odors) (jittered trials; `n_trials` is then taken from
            the array).
        fit_floor: False holds b at `floor0` (K and c still calibrated jointly).
        c_bounds: the range of c; floors at which the target rate needs c outside it are
            excluded, and when no floor inside the range reaches the band target, the one
            nearest to it is returned with `floor_at_bound` True.

    Returns dict: K, floor, c, iterations, converged, history (list of per-round dicts), the
    achieved rate / band_share / participation_ratio_resp, and K_at_bound / floor_at_bound.
    """
    arr = on_counts.to_numpy(dtype=float) if isinstance(on_counts, pd.DataFrame) else np.asarray(on_counts, dtype=float)
    if arr.ndim == 2:
        trials = arr.T[:, :, None]                          # groups (odors) x units x trials
    elif arr.ndim == 4:
        n_trials = arr.shape[1]
        trials = np.concatenate([np.moveaxis(arr[r], 0, -1).transpose(1, 0, 2) for r in range(arr.shape[0])])
    else:
        raise ValueError(f'on_counts must be (units, odors) or (recordings, trials, units, odors), not {arr.shape}')
    # identical count tuples read out identically: group them per odor (and recording)
    groups = []
    for g in range(trials.shape[0]):
        u, cnt = np.unique(trials[g], axis=0, return_counts=True)
        groups.append((u, cnt))
    n_units = np.array([trials.shape[1]] * trials.shape[0], dtype=float)
    width = max(len(c) for _, c in groups)
    counts = np.zeros((len(groups), width, trials.shape[2]))
    weights = np.zeros((len(groups), width))
    for g, (u, cnt) in enumerate(groups):
        counts[g, :len(cnt)] = u
        weights[g, :len(cnt)] = cnt
    sd_mean = float(noise_sd) / np.sqrt(n_trials)

    def signal(K, b):
        x = counts
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(x > 0, (x + b) / (x + b + K), 0.0)
        return r.mean(axis=2)

    def solve_c(K, b):
        return _solve_c((signal(K, b), weights, n_units), target_rate, sd_mean, thresh, c_bounds)

    def c_ok(c):
        return c_bounds[0] * 1.0001 < c < c_bounds[1] * 0.9999

    def band_err(bb):
        # nan where the rate cannot be reached with c inside its bounds (e.g. b near -1, where a
        # model tuned below the target rate reaches it only by inflating c without limit)
        c = solve_c(K, bb)
        return stats_at(K, bb, c)['band_share'] - target_band if c_ok(c) else np.nan

    def stats_at(K, b, c):
        return low_end_stats(c * signal(K, b), sd_mean, band=band, thresh=thresh, weights=weights, n_units=n_units)

    def root(f, lo, hi):
        flo, fhi = f(lo), f(hi)
        if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
            return (lo if abs(flo) < abs(fhi) else hi), True
        from scipy.optimize import brentq
        return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12)), False

    def root_nearest(f, grid, x0):
        # the band share need not be monotone in b once c is re-solved at every b (a model tuned
        # below the target rate reaches it only through its one-spike entries), so the root nearest
        # to the starting floor is taken among the sign changes on a grid
        vals = np.array([f(x) for x in grid])
        ok = np.isfinite(vals)
        brackets = [i for i in range(len(grid) - 1) if ok[i] and ok[i + 1] and vals[i] * vals[i + 1] <= 0]
        if not brackets:
            if not ok.any():
                # the target rate is out of reach at every floor (c at its bounds): keep the start
                return float(x0), True
            i = int(np.nanargmin(np.abs(np.where(ok, vals, np.nan))))
            return float(grid[i]), True
        from scipy.optimize import brentq
        roots = [float(brentq(f, grid[i], grid[i + 1], xtol=1e-10, rtol=1e-12)) if vals[i] != 0 else float(grid[i])
            for i in brackets]
        return min(roots, key=lambda r: abs(r - x0)), False

    b_grid = np.unique(np.concatenate([np.linspace(floor_bounds[0], 0.0, 21),
        np.geomspace(0.05, floor_bounds[1], 40)]))
    K, b = float(K0), float(floor0)
    history = []
    K_at_bound = floor_at_bound = False
    converged = False
    for it in range(1, max_iter + 1):
        K_prev, b_prev = K, b
        if fit_floor:
            b, floor_at_bound = root_nearest(band_err, b_grid, float(floor0))
        fk = lambda lk: stats_at(np.exp(lk), b, solve_c(np.exp(lk), b))['participation_ratio_resp'] - target_pr
        lk, K_at_bound = root(fk, np.log(K_bounds[0]), np.log(K_bounds[1]))
        K = float(np.exp(lk))
        c = solve_c(K, b)
        st = stats_at(K, b, c)
        history.append(dict(iteration=it, K=K, floor=b, c=c, **st))
        if abs(np.log(K) - np.log(K_prev)) < tol and abs(b - b_prev) < tol and it > 1 or (not fit_floor and it > 1):
            converged = True
            break
    return dict(K=K, floor=b, c=c, iterations=it, converged=converged, history=history, K_at_bound=K_at_bound,
        floor_at_bound=floor_at_bound, c_at_bound=not c_ok(c), noise_sd=float(noise_sd),
        n_trials=n_trials, **st)


def cosine_terms(mat: pd.DataFrame, thresh: float = KC_RESPONSE_Z) -> pd.DataFrame:
    """Each odor pair's cosine similarity across units split into three terms by unit: both
    entries responders (above `thresh`), one a responder and the other sub-threshold, both
    sub-threshold; each term is that part of the dot product divided by the product of the two
    odors' lengths, so the three add up to the cosine. Columns 'both', 'one', 'neither', 'cosine';
    index canonical pairs."""
    m = abbrev_columns(mat).astype(float)
    odors = odor_order(m.columns)
    X = m[odors].to_numpy()
    R = (X > thresh).astype(float)
    S = 1.0 - R
    norms = np.linalg.norm(X, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        den = np.outer(norms, norms)
        both = ((X * R).T @ (X * R)) / den
        one = ((X * R).T @ (X * S) + (X * S).T @ (X * R)) / den
        neither = ((X * S).T @ (X * S)) / den
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    rows = [(both[idx[a], idx[b]], one[idx[a], idx[b]], neither[idx[a], idx[b]]) for a, b in pl]
    out = pd.DataFrame(rows, columns=['both', 'one', 'neither'],
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))
    out['cosine'] = out.sum(axis=1)
    return out


_KC_NOISE_SD: Dict[str, float] = {}


def kc_trial_noise_sd(genotype: str = 'TNT3c') -> float:
    """The KC data's measurement noise: the trial-to-trial sd of the non-responding entries
    (trial mean <= 1 z), the root of the mean over those entries of their variance across the
    three trials (ddof 1), pooled over a genotype's recordings (Yang's per-trial values,
    `noise_floor_trial_distances.load_trials`). Cached."""
    if genotype not in _KC_NOISE_SD:
        from noise_floor_trial_distances import load_trials, recording_arrays
        var = []
        for by_odor in recording_arrays(load_trials(genotype)).values():
            X = np.stack([by_odor[o] for o in APL16_ODORS if o in by_odor], axis=-1)   # trials x KCs x odors
            v = X.var(axis=0, ddof=1)
            var.append(v[X.mean(axis=0) <= KC_RESPONSE_Z])
        _KC_NOISE_SD[genotype] = float(np.sqrt(np.concatenate(var).mean()))
    return _KC_NOISE_SD[genotype]


def kc_noise_floor(n_sd: float = 2.0) -> float:
    """The noise-defined floor of the 'floor' reading of the KC data: `n_sd` times the sd of a
    trial mean's noise, kc_trial_noise_sd('TNT3c') / sqrt(3) (the control flies' measurement
    noise, used for both genotypes and for the models, as every low-end parameter is set on the
    control flies)."""
    return float(n_sd * kc_trial_noise_sd('TNT3c') / np.sqrt(3))


# ---------------------------------------------------------------------------
# the recordings' per-KC normalization on the model's side (E73): Yang's values appear to be each
# KC's fluorescence divided by its sd over the whole recording (a z-score over the session), so a
# KC's response is relative to its own session; these read a model's spike counts the same way
ZCELL_MAPS = ('linear', 'saturating')
ZCELL_READINGS = ('graded', 'thresholded', 'binary')
OPK_EDGES = ((0, 0), (1, 1), (2, 3), (4, 7), (8, None))


def _zcell_amplitude(x: np.ndarray, amp: str, K: Optional[float]) -> np.ndarray:
    if amp == 'linear':
        return x
    if amp == 'saturating':
        if K is None or not K > 0:
            raise ValueError(f'the saturating map needs K > 0, not {K}')
        return x / (x + float(K))
    raise ValueError(f'unknown amplitude map {amp!r} (one of {ZCELL_MAPS})')


def session_zscore(counts, *, q: float, noise, amp: str = 'linear', K: Optional[float] = None,
    noise_ratio: float = 0.385, n_trials: int = 3, n_blank: int = 1, divisor: Optional[float] = None,
    center: bool = False, drift: float = 0.0) -> Dict[str, object]:
    """A model's KC responses read as the recordings appear to be: each KC's session (the panel's odors
    and `n_blank` blank stimuli, `n_trials` trials each) assembled from its spike counts and divided by the
    KC's sd over the whole session (E73).

    Each trial's response window holds the share `q` of the session's frames; inside it the KC's signal is
    its amplitude a (`amp`: 'linear', a = n; 'saturating', a = n / (n + K)), outside it 0; every frame
    carries independent noise of sd `noise` (amplitude units: the baseline-noise SD) and the trace a slow
    drift of sd drift x noise, which the pre-odor baseline removes from every response but which adds to the
    session's variance. The KC's session sd is S = sqrt(q mean(a^2) - (q mean(a))^2 + noise^2 (1 + drift^2))
    (means over the session's trials), and a trial reads
    a / S (the window relative to the pre-odor baseline, both in the KC's session z units), or with
    `center` (a - q mean(a)) / S (the window mean of the plain session z-score, no baseline subtracted),
    plus measurement noise of sd noise_ratio x noise / S (a silent KC's trial sd is `noise_ratio`). The
    per-KC mean of squared trial values cannot exceed about 1 / q (the edge of part A of E73). Yang's APL16
    responses are made this way (each KC's Fc z-scored over the whole recording, a trial's response the mean
    z 2-8 s after onset minus the mean over the 5 s before: add_to_reports/yang_response_definition.md), with
    the response windows covering a share 51 x 18 / 7194 = 0.128 of the recording. drift changes no reading
    when the noise is calibrated (noise sqrt(1 + drift^2) and noise_ratio / sqrt(1 + drift^2) at drift 0
    read identically); it sets the units, so that S / noise is the whole-trace SD over the baseline-noise SD.

    Args:
        counts: spike counts, (units x odors) DataFrame or array (the same counts on every trial), or
            (recordings, trials, units, odors) per-trial counts (jittered trials; each recording a session).
        q: share of the session's frames inside the response windows.
        noise: frame-noise sd in amplitude units, one value for all units (or an array over units).
        amp, K: the amplitude map (`ZCELL_MAPS`); K the saturating map's half-saturation.
        noise_ratio: sd of a response's measurement noise relative to the frame-noise sd.
        n_trials: trials per odor when `counts` is 2-D.
        n_blank: blank (solvent) stimuli in the session, amplitude 0.
        divisor: the control: every unit's amplitude divided by this one value, nothing subtracted (no
            per-KC normalization), measurement noise sd noise_ratio x noise / divisor.
        center: subtract the session mean (see above); ignored with `divisor`.
        drift: the slow drift's sd relative to `noise`.

    Returns dict: trials ((recordings, trials, units, odors) noise-free trial values in z; trials is 1
    for 2-D counts, whose trials are identical), center (their mean over trials, (recordings, units,
    odors)), trial_sd ((recordings, units) sd of one trial's measurement noise in z), n_trials, blank
    ((recordings, units) the blank's value in z), mean_sq ((recordings, units) expected per-unit mean of
    squared trial values over the session, the blank's included), scale_ratio ((recordings, units) S / noise,
    the whole-trace SD over the baseline-noise SD), odors (column labels or None), and the parameters.
    """
    odors = list(counts.columns) if isinstance(counts, pd.DataFrame) else None
    x = counts.to_numpy(dtype=float) if isinstance(counts, pd.DataFrame) else np.asarray(counts, dtype=float)
    if x.ndim == 2:
        x = x[None, None]
        T = int(n_trials)
    elif x.ndim == 4:
        T = x.shape[1]
    else:
        raise ValueError(f'counts must be (units, odors) or (recordings, trials, units, odors), not {x.shape}')
    a = _zcell_amplitude(x, amp, K)
    n_stim = a.shape[3] + int(n_blank)
    # session means over the trials: the blank trials add zeros
    mean_a = a.mean(axis=1).sum(axis=-1) / n_stim                      # recordings x units
    mean_a2 = (a ** 2).mean(axis=1).sum(axis=-1) / n_stim
    sig = np.broadcast_to(np.asarray(noise, dtype=float), mean_a.shape[-1:])[None, :]
    if divisor is not None:
        S = np.full(mean_a.shape, float(divisor))
        mu = np.zeros_like(mean_a)
    else:
        mu = q * mean_a
        S = np.sqrt(np.maximum(q * mean_a2 - mu ** 2, 0.0) + sig ** 2 * (1 + float(drift) ** 2))
        if not center:
            mu = np.zeros_like(mu)
    with np.errstate(invalid='ignore', divide='ignore'):
        vals = (a - mu[:, None, :, None]) / S[:, None, :, None]
        blank = -mu / S
        trial_sd = noise_ratio * sig / S
    vals = np.nan_to_num(vals)
    blank, trial_sd = np.nan_to_num(blank), np.nan_to_num(trial_sd)
    mean_sq = ((vals ** 2).mean(axis=1).sum(axis=-1) + n_blank * blank ** 2) / n_stim + trial_sd ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        scale_ratio = S / sig
    return dict(trials=vals, center=vals.mean(axis=1), trial_sd=trial_sd, n_trials=T, blank=blank, mean_sq=mean_sq,
        scale_ratio=scale_ratio, odors=odors, q=float(q), noise=noise, amp=amp, K=K, noise_ratio=float(noise_ratio),
        n_blank=int(n_blank), divisor=divisor, center_mean=bool(center), drift=float(drift))


def _tail_moments(m: np.ndarray, sd: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(x > thresh), E[x 1{x > thresh}] and E[x^2 1{x > thresh}] for x ~ N(m, sd^2), elementwise
    (sd broadcast against m; sd 0 is the noise-free indicator)."""
    from scipy.special import ndtr
    m = np.asarray(m, dtype=float)
    sd = np.broadcast_to(np.asarray(sd, dtype=float), m.shape)
    pos = sd > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        a = np.where(pos, (thresh - m) / np.where(pos, sd, 1.0), 0.0)
        P = np.where(pos, ndtr(-a), (m > thresh).astype(float))
        phi = np.exp(-0.5 * a ** 2) / np.sqrt(2 * np.pi)
        E1 = np.where(pos, m * P + sd * phi, m * P)
        E2 = np.where(pos, (m ** 2 + sd ** 2) * P + sd * phi * (m + thresh), m ** 2 * P)
    return P, E1, E2


def _zcell_odors(z: Dict[str, object], n_odors: int) -> List[str]:
    odors = z.get('odors')
    if odors is None:
        if n_odors != len(APL16_ODORS):
            raise ValueError('a reading of an unlabeled array needs 16 columns (the APL16 odors)')
        return list(APL16_ODORS)
    return [odor_abbrev(o) for o in odors]


def zcell_pair_distances(z: Dict[str, object], reading: str = 'graded', thresh: float = KC_RESPONSE_Z) -> pd.Series:
    """Cosine odor-odor distances of a `session_zscore` reading, in expectation over its measurement noise
    (closed form, the ratio of the expectations as `expected_pair_distances_noisy`: independent noise adds
    to each odor's squared length and nothing to the products of different odors), mean over its
    recordings: 'graded' (all values, negative ones included: the data's cosine), 'thresholded' (trial
    means <= `thresh` set to 0) or 'binary' (trial mean > `thresh`). Canonical pair Series."""
    C = z['center']
    s = z['trial_sd'] / np.sqrt(z['n_trials'])
    odors = _zcell_odors(z, C.shape[2])
    order = odor_order(odors)
    idx = [odors.index(o) for o in order]
    D = []
    for r in range(C.shape[0]):
        m = C[r][:, idx]
        sd = s[r][:, None]
        if reading == 'graded':
            E1, E2 = m, m ** 2 + sd ** 2
        elif reading in ('thresholded', 'binary'):
            P, E1, E2 = _tail_moments(m, sd, thresh)
            if reading == 'binary':
                E1 = E2 = P
        else:
            raise ValueError(f'unknown reading {reading!r} (one of {ZCELL_READINGS})')
        n2 = E2.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            D.append(1 - (E1.T @ E1) / np.sqrt(np.outer(n2, n2)))
    D = np.nanmean(np.stack(D), axis=0) if len(D) > 1 else D[0]
    pl = pair_list(order)
    pos = {o: i for i, o in enumerate(order)}
    return pd.Series([D[pos[a], pos[b]] for a, b in pl], index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))


def poisson_binomial_bins(P: np.ndarray, edges=OPK_EDGES) -> np.ndarray:
    """Mean over units of the distribution of the number of odors a unit responds to, each odor
    independently with its probability (P: units x odors), summed into `edges` bins ((lo, hi) inclusive,
    hi None for no upper limit): the expected odors-per-KC distribution."""
    P = np.asarray(P, dtype=float)
    n_u, n_o = P.shape
    dist = np.zeros((n_u, n_o + 1))
    dist[:, 0] = 1.0
    for j in range(n_o):
        p = P[:, j:j + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - p) + dist[:, :-1] * p
        dist[:, 0] = dist[:, 0] * (1 - p[:, 0])
    mean = dist.mean(axis=0)
    return np.array([mean[lo:(n_o + 1 if hi is None else hi + 1)].sum() for lo, hi in edges])


def zcell_stats(z: Dict[str, object], thresh: float = KC_RESPONSE_Z) -> Dict[str, object]:
    """Statistics of a `session_zscore` reading under the data's criterion (trial mean > `thresh`), in
    expectation over the measurement noise, pooled over its recordings: `rate` (share of entries
    responding) and `rate_per_odor` (Series), `resp_amp_mean` and `resp_amp_cv` (the responders' trial
    means), `share_above_4` and `share_05_15` (entries with a trial mean above 4 z, and in (0.5, 1.5] z),
    `opk` (the odors-per-KC distribution over 0, 1, 2-3, 4-7, 8 or more odors), `silent_share` and
    `opk_8plus`, `nonresp_trial_sd` (root mean trial-to-trial variance of the non-responding entries, as
    `kc_trial_noise_sd` measures the data: measurement noise plus the trials' own variation), and the
    per-unit mean square's upper edge (`mean_sq_max`, `_p999`, `_p99`, `_median`, `edge_gap` = max - 99.9th
    percentile), `max_trial_mean` and `max_trial` (noise-free); by breadth (each unit at its expected number
    of odors responded to, rounded, 11 or more pooled), `scale_ratio_by_breadth` (the median whole-trace SD
    over the baseline-noise SD) and `amp_base_by_breadth` (the responders' mean amplitude in baseline-noise
    units, z x that ratio), the checks against the KC data's (notes/yang_response_normalization/)."""
    from scipy.special import ndtr
    C = z['center']
    T = z['n_trials']
    t = z['trial_sd']
    sd = t / np.sqrt(T)
    sdb = sd[:, :, None]
    P, E1, E2 = _tail_moments(C, sdb, thresh)
    odors = _zcell_odors(z, C.shape[2])
    with np.errstate(invalid='ignore', divide='ignore'):
        amp_mean = float(E1.sum() / P.sum())
        amp_var = float(E2.sum() / P.sum()) - amp_mean ** 2
        P4 = _tail_moments(C, sdb, 4.0)[0]
        band = ndtr((1.5 - C) / np.where(sdb > 0, sdb, np.inf)) - ndtr((0.5 - C) / np.where(sdb > 0, sdb, np.inf))
    if not (sd > 0).all():
        band = np.where(sdb > 0, band, ((C > 0.5) & (C <= 1.5)).astype(float))
    # the trials' own variation (jittered trials) adds to the measurement noise
    tr = z['trials']
    var_tr = tr.var(axis=1, ddof=1) if tr.shape[1] > 1 else np.zeros_like(C)
    w = 1 - P
    nonresp_var = float((w * (t[:, :, None] ** 2 + var_tr)).sum() / w.sum())
    opk = np.mean([poisson_binomial_bins(P[r]) for r in range(P.shape[0])], axis=0)
    ms = z['mean_sq'].ravel()
    breadth = np.minimum(np.rint(P.sum(axis=2)), 11).ravel()
    sr = np.asarray(z['scale_ratio'], dtype=float).ravel()
    e1b = (E1 * z['scale_ratio'][:, :, None]).sum(axis=2).ravel()
    pb = P.sum(axis=2).ravel()
    sr_b, amp_b = {}, {}
    for b in range(12):
        m = breadth == b
        if m.any():
            sr_b[b] = float(np.median(sr[m]))
            if b > 0 and pb[m].sum() > 0:
                amp_b[b] = float(e1b[m].sum() / pb[m].sum())
    return dict(rate=float(P.mean()), rate_per_odor=pd.Series(P.mean(axis=(0, 1)), index=odors),
        resp_amp_mean=amp_mean, resp_amp_cv=float(np.sqrt(max(amp_var, 0.0)) / amp_mean) if amp_mean > 0 else np.nan,
        share_above_4=float(P4.mean()), share_05_15=float(band.mean()), opk=opk, silent_share=float(opk[0]),
        opk_8plus=float(opk[-1]), nonresp_trial_sd=float(np.sqrt(nonresp_var)), mean_sq_max=float(ms.max()),
        mean_sq_p999=float(np.quantile(ms, 0.999)), mean_sq_p99=float(np.quantile(ms, 0.99)),
        mean_sq_median=float(np.median(ms)), edge_gap=float(ms.max() - np.quantile(ms, 0.999)),
        max_trial_mean=float(C.max()), max_trial=float(tr.max()), scale_ratio_by_breadth=sr_b, amp_base_by_breadth=amp_b)


def _zcell_rate(on_counts, q, noise, amp, K, noise_ratio, n_trials, n_blank, center, drift=0.0, thresh=KC_RESPONSE_Z
    ) -> Tuple[float, Dict]:
    z = session_zscore(on_counts, q=q, noise=noise, amp=amp, K=K, noise_ratio=noise_ratio, n_trials=n_trials,
        n_blank=n_blank, center=center, drift=drift)
    P = _tail_moments(z['center'], (z['trial_sd'] / np.sqrt(z['n_trials']))[:, :, None], thresh)[0]
    return float(P.mean()), z


def calibrate_session_zscore(on_counts, *, q: float, amp: str = 'linear', target_rate: float,
    target_amp: Optional[float] = None, target_silent_sd: Optional[float] = None, K: Optional[float] = None,
    noise_ratio: float = 0.385, n_trials: int = 3, n_blank: int = 1, center: bool = False, drift: float = 0.0,
    noise_bounds: Tuple[float, float] = (1e-4, 1e4), K_bounds: Tuple[float, float] = (1e-2, 1e3), n_K_grid: int = 25,
    tol: float = 1e-4, max_iter: int = 20) -> Dict[str, object]:
    """The parameters of `session_zscore` set on the APL-on counts alone (then used unchanged for APL off,
    so the APL effect stays a prediction): the frame noise so that the APL-on response rate under the
    data's criterion (expected trial mean > 1 z) is `target_rate`; for the saturating map with
    `target_amp`, also K, so that the responders' mean amplitude (z) is `target_amp`; with
    `target_silent_sd`, the noise ratio so that the trial-to-trial sd of the non-responding entries
    (`zcell_stats`' `nonresp_trial_sd`) is that value (iterated to a fixed point; it barely moves the other
    two). The linear map has the noise as its only free parameter, so its responders' amplitude is a
    prediction. The rate falls monotonically with the noise (root in log noise, Brent); K is the root of the
    amplitude error nearest the grid point with the smallest error among the sign changes on a log grid
    over `K_bounds` (the amplitude, with the noise re-solved at every K, need not be monotone in K), or the
    grid point with the smallest error when there is no sign change (`K_at_bound`).

    Args:
        on_counts: APL-on spike counts, (units x odors) or (recordings, trials, units, odors).
        K: the saturating map's K when it is not calibrated (no `target_amp`).

    Returns dict: noise, K, noise_ratio, rate, resp_amp_mean, nonresp_trial_sd (achieved), noise_at_bound,
    K_at_bound, converged, iterations, q, amp, center.
    """
    from scipy.optimize import brentq
    lo_n, hi_n = np.log(noise_bounds[0]), np.log(noise_bounds[1])

    def solve_noise(Kv, nr):
        f = lambda ln: _zcell_rate(on_counts, q, np.exp(ln), amp, Kv, nr, n_trials, n_blank, center, drift)[0] - target_rate
        flo, fhi = f(lo_n), f(hi_n)
        if flo * fhi > 0:
            return float(np.exp(lo_n if abs(flo) < abs(fhi) else hi_n)), True
        return float(np.exp(brentq(f, lo_n, hi_n, xtol=1e-10, rtol=1e-10))), False

    def stats_at(Kv, nr):
        noise, nb = solve_noise(Kv, nr)
        z = session_zscore(on_counts, q=q, noise=noise, amp=amp, K=Kv, noise_ratio=nr, n_trials=n_trials,
            n_blank=n_blank, center=center, drift=drift)
        return noise, nb, z

    def amp_of(z):
        C = z['center']
        sd = (z['trial_sd'] / np.sqrt(z['n_trials']))[:, :, None]
        P, E1, _ = _tail_moments(C, sd, KC_RESPONSE_Z)
        return float(E1.sum() / P.sum()) if P.sum() > 0 else np.nan

    fit_K = amp == 'saturating' and target_amp is not None
    if amp == 'saturating' and not fit_K and K is None:
        raise ValueError('the saturating map needs K, or target_amp to calibrate it')
    nr = float(noise_ratio)
    Kv = K
    K_at_bound = False
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        fk = lambda lk: amp_of(stats_at(np.exp(lk), nr)[2]) - target_amp
        if fit_K and it > 1 and not K_at_bound:
            # later rounds (the noise ratio moved a little): the root near the previous K
            lo_k, hi_k = np.log(Kv) - 0.7, np.log(Kv) + 0.7
            if fk(lo_k) * fk(hi_k) <= 0:
                Kv = float(np.exp(brentq(fk, lo_k, hi_k, xtol=1e-8, rtol=1e-8)))
            else:
                K_at_bound = True
        if fit_K and (it == 1 or K_at_bound):
            grid = np.linspace(np.log(K_bounds[0]), np.log(K_bounds[1]), n_K_grid)
            err = np.array([amp_of(stats_at(np.exp(g), nr)[2]) - target_amp for g in grid])
            ok = np.isfinite(err)
            best = int(np.nanargmin(np.where(ok, np.abs(err), np.nan)))
            brackets = [i for i in range(len(grid) - 1) if ok[i] and ok[i + 1] and err[i] * err[i + 1] <= 0]
            if brackets:
                i = min(brackets, key=lambda i: abs(i + 0.5 - best))
                Kv = float(np.exp(brentq(fk, grid[i], grid[i + 1], xtol=1e-8, rtol=1e-8)))
                K_at_bound = False
            else:
                Kv, K_at_bound = float(np.exp(grid[best])), True
        noise, noise_at_bound, z = stats_at(Kv, nr)
        if target_silent_sd is None:
            converged = True
            break
        st = zcell_stats(z)
        new_nr = nr * float(target_silent_sd) / st['nonresp_trial_sd']
        if abs(new_nr - nr) < tol * nr:
            nr = new_nr
            noise, noise_at_bound, z = stats_at(Kv, nr)
            converged = True
            break
        nr = new_nr
    st = zcell_stats(z)
    return dict(noise=noise, K=Kv, noise_ratio=nr, rate=st['rate'], resp_amp_mean=st['resp_amp_mean'],
        nonresp_trial_sd=st['nonresp_trial_sd'], noise_at_bound=bool(noise_at_bound), K_at_bound=bool(K_at_bound),
        converged=converged, iterations=it, q=float(q), amp=amp, center=bool(center), drift=float(drift))


# ---------------------------------------------------------------------------
# the per-pair APL effect split by the number of odors each KC responds to (E74): a pair's cosine
# similarity is a sum over units, so each condition's similarity splits into the contributions of the
# units in each odors-per-KC class (classes set per condition), and the APL effect (APL-on distance minus
# APL-off distance = APL-off similarity minus APL-on similarity) into per-class effects
OPK_CLASSES = ('0', '1', '2_3', '4_7', '8plus')
CLASS_TERMS = ('both', 'one', 'neither', 'total')


def opk_class_labels(edges=OPK_EDGES) -> List[str]:
    """Labels of odors-per-KC classes ((lo, hi) inclusive, hi None for no upper limit): '0', '2_3',
    '8plus', ...; `OPK_CLASSES` for `OPK_EDGES`."""
    return [f'{lo}' if hi == lo else (f'{lo}plus' if hi is None else f'{lo}_{hi}') for lo, hi in edges]


def _ordered_matrix(mat, odors: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str], List[int]]:
    """(units x odors float array with its columns in `odor_order`, that order, the column positions taken)
    of a DataFrame (odor labels abbreviated) or an array whose columns are `odors` (default `APL16_ODORS`)."""
    if isinstance(mat, pd.DataFrame):
        labels = [odor_abbrev(c) for c in mat.columns]
        X = mat.to_numpy(dtype=float)
    else:
        X = np.asarray(mat, dtype=float)
        labels = [odor_abbrev(o) for o in (APL16_ODORS if odors is None else odors)]
    if X.ndim != 2 or X.shape[1] != len(labels) or len(set(labels)) != len(labels):
        raise ValueError(f'a (units x odors) matrix with {len(labels)} distinct odor columns, not {X.shape}')
    order = odor_order(labels)
    idx = [labels.index(o) for o in order]
    return X[:, idx], order, idx


def class_cosine_terms_array(X: np.ndarray, R: np.ndarray, edges=OPK_EDGES) -> np.ndarray:
    """The core of `class_cosine_terms` on arrays: X (units x odors) values, R (units x odors) bool
    responders, columns in the order the pairs are wanted in (pairs (i, j), i < j, row-major). Returns
    (pairs, classes + 1, 4): per pair, each class's contribution to the cosine similarity (the last class
    all units), split into the terms `CLASS_TERMS`."""
    X = np.asarray(X, dtype=float)
    R = np.asarray(R, dtype=bool)
    n_kc = R.sum(axis=1)
    XR = np.where(R, X, 0.0)
    XS = X - XR
    norms = np.sqrt((X ** 2).sum(axis=0))
    iu = np.triu_indices(X.shape[1], 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        den = np.outer(norms, norms)[iu]
    masks = [(n_kc >= lo) & (n_kc <= (X.shape[1] if hi is None else hi)) for lo, hi in edges]
    masks.append(np.ones(X.shape[0], dtype=bool))
    out = np.empty((len(iu[0]), len(masks), len(CLASS_TERMS)))
    for c, m in enumerate(masks):
        r, s = XR[m], XS[m]
        rs = r.T @ s
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, c, 0] = (r.T @ r)[iu] / den
            out[:, c, 1] = (rs + rs.T)[iu] / den
            out[:, c, 2] = (s.T @ s)[iu] / den
    out[:, :, 3] = out[:, :, :3].sum(axis=2)
    return out


def opk_class_shares(resp, edges=OPK_EDGES) -> np.ndarray:
    """Share of units in each odors-per-KC class of a (units x odors) bool responder matrix."""
    R = np.asarray(resp.to_numpy() if isinstance(resp, pd.DataFrame) else resp, dtype=bool)
    n_kc = R.sum(axis=1)
    return np.array([np.mean((n_kc >= lo) & (n_kc <= (R.shape[1] if hi is None else hi))) for lo, hi in edges])


def _class_columns(edges) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([opk_class_labels(edges) + ['all'], list(CLASS_TERMS)], names=['class', 'term'])


def _responders(X: np.ndarray, resp, idx: List[int], order: List[str], thresh: float) -> np.ndarray:
    """The responders of an `_ordered_matrix` X: X > thresh, or `resp` in X's column order (a DataFrame by its
    odor labels, an array by the positions `idx` of the matrix it belongs to)."""
    if resp is None:
        return X > thresh
    if isinstance(resp, pd.DataFrame):
        R = abbrev_columns(resp)[order].to_numpy(dtype=bool)
    else:
        R = np.asarray(resp, dtype=bool)[:, idx]
    if R.shape != X.shape:
        raise ValueError(f'responders {R.shape} for a matrix {X.shape}')
    return R


def class_cosine_terms(mat, resp=None, *, thresh: float = KC_RESPONSE_Z, edges=OPK_EDGES,
    odors: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Each odor pair's cosine similarity across units split by the units' odors-per-KC class and, within a
    class, into `cosine_terms`' three terms: both entries responders ('both'), one a responder and the other
    sub-threshold ('one'), both sub-threshold ('neither'), with their sum ('total'). Each part is that part
    of the dot product over the product of the two odors' lengths over all units, so the classes' totals add
    up to the cosine (class 'all', computed over all units).

    Args:
        mat: (units x odors) values: a DataFrame (odor labels, abbreviated) or an array whose columns are
            `odors` (default `APL16_ODORS`).
        resp: (units x odors) responders, in `mat`'s column order (a bool array or DataFrame); default
            `mat > thresh` (e.g. a model read out as n / (n + K) whose responders are its two-spike entries
            passes resp=counts >= 2).
        thresh: the response criterion when `resp` is None.
        edges: the classes, (lo, hi) inclusive numbers of odors responded to (hi None: no upper limit).

    Returns DataFrame indexed by canonical pairs, columns (class, term) (`opk_class_labels(edges)` + 'all'
    x `CLASS_TERMS`); NaN for pairs with an odor of zero length.
    """
    X, order, idx = _ordered_matrix(mat, odors)
    R = _responders(X, resp, idx, order, thresh)
    arr = class_cosine_terms_array(X, R, edges)
    pl = pair_list(order)
    return pd.DataFrame(arr.reshape(len(pl), -1), index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']),
        columns=_class_columns(edges))


def apl_effect_by_class(on, off, *, resp_on=None, resp_off=None, thresh_on: float = KC_RESPONSE_Z,
    thresh_off: float = KC_RESPONSE_Z, edges=OPK_EDGES, odors: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The per-pair APL effect split by odors-per-KC class (E74): each condition's `class_cosine_terms`
    (classes set by that condition's responses), averaged over its recordings, and the effect, APL-off
    similarity minus APL-on similarity (= APL-on distance minus APL-off distance, as `apl_effect`), per class
    and term. Class 'all' term 'total' of the effect is `apl_effect` of the two conditions' cosine distances
    (mean over recordings).

    Args:
        on, off: each condition's (units x odors) values, or a list of them (recordings; averaged with
            equal weight, NaN pairs of a recording skipped).
        resp_on, resp_off: each condition's responders (one per matrix, as `class_cosine_terms`' `resp`), or
            None for values above `thresh_on` / `thresh_off`.
        edges, odors: as `class_cosine_terms`.

    Returns DataFrame indexed by canonical pairs, columns (block, class, term), block 'on', 'off' (the
    similarity contributions) and 'effect' (off minus on); attrs['class_shares'] {'on', 'off'}: the share of
    units in each class (mean over recordings).
    """
    blocks, shares = {}, {}
    for cond, mats, resps, thresh in (('on', on, resp_on, thresh_on), ('off', off, resp_off, thresh_off)):
        if isinstance(mats, (list, tuple)):
            resps = [None] * len(mats) if resps is None else list(resps)
            if len(resps) != len(mats):
                raise ValueError(f'{cond}: {len(mats)} matrices but {len(resps)} responder matrices')
        else:
            mats, resps = [mats], [resps]
        arrs, sh, order = [], [], None
        for m, r in zip(mats, resps):
            X, o, idx = _ordered_matrix(m, odors)
            if order is not None and o != order:
                raise ValueError('the matrices have different odors')
            order = o
            R = _responders(X, r, idx, o, thresh)
            arrs.append(class_cosine_terms_array(X, R, edges))
            sh.append(opk_class_shares(R, edges))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            a = np.nanmean(np.stack(arrs), axis=0) if len(arrs) > 1 else arrs[0]
        pl = pair_list(order)
        blocks[cond] = pd.DataFrame(a.reshape(len(pl), -1), index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']),
            columns=_class_columns(edges))
        shares[cond] = pd.Series(np.mean(sh, axis=0), index=opk_class_labels(edges))
    if not blocks['on'].index.equals(blocks['off'].index):
        raise ValueError('the two conditions have different odors')
    blocks['effect'] = blocks['off'] - blocks['on']
    out = pd.concat(blocks, axis=1, names=['block'])
    out.attrs['class_shares'] = shares
    return out


def resample_recording_means(rec_rdms: Dict[str, pd.DataFrame], rng: np.random.Generator,
    *, unit: str = 'recording') -> Dict[str, pd.Series]:
    """One bootstrap replicate of {genotype: mean pair Series}.

    Args:
        rec_rdms: {genotype: per-recording pair DataFrame} (rows (date, fly_num,
            hemibrain), as `yang_recording_rdms`)
        unit: 'recording' draws recordings with replacement within each genotype;
            'fly' draws flies (date, fly_num) with replacement, keeping all hemispheres
            of each drawn fly (the two hemispheres of a fly are not independent)

    Recordings are then averaged with equal weight, as in the report's recipe.
    """
    means = {}
    for g, df in rec_rdms.items():
        if unit == 'recording':
            idx = rng.integers(0, len(df), len(df))
            means[g] = df.iloc[idx].mean()
        elif unit == 'fly':
            flies = df.index.droplevel('hemibrain').unique()
            drawn = flies[rng.integers(0, len(flies), len(flies))]
            parts = [df.xs(f, level=('date', 'fly_num'), drop_level=False) for f in drawn]
            means[g] = pd.concat(parts).mean()
        else:
            raise ValueError(unit)
    return means


def bootstrap_recordings(fn: Callable[[Dict[str, pd.Series]], Dict[str, float]],
    rec_rdms: Dict[str, pd.DataFrame], *, n_boot: int = 1000, seed: int = 0,
    ci: Tuple[float, float] = (5, 95), unit: str = 'recording') -> pd.DataFrame:
    """Bootstraps statistics over recordings (or flies).

    Args:
        fn: maps {genotype: mean pair Series} -> {stat: value}
        rec_rdms: {genotype: per-recording pair DataFrame} (`yang_recording_rdms`)
        n_boot: resamples (drawn with replacement within each genotype)
        unit: 'recording' or 'fly' (see `resample_recording_means`)

    Returns DataFrame indexed by stat: observed, boot_mean, boot_sd, lo, hi.
    """
    rng = np.random.default_rng(seed)
    observed = fn({g: df.mean() for g, df in rec_rdms.items()})
    draws = {k: [] for k in observed}
    for _ in range(n_boot):
        means = resample_recording_means(rec_rdms, rng, unit=unit)
        for k, v in fn(means).items():
            draws[k].append(v)
    rows = {}
    for k, v in observed.items():
        arr = np.asarray(draws[k], dtype=float)
        rows[k] = dict(observed=v, boot_mean=np.nanmean(arr), boot_sd=np.nanstd(arr),
            lo=np.nanpercentile(arr, ci[0]), hi=np.nanpercentile(arr, ci[1])
        )
    return pd.DataFrame(rows).T


def split_half_reliability(rec_rdms: pd.DataFrame, *, n_splits: int = 200, seed: int = 0,
    method: str = 'spearman') -> Dict[str, float]:
    """Reliability of a mean pair-distance matrix, from random split halves of recordings.

    Returns split-half correlation (mean over splits, 5th / 95th percentiles), the
    Spearman-Brown estimate for the full set of recordings (2r / (1 + r)), and
    `ceiling`: sqrt of that, the expected correlation of a noise-free model that matched
    the true mean matrix perfectly with the observed mean.
    """
    rng = np.random.default_rng(seed)
    n = len(rec_rdms)
    corr = stats.spearmanr if method == 'spearman' else stats.pearsonr
    vals = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a = rec_rdms.iloc[idx[:n // 2]].mean()
        b = rec_rdms.iloc[idx[n // 2:]].mean()
        vals.append(corr(a, b)[0])
    vals = np.asarray(vals)
    r = float(np.nanmean(vals))
    full = 2 * r / (1 + r)
    return dict(n_recordings=n, split_half=r, split_half_lo=float(np.nanpercentile(vals, 5)),
        split_half_hi=float(np.nanpercentile(vals, 95)), spearman_brown_full=full,
        ceiling=float(np.sqrt(full)),
    )


# ---------------------------------------------------------------------------
# per-recording KC x odor matrices and RDM pipeline variants (E7; atlas)
RECORDING_KEYS: List[str] = ['date', 'fly_num', 'hemibrain']


def yang_recording_matrices(genotype: str, with_pfo: bool = False
    ) -> Dict[Tuple[str, int, str], pd.DataFrame]:
    """{(date, fly_num, hemibrain): KCs x odors trial-mean matrix} for one genotype.

    Odor columns are in `APL16_ODORS` order (plus `SOLVENT` last if `with_pfo`). Values
    are Yang's z-scored fluorescence trial means (negatives included).
    """
    tm = yang_trialmeans()
    tm = tm[tm.genotype == genotype]
    odors = list(APL16_ODORS) + ([SOLVENT] if with_pfo else [])
    mats = {}
    for rec, sub in tm.groupby(RECORDING_KEYS, sort=True):
        mat = sub.pivot(index='roi', columns='odor', values='value')[odors]
        assert not mat.isna().any().any(), rec
        mats[(str(rec[0]), int(rec[1]), str(rec[2]))] = mat
    return mats


def transform_matrix(mat: pd.DataFrame, *, values: str = 'graded', cells: str = 'all',
    thresh: float = 1.0) -> pd.DataFrame:
    """One RDM pipeline variant applied to a KCs x odors matrix.

    Args:
        values: 'graded' (as is), 'binary' (response > thresh -> 1.0 else 0.0), or
            'thresholded' (graded values kept where > thresh, else 0)
        cells: 'all' KCs, or 'responsive' (KCs with a response > thresh to at least one
            odor column of `mat`)
        thresh: response threshold (Yang's data are z-scores; she has used 0.75 and 1)
    """
    m = mat.astype(float)
    if cells == 'responsive':
        m = m.loc[(m > thresh).any(axis='columns')]
    elif cells != 'all':
        raise ValueError(cells)
    if values == 'binary':
        m = (m > thresh).astype(float)
    elif values == 'thresholded':
        m = m.where(m > thresh, 0.0)
    elif values != 'graded':
        raise ValueError(values)
    return m


def _stack_recordings(rows: Dict[tuple, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=RECORDING_KEYS)
    return df


def recording_rdms_from_matrices(mats: Dict[tuple, pd.DataFrame], *,
    measure: str = 'cosine', drop_silent: bool = False, **transform_kws) -> pd.DataFrame:
    """Per-recording pair distances (rows recordings, columns canonical pairs) for one
    pipeline variant (`transform_kws` -> `transform_matrix`; `measure`, `drop_silent` ->
    `pair_distances`). Defaults reproduce `yang_recording_rdms(genotype)`.
    """
    return _stack_recordings({rec: pair_distances(transform_matrix(m, **transform_kws),
        measure=measure, drop_silent=drop_silent) for rec, m in mats.items()
    })


def recording_response_fractions(mats: Dict[tuple, pd.DataFrame], thresh: float = 1.0
    ) -> pd.DataFrame:
    """Rows recordings, columns odors: fraction of KCs with response > thresh."""
    return _stack_recordings({rec: (m > thresh).mean(axis='index') for rec, m in mats.items()})


def match_threshold(mats: Dict[tuple, pd.DataFrame], target_frac: float, *,
    lo: float = -1.0, hi: float = 8.0, tol: float = 1e-4) -> float:
    """Threshold at which the recording-mean response fraction (mean over odors) equals
    `target_frac`, by bisection. Pass a single-recording dict to match one recording.
    """
    def frac(th: float) -> float:
        return float(recording_response_fractions(mats, th).mean(axis='columns').mean())

    assert frac(lo) > target_frac > frac(hi), (frac(lo), target_frac, frac(hi))
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if frac(mid) > target_frac:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def density_null_similarity(mat: pd.DataFrame) -> pd.Series:
    """Expected cosine similarity between odor columns of a KCs x odors matrix if KC
    identities were shuffled independently within each odor (per-odor response
    distributions, hence response counts, kept).

    Exact permutation expectation: E[a . b] = N mean(a) mean(b), so
    E[cos] = N mean(a) mean(b) / (|a| |b|); for binary columns this is sqrt(p_a p_b).
    Returned as a canonical pair Series of similarities (not distances).
    """
    m = abbrev_columns(mat).astype(float)
    odors = odor_order(m.columns)
    X = m[odors].to_numpy().T
    n = X.shape[1]
    means = X.mean(axis=1)
    norms = np.linalg.norm(X, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = n * np.outer(means, means) / np.outer(norms, norms)
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([S[idx[a], idx[b]] for a, b in pl],
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b'])
    )


def recording_density_null_rdms(mats: Dict[tuple, pd.DataFrame], **transform_kws
    ) -> pd.DataFrame:
    """Per-recording expected 1 - cosine under the within-odor KC-identity shuffle null."""
    return _stack_recordings({rec: 1 - density_null_similarity(
        transform_matrix(m, **transform_kws)) for rec, m in mats.items()
    })


def shuffled_pair_distances(mat: pd.DataFrame, rng: np.random.Generator, *,
    measure: str = 'cosine', **transform_kws) -> pd.Series:
    """Pair distances after independently permuting KC identities within each odor
    column (one draw from the density null; `density_null_similarity` is its mean)."""
    m = transform_matrix(mat, **transform_kws)
    X = m.to_numpy().copy()
    for j in range(X.shape[1]):
        X[:, j] = X[rng.permutation(X.shape[0]), j]
    return pair_distances(pd.DataFrame(X, index=m.index, columns=m.columns), measure=measure)


# ---------------------------------------------------------------------------
# the three repeats of the recordings (E81, the flies' variability across the three repeats; atlas):
# single-trial distances by kind of trial pair, the same-repeat gap, noise-free distances from products
# of different repeats, and the parts of a response's variability across repeats. A recording (or a
# model's pseudo-recording) is an array (n_repeats, units, odors) of single-trial values; in the model,
# whose trials are exchangeable, the trial index stands in for the repeat.
REPEAT_KINDS: Tuple[str, ...] = ('same_repeat', 'different_repeat', 'adjacent', 'first_third')
TRIAL_VALUES: Tuple[str, ...] = ('graded', 'thresholded', 'binary')
YANG_DATASETS: Dict[str, int] = {'TNT': 19, 'TNT3c': 20}
_YANG_TRIALS: Dict[tuple, Dict[tuple, Tuple[np.ndarray, List[str]]]] = {}


def yang_recording_trials(genotype: str, units: str = 'z', odors: Optional[Sequence[str]] = None
    ) -> Dict[Tuple[str, int, str], np.ndarray]:
    """One genotype's single-trial KC responses, per recording.

    Args:
        genotype: 'TNT3c' (control flies) or 'TNT' (APL output blocked (TNT)).
        units: 'z' (Yang's whole-recording z, olfdb's values: data/yang_apl16_kc_olfdb/<genotype>/
            dataset_<id>_long.csv) or 'noise' (baseline-noise units: data/yang_apl16_kc_noise_units/
            per_trial_long.parquet, the realism bands' units).
        odors: odor abbreviations, in the order of the last axis (default `APL16_ODORS`; 'pfo' is the
            solvent).

    Returns {(date, fly_num, hemibrain): array (n_repeats, KCs, odors)}, repeats in the order of olfdb's
    `repeat` (taken as the repeat's number in the session), KCs sorted by roi. Cached per (genotype,
    units); the arrays returned are copies.
    """
    odors = list(APL16_ODORS if odors is None else odors)
    key = (genotype, units)
    if key not in _YANG_TRIALS:
        cols = RECORDING_KEYS + ['roi', 'odor', 'repeat', 'value']
        if units == 'z':
            d = pd.read_csv(yang_data_dir() / genotype / f'dataset_{YANG_DATASETS[genotype]}_long.csv',
                usecols=cols)
        elif units == 'noise':
            d = pd.read_parquet(shared_dir() / 'data' / 'yang_apl16_kc_noise_units' / 'per_trial_long.parquet',
                columns=['genotype'] + cols)
            d = d[d['genotype'] == genotype].drop(columns='genotype')
        else:
            raise ValueError(f"units must be 'z' or 'noise', not {units!r}")
        d['abbrev'] = d['odor'].map(odor_abbrev)
        out = {}
        for rec, sub in d.groupby(RECORDING_KEYS, sort=True):
            w = sub.set_index(['repeat', 'roi', 'abbrev'])['value'].unstack('abbrev').sort_index()
            reps = sorted(w.index.get_level_values('repeat').unique())
            rois = sorted(w.index.get_level_values('roi').unique())
            w = w.reindex(pd.MultiIndex.from_product([reps, rois], names=['repeat', 'roi']))
            if w.isna().any().any():
                raise ValueError(f'{genotype} {rec}: missing trials')
            arr = w.to_numpy(dtype=float).reshape(len(reps), len(rois), w.shape[1])
            out[(str(rec[0]), int(rec[1]), str(rec[2]))] = (arr, list(w.columns))
        _YANG_TRIALS[key] = out
    res = {}
    for rec, (arr, cols) in _YANG_TRIALS[key].items():
        missing = [o for o in odors if o not in cols]
        if missing:
            raise KeyError(f'{genotype} {rec}: no odors {missing}')
        res[rec] = arr[:, :, [cols.index(o) for o in odors]].copy()
    return res


def trial_values(trials: np.ndarray, values: str = 'graded', thresh: float = KC_RESPONSE_Z) -> np.ndarray:
    """Single trials read as `values`: 'graded' (unchanged), 'thresholded' (values at or below `thresh`
    set to 0, trial by trial, as a non-responding model KC contributes 0) or 'binary' (1 above `thresh`,
    else 0). Works on any array shape."""
    X = np.asarray(trials, dtype=float)
    if values == 'graded':
        return X
    if values == 'thresholded':
        return np.where(X > thresh, X, 0.0)
    if values == 'binary':
        return (X > thresh).astype(float)
    raise ValueError(f'values must be one of {TRIAL_VALUES}, not {values!r}')


def trial_cosines(trials: np.ndarray) -> np.ndarray:
    """Cosine similarities between every pair of single-trial response vectors of one recording.

    Args:
        trials: (n_repeats, units, odors).

    Returns (n_repeats, odors, n_repeats, odors): [i, a, j, b] is the cosine between the units' responses
    to odor a in repeat i and to odor b in repeat j; NaN where either vector is all zero.
    """
    X = np.asarray(trials, dtype=float)
    R, U, O = X.shape
    Y = X.transpose(0, 2, 1).reshape(R * O, U)
    G = Y @ Y.T
    n = np.sqrt(np.diag(G))
    with np.errstate(invalid='ignore', divide='ignore'):
        C = G / np.outer(n, n)
    C[(n == 0)[:, None] | (n == 0)[None, :]] = np.nan
    return C.reshape(R, O, R, O)


def _repeat_masks(n_rep: int) -> Dict[str, np.ndarray]:
    """{kind: (n_rep, n_rep) bool} of the repeat pairs (i, j) of each kind (ordered pairs)."""
    i, j = np.meshgrid(np.arange(n_rep), np.arange(n_rep), indexing='ij')
    return {'same_repeat': i == j, 'different_repeat': i != j, 'adjacent': np.abs(i - j) == 1,
        'first_third': np.abs(i - j) == 2}


def _kind_means(C: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(odors, odors) mean of C[i, a, j, b] over the repeat pairs in `mask` (NaN skipped; NaN if none)."""
    if not mask.any():
        return np.full(C.shape[1:4:2], np.nan)
    sel = C.transpose(0, 2, 1, 3)[mask]                      # (n_selected, odors, odors)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(sel, axis=0)


def distances_by_kind(trials: np.ndarray, odors: Sequence[str], *, values: str = 'graded',
    thresh: float = KC_RESPONSE_Z) -> pd.DataFrame:
    """One recording's single-trial cosine distances (1 - cosine), averaged by kind of trial pair.

    Args:
        trials: (n_repeats, units, odors), odors in the order of `odors`.
        values, thresh: how the single trials are read (`trial_values`).

    Returns a DataFrame indexed (odor_a, odor_b): the canonical two-odor pairs (in `odors` order), then
    one same-odor row per odor (odor_a == odor_b); columns `REPEAT_KINDS`: 'same_repeat' (the two odors
    in the same repeat; NaN for a same-odor row), 'different_repeat' (any two different repeats),
    'adjacent' (repeats one apart) and 'first_third' (repeats two apart; NaN with fewer than three
    repeats). Each entry is the mean over the trial pairs of that kind (NaN cosines skipped).
    """
    odors = list(odors)
    X = trial_values(trials, values, thresh)
    C = trial_cosines(X)
    masks = _repeat_masks(X.shape[0])
    D = {k: 1 - _kind_means(C, m) for k, m in masks.items()}
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    rows = [(a, b) for a, b in pl] + [(o, o) for o in odors]
    out = pd.DataFrame({k: [D[k][idx[a], idx[b]] for a, b in rows] for k in REPEAT_KINDS},
        index=pd.MultiIndex.from_tuples(rows, names=['odor_a', 'odor_b']))
    out.loc[[(o, o) for o in odors], 'same_repeat'] = np.nan
    return out


def same_repeat_gap(by_kind: pd.DataFrame, pairs: Optional[Sequence[Tuple[str, str]]] = None) -> float:
    """The same-repeat gap of one recording: the mean over two-odor pairs (default all of `by_kind`'s;
    `pairs` restricts them, e.g. to the six pairs with near-identical ORN input; either order) of the
    single-trial distance between different repeats minus the distance within the same repeat
    (`distances_by_kind`). Positive when two odors of one repeat are more alike than two odors of
    different repeats, i.e. when part of the trial-to-trial variability is shared by the odors of a
    repeat."""
    a = by_kind.index.get_level_values(0)
    b = by_kind.index.get_level_values(1)
    keep = np.asarray(a != b)
    if pairs is not None:
        want = {tuple(sorted(p)) for p in pairs}
        keep &= np.array([tuple(sorted(x)) in want for x in by_kind.index])
        if not keep.any():
            raise KeyError(f'none of {sorted(want)} in by_kind')
    g = by_kind.loc[keep, 'different_repeat'] - by_kind.loc[keep, 'same_repeat']
    return float(g.mean())


def _noise_free_matrix(C: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(odors, odors) noise-free distances 1 - C_ab / sqrt(C_aa C_bb) from single-trial cosines `C`
    (`trial_cosines`) over the repeat pairs in `mask` (which must exclude i == j)."""
    M = _kind_means(C, mask)
    d = np.diag(M).copy()
    d[~(d > 0)] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1 - M / np.sqrt(np.outer(d, d))


def noise_free_pair_distances(trials: np.ndarray, odors: Sequence[str], *, repeats: str = 'different',
    values: str = 'graded', thresh: float = KC_RESPONSE_Z) -> pd.Series:
    """Noise-free cosine distances of one recording from products of different repeats.

    For each pair (a, b): 1 - C_ab / sqrt(C_aa C_bb), C_ab the mean single-trial cosine between odor a in
    repeat i and odor b in repeat j over the repeat pairs i != j selected by `repeats`, C_aa the same for
    odor a with itself. Noise independent between repeats adds nothing, in expectation, to a product of
    two different repeats, whether or not it is shared by the odors of a repeat; a drift common to an
    odor's repeats acts on C_ab and C_aa alike and cancels in the ratio. Not clipped: a near-identical pair
    can come out slightly below 0. (Cosines rather than dot products, so a gain of a whole presentation,
    which a cosine does not see, does not weigh a trial in.)

    Args:
        trials: (n_repeats, units, odors).
        repeats: 'different' (every i != j), 'adjacent' (|i - j| == 1) or 'first_third' (|i - j| == 2).
        values, thresh: how the single trials are read (`trial_values`).

    Returns a canonical pair Series (index (odor_a, odor_b)).
    """
    kind = {'different': 'different_repeat', 'adjacent': 'adjacent', 'first_third': 'first_third'}.get(repeats)
    if kind is None:
        raise ValueError(f"repeats must be 'different', 'adjacent' or 'first_third', not {repeats!r}")
    odors = list(odors)
    X = trial_values(trials, values, thresh)
    D = _noise_free_matrix(trial_cosines(X), _repeat_masks(X.shape[0])[kind])
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    return pd.Series([D[idx[a], idx[b]] for a, b in pl],
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))


def repeat_pair_table(trials: np.ndarray, odors: Sequence[str], *, values: str = 'graded',
    thresh: float = KC_RESPONSE_Z) -> pd.DataFrame:
    """Every pair distance of one recording that E81 compares, one row per canonical two-odor pair.

    Columns: 'trial_mean' (1 - cosine of the two odors' trial means, the main target's distance; the
    trial mean of the trials read as `values`), 'corrected' (the noise correction in use,
    `noise_corrected_pair_distances`: noise assumed independent between odors), 'noise_free',
    'noise_free_adjacent', 'noise_free_first_third' (`noise_free_pair_distances` with each `repeats`),
    and the single-trial distances by kind (`REPEAT_KINDS`, from `distances_by_kind`).
    """
    odors = list(odors)
    X = trial_values(trials, values, thresh)
    C = trial_cosines(X)
    masks = _repeat_masks(X.shape[0])
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    ia = [idx[a] for a, _ in pl]
    ib = [idx[b] for _, b in pl]
    xbar = X.mean(axis=0)
    nrm = np.linalg.norm(xbar, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (xbar.T @ xbar) / np.outer(nrm, nrm)
    cols = {'trial_mean': 1 - S[ia, ib],
        'corrected': noise_corrected_pair_distances(X, odors).to_numpy() if X.shape[0] > 1 else np.full(len(pl), np.nan)}
    for name, kind in (('noise_free', 'different_repeat'), ('noise_free_adjacent', 'adjacent'),
            ('noise_free_first_third', 'first_third')):
        cols[name] = _noise_free_matrix(C, masks[kind])[ia, ib]
    for kind in REPEAT_KINDS:
        cols[kind] = (1 - _kind_means(C, masks[kind]))[ia, ib]
    return pd.DataFrame(cols, index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))


def presentation_pairs(trials: np.ndarray, odors: Sequence[str], order: pd.DataFrame, *, values: str = 'graded',
    thresh: float = KC_RESPONSE_Z, volume_rate: Optional[float] = None) -> pd.DataFrame:
    """One recording's single-trial distance between every two presentations of two different odors, with the
    time between them (E81 part A6, the stimulus order).

    Args:
        trials: (n_repeats, units, odors), odors in the order of `odors` (`yang_recording_trials`).
        odors: the odors of the last axis. Presentations of other stimuli (the solvent) form no pairs but keep
            their places in the order.
        order: the recording's presentations, one row each: 'presentation' (index in time order), 'block',
            'odor' (abbreviation), 'repeat' (the index of the presentation's trial on the first axis of
            `trials`) and, with `volume_rate`, 'onset_volume'.
        values, thresh: how the single trials are read (`trial_values`).
        volume_rate: volumes per second, for 'separation_s'.

    Returns a DataFrame, one row per pair of presentations of two different odors of `odors`: 'odor_a',
    'odor_b' (the canonical pair, as `pair_list` orders it), 'presentation_a', 'presentation_b', 'block_a',
    'block_b' (of odor_a's and odor_b's presentation), 'position_a', 'position_b' (in the block, when `order` has
    'position'), 'same_block', 'separation' (presentations apart, solvent presentations counted), 'separation_s'
    (seconds between the onsets; NaN without `volume_rate`) and 'distance' (1 - cosine of the two single trials;
    NaN where either is all zero).
    """
    odors = list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    o = order[order['odor'].isin(odors)].sort_values('presentation')
    if o.duplicated(['odor', 'repeat']).any():
        raise ValueError('order: an (odor, repeat) given to two presentations')
    if (o['repeat'] < 0).any() or (o['repeat'] >= trials.shape[0]).any():
        raise ValueError('order: a repeat outside the trials')
    X = trial_values(trials, values, thresh)
    C = trial_cosines(X)
    rank = {p: i for i, p in enumerate(odor_order(odors))}
    pres = o['presentation'].to_numpy()
    blk = o['block'].to_numpy()
    rep = o['repeat'].to_numpy()
    oi = np.array([idx[x] for x in o['odor']])
    onset = o['onset_volume'].to_numpy(dtype=float) if volume_rate is not None else np.full(len(o), np.nan)
    i, j = np.triu_indices(len(o), k=1)
    keep = oi[i] != oi[j]
    i, j = i[keep], j[keep]
    swap = np.array([rank[odors[oi[a]]] > rank[odors[oi[b]]] for a, b in zip(i, j)], dtype=bool)
    ia, ib = np.where(swap, j, i), np.where(swap, i, j)
    with np.errstate(invalid='ignore'):
        dist = 1 - C[rep[ia], oi[ia], rep[ib], oi[ib]]
    out = {'odor_a': [odors[k] for k in oi[ia]], 'odor_b': [odors[k] for k in oi[ib]],
        'presentation_a': pres[ia], 'presentation_b': pres[ib], 'block_a': blk[ia], 'block_b': blk[ib]}
    if 'position' in o.columns:
        pos = o['position'].to_numpy()
        out.update(position_a=pos[ia], position_b=pos[ib])
    out.update(same_block=blk[ia] == blk[ib], separation=np.abs(pres[ib] - pres[ia]),
        separation_s=np.abs(onset[ib] - onset[ia]) / (volume_rate if volume_rate else np.nan), distance=dist)
    return pd.DataFrame(out)


def repeat_response_stats(trials: np.ndarray, odors: Sequence[str], *, thresh: float = KC_RESPONSE_Z
    ) -> pd.DataFrame:
    """Per repeat and odor of one recording: 'rate' (fraction of units whose single-trial value is above
    `thresh`), 'amplitude' (the mean single-trial value of those units; NaN if none), 'n_resp', 'n_units'.
    Indexed (repeat (0-based), odor)."""
    X = np.asarray(trials, dtype=float)
    R, U, O = X.shape
    resp = X > thresh
    n_resp = resp.sum(axis=1)                                   # (R, O)
    with np.errstate(invalid='ignore', divide='ignore'):
        amp = np.where(resp, X, 0.0).sum(axis=1) / n_resp
    odors = list(odors)
    idx = pd.MultiIndex.from_product([range(R), odors], names=['repeat', 'odor'])
    return pd.DataFrame({'rate': (n_resp / U).ravel(), 'amplitude': amp.ravel(), 'n_resp': n_resp.ravel(),
        'n_units': np.full(R * O, U)}, index=idx)


def _pair_cov_sums(h: np.ndarray, A: np.ndarray, B: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Sum over units, repeats and ordered odor pairs o != p with (unit, o) in A and (unit, p) in B
    (B = A if None) of h[r, k, o] h[r, k, p], and the number of such (unit, o, p) triples.
    A, B: (units, odors) bool; when B is given, A must be a subset of B."""
    hA = np.where(A[None], h, 0.0)
    if B is None:
        s = hA.sum(axis=2)
        q = (hA ** 2).sum(axis=2)
        nA = A.sum(axis=1)
        return float((s ** 2 - q).sum()), float((nA * (nA - 1)).sum())
    hB = np.where(B[None], h, 0.0)
    total = float((hA.sum(axis=2) * hB.sum(axis=2) - (hA ** 2).sum(axis=2)).sum())
    return total, float((A.sum(axis=1) * (B.sum(axis=1) - 1)).sum())


def _repeat_gains(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The gain of each presentation and the deviations of one recording (`repeat_variance_parts`).

    The dot products over units between two different repeats of one odor, P_rs = x_r . x_s (r != s), are
    g_r g_s |m|^2 in expectation whatever noise is independent between repeats, so each repeat's gain follows
    from them: a_r = sqrt(g_r^2 |m|^2) from log P_rs = log a_r + log a_s by least squares (with three repeats
    exactly a_r^2 = P_rs P_rt / P_st), and the relative gain is a_r over the mean of the a's.

    Returns (d, gdev, m, Q): d (R, U, O) the deviations from each entry's mean over repeats; gdev (R, O) each
    presentation's relative gain minus 1 (0 for every presentation of an odor with a product not positive, and
    with fewer than three repeats, where gain and noise cannot be told apart); m (U, O) the trial means; Q
    (U, O) the mean over i != j of x_i x_j per entry, an estimate of the entry's squared noise-free mean.
    """
    R, U, O = X.shape
    m = X.mean(axis=0)
    d = X - m[None]
    gdev = np.zeros((R, O))
    if R >= 3:
        P = np.einsum('iko,jko->ijo', X, X)                      # (R, R, O)
        iu, ju = np.triu_indices(R, 1)
        A = np.zeros((len(iu), R))
        A[np.arange(len(iu)), iu] = 1.0
        A[np.arange(len(iu)), ju] = 1.0
        Pu = P[iu, ju]                                           # (n_pairs, O)
        ok = (Pu > 0).all(axis=0)
        if ok.any():
            la = np.linalg.lstsq(A, np.log(Pu[:, ok]), rcond=None)[0]     # (R, n_ok)
            a = np.exp(la)
            gdev[:, ok] = a / a.mean(axis=0) - 1.0
    S = X.sum(axis=0)
    Q = (S ** 2 - (X ** 2).sum(axis=0)) / (R * (R - 1))
    return d, gdev, m, Q


def repeat_variance_parts(trials: np.ndarray, odors: Sequence[str], *, thresh: float = KC_RESPONSE_Z,
    n_amp_bins: int = 4) -> pd.DataFrame:
    """The parts of one recording's variability across repeats.

    Each entry's deviation from its own mean over repeats, d[r, k, o] = x[r, k, o] - mean_r x[., k, o],
    is split into three parts, with x the units' (KCs') single-trial values:
    - 'gain': common to all units of a presentation (repeat r of odor o), proportional to the units'
      noise-free response to that odor (a cosine does not see it; it also takes up a run-down common to the
      whole repeat). The presentation's relative gain comes from products of different repeats
      (`_repeat_gains`), so the noise in the trial mean does not attenuate it as a regression on the trial
      mean would; the part's variance in an entry is the variance over repeats of its odor's relative gains
      times the entry's squared noise-free mean (estimated from products of different repeats);
    - 'shared': common to one unit's odors within a repeat: the covariance between two different odors'
      residual deviations (the gain part, relative gain deviation times trial mean, removed) of the same unit
      in the same repeat;
    - 'rest': the remainder, var_total - var_gain - var_shared (independent between the odors of a repeat).
    Variances are per trial (sums over repeats of squared deviations, or of products, divided by
    n_repeats - 1, which is unbiased for a part independent between repeats), averaged over the entries
    (or, for the shared part, the unit x odor-pair products) of a class.

    Args:
        trials: (n_repeats, units, odors).
        thresh: an entry (unit, odor) is a responder when its trial mean is above `thresh`, silent
            otherwise; the shared part of 'all', 'responders' and 'silent' is taken over pairs of odors
            whose two entries are both in the class.
        n_amp_bins: the responders' entries are also split into this many quantile bins of their trial
            mean ('responders_amp1' the lowest); a bin's shared part is the covariance of its entries
            with the same unit's other responder entries, to show whether each part scales with the
            amplitude.

    Returns a DataFrame indexed by class ('all', 'responders', 'silent', 'responders_amp1', ...): n_entries,
    n_pairs (unit x ordered odor pairs behind the shared part), mean_amplitude (mean trial mean),
    var_total, var_gain, var_shared, var_rest, share_gain, share_shared, share_rest (of var_total),
    sd_trial (sqrt(var_total)) and corr_shared (var_shared over var_total - var_gain: the correlation of
    two odors' deviations within a repeat once the gain part is removed).
    """
    X = np.asarray(trials, dtype=float)
    R = X.shape[0]
    if R < 2:
        raise ValueError('need at least two repeats')
    d, gdev, m, Q = _repeat_gains(X)
    h = d - gdev[:, None, :] * m[None]
    v_tot = (d ** 2).sum(axis=0) / (R - 1)                      # per entry
    v_gain = ((gdev ** 2).sum(axis=0) / (R - 1))[None, :] * Q
    resp = m > thresh
    classes = {'all': (np.ones_like(resp), None), 'responders': (resp, None), 'silent': (~resp, None)}
    if n_amp_bins and resp.sum() >= n_amp_bins:
        edges = np.quantile(m[resp], np.linspace(0, 1, n_amp_bins + 1))
        b = np.clip(np.searchsorted(edges, m, side='right') - 1, 0, n_amp_bins - 1)
        for i in range(n_amp_bins):
            classes[f'responders_amp{i + 1}'] = (resp & (b == i), resp)
    rows = {}
    for name, (A, B) in classes.items():
        n = int(A.sum())
        if n == 0:
            rows[name] = dict(n_entries=0, n_pairs=0)
            continue
        s, npairs = _pair_cov_sums(h, A, B)
        vt, vg = float(v_tot[A].mean()), float(v_gain[A].mean())
        vs = s / (R - 1) / npairs if npairs > 0 else np.nan
        rows[name] = dict(n_entries=n, n_pairs=int(npairs), mean_amplitude=float(m[A].mean()), var_total=vt,
            var_gain=vg, var_shared=vs, var_rest=vt - vg - vs, share_gain=vg / vt, share_shared=vs / vt,
            share_rest=(vt - vg - vs) / vt, sd_trial=float(np.sqrt(vt)), corr_shared=vs / (vt - vg))
    return pd.DataFrame(rows).T.rename_axis('class')


def coresponder_deviation_corr(trials: np.ndarray, odors: Sequence[str], *, thresh: float = KC_RESPONSE_Z,
    min_units: int = 5, remove_gain: bool = False) -> pd.Series:
    """Per two-odor pair of one recording: the correlation, over the units responding to both odors (trial
    mean above `thresh`) and the repeats, between the two odors' deviations from each unit's mean over
    repeats (the shared state among co-responders; with `remove_gain`, after the gain part of
    `repeat_variance_parts`). NaN for pairs with fewer than `min_units` such units.
    Returns a canonical pair Series."""
    X = np.asarray(trials, dtype=float)
    d, gdev, m, _ = _repeat_gains(X)
    if remove_gain:
        d = d - gdev[:, None, :] * m[None]
    resp = m > thresh
    odors = list(odors)
    pl = pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    vals = []
    for a, b in pl:
        ia, ib = idx[a], idx[b]
        k = resp[:, ia] & resp[:, ib]
        if k.sum() < min_units:
            vals.append(np.nan)
            continue
        da, db = d[:, k, ia].ravel(), d[:, k, ib].ravel()
        den = np.sqrt((da ** 2).sum() * (db ** 2).sum())
        vals.append(float((da * db).sum() / den) if den > 0 else np.nan)
    return pd.Series(vals, index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))


# ---------------------------------------------------------------------------
# plots (matplotlib; sequential single-hue maps for distances, diverging for residuals)
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_square(ax, pairs: pd.Series, title: str = '', vmin: float = 0.0,
    vmax: float = 1.0, cmap: str = 'Blues', cbar_label: str = '1 - similarity') -> None:
    """Heatmap of a pair Series as an odor x odor matrix."""
    sq = square_from_pairs(pairs)
    im = ax.imshow(sq.to_numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(sq)))
    ax.set_xticklabels(sq.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sq)))
    ax.set_yticklabels(sq.index, fontsize=7)
    ax.set_title(title, fontsize=9)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)


def plot_on_vs_off(ax, on: pd.Series, off: pd.Series, *, label: str = 'model',
    color: str = CONDITION_COLORS['model'], data_on: Optional[pd.Series] = None,
    data_off: Optional[pd.Series] = None, highlight_block: str = FRUITY_SMALL_BLOCK
    ) -> None:
    """Scatter of APL-on vs APL-off pair distances with the isotonic fit; block pairs marked.

    Optionally overlays the data (control vs TNT) in a second color.
    """
    on, off = align_pairs(on, off)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(off, on)
    xs = np.linspace(off.min(), off.max(), 100)
    blk = np.array([block_label(a, b) == highlight_block for a, b in on.index])
    ax.scatter(off[~blk], on[~blk], s=12, color=color, alpha=0.7, label=label,
        edgecolors='none'
    )
    ax.scatter(off[blk], on[blk], s=26, facecolors='none', edgecolors=color, linewidths=1.2,
        label=f'{label}: {highlight_block} pairs'
    )
    ax.plot(xs, iso.predict(xs), color=color, lw=1.5, alpha=0.9)
    if data_on is not None and data_off is not None:
        d_on, d_off = align_pairs(data_on, data_off)
        dc = CONDITION_COLORS['data']
        ax.scatter(d_off, d_on, s=12, color=dc, alpha=0.7, label='data (TNT3c vs TNT)',
            edgecolors='none'
        )
        iso_d = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(d_off, d_on)
        xd = np.linspace(d_off.min(), d_off.max(), 100)
        ax.plot(xd, iso_d.predict(xd), color=dc, lw=1.5, alpha=0.9)
    lim = [0, 1.05]
    ax.plot(lim, lim, color='#bbbbbb', lw=0.8, ls='--')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('1 - similarity, APL off (TNT)')
    ax.set_ylabel('1 - similarity, APL on (control)')
    ax.legend(fontsize=7, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot_per_odor_vs_drive(ax, per_odor: pd.DataFrame, *, col: str = 'iso_resid',
    color: str = CONDITION_COLORS['model'], label: str = 'model') -> None:
    """Per-odor APL effect (mean isotonic residual of its pairs) vs ORN drive, labeled."""
    d = per_odor.dropna(subset=['drive', col])
    ax.scatter(d['drive'], d[col], s=18, color=color, label=label, edgecolors='none')
    for o, row in d.iterrows():
        ax.annotate(o, (row['drive'], row[col]), fontsize=6, xytext=(2, 2),
            textcoords='offset points', color='#444444'
        )
    ax.axhline(0, color='#bbbbbb', lw=0.8, ls='--')
    ax.set_xlabel('ORN drive (mean over glomeruli)')
    ax.set_ylabel(f'per-odor APL effect ({col})')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def plot_model_vs_data(ax, model: pd.Series, data: pd.Series, *, label: str = '',
    color: str = CONDITION_COLORS['model']) -> None:
    """Scatter of model vs data pair distances, with unity line."""
    m, d = align_pairs(model, data)
    ax.scatter(d, m, s=12, color=color, alpha=0.75, edgecolors='none', label=label)
    lim = [0, 1.05]
    ax.plot(lim, lim, color='#bbbbbb', lw=0.8, ls='--')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('data: 1 - similarity')
    ax.set_ylabel('model: 1 - similarity')
    if label:
        ax.legend(fontsize=7, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
def selftest(n_boot: int = 500) -> None:
    """Reproduces the report's data-vs-data numbers and prints recording-bootstrap CIs."""
    pd.set_option('display.width', 160)
    rep = report_matrices()
    recs = {g: yang_recording_rdms(g, 'cosine') for g in ('TNT3c', 'TNT')}
    for g in recs:
        diff = (recs[g].mean() - rep[g]).abs().max()
        print(f'{g}: {len(recs[g])} recordings; recording-mean cosine vs report matrix: '
            f'max |diff| = {diff:.2e}'
        )
    orn = orn16_consensus_pairs()
    drive = consensus_orn_drive()
    st, tabs = compare_on_off(rep['TNT3c'], rep['TNT'], orn=orn, drive=drive)
    print('\ncontrol (TNT3c) vs TNT, 120 pairs (report: spearman 0.869, lin R2 0.848, '
        'iso R2 0.94, block +0.055, spearman ORN 0.655 -> 0.748, per-odor vs drive 0.50):'
    )
    for k, v in st.items():
        print(f'  {k:>34s}: {v:.4f}' if isinstance(v, float) else f'  {k:>34s}: {v}')
    print('\nblocks (mean isotonic residual):')
    print(tabs['blocks'].sort_values('iso_resid', ascending=False).round(3).to_string())
    print('\nreference comparisons (66 megamat pairs):')
    ref = reference_pairs()
    rows = {g: compare_to_references(rep[g], ref) for g in ('TNT3c', 'TNT')}
    rows['WT (Remy)'] = compare_to_references(ref['kc_soma'], ref)
    print(pd.DataFrame(rows).round(3).to_string())

    def fn(means: Dict[str, pd.Series]) -> Dict[str, float]:
        s, _ = compare_on_off(means['TNT3c'], means['TNT'], orn=orn, drive=drive)
        keys = ['mean_on', 'mean_off', 'spearman_on_off', 'lin_r2', 'iso_r2',
            'block_fruity_small_iso', 'block_other_between_iso', 'spearman_orn_on',
            'spearman_orn_off', 'gap_closed', 'apl_share_resid_var',
            'spearman_per_odor_iso_vs_drive',
        ]
        out = {k: s[k] for k in keys}
        for g in ('TNT3c', 'TNT'):
            r = compare_to_references(means[g], ref)
            out[f'{g}_spearman_vcf_hyp'] = r['spearman_vcf_hyp']
            out[f'{g}_spearman_resid_vs_vcf_hyp'] = r['spearman_resid_vs_vcf_hyp']
            out[f'{g}_spearman_resid_vs_vcf_raw'] = r['spearman_resid_vs_vcf_raw']
            out[f'{g}_spearman_kc_claw'] = r['spearman_kc_claw']
        return out

    print(f'\nbootstrap over recordings ({n_boot} draws; 90% CI):')
    print(bootstrap_recordings(fn, recs, n_boot=n_boot).round(3).to_string())
    print('\nsplit-half reliability of the mean matrices (Spearman; ceiling for a model):')
    for g in ('TNT3c', 'TNT'):
        for bz in (None, 1.0):
            r = split_half_reliability(yang_recording_rdms(g, 'cosine', binarize=bz))
            print(f"  {g} {'binarized' if bz else 'raw':9s}: split-half {r['split_half']:.3f} "
                f"[{r['split_half_lo']:.3f}, {r['split_half_hi']:.3f}], full-sample "
                f"{r['spearman_brown_full']:.3f}, model ceiling {r['ceiling']:.3f}")
    print('\nbinarized data (trial-mean > 1), recording-mean 1 - cosine: '
        + ', '.join(f'{g} {yang_mean_rdm(g, binarize=1.0).mean():.3f}' for g in ('TNT3c', 'TNT'))
        + ' (raw: ' + ', '.join(f'{g} {rep[g].mean():.3f}' for g in ('TNT3c', 'TNT')) + ')'
    )
    print('\n' + APPROX_CONC_NOTE)


if __name__ == '__main__':
    selftest()
