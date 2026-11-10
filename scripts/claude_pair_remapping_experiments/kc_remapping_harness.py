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


YANG_EFFECT_VALUES = ('graded', 'thresholded', 'noise_corrected')


def yang_recording_rdms_values(genotype: str, values: str = 'graded') -> pd.DataFrame:
    """Per-recording cosine pair distances (rows recordings, columns pairs) of the KC data
    transformed as `values` says: 'graded' (as `yang_recording_rdms`), 'thresholded' (trial
    means <= 1 set to 0 before the distance, as a model KC that does not respond contributes 0)
    or 'noise_corrected' (each recording's cosine corrected for trial-to-trial noise,
    `noise_floor_trial_distances.analyse`: the distance a noise-free recording would show)."""
    if values == 'graded':
        return yang_recording_rdms(genotype, 'cosine')
    if values == 'thresholded':
        return recording_rdms_from_matrices({k: transform_matrix(m, values='thresholded')
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
    (`readout(..., 'binary')`). `values` 'thresholded' or 'noise_corrected' (cosine only, without
    `binarize`) computes it from the data transformed as `yang_recording_rdms_values` says.
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


def readout(counts: pd.DataFrame, kind: str = 'raw', param=None) -> pd.DataFrame:
    """A readout of a model's spike counts (rows KCs, columns odors), elementwise.

    Args:
        counts: spike counts, e.g. the archived `<tag>_<condition>_spike_counts.parquet`.
        kind: 'raw' (the counts; the target as defined); 'sqrt' (n ** 0.5); 'power'
            (n ** param, 0 < param <= 1); 'saturating' (n / (n + param), half-saturation at
            `param` spikes); 'log1p' (log(1 + n / param)); 'binary' (1 where n >= param,
            default 2, i.e. the model's `responses`); 'rank_match' (each count mapped through
            the lookup `param` from `rank_match_map`, a monotone count -> amplitude map).
        param: the readout's parameter (see `kind`); None for 'raw' and 'sqrt'.

    Every readout keeps 0 at 0 and is monotone in the count, so which KCs respond, and the
    response rates, are the same under every readout.
    """
    x = counts.astype(float)
    if kind == 'raw':
        return x
    if kind == 'sqrt':
        return x ** 0.5
    if kind == 'power':
        return x ** float(param)
    if kind == 'saturating':
        return x / (x + float(param))
    if kind == 'log1p':
        return np.log1p(x / float(param))
    if kind == 'binary':
        return (x >= (2.0 if param is None else float(param))).astype(float)
    if kind == 'rank_match':
        lut = param
        vals = np.interp(x.to_numpy(), lut.index.to_numpy(dtype=float), lut.to_numpy(dtype=float))
        return pd.DataFrame(np.where(x.to_numpy() > 0, vals, 0.0), index=x.index, columns=x.columns)
    raise ValueError(f'unknown readout {kind!r} (one of {READOUT_KINDS})')


def readout_pair_distances(counts: pd.DataFrame, kind: str = 'raw', param=None) -> pd.Series:
    """Cosine odor-odor distances across KCs after `readout`: the model's side of the per-pair
    APL effect under that readout ('raw' gives the `spike_counts_cosine` distances)."""
    return pair_distances(readout(counts, kind, param), measure='cosine')


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
    responders_min: Optional[float] = None) -> float:
    """A concentration statistic of the readout of `on_counts`: an `amplitude_stats` column
    (mean over odors) or 'responder_cv' (`responder_cv` of the readout among responders).
    `responders_min` (e.g. 2, the model's response threshold) zeroes the readout of the
    non-responding entries first, like-for-like with the data's statistics on responders only
    (trial means <= 1 set to 0); 'responder_cv' always uses responders (default 2 spikes)."""
    r = readout(on_counts, kind, param)
    if stat == 'responder_cv':
        thr = 2.0 if responders_min is None else responders_min
        return responder_cv(r.to_numpy()[on_counts.to_numpy() >= thr])
    if responders_min is not None:
        r = r.where(on_counts >= responders_min, 0.0)
    return float(amplitude_stats(r)[stat].mean())


def calibrate_readout(on_counts: pd.DataFrame, kind: str, target: float, stat: str = 'participation_ratio',
    bounds: Tuple[float, float] = (1e-3, 1e3), iters: int = 60, responders_min: Optional[float] = None) -> float:
    """The parameter of a one-parameter readout ('power', 'saturating' or 'log1p') at which the
    APL-on `readout_stat` (`stat`, `responders_min`) equals `target`, e.g. the control flies'
    participation ratio: bisection on a log scale. The parameter is fixed on the APL-on counts
    alone and then used unchanged for APL off, so the APL effect is not fitted. Returns the bound
    nearest to the target when it is not bracketed (for 'power' the upper bound is 1, raw counts)."""
    if kind == 'power':
        bounds = (max(bounds[0], 1e-3), 1.0)
    f = lambda p: readout_stat(on_counts, kind, p, stat, responders_min)
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
