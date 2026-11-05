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
    """The report's 16-odor ORN matrix (1 - r of consensus glomerular mean dF/F)."""
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
