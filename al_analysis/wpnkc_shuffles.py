"""Shuffled (null model) versions of PN->KC connectivity matrices (`wPNKC`).

For testing how much of the model's odor-odor structure comes from the specific
connectome wiring, rather than from claw count and glomerulus frequency statistics alone.
Outputs of `shuffle_wPNKC` can be passed to `fit_mb_model` via its `_wPNKC` keyword
(along with the same `one_row_per_claw` / `prat_claws` / etc flags that would produce
the unshuffled matrix).
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from al_analysis.mb_model import (KC_ID, KC_TYPE, CLAW_ID, connectome_wPNKC,
    get_connectome_wPNKC_params
)


SHUFFLE_METHODS: Tuple[str, ...] = ('claw_permutation', 'curveball')

# name -> (`shuffle_wPNKC` method, within_kc_type), for `shuffled_wPNKC`
NAMED_SHUFFLES: Dict[str, Tuple[str, bool]] = {
    'claw_permutation': ('claw_permutation', False),
    'claw_permutation_within_type': ('claw_permutation', True),
    'curveball': ('curveball', False),
    'curveball_within_type': ('curveball', True),
}

# default number of Curveball trades, per KC
CURVEBALL_TRADES_PER_KC: int = 5


def curveball(binary: np.ndarray, rng: np.random.Generator, n_iters: int
    ) -> np.ndarray:
    """Returns `binary` randomized by Curveball trades, with row and column sums kept.

    Curveball algorithm from Strona et al. 2014 (Nat Commun 5:4114, "A fast and unbiased
    procedure to randomize ecological binary matrices with fixed row and column
    totals"). Each trade picks two distinct rows at random, pools the columns that are 1
    in exactly one of the two rows, and randomly re-distributes the pool between them,
    with each row getting back as many as it gave. Columns that are 1 in both rows stay
    1 in both.

    Args:
        binary: 2D array of 0/1 (or boolean) values. Not modified.
        rng: e.g. from `np.random.default_rng(seed)`
        n_iters: number of trades (pairs of rows drawn, including pairs with nothing to
            trade). `shuffle_wPNKC` uses 5x the number of rows by default.

    Returns array of same shape and dtype as `binary`, with same row and column sums.
    """
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(f'binary must be 2D (got shape {binary.shape})')

    if not np.isin(binary, (0, 1)).all():
        raise ValueError('binary must only contain 0/1 (or boolean) values')

    m = binary.astype(bool)
    n_rows = m.shape[0]
    if n_rows < 2 or n_iters <= 0:
        return binary.copy()

    rows_a = rng.integers(n_rows, size=n_iters)
    # uniform over rows other than rows_a
    rows_b = rng.integers(n_rows - 1, size=n_iters)
    rows_b[rows_b >= rows_a] += 1

    for a, b in zip(rows_a, rows_b):
        row_a = m[a]
        row_b = m[b]
        a_only = row_a & ~row_b
        b_only = row_b & ~row_a
        n_a_only = np.count_nonzero(a_only)
        if n_a_only == 0 or not b_only.any():
            continue

        pool = rng.permutation(np.flatnonzero(a_only | b_only))
        to_a = pool[:n_a_only]
        to_b = pool[n_a_only:]
        m[a, to_b] = False
        m[a, to_a] = True
        m[b, to_a] = False
        m[b, to_b] = True

    return m.astype(binary.dtype)


def _wPNKC_values(wPNKC: pd.DataFrame, one_row_per_claw: bool) -> np.ndarray:
    """Returns `wPNKC` values, after checking they are valid for its format.
    """
    if wPNKC.index.has_duplicates or wPNKC.columns.has_duplicates:
        raise ValueError('wPNKC index and columns must not have duplicates')

    if KC_ID not in wPNKC.index.names:
        raise ValueError(f'wPNKC index must have a {KC_ID!r} level')

    values = wPNKC.to_numpy()
    if values.dtype == bool:
        if not one_row_per_claw:
            raise ValueError('boolean wPNKC only supported for one-row-per-claw input '
                '(pass integer claw counts for one-row-per-KC input)'
            )
    elif not (np.issubdtype(values.dtype, np.number) and (values >= 0).all() and
            (values == np.round(values)).all()):
        raise ValueError('wPNKC values must be non-negative integer claw counts')

    if one_row_per_claw and not (
            np.isin(values, (0, 1)).all() and (values.sum(axis=1) <= 1).all()):
        raise ValueError('one-row-per-claw wPNKC values must be 0/1, with at most one 1 '
            'per row'
        )

    return values


def _kc_codes(wPNKC: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """Returns KC code (0 to n_kcs - 1, in order of first appearance) per row, & n_kcs.
    """
    row_kc, kc_ids = pd.factorize(wPNKC.index.get_level_values(KC_ID))
    if (row_kc < 0).any():
        raise ValueError(f'wPNKC {KC_ID!r} index level must not have NaN')

    return row_kc, len(kc_ids)


def _kc_groups(wPNKC: pd.DataFrame, row_kc: np.ndarray, n_kcs: int) -> np.ndarray:
    """Returns integer code of each KC's `KC_TYPE` (length n_kcs).
    """
    if KC_TYPE not in wPNKC.index.names:
        raise ValueError(f'within_kc_type=True requires {KC_TYPE!r} wPNKC index level')

    # NaN types -> code -1 (grouped together)
    row_type, _ = pd.factorize(wPNKC.index.get_level_values(KC_TYPE))

    _, first_row = np.unique(row_kc, return_index=True)
    assert len(first_row) == n_kcs
    kc_group = row_type[first_row]
    if not np.array_equal(kc_group[row_kc], row_type):
        raise ValueError(f'{KC_TYPE!r} must be the same for all rows of each KC')

    return kc_group


def _kc_counts(wPNKC: pd.DataFrame) -> np.ndarray:
    """Returns array (# KCs, # columns) of # claws per (KC, column), for either format.

    KCs are in order of first appearance in `wPNKC` (one per row, for one-row-per-KC
    input).
    """
    one_row_per_claw = CLAW_ID in wPNKC.index.names
    values = _wPNKC_values(wPNKC, one_row_per_claw).astype(np.int64)
    if not one_row_per_claw:
        return values

    row_kc, n_kcs = _kc_codes(wPNKC)
    counts = np.zeros((n_kcs, values.shape[1]), dtype=np.int64)
    np.add.at(counts, row_kc, values)
    return counts


def shuffle_wPNKC(wPNKC: pd.DataFrame, method: str = 'claw_permutation', *,
    within_kc_type: bool = False, seed: int = 0, n_iters: Optional[int] = None
    ) -> pd.DataFrame:
    """Returns shuffled copy of `wPNKC`, with same index, columns and dtypes.

    Args:
        wPNKC: from `connectome_wPNKC`, in either of two formats:
            - one row per KC (index without `CLAW_ID` level), with # of claws from each
              glomerulus as values (e.g. `weight_divisor=20`)
            - one row per claw (index with `CLAW_ID` level, e.g. `prat_claws=True`),
              with 0/1 values and at most one 1 per row. Index entries (e.g. claw
              coordinates) stay with their rows, and only which column holds each row's
              1 changes.

            All-0 rows (KCs or claws without input) are left as is. Columns are
            typically glomeruli, but any column labels (e.g. boutons) are treated the
            same way.

        method: 'claw_permutation' randomly permutes glomerulus (column) labels over
            all claws. This exactly preserves the number of claws of each KC and the
            total number of claws from each glomerulus, and destroys any other
            structure (including the number of distinct glomeruli per KC, and the
            number of claws per KC-glomerulus pair).

            'curveball' randomizes the binary KC x glomerulus matrix (whether each KC
            has any claw from each glomerulus) with `curveball`. This exactly preserves
            the number of distinct glomeruli of each KC and the number of KCs each
            glomerulus contacts. Each KC keeps its claws and their multiplicity pattern
            (e.g. 2 claws from one glomerulus and 1 from each of two others), with its
            original glomeruli mapped to its new ones by a random one-to-one assignment.
            Because of that random assignment, the total number of claws from each
            glomerulus is only approximately preserved.

        within_kc_type: if True, shuffles separately within each `KC_TYPE` index level
            value ('unknown' is its own group), so per-type totals are also preserved
            (claws per glomerulus for 'claw_permutation', KCs per glomerulus for
            'curveball').

        seed: for `np.random.default_rng`. Output is reproducible for a given seed.

        n_iters: number of Curveball trades (only for `method='curveball'`). Default is
            5x the number of KCs. With `within_kc_type=True`, trades are split across
            types in proportion to their number of KCs.
    """
    if method not in SHUFFLE_METHODS:
        raise ValueError(f'{method=} not in {SHUFFLE_METHODS}')

    if n_iters is not None and method != 'curveball':
        raise ValueError("n_iters only used for method='curveball'")

    one_row_per_claw = CLAW_ID in wPNKC.index.names
    values = _wPNKC_values(wPNKC, one_row_per_claw)
    n_cols = values.shape[1]

    rng = np.random.default_rng(seed)

    if one_row_per_claw:
        row_kc, n_kcs = _kc_codes(wPNKC)
        claw_rows = np.flatnonzero(values.any(axis=1))
        claw_col = values[claw_rows].argmax(axis=1)
        claw_kc = row_kc[claw_rows]
    else:
        n_kcs = len(values)
        row_kc = np.arange(n_kcs)
        # one element per claw, with claws of each KC consecutive
        flat = np.repeat(np.arange(values.size), values.ravel().astype(np.int64))
        claw_kc, claw_col = np.divmod(flat, n_cols)

    if within_kc_type:
        kc_group = _kc_groups(wPNKC, row_kc, n_kcs)
    else:
        kc_group = np.zeros(n_kcs, dtype=np.int64)
    groups = np.unique(kc_group)

    if method == 'claw_permutation':
        new_col = claw_col.copy()
        claw_group = kc_group[claw_kc]
        for group in groups:
            mask = claw_group == group
            new_col[mask] = rng.permutation(claw_col[mask])
    else:
        counts = np.zeros((n_kcs, n_cols), dtype=np.int64)
        np.add.at(counts, (claw_kc, claw_col), 1)
        binary = counts > 0

        new_binary = binary.copy()
        for group in groups:
            rows = np.flatnonzero(kc_group == group)
            if n_iters is None:
                group_iters = CURVEBALL_TRADES_PER_KC * len(rows)
            else:
                group_iters = int(round(n_iters * len(rows) / n_kcs))

            new_binary[rows] = curveball(binary[rows], rng, group_iters)

        # (both in row-major order, and equal since row sums are preserved)
        old_kc, old_col = np.nonzero(binary)
        new_kc, new_kc_col = np.nonzero(new_binary)
        assert np.array_equal(old_kc, new_kc)

        # random one-to-one map from each KC's old glomeruli to its new glomeruli
        within_kc_order = np.lexsort((rng.random(len(new_kc)), new_kc))
        col_map = np.full((n_kcs, n_cols), -1, dtype=np.int64)
        col_map[old_kc, old_col] = new_kc_col[within_kc_order]
        new_col = col_map[claw_kc, claw_col]
        assert (new_col >= 0).all()

    if one_row_per_claw:
        new_values = np.zeros_like(values)
        new_values[claw_rows, new_col] = 1
    else:
        new_values = np.bincount(claw_kc * n_cols + new_col, minlength=values.size
            ).reshape(values.shape).astype(values.dtype)

    shuffled = pd.DataFrame(new_values, index=wPNKC.index.copy(),
        columns=wPNKC.columns.copy()
    )
    if wPNKC.dtypes.nunique() > 1:
        shuffled = shuffled.astype(wPNKC.dtypes.to_dict())

    return shuffled


def shuffled_wPNKC(name: str, *, seed: int = 0, **model_kws) -> pd.DataFrame:
    """Returns shuffled connectome wPNKC, to pass to `fit_mb_model` as `_wPNKC`.

    Output is meant to be used as `fit_mb_model(..., _wPNKC=<output>, **model_kws)`.

    Args:
        name: key of `NAMED_SHUFFLES` ('<method>' or '<method>_within_type'), giving
            `method` and `within_kc_type` for `shuffle_wPNKC`

        seed: passed to `shuffle_wPNKC`

        **model_kws: `fit_mb_model` kwargs. The subset that are `connectome_wPNKC`
            arguments (see `get_connectome_wPNKC_params`) select the matrix to shuffle
            (e.g. `prat_claws=True` for one row per claw, or `weight_divisor=20`).
            Others are ignored.
    """
    if name not in NAMED_SHUFFLES:
        raise ValueError(f'{name=} not in {list(NAMED_SHUFFLES)}')

    # fit_mb_model only accepts _wPNKC with this (default) value
    if model_kws.get('pn2kc_connections', 'hemibrain') != 'hemibrain':
        raise ValueError("pn2kc_connections must be 'hemibrain' (or unset)")

    method, within_kc_type = NAMED_SHUFFLES[name]
    wPNKC = connectome_wPNKC(**get_connectome_wPNKC_params(model_kws))
    return shuffle_wPNKC(wPNKC, method, within_kc_type=within_kc_type, seed=seed)


def wPNKC_shuffle_summary(original: pd.DataFrame, shuffled: pd.DataFrame
    ) -> pd.Series:
    """Returns Series of checks and statistics comparing shuffled `wPNKC` to original.

    Args:
        original: `wPNKC` input to `shuffle_wPNKC` (either format it supports)
        shuffled: output of `shuffle_wPNKC(original, ...)`, or any other matrix with the
            same index and columns

    Returns Series with:
        - n_kcs, n_claws: of `original`

        - frac_claws_changed: fraction of claws whose glomerulus changed. For
          one-row-per-KC input (where claws are not individually identified), the
          minimum such fraction, `1 - sum(min(original, shuffled)) / n_claws`.

        - kc_n_claws_preserved: whether each KC has the same number of claws

        - kc_n_glomeruli_preserved: whether each KC has the same number of distinct
          glomeruli

        - max_abs_glom_n_claws_change: max (over glomeruli) absolute change in total
          number of claws from the glomerulus

        - max_abs_glom_n_kcs_change: max (over glomeruli) absolute change in number of
          KCs with any claw from the glomerulus

        - coinput_corr: Pearson correlation, across glomerulus pairs, between original
          and shuffled number of KCs with claws from both glomeruli of the pair. 1 if
          co-input structure is unchanged. Shuffles should lower it, but not to 0, since
          glomerulus frequencies (which they preserve) also shape co-input counts.

        - coinput_resid_corr: as `coinput_corr`, but on residuals after subtracting the
          part of each matrix's co-input counts proportional to the product of the
          two glomeruli's # of KCs (least-squares fit across pairs). Roughly 0 for
          shuffles that leave only glomerulus frequencies.
    """
    if not (original.index.equals(shuffled.index) and
            original.columns.equals(shuffled.columns)):
        raise ValueError('original and shuffled must have same index and columns')

    orig_counts = _kc_counts(original)
    shuf_counts = _kc_counts(shuffled)
    n_claws = int(orig_counts.sum())

    if CLAW_ID in original.index.names:
        orig_values = original.to_numpy()
        claw_rows = orig_values.any(axis=1)
        frac_changed = (orig_values[claw_rows] != shuffled.to_numpy()[claw_rows]
            ).any(axis=1).mean()
    else:
        frac_changed = 1 - np.minimum(orig_counts, shuf_counts).sum() / n_claws

    orig_binary = (orig_counts > 0).astype(np.int64)
    shuf_binary = (shuf_counts > 0).astype(np.int64)

    triu = np.triu_indices(orig_counts.shape[1], k=1)

    def coinput_and_resid(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coinput = (binary.T @ binary)[triu].astype(float)
        n_kcs_per_glom = binary.sum(axis=0).astype(float)
        freq_product = np.outer(n_kcs_per_glom, n_kcs_per_glom)[triu]
        scale = (coinput * freq_product).sum() / (freq_product ** 2).sum()
        return coinput, coinput - scale * freq_product

    orig_coinput, orig_resid = coinput_and_resid(orig_binary)
    shuf_coinput, shuf_resid = coinput_and_resid(shuf_binary)
    coinput_corr = np.corrcoef(orig_coinput, shuf_coinput)[0, 1]
    coinput_resid_corr = np.corrcoef(orig_resid, shuf_resid)[0, 1]

    return pd.Series({
        'n_kcs': len(orig_counts),
        'n_claws': n_claws,
        'frac_claws_changed': float(frac_changed),
        'kc_n_claws_preserved': bool(np.array_equal(orig_counts.sum(axis=1),
            shuf_counts.sum(axis=1)
        )),
        'kc_n_glomeruli_preserved': bool(np.array_equal(orig_binary.sum(axis=1),
            shuf_binary.sum(axis=1)
        )),
        'max_abs_glom_n_claws_change': int(
            np.abs(orig_counts.sum(axis=0) - shuf_counts.sum(axis=0)).max()
        ),
        'max_abs_glom_n_kcs_change': int(
            np.abs(orig_binary.sum(axis=0) - shuf_binary.sum(axis=0)).max()
        ),
        'coinput_corr': float(coinput_corr),
        'coinput_resid_corr': float(coinput_resid_corr),
    }, dtype=object)


def _column_groups(columns: pd.Index, by: Optional[str]) -> Tuple[np.ndarray, pd.Index]:
    """Returns (group code per column, group labels): the columns grouped by MultiIndex
    level `by` when they have it, else each column its own group.
    """
    if by is not None and isinstance(columns, pd.MultiIndex) and by in columns.names:
        codes, groups = pd.factorize(columns.get_level_values(by))
        return np.asarray(codes), pd.Index(groups)

    return np.arange(len(columns)), pd.Index(columns)


def coinput_counts(wPNKC: pd.DataFrame, by: Optional[str] = 'glomerulus') -> pd.DataFrame:
    """Returns # of KCs with claws from both glomeruli of each glomerulus pair.

    Args:
        wPNKC: either `shuffle_wPNKC` format.
        by: column level to group the columns by before counting (default 'glomerulus',
            the first level of the bouton model's (glomerulus, pn_id, bouton_id) columns).
            None, or a level the columns do not have, counts per column.

    Returns symmetric (groups x groups) DataFrame of counts; the diagonal is each group's
    number of KCs (with any claw from it).
    """
    counts = _kc_counts(wPNKC)
    codes, groups = _column_groups(wPNKC.columns, by)
    onehot = np.zeros((len(codes), len(groups)), dtype=np.int64)
    onehot[np.arange(len(codes)), codes] = 1
    binary = ((counts @ onehot) > 0).astype(np.int64)
    return pd.DataFrame(binary.T @ binary, index=groups, columns=groups)


def lesion_coinput(wPNKC: pd.DataFrame, targets: Dict[Tuple, int], *, seed: int = 0,
    within_kc_type: bool = True, protect: Optional[Iterable[Tuple]] = None,
    by: Optional[str] = 'glomerulus', max_tries: int = 2000, strict: bool = True,
    passes: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns one-row-per-claw `wPNKC` with the number of KCs sharing each listed pair of
    glomeruli brought to a target, by claw swaps that keep every margin, and the swaps made.

    For each pair in `targets` (in order), while more KCs than its target have claws from
    both its glomeruli: a random such KC gives up all its claws from the glomerulus of the
    pair it has fewer claws from (ties random); each of those claws exchanges its column
    (bouton) with a claw of a random other KC (of the same `KC_TYPE` with `within_kc_type`)
    that has claws from neither glomerulus of the pair, chosen so that no pair in `targets`
    or `protect` gains a KC (the partner claw's glomerulus may not form a listed pair with a
    glomerulus the first KC keeps, nor the moved glomerulus with one the partner KC has) and
    no pair in `protect` loses one either (neither KC may give up the last claw of a
    glomerulus that forms a protected pair with another of its glomeruli). While fewer KCs
    than the target share the pair (a transplant of co-input, e.g. into a shuffled wiring):
    a random KC with claws from one glomerulus of the pair and none from the other takes one
    claw of the other from a random KC of the same group that has the other and not the
    first, giving one of its own claws from a third glomerulus in exchange, so that no
    protected pair changes (other target pairs may change, since a transplanted co-input
    structure has its pairs on common KCs; with `passes` > 1 the targets are revisited until
    every pair is at its target or a pass makes no progress). So every KC keeps its number
    of claws, every column (bouton, hence glomerulus) its number of claws (per KC type with
    `within_kc_type`), no listed pair gains a KC while pairs are being lowered, and
    protected pairs keep their counts; a KC's number of distinct glomeruli and the counts of
    unlisted pairs change incidentally, as under a claw permutation restricted to the swapped
    claws.

    Args:
        wPNKC: one row per claw (`CLAW_ID` index level), 0/1 values.
        targets: {(glom_a, glom_b): target # of KCs}, glomeruli being labels of column
            level `by` (or columns). A target equal to the current count changes nothing
            for that pair; below it lowers the count, above it raises it.
        seed: for `np.random.default_rng`. Output is reproducible for a given seed.
        within_kc_type: partners come from the same `KC_TYPE` as the KC giving up a claw.
        protect: pairs whose counts may not change (other targets may fall below their
            targets incidentally, never rise).
        max_tries: random partner draws per claw, and KC draws per pair, before giving up.
        strict: raise when a pair cannot be brought to its target (e.g. every KC sharing it
            would break a protected pair by giving up either glomerulus); False leaves such
            a pair where it got to and goes on (compare the result's counts with the
            targets, e.g. with `coinput_counts`).
        passes: how many times to go over `targets` (later pairs' swaps can move earlier
            pairs' counts, mostly when raising); stops early once every pair is at its
            target or a pass changed nothing.

    Returns (lesioned copy of `wPNKC`, DataFrame of the swaps: glom_a, glom_b (the pair
    being changed), n_before (its count before the step), kc, moved (the glomerulus the KC
    gave up), partner_kc, received (the glomerulus it got), claw_row, partner_row (row
    positions in `wPNKC`)).

    Raises:
        RuntimeError: a pair could not be brought to its target (`strict`).
    """
    if CLAW_ID not in wPNKC.index.names:
        raise ValueError('lesion_coinput needs one-row-per-claw wPNKC (a CLAW_ID index level)')

    values = _wPNKC_values(wPNKC, True)
    row_kc, n_kcs = _kc_codes(wPNKC)
    claw_rows = np.flatnonzero(values.any(axis=1))
    claw_col = values[claw_rows].argmax(axis=1)
    claw_kc = row_kc[claw_rows]
    col_group, groups = _column_groups(wPNKC.columns, by)
    claw_glom = col_group[claw_col].copy()
    n_gloms = len(groups)
    gidx = {g: i for i, g in enumerate(groups)}
    if within_kc_type:
        kc_group = _kc_groups(wPNKC, row_kc, n_kcs)
    else:
        kc_group = np.zeros(n_kcs, dtype=np.int64)

    def pair_idx(p) -> Tuple[int, int]:
        if len(p) != 2 or p[0] not in gidx or p[1] not in gidx:
            raise ValueError(f'pair {p!r} is not two glomeruli of the wiring')
        if p[0] == p[1]:
            raise ValueError(f'pair {p!r} names one glomerulus twice')
        return gidx[p[0]], gidx[p[1]]

    listed = np.zeros((n_gloms, n_gloms), dtype=bool)
    for p in list(targets) + list(protect or ()):
        a, b = pair_idx(p)
        listed[a, b] = listed[b, a] = True
    protected = np.zeros((n_gloms, n_gloms), dtype=bool)
    for p in (protect or ()):
        a, b = pair_idx(p)
        protected[a, b] = protected[b, a] = True

    counts = np.zeros((n_kcs, n_gloms), dtype=np.int64)
    np.add.at(counts, (claw_kc, claw_glom), 1)
    # the claws of each KC (swaps change a claw's column, never its KC)
    order = np.argsort(claw_kc, kind='stable')
    bounds = np.searchsorted(claw_kc[order], np.arange(n_kcs + 1))
    kc_claws = [order[bounds[k]:bounds[k + 1]] for k in range(n_kcs)]
    group_claws = {g: np.flatnonzero(kc_group[claw_kc] == g) for g in np.unique(kc_group)}
    kc_ids = wPNKC.index.get_level_values(KC_ID)

    rng = np.random.default_rng(seed)
    new_col = claw_col.copy()
    swaps: List[dict] = []

    def partner_for(k: int, moved: int, other: int) -> Optional[int]:
        """A claw (index into claw_rows) of another KC of k's group with neither `moved` nor
        `other`, whose glomerulus can come to k without any listed pair gaining a KC."""
        pool = group_claws[kc_group[k]]
        keeps_k = counts[k] > 0
        if counts[k, moved] == 1:
            keeps_k = keeps_k.copy()
            keeps_k[moved] = False
        for _ in range(max_tries):
            r2 = pool[rng.integers(len(pool))]
            k2 = claw_kc[r2]
            if k2 == k or counts[k2, moved] > 0 or counts[k2, other] > 0:
                continue
            x = claw_glom[r2]
            if x == moved or listed[x][keeps_k].any():
                continue
            keeps_k2 = counts[k2] > 0
            if counts[k2, x] == 1:
                # k2 gives up x: no protected pair (x, y) may lose k2
                if protected[x][keeps_k2].any():
                    continue
                keeps_k2 = keeps_k2.copy()
                keeps_k2[x] = False
            if listed[moved][keeps_k2].any():
                continue
            return int(r2)
        return None

    def donor_for(k: int, need: int, have: int) -> Optional[Tuple[int, int]]:
        """(claw of k to give, claw of another KC of k's group to take) so that k gains
        glomerulus `need` (it has `have`) without any other listed pair gaining a KC or a
        protected pair losing one; or None."""
        pool = group_claws[kc_group[k]]
        own = [r for r in kc_claws[k] if claw_glom[r] != have and claw_glom[r] != need]
        if not own:
            return None
        # `need` arrives at k: no protected pair (need, y) may gain k
        others_k = counts[k] > 0
        others_k = others_k.copy()
        others_k[have] = False
        if protected[need][others_k].any():
            return None
        for _ in range(max_tries):
            r2 = pool[rng.integers(len(pool))]
            k2 = int(claw_kc[r2])
            if k2 == k or claw_glom[r2] != need or counts[k2, have] > 0:
                continue
            r1 = int(own[rng.integers(len(own))])
            x = int(claw_glom[r1])
            # k gives up x: no protected pair (x, y) may lose k
            if counts[k, x] == 1 and protected[x][counts[k] > 0].any():
                continue
            # x arrives at k2, which gives up one `need` claw
            keeps_k2 = counts[k2] > 0
            if counts[k2, need] == 1:
                if protected[need][keeps_k2].any():
                    continue
                keeps_k2 = keeps_k2.copy()
                keeps_k2[need] = False
            if protected[x][keeps_k2].any():
                continue
            return r1, int(r2)
        return None

    for _pass in range(max(int(passes), 1)):
      n_swaps_before = len(swaps)
      for p, target in targets.items():
        a, b = pair_idx(p)
        tries = 0
        while True:
            both = np.flatnonzero((counts[:, a] > 0) & (counts[:, b] > 0))
            if len(both) == target:
                break
            if tries >= max_tries:
                if strict:
                    raise RuntimeError(f'pair {p!r}: stuck at {len(both)} KCs sharing it '
                        f'(target {target}) after {tries} attempts'
                    )
                break
            tries += 1
            if len(both) < target:
                # raise: a KC with one glomerulus of the pair takes a claw of the other
                one = np.flatnonzero((counts[:, a] > 0) != (counts[:, b] > 0))
                if len(one) == 0:
                    continue
                k = int(one[rng.integers(len(one))])
                have, need = (a, b) if counts[k, a] > 0 else (b, a)
                found = donor_for(k, need, have)
                if found is None:
                    continue
                r1, r2 = found
                k2, x = int(claw_kc[r2]), int(claw_glom[r1])
                new_col[r1], new_col[r2] = new_col[r2], new_col[r1]
                claw_glom[r1], claw_glom[r2] = need, x
                counts[k, x] -= 1
                counts[k, need] += 1
                counts[k2, need] -= 1
                counts[k2, x] += 1
                swaps.append(dict(glom_a=p[0], glom_b=p[1], n_before=len(both),
                    kc=kc_ids[claw_rows[r1]], moved=groups[x],
                    partner_kc=kc_ids[claw_rows[r2]], received=groups[need],
                    claw_row=int(claw_rows[r1]), partner_row=int(claw_rows[r2])
                ))
                continue
            k = int(both[rng.integers(len(both))])
            ca, cb = counts[k, a], counts[k, b]
            prefer = [(a, b), (b, a)] if ca < cb or (ca == cb and rng.random() < 0.5) else [(b, a), (a, b)]
            # k may not give up a glomerulus that forms a protected pair with another of its own
            choice = [(m, o) for m, o in prefer if not protected[m][counts[k] > 0].any()]
            if not choice:
                continue
            moved, other = choice[0]
            n_before = len(both)
            for r1 in [r for r in kc_claws[k] if claw_glom[r] == moved]:
                r2 = partner_for(k, moved, other)
                if r2 is None:
                    # k keeps both glomeruli (its swaps so far still keep every margin);
                    # another KC is tried
                    break
                k2, x = int(claw_kc[r2]), int(claw_glom[r2])
                new_col[r1], new_col[r2] = new_col[r2], new_col[r1]
                claw_glom[r1], claw_glom[r2] = x, moved
                counts[k, moved] -= 1
                counts[k, x] += 1
                counts[k2, x] -= 1
                counts[k2, moved] += 1
                swaps.append(dict(glom_a=p[0], glom_b=p[1], n_before=n_before,
                    kc=kc_ids[claw_rows[r1]], moved=groups[moved],
                    partner_kc=kc_ids[claw_rows[r2]], received=groups[x],
                    claw_row=int(claw_rows[r1]), partner_row=int(claw_rows[r2])
                ))
      at_target = all(int(np.count_nonzero((counts[:, pair_idx(p)[0]] > 0) & (counts[:, pair_idx(p)[1]] > 0))) == t
          for p, t in targets.items())
      if at_target or len(swaps) == n_swaps_before:
          break

    new_values = np.zeros_like(values)
    new_values[claw_rows, new_col] = 1
    lesioned = pd.DataFrame(new_values, index=wPNKC.index.copy(),
        columns=wPNKC.columns.copy()
    )
    if wPNKC.dtypes.nunique() > 1:
        lesioned = lesioned.astype(wPNKC.dtypes.to_dict())

    cols = ['glom_a', 'glom_b', 'n_before', 'kc', 'moved', 'partner_kc', 'received',
        'claw_row', 'partner_row'
    ]
    return lesioned, pd.DataFrame(swaps, columns=cols)
