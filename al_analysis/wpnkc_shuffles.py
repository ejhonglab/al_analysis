"""Shuffled (null model) versions of PN->KC connectivity matrices (`wPNKC`).

For testing how much of the model's odor-odor structure comes from the specific
connectome wiring, rather than from claw count and glomerulus frequency statistics alone.
Outputs of `shuffle_wPNKC` can be passed to `fit_mb_model` via its `_wPNKC` keyword
(along with the same `one_row_per_claw` / `prat_claws` / etc flags that would produce
the unshuffled matrix).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from al_analysis.mb_model import KC_ID, KC_TYPE, CLAW_ID


SHUFFLE_METHODS: Tuple[str, ...] = ('claw_permutation', 'curveball')

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
