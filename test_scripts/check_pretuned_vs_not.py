#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

from al_analysis import model_responses_cache_name, model_spikecounts_cache_name


def main():
    root = Path('../pebbled_6f/pdf/ijroi/mb_modeling/megamat')

    h1 = (
        'dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_hemibrain__'
        'target-sp_0.0915'
    )
    h2 = (
        'tuned-on_megamat__dff_scale-to-avg-max__data_pebbled__hallem-tune_False__'
        'pn2kc_hemibrain__fixed-thr_204__wAPLKC_3.78'
    )

    u1 = (
        'dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_uniform__'
        'n-claws_7__target-sp_0.0915__n-seeds_3'
    )
    u2 = (
        'tuned-on_megamat__dff_scale-to-avg-max__data_pebbled__hallem-tune_False__'
        'pn2kc_uniform__n-claws_7__n-seeds_3'
    )

    name_pairs = [(h1, h2), (u1, u2)]

    for n1, n2 in name_pairs:
        d1 = root / n1
        d2 = root / n2

        r1 = pd.read_pickle(d1 / model_responses_cache_name)
        r2 = pd.read_pickle(d2 / model_responses_cache_name)
        # TODO failing now (2025-03-12)... i assume this was working when i initially
        # ran it? is it b/c one of the input dirs has since diverged (and how could that
        # be?)?
        assert r1.equals(r2)

        s1 = pd.read_pickle(d1 / model_spikecounts_cache_name)
        s2 = pd.read_pickle(d2 / model_spikecounts_cache_name)
        assert s1.equals(s2)


if __name__ == '__main__':
    main()

