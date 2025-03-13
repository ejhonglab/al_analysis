#!/usr/bin/env python3

from al_analysis import connectome_wPNKC


def main():
    wPNKC = connectome_wPNKC(_use_matt_wPNKC=True)
    # TODO TODO why are these max values in the 1-4 range? synapse counts (/
    # weights) in what prat gave me are almost all >10 (he sets cutoff of >= 5 to
    # even count connection...). does matt's code actually do anything with any of
    # these >1? wPNKC.max()

    wPNKC2 = connectome_wPNKC()

    # TODO TODO does distribution of synapse counts in wPNKC (from matt's gkc-...)
    # match what we expect? does matt already make this somewhere?
    # TODO TODO what about double draw (KC drawing from same glomerulus)
    # frequencies? those match the literature? matt make this somewhere?

    # TODO try to recreate matt's wPNKC matrix by subsetting prat's stuff to
    # hallem glomeruli and body IDs of KCs that have some connections from them?

    # TODO matt only had 67 VM2 connections, but it seems even in 1.1 and 1.2 had 150
    # VM2 connections (and 1.0 right?) (nor did earlier version seem to have diff VA1v
    # connections, right?)
    # TODO TODO investigate wPNKC differences further! (VM2 & VA1v diffs first)

    # TODO make distribution of this, like in matt-hemibrain/docs/mb-claws.html
    # (seems to match up pretty well by inspecting .value_counts())
    #wPNKC2.sum(axis='columns')

    # TODO delete
    #w2 = wPNKC2[wPNKC.columns].copy()
    # ipdb> (w2.sum(axis='columns') > 0).sum()
    # 1652
    #w2 = w2[(w2.sum(axis='columns') > 0)].copy()

    # ipdb> w2.index.isin(wPNKC.index).all()
    # False
    # ipdb> wPNKC.index.isin(w2.index).all()
    # False
    #
    # ipdb> len(set(wPNKC.index) - set(w2.index))
    # 43
    # ipdb> len(set(w2.index) - set(wPNKC.index))
    # 65
    #
    # ipdb> w2.shape
    # (1652, 22)
    # ipdb> wPNKC.shape
    # (1630, 22)

    # TODO maybe compare values between shared bodyids across w2 and wPNKC?

    # TODO TODO seems like wPNKC tends to have some connections w2 doesn't...
    # what's up with that?
    #
    # glomerulus
    # DL5     0
    # VM3     0
    # DL1     0
    # DC1     0
    # DM2     0
    # DA3     0
    # VC3     0
    # DA4l    0
    # VM2     1
    # DM3     0
    # VA1v    0
    # VA5     0
    # DM4     0
    # DL3     0
    # DM6     0
    # VC4     0
    # VA6     0
    # DM5     0
    # VM5d    0
    # DL4     0
    # VA1d    0
    # VM5v    0
    # dtype: int64
    # ipdb> (w2[w2.index.isin(wPNKC.index)] - wPNKC[wPNKC.index.isin(w2.index)]).max().max()
    # 1
    # ipdb> (w2[w2.index.isin(wPNKC.index)] - wPNKC[wPNKC.index.isin(w2.index)]).min().min()
    # -2
    # ipdb> (w2[w2.index.isin(wPNKC.index)] - wPNKC[wPNKC.index.isin(w2.index)]).min()
    # glomerulus
    # DL5    -1
    # VM3    -1
    # DL1    -1
    # DC1    -1
    # DM2    -1
    # DA3    -1
    # VC3    -2
    # DA4l   -1
    # VM2    -1
    # DM3    -1
    # VA1v   -1
    # VA5    -1
    # DM4    -1
    # DL3     0
    # DM6    -2
    # VC4    -1
    # VA6    -1
    # DM5    -1
    # VM5d   -1
    # DL4    -1
    # VA1d   -1
    # VM5v   -1
    #
    # looks like ~70% of KCs have same connections:
    # ipdb> (w2[w2.index.isin(wPNKC.index)].sort_index() == wPNKC[wPNKC.index.isin(w2.index)].sort_index())
    # .all(axis='columns').sum()
    # 1143
    # ipdb> w2.shape
    # (1652, 22)
    # ipdb> 1143/1652
    # 0.6918886198547215
    #
    # ipdb> w2[w2.index.isin(wPNKC.index)].sort_index()[(w2[w2.index.isin(wPNKC.index)].sort_index() != wPN
    # KC[wPNKC.index.isin(w2.index)].sort_index())].sum()
    # glomerulus
    # DL5      0.0
    # VM3      0.0
    # DL1      0.0
    # DC1      3.0
    # DM2      0.0
    # DA3      0.0
    # VC3     23.0
    # DA4l     0.0
    # VM2     82.0
    # DM3      0.0
    # VA1v    16.0
    # VA5      0.0
    # DM4      0.0
    # DL3      0.0
    # DM6      1.0
    # VC4      0.0
    # VA6      0.0
    # DM5      2.0
    # VM5d     3.0
    # DL4      0.0
    # VA1d     0.0
    # VM5v     0.0

    # TODO are they close enough at this point?
    # maybe the differences don't matter?

    #
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

