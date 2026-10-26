"""Do hce-10k and gen1 (19.9M) label the same boards differently?

Boards are canonicalized over the left-right mirror (training mirrors anyway).
Cross-source label differences are compared against a within-source baseline:
each dataset split in two by game id, same metric on boards both halves share.
"""
import glob
import os

import numpy as np
import pandas as pd

from common import REPO_ROOT as REPO, Dataset, MIRROR_PERM

REL = os.path.join(REPO, "V2.0.0_release_dataset")
HCE = [os.path.join(REL, "hce_10kn.csv"), os.path.join(REL, "hce_10kn_big.csv")]
GEN1 = []
for part in ("gen1-10k", "gen1-10k-aws", "gen1-10k-aws2"):
    GEN1 += sorted(glob.glob(os.path.join(REPO, "training", "data", part, "w*.csv")))


def pack(b):
    lo = np.zeros(len(b), dtype=np.uint64)
    for sq in range(32):
        lo |= b[:, sq].astype(np.uint64) << np.uint64(2 * sq)
    hi = np.zeros(len(b), dtype=np.uint16)
    for sq in range(32, 36):
        hi |= b[:, sq].astype(np.uint16) << np.uint16(2 * (sq - 32))
    return lo, hi


def canon_keys(boards):
    lo, hi = pack(boards)
    mlo, mhi = pack(boards[:, MIRROR_PERM])
    take_m = (mhi < hi) | ((mhi == hi) & (mlo < lo))
    return np.where(take_m, mlo, lo), np.where(take_m, mhi, hi)


def per_board(ds, label):
    lo, hi = canon_keys(ds.boards)
    df = pd.DataFrame({"lo": lo, "hi": hi, "y": ds.outcomes, "mv": ds.move_nums, "g": ds.game_ids})
    print(f"{label}: {len(df):,} rows")
    return df


def agg(df):
    return df.groupby(["lo", "hi"]).agg(n=("y", "size"), y=("y", "mean"), mv=("mv", "min"))


def compare(a, b, name):
    """Boards in both a and b: how far apart are their mean labels?"""
    m = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    if m.empty:
        print(f"  {name}: no shared boards")
        return m
    w = np.minimum(m.n_a, m.n_b)                  # weight: rows backing both means
    d = (m.y_a - m.y_b).abs()
    opp = (np.sign(m.y_a) * np.sign(m.y_b) < 0) & (m.y_a.abs() >= 0.5) & (m.y_b.abs() >= 0.5)
    print(f"  {name}: {len(m):,} shared boards | mean |dlabel| {d.mean():.3f} "
          f"(weighted {np.average(d, weights=w):.3f}) | "
          f"strong opposite (both |mean|>=0.5) {100 * opp.mean():.1f}% "
          f"(weighted {100 * np.average(opp, weights=w):.1f}%)")
    return m


def halves(df):
    odd = (df.g % 2).astype(bool)
    return agg(df[~odd]), agg(df[odd])


def by_move(m, name):
    # Bucket by earliest move number the board appears at in either set.
    mv = np.minimum(m.mv_a, m.mv_b)
    buckets = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 999)]
    print(f"  {name} by earliest move_num:")
    for lo_, hi_ in buckets:
        s = m[(mv >= lo_) & (mv <= hi_)]
        if s.empty:
            print(f"    {lo_}-{hi_}: 0")
            continue
        d = (s.y_a - s.y_b).abs()
        opp = (np.sign(s.y_a) * np.sign(s.y_b) < 0) & (s.y_a.abs() >= 0.5) & (s.y_b.abs() >= 0.5)
        print(f"    move {lo_}-{hi_}: {len(s):>10,} boards | rows hce {int(s.n_a.sum()):>11,} gen1 {int(s.n_b.sum()):>11,} | "
              f"mean |d| {d.mean():.3f} | strong opposite {100 * opp.mean():.1f}% | "
              f"mean label hce {np.average(s.y_a, weights=s.n_a):+.3f} gen1 {np.average(s.y_b, weights=s.n_b):+.3f}")


if __name__ == "__main__":
    hce = per_board(Dataset.from_files(HCE, outcome_scale=1.0), "hce-10k")
    gen = per_board(Dataset.from_files(GEN1, outcome_scale=1.0), "gen1 19.9M")

    A, B = agg(hce), agg(gen)
    print(f"\nunique boards (mirror-canonical): hce {len(A):,}  gen1 {len(B):,}")

    print("\nCROSS-SOURCE (hce vs gen1)")
    m = compare(A, B, "hce vs gen1")
    rows_hce = int(m.n_a.sum()); rows_gen = int(m.n_b.sum())
    print(f"  rows on shared boards: hce {rows_hce:,} ({100 * rows_hce / len(hce):.1f}% of hce), "
          f"gen1 {rows_gen:,} ({100 * rows_gen / len(gen):.1f}% of gen1)")
    by_move(m, "hce vs gen1")

    print("\nWITHIN-SOURCE BASELINE (each set split in two by game id)")
    ha, hb = halves(hce)
    mh = compare(ha, hb, "hce half vs half")
    by_move(mh, "hce halves")
    ga, gb = halves(gen)
    mg = compare(ga, gb, "gen1 half vs half")
    by_move(mg, "gen1 halves")
