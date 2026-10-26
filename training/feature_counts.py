"""How many training examples does each of the 5886 pair features get?

Counts feature activations over a dataset's rows (before the train/val split
and mirroring; training sees ~0.8 x 2 = 1.6x these counts). Reads one file at
a time in chunks so it stays small in RAM and leaves the GPU alone.

Usage: python training/feature_counts.py <dataset>[+<dataset>...]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from common import REPO_ROOT

FEATURE_COUNT = 5886
RELEASE_FILES = {
    "hce-10k-part1": ["hce_10kn.csv"],
    "hce-10k-part2": ["hce_10kn_big.csv"],
    "hce-10k":       ["hce_10kn.csv", "hce_10kn_big.csv"],
}

# Same layout as train_pair.py.
_PAIR_IDX = np.full((36, 36), -1, dtype=np.int64)
_k = 0
for _a in range(36):
    for _b in range(_a + 1, 36):
        _PAIR_IDX[_a, _b] = _PAIR_IDX[_b, _a] = _k
        _k += 1
_SAME_OFF = np.array([0, 108, 738, 1368])
_CROSS_OFF = np.zeros((4, 4), dtype=np.int64)
for _lo, _hi, _off in ((1, 2, 1998), (1, 3, 3294), (2, 3, 4590)):
    _CROSS_OFF[_lo, _hi] = _CROSS_OFF[_hi, _lo] = _off


def encode(boards, max_pieces=12):
    """numpy port of train_pair.encode_batch: (B, 36) -> (B, slots), FEATURE_COUNT = empty."""
    b = boards.astype(np.int64)
    sq_all = np.arange(36)
    keys = np.sort(np.where(b != 0, sq_all, sq_all + 36), axis=1)[:, :max_pieces]
    valid = keys < 36
    sq = keys % 36
    t = np.where(valid, np.take_along_axis(b, sq, 1), 0)
    single = np.where(valid, sq * 3 + t - 1, FEATURE_COUNT)
    i, j = np.triu_indices(max_pieces, 1)
    si, sj, ti, tj = sq[:, i], sq[:, j], t[:, i], t[:, j]
    same = _SAME_OFF[ti] + _PAIR_IDX[si, sj]
    lo_first = ti < tj
    cross = _CROSS_OFF[ti, tj] + np.where(lo_first, si, sj) * 36 + np.where(lo_first, sj, si)
    pair = np.where(valid[:, i] & valid[:, j], np.where(ti == tj, same, cross), FEATURE_COUNT)
    return np.concatenate([single, pair], axis=1)


def files_for(dataset):
    files = []
    for part in dataset.split("+"):
        if part in RELEASE_FILES:
            files += [os.path.join(REPO_ROOT, "V2.0.0_release_dataset", f) for f in RELEASE_FILES[part]]
        else:
            files += sorted(glob.glob(os.path.join(REPO_ROOT, "training", "data", part, "w*.csv")))
    return files


def main(dataset):
    counts = np.zeros(FEATURE_COUNT + 1, dtype=np.int64)
    rows = 0
    for f in files_for(dataset):
        for chunk in pd.read_csv(f, header=None, usecols=range(36), dtype=np.int8, chunksize=1_000_000):
            counts += np.bincount(encode(chunk.to_numpy()).ravel(), minlength=FEATURE_COUNT + 1)
            rows += len(chunk)
    c = counts[:FEATURE_COUNT]
    params = FEATURE_COUNT * 1024 + 1024 + 1024 + 1
    train_rows = rows * 0.8 * 2
    print(f"{dataset}: {rows:,} rows -> ~{train_rows / 1e6:.1f}M train rows (80% split, mirrored)")
    print(f"  params {params:,} -> {train_rows / params:.1f} train rows per parameter")
    print(f"  active features per row: {c.sum() / rows:.1f}")
    print("  examples per feature (raw rows; training sees ~1.6x):")
    for q in (0, 1, 5, 10, 25, 50, 75, 100):
        print(f"    {q:>3}th percentile: {int(np.percentile(c, q)):>12,}")
    for lim in (0, 100, 1_000, 10_000, 100_000):
        print(f"    features with <= {lim:>7,} examples: {int((c <= lim).sum()):>5} of {FEATURE_COUNT}")
    for name, lo, hi in (("singletons", 0, 108), ("same-type pairs", 108, 1998), ("cross-type pairs", 1998, 5886)):
        s = c[lo:hi]
        print(f"    {name:<17} median {int(np.median(s)):>11,}  min {int(s.min()):>9,}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gen1-10k+gen1-10k-aws+gen1-10k-aws2")
