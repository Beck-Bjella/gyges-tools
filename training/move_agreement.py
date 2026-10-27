"""Move agreement: does each net, looking one move ahead, pick the move the data's player played?

For sampled positions from a dataset (move 4 on — the first 4 plies of data gen games are
random), every legal move's resulting position comes from `gyges-tools children`. Each net
scores those children (from the opponent's side, so the net's pick is the child with the
lowest score) and is checked against the move actually played (the next row of the game).
Positions where a move wins on the spot are left out (every net finds those).

Reports, per move-number bucket:
  agree  = share of positions where the net's top pick is the played move
  rank   = average share of legal moves the net rates above the played move (0 = it's the top)
  random = what picking a legal move at random would score on agree

Usage: python training/move_agreement.py [dataset] [positions] [net ...]
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from common import REPO_ROOT
from phase_analysis import DEFAULT_NETS, load_net, predict
from train_pair import device

BUCKETS = [(4, 5), (6, 9), (10, 14), (15, 19), (20, 999)]
SCRATCH = os.path.join(REPO_ROOT, "results", "tmp_move_agreement")
TOOLS = os.path.join(REPO_ROOT, "target", "release", "gyges-tools.exe" if os.name == "nt" else "gyges-tools")


def sample_positions(dataset, n, seed=1):
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "training", "data", dataset, "w*.csv")))
    df = pd.concat([pd.read_csv(f, header=None, dtype=np.int64) for f in files], ignore_index=True)
    sq = list(range(36))
    # Next row of the same game = the position after the move that was played.
    nxt = df.shift(-1)
    has_next = (nxt[38] == df[38]) & (nxt[37] == df[37] + 1)
    pool = df.index[has_next & (df[37] >= 4)]
    pick = np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False)
    pos = df.loc[pick, sq].to_numpy()
    played = nxt.loc[pick, sq].to_numpy().astype(np.int64)
    return pos, played, df.loc[pick, 37].to_numpy()


def children_of(pos):
    os.makedirs(SCRATCH, exist_ok=True)
    inp, out = os.path.join(SCRATCH, "positions.csv"), os.path.join(SCRATCH, "children.csv")
    with open(inp, "w") as f:
        for i, b in enumerate(pos):
            f.write(f"{i}," + ",".join(map(str, b)) + "\n")
    print(subprocess.run([TOOLS, "children", inp, out], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True).stdout.strip())
    ch = pd.read_csv(out, header=None, dtype=np.int64)
    return ch[0].to_numpy(), ch[1].to_numpy(), ch.iloc[:, 2:38].to_numpy()


def main(dataset, n, nets):
    pos, played, mv = sample_positions(dataset, n)
    pid, win, kids = children_of(pos)

    # The played move must be one of the generated moves — checks orientation end to end.
    is_played = (kids == played[pid]).all(axis=1)
    found = np.bincount(pid[is_played], minlength=len(pos)) > 0
    if not found.all():
        raise SystemExit(f"played move not among generated children for {(~found).sum()} positions")
    keep = np.bincount(pid[win == 1], minlength=len(pos)) == 0         # no immediate win available
    n_kids = np.bincount(pid, minlength=len(pos))
    print(f"{len(pos):,} positions, played move found for all; {keep.sum():,} without an immediate win, "
          f"{n_kids[keep].mean():.1f} legal moves on average")

    boards = torch.from_numpy(kids.astype(np.int8)).to(device)
    rows = []
    for name in nets:
        score = predict(load_net(name), boards)          # from the opponent's side
        order = np.lexsort((score, pid))                 # by position, then score ascending
        first = np.r_[0, np.flatnonzero(np.diff(pid[order])) + 1]
        best = order[first]                              # each position's lowest-scored child
        agree = np.zeros(len(pos), bool)
        agree[pid[best]] = is_played[best]
        # Share of moves rated better than the played one.
        played_score = np.full(len(pos), np.inf)
        np.minimum.at(played_score, pid[is_played], score[is_played])
        better = np.bincount(pid, weights=(score < played_score[pid]).astype(float), minlength=len(pos))
        rank = better / np.maximum(n_kids - 1, 1)
        for lo, hi in BUCKETS:
            m = keep & (mv >= lo) & (mv <= hi)
            rows.append(dict(net=name, phase=f"{lo}-{hi}" if hi < 999 else f"{lo}+", n=int(m.sum()),
                             agree=round(float(agree[m].mean()), 4), rank=round(float(rank[m].mean()), 4),
                             random=round(float((1 / n_kids[m]).mean()), 4)))
        m = keep
        rows.append(dict(net=name, phase="all", n=int(m.sum()), agree=round(float(agree[m].mean()), 4),
                         rank=round(float(rank[m].mean()), 4), random=round(float((1 / n_kids[m]).mean()), 4)))
    out = pd.DataFrame(rows)
    # Rerunning nets replaces their rows; other nets' results are kept.
    path = os.path.join(REPO_ROOT, "results", f"move_agreement_{dataset}.csv")
    saved = pd.concat([pd.read_csv(path).query("net not in @nets"), out]) if os.path.exists(path) else out
    saved.to_csv(path, index=False)
    short = {nm: nm.replace("pair1024", "").replace("+gen1-10k-aws+gen1-10k-aws2", "+gen1(20M)")
             .replace("gen1-10k+gen1-10k-aws_", "gen1(5M)_")[:40] for nm in nets}
    phases = [f"{lo}-{hi}" if hi < 999 else f"{lo}+" for lo, hi in BUCKETS] + ["all"]
    for metric in ("agree", "rank"):
        t = out.pivot(index="phase", columns="net", values=metric).reindex(phases)[nets].rename(columns=short)
        base = out[out.net == nets[0]].set_index("phase").reindex(phases)
        t.insert(0, "n", base.n)
        if metric == "agree":
            t.insert(1, "random", base.random)
        print(f"\n{metric.upper()} by move number")
        print(t.to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "gen2-10k-aws", int(a[1]) if len(a) > 1 else 50_000, a[2:] or DEFAULT_NETS)
