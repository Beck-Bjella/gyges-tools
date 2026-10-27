"""How well does each net predict game results at each phase of the game?

Runs every net over one dataset (default gen2-10k-aws: well played, and none of the
gen0/gen1 nets were trained on it) and reports, per move-number bucket:
  R2   = 1 - MSE / variance of the labels in that bucket (0 = no better than the
         bucket's average result, 1 = perfect), labels scaled x0.75 as in training
  acc  = how often the sign of the eval matches the winner (draws left out)

Usage: python training/phase_analysis.py [dataset] [net ...]   (nets = files in engines/weights/)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

from common import REPO_ROOT
from train_pair import FEATURE_COUNT, device, encode_batch

BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 14), (15, 19), (20, 29), (30, 999)]
DEFAULT_NETS = [
    "pair1024_hce-10k_e9",
    "pair1024_gen1-10k+gen1-10k-aws_e6",
    "pair1024_gen1-1k_e5",
    "pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9",
    "pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12",
    "pair1024x32_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12",
]
L1, TAIL = 1024, 32


def load_net(name):
    w = np.fromfile(os.path.join(REPO_ROOT, "engines", "weights", f"{name}.bin"), dtype=np.float32)
    n1 = L1 * FEATURE_COUNT
    w1t = np.zeros((FEATURE_COUNT + 1, L1), dtype=np.float32)       # + zero padding row
    w1t[:FEATURE_COUNT] = w[:n1].reshape(L1, FEATURE_COUNT).T
    b1 = w[n1:n1 + L1]
    rest = w[n1 + L1:]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    if len(rest) == L1 + 1:
        return dict(w1t=t(w1t), b1=t(b1), w2=t(rest[:L1]), b2=float(rest[L1]))
    assert len(rest) == TAIL * L1 + TAIL + TAIL + 1, f"unexpected size for {name}"
    w2 = rest[:TAIL * L1].reshape(TAIL, L1)
    b2 = rest[TAIL * L1:TAIL * L1 + TAIL]
    w3 = rest[TAIL * L1 + TAIL:TAIL * L1 + 2 * TAIL]
    return dict(w1t=t(w1t), b1=t(b1), tw2=t(w2), tb2=t(b2), w3=t(w3), b3=float(rest[-1]))


@torch.no_grad()
def predict(net, boards, batch=65536):
    out = []
    for i in range(0, len(boards), batch):
        idx = encode_batch(boards[i:i + batch], 12)
        h = torch.relu(torch.nn.functional.embedding_bag(idx, net["w1t"], mode="sum") + net["b1"])
        if "w2" in net:
            y = h @ net["w2"] + net["b2"]
        else:
            y = torch.relu(h @ net["tw2"].T + net["tb2"]) @ net["w3"] + net["b3"]
        out.append(torch.tanh(y).cpu())
    return torch.cat(out).numpy()


def main(dataset, nets):
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "training", "data", dataset, "w*.csv")))
    cols = list(range(36)) + [36, 37]
    df = pd.concat([pd.read_csv(f, header=None, usecols=cols, dtype=np.int8) for f in files], ignore_index=True)
    boards = torch.from_numpy(df.iloc[:, :36].to_numpy()).to(device)
    y = df[36].to_numpy().astype(np.float32) * 0.75
    mv = df[37].to_numpy()
    print(f"{dataset}: {len(df):,} positions")
    rows = []
    for name in nets:
        p = predict(load_net(name), boards)
        for lo, hi in BUCKETS:
            m = (mv >= lo) & (mv <= hi)
            yb, pb = y[m], p[m]
            var = yb.var()
            r2 = 1 - ((pb - yb) ** 2).mean() / var if var > 0 else float("nan")
            dec = yb != 0
            acc = (np.sign(pb[dec]) == np.sign(yb[dec])).mean()
            rows.append(dict(net=name, phase=f"{lo}-{hi}" if hi < 999 else f"{lo}+", n=int(m.sum()),
                             r2=round(float(r2), 4), acc=round(float(acc), 4)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(REPO_ROOT, "results", f"phase_{dataset}.csv"), index=False)
    short = {n: n.replace("pair1024", "").replace("+gen1-10k-aws+gen1-10k-aws2", "+gen1(20M)")
             .replace("gen1-10k+gen1-10k-aws_", "gen1(5M)_")[:40] for n in nets}
    for metric in ("r2", "acc"):
        t = out.pivot(index="phase", columns="net", values=metric).reindex(
            [f"{lo}-{hi}" if hi < 999 else f"{lo}+" for lo, hi in BUCKETS])[nets].rename(columns=short)
        t.insert(0, "n", out[out.net == nets[0]].set_index("phase").n.reindex(t.index))
        print(f"\n{metric.upper()} by move number")
        print(t.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "gen2-10k-aws", args[1:] or DEFAULT_NETS)
