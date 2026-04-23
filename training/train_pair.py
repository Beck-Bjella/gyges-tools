"""NNUE training for Gyges with pair-factored feature encoding.

Features: 5886 total.
  - 108 singletons (36 squares * 3 piece types)
  - 1890 same-type pairs (3 types * C(36,2) = 3 * 630)
  - 3888 cross-type ordered pairs (3 type-combos * 36 * 36)

Network is a sparse-input MLP (FEATURE_COUNT -> HIDDEN -> 1) trained to predict
game outcome (Tanh output in [-1, 1]).
"""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from common import REPO_ROOT, WEIGHTS_DIR, device, load_and_split, save_weights

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

HIDDEN_SIZE = 64
OUTPUT_PREFIX = f"nnue_h{HIDDEN_SIZE}_pairs"

# ── Pair encoding layout (5886 features total) ──────────────────────────────
#   [   0,  108)  singletons            36 sq × 3 piece types
#   [ 108,  738)  1-1 pairs (unordered) C(36,2) = 630
#   [ 738, 1368)  2-2 pairs (unordered) 630
#   [1368, 1998)  3-3 pairs (unordered) 630
#   [1998, 3294)  1-2 pairs (ordered)   36 × 36 = 1296
#   [3294, 4590)  1-3 pairs (ordered)   1296
#   [4590, 5886)  2-3 pairs (ordered)   1296
FEATURE_COUNT = 5886
MAX_ACTIVE    = 100    # safe upper bound on bits fired per position (actual ~78)
_PAIR_11_OFFSET = 108
_PAIR_22_OFFSET = 738
_PAIR_33_OFFSET = 1368
_PAIR_12_OFFSET = 1998
_PAIR_13_OFFSET = 3294
_PAIR_23_OFFSET = 4590


def _build_pair_idx():
    # Precomputed unordered-pair index: table[a,b] == table[b,a] in [0, 630)
    table = np.full((36, 36), -1, dtype=np.int32)
    k = 0
    for a in range(36):
        for b in range(a + 1, 36):
            table[a, b] = k
            table[b, a] = k
            k += 1
    return table

_PAIR_IDX = _build_pair_idx()


class GygesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_COUNT, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_position(board_36):
    """Return the active feature indices for a position as a list of ints."""
    indices = []
    sqs = ([], [], [])   # sqs[0]=ones, sqs[1]=twos, sqs[2]=threes
    for sq in range(36):
        t = int(board_36[sq])
        if t == 0:
            continue
        sqs[t - 1].append(sq)
        indices.append(sq * 3 + (t - 1))   # singleton

    # Same-type pairs (unordered)
    for t_idx, off in ((0, _PAIR_11_OFFSET), (1, _PAIR_22_OFFSET), (2, _PAIR_33_OFFSET)):
        g = sqs[t_idx]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                indices.append(off + int(_PAIR_IDX[g[i], g[j]]))

    # Cross-type pairs (ordered by lower piece-type first)
    for (t_lo, t_hi, off) in ((0, 1, _PAIR_12_OFFSET), (0, 2, _PAIR_13_OFFSET), (1, 2, _PAIR_23_OFFSET)):
        for a in sqs[t_lo]:
            for b in sqs[t_hi]:
                indices.append(off + a * 36 + b)

    return indices


def build_batch(indices_gpu, idx, out):
    """Scatter sparse indices for batch `idx` into preallocated dense `out` [B, FEATURE_COUNT+1]."""
    out.zero_()
    batch_idx = indices_gpu[idx]
    out.scatter_(1, batch_idx, 1.0)
    return out[:, :FEATURE_COUNT]


if __name__ == '__main__':
    files = [os.path.join(REPO_ROOT, f"training_data_{i}.csv") for i in range(4)]
    train_data, val_data, y_train, y_val = load_and_split(files)

    # Sparse encoding: each position becomes a list of active feature indices.
    # Pad to MAX_ACTIVE with FEATURE_COUNT (the "sink" index) so scatter is safe.
    def encode_all(boards):
        out = np.full((len(boards), MAX_ACTIVE), FEATURE_COUNT, dtype=np.int64)
        for i in range(len(boards)):
            active = encode_position(boards[i])
            out[i, :len(active)] = active
        return out

    print("Encoding positions...")
    X_train = encode_all(train_data)
    X_val   = encode_all(val_data)
    print(f"Train indices: {X_train.shape}  Val indices: {X_val.shape}")

    X_train_gpu = torch.tensor(X_train, dtype=torch.long,    device=device)
    y_train_gpu = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_gpu   = torch.tensor(X_val,   dtype=torch.long,    device=device)
    y_val_gpu   = torch.tensor(y_val,   dtype=torch.float32, device=device)

    model     = GygesNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    epochs = 100
    batch_size = 1024
    n_train = X_train_gpu.shape[0]
    n_val   = X_val_gpu.shape[0]

    # Preallocated scatter buffer reused across batches (saves allocation per step).
    scatter_buf = torch.empty(batch_size, FEATURE_COUNT + 1, device=device)

    run_dir = os.path.join(WEIGHTS_DIR, f"{OUTPUT_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_dir}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = torch.zeros(1, device=device)
        n_train_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            buf = scatter_buf if idx.shape[0] == batch_size else torch.empty(idx.shape[0], FEATURE_COUNT + 1, device=device)
            X = build_batch(X_train_gpu, idx, buf)
            y = y_train_gpu[idx]
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
            n_train_batches += 1

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = torch.zeros(1, device=device)
            n_val_batches = 0
            with torch.no_grad():
                for i in range(0, n_val, batch_size):
                    end = min(i + batch_size, n_val)
                    vidx = torch.arange(i, end, device=device)
                    buf = scatter_buf if vidx.shape[0] == batch_size else torch.empty(vidx.shape[0], FEATURE_COUNT + 1, device=device)
                    X = build_batch(X_val_gpu, vidx, buf)
                    val_loss += loss_fn(model(X), y_val_gpu[i:end])
                    n_val_batches += 1

            avg_val   = (val_loss   / n_val_batches).item()
            avg_train = (train_loss / n_train_batches).item()

            print(f"Epoch {epoch + 1:>3}/{epochs} | "
                  f"train: {avg_train:.4f} | "
                  f"val: {avg_val:.4f}")

            out_path = os.path.join(run_dir, f"e{epoch + 1}.bin")
            save_weights(model, out_path)
            print(f"  Saved {out_path}")
