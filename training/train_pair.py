"""NNUE training for Gyges with pair-factored feature encoding.

Features: 5886 total.
  - 108 singletons (36 squares * 3 piece types)
  - 1890 same-type pairs (3 types * C(36,2) = 3 * 630)
  - 3888 cross-type ordered pairs (3 type-combos * 36 * 36)

Network is a sparse-input MLP (FEATURE_COUNT -> HIDDEN -> 1) trained to predict
game outcome (Tanh output in [-1, 1]).
"""
import glob
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from common import REPO_ROOT, WEIGHTS_DIR, Dataset, device

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# With DATASET = "hce-10k", this exact config (h1024, dropout 0.5) produced
# pair1024_hce-10k_e9 (originally e9_pair_h1024_combined.bin), the network
# shipped in GygesRust v2.0.0. Dataset backup: V2.0.0_release_dataset/README.md.
#
# Runs are saved to training/weights/pair{H}_{DATASET}[_{TAG}]/e{N}.bin (see the
# NOTES.md registry). Set TAG when re-training the same arch on the same data
# with something changed, e.g. TAG = "drop0.3".
DATASET     = "hce-10k"
TAG         = ""
HIDDEN_SIZE = 1024
RUN_NAME    = f"pair{HIDDEN_SIZE}_{DATASET}" + (f"_{TAG}" if TAG else "")

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
    """Pair-encoding NNUE.

    Uses EmbeddingBag for the first layer instead of Linear over a dense scatter.
    Mathematically equivalent (sum of weight rows for active features) but ~10x
    faster — no dense (B, FEATURE_COUNT) tensor, no zero-multiplications.

    The +1 in num_embeddings is for the sink/padding index used by encode_all
    to fill MAX_ACTIVE slots when fewer features fire.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.EmbeddingBag(
            FEATURE_COUNT + 1, HIDDEN_SIZE,
            mode='sum', padding_idx=FEATURE_COUNT,
        )
        # PyTorch's default EmbeddingBag init is N(0,1), which is huge: summing
        # ~78 active features per sample blows pre-activation std to ~8.8 and
        # saturates the downstream Tanh, killing gradients. Re-init to match
        # nn.Linear's default (uniform within ±1/sqrt(fan_in)) so training behaves
        # like the dense scatter+Linear version we replaced.
        bound = 1.0 / (FEATURE_COUNT ** 0.5)
        nn.init.uniform_(self.fc1.weight, -bound, bound)
        # Keep the padding row exactly zero so it contributes nothing in the sum.
        with torch.no_grad():
            self.fc1.weight[FEATURE_COUNT].zero_()
        self.fc1_bias = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, indices):
        # indices: (B, MAX_ACTIVE) int64
        h = self.fc1(indices) + self.fc1_bias
        h = torch.relu(h)
        h = self.dropout(h)
        out = self.fc2(h).squeeze(-1)
        return torch.tanh(out)


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


def save_weights_pair(model, path):
    """Save weights in the engine's expected layout.

    Engine expects fc1 in Linear's `(HIDDEN, FEATURE_COUNT)` shape, but
    EmbeddingBag stores it as `(FEATURE_COUNT+1, HIDDEN)`. Transpose at save
    time and drop the padding row so the .bin format matches what engine reads.

    Layout: fc1.weight [HIDDEN, FEATURE_COUNT] + fc1_bias [HIDDEN] +
            fc2.weight [1, HIDDEN]              + fc2.bias [1]
    """
    with open(path, "wb") as f:
        # fc1: drop padding row, transpose to (HIDDEN, FEATURE_COUNT)
        fc1_w = model.fc1.weight.data[:FEATURE_COUNT].t().contiguous()
        f.write(fc1_w.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc1_bias.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc2.weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc2.bias.detach().cpu().numpy().astype(np.float32).tobytes())


if __name__ == '__main__':
    if DATASET == "hce-10k":
        # The release data: hce-10k-part1 + hce-10k-part2, read straight from the
        # frozen archive. Loading them in this order gives the same rows in the
        # same order as the old hce_10kn_combined.csv, so the seeded split matches e9.
        RELEASE_DIR = os.path.join(REPO_ROOT, "V2.0.0_release_dataset")
        files = [
            os.path.join(RELEASE_DIR, "hce_10kn.csv"),
            os.path.join(RELEASE_DIR, "hce_10kn_big.csv"),
        ]
    else:
        # Data gen output: one shard per worker in training/data/<DATASET>/.
        shard_dir = os.path.join(REPO_ROOT, "training", "data", DATASET)
        files = sorted(glob.glob(os.path.join(shard_dir, "w*.csv")))
        if not files:
            raise SystemExit(f"no shards found in {shard_dir}")

    # Optional pruning knobs — leave at 0 to disable
    MIN_GAME_MOVES = 0   # drop games shorter than N positions (random-opening junk)
    MIN_MOVE_NUM   = 0   # drop positions earlier than this within their game

    ds = Dataset.from_files(files)
    ds.stats("Raw")
    if MIN_GAME_MOVES > 0:
        ds = ds.filter_min_moves(MIN_GAME_MOVES)
        ds.stats(f"After filter_min_moves({MIN_GAME_MOVES})")
    if MIN_MOVE_NUM > 0:
        ds = ds.filter_min_move_num(MIN_MOVE_NUM)
        ds.stats(f"After filter_min_move_num({MIN_MOVE_NUM})")

    train, val = ds.split(test_size=0.2)
    train = train.mirror_augment()
    train.stats("Train (after mirroring)")
    val.stats("Val (no mirror)")
    print()

    # Sparse encoding: each position becomes a list of active feature indices.
    # Pad to MAX_ACTIVE with FEATURE_COUNT (the "sink" index) so scatter is safe.
    # int16 is enough (FEATURE_COUNT=5886 << 32k); 4× memory savings vs int64.
    def encode_all(boards):
        out = np.full((len(boards), MAX_ACTIVE), FEATURE_COUNT, dtype=np.int16)
        for i in range(len(boards)):
            active = encode_position(boards[i])
            out[i, :len(active)] = active
        return out

    print("Encoding positions...")
    X_train = encode_all(train.boards)
    X_val   = encode_all(val.boards)
    y_train = train.outcomes
    y_val   = val.outcomes
    print(f"Train indices: {X_train.shape}  Val indices: {X_val.shape}")

    # Push encoded indices to GPU as int16 (~6GB for 30M positions vs 24GB int64).
    # EmbeddingBag needs int64 indices, so we convert per-batch from the int16 slice.
    X_train_gpu = torch.from_numpy(X_train).to(device)   # int16
    y_train_gpu = torch.from_numpy(y_train).to(device)
    X_val_gpu   = torch.from_numpy(X_val).to(device)
    y_val_gpu   = torch.from_numpy(y_val).to(device)

    model     = GygesNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    epochs = 100
    batch_size = 4096
    n_train = X_train_gpu.shape[0]
    n_val   = X_val_gpu.shape[0]

    run_dir = os.path.join(WEIGHTS_DIR, RUN_NAME)
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} already exists — set a TAG for this run")
    os.makedirs(run_dir)
    print(f"Run: {run_dir}  (started {datetime.now():%Y-%m-%d %H:%M})")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = torch.zeros(1, device=device)
        n_train_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            X_idx = X_train_gpu[idx].long()  # int16 slice → int64 for EmbeddingBag
            y     = y_train_gpu[idx]
            pred = model(X_idx)
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
            n_train_batches += 1

        if (epoch + 1) % 1 == 0:
            model.eval()
            val_loss = torch.zeros(1, device=device)
            n_val_batches = 0
            with torch.no_grad():
                for i in range(0, n_val, batch_size):
                    end = min(i + batch_size, n_val)
                    X_idx = X_val_gpu[i:end].long()
                    y     = y_val_gpu[i:end]
                    val_loss += loss_fn(model(X_idx), y)
                    n_val_batches += 1

            avg_val   = (val_loss   / n_val_batches).item()
            avg_train = (train_loss / n_train_batches).item()

            print(f"Epoch {epoch + 1:>3}/{epochs} | "
                  f"train: {avg_train:.4f} | "
                  f"val: {avg_val:.4f}")

            out_path = os.path.join(run_dir, f"e{epoch + 1}.bin")
            save_weights_pair(model, out_path)
            print(f"  Saved {out_path}")
