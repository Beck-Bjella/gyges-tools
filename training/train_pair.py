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

from common import REPO_ROOT, WEIGHTS_DIR, Dataset, append_result, device

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# With DATASET = "hce-10k", this exact config (h1024, dropout 0.5) produced
# pair1024_hce-10k_e9 (originally e9_pair_h1024_combined.bin), the network
# shipped in GygesRust v2.0.0. Dataset backup: V2.0.0_release_dataset/README.md.
#
# Runs are saved to training/weights/pair{H}_{DATASET}[_{TAG}]/e{N}.bin (see the
# NOTES.md registry). Set TAG when re-training the same arch on the same data
# with something changed, e.g. TAG = "drop0.3". No tag means the release recipe
# (DROPOUT 0.5 and the optimizer settings below).
DATASET     = "hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2"
TAG         = ""
HIDDEN_SIZE = 1024
DROPOUT     = 0.5    # 0.5 = release recipe; tag the run if changed
LR_DECAY    = 1.0    # lr multiplier after each epoch without a new best val (1.0 = off, release recipe)
SINGLETONS  = True   # encode the 108 single-piece features (release recipe). False = pairs only:
                     # their weights stay zero, so the saved net is still the 5886 layout the engine loads
TAIL        = 0      # 0 = HIDDEN -> 1 (release recipe). N = HIDDEN -> N -> ReLU -> 1 ("pair1024x32");
                     # needs an engine build that reads the tail
RUN_NAME    = (f"pair{HIDDEN_SIZE}" + (f"x{TAIL}" if TAIL else "") + f"_{DATASET}"
               + (f"_{TAG}" if TAG else ""))

# ── Pair encoding layout (5886 features total) ──────────────────────────────
#   [   0,  108)  singletons            36 sq × 3 piece types
#   [ 108,  738)  1-1 pairs (unordered) C(36,2) = 630
#   [ 738, 1368)  2-2 pairs (unordered) 630
#   [1368, 1998)  3-3 pairs (unordered) 630
#   [1998, 3294)  1-2 pairs (ordered)   36 × 36 = 1296
#   [3294, 4590)  1-3 pairs (ordered)   1296
#   [4590, 5886)  2-3 pairs (ordered)   1296
FEATURE_COUNT = 5886
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

    The +1 in num_embeddings is for the sink/padding index used by encode_batch
    to fill the slots of empty piece slots.
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
            if not SINGLETONS:
                # Never active, so never updated: zero rows = a pairs-only net.
                self.fc1.weight[:_PAIR_11_OFFSET].zero_()
        self.fc1_bias = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2     = nn.Linear(HIDDEN_SIZE, TAIL or 1)
        self.fc3     = nn.Linear(TAIL, 1) if TAIL else None

    def forward(self, indices):
        # indices: (B, MAX_ACTIVE) int64
        h = self.fc1(indices) + self.fc1_bias
        h = torch.relu(h)
        h = self.dropout(h)
        out = self.fc2(h)
        if self.fc3 is not None:
            out = self.fc3(torch.relu(out))
        return torch.tanh(out.squeeze(-1))


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


# encode_batch tables: unordered pair index, same-type and cross-type offsets.
_PAIR_IDX_FLAT = torch.from_numpy(_PAIR_IDX.reshape(-1).astype(np.int64)).to(device)
_SAME_OFF  = torch.tensor([0, _PAIR_11_OFFSET, _PAIR_22_OFFSET, _PAIR_33_OFFSET], device=device)
_CROSS_OFF = torch.zeros(4, 4, dtype=torch.int64, device=device)
for (_lo, _hi, _off) in ((1, 2, _PAIR_12_OFFSET), (1, 3, _PAIR_13_OFFSET), (2, 3, _PAIR_23_OFFSET)):
    _CROSS_OFF[_lo, _hi] = _CROSS_OFF[_hi, _lo] = _off
_SQ = torch.arange(36, device=device)


def encode_batch(boards, max_pieces):
    """GPU version of encode_position for a (B, 36) int8 board tensor.

    Returns (B, P + P(P-1)/2) int64 indices, P = max_pieces: one singleton per
    piece slot and one feature per slot pair, FEATURE_COUNT where a slot is
    empty. Same active set as encode_position (order differs; the sum doesn't
    care), so training on the fly matches pre-encoding without storing ~200
    bytes a position.
    """
    b = boards.long()
    # Occupied squares first, ascending; empty squares sorted after them.
    keys = torch.where(b != 0, _SQ, _SQ + 36)
    keys = torch.sort(keys, dim=1).values[:, :max_pieces]
    valid = keys < 36
    sq = keys % 36
    t = torch.where(valid, b.gather(1, sq), 0)

    single = torch.where(valid & SINGLETONS, sq * 3 + t - 1, FEATURE_COUNT)

    i, j = torch.triu_indices(max_pieces, max_pieces, offset=1, device=device)
    si, sj, ti, tj = sq[:, i], sq[:, j], t[:, i], t[:, j]   # si < sj where valid
    same = _SAME_OFF[ti] + _PAIR_IDX_FLAT[si * 36 + sj]
    lo_first = ti < tj                                      # lower piece type first
    cross = (_CROSS_OFF[ti, tj]
             + torch.where(lo_first, si, sj) * 36
             + torch.where(lo_first, sj, si))
    pair = torch.where(ti == tj, same, cross)
    pair = torch.where(valid[:, i] & valid[:, j], pair, FEATURE_COUNT)

    return torch.cat([single, pair], dim=1)


def check_encode_batch(boards, max_pieces):
    """Fail fast if encode_batch and encode_position disagree on any board."""
    got = encode_batch(torch.from_numpy(boards).to(device), max_pieces).cpu().numpy()
    for k in range(len(boards)):
        want = sorted(x for x in encode_position(boards[k]) if SINGLETONS or x >= _PAIR_11_OFFSET)
        have = sorted(int(x) for x in got[k] if x != FEATURE_COUNT)
        if want != have:
            raise SystemExit(f"encode_batch mismatch on board {boards[k].tolist()}")


def save_weights_pair(model, path):
    """Save weights in the engine's expected layout.

    Engine expects fc1 in Linear's `(HIDDEN, FEATURE_COUNT)` shape, but
    EmbeddingBag stores it as `(FEATURE_COUNT+1, HIDDEN)`. Transpose at save
    time and drop the padding row so the .bin format matches what engine reads.

    Layout: fc1.weight [HIDDEN, FEATURE_COUNT] + fc1_bias [HIDDEN] +
            fc2.weight [1, HIDDEN]              + fc2.bias [1]
    With a tail: fc2.weight [TAIL, HIDDEN] + fc2.bias [TAIL] + fc3.weight [1, TAIL] + fc3.bias [1].
    """
    with open(path, "wb") as f:
        # fc1: drop padding row, transpose to (HIDDEN, FEATURE_COUNT)
        fc1_w = model.fc1.weight.data[:FEATURE_COUNT].t().contiguous()
        f.write(fc1_w.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc1_bias.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc2.weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc2.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        if model.fc3 is not None:
            f.write(model.fc3.weight.detach().cpu().numpy().astype(np.float32).tobytes())
            f.write(model.fc3.bias.detach().cpu().numpy().astype(np.float32).tobytes())


if __name__ == '__main__':
    # The release data is read straight from the frozen archive. hce-10k loads
    # part1 then part2: the same rows in the same order as the old
    # hce_10kn_combined.csv, so the seeded split matches e9.
    RELEASE_DIR = os.path.join(REPO_ROOT, "V2.0.0_release_dataset")
    RELEASE_FILES = {
        "hce-10k-part1": ["hce_10kn.csv"],
        "hce-10k-part2": ["hce_10kn_big.csv"],
        "hce-10k":       ["hce_10kn.csv", "hce_10kn_big.csv"],
    }

    # Everything else is data gen output: one shard per worker in
    # training/data/<dataset>/. A mix ("hce-10k+gen1-10k") loads every part in order.
    # "<dataset>@N" takes only its first N shards (in the same sorted-name order used
    # for whole datasets, so earlier runs keep their row order), for size-matched subsets.
    files = []
    for part in DATASET.split("+"):
        if part in RELEASE_FILES:
            files += [os.path.join(RELEASE_DIR, f) for f in RELEASE_FILES[part]]
            continue
        name, _, n_shards = part.partition("@")
        shard_dir = os.path.join(REPO_ROOT, "training", "data", name)
        part_files = sorted(glob.glob(os.path.join(shard_dir, "w*.csv")))
        if not part_files:
            raise SystemExit(f"no shards found in {shard_dir}")
        if n_shards:
            part_files = part_files[:int(n_shards)]
        files += part_files

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

    n_rows = len(ds)
    train, val = ds.split(test_size=0.2)
    del ds
    train = train.mirror_augment()
    train.stats("Train (after mirroring)")
    val.stats("Val (no mirror)")
    print()

    # Boards go to the GPU as int8 (36 bytes a position, ~2.6 GB for a 40M-row
    # mix) and are encoded a batch at a time. Pre-encoding all of them took
    # ~200 bytes a position and didn't fit in 8 GB past ~20M rows.
    max_pieces = int(max((train.boards != 0).sum(1).max(), (val.boards != 0).sum(1).max()))
    check_encode_batch(np.concatenate([train.boards[:1000], train.boards[-1000:], val.boards[:1000]]),
                       max_pieces)
    print(f"encode_batch matches encode_position (max {max_pieces} pieces a board)")

    X_train_gpu = torch.from_numpy(train.boards).to(device)   # int8 boards
    y_train_gpu = torch.from_numpy(train.outcomes).to(device)
    X_val_gpu   = torch.from_numpy(val.boards).to(device)
    y_val_gpu   = torch.from_numpy(val.outcomes).to(device)
    del train, val

    LR = 1e-3
    model     = GygesNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    epochs = 100
    patience = 3     # stop after this many epochs without a new best val
    batch_size = 4096
    n_train = X_train_gpu.shape[0]
    n_val   = X_val_gpu.shape[0]

    run_dir = os.path.join(WEIGHTS_DIR, RUN_NAME)
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} already exists — set a TAG for this run")
    os.makedirs(run_dir)
    started = f"{datetime.now():%Y-%m-%d %H:%M}"
    print(f"Run: {run_dir}  (started {started})")
    config = (f"dataset {DATASET}\nhidden {HIDDEN_SIZE}\ndropout {DROPOUT}\nlr {LR}\n"
              f"lr_decay {LR_DECAY}\nbatch {batch_size}\npatience {patience}\nsingletons {SINGLETONS}\ntail {TAIL}\n")
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(config)
    print("Config: " + ", ".join(config.splitlines()))
    append_result("runs", {
        "run": RUN_NAME, "started": started, "dataset": DATASET, "rows": n_rows,
        "train_rows": n_train, "val_rows": n_val, "hidden": HIDDEN_SIZE, "dropout": DROPOUT,
        "lr": LR, "lr_decay": LR_DECAY, "batch": batch_size, "patience": patience,
        "min_game_moves": MIN_GAME_MOVES, "min_move_num": MIN_MOVE_NUM,
        "mirror": 1, "split": "by position, seed 42, 20% val",
        "note": "; ".join(x for x in ("" if SINGLETONS else "pairs only (no single-piece features)",
                                     f"tail {HIDDEN_SIZE}->{TAIL}->1" if TAIL else "") if x),
    })

    best_val, since_best = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = torch.zeros(1, device=device)
        n_train_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            X_idx = encode_batch(X_train_gpu[idx], max_pieces)
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
                    X_idx = encode_batch(X_val_gpu[i:end], max_pieces)
                    y     = y_val_gpu[i:end]
                    val_loss += loss_fn(model(X_idx), y)
                    n_val_batches += 1

            avg_val   = (val_loss   / n_val_batches).item()
            avg_train = (train_loss / n_train_batches).item()

            line = (f"Epoch {epoch + 1:>3}/{epochs} | "
                    f"train: {avg_train:.4f} | "
                    f"val: {avg_val:.4f}")
            print(line)
            with open(os.path.join(run_dir, "log.txt"), "a") as f:
                f.write(line + "\n")
            append_result("epochs", {
                "run": RUN_NAME, "epoch": epoch + 1, "train": round(avg_train, 6),
                "val": round(avg_val, 6), "lr": optimizer.param_groups[0]["lr"],
            })

            out_path = os.path.join(run_dir, f"e{epoch + 1}.bin")
            save_weights_pair(model, out_path)
            print(f"  Saved {out_path}")

            if avg_val < best_val:
                best_val, since_best = avg_val, 0
            else:
                since_best += 1
                if since_best >= patience:
                    print(f"No new best val for {patience} epochs — stopping")
                    break
                if LR_DECAY != 1.0:
                    for group in optimizer.param_groups:
                        group["lr"] *= LR_DECAY
                    print(f"  lr -> {optimizer.param_groups[0]['lr']:.2e}")
