import os
import glob
import json
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(SCRIPT_DIR, "policy_data")
WEIGHTS_DIR   = os.path.join(SCRIPT_DIR, "policy_weights")
OUTPUT_PREFIX = "policy_weights"

# Rank-based label weighting. All moves contribute, but weighted by geometric
# decay 0.85^i so low-ranked moves are nearly zero. Ignores raw scores — robust
# to mate outliers and wild per-position score ranges.
RANK_DECAY = 0.85
# Skip positions whose top score is a mate — search finds those trivially,
# and they pollute training with one-hot labels.
MATE_THRESHOLD = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


class PolicyNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(144, hidden),
            nn.ReLU(),
        )
        self.source_head = nn.Linear(hidden, 36)
        self.pickup_head = nn.Linear(hidden, 36)
        self.dest_head   = nn.Linear(hidden, 36)

    def forward(self, x):
        h = self.body(x)
        return self.source_head(h), self.pickup_head(h), self.dest_head(h)


class PolicyDataset(Dataset):
    def __init__(self, features, src_t, pickup_t, dest_t, has_pickup):
        self.features   = torch.tensor(features,   dtype=torch.float32).to(device)
        self.src_t      = torch.tensor(src_t,      dtype=torch.float32).to(device)
        self.pickup_t   = torch.tensor(pickup_t,   dtype=torch.float32).to(device)
        self.dest_t     = torch.tensor(dest_t,     dtype=torch.float32).to(device)
        self.has_pickup = torch.tensor(has_pickup, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.src_t[idx], self.pickup_t[idx],
                self.dest_t[idx], self.has_pickup[idx])


# Move status codes from the engine's policy CSV.
STATUS_UNSEARCHED = 0  # legal but no score signal — exclude from labels
STATUS_FAIL_LOW   = 1  # upper bound only — exclude from rank labels
STATUS_RELIABLE   = 2  # exact / fail-high lower bound — use for labels


def flip_sq(sq):
    return 35 - sq


def flip_pieces(pieces):
    return [pieces[35 - i] for i in range(36)]


def encode_board(pieces):
    # 36 piece ints → 144-feature one-hot (4 piece types per square)
    feats = np.zeros(144, dtype=np.float32)
    for sq in range(36):
        feats[sq * 4 + int(pieces[sq])] = 1.0
    return feats


def moves_to_heatmaps(moves, decay=RANK_DECAY):
    # moves: list of [src, pickup_or_null, dest, score, status], sorted by score descending
    # Only status==2 moves contribute to the rank-decay label.
    reliable = [m for m in moves if m[4] == STATUS_RELIABLE]

    seen = set()
    unique = []
    for m in reliable:
        key = (m[0], m[1], m[2])
        if key not in seen:
            seen.add(key)
            unique.append(m)

    P_src    = np.zeros(36, dtype=np.float32)
    P_pickup = np.zeros(36, dtype=np.float32)
    P_dest   = np.zeros(36, dtype=np.float32)
    pickup_mass = 0.0

    if not unique:
        return P_src, P_pickup, P_dest, 0.0

    weights = np.array([decay ** i for i in range(len(unique))], dtype=np.float64)
    weights /= weights.sum()

    for m, w in zip(unique, weights):
        src, pickup, dest = m[0], m[1], m[2]
        P_src[src]   += w
        P_dest[dest] += w
        if pickup is not None:
            P_pickup[pickup] += w
            pickup_mass      += w

    if pickup_mass > 1e-8:
        P_pickup /= pickup_mass

    return P_src, P_pickup, P_dest, float(pickup_mass)


def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "policy_w*.csv")))
    if not files:
        raise FileNotFoundError(f"No policy_w*.csv files found in {data_dir}")
    print(f"Loading {len(files)} files from {data_dir}")

    dfs = [pd.read_csv(f, header=None, names=["pieces", "player", "moves"],
                        engine="python", on_bad_lines="skip") for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  total rows: {len(df)}")

    feats_l, src_l, pickup_l, dest_l, has_pickup_l = [], [], [], [], []
    skipped_empty    = 0
    skipped_mate     = 0
    skipped_no_label = 0

    for _, row in df.iterrows():
        pieces = [int(x) for x in str(row["pieces"]).split(",")]
        player = int(row["player"])
        try:
            moves = json.loads(row["moves"])
        except Exception:
            skipped_empty += 1
            continue
        if len(moves) == 0:
            skipped_empty += 1
            continue

        # Orient board + moves to side-to-move's perspective (matches inference flip).
        if player == 2:
            pieces = flip_pieces(pieces)
            moves = [
                [flip_sq(m[0]),
                 None if m[1] is None else flip_sq(m[1]),
                 flip_sq(m[2]),
                 m[3], m[4]]
                for m in moves
            ]

        # Filter mates by checking the top reliable score.
        reliable_scores = [m[3] for m in moves if m[4] == STATUS_RELIABLE and m[3] is not None]
        if not reliable_scores:
            skipped_no_label += 1
            continue
        if abs(max(reliable_scores, key=abs)) > MATE_THRESHOLD:
            skipped_mate += 1
            continue

        feat = encode_board(pieces)
        P_src, P_pickup, P_dest, pickup_mass = moves_to_heatmaps(moves)

        feats_l.append(feat)
        src_l.append(P_src)
        pickup_l.append(P_pickup)
        dest_l.append(P_dest)
        has_pickup_l.append(1.0 if pickup_mass > 1e-8 else 0.0)

    if skipped_empty:
        print(f"  skipped {skipped_empty} rows with empty move list")
    if skipped_no_label:
        print(f"  skipped {skipped_no_label} rows with no status-2 (reliable) moves")
    if skipped_mate:
        print(f"  skipped {skipped_mate} rows with mate scores")

    return (np.asarray(feats_l, dtype=np.float32),
            np.asarray(src_l,   dtype=np.float32),
            np.asarray(pickup_l, dtype=np.float32),
            np.asarray(dest_l,  dtype=np.float32),
            np.asarray(has_pickup_l, dtype=np.float32))


def cross_entropy_per_sample(logits, target_probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1)


def save_weights(model, path):
    with open(path, "wb") as f:
        for _, param in model.named_parameters():
            f.write(param.detach().cpu().numpy().astype(np.float32).tobytes())


if __name__ == '__main__':
    feats, src_t, pickup_t, dest_t, has_pickup = load_data(DATA_DIR)

    print(f"Feature matrix shape: {feats.shape}")
    print(f"Positions with pickup-drop moves: {has_pickup.sum():.0f} / {len(has_pickup)} "
          f"({100 * has_pickup.mean():.1f}%)")

    idx = np.arange(len(feats))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_ds = PolicyDataset(feats[train_idx], src_t[train_idx], pickup_t[train_idx],
                             dest_t[train_idx], has_pickup[train_idx])
    val_ds   = PolicyDataset(feats[val_idx],   src_t[val_idx],   pickup_t[val_idx],
                             dest_t[val_idx],   has_pickup[val_idx])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1024)

    model     = PolicyNet(hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200

    run_dir = os.path.join(WEIGHTS_DIR, f"{OUTPUT_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_dir}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, sT, pT, dT, hP in train_loader:
            src_logits, pickup_logits, dest_logits = model(X)
            loss_src  = cross_entropy_per_sample(src_logits,  sT).mean()
            loss_dest = cross_entropy_per_sample(dest_logits, dT).mean()
            pickup_ce = cross_entropy_per_sample(pickup_logits, pT)
            loss_pickup = (pickup_ce * hP).sum() / (hP.sum() + 1e-8)
            loss = loss_src + loss_dest + loss_pickup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, sT, pT, dT, hP in val_loader:
                src_logits, pickup_logits, dest_logits = model(X)
                loss_src  = cross_entropy_per_sample(src_logits,  sT).mean()
                loss_dest = cross_entropy_per_sample(dest_logits, dT).mean()
                pickup_ce = cross_entropy_per_sample(pickup_logits, pT)
                loss_pickup = (pickup_ce * hP).sum() / (hP.sum() + 1e-8)
                val_loss += (loss_src + loss_dest + loss_pickup).item()

        avg_val = val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:>3}/{epochs} | "
                  f"train: {train_loss / len(train_loader):.4f} | "
                  f"val: {avg_val:.4f}")

            out_path = os.path.join(run_dir, f"e{epoch + 1}.bin")
            save_weights(model, out_path)
            print(f"  Saved {out_path}")
