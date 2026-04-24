"""NNUE training for Gyges with the flat (144-feature) baseline encoding.

Features: 144 = 36 squares * 4 piece states (empty, type1, type2, type3).
Dense input MLP (144 -> 256 -> 1), trained to predict game outcome.

This is the pre-pair-encoding baseline kept for comparison.
"""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from common import REPO_ROOT, WEIGHTS_DIR, Dataset, device, save_weights

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

HIDDEN_SIZE = 256
OUTPUT_PREFIX = f"nnue_h{HIDDEN_SIZE}"
FEATURE_COUNT = 144


class GygesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_COUNT, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_board(board_36):
    """144 one-hot features = 36 squares * 4 piece states."""
    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    for sq in range(36):
        piece_type = int(board_36[sq])
        features[sq * 4 + piece_type] = 1.0
    return features


if __name__ == '__main__':
    files = (
        [os.path.join(REPO_ROOT, "training", "data", "hce_100kn.csv")] +
        [os.path.join(REPO_ROOT, f"training_data_{i}.csv") for i in range(4)]
    )
    
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

    print("Encoding positions...")
    X_train = np.stack([encode_board(train.boards[i]) for i in range(len(train))])
    X_val   = np.stack([encode_board(val.boards[i])   for i in range(len(val))])
    y_train = train.outcomes
    y_val   = val.outcomes
    print(f"Train features: {X_train.shape}  Val features: {X_val.shape}")

    X_train_gpu = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_gpu = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_gpu   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_val_gpu   = torch.tensor(y_val,   dtype=torch.float32, device=device)

    model     = GygesNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    epochs = 100
    batch_size = 1024
    n_train = X_train_gpu.shape[0]
    n_val   = X_val_gpu.shape[0]

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
            X, y = X_train_gpu[idx], y_train_gpu[idx]
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
                    X = X_val_gpu[i:i + batch_size]
                    y = y_val_gpu[i:i + batch_size]
                    val_loss += loss_fn(model(X), y)
                    n_val_batches += 1

            avg_val   = (val_loss   / n_val_batches).item()
            avg_train = (train_loss / n_train_batches).item()

            print(f"Epoch {epoch + 1:>3}/{epochs} | "
                  f"train: {avg_train:.4f} | "
                  f"val: {avg_val:.4f}")

            out_path = os.path.join(run_dir, f"e{epoch + 1}.bin")
            save_weights(model, out_path)
            print(f"  Saved {out_path}")
