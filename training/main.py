import os
import glob
import copy

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
OUTPUT_NAME = "final_weights.bin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


class GygesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(180, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class GygesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def encode_board(data):
    # data: 72 values [piece0, ctrl0, piece1, ctrl1, ...]
    # Output: 180 features — 5 per square (4 one-hot piece type + 1 scalar control)
    features = np.zeros(180, dtype=np.float32)
    for sq in range(36):
        piece_type = int(data[sq * 2])
        control = float(data[sq * 2 + 1])
        features[sq * 5 + piece_type] = 1.0
        features[sq * 5 + 4] = control
    return features


def mirror_board(data):
    # data: 72 values [piece0, ctrl0, ..., piece35, ctrl35]
    # Flip left-right within each row, keeping piece+control pairs together
    mirrored = data.copy()
    for row in range(6):
        for col in range(3):
            left  = (row * 6 + col) * 2
            right = (row * 6 + (5 - col)) * 2
            mirrored[left],   mirrored[right]   = data[right],   data[left]
            mirrored[left+1], mirrored[right+1] = data[right+1], data[left+1]
    return mirrored


if __name__ == '__main__':
    files = [os.path.join(DATA_DIR, f"converted_{i}.csv") for i in range(24)]

    df = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} files")

    data     = df.iloc[:, :72].values   # 36 squares × 2 (piece + control)
    # data[:, 1::2] = 0  # zero every control column (for no-control experiments)

    outcomes = df.iloc[:, 72].values.astype(np.float32)
    outcomes = outcomes * 0.75

    # ── Dataset diagnostics ──────────────────────────────────────────────
    def print_stats(tag, data_arr, outcomes_arr):
        n = len(outcomes_arr)
        wins   = int((outcomes_arr > 0).sum())
        losses = int((outcomes_arr < 0).sum())
        unique = len(np.unique(data_arr, axis=0))
        dupes  = n - unique

        print(f"\n{'─' * 50}")
        print(f"  {tag}")
        print(f"{'─' * 50}")
        print(f"  Total positions:  {n:,}")
        print(f"  Unique boards:    {unique:,} ({100 * unique / n:.1f}%)")
        print(f"  Duplicates:       {dupes:,} ({100 * dupes / n:.1f}%)")
        print(f"  Wins: {wins:,} ({100 * wins / n:.1f}%)  |  Losses: {losses:,} ({100 * losses / n:.1f}%)")

    print_stats("Raw dataset", data, outcomes)

    # Mirror augmentation
    mirrored_data = np.array([mirror_board(data[i]) for i in range(len(data))])
    data     = np.vstack([data, mirrored_data])
    outcomes = np.concatenate([outcomes, outcomes])

    print_stats("After mirroring", data, outcomes)
    print()

    print("Encoding positions...")
    features = np.array([encode_board(data[i]) for i in range(len(data))])
    print(f"Feature matrix shape: {features.shape}")  # should be (N, 180)

    X_train, X_val, y_train, y_val = train_test_split(
        features, outcomes, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(GygesDataset(X_train, y_train), batch_size=1024, shuffle=True)
    val_loader   = DataLoader(GygesDataset(X_val,   y_val),   batch_size=1024)

    model      = GygesNet().to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn    = nn.MSELoss()

    epochs = 50
    best_val = float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += loss_fn(model(X), y).item()

        avg_val = val_loss / len(val_loader)
        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:>3}/{epochs} | "
                  f"train: {train_loss / len(train_loader):.4f} | "
                  f"val: {avg_val:.4f}")

    print(f"\nBest val loss {best_val:.4f} at epoch {best_epoch}")
    model.load_state_dict(best_state)
    model.eval()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    out_path = os.path.join(WEIGHTS_DIR, OUTPUT_NAME)
    with open(out_path, "wb") as f:
        for name, param in model.named_parameters():
            data_out = param.detach().cpu().numpy().astype(np.float32)
            f.write(data_out.tobytes())
            print(f"{name}: shape={data_out.shape}")

    print(f"Saved {out_path}")
