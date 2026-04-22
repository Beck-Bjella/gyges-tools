import os
from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
OUTPUT_PREFIX = f"nnue_h{256}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


class GygesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(144, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_board(board_36):
    # 144 one-hot features = 36 squares × 4 piece states (empty, type1, type2, type3)
    features = np.zeros(144, dtype=np.float32)
    for sq in range(36):
        piece_type = int(board_36[sq])
        features[sq * 4 + piece_type] = 1.0
    return features


def mirror_board(board_36):
    # Left-right flip within each row
    mirrored = board_36.copy()
    for row in range(6):
        for col in range(3):
            left  = row * 6 + col
            right = row * 6 + (5 - col)
            mirrored[left], mirrored[right] = board_36[right], board_36[left]
    return mirrored


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


def save_weights(model, path):
    # Layout: fc1.weight [256,144] + fc1.bias [256] + fc2.weight [1,256] + fc2.bias [1]
    # Total = 36864 + 256 + 256 + 1 = 37377 floats = 149508 bytes
    with open(path, "wb") as f:
        for _, param in model.named_parameters():
            f.write(param.detach().cpu().numpy().astype(np.float32).tobytes())


if __name__ == '__main__':
    files = [os.path.join(DATA_DIR, "hce_100kn.csv")]

    df = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} files")

    data     = df.iloc[:, :36].values.astype(np.int64)
    outcomes = df.iloc[:, 36].values.astype(np.float32)
    outcomes = outcomes * 0.75

    print_stats("Raw dataset", data, outcomes)

    # Mirror augmentation (L-R flip)
    mirrored_data = np.array([mirror_board(data[i]) for i in range(len(data))])
    data     = np.vstack([data, mirrored_data])
    outcomes = np.concatenate([outcomes, outcomes])

    print_stats("After mirroring", data, outcomes)
    print()

    print("Encoding positions...")
    features = np.stack([encode_board(data[i]) for i in range(len(data))])
    print(f"Feature matrix shape: {features.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        features, outcomes, test_size=0.2, random_state=42
    )

    X_train_gpu = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_gpu = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_gpu   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_val_gpu   = torch.tensor(y_val,   dtype=torch.float32, device=device)

    model      = GygesNet().to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn    = nn.MSELoss()

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
