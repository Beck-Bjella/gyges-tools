"""Shared utilities for Gyges training scripts."""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mirror_board(board_36):
    """Left-right flip within each row: sq (row, col) -> (row, 5-col)."""
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

    print(f"\n{'-' * 50}")
    print(f"  {tag}")
    print(f"{'-' * 50}")
    print(f"  Total positions:  {n:,}")
    print(f"  Unique boards:    {unique:,} ({100 * unique / n:.1f}%)")
    print(f"  Duplicates:       {dupes:,} ({100 * dupes / n:.1f}%)")
    print(f"  Wins: {wins:,} ({100 * wins / n:.1f}%)  |  Losses: {losses:,} ({100 * losses / n:.1f}%)")


def save_weights(model, path):
    """Write fc weights + biases as raw f32, ordered by named_parameters()."""
    with open(path, "wb") as f:
        for _, param in model.named_parameters():
            f.write(param.detach().cpu().numpy().astype(np.float32).tobytes())


def load_and_split(files, test_size=0.2, random_state=42):
    """Load CSVs, print stats, split train/val, mirror-augment train only.

    Reads columns 0..35 as the board and column 36 as the outcome (ignores any
    extra columns the data-gen code may add in the future, e.g. move_num/game_id).

    Returns (train_boards, val_boards, y_train, y_val) as numpy arrays.
    """
    df = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} files")

    data     = df.iloc[:, :36].values.astype(np.int64)
    outcomes = df.iloc[:, 36].values.astype(np.float32)
    outcomes = outcomes * 0.75  # scale targets inside Tanh's linear region

    print_stats("Raw dataset", data, outcomes)

    # Split BEFORE mirroring so a position and its mirror can't straddle train/val.
    train_data, val_data, y_train, y_val = train_test_split(
        data, outcomes, test_size=test_size, random_state=random_state
    )

    # Mirror augmentation on train only.
    mirrored_train = np.array([mirror_board(train_data[i]) for i in range(len(train_data))])
    train_data = np.vstack([train_data, mirrored_train])
    y_train    = np.concatenate([y_train, y_train])

    print_stats("Train (after mirroring)", train_data, y_train)
    print_stats("Val (no mirror)",         val_data,   y_val)
    print()
    return train_data, val_data, y_train, y_val
