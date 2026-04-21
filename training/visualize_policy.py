import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from policy import DATA_DIR, moves_to_heatmaps, RANK_DECAY, flip_pieces, flip_sq

N_SAMPLES = 4
MIN_MOVES = 10
SEED      = None  # set to an int for reproducible samples


def entropy(p):
    p = p[p > 1e-9]
    return float(-(p * np.log(p)).sum())


def safe_len(s):
    try:    return len(json.loads(s))
    except: return -1


def load_sample_rows(data_dir, n, min_moves, seed):
    files = sorted(glob.glob(os.path.join(data_dir, "policy_w*.csv")))
    if not files:
        raise FileNotFoundError(f"No policy_w*.csv in {data_dir}")

    df = pd.concat(
        [pd.read_csv(f, header=None, names=["pieces", "player", "moves"],
                      engine="python", on_bad_lines="skip") for f in files],
        ignore_index=True,
    )
    counts = df["moves"].apply(safe_len)
    bad = int((counts < 0).sum())
    if bad:
        print(f"Dropped {bad} malformed rows")
    df = df[counts >= min_moves].reset_index(drop=True)
    print(f"Loaded {len(df)} usable rows from {len(files)} file(s)")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def as_grid(vec36):
    # Row 0 of the array holds squares 0–5 (player's home rank).
    # With origin="lower" in imshow, row 0 renders at the bottom.
    return np.asarray(vec36).reshape(6, 6)


def draw_heatmap(ax, grid, title, cmap="viridis"):
    im = ax.imshow(grid, cmap=cmap, origin="lower", vmin=0)
    hi = max(grid.max(), 1e-9)
    for i in range(6):
        for j in range(6):
            v = grid[i, j]
            if v > 0.005:
                color = "white" if v < hi * 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_board(ax, pieces):
    grid = as_grid(pieces)
    ax.imshow(grid, cmap=plt.cm.get_cmap("Blues", 4), origin="lower", vmin=0, vmax=3)
    for i in range(6):
        for j in range(6):
            v = int(grid[i, j])
            if v > 0:
                color = "white" if v >= 2 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Board", fontsize=10)


def visualize(rows):
    _, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)

    for i, row in rows.iterrows():
        pieces = [int(x) for x in str(row["pieces"]).split(",")]
        player = int(row["player"])
        moves  = json.loads(row["moves"])
        if player == 2:
            pieces = flip_pieces(pieces)
            moves = [
                [flip_sq(m[0]),
                 None if m[1] is None else flip_sq(m[1]),
                 flip_sq(m[2]),
                 m[3], m[4]]
                for m in moves
            ]
        P_src, P_pickup, P_dest, pickup_mass = moves_to_heatmaps(moves)

        print(f"-- Sample {i}: {len(moves)} moves | pickup-mass {pickup_mass:.2f} | "
              f"H(src)={entropy(P_src):.2f}  H(pickup)={entropy(P_pickup):.2f}  H(dest)={entropy(P_dest):.2f}")

        draw_board(  axes[i][0], pieces)
        draw_heatmap(axes[i][1], as_grid(P_src),    f"P(source)  H={entropy(P_src):.2f}")
        draw_heatmap(axes[i][2], as_grid(P_pickup), f"P(pickup | pickup-drop)  mass={pickup_mass:.2f}")
        draw_heatmap(axes[i][3], as_grid(P_dest),   f"P(dest)  H={entropy(P_dest):.2f}")

    plt.suptitle(f"Policy targets  (rank decay = {RANK_DECAY})", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_heatmaps.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    rows = load_sample_rows(DATA_DIR, N_SAMPLES, MIN_MOVES, SEED)
    visualize(rows)