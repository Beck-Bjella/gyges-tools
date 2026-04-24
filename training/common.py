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


# ── Board utilities ────────────────────────────────────────────────────────

def mirror_board(board_36):
    """Left-right flip within each row: sq (row, col) -> (row, 5-col)."""
    mirrored = board_36.copy()
    for row in range(6):
        for col in range(3):
            left  = row * 6 + col
            right = row * 6 + (5 - col)
            mirrored[left], mirrored[right] = board_36[right], board_36[left]
    return mirrored


def save_weights(model, path):
    """Write fc weights + biases as raw f32, ordered by named_parameters()."""
    with open(path, "wb") as f:
        for _, param in model.named_parameters():
            f.write(param.detach().cpu().numpy().astype(np.float32).tobytes())


# ── Dataset ────────────────────────────────────────────────────────────────

def _is_starting_board(board):
    """Heuristic for the start of a game: front + back ranks fully occupied,
    middle empty. Used as a fallback when game_ids aren't present in the data."""
    return (np.all(board[:6]   != 0) and
            np.all(board[6:30] == 0) and
            np.all(board[30:]  != 0))


class Dataset:
    """A Gyges training dataset.

    Holds boards (N, 36) and outcomes (N,) plus optional move_num and game_id
    metadata. Auto-detects 37-col (legacy) vs 39-col (with metadata) CSV format.

    All filter / split / augment methods return a new Dataset instead of mutating.
    """

    def __init__(self, boards, outcomes, move_nums=None, game_ids=None):
        self.boards    = np.asarray(boards,   dtype=np.int64)
        self.outcomes  = np.asarray(outcomes, dtype=np.float32)
        self.move_nums = np.asarray(move_nums, dtype=np.int64) if move_nums is not None else None
        self.game_ids  = np.asarray(game_ids,  dtype=np.int64) if game_ids  is not None else None
        assert len(self.boards) == len(self.outcomes)
        if self.move_nums is not None: assert len(self.move_nums) == len(self.boards)
        if self.game_ids  is not None: assert len(self.game_ids)  == len(self.boards)

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_files(cls, files, outcome_scale=0.75):
        """Load one or more CSVs. Auto-detects 37-col vs 39-col format.

        Outcomes are scaled (default 0.75) to sit in Tanh's linear region.
        """
        df = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)
        ncols = df.shape[1]
        if ncols not in (37, 39):
            raise ValueError(f"Unexpected CSV column count: {ncols} (expected 37 or 39)")

        boards    = df.iloc[:, :36].values.astype(np.int64)
        outcomes  = df.iloc[:, 36].values.astype(np.float32) * outcome_scale
        if ncols == 39:
            move_nums = df.iloc[:, 37].values.astype(np.int64)
            game_ids  = df.iloc[:, 38].values.astype(np.int64)
        else:
            move_nums = None
            game_ids  = None

        ds = cls(boards, outcomes, move_nums, game_ids)
        print(f"Loaded {len(ds):,} rows from {len(files)} file(s) — "
              f"{'with metadata (move_num, game_id)' if ds.has_metadata else 'legacy 37-col format'}")
        return ds

    # ── Properties ────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.outcomes)

    @property
    def has_metadata(self):
        return self.move_nums is not None and self.game_ids is not None

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self, tag="Dataset"):
        """Quick one-screen summary. Use during training."""
        n = len(self)
        wins   = int((self.outcomes > 0).sum())
        losses = int((self.outcomes < 0).sum())
        draws  = n - wins - losses
        unique = len(np.unique(self.boards, axis=0))
        dupes  = n - unique

        print(f"\n{'-' * 50}")
        print(f"  {tag}")
        print(f"{'-' * 50}")
        print(f"  Total positions:  {n:,}")
        print(f"  Unique boards:    {unique:,} ({100 * unique / n:.1f}%)")
        print(f"  Duplicates:       {dupes:,} ({100 * dupes / n:.1f}%)")
        print(f"  Wins: {wins:,} ({100 * wins / n:.1f}%)  |  "
              f"Losses: {losses:,} ({100 * losses / n:.1f}%)" +
              (f"  |  Draws: {draws:,} ({100 * draws / n:.1f}%)" if draws else ""))

        if self.has_metadata:
            n_games = len(np.unique(self.game_ids))
            avg_len = n / n_games
            print(f"  Games:            {n_games:,}  (avg {avg_len:.1f} positions/game)")
            print(f"  Move number:      min {self.move_nums.min()}  "
                  f"max {self.move_nums.max()}  "
                  f"mean {self.move_nums.mean():.1f}")

    def _game_lengths(self):
        """Return an array of per-game position counts (using game_id when
        present, falling back to the starting-position heuristic for legacy data)."""
        if self.has_metadata:
            _, counts = np.unique(self.game_ids, return_counts=True)
            return counts
        # Legacy: scan for game starts via heuristic
        lengths = []
        cur = 0
        for i in range(len(self)):
            if i > 0 and _is_starting_board(self.boards[i]):
                lengths.append(cur)
                cur = 1
            else:
                cur += 1
        if cur > 0:
            lengths.append(cur)
        return np.array(lengths)

    def analyze(self, tag="Dataset"):
        """Detailed breakdown: stats + game length distribution + cumulative
        threshold table. Use for one-off inspection (CLI, post-collection)."""
        self.stats(tag)
        lengths = self._game_lengths()
        if len(lengths) == 0:
            print("  No games detected.")
            return

        n = len(self)
        sorted_l = np.sort(lengths)
        print(f"\n  Game length:")
        print(f"    Min / Max:  {sorted_l[0]} / {sorted_l[-1]}")
        print(f"    Mean:       {lengths.mean():.1f}")
        print(f"    Median:     {sorted_l[len(sorted_l)//2]}")

        buckets = [(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,40),(40,60),(60,10**9)]
        total_games = len(lengths)
        print(f"\n  Game length distribution:")
        print(f"    {'bucket':<10} {'games':>9} {'% games':>9} {'positions':>12} {'% positions':>12}")
        for lo, hi in buckets:
            inb = lengths[(lengths >= lo) & (lengths < hi)]
            n_g = len(inb)
            n_p = int(inb.sum())
            hi_str = "inf" if hi >= 10**9 else str(hi)
            print(f"    {lo}-{hi_str:<6} {n_g:>9,} {100*n_g/total_games:>8.1f}% "
                  f"{n_p:>12,} {100*n_p/n:>11.1f}%")

        print(f"\n  Positions kept by min-moves threshold:")
        print(f"    {'threshold':<10} {'games':>9} {'% games':>9} {'positions':>12} {'% positions':>12}")
        for t in [5, 10, 15, 20, 25, 30]:
            keep = lengths[lengths >= t]
            n_g = len(keep)
            n_p = int(keep.sum())
            print(f"    >= {t:<7} {n_g:>9,} {100*n_g/total_games:>8.1f}% "
                  f"{n_p:>12,} {100*n_p/n:>11.1f}%")

    # ── Filtering ─────────────────────────────────────────────────────────

    def _select(self, mask):
        """Return a new Dataset with rows where `mask` is True."""
        return Dataset(
            self.boards[mask],
            self.outcomes[mask],
            self.move_nums[mask] if self.move_nums is not None else None,
            self.game_ids[mask]  if self.game_ids  is not None else None,
        )

    def filter_min_moves(self, min_moves):
        """Drop entire games shorter than `min_moves` positions.

        Uses game_id when present; otherwise falls back to detecting game
        boundaries via the starting-position heuristic.
        """
        if self.has_metadata:
            unique_ids, counts = np.unique(self.game_ids, return_counts=True)
            keep_ids = set(unique_ids[counts >= min_moves].tolist())
            mask = np.fromiter((gid in keep_ids for gid in self.game_ids),
                               dtype=bool, count=len(self))
        else:
            mask = np.zeros(len(self), dtype=bool)
            game_start = 0
            for i in range(len(self)):
                if i > 0 and _is_starting_board(self.boards[i]):
                    if i - game_start >= min_moves:
                        mask[game_start:i] = True
                    game_start = i
            if len(self) - game_start >= min_moves:
                mask[game_start:] = True
        return self._select(mask)

    def filter_min_move_num(self, min_move_num):
        """Drop positions with move_num < min_move_num (early-game noise).
        Requires metadata (new-format CSV)."""
        if not self.has_metadata:
            raise ValueError("filter_min_move_num requires metadata (new-format CSV)")
        return self._select(self.move_nums >= min_move_num)

    # ── Split + augment ──────────────────────────────────────────────────

    def split(self, test_size=0.2, random_state=42):
        """Train/val split. Returns (train_ds, val_ds).

        Note: positions from the same game can land in different splits.
        For game-aware splitting, a future version could use GroupShuffleSplit.
        """
        idx_train, idx_val = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=random_state
        )
        return self._select(idx_train), self._select(idx_val)

    def mirror_augment(self):
        """Return a new Dataset with each row plus its left-right mirror."""
        mirrored = np.array([mirror_board(b) for b in self.boards])
        boards   = np.vstack([self.boards, mirrored])
        outcomes = np.concatenate([self.outcomes, self.outcomes])
        if self.has_metadata:
            move_nums = np.concatenate([self.move_nums, self.move_nums])
            game_ids  = np.concatenate([self.game_ids,  self.game_ids])
        else:
            move_nums = None
            game_ids  = None
        return Dataset(boards, outcomes, move_nums, game_ids)
