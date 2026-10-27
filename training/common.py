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
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def append_result(name, row):
    """Append `row` (a dict) to results/<name>.csv, writing the header if the
    file is new. The results/ tables are what the plots read."""
    import csv
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Board utilities ────────────────────────────────────────────────────────

# Left-right flip within each row: sq (row, col) -> (row, 5-col).
MIRROR_PERM = np.array([row * 6 + (5 - col) for row in range(6) for col in range(6)])


def mirror_board(board_36):
    """Left-right flip within each row: sq (row, col) -> (row, 5-col)."""
    return board_36[MIRROR_PERM]


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
        # Squares are 0-3, so int8: 36 bytes a board (40M boards = 1.4 GB).
        self.boards    = np.asarray(boards,   dtype=np.int8)
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

        If any file is 37-col legacy, the whole batch falls back to no-metadata
        mode (otherwise pd.concat would NaN-fill the missing columns and
        astype(int64) would produce i64::MIN garbage). 47-col files (39 + the
        search columns data gen writes since 2026-10-27) load as 39-col.

        Outcomes are scaled (default 0.75) to sit in Tanh's linear region.
        """
        ncols_per_file = [pd.read_csv(f, header=None, nrows=1).shape[1] for f in files]
        for n in ncols_per_file:
            if n not in (37, 39, 47):
                raise ValueError(f"Unexpected CSV column count: {n} (expected 37, 39 or 47)")

        has_metadata = all(n in (39, 47) for n in ncols_per_file)

        # Read one file at a time with small dtypes and keep only numpy arrays,
        # so a 40M-row mix fits in RAM (default int64 frames would need ~12 GB).
        dtypes = {i: np.int8 for i in range(37)}
        if has_metadata:
            dtypes.update({37: np.int32, 38: np.int64})
        boards, outcomes, move_nums, game_ids = [], [], [], []
        for f in files:
            # Trim any 39-col files down to 37 if mixed with legacy files.
            df = pd.read_csv(f, header=None, usecols=list(dtypes), dtype=dtypes)
            boards.append(df.iloc[:, :36].to_numpy())
            outcomes.append(df.iloc[:, 36].to_numpy().astype(np.float32) * outcome_scale)
            if has_metadata:
                move_nums.append(df.iloc[:, 37].to_numpy())
                game_ids.append(df.iloc[:, 38].to_numpy())
            del df

        boards   = np.concatenate(boards)
        outcomes = np.concatenate(outcomes)
        if has_metadata:
            move_nums = np.concatenate(move_nums)
            game_ids  = np.concatenate(game_ids)
        else:
            move_nums = None
            game_ids  = None

        ds = cls(boards, outcomes, move_nums, game_ids)
        mixed_note = "" if all(n == ncols_per_file[0] for n in ncols_per_file) else " (mixed formats — fell back to legacy)"
        print(f"Loaded {len(ds):,} rows from {len(files)} file(s) — "
              f"{'with metadata (move_num, game_id)' if ds.has_metadata else 'legacy 37-col format'}"
              f"{mixed_note}")
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
        # Pack each board into 72 bits (2 bits a square: 64 + 8) and dedupe the
        # two packed columns; deduping all 36 columns needs ~10 GB at 60M rows.
        lo = np.zeros(n, dtype=np.uint64)
        for sq in range(32):
            lo |= self.boards[:, sq].astype(np.uint64) << np.uint64(2 * sq)
        hi = np.zeros(n, dtype=np.uint8)
        for sq in range(32, 36):
            hi |= self.boards[:, sq].astype(np.uint8) << np.uint8(2 * (sq - 32))
        unique = n - int(pd.DataFrame({"lo": lo, "hi": hi}).duplicated().sum())
        del lo, hi
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
        mirrored = self.boards[:, MIRROR_PERM]
        boards   = np.vstack([self.boards, mirrored])
        outcomes = np.concatenate([self.outcomes, self.outcomes])
        if self.has_metadata:
            move_nums = np.concatenate([self.move_nums, self.move_nums])
            game_ids  = np.concatenate([self.game_ids,  self.game_ids])
        else:
            move_nums = None
            game_ids  = None
        return Dataset(boards, outcomes, move_nums, game_ids)
