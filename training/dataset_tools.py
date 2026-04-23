"""
Dataset inspection and filtering tools for Gyges training data.

Usage:
    # Inspect one or more raw CSVs and print a detailed breakdown:
    python dataset_tools.py analyze <file1> [<file2> ...]

    # Create a filtered dataset (drop games shorter than N moves):
    python dataset_tools.py filter --min-moves 10 --out clean.csv <file1> [<file2> ...]

The CSV format is assumed to be 36 squares + 1 outcome column, no header.
Game boundaries are detected by spotting "starting positions" — rows where
front and back ranks are fully occupied and the middle is empty.
"""
import argparse
import os
import sys


# ── Game boundary detection ─────────────────────────────────────────────────

def is_starting_position(cols):
    """Return True if this row is a starting position (new game marker)."""
    if len(cols) < 37:
        return False
    try:
        front  = [int(x) for x in cols[:6]]
        middle = [int(x) for x in cols[6:30]]
        back   = [int(x) for x in cols[30:36]]
    except ValueError:
        return False
    return (all(x == 0 for x in middle)
            and all(x != 0 for x in front)
            and all(x != 0 for x in back))


def iter_games(files):
    """Yield (game_rows_list) for each game across the given files, in order."""
    current = []
    for path in files:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split(',')
                if is_starting_position(cols):
                    if current:
                        yield current
                    current = [line]
                else:
                    current.append(line)
    if current:
        yield current


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze(files):
    total_rows = 0
    game_lengths = []
    wins = losses = draws = 0
    unique_boards = set()
    sym_starts = asym_starts = 0

    for game in iter_games(files):
        game_lengths.append(len(game))
        for row in game:
            total_rows += 1
            cols = row.split(',')
            board = tuple(cols[:36])
            unique_boards.add(board)
            outcome = cols[36].strip()
            if outcome == '1':
                wins += 1
            elif outcome == '-1':
                losses += 1
            else:
                draws += 1

        # Check if first position of game is symmetric or independent
        first_cols = game[0].split(',')
        if is_starting_position(first_cols):
            front = [int(x) for x in first_cols[:6]]
            back  = [int(x) for x in first_cols[30:36]]
            if back == front[::-1]:
                sym_starts += 1
            else:
                asym_starts += 1

    dupes = total_rows - len(unique_boards)
    total_games = len(game_lengths)
    if total_games == 0:
        print("No games detected.")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset analysis")
    print(f"{'='*60}")
    print(f"  Files: {len(files)}")
    for f in files:
        sz = os.path.getsize(f) if os.path.exists(f) else 0
        print(f"    {f}  ({sz/1024/1024:.2f} MB)")
    print()

    print(f"  Positions:     {total_rows:,}")
    print(f"  Games:         {total_games:,}")
    print(f"  Unique boards: {len(unique_boards):,} ({100*len(unique_boards)/total_rows:.1f}%)")
    print(f"  Duplicates:    {dupes:,} ({100*dupes/total_rows:.1f}%)")
    print()

    print(f"  Outcomes:")
    print(f"    Wins:   {wins:,} ({100*wins/total_rows:.1f}%)")
    print(f"    Losses: {losses:,} ({100*losses/total_rows:.1f}%)")
    if draws:
        print(f"    Draws:  {draws:,} ({100*draws/total_rows:.1f}%)")
    print()

    print(f"  Starting-line symmetry:")
    total_starts = sym_starts + asym_starts
    if total_starts:
        print(f"    Symmetric (gen_board):       {sym_starts:,} ({100*sym_starts/total_starts:.1f}%)")
        print(f"    Independent (gen_board_indep): {asym_starts:,} ({100*asym_starts/total_starts:.1f}%)")
    else:
        print(f"    (no starting positions detected)")
    print()

    # Game length stats
    game_lengths_sorted = sorted(game_lengths)
    median = game_lengths_sorted[total_games // 2]
    avg = sum(game_lengths) / total_games
    print(f"  Game length:")
    print(f"    Min / Max:  {game_lengths_sorted[0]} / {game_lengths_sorted[-1]}")
    print(f"    Mean:       {avg:.1f}")
    print(f"    Median:     {median}")
    print()

    # Histogram buckets
    buckets = [(0,5), (5,10), (10,15), (15,20), (20,25), (25,30), (30,40), (40,60), (60, 10**9)]
    print(f"  Game length distribution:")
    print(f"    {'bucket':<10} {'games':>9} {'% games':>9} {'positions':>12} {'% positions':>12}")
    for lo, hi in buckets:
        g = [gl for gl in game_lengths if lo <= gl < hi]
        pos = sum(g)
        hi_str = "inf" if hi >= 10**9 else str(hi)
        label = f"{lo}-{hi_str}"
        print(f"    {label:<10} {len(g):>9,} {100*len(g)/total_games:>8.1f}% "
              f"{pos:>12,} {100*pos/total_rows:>11.1f}%")
    print()

    # Cumulative thresholds
    print(f"  Positions kept by min-moves threshold:")
    print(f"    {'threshold':<10} {'games':>9} {'% games':>9} {'positions':>12} {'% positions':>12}")
    for t in [5, 10, 15, 20, 25, 30]:
        kept_games = sum(1 for gl in game_lengths if gl >= t)
        kept_pos   = sum(gl for gl in game_lengths if gl >= t)
        print(f"    >= {t:<7} {kept_games:>9,} {100*kept_games/total_games:>8.1f}% "
              f"{kept_pos:>12,} {100*kept_pos/total_rows:>11.1f}%")
    print(f"{'='*60}\n")


# ── Filtering ───────────────────────────────────────────────────────────────

def filter_dataset(files, min_moves, out_path):
    """Copy games with >= min_moves positions into a single output CSV."""
    total_games = kept_games = 0
    total_rows = kept_rows = 0

    with open(out_path, 'w', newline='') as out:
        for game in iter_games(files):
            total_games += 1
            total_rows += len(game)
            if len(game) >= min_moves:
                kept_games += 1
                kept_rows += len(game)
                for row in game:
                    out.write(row + '\n')

    print(f"Filter: dropped games with fewer than {min_moves} moves")
    print(f"  Games:     {kept_games:,} / {total_games:,} kept ({100*kept_games/total_games:.1f}%)")
    print(f"  Positions: {kept_rows:,} / {total_rows:,} kept ({100*kept_rows/total_rows:.1f}%)")
    print(f"  Output:    {out_path}  ({os.path.getsize(out_path)/1024/1024:.2f} MB)")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gyges dataset tools")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_analyze = sub.add_parser('analyze', help='Print a detailed dataset breakdown')
    p_analyze.add_argument('files', nargs='+', help='Input CSV file(s)')

    p_filter = sub.add_parser('filter', help='Create a filtered dataset')
    p_filter.add_argument('files', nargs='+', help='Input CSV file(s)')
    p_filter.add_argument('--min-moves', type=int, required=True,
                          help='Drop games shorter than this')
    p_filter.add_argument('--out', required=True, help='Output CSV path')

    args = parser.parse_args()

    for f in args.files:
        if not os.path.exists(f):
            print(f"Error: file not found: {f}", file=sys.stderr)
            sys.exit(1)

    if args.cmd == 'analyze':
        analyze(args.files)
    elif args.cmd == 'filter':
        filter_dataset(args.files, args.min_moves, args.out)


if __name__ == '__main__':
    main()
