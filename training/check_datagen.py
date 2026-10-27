"""Validate a data gen dataset (47-column format) end to end.

Replays every game from its rows: applying each row's `best` move to its board and
flipping it must give the next row of the same game. Also checks results, the random
opening plies, and that the search columns agree with each other. Exits non-zero on
any failure.

Usage: python training/check_datagen.py <dataset>
"""
import glob
import os
import sys

from common import REPO_ROOT

FORCED = 100000.0
RANDOM_PLIES = 4          # data gen randomizes the first two full moves


def make_move(board, mv):
    b = board[:]
    if len(mv) == 2:
        a, c = mv
        b[c], b[a] = b[a], 0
    else:
        a, m, c = mv
        p1, p2 = b[a], b[m]
        b[a], b[m], b[c] = 0, p1, p2
    return b


def flip(board):
    return [board[(5 - i // 6) * 6 + (5 - i % 6)] for i in range(36)]


def check(dataset):
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "training", "data", dataset, "w*.csv")))
    problems = {}
    stats = dict(rows=0, games=0, forced=0, forced_vs_result_disagree=0, last_move_wins=0, draws=0,
                 decided_at_root=0, lost_pv_differs=0, max_eval=0.0, min_proven=float("inf"))

    def bad(kind, where):
        problems.setdefault(kind, []).append(where)

    for f in files:
        rows = [line.rstrip("\n").split(",") for line in open(f)]
        stats["rows"] += len(rows)
        for i, r in enumerate(rows):
            where = f"{os.path.basename(f)}:{i + 1}"
            if len(r) != 47:
                bad("column count", where)
                continue
            board = [int(x) for x in r[:36]]
            result, move_num, game = int(r[36]), int(r[37]), int(r[38])
            score, ply, forced, best, nodes, pv, by_depth = r[39], int(r[40]), int(r[42]), r[43], r[44], r[45], r[46]

            # Random opening plies carry no score; every other row has one.
            if (move_num < RANDOM_PLIES) != (score == "nan"):
                bad("random plies / nan score", where)
            if score != "nan":
                s = float(score)
                want = 1 if s >= FORCED else -1 if s <= -FORCED else 0
                if forced != want:
                    bad("forced flag vs score", where)
                # Eval scores are tanh x 10000 (|s| <= 10000); proven results are ~1e8.
                if not (abs(s) <= 10000 or abs(s) >= 1e7):
                    bad("score between the eval range and the proven range", where)
                stats["max_eval"] = max(stats["max_eval"], abs(s) if abs(s) <= 10000 else 0)
                if abs(s) >= 1e7:
                    stats["min_proven"] = min(stats["min_proven"], abs(s))
                if forced:
                    stats["forced"] += 1
                    if result != 0 and forced != result:
                        stats["forced_vs_result_disagree"] += 1
                # Last per-depth entry = final ply and score; plies odd and increasing.
                if by_depth:
                    pairs = [p.split(":") for p in by_depth.split(" ")]
                    plies = [int(p) for p, _ in pairs]
                    if plies[-1] != ply or abs(float(pairs[-1][1]) - s) > 0.11:
                        bad("depth_scores last entry vs ply/score", where)
                    if plies != sorted(set(plies)) or any(p % 2 == 0 for p in plies):
                        bad("depth_scores plies not 1,3,5,...", where)
                if best != "null" and pv and pv.split(" ")[0] != best:
                    if forced == -1:
                        stats["lost_pv_differs"] += 1          # proven loss: engine plays another move
                    else:
                        bad("pv does not start with best", where)
            if best != "null" and int(nodes) <= 0:
                if ply == 0 and (move_num < RANDOM_PLIES or (score != "nan" and forced != 0)):
                    stats["decided_at_root"] += 1              # decided at the root (forced, or a random ply): no nodes/pv
                else:
                    bad("nodes missing", where)

            # Game replay: next row of the same game = this board + best move, flipped.
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            same_game = nxt is not None and len(nxt) == 47 and int(nxt[38]) == game
            if same_game:
                if int(nxt[37]) != move_num + 1:
                    bad("move_num sequence", where)
                if int(nxt[36]) != -result:
                    bad("result does not alternate", where)
                if best not in ("null", "none"):
                    after = flip(make_move(board, [int(x) for x in best.split("|")]))
                    if after != [int(x) for x in nxt[:36]]:
                        bad("replay: board + best != next row", where)
            else:
                stats["games"] += 1
                # Last row of a game: the side to move made the final move.
                if result == 0:
                    stats["draws"] += 1
                elif best not in ("null", "none") and best.split("|")[-1] == "37" and result == 1:
                    stats["last_move_wins"] += 1
                else:
                    bad("last row: final move is not a win for the mover", where)

    print(f"{dataset}: {stats['rows']:,} rows, {stats['games']:,} games "
          f"({stats['draws']} draws, {stats['last_move_wins']:,} ended by the mover reaching the goal)")
    print(f"  forced rows {stats['forced']:,}, disagreeing with the game result: "
          f"{stats['forced_vs_result_disagree']}")
    print(f"  scores: largest eval |score| {stats['max_eval']:.1f} (bound 10000), "
          f"smallest proven |score| {stats['min_proven']:.0f}")
    print(f"  expected engine behaviour: {stats['decided_at_root']} rows decided at the root "
          f"(no nodes/pv), {stats['lost_pv_differs']} proven losses where the pv's first move != best")
    if problems:
        for kind, where in problems.items():
            print(f"  FAIL {kind}: {len(where)} rows (first: {where[:3]})")
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    check(sys.argv[1])
