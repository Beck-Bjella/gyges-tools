# results/

The permanent record of who played whom and how every net was trained. The
plots read these tables; nothing in a figure is typed in by hand.

| File | One row per | Written by |
|---|---|---|
| `matches.csv` | finished match (`gyges-tools sprt …` or `run_sprt`) | the match runner, automatically at the end of a match |
| `runs.csv` | training run: dataset, rows, hidden, dropout, lr, lr_decay, batch, patience, filters, split | `training/train_pair.py`, at the start of a run |
| `epochs.csv` | epoch of a training run: train loss, val loss, lr | `training/train_pair.py`, after every epoch |
| `datasets.csv` | dataset: rows, who played it, nodes, generation, rules | by hand when a dataset is added (mirror of the NOTES.md registry) |
| `figures/` | figure (`.png` + `.pdf`) | `python training/plots.py` |

Net names are `{run}_e{N}`, so a match row joins to `runs.csv` (strip `_eN`) and
from there to `datasets.csv` (split the dataset on `+`).

Rows from before 2026-10-25 were backfilled from the summary boxes in the
session transcript and the match logs (`note` says which); matches stopped
early say so in `note`. Matches with incomplete records (older April runs, the
6-pair smoke test, the 43-pair match the laptop slept through) were left out.

`matches.csv` columns: `pair_score` is the candidate (`net_a`) score per pair of
games, `ll…ww` the pentanomial counts, `a_wins/b_wins/draws` game counts,
`a_depth/b_depth` average search depth, `a_ms/b_ms` average think time per move.
