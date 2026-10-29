# Gyges NN Eval — Experiment Tracker

## Experiments (after cleanup)
14. Independent starting lines dataset — gen_board_independent() already in code
15. Compare independent lines model vs symmetric lines model
16. Dataset size scaling test (100k / 300k / 700k / 1.4M)
17. NN vs HCE at deeper search (depth 7, 9)
18. Self-play data generation with NN engine (need higher node count)

19. Train model WITHOUT control features (144 inputs, piece type only) — test if control is masking positional learning
20. Try CNN architecture — reshape board as 6x6 grid with piece/control channels, let conv layers learn spatial patterns (AlphaZero-style)

## IDEAS
The NN's might be really strong in endgames assuming they dont throw the early game. Thus after a certin depth average mabye the HCE cant outperform. This means an opening book based on the NN's best moves, and then a hybrid combo or smth?


## Naming scheme (since 2026-10-23)

Names say what the thing is; this file is the registry with the details.

| Thing | Pattern | Example |
|---|---|---|
| Dataset | `{who played}-{nodes per move}[-tag]`. HCE data is `hce-…`; NN self-play is `gen{N}-…` | `hce-10k`, `gen1-5k` |
| Generations | gen{N} = who played the games — see **Generations** below | |
| Dataset files | `training/data/<dataset>/w{NN}.csv` (one shard per data-gen worker) | `training/data/gen1-5k/w03.csv` |
| Mix | datasets joined with `+`, no file of its own | `hce-10k+hce-100k` |
| Training run | `{encoding}{width}_{dataset}[_tag]` → `training/weights/<run>/e{N}.bin` | `pair1024_hce-10k/e9.bin` |
| Net | `{run}_e{N}` → `engines/weights/<net>.bin` | `pair1024_hce-10k_e9` |
| Engine | `gyges-v{version}.exe`, dev builds `gyges-dev-{what}.exe` | `gyges-v2.0.0.exe` |

Encodings: `pair{H}` = 5886 → H → 1, `pair{H}x32` = with a 32 tail, `flat{H}` = 144 → H → 1. Run tags only when the same arch is trained on the same data again (`_drop0.3`; `_try{N}` for old runs with no recorded config).
**No tag = the release recipe** (`train_pair.py` defaults): dropout 0.5, Adam lr 1e-3, batch 4096, mirror augment, seeded 80/20 split. A tag names what differs from it (`_drop0.3` = dropout 0.3, everything else default; `_lrdecay0.3` = lr ×0.3 after each epoch without a new best val; `_mm2` = positions with move_num < 2 dropped). Since 2026-10-25 runs stop after 3 epochs without a new best val, and each run folder has `config.txt` + `log.txt`.
The v2.0.0 engine only loads `pair1024` nets; other widths need a dev build with `L1_SIZE` changed.
Game ids in new datasets are `(hash(dataset name) % 100000 * 1000 + worker) * 10M + game`, so they don't repeat across datasets.

## Generations

A generation is **who played the games**. A new generation starts when a clearly stronger net becomes the player; dropping the random-opening rows (moves 4+) counts as one, since it made the gen1 net ~+70 Elo stronger on the same games.

| Gen | Games played by | Datasets | Nets trained on it |
|---|---|---|---|
| **gen0** | HCE (hand-crafted eval) | `hce-10k` (= part1 + part2), `hce-5k`, `hce-100k` | **release net** `pair1024_hce-10k_e9` (shipped in v2.0.0) |
| **gen1** | release net `pair1024_hce-10k_e9` | `gen1-10k`, `gen1-10k-aws`, `gen1-10k-aws2` (19.87M together), `gen1-1k` | **gen1 e9** `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` (all rows); **best net** `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_from4_e10` (moves 4+) |
| **gen2** | gen1 e9 | `gen2-10k-aws`, `gen2-20k-aws` | ~4M test nets only (see ~5M nets below); pooled with gen1 in mixes |
| **gen3** | best net (gen1 data, moves 4+) | `test-gen3-10k-min3-nb` (10-min check), `gen3-10k-laptop` (4-ply pilot, not used), `test-gen3-10k-min3-nb8` (8-ply check); next AWS set | — |

Recipes (how the games were played and which rows are trained on):

| Recipe | Opening | Search | Rows trained | Used for |
|---|---|---|---|---|
| **old** | `searched`: 4 random non-losing plies (first mover gets a head start) | node cap only (10k: ~11% of moves stop at ply 1) | all rows until 2026-10-27, then **moves 4+** (`MIN_MOVE_NUM = 4`, run tag `_from4`) | gen0–gen2 |
| **new** | `nearbest`: 8 random plies (datagen `random_moves` 4) among moves within 0.1 of the best two-ply score (4 plies gave too few distinct openings: duplicate games) | node cap + `minPly 3` (every move sees the reply) | all rows (moves 4+ as an ablation) | gen3 on (needs `engines/gyges-v2.0.0-datagen.exe`, GygesRust branch `datagen-options`) |

## Datasets

Row format: 39 columns (`sq0..sq35, result, move_num, game_id`) up to gen2-10k-aws; from 2026-10-27 data gen writes 47 (`+ score, ply, time_s, forced, best, nodes, pv, depth_scores` — see ENGINE_HISTORY "Why the next data saves the search"). Check a new dataset with `python training/check_datagen.py <dataset>`.

Everything up to `hce-10k` was generated under the pre-2.0.0 rules (board-unchanged moves still legal; ~1% of moves in hce-10k). Locations as of 2026-10-23.

| Dataset | File(s) | Rows | Played by / nodes | Starts | Notes |
|---|---|---|---|---|---|
| `hce-100k` | `hce_100kn.csv` | 833k | HCE / 100k | Symmetric (gen_board) | `DATASET_BACKUP/` (being deleted). 37-col (no move_num/game_id). AWS + local overnight run; every start has `back[i] == front[5-i]`. |
| `hce-5k` | `hce_5kn.csv` | 6.12M | HCE / 5k | Independent | `DATASET_BACKUP/` (being deleted). Local 14-worker run 2026-04-23/24. Unique 92.2%, median game length 16. |
| `hce-10k-part1` | `hce_10kn.csv` | 4.57M | HCE / 10k | Independent | `V2.0.0_release_dataset/` (frozen). Dedicated machine, 2026-04-23/24. Unique 92.9%, median length 17. |
| `hce-10k-part2` | `hce_10kn_big.csv` | 15.20M | HCE / 10k | Independent | `V2.0.0_release_dataset/` (frozen). 15-worker run, ~2026-04-26. Node budget not recorded in code; game stats match part1 exactly. Game ids overlap part1's. |
| **`hce-10k`** | = part1 + part2 (was `hce_10kn_combined.csv`) | 19.77M | HCE / 10k | | **The release data: training set of `pair1024_hce-10k_e9`, shipped in v2.0.0.** `train_pair.py` loads the two archive files directly. |
| `hce-10k+hce-100k` | mix | 20.61M | | | hce-100k is ~4% of rows ("DUO"). |
| `hce-10k+hce-100kx8` | mix | 26.44M | | | hce-100k listed 8 times, ~25% of rows. |
| `gen1-10k` | `training/data/gen1-10k/` | 0.35M | `pair1024_hce-10k_e9` / 10k on every move (random openers too) | Independent | First NN self-play data, first data under 2.0.0 rules. Laptop, 12 workers, 2026-10-23 19:55-22:22. Unique 96.3%, median length 18. |
| `gen1-10k-aws` | `training/data/gen1-10k-aws/` | 4.35M | same as `gen1-10k` | Independent | Same config as `gen1-10k`, run on AWS c8a.48xlarge, 190 workers, ~1 h 15 min on 2026-10-24 (stopped to match hce-10k-part1 size; instance stopped, can resume). Separate name so ids don't collide; train on both together. |
| `gen1-10k-aws2` | `training/data/gen1-10k-aws2/` | 15.16M | same as `gen1-10k` | Independent | Same config, AWS c8a.48xlarge, 190 workers, ~4 h on 2026-10-24. Separate name so `gen1-10k-aws` (what `pair1024_gen1-10k+gen1-10k-aws_e6` was trained on) stays unchanged. |
| `gen1-1k` | `training/data/gen1-1k/` | 4.73M | `pair1024_hce-10k_e9` / **1k** on every move (random openers too) | Independent | Same player as gen1-10k, 10× fewer nodes — tests whether data-gen search depth matters. Laptop 2026-10-25, 12 then 15 workers (15 = 1.36M rows/h vs 1.27M), ~3.5 h. 4,733,829 rows. |
| `gen2-10k-aws` | `training/data/gen2-10k-aws/` | 4.85M | `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` / 10k on every move | Independent | **First gen2 data**: played by the gen1 19.9M net (+65 Elo over the release net at 10k). AWS c8a.48xlarge, 190 workers, 1.35 h on 2026-10-26, 3.6M rows/h, 4,852,079 rows, 93.0% unique. Size-matched to gen1 4.7M. CloudWatch alarm (CPU < 5% for 3×5 min → Stop) stopped the instance after the run. |
| `gen1-10k-aws2@59` | first 59 shards (sorted name order) of `gen1-10k-aws2` | 4.72M | same as `gen1-10k` | Independent | Subset syntax `<dataset>@N` in train_pair.py (no files copied). Size-matched second half for the gen1+gen1 control mix. |
| `gen1-10k+gen1-10k-aws+gen1-10k-aws2` | mix | 19.87M | | | All gen1 data so far; size-matched to `hce-10k` (19.77M). |
| `hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2` | mix | 39.64M | | | All HCE release data + all gen1 data. Unique 88.8%. Tests whether more data helps even when half of it is weaker (HCE) labels. |
| `gen2-20k-aws` | `training/data/gen2-20k-aws/` | 5.03M | `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` / 20k on every move | Independent, 4 random plies (`opening searched`) | **Depth test**: same player as `gen2-10k-aws` at twice the nodes. First 47-column data (search columns: score, ply, time, forced, best, nodes, pv, depth_scores). AWS c8a.48xlarge, 190 workers, 2.7 h on 2026-10-27, 1.86M rows/h, 5,028,120 rows, 284,761 games, 92.9% unique. `check_datagen.py`: all pass; 1.72M proven rows (34%), 1,762 (0.10%) disagree with the game result. Includes `gen2-20k.log`. First download was cut short (server tar still writing + idle alarm) — redone, archive md5 6c94ec5f99d10ab058732087556f8d10. |
| `test-none-10k` | `training/data/test-none-10k/` | 29,934 | `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` / 10k | Independent, **no random plies** (`opening none`) | Test set for real openings: local, 12 workers, 15 min on 2026-10-27, 1,455 games. Games from the same start board are identical (deterministic), so ≤ 8,100 games per player/node setting. 47 columns. |

Older time-limited data (`hce_p3r2_1s_161k`, `hce_p3r2_2s_56k`, the 1s game folders) has no registry name — being deleted.

**Test sets** (small, laptop, 47 columns; for measuring data and nets, never trained on). Details in `results/datasets.csv`.

| Dataset | Played by (gen) / nodes | Recipe | Games | Purpose |
|---|---|---|---|---|
| `test-gen2-10k-search` | gen1 e9 (gen2) / 10k | old | 767 | proof timing vs gen2-20k-aws |
| `test-none-best-10k` | best net (gen3) / 10k | no random plies | 389 decisive (81% draws) | real openings, best net |
| `test-none-release-10k` | release (gen1) / 10k | no random plies | 1,688 | real openings — the reference for "real play" (first mover 46.4%) |
| `test-release-1k` / `-3k` / `-10k` | release (gen1) / 1k, 3k, 10k | old | 12,719 / 4,080 / 1,344 | node-budget comparison (blind ply-1 moves, game quality) |
| `test-release-10k-min3` | release (gen1) / 10k | old opening + floor 3 (tools-side re-search) | 1,140 | floor test |
| `test-release-10k-min3-nb` | release (gen1) / 10k | new, tools-side version | 477 | near-best opening test (tools-side, since removed) |
| `test-release-10k-min3-nb-engine` | release (gen1) / 10k | new (engine options) | 660 | same, engine-side; speed check |
| `test-gen3-10k-min3-nb` | best net (gen3) / 10k | new, 4 random plies | — | draw rate / speed of the gen3 player before AWS |
| `gen3-10k-laptop` | best net (gen3) / 10k | new, 4 random plies | — | overnight pilot (713k rows): floor check; found duplicate games (2.7%) |
| `test-gen3-10k-min3-nb8` | best net (gen3) / 10k | new, 8 random plies | — | the AWS recipe: duplicates, fairness, speed |

## Trained Models

Run folders in `training/weights/`. "Net" = the epoch that got used, if one was.

**~5M comparison nets** — one per data source, all release recipe, h1024 pair, size-matched (~4.6–4.9M games' worth of rows). These are the nets to compare data sources and recipes with; results in ENGINE_HISTORY ("~5M nets: one table").

| Net | Data (gen) | Player / nodes | Rows trained | What it isolates |
|---|---|---|---|---|
| `pair1024_hce-10k-part1_e6` | hce-10k-part1 (gen0) | HCE / 10k | 4.57M | gen0 baseline |
| `pair1024_gen1-10k+gen1-10k-aws_e6` | gen1 4.7M (gen1) | release / 10k | 4.70M | gen1 baseline, all rows |
| `pair1024_gen1-10k+gen1-10k-aws_from4_e6` | same | same | 3.66M (moves 4+) | moves 4+ on gen1 |
| `pair1024_gen1-10k+gen1-10k-aws_from4_min11_e5` | same | same | 3.52M (moves 4+, games ≤ 10 rows dropped) | short-game filter |
| `pair1024_gen1-1k_e5` | gen1-1k (gen1) | release / **1k** | 4.73M | data-gen depth: 1k vs 10k |
| `pair1024_gen1-1k_from4_e5` | same | same | 3.65M (moves 4+) | moves 4+ on 1k data |
| `pair1024_gen2-10k-aws_e6` | gen2-10k-aws (gen2) | gen1 e9 / 10k | 4.85M | gen2 vs gen1, all rows |
| `pair1024_gen2-10k-aws_from4_e6` | same | same | 3.77M (moves 4+) | gen2 vs gen1, moves 4+ |
| `pair1024_gen2-20k-aws_from4_e7` | gen2-20k-aws (gen2) | gen1 e9 / **20k** | 3.89M (moves 4+) | data-gen depth: 20k vs 10k |
| `pair1024_gen2-20k-aws_from4_blend0.5_e8` | same | same | 3.89M (moves 4+, blend labels) | label ablation (search score blended in) |

| Run | Original name | Epochs (used) | Val loss / notes |
|---|---|---|---|
| `flat64_hce-100k` | nnue_h64 (20260421_201800) | 100 (e100) | 0.3830 (train 0.3907). Run folder gone; net in `engines/legacy/flat64_hce-100k_e100.bin` |
| `flat128_hce-100k` | nnue_h128 (20260421_203630) | 100 (e100) | 0.3691 (train 0.3704). `engines/legacy/flat128_hce-100k_e100.bin` |
| `flat256_hce-100k` | nnue_h256 (20260421_202315) | 100 (e100) | 0.3613 (train 0.3509). `engines/legacy/flat256_hce-100k_e100.bin` |
| `flat512_hce-100k` | nnue_h512 (20260421_203023) | 100 (e60) | 0.3590 @ e100, best 0.3574 @ e60. `engines/legacy/flat512_hce-100k_e60.bin` |
| `flat256x32_hce-100k` | nnue_h256x32 (20260422) | 100 (e50) | best 0.3546 @ e50. Weights not kept. |
| `flat256_hce-5k` | nnue_h256_5k_6m (20260424_124044) | 60 (e60) | 0.4326 best @ e60 — different val set from the hce-100k runs |
| `pair64_hce-5k` | nnue_h64_pairs_5k_6m (20260424_144807) | 21 (e20) | ~0.4228 @ e20, dropout 0.4 |
| `flat256_hce-10k-part1` | nnue_h256_10k (20260424_173137) | 80 (e60/e70) | 0.4141 best @ e60 |
| `pair64_hce-10k-part1` | nnue_h64_pairs_10k (20260426_015734) | 26 | not logged |
| `pair128_hce-10k-part1` | nnue_h128_pairs_10k (20260426_022729) | 52 | not logged |
| `pair256_hce-10k-part1` | nnue_h256_pairs_10k (20260426_145011) | 19 | not logged |
| `flat512_hce-10k-part1` | nnue_h512_10k (20260426_151937) | 60 | not logged |
| `pair512_hce-10k-part1` | nnue_h512_pairs_10k (20260426_160534) | 21 | not logged |
| `pair1024_hce-10k-part1` | nnue_h1024_pairs_10k (20260426_193557) | 53 (e6) | not logged; dropout maybe 0.5. Net `pair1024_hce-10k-part1_e6` (was `e6_pair_h1024_10k.bin`) |
| `pair1024_hce-10k_try1` … `_try4` | h1024 combined, 20260427 140107 / 161023 / 203955 / 235517 | 21 / 10 / 8 / 6 | configs not recorded |
| `pair2048_hce-10k_try1` | h2048 combined (20260427_225403) | 2 | stopped early |
| `pair2048_hce-10k` | nnue_h2048_pairs_10k_combined (20260428_201225) | 6 (e5) | not logged — did not beat h1024. Net deleted (duplicate, no engine loads 2048) |
| `pair1024x32_hce-10k` | nnue_h1024x32_pairs_10k_combined (20260429_001055) | 6 (e5) | not logged, dropout 0.3. Net deleted (duplicate, no engine loads x32) |
| **`pair1024_hce-10k`** | nnue_h1024_pairs_10k_combined (20260429_172819) | 11 (**e9**) | 0.3901 (plateau e8-e11), dropout 0.5. **Net `pair1024_hce-10k_e9` = shipped in GygesRust v2.0.0** as `weights/default.bin` (md5 75a3b36a…), was `e9_pair_h1024_combined.bin` |
| `pair1024_hce-10k+hce-100k` | nnue_h1024_pairs_10k_plus_100k (20260429_185729) | 10 (e9) | 0.3880 @ e10 — different val set. "DUO", coin flip vs the release net. Net `pair1024_hce-10k+hce-100k_e9` |
| `pair1024_gen1-10k+gen1-10k-aws` | — (2026-10-24, current train_pair.py) | 8 (**e6**) | 0.3539 @ e6 (e7 0.3538, e8 0.3544 back up), train 0.3259 @ e6; dropout 0.5, release recipe. First net trained on NN self-play data (4.70M rows, size-matched to hce-10k-part1). Different val set — not comparable to 0.3901. Net `pair1024_gen1-10k+gen1-10k-aws_e6` |
| `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2` | — (2026-10-24, current train_pair.py) | 12 (**e9**) | 0.3445 @ e9 (e10 0.3444, e11/e12 0.3443), train 0.3350 @ e9; dropout 0.5, release recipe. 19.87M rows — size-matched to `hce-10k`. Net `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` |
| `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_nosingle` | — (2026-10-26) | 14, stopped itself (**e9**) | 0.3447 @ e9 (best 0.3446 @ e10/11); release recipe but **pairs only** (`SINGLETONS = False`: the 108 single-piece features never encoded, their weight rows saved as zero, so v2.0.0 loads it unchanged). Same data and split as the run above (best 0.3443). Net `…_nosingle_e9` |
| `pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2` | — (2026-10-25, GPU-encoding train_pair.py) | 14, stopped by hand (**e12**) | 0.3664 @ e12 (best 0.3663 @ e14), train 0.3616 @ e12; release recipe (dropout 0.5). 39.64M rows, 63.4M mirrored train. Val set is half HCE games — not comparable to gen1-only val. Net `pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12` |
| `pair1024x32_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2` | — (2026-10-26/27) | 14, stopped by hand (**e12** tested) | 0.3593 @ e12 (0.3592 @ e14), train 0.3534 @ e12; release recipe + **tail** `1024→32→ReLU→1` (`TAIL = 32`). Same data/split as the no-tail mix (best 0.3663). Needs the dev engine `engines/gyges-dev-x32.exe` (GygesRust worktree `GygesRust-dev-tail`, branch `dev-tail`, uncommitted); play it as `dev:<net>` in `gyges-tools sprt`. Net `pair1024x32_…_e12` |
| `pair1024_gen1-1k` | — (2026-10-25, GPU-encoding train_pair.py) | 9, stopped itself (**e5**) | 0.4182 @ e5 (best 0.4181 @ e6, back up from e7), train 0.3951 @ e5; release recipe (dropout 0.5). 4.73M rows of 1k-node games, size-matched to `pair1024_gen1-10k+gen1-10k-aws` (val 0.3539 there — different val set, but 1k-game results are clearly harder to predict). Net `pair1024_gen1-1k_e5` |
| `pair1024_gen2-10k-aws` | — (2026-10-26) | 10, stopped itself (**e6**) | 0.3414 @ e6 (best 0.3413 @ e7), train 0.3148 @ e6; release recipe. 4.85M rows of gen2 games. Val clearly lower than gen1 4.7M (0.3539) — different val set, but cleaner labels. Net `pair1024_gen2-10k-aws_e6` |
| `pair1024_gen1-10k+gen1-10k-aws+hce-10k-part1` | — (2026-10-26) | 9, stopped itself (**e6**) | 0.3756 @ e6 (best); release recipe. 9.28M = gen1 4.7M + gen0 4.57M (half HCE val set). Net `…+hce-10k-part1_e6` |
| `pair1024_gen1-10k+gen1-10k-aws+gen2-10k-aws` | — (2026-10-26) | 11, stopped itself (**e7**) | 0.3447 @ e7 (best 0.3446 @ e8); release recipe. 9.56M = gen1 4.7M + gen2 4.85M. Net `…+gen2-10k-aws_e7` |
| `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2@59` | — (2026-10-26) | 10, stopped itself (**e7**) | 0.3504 @ e7 (best); release recipe. 9.42M = gen1 4.7M + 4.72M more gen1 (control). Net `…+gen1-10k-aws2@59_e7` |
| `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2+gen2-10k-aws` | — (2026-10-26) | 16, stopped by hand (**e12**) | 0.3409 @ e12 (best 0.3407 @ e16, creeping ≤0.0001); release recipe. 24.72M = gen1 19.87M + gen2 4.85M. Net `…+gen2-10k-aws_e12` |
| `pair1024_gen1-10k+gen1-10k-aws_from4` | — (2026-10-27) | 9, stopped itself (**e6**) | 0.3010 @ e6; release recipe on gen1 4.7M with rows of moves 0–3 (the random plies) dropped (`MIN_MOVE_NUM = 4`): 3.66M rows. Val not comparable (no opening rows). Net `pair1024_gen1-10k+gen1-10k-aws_from4_e6` |
| `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_from4` | — (2026-10-27) | 15, stopped itself (**e10**) | 0.2892 @ e10 (best 0.2890 @ e12); release recipe on gen1 19.87M with moves 0–3 dropped: 15.46M rows. Val not comparable (no opening rows). Net `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_from4_e10` |
| `pair1024_gen2-10k-aws_from4` | — (2026-10-27) | 9, stopped itself (**e6**) | 0.2837 @ e6; release recipe on gen2-10k-aws with moves 0–3 dropped: 3.77M rows. The 10k side of the gen2-20k-aws depth comparison. Net `pair1024_gen2-10k-aws_from4_e6` |
| `pair1024_gen2-20k-aws_from4` | — (2026-10-27) | 10, stopped itself (**e7**) | 0.2497 @ e7 (e6 0.2499 just outside 0.0002); release recipe on gen2-20k-aws with moves 0–3 dropped: 3.89M rows. Depth test: vs `pair1024_gen2-10k-aws_from4_e6` (same player at 10k, 3.77M rows; val 0.2837 — different val set). Net `pair1024_gen2-20k-aws_from4_e7` |
| `pair1024_gen2-20k-aws_from4_blend0.5` | — (2026-10-27) | 11, stopped itself (**e8**) | 0.13914 @ e8 (val on blended targets — not comparable); same 3.89M rows as `…_from4`, `LABELS = "blend"`, λ 0.5: 1.72M proven rows → 0.75 × proven result, 2.17M → 0.5 × 0.75 × result + 0.5 × 0.75 × calibrated search score (score 0 → −0.263, ±0.5 → +0.670 / −0.875). Net `pair1024_gen2-20k-aws_from4_blend0.5_e8` |
| `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2+gen2-10k-aws+gen2-20k-aws_from4` | — (2026-10-28) | 18, stopped itself (**e12**) | 0.27593 @ e12 (best 0.27586 @ e15); release recipe, moves 4+, result labels on gen1 19.87M + gen2-10k 4.85M + gen2-20k 5.03M = 29.75M → 23.1M rows. First attempt stopped at e8 (shutdown) — kept as `…_from4_stopped-e8`. Net `…+gen2-20k-aws_from4_e12` |
| `pair1024_gen1-10k+gen1-10k-aws_from4_min11` | — (2026-10-27) | 8, stopped itself (**e5**) | 0.3074 @ e5; release recipe on gen1 4.7M with games of ≤ 10 rows dropped (`MIN_GAME_MOVES = 11`, −7.0% rows) and then moves 0–3 (3.52M rows). Val above `_from4` (0.3010) as expected — the dropped short games are mostly easy, proven positions. Net `pair1024_gen1-10k+gen1-10k-aws_from4_min11_e5` |
| `pair1024_gen1-1k_from4` | — (2026-10-28) | 8, stopped itself (**e5**) | release recipe on gen1-1k with moves 0–3 dropped: 3.65M rows. Moves 4+ on the 1k data (least opening bias). Net `pair1024_gen1-1k_from4_e5` |
| — | nnue_h1024_pairs_10k_plus_100kx8 (20260429_213158) | — | trained on `hce-10k+hce-100kx8`, never saved a checkpoint; empty dir removed |

Empty run dirs removed 2026-10-23 (no checkpoints): 20260424_141232, 20260427_013035, 20260427_015132, 20260429_213158.

## Other
═════════════════════════════════════════════════════════════════
  ANALYSIS COMPLETE  —  HCE  vs  NN 256 10Kn  (794 games)
═════════════════════════════════════════════════════════════════
  Results:
    HCE          |   196 wins   (24.7%)
    NN 256 10Kn  |   551 wins   (69.4%)
    Draws        |    47 draws  (5.9%)

  Set results  (397 sets):
    HCE          |    16 full wins (4.0%)     14 partial wins (3.5%)
    NN 256 10Kn  |   192 full wins (48.4%)     17 partial wins (4.3%)
    Tied         |   158 tied sets (39.8%)

  Think times:
    HCE          |  avg  1402.9ms  min     0ms  max  6823ms  avg depth  4.08  (8344 moves)
    NN 256 10Kn  |  avg  1247.6ms  min     0ms  max  6817ms  avg depth  3.96  (8509 moves)

  Game length (moves):
    HCE          |  wins avg  19.4  losses avg  21.5
    NN 256 10Kn  |  wins avg  21.5  losses avg  19.4
    Draws        |  avg  25.2
═════════════════════════════════════════════════════════════════
