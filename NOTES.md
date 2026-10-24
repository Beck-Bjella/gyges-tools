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
| Generations | gen1 = played by the release net (trained on HCE data), gen2 = played by a net trained on gen1 data, ... | |
| Dataset files | `training/data/<dataset>/w{NN}.csv` (one shard per data-gen worker) | `training/data/gen1-5k/w03.csv` |
| Mix | datasets joined with `+`, no file of its own | `hce-10k+hce-100k` |
| Training run | `{encoding}{width}_{dataset}[_tag]` → `training/weights/<run>/e{N}.bin` | `pair1024_hce-10k/e9.bin` |
| Net | `{run}_e{N}` → `engines/weights/<net>.bin` | `pair1024_hce-10k_e9` |
| Engine | `gyges-v{version}.exe`, dev builds `gyges-dev-{what}.exe` | `gyges-v2.0.0.exe` |

Encodings: `pair{H}` = 5886 → H → 1, `pair{H}x32` = with a 32 tail, `flat{H}` = 144 → H → 1. Run tags only when the same arch is trained on the same data again (`_drop0.3`; `_try{N}` for old runs with no recorded config).
The v2.0.0 engine only loads `pair1024` nets; other widths need a dev build with `L1_SIZE` changed.
Game ids in new datasets are `(hash(dataset name) % 100000 * 1000 + worker) * 10M + game`, so they don't repeat across datasets.

## Datasets

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

Older time-limited data (`hce_p3r2_1s_161k`, `hce_p3r2_2s_56k`, the 1s game folders) has no registry name — being deleted.

## Trained Models

Run folders in `training/weights/`. "Net" = the epoch that got used, if one was.

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
