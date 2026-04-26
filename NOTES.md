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


## Datasets
| Name | Positions (raw/mirrored) | Starting Lines | Node Limit | Notes |
|------|--------------------------|---------------|------------|-------|
| 100n (`hce_100kn.csv`) | 757k / 1.4M | Symmetric (gen_board) — back rank == reverse of front rank | 100k | AWS + local overnight run. Confirmed via inspection 2026-04-23: every starting position satisfies `back[i] == front[5-i]`. Code switched to `gen_board_independent` in commit ff89070 (2026-04-17 12:56), but this dataset was generated on prior code. |
| 5kn (`hce_5kn.csv`) | 6.12M / 9.79M | Independent (gen_board_independent) | 5k | Local 14-worker run after busy-wait fix (2026-04-23/24). 39-col format with move_num + game_id. Unique 92.2%, dupes 7.8%, median game length 16. |
| 10kn (`hce_10kn.csv`) | 4.57M / 7.31M | Independent (gen_board_independent) | 10k | Dedicated machine overnight (2026-04-23/24). 39-col format. Unique 92.9%, dupes 7.1%, median game length 17. |

## Trained Models
| Name | Architecture | Epochs | Dataset | Val Loss |
|------|-------------|--------|---------|----------|
| nnue_h64  (20260421_201800) | 144->64->1  | 100 | 100n (1.67M w/ mirror) | 0.3830 (train 0.3907) |
| nnue_h128 (20260421_203630) | 144->128->1 | 100 | 100n (1.67M w/ mirror) | 0.3691 (train 0.3704) |
| nnue_h256 (20260421_202315) | 144->256->1 | 100 | 100n (1.67M w/ mirror) | 0.3613 (train 0.3509) |
| nnue_h512 (20260421_203023) | 144->512->1 | 100 | 100n (1.67M w/ mirror) | 0.3590 (train 0.3280) — best val 0.3574 @ e60 |
| nnue_h256x32 (20260422)     | 144->256->32->1 | 100 | 100n (1.67M w/ mirror) | 0.3546 best @ e50 |
| nnue_h256_5k_6m (20260424_124044)   | 144->256->1     | 100 | 5kn (9.79M w/ mirror)  | 0.4326 best @ e60 — diff val set, not directly comparable to 100n runs |
| nnue_h256_10k (20260424_173137)     | 144->256->1     | 100 | 10kn (7.31M w/ mirror) | 0.4141 best @ e60 — diff val set |
| nnue_h64_pairs_5k_6m (20260424_144807) | 5886->64->1 | 100+ | 5kn (9.79M w/ mirror) | ~0.4228 @ e20 (still training when checkpointed) |

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
