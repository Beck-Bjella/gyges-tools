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
| 100n | 757k / 1.4M | Mirrored | 100k | AWS + local overnight run |

## Trained Models
| Name | Architecture | Epochs | Dataset | Val Loss |
|------|-------------|--------|---------|----------|
| nnue_h64  (20260421_201800) | 144->64->1  | 100 | 100n (1.67M w/ mirror) | 0.3830 (train 0.3907) |
| nnue_h128 (20260421_203630) | 144->128->1 | 100 | 100n (1.67M w/ mirror) | 0.3691 (train 0.3704) |
| nnue_h256 (20260421_202315) | 144->256->1 | 100 | 100n (1.67M w/ mirror) | 0.3613 (train 0.3509) |
| nnue_h512 (20260421_203023) | 144->512->1 | 100 | 100n (1.67M w/ mirror) | 0.3590 (train 0.3280) — best val 0.3574 @ e60 |

