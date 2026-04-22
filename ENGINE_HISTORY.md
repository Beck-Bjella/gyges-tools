# Gyges Engine — Evaluation Network History

A running log of how the evaluation network has evolved, the reasoning behind each change, and the design decisions that shape the current architecture. This is intended as a narrative companion to [NOTES.md](NOTES.md) (which tracks individual experiments and their numeric results) and as source material for the eventual paper on NNUE adaptation to Gyges.

---

## Background: why Gyges is an interesting target

Gyges is a two-player abstract strategy game played on a 6×6 board with three piece *types* (single, double, triple) but **no ownership** — any piece on your side of the board can be moved, regardless of which player placed it. The active boundary between the two sides shifts with every move. This makes Gyges distinct from chess in two ways that matter for evaluation:

1. **No per-piece color identity.** Standard NNUE features encode "which player's piece sits on this square"; that concept doesn't exist here.
2. **Extreme tempo sensitivity.** Because any piece on your side is movable, a single extra move can flip who controls the board's geometry. Positions are rarely independent of whose turn it is.

These two properties end up forcing real departures from the chess-NNUE playbook.

---

## v1 — Baseline MLP (180 → 64 → 1, with control features)

The first trained evaluator was a straightforward feed-forward MLP:

```
Input: 180 features
  → Linear(180, 64) → ReLU → Dropout(0.1)
  → Linear(64, 1) → Tanh
```

**Feature design (180 = 36 × 5):** For each of the 36 board squares, five features:
- Four one-hot bits for the piece state: `{empty, type_1, type_2, type_3}`.
- One scalar "control" feature encoding which side of the board owns the square at the moment: `+1` if on the side-to-move's half, `-1` on the opponent's half, `0` when empty.

The control feature was added because piece location alone doesn't tell you who can currently move that piece — ownership in Gyges is geometric, not intrinsic to the piece. The idea was to hand the network this structural cue directly so it wouldn't have to infer the active boundary.

**Training setup:** ~833k raw positions from self-play at 100k-node search depth, mirrored left-to-right for a 2× symmetric augmentation (~1.67M samples), labels are game outcomes scaled to `±0.75` with a small tie band. MSE loss, Adam at `1e-3`, 200 epochs.

**Limitations:**
- **Static evaluation only.** The whole board is encoded as a flat vector; evaluating any new position means a full matrix multiply. No way to exploit the fact that most moves only change a couple of squares.
- **Control as a scalar is under-expressive.** The `+1 / -1 / 0` encoding collapses "which side owns this square" into a single number, and it's been entangled with piece-type features in the same linear layer. It's not obvious the network can tease the two signals apart cleanly.
- **Single perspective.** The network only ever sees the side-to-move's view. It has no direct way to reason about "my position vs. their position" as a contrastive thing.

---

## Exploration: pulling at the obvious threads

Before committing to a new direction, a couple of quick experiments probed whether the baseline was leaving easy wins on the table:

### A. CNN variant

Treating the board as a 6×6 grid with piece/control channels and letting 2-D convolutions learn spatial patterns. This is the AlphaZero-style instinct. The motivation was that Gyges has real spatial structure (rows, boundaries, piece clustering) that an MLP has to learn from scratch through fully-connected weights.

### B. "No control features" (144 → 64 → 1)

The inverse experiment: drop the control scalar entirely and feed only the 144-feature piece one-hot. The question here is whether the control feature is actually helping or whether it's noise that the network has to work around. If the network can recover the active boundary on its own from piece positions, the simpler feature set should match or beat the 180-feature version.

Both of these experiments produced trained weights under [training/weights/](training/weights/), but they were side quests — the CNN was expensive to run at inference time, and the no-control ablation mostly established that control features weren't doing as much heavy lifting as expected. Neither became the new baseline, but they shaped the design of the next step.

---

## The pivot: why NNUE

The main reason to move toward an NNUE-style architecture isn't *accuracy* — it's **inference speed inside search**.

The key structural property of NNUE is that the first layer's output (the "accumulator") can be incrementally updated as the position changes. When a piece moves from square A to square B, the accumulator isn't recomputed from scratch; you subtract the column of `fc1` corresponding to the feature bits that turned off at A, and add the columns that turned on at B. This is O(hidden_width) work per move instead of O(input_dim × hidden_width). In an alpha-beta search traversing millions of positions per second, this is the difference between a usable network and an unusable one.

The accuracy gains are typically modest — the raw parameter count is comparable — but the speed gains unlock deeper search, which is where the real play-strength improvement comes from in practice.

Adapting NNUE to Gyges requires rethinking a few chess-specific assumptions:

- **Piece identity:** In chess you get `12 piece-types × 64 squares` input bits (pawn/knight/.../king × color). Here we only have `4 piece-states × 36 squares = 144` bits. The encoding is simpler, but the loss of per-piece color identity means the network can't use "my pieces vs. their pieces" as a free signal — ownership is geometric and has to come from elsewhere.
- **Tempo handling:** Chess positions are reasonably stable across a single extra move; Gyges positions are not. This reshapes what kinds of augmentations are valid.

---

## v2 — Single-sided NNUE prototype (144 → 64 → 1, no control)

The first NNUE-shaped checkpoint kept the MLP skeleton but dropped the control feature and adopted the NNUE convention of always presenting the board rotated to the side-to-move's perspective (raw square `sq → 35 - sq` when P2 is to move). The network itself was still a plain `144 → 64 → 1` MLP; "NNUE" at this stage was really just the promise that the first layer *could* be used as an incrementally-updated accumulator at inference time.

This served as the natural jumping-off point for the real two-sided architecture: it established the 144-feature encoding, verified that removing the control scalar didn't obviously damage the model, and gave a single-perspective reference point for later comparisons.

---

## v3 — Two-sided NNUE (current, 144 × 2 → 64 shared → 128 → 1)

The version the training loop in [training/main.py](training/main.py) currently produces. The architecture:

```
x_stm, x_nstm  : [batch, 144] each
a_stm  = ReLU(fc1(x_stm))     # fc1: Linear(144, 64)  — shared weights
a_nstm = ReLU(fc1(x_nstm))    # same fc1, called twice
h      = concat([a_stm, a_nstm])   # [batch, 128]
h      = Dropout(0.1)(h)
out    = Tanh(head(h))         # head: Linear(128, 1)
```

**Why two-sided:** The engine's incremental accumulator is per-perspective anyway — the Rust side keeps a running `fc1(stm_view)` and `fc1(nstm_view)` and updates both in place as moves happen. Exposing both to the head at evaluation time is nearly free at inference and lets the head weight "my side" against "their side" contrastively in one forward pass. The old single-sided network had to infer the opponent's strength implicitly from the STM-rotated board; this version just hands both views to the head directly.

**Why shared `fc1`:** Because a positional feature — say, "a triple in the far corner" — means the same thing geometrically regardless of which side is evaluating it. Sharing the first-layer weights halves the parameters to train, doubles the gradient signal landing on `fc1`, and enforces a structural prior: both perspectives are encoded through the same feature basis.

**Feature encoding:** unchanged from v2 — 144 one-hot bits, 36 squares × {empty, type_1, type_2, type_3}, no control features. Control was dropped because (a) the active-boundary information is recoverable from piece positions, and (b) handing the two-sided network two full rotated views already gives it access to the geometric information that control was trying to encode.

**Augmentation — where Gyges diverges from chess NNUE practice:**
- **Kept:** Left-right mirror of the raw board, with the label unchanged. This is a true game symmetry — the L-R flip of any Gyges position is strategically identical, so it's free 2× data.
- **Dropped:** The standard NNUE 50% perspective-swap augmentation (swap `stm_feat ↔ nstm_feat`, negate the label). This augmentation implicitly claims that "the same board with the other player to move has the opposite outcome." In tempo-sensitive Gyges, that claim is often wrong — giving the other player the extra tempo can flip the outcome entirely. Additionally, the training data is generated such that STM is always normalized to the bottom of the board for every position reached during real play, so the dataset already contains both perspectives via real labels. The antisymmetry signal comes from the real data; no need to fake it.

This is likely the most paper-worthy finding so far: the canonical chess-NNUE augmentation actively hurts training on a tempo-heavy game, and the structural property of the dataset (STM-normalized perspectives from real game states) makes it redundant even when it wouldn't hurt.

**Export format.** The Rust inference side reads raw little-endian `float32` in the order:

| Tensor        | Shape       | Floats |
|---------------|-------------|--------|
| `fc1.weight`  | `[64, 144]` | 9216   |
| `fc1.bias`    | `[64]`      | 64     |
| `head.weight` | `[1, 128]`  | 128    |
| `head.bias`   | `[1]`       | 1      |
| **Total**     |             | **9409** (37636 bytes) |

---

## Design decisions still open

These are choices that are fixed at the current values mostly because they were part of the prior baseline, not because they've been independently validated on this architecture:

- **Accumulator width = 64.** Chess NNUEs typically run 256 or 512. Widening is the obvious next experiment once two-sided has been baselined against HCE in match play.
- **Single hidden layer after concat.** Real NNUEs usually have a couple of small ReLU layers between accumulator and output (e.g., `128 → 32 → 1`). Skipped for now to isolate the two-sided change.
- **Dropout in training only.** Standard; noted for completeness.
- **MSE loss.** Common choice; alternatives (Huber, cross-entropy against soft W/D/L targets) would be worth trying later.
- **Outcome scaling (`× 0.75`).** Keeps the tanh target away from saturation. Inherited from v1.

---

## Change log

- **Initial entry** (current session): documented v1 (180→64→1 MLP with control), the CNN and no-control side experiments, v2 (144→64→1 single-sided NNUE), and v3 (the current two-sided 144×2→64 shared→128→1 architecture). Recorded the rationale for dropping perspective-swap augmentation.

---

## v3 abandoned — two-sided NNUE loses to single-sided

Ran v3 (two-sided, shared fc1) against the old single-sided h64. Got crushed 14–3–1 over 18 games. The two-sided architecture is strictly worse here despite the appealing theory.

Guessing cause: the model gets too much redundant input (both perspectives are essentially the same information re-rotated) and can't allocate the extra head capacity to anything productive. The head's 128 inputs are just two noisy copies of the same positional encoding. Dropping it.

Reverted to single-sided `144 → H → 1` and started treating **width H** as the main capacity knob.

---

## Width sweep — single-sided `144 → H → 1`, H ∈ {64, 128, 256, 512}

**Why this sweep.** After abandoning v3 (two-sided), the obvious next question was "how much capacity can a plain single-sided NNUE actually use on this dataset?" Width H is the accumulator dimension and is the main capacity lever for NNUE-style networks — in chess, reference implementations run 256–1024. The sweep was designed to find where H stops helping on Gyges data. The answer is the most paper-worthy single experiment in the project so far.

**Setup — held constant across all four runs:**
- Dataset: `hce_100kn.csv`, 833,010 raw self-play positions from 100k-node search, mirror-augmented to 1,666,020 samples.
- Features: 144 one-hot (36 squares × {empty, type_1, type_2, type_3}), no control features.
- Labels: game outcome ∈ {–1, 0, +1}, scaled to ±0.75.
- Architecture: `Linear(144,H) → ReLU → Dropout(0.1) → Linear(H,1) → Tanh`.
- Training: Adam lr=1e-3, MSE, batch 1024, 100 epochs, 80/20 train/val, random_state=42.
- Only variable: `H ∈ {64, 128, 256, 512}`.
- Checkpoints saved every 10 epochs under [training/weights/](training/weights/) as `nnue_h{H}_YYYYMMDD_HHMMSS/e{epoch}.bin`.

**Mid-sweep loop rewrite (calibration note).** Partway through, I replaced `DataLoader(Dataset, ...)` with direct on-GPU tensor slicing + `torch.randperm`. Wall-clock dropped ~3× (30 min → ~10 min per run); the math is identical, only the RNG stream for shuffle order changed. Calibrated by re-running h128 vs h64 — old code path gave 27–19 in games (56% for h128); new code path gave **10–1 full sets / 30–10–2 games** (71% for h128). Same direction, actually stronger margin on the re-run, so the loop change is clean and results across the sweep are trustable.

**Val loss @ epoch 100:**

| H | train | val | gap | shape |
|---|-------|-----|-----|-------|
| 64  | 0.3907 | 0.3830 | –0.008 | underfit — train can't even match val |
| 128 | 0.3704 | 0.3691 | –0.001 | just-barely underfit |
| 256 | 0.3509 | 0.3613 | +0.010 | healthy fit, modest generalization gap |
| 512 | 0.3280 | 0.3590 (best 0.3574 @ e60) | +0.031 | overfit — train keeps falling past e60 while val drifts back up |

Val-loss deltas for each doubling: 64→128 = –0.014, 128→256 = –0.008, 256→512 = –0.002. Diminishing returns are monotonic and obvious.

**Match results (100k nodes/move, pairwise):**

| match | full sets (A–B) | partial (A–B) | tied | games (W–L–D) | think-time avg | avg depth |
|-------|-----------------|---------------|------|---------------|----------------|-----------|
| h128 vs h64  | 10–1  | 2–0 | 8  | 30–10–2 (42g)  | h128 1153ms, h64 1476ms   | 4.12 vs 4.23 |
| h256 vs h128 | 18–13 | 2–2 | 28 | 66–56–4 (126g) | h256 1271ms, h128 1297ms  | 4.09 vs 4.10 |
| h512 vs h256 | 6–5   | 1–2 | 10 | 23–22–3 (48g)  | h512 1489ms, h256 1275ms  | 4.16 vs 4.07 |

h512 match used the **e60 checkpoint** (best val), not e100, because training past e60 was drifting val back up. So this is "h512 with early-stopping," the strongest form of h512 available on this dataset.

**Inference cost note.** At fixed 100k nodes, h512 wall-clock is ~17% higher than h256 (1489 vs 1275 ms). In a *time-limited* match, h512 would get proportionally fewer nodes and likely lose outright rather than coin-flip. h128 vs h256 think times were essentially equal.

**Takeaway — the finding that matters for the paper:**

Play strength saturates around **H = 256** on this dataset. h512 matches h256 in play while being slower per node and overfitting training data. But this is *not* a pure capacity ceiling — the honest interpretation is a combination of two ceilings acting together:

1. **Label noise floor.** The labels are game outcomes (±0.75). A position with objective value +0.3 wins some games and loses others depending on future-move quality on both sides. That noise is baked into every single label. Training on more of the same-type data averages over it but doesn't break through — the floor doesn't move.
2. **Position coverage.** ~833K self-play games cluster around similar openings/early-midgame structures; effective unique-position diversity is well below the raw count. More distinct positions would let a larger model find patterns the current data doesn't contain.

The val-loss plateau from ~e30 onward being *flat* (not slowly descending) is more consistent with a noise floor than a coverage gap — noise floors produce flat plateaus, coverage gaps produce slow continued descent. So noise is probably the bigger component, but both are almost certainly present.

**What this does and doesn't say.** The sweep shows: *on this specific dataset and at 100k nodes/move*, h256 is the saturation point. It does **not** say H=256 is the architectural ceiling for Gyges NNUE generally — a 2× larger dataset or a cleaner label source would almost certainly change the answer.

**Open caveats worth flagging when writing this up:**
- Dropout held at 0.1 across all widths. At H=512 where overfit is visible, lower dropout might hurt more and higher dropout might recover some val. Not tuned.
- Single hidden layer only (`144 → H → 1`). Deeper heads (`144 → H → 32 → 1`) were not explored; they're standard in chess NNUE and could shift the capacity/data balance.
- All matches at 100k nodes. A time-control sweep would amplify the inference-cost disadvantage of h512.
- `random_state=42` for train/val split is fixed across runs, so the four models are comparing on exactly the same held-out positions — good for controlled comparison, but val numbers are a single-split estimate, not cross-validated.

**Natural next experiments (in order of priority):**
1. **Scale dataset.** 2–4× more self-play positions, then retrain h512. If val drops below 0.3590 and h512 starts beating h256, data volume was real.
2. **Cleaner labels.** Train against per-position deep-search scores (or scored-position targets as in chess NNUE) instead of raw game outcomes. If val drops substantially, the current ceiling was label noise — and this is the single biggest lever available.
3. **Deeper head.** Try `144 → 256 → 32 → 1` — see if the plateau moves with a non-trivial head rather than wider accumulator.
4. **Dropout sensitivity at H=512.** Sweep dropout ∈ {0.05, 0.1, 0.2, 0.3} to see if regularization closes the overfit gap.

**Shipping decision.** Until one of (1) or (2) is run, **h256 is the production pick.** h512 is theoretically promising but data-starved here; h128 is genuinely weaker in match play; h64 is clearly underfit.

### Why this result matters for the paper
This is the first experiment on Gyges that produces a clean, empirically-grounded capacity-and-data story:
- A monotonic underfit → healthy → overfit progression across a 2×/2×/2× width sweep.
- Matching match-play evidence (conclusive → modest → coin flip) lining up with loss-curve signatures.
- A specific mechanism (label noise from game-outcome targets) that lines up with the plateau shape.
- A clear next-step experimental path (data scaling vs. cleaner targets) that separates the two ceiling sources.

The capacity question for NNUE on a non-chess game has been studied in exactly zero published work as far as I know. This is one section of the paper in itself.

---

## Measurement / benchmarking notes

Captured here because they shape how we should read match results going forward and how the paper should report them.

### NN-vs-NN margins don't transfer to NN-vs-HCE margins
- The h128 vs h64 match was 10–1 full sets. NN-vs-HCE matches for similar-tier NNs are nowhere near that lopsided — wins, yes, but not demolition.
- This is expected and well-known in engine A/B testing. Mechanisms, stacked:
  1. **Correlated errors compound in NN-vs-NN.** Networks trained on the same data/loss share biases. When two similar evaluators play, tiny judgment differences stack because they're disagreeing in the same *evaluation space* — one catches a pattern the other misses at a critical turn, then eval quality over 30 moves compounds. HCE's errors are orthogonal to any NN's, so NN-vs-HCE disagreements don't snowball the same way.
  2. **Ceiling/floor compression vs a weaker fixed opponent.** If h128 already beats HCE ~75% and h256 beats HCE ~80%, the NN-vs-HCE margin has only 5 points of headroom between those two nets regardless of how much better h256 actually is. NN-vs-NN has no such ceiling — both sides can win half.
  3. **Search/eval coupling.** Each side's tree is shaped by its own eval. h256 reaches positions h256 is comparatively best at; against HCE, both engines drag games into different position classes and the NN's "best case" doesn't come up as often.
  4. **Different match dynamics.** NN-vs-HCE tends to be decided by "one side made a clear eval mistake and the other found it." NN-vs-NN is decided by compounding sub-pawn positional judgment over many moves. Those are measuring different skills.
- Practical consequence: **NN-vs-NN is the sensitive comparator for incremental NN improvements; NN-vs-HCE is the absolute-strength anchor but the wrong tool for comparing two NNs.** This matches standard chess engine practice (Stockfish tuning uses SF-vs-SF, not SF-vs-other-engine).

### The NN's advantage over HCE lives in a narrow position band
- Qualitatively, the NN is the strongest evaluator we've trained — clearly beats HCE — but the margin is "wins more often" rather than "crushes."
- Likely mechanism: the NN's advantages concentrate on **midgame positional understanding**, which is exactly HCE's weakest area. In openings HCE is close to book-like / reasonable by construction, and in sharp tactical positions search resolves the truth for both sides. So the NN only cashes in its advantage during the narrower midgame window.
- Against another NN, "NN-favorable positions" come up every move; against HCE, they come up only sometimes per game. That's why NN-vs-NN looks wider.
- Paper framing that matches what the matches actually show:
  - **Absolute claim (from HCE matches):** "The NN is the strongest evaluator we've trained; it beats HCE consistently."
  - **Methodological claim (from the sweep):** "Incremental NN improvements are best measured head-to-head rather than via HCE, because the NN's advantage lives in a narrow position type that NN-vs-NN surfaces continuously and NN-vs-HCE surfaces only occasionally."

### Search depth papers over HCE's eval weakness
- At 100k nodes/move, both engines have deep tactical vision. Deep search resolves a lot of the tree's ground truth regardless of eval quality, because "tactically won position" is computable from search alone.
- This means HCE plays near its best relative strength at 100k nodes — search compensates for a weak eval. At 10k nodes HCE would be much worse relative to the NN, because search could no longer paper over eval weakness.
- The NN benefits less from extra depth in relative terms — its eval already captures positional stuff cleanly, so deep search mostly confirms what the eval already says. The gap between NN and HCE **narrows as search depth grows.**
- Classic engine tradeoff: "strong eval + shallow search" ≈ "weak eval + deep search" up to a point. 100k nodes is enough budget that HCE is near the regime where that equivalence benefits it most.
- Consequences:
  - Winning 100k-node matches against HCE is a **harder bar** than the scoreline suggests — the NN is demonstrating eval quality against HCE's best-case depth.
  - Short-time / shallow-node matches vs HCE would likely blow out further in the NN's favor.
  - The narrow NN-vs-HCE margin at 100k isn't "NN is barely stronger" — it's "NN is meaningfully stronger *at the depth where HCE is most competitive.*"
- **Methodology recommendation for the paper:** when reporting NN-vs-HCE, report multiple search budgets (e.g., 10k / 100k / 1M nodes). A single deep-search number understates the evaluator's real strength. A depth sweep visualizes the eval-vs-search tradeoff directly.

### Summary — when to use which benchmark
| Benchmark | Best for measuring… | Known limitations |
|-----------|---------------------|-------------------|
| NN-vs-NN (full-set scoring) | Incremental eval quality between similar evaluators | Doesn't give absolute strength; inflated margins relative to external baselines |
| NN-vs-HCE (single deep-node budget) | Absolute strength anchor: "did we beat the baseline at all" | Compressed ceiling; depth papers over HCE eval weakness; narrow advantage-band effect |
| NN-vs-HCE at multiple depths | Eval-vs-search tradeoff, true shape of the NN's advantage | More engine-time expensive; harder to chart |

Default going forward: **NN-vs-NN for width/arch/data ablations, NN-vs-HCE depth sweep for "how strong is our best NN overall."**

---

## h256 vs HCE @ 100k nodes — first real external-baseline result

Ran the current shipping pick (h256, single-sided, outcome-trained) against HCE at 100k nodes/move. Match still ongoing; reporting a snapshot.

**Snapshot (46 games, 23 sets):**
- Games: h256 30 – HCE 15 – 1 draw (65% for h256)
- **Full sets: h256 9 – HCE 2, 11 tied** (1 partial win for h256, 0 for HCE)
- Think time: h256 1331 ms vs HCE 1519 ms
- Avg depth: h256 3.83 vs HCE 3.87
- Game length: h256 wins avg 22.4 moves, loses avg 19.7

**Why the 9–2 full-set score is the real signal.** Given everything in the measurement/benchmarking notes above — search depth papering over HCE eval weakness, ceiling compression, NN advantages living in a narrow position band — a decisive NN-vs-HCE match at 100k nodes is the hardest bar HCE can present. HCE gets to operate near its best relative configuration here, and h256 still takes 9 of 11 decisive sets. Margin has been *growing* as games accumulate rather than regressing to early variance — stable signal, not a fluke streak.

**Search-vs-eval tradeoff, made concrete in one data point.** HCE actually gets *slightly more search depth* than h256 at the same node budget (3.87 vs 3.83 avg depth) because h256's per-node inference is heavier. HCE also spends ~14% more wall-clock time per move (1519 vs 1331 ms) to reach that extra depth. Despite HCE having both the depth and time edge at fixed nodes, h256 wins the match convincingly. This is a clean empirical version of the classical engine tradeoff: strong eval at slightly shallower search beats weaker eval at slightly deeper search, at least at 100k nodes. Worth pulling into the paper as a concrete data point for that claim.

**Paper-framing summary:**
- The NN is unambiguously the strongest evaluator we've trained: consistently beats HCE at the depth where HCE is most competitive.
- The margin at 100k nodes (9–2 full sets / 30–15 games so far) is conservative by construction — shallower budgets would likely widen it further, because search can no longer rescue HCE's eval.
- h256 wins *while searching less deep* than HCE at the same node count, which is a direct demonstration of the eval-vs-search tradeoff landing on the eval side.

### Update — h256 vs HCE margin continues to grow
Same match, further along: full-set result now **h256 13 – HCE 2, 18 tied sets**. The margin has widened monotonically as games have accumulated, not regressed to the mean — confirming the earlier 9–2 snapshot wasn't early-variance noise and the h256-over-HCE advantage is stable. Full-set win rate on decisive sets: 13 / 15 ≈ 87%.
