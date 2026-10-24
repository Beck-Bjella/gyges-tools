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

---

## Deeper head — `144 → 256 → 32 → 1` (h256x32)

**Why this experiment.** The width sweep saturated around H=256 on this dataset. Open question from the sweep's "next experiments" list: does the plateau move if instead of widening the accumulator, you add a non-trivial nonlinear head? The flat `144 → 256 → 1` is structurally close to a generalized linear model — every feature has to carry its weight in isolation, with no way to combine "piece A at sq X *together with* piece B at sq Y." Stacking a small `→32→1` ReLU stage gives the network a place to compose features. This is also the canonical chess-NNUE shape (wide accumulator + small dense tail), so it's the most natural architectural axis to vary next.

**Setup.** Identical to the width sweep except for the architecture: `Linear(144,256) → ReLU → Dropout(0.1) → Linear(256,32) → ReLU → Linear(32,1) → Tanh`. Same dataset (`hce_100kn.csv`, 1.67M w/ mirror), same Adam lr=1e-3, MSE, batch 1024, 100 epochs, 80/20 split, random_state=42, checkpoints every 10 epochs.

**Training result:** **best val 0.3546 @ e50** (vs 0.3613 @ e100 for flat h256 — ~2% lower val loss). Train kept descending past e50 while val drifted back up, same overfit signature as h512. Picked **e50 checkpoint** for matches following the same best-val-checkpoint rule used for h512 (e60). Export layout: `fc1.weight [256,144] + fc1.bias [256] + fc2.weight [32,256] + fc2.bias [32] + fc3.weight [1,32] + fc3.bias [1]` = 45,377 floats / 181,508 bytes.

**Match — h256x32 (e50) vs h256 (e100), 100k nodes/move:**

| stage | full sets (x32–256) | partial (x32–256) | tied | games (x32–256–D) | think-time avg | avg depth |
|-------|--------------------|--------------------|------|---------------------|----------------|-----------|
| early snapshot (42g)  | 7–4   | 2–0 | 8  | 24–16–2  | x32 1668ms, 256 1424ms | 4.22 vs 4.25 |
| **final (298g)**      | **43–21** | 3–3 | 79 | **168–124–6** | x32 1611ms, 256 1389ms | 4.18 vs 4.16 |

**Statistical read.** 56.4% raw on N=298 → 2σ band at this N is ±5.8%, so the result just clears statistical significance. Full-set ratio 43–21 = 2.05:1, or 67% of decisive sets — solidly significant at the set level. Tied sets are 53% (consistent with both engines being strong, edge being modest). The training-loss delta (~2%) and the play-strength delta line up almost exactly.

**Inference cost.** x32 spends ~16% more time per move (1611 vs 1389 ms) for essentially the same depth (4.18 vs 4.16). The extra layer has real per-eval cost. Two implications:
- In a *time-limited* match the edge would shrink — x32 would get fewer searched nodes per move.
- A `256 → 16 → 1` (or `256 → 8 → 1`) tail might be net stronger overall: keeps the architectural-composition benefit while costing less per inference. Worth testing.

**Takeaway.** Adding a small nonlinear head to the saturated h256 accumulator produced a real (modest) play-strength gain that matches its val-loss gain almost exactly. The plateau *did* move slightly with depth-in-head, not just width-in-accumulator — partial evidence that the H=256 saturation was at least partly structural (the flat head couldn't use further capacity productively), not purely data-noise-floor. But the move is small enough that the noise-floor explanation from the width sweep still holds for the bulk of the plateau.

**Open from this entry:**
- Smaller tail sizes (`256 → 16 → 1`, `256 → 8 → 1`) — likely the right next architectural axis.
- Two-stage tail (`256 → 32 → 32 → 1`) — full canonical NNUE shape.
- Time-controlled rematch (not just node-controlled) to measure the real-strength-vs-inference-cost tradeoff directly.
- h256x32 vs HCE — open question whether the gain over flat h256 transfers proportionally to the HCE matchup, or compresses (NN-vs-HCE has narrower dynamic range, see measurement notes above).

---

## Note (2026-04-23) — existing 100n dataset is symmetric-starts, not independent

While prepping a fresh data-collection run with `gen_board_independent()`, dug into when the swap from `gen_board()` happened. Found in `git log`: commit `ff89070` (2026-04-17 12:56) introduced `gen_board_independent` and swapped both data-gen call sites. The current `hce_100kn.csv` was modified Apr 17 at 14:20 — only 84 minutes later, and birth time matches modify time, suggesting the file was *transferred* (likely from AWS) rather than written-in-place. NOTES.md had said "Mirrored" for starting lines, ambiguously.

Confirmed via direct inspection: ran a check on the first 500 and last 500 rows looking for "starting positions" (only back ranks occupied, middle empty). All 10 examples checked (drawn from head and tail) satisfy `back[i] == front[5-i]` exactly — i.e. perfect rotational mirrors. Independent generation would produce this with probability ~1/720 per position; ~30 in a row is statistical impossibility.

Conclusion: the entire 833k dataset is from `gen_board()` (symmetric starts). Both AWS and local portions used pre-`ff89070` code. The val ceiling at 0.352 we've been hitting was on this restricted opening distribution.

**Implications:**
- Tonight's overnight run on `gen_board_independent` will produce a distributionally different dataset, not just a bigger one.
- "More data" and "wider opening distribution" are now coupled — can't separate them with what we have.
- For the paper, want to be clear: pair-encoding ceiling at 0.352 was measured on symmetric-only data.

NOTES.md updated to reflect "Symmetric (gen_board)" with timestamp pointer back to here.

---

## Paper framing crystallization (2026-04-24) — "hybrid NNUE/CNN with W/L labels"

Spent an evening bouncing around architecture/data experiments and the paper's contribution came into focus. Writing it down so I can read it back later.

**The core realization:** this project isn't just "NNUE adaptation for Gyges." It's a methodology paper for a neglected gap between classical NNUE and AlphaZero-style approaches. Specifically:

- **Classical NNUE** (Stockfish) needs a strong-eval oracle for training labels. In chess, Stockfish's own HCE provides these at great depth. In Gyges, nothing produces good mid-game scores — not my HCE, not prior NNs, not anything. So you can't bootstrap an NNUE the Stockfish way because there's no oracle to train against.
- **AlphaZero CNN + MCTS** is overkill for Gyges. State space is small (6×6 board, 12 pieces), doesn't justify the compute. And CNN inference is slow for search — dense convolutions can't be incrementally updated when one piece moves.
- **Flat NNUE** ("piece at square" features, the simplest adaptation) doesn't capture enough positional structure. Gyges positional play is about piece *relationships* (walls, 1s backed by 2s/3s), not raw piece placement.

So there's a gap: games with meaningful positional relationships that are too small for AlphaZero and too oracle-less for Stockfish-NNUE. Gyges sits in that gap. So do Arimaa, Lines of Action, Tafl variants, many modern abstract strategy games.

**My hybrid answers both sides of the gap:**

1. **NNUE's sparse accumulator** for inference speed (CNN can't do this — that's why it's slow)
2. **Hand-engineered pair features** for CNN-like relational inductive bias (CNN would learn these from scratch; I encode them directly, trading learning flexibility for inference compatibility)
3. **W/L outcome labels** to bypass the eval-oracle dependency (AlphaZero-style label generation, without the MCTS)

Each component fills a specific gap. The combination is the contribution.

**Example abstract** (for where to aim the paper):

> "For abstract strategy games where classical NNUE requires an unavailable strong-eval oracle and AlphaZero-style approaches are computationally excessive, we propose a hybrid evaluator combining NNUE's sparse-accumulator inference with hand-engineered relational pair features trained on outcome labels. We demonstrate the methodology on Gyges, a 6×6 abstract strategy game with no piece ownership and strong tempo sensitivity. Our approach outperforms both flat-NNUE baselines and pure-outcome CNN architectures at the same data scale, while maintaining millions-of-evals-per-second inference compatibility. We argue this hybrid applies broadly to a class of abstract strategy games previously underserved by existing neural evaluator methodologies."

**Why this framing is stronger than what I was pitching before:**
- Target audience broadens from Gyges enthusiasts to abstract-strategy-game researchers in general
- Gyges becomes a case study demonstrating the methodology, not the main contribution
- Provides a reusable design recipe: identify domain relationships → encode as pair features → use NNUE architecture → train on W/L
- Opens clear follow-up work: which other games does this apply to? how far does it scale?

**Key scientific questions the paper needs to answer:**
1. Does pair encoding *scale better with data* than flat encoding (steeper slope, not just lower point)?
2. How does label quality (node count of generating games) trade off against label quantity?
3. What's the inference cost relative to CNN alternatives at comparable strength?

**Status as of today:** flat-6M (on 5k-node data) just crushed flat-833k (on 100k-node data) in matches ~65% to 35%. Confirms volume matters for flat. Pair-h64 on the same 6M is training now; that match (pair-6M vs flat-6M) is the linchpin experiment for the paper's encoding claim.

Memory file `project_paper_goal.md` updated with this framing in full detail.

---

## Design rationale — why each hybrid component is motivated by a specific Gyges property (2026-04-24)

Talking through why standard techniques fail on Gyges and how each part of the hybrid is a response to a specific failure. This is the "Method motivation" section for the paper — writing it down so I don't lose the chain of reasoning.

**Gyges structural properties that break standard approaches:**

1. **Complex move space, hundreds of options per turn, path-based.** Moves involve piece bouncing off other pieces, pathing through squares, multi-step trajectories. Move generation itself is expensive. Branching factor is high (~50-100 at typical mid-game).

2. **No piece ownership.** Any piece on your side is movable, regardless of who placed it originally. There's no "my pieces" vs "opponent's pieces." This immediately breaks HalfKP-style features that encode "my king position + opponent's piece at square X."

3. **Single moves can massively change board state.** Because of the bouncing mechanic, one move can relocate multiple pieces or cross half the board. Unlike chess where a move is a small perturbation, Gyges moves are high-magnitude. A 1-ply look-ahead shows a dramatically different board.

4. **No quiet positions — polarity always flips.** Every move threatens to change the balance. There's no "this position is stable, evaluate it statically and move on" — the next move could flip the position's evaluation entirely.

5. **No strong eval oracle.** My HCE and prior NNs don't produce reliable mid-game scores. Neither does anything else. So there's no Stockfish-style scaffold to train on.

**What each standard approach does and why it fails:**

- **Classical NNUE (Stockfish-NNUE):** Requires strong eval labels from a deep-search oracle. Gyges has no such oracle. **Fails: can't train.**
- **AlphaZero (CNN + MCTS):** Requires compute proportional to state space + MCTS infrastructure. 6×6 board doesn't justify it. CNN inference is slow per eval (can't be incrementally updated). **Fails: overkill + slow.**
- **Alpha-beta with HCE:** Branching factor crushes depth. Static eval is noisy because moves are high-magnitude — scores at the top of search are all similar, pruning is weakened. **Fails: weak at depth.**
- **Quiescence search:** No quiet positions exist. Polarity flips every move. Can't extend search at "stable" points because there are none. **Fails: assumption doesn't hold.**
- **Policy networks:** Move space is structural/pathing, not classifiable. Even if policy was learned, alpha-beta deep search beats one-ply policy selection — the strategy IS "search deep, don't lose." **Fails: doesn't capture the actual winning strategy.**
- **Flat NNUE (144 features, piece-at-square):** Doesn't capture relational structure. Same type-3 at different positions relative to neighboring pieces means very different things. Features are too coarse. **Fails: not expressive enough for structural play.**

**How each hybrid component addresses a specific failure:**

| Game property | Standard technique that fails | Hybrid component that fixes it |
|---|---|---|
| High branching + path-based moves | Alpha-beta pruning weakened; need many nodes/sec | NNUE's sparse accumulator → millions of evals/sec |
| No piece ownership | HalfKP-style features don't apply | Pair features encode structural relationships, not control |
| High-magnitude moves | Static eval noisy, similar scores at top | Rich pair features reduce eval noise via relational context |
| No quiet positions | Q-search impossible | Fast eval enables deeper full-width search instead |
| No strong eval oracle | Classical NNUE training doesn't work | W/L outcome labels from self-play |
| Policy nets don't capture moves | Can't shortcut search | Fast eval means alpha-beta can search deep enough anyway |

Every component is load-bearing for a specific reason. Nothing is there because "it sounded cool." That's what makes this a *designed* methodology rather than a collection of techniques.

**Why this generalizes to the target class:**

These same properties appear in other abstract strategy games:
- **Arimaa:** high branching (17² typical moves), pieces can be pushed (unstable control), structural eval matters (trap positioning)
- **Lines of Action:** structure-based win condition (connectivity), high branching, hard static eval
- **Tafl variants:** asymmetric pieces but structure/positioning dominates
- **Octi:** piece-movement with directional arms, positional play, no clear eval heuristics

Each of these breaks the same standard techniques for the same structural reasons. The methodology applies because the *design rationale* applies — not just because the games happen to look similar.

**This is the "design motivation" section of the paper written out.** When the paper gets written, unpack each row of the table above with a paragraph explaining the game property, the technique it breaks, and how the hybrid component addresses it.

---

## Match results — flat-10k vs HCE & flat-5k-6M comparisons (2026-04-24)

Logging the day's match results that actually informed strategy. All matches at 100k nodes/move, time-cap 120s, no randomization.

**Flat-h256 trained on 6.1M positions of 5k-node data vs flat-h256 trained on 833k positions of 100k-node data:**
- 87 wins (50.6%) – 74 wins (43.0%) – 11 draws (6.4%), 172 games
- Set results: 19 full + 5 partial vs 14 full + 2 partial, 46 tied (53%)
- Take: at 7× volume vs 1/20 quality ratio, **volume wins clearly** (~65% / 35% in decisive games)
- Confirms that more low-node data beats less high-node data at extreme trade-off ratios

**Flat-h256 trained on 4.5M positions of 10k-node data vs flat-h256 trained on 6.1M positions of 5k-node data:**
- 53 wins (60.2%) – 27 wins (30.7%) – 8 draws (9.1%), 88 games
- Set results: 15 full + 7 partial vs 5 full + 1 partial, 16 tied (36%)
- Take: at 1.3× volume vs 1/2 quality ratio, **quality wins clearly** (~60% / 31%)
- The trade-off curve has an inflection: extreme-volume regime favors more data, but moderate-volume regime favors better labels

**Flat-h256 (10k-trained) vs HCE @ 100k nodes:**
- 30 wins (65.2%) – 12 wins (26.1%) – 4 draws (8.7%), 46 games
- Set results: 10 full + 1 partial vs 0 full + 3 partial, 9 tied
- Take: HCE has no full-set wins. NN dominates decisive sets 11-3. Comparable or slightly better than the old 833k h256 baseline against HCE (which was 13-2 full sets / 2:1 ratio).
- **The strength chain holds against HCE.** Flat-10k does NOT regress against the external baseline despite being trained on different distribution than HCE.

**Pair-h64 + dropout 0.4 (5k-6M data, e20) vs flat-h256 (5k-6M):**
- 28 wins (45.2%) – 31 wins (50.0%) – 3 draws (4.8%), 62 games
- Set results: 6 full + 1 partial vs 7 full + 2 partial, 15 tied
- Take: dead even / pair very slight loss within statistical noise (2σ = ±12.5%)
- Caveat: pair was undertrained (val still dropping at e20), used overly aggressive dropout for the data-rich regime
- Real pair test = pair-h64 + dropout 0.2 on 10k data, not yet run

**Quality-vs-volume trade-off curve emerging from these results:**

| Comparison | Quality ratio | Volume ratio | Winner |
|---|---|---|---|
| Flat-100k-833k vs Flat-5k-6M | 1/20 | 7× | Volume (5k won 65%) |
| Flat-10k-3.7M vs Flat-5k-6M | 1/2 | 1/1.3 | Quality (10k won 60%) |

Inflection point exists somewhere between these regimes. For paper: characterize this curve as a finding — outcome-based training has a quality-volume sweet spot that depends on label-noise vs signal scaling.

**Strategic implication going forward:**

1. **10k-node data is the strongest single source for production training.** Switch all data generation to 10k.
2. **Combining datasets (5k+10k+100k) for diversity may further help** but untested. The 100k data's clean endgame labels could disproportionately improve hard-to-evaluate positions.
3. **Pair encoding's real test is pair-h64 + dropout 0.2 on 10k.** Pending. Will determine if encoding contributes additional strength beyond what data quality alone provides.

**NOTES.md updated** to reflect the new datasets (5kn, 10kn) and trained models. Match-result entries live here in ENGINE_HISTORY per the established convention.

### Update — long-run flat-10k vs HCE @ 100k nodes (794 games / 397 sets)

Same matchup as the earlier snapshot, now at high statistical power. Definitive result.

- Games: NN 551 (69.4%) – HCE 196 (24.7%) – 47 draws (5.9%)
- **Full sets: NN 192 – HCE 16, 158 tied** (192/(192+16) = 92.3% of decisive full-set wins)
- Partial sets: NN 17 – HCE 14
- Set ratio counting ties as half: NN 288 – HCE 109 = **2.64:1**
- Think time: NN 1248ms vs HCE 1403ms (NN ~11% faster per move)
- Avg depth: NN 3.96 vs HCE 4.08 (HCE searches slightly deeper at same node budget)
- Game length: NN wins avg 21.5 moves, loses avg 19.4 — NN tends to win longer, more patient games

**Compared to the old h256 (833k symmetric) baseline vs HCE (which was 2:1 ratio):**
Flat-10k vs HCE is **2.64:1 — roughly 32% stronger** in the standard ratio metric. So the data-pipeline improvements (independent starts, low-node high-volume generation) genuinely produce a meaningfully stronger evaluator against the external baseline, not just against other NNs.

**HCE has only 4% full-set wins out of 397 sets.** That's basically "HCE never wins decisively against this NN." The 158 tied sets reflect engines that play similarly enough that all positions in a set draw — the NN's edge concentrates in the positions where eval quality matters most.

**Notable: NN searches less deep AND is faster per move, yet wins this decisively.** Direct empirical demonstration of the eval-vs-search tradeoff: better eval at slightly shallower depth beats weaker eval at slightly deeper depth at this node budget.

**This is now the new "old eval baseline" for the paper.** Any further improvement (better data mix, pair encoding, etc.) gets compared against flat-10k. The bar is high: 69% raw / 92% decisive-set win rate vs HCE.

## Catch-up entries — pair-encoding scaling era (2026-04-29)

Backfilled because results were piling up undocumented. Numbers below are quoted verbatim from match-output paste; anything I don't have raw data for is flagged explicitly.

### Methodology updates landed since last entry

- **Match analyzer parallelized.** New `analyze_parallel(set_count, thread_count)` in `src/engine_analyzer.rs`: each worker owns its own engine pair, pulls set indices off an atomic counter, sends `SetResult`s back to main over an mpsc channel. Main accumulates and prints rolling summary per set. Workers print `[wN] set X done — E1 a / E2 b / draws c` lines as they finish. Single-worker mode falls back to verbose per-move printing for live watching. Test runs use 12 workers.
- **Independent starting boards.** `analyze_worker` now calls `gen_board_independent()` instead of the symmetric `gen_board()`. Results from this point forward sample the full starting-position distribution, not just the symmetric subset. *Caveat:* high tied-set rate barely moved (39.3% → 39.9% in the comparison runs), so deterministic-play convergence — not opening narrowness — is what was driving ties.
- **Mixed-format CSV loader fix.** `Dataset.from_files` was producing `i64::MIN` for missing metadata when concatenating 39-col + 37-col CSVs (the new 10kn-with-metadata files combined with legacy 100kn). Fix in `training/common.py`: load files separately, fall back to legacy 37-col mode if any file lacks metadata. Board features were always correct; only the unused move_num/game_id columns were corrupted, so prior runs that didn't use the filter knobs were unaffected.

### Pair-h1024x32 vs HCE @ 100k nodes (410 games / 205 sets — first documented pair-vs-HCE result)

Architecture: pair encoding (5886 features) → h1024 → 32 → 1, trained on `hce_10kn_combined.csv` (combined 10kn dataset, ~19.77M raw positions). Match conditions: 100k node budget, max-time 120s, no randomization, independent starting boards, parallel 12-worker.

- Games: NN 323 (78.8%) – HCE 71 (17.3%) – 16 draws (3.9%)
- **Full sets: NN 122 – HCE 2, 65 tied (31.7%)** — NN takes 122/(122+2) = 98.4% of decisive full sets
- Partial sets: NN 14 – HCE 2
- Think time: NN 6059ms vs HCE 3206ms (NN ~89% slower per move from heavier eval)
- Avg depth: NN 3.86 vs HCE 3.88 — **same depth despite 2x slower eval per move.** The eval cost is paying for itself at search time.
- Game length: NN wins avg 20.2, loses avg 20.5 — symmetric, no length-bias

**Compared to the prior "best documented baseline" (flat-h256 / 10kn vs HCE, 794 games):**

| Metric | flat-h256 vs HCE | pair-h1024x32 vs HCE |
|---|---|---|
| NN raw win rate | 69.4% | **78.8%** (+9.4 pp) |
| HCE full-set wins | 4.0% (16/397) | **1.0%** (2/205) |
| NN decisive-set share | 92.3% | **98.4%** |
| Set ratio (ties = ½) | **2.64 : 1** | **~4.6 : 1** |

So pair encoding at h1024 + tail layer is roughly 75% stronger than flat-h256 by the standard set-ratio metric, against the same external baseline. HCE practically never decisively wins against this NN.

**Caveat — the NN-vs-NN story is much narrower than the NN-vs-HCE story.** See below.

### Pair-h1024x32 vs Pair-h1024 (no tail), both on `hce_10kn_combined.csv` (486 games / 243 sets)

Architectural-isolation match. Three variables differ between the nets, not just the tail:
1. Tail layer: x32 has `Linear(1024, 32) → ReLU → Linear(32, 1)`; plain h1024 goes straight to `Linear(1024, 1)`
2. Dropout: x32 = 0.3, plain h1024 = 0.5 (per user's recollection — original h1024 dropout not logged)
3. Dataset: both trained on the same combined 10kn set; this *is* matched

So this match cleanly tells you "x32 architecture (with its co-tuned dropout) vs h1024 architecture (with its co-tuned dropout)" — not "tail vs no-tail" alone.

- Games: x32 244 (50.2%) – h1024 210 (43.2%) – 32 draws (6.6%)
- Full sets: x32 65 – h1024 51, 97 tied (39.9%)
- Partial: x32 18 – h1024 12
- Think time: x32 5535ms vs h1024 3501ms (x32 ~58% slower)
- Avg depth: x32 4.16 vs h1024 4.09 (essentially matched)

**Take:** x32 is marginally stronger (~+7 pp raw, ~14:12 in decisive full sets) but pays substantial wall-clock cost. The strength gain is real, but small relative to the cost. **At fixed node budget x32 wins; at fixed time budget x32 likely evaporates because each move spends more compute.**

**Decision: drop x32 from the active rotation, keep plain h1024 as the architectural baseline going forward.** Weights file `e5_pair_h1024x32_combined.bin` retained for paper comparison if needed.

### Plain h1024 trained on bigger combined dataset vs original `e6_pair_h1024_10k.bin` (224 games / 112 sets)

Dataset isolation experiment intent: same architecture (h1024 no tail), train on the bigger `hce_10kn_combined.csv` instead of the original smaller `hce_10kn.csv`, see if the bigger data wasn't broken / was helpful. Used dropout 0.5.

**Confound:** the original h1024_10k training didn't have its dropout value logged. User recalls 0.5 for the original but isn't certain — could have been lower. So this match is "bigger data + (maybe) different dropout" vs "smaller data + original training", not a pure data-isolation A/B.

- Games: COMBINED 131 (58.5%) – 10K 80 (35.7%) – 13 draws (5.8%)
- **Full sets: COMBINED 42 – 10K 17, 40 tied (35.7%)** — COMBINED takes 42/59 = 71% of decisive full sets, ratio ~2.47:1
- Partial: COMBINED 7 – 10K 6
- Think time / depth basically matched (4286ms / 4.10 vs 4229ms / 4.07)
- User noted result was "consistent for 50+ games" before the snapshot — direction is stable, not early-variance noise

**Take:** the new h1024 (bigger combined data, dropout 0.5) is clearly stronger than the original `e6_pair_h1024_10k`. ~2-3x in decisive sets is well outside any reasonable noise floor at this sample size.

**Conclusions, with appropriate caveats:**
1. The bigger 10k_combined data is at least non-harmful, and almost certainly helpful — this rules out the "the new 15M positions are corrupting the model" worry.
2. Whether the strength gain is from more data, the dropout change, or both is undetermined. User's pragmatic call: doesn't matter for production purposes, adopt as new baseline. Could be cleanly isolated by retraining on the smaller 10k set with dropout 0.5; not deemed worth the compute.
3. **The "h1024 is extraction-bound on the smaller 10kn data" theory from the prior era now needs revising.** More data (or better regularization) clearly does buy more strength, so the previous extraction-bound observation was specific to that training configuration, not a general ceiling on the encoding.

**This is the new "current best" replacing `e6_pair_h1024_10k`.** Going forward, "PAIR h1024 COMBINED" / "PAIR h1024" without qualifier refers to this new model.

### Pair-h1024 trained on (10k_combined + 100kn) vs Pair-h1024 trained on (10k_combined alone) — "DUO vs COMBINED" (536 games / 268 sets)

Combo experiment. Same architecture (h1024 no tail), same dropout (0.5), same training procedure — only the training data changed. Combo run added `hce_100kn.csv` (~833k positions of 100k-node-budget HCE games) to the existing `hce_10kn_combined.csv`. The 100kn rows were only **4% of the combined dataset** at this mix.

Hypothesis being tested: does the cleaner endgame labeling from 100k-node search add signal that 10k-node games miss?

- Games: DUO 256 (47.8%) – COMBINED 246 (45.9%) – 34 draws (6.3%)
- Full sets: DUO 63 – COMBINED 58, 113 tied (42.2%)
- Partial: DUO 17 – COMBINED 17
- Decisive full-set ratio: 63 / (63+58) = **52% — within statistical noise of 50/50**
- Think time / depth matched
- User noted DUO was "slightly up the whole time" — the lean is real but the magnitude is consistent with sampling variance at 502 decisive games (1σ ≈ 11 wins)

**Take:** at 4% mix, the 100kn data did not produce a measurable strength gain. Two non-mutually-exclusive interpretations:

1. **Encoding is genuinely capped** — the 100kn signal is real in principle but the network can't extract additional information from any data at this width.
2. **100kn volume too small to move the gradient** — 4% effective contribution is small enough that even valuable signal could be drowned by 96% 10kn gradient pull. Not the same conclusion as encoding-limited.

This experiment can't distinguish (1) from (2). Disambiguating requires either oversampling/weighting the 100kn data, or generating substantially more 100kn data so the proportion is healthier.

### Val curves from the two h1024 runs

Provided here so the convergence shape is on record. **Val sets differ between the two runs** — the (10k+100kn) val split contains 100kn positions the (10k_combined alone) val split never saw — so val loss numbers are NOT directly comparable. Engine match results are the comparator (above). Listing for completeness:

| Epoch | h1024 on 10k_combined | h1024 on 10k_combined + 100kn |
|---|---|---|
| 1 | train 0.4170 / val 0.4022 | train 0.4148 / val 0.4000 |
| 5 | 0.3892 / 0.3917 | 0.3877 / 0.3896 |
| 10 | 0.3754 / 0.3901 | 0.3743 / 0.3880 |
| 11 | 0.3733 / 0.3901 (val plateau) | run crashed — traceback truncated by user, not diagnosed |

Val plateau in both cases lands in the e8-e11 window. Best epochs both runs were picked by user from this range (specific epoch index of the weights actually used in the matches above is not in my data — both should be checkpoints in `training/weights/nnue_h1024_pairs_*` directories from 2026-04-29).

### Things deliberately NOT in this catch-up

- **Pair-h1024 vs HCE.** User recalls a result like "49-1" but it isn't in either ENGINE_HISTORY or NOTES — never logged. Not invented here. Direction of the result (NN dominant) is consistent with x32-vs-HCE above, but specific numbers are not on record. Re-running pair-h1024 vs HCE at 100k nodes under the new methodology (independent starts, parallel analyzer) would establish a clean baseline if needed.
- **h2048 result.** Memory says "h2048 didn't help" but I have no raw match data in this catch-up. Not characterizing the magnitude.
- **Original `e6_pair_h1024_10k` training config.** Specifically the dropout value used at training time — user recalls 0.5 but this is uncertain. The match results above stand independent of this.

### Open question / next experiment in flight

To disambiguate the DUO vs COMBINED ambiguity, a weighted-100kn run is in progress: same combo dataset but with `hce_100kn.csv` listed 8 times in the file list (8x oversampling → ~25% effective contribution instead of 4%). If this run produces a clear strength win against COMBINED, the 100kn signal is real and the prior result was a volume issue. If it produces another coin-flip, encoding-limit becomes the strong read.

Match-test results pending.

---

## 2026-10-23 — back from the search detour: v2.0.0 shipped, tools synced to 2.0.0 rules

**What happened to the open threads from 04-29:**
- The 100kn×8 oversampling run (`nnue_h1024_pairs_10k_plus_100kx8_20260429_213158`) never saved a checkpoint — the run dir is empty. Never match-tested. DUO-vs-COMBINED stays unresolved.
- h2048 didn't beat h1024 (per notes from ~05-04). Match numbers were never logged.
- May → Oct went to search work in GygesRust (lazy SMP / coop / scout → YBWC, merged 06-13), tested with the SPRT runner here. Those parallel-search SPRT results weren't logged in this file.

**v2.0.0 released 2026-10-22.** Ships `e9_pair_h1024_combined` as the embedded `weights/default.bin` (md5 match checked: `75a3b36a…`). That's epoch 9 of `nnue_h1024_pairs_10k_combined_20260429_172819` — 5886 → 1024 → 1, dropout 0.5, on `hce_10kn_combined.csv`. The benchmark report is in GygesRust `benchmarks/2.0.0/results/report.md`. The NN-vs-HCE part (same 2.0.0 engine, only the eval differs, 1 thread, 100 pairs per budget):

| nodes | WW | WD | DD | LD | LL | decisive share |
|---|---|---|---|---|---|---|
| 20k | 67 | 3 | 30 | 0 | 0 | 100% |
| 50k | 68 | 2 | 26 | 0 | 4 | 94% |
| 100k | 65 | 1 | 32 | 0 | 2 | 97% |

NN runs at ~0.65× HCE nodes/sec. Side note for later: the engine's NN eval re-sums every active feature row from scratch every node (f32, no incremental accumulator), so the "sparse accumulator for speed" part of the method isn't actually implemented in the engine yet.

**Rules change in 2.0.0** (GygesRust 62274ce, 10-09): every move must change the board. Movegen no longer produces (a) a bounce that ends on its own start square or (b) a drop-swap that puts the same piece type back on the start square.

The tools never checked move legality — the runner just applied whatever the engine sent. So the rule only reached the tools through whichever engine binary was being run, and both binaries in `engines/` predate it. Changes:
- Every engine move is now checked against the `gyges` library's legal move list (`BoardState::is_legal`) before it's applied. Illegal move → that engine forfeits (match) or the game is discarded (data gen). Logged as `[EVENT]` on stderr. So the sim follows whatever rules the library has, regardless of engine age. Unit tests cover both kinds of board-unchanged move.
- **Bug found:** 2.0.0 rejects `setoption maxNodes 0` and keeps its *previous* limit. The tools used 0 to mean "no cap" (data gen `[0, 0, 5000]`), so under 2.0.0 the opening moves would have silently inherited the last game's cap. Fixed by sending a huge cap instead of 0.
- Folded in the robustness changes from the v2.0.0 benchmark patches (`tools_runner_*.patch`): hard per-move limit of 2× time + 5s (engine killed + forfeits), crashed/killed engines restart between games, a 1.1.0 `100|100|100` resign counts as a loss, `bestmove` lines without score/time are OK.
- Built `engines/gyges_v2.0.0.exe` from the v2.0.0 tag. The default SPRT config is now v2.0.0 vs v2.0.0, with `weightsPath` swapped on the candidate side, 1 search thread each, 100k nodes, 4 games at once (each 2.0.0 engine allocates a ~400MB TT).

**Caveat — all training data is pre-2.0.0 rules.** Counted board-unchanged moves in `hce_10kn_combined.csv` (row i+1 = row i flipped, same game): **185,515 of 18.6M in-game moves (~1.0%), in ~164k of ~885k games (~19%)**. So e9 learned from games where HCE sometimes passed. The positions themselves are still legal, but the outcomes come from games played under the old rules. Each pass also leaves an exact "same board, other side to move" pair in the data — the tempo-swap pair we argued against faking via augmentation. Worth knowing for the paper. Any newly generated data will be under the new rules, so old-vs-new data mixes have a rules confound.

**Cleanup:**
- `engine_analyzer.rs` split into `game.rs` (engine I/O + one game + legality) and `data_gen.rs`. The match worker moved into `match_runner.rs`. A `Schedules` struct replaces the four schedule vectors.
- Removed `convert_dataset` (control-feature re-encoder, dead since control features were dropped) and with it the `gyges_engine` dependency.
- The summary box now prints the limits line so pasted results carry their conditions.
- Smoke-tested: 6-pair SPRT v2 HCE vs v2 NN at 3k nodes ran clean (no false `[EVENT]`s over 255 validated moves). A 2-worker × 3-game data gen wrote correct 39-col rows.

**Dataset backup:** e9's training set = `DATASET_BACKUP/hce_10kn.csv` + `hce_10kn_big.csv` (byte concat, md5 verified). `hce_10kn_big.csv` had no backup. Its 15 raw shards were sitting in the repo root, where the next data-gen run would have appended to them. Copied the big file and moved the shards to `DATASET_BACKUP_V2/` with hashes + rebuild steps in its README. `DATASET_BACKUP/` untouched. Still only on this one SSD.

**Comparability note:** results from here on use the v2.0.0 binary on both sides plus the pentanomial SPRT box. Older NN-vs-NN results used older engine builds (`gyges_large_1024.exe`, etc.) and full/partial/tied set counts. Directionally comparable, but not the same instrument.

### Update (same day) — verified e9's training data + script

**Is `hce_10kn_big.csv` really 10k-node data?** The code doesn't say for sure: the commented data-gen block at `a7de0d7` (Apr 26) reads `vec![0, 0, 5_000]; // 10k node limit`, so the value and the comment disagree. Compared per-game stats instead:

| dataset | rows | games | mean len | len p10/25/50/75/90 | first-mover win | unique |
|---|---|---|---|---|---|---|
| hce_5kn | 6.12M | 367k | 16.67 | 7/12/16/21/26 | 52.20% | 92.19% |
| hce_10kn | 4.57M | 264k | 17.33 | 7/12/17/21/27 | 54.37% | 92.92% |
| hce_10kn_big | 15.20M | 877k | 17.34 | 7/12/17/21/27 | 54.42% | 90.86% |

`big` matches 10kn almost exactly and is clearly off from 5kn, so it's 10k-node data. That makes `hce_10kn_combined` = all the 10k-node data we have (10kn + big). Not proof from a log, but a strong fingerprint.

**Does the current `train_pair.py` match what made e9?** Loaded the shipped `e9_pair_h1024_combined.bin` into the current script's `GygesNet`, rebuilt the same split (`random_state=42`) from `hce_10kn_combined.csv`, and recomputed val: **0.3901**, matching the logged plateau (e10/e11 = 0.3901). Re-saving through `save_weights_pair` gives back the shipped file byte for byte. So the data, split, encoding and export layout are confirmed. Dropout/lr/batch can't be read back from the weights, but nothing suggests they changed (0.5 / 1e-3 / 4096 is what every h1024 run used).

Known non-reproducible bit: no torch seed (init + shuffle), so a rerun gives a different e9. The split is seeded.

### Update (same day) — removed `train_flat.py`

Dropped the flat 144-feature training script; pair encoding won and flat isn't going anywhere. Also dropped `common.save_weights`, which only flat used. The last version is in commit `768207a` if a flat baseline is ever needed again for the paper's flat-vs-pair scaling comparison. Note that the 2.0.0 engine only loads the 5886 → 1024 → 1 pair layout, so flat nets would also need the old `engines/baselines/gyges_144x*.exe` builds to be match-tested. The flat results above and the flat weights on disk are untouched.

### Update (same day) — dataset cleanup + `V2.0.0_release_dataset/`

Renamed `DATASET_BACKUP_V2/` → `V2.0.0_release_dataset/` and moved `hce_10kn.csv` in from `DATASET_BACKUP/`. The folder now holds both halves of e9's training set (`hce_10kn.csv` + `hce_10kn_big.csv`; concat md5 re-verified = `2535d843…`). Started deleting old data: the 1s/2s time-limited sets are gone from the repo `DATASET_BACKUP/` (the old `Desktop/DATASET_BACKUP/` duplicate still has them). Plan is to keep only the 10k-node data.

Gotcha found while explaining the combined set: both 10k data-gen runs numbered games the same way (`worker * 10M + game`), so game IDs repeat between the two halves (~885k distinct IDs for ~1.14M games). e9 is unaffected (game filters were off), but `filter_min_moves` or any game-aware split would merge unrelated games on the combined file. New naming scheme will give each data-gen run its own ID range.

### Update (same day) — naming scheme, engines cleanup, data gen ready for NN data

**Naming scheme.** Everything gets a short id, and NOTES.md is the registry (old name → id for every dataset and run). Datasets `d{NN}-{who played}-{nodes}`, training runs `m{NN}-{arch}-{dataset}/e{N}.bin`, nets `m{NN}e{N}`, engines `gyges-v{version}.exe`. The shipped net `e9_pair_h1024_combined` is now **m22e9**, and its training set "10kn_combined" is **d05** (= d03 `hce_10kn` + d04 `hce_10kn_big`). Renamed all `training/weights/` run dirs (m06–m23; m01–m05 only survive as nets in `engines/legacy/`) and removed 4 empty run dirs. The archive `V2.0.0_release_dataset/` keeps its original file names — it's frozen.

**Engines cleanup.** Deleted the old pre-2.0.0 builds (`gyges_large_1024.exe`, `gyges_engine.exe`) and the weights that were md5 duplicates of training checkpoints and that no current engine can load (h2048, h1024x32, flat e70, a second e6 copy). Moved the four flat width-sweep nets (only copies left) and their `gyges_144x*` builds to `engines/legacy/`. `engines/` is now `gyges-v2.0.0.exe` + `weights/{m14e6, m22e9, m23e9}.bin` (all loadable by 2.0.0) + `legacy/`.

**Runner: separate schedules per engine**, so asymmetric budgets work (NN@10k vs HCE@100k, etc.). When budgets differ, the summary box shows both.

**Data gen** now writes `training/data/<dataset>/w{NN}.csv`, no more shards in the repo root. Game ids are `(dataset_num*1000 + worker) * 10M + game`, so they can't collide across runs like d03/d04 did. Restarting with the same dataset name resumes each worker's numbering. Default config = **d08-m22e9-10k**: 2.0.0 with its embedded net, 10k nodes on *every* move including the two randomized openers (uncapped, 2.0.0 spends the full 5s on each random move). Smoke-tested at 2k nodes (ids + resume correct). `train_pair.py` names runs from `MODEL_ID` / `DATASET_ID` and refuses to overwrite an existing run dir.

**Started: m22e9 NN @10k vs HCE @100k** — same 2.0.0 binary, 1 search thread each, 4 games at once, 200 pairs. Question: how much HCE search does the NN's eval equal at the budget we'd generate data with? Results to follow.

### Update (same day) — switched to readable names; NN@10k vs HCE@100k result

**Naming, take 2.** The short-id scheme (d08 / m22e9) was confusing, so it's replaced with names that say what things are: datasets `{who played}-{nodes}` (`hce-10k` = release data = `hce-10k-part1` + `-part2`; NN self-play is `gen1-5k`, `gen2-…`), runs `{encoding}{width}_{dataset}` (`pair1024_hce-10k`), nets `{run}_e{N}`. **The release net is `pair1024_hce-10k_e9`.** Registry + old→new table in NOTES.md. Run folders and engine nets renamed again; the archive keeps its original file names.

**Data gen safe-stop.** Each game is now written + flushed as it finishes (it was every 50 games — stopping a run early would lose up to 50 games/worker and could leave half a row, which also broke resume). On restart a half-written last row is trimmed. Added an optional time budget (`max_hours`). Game ids now come from a hash of the dataset name instead of a number. Smoke-tested: cut a row in half by hand → trimmed on resume, all rows 39 cols, ids continued.

**Match: `pair1024_hce-10k_e9` @10k nodes vs HCE @100k nodes** (same 2.0.0 binary, 1 search thread each, 4 games at once, independent boards). Stopped at 177 of 200 pairs:

| | |
|---|---|
| pair score (NN) | **0.462**, 95% CI [0.411, 0.513] |
| pentanomial LL / LD / DD / WD / WW | 47 / 7 / 84 / 4 / 35 |
| games NN – HCE – draw | 158 – 185 – 11 |
| think time / depth | NN 249 ms, 2.64 · HCE 1564 ms, 3.95 |
| forfeits | 0 |

So NN at 10k nodes is a bit weaker than HCE at 10× the nodes, but close (0.46, CI barely includes 0.5) — while thinking ~6× less wall-clock per move. For data gen: NN@10k game quality ≈ just under HCE@100k. For reference, the release net was trained on HCE@10k games. Next: NN@10k vs HCE@200k to bracket it.

### Update (same day) — NN@10k vs HCE@200k

Same setup, stopped at 63 pairs once the result was clear:

| | |
|---|---|
| pair score (NN) | **0.341**, 95% CI [0.254, 0.429] |
| pentanomial LL / LD / DD / WD / WW | 29 / 0 / 24 / 2 / 8 |
| games NN – HCE – draw | 42 – 82 – 2 |
| think time / depth | NN 216 ms, 2.58 · HCE 2781 ms, 4.10 |
| forfeits | 0 |

**Bracket: the release net at 10k nodes ≈ HCE at a bit over 100k nodes** (0.46 vs 100k, 0.34 vs 200k), i.e. its eval is worth roughly 10× the search. It searches ~1.5 ply shallower (2.6 vs 3.95-4.1) and thinks 6-13× less per move.

How the games end fits that: in both matches the NN's wins are long games and its losses are short ones (vs 100k: NN wins avg 22.5 moves, losses 17.8; vs 200k: 23.6 vs 16.9). HCE wins by finding short tactical knockouts with its deeper search; the NN wins by out-positioning it over a longer game, which deeper search can't fully make up for. For gen1 data this means the NN's 10k-node games should be much better played than the HCE 10k games in `hce-10k`, especially in the long positional middlegames.

### Update (same day) — gen1-10k data gen started

Started **`gen1-10k`** on the laptop ~19:55: release net `pair1024_hce-10k_e9` (2.0.0, 1 thread) playing both sides at 10k nodes on every move, randomize `[1, 1, 0]` (first two moves per side random — same as `hce-10k`), independent boards, 12 workers, 10 h budget. Kept the random openers so the only difference from `hce-10k` is who played. The opening-noise question (~4 positions/game get labels that partly reflect a random move) gets tested at training time with `MIN_MOVE_NUM` instead.

`randomize` in 2.0.0 = search normally, drop root moves proven losing, then pick uniformly from the rest (plays a forced win if one is found). It's needed: at fixed nodes on 1 thread the engine is deterministic, and there are only 8,100 start boards (90 × 90), so without it there'd be at most ~16k distinct games.

Rates at 26 min: **~12k positions/h per worker, ~139k/h total**, 18.1 positions/game (hce-10k: 17.3), ~1% draws, 0 forfeits, 0 time-cap hits. The 10 h run should give ~1.4M positions (~7% of hce-10k). At this rate a full 20M set is ~5½ laptop-days, so the bulk will run on AWS (c8a, Zen 5 Turin, 1 vCPU = 1 physical core; us-east-1 on-demand quota is 128 vCPU, increase to 192 requested for on-demand + spot). Things to watch there: each NN engine holds its own 24 MB weight copy, so ~100 engines may become memory-bound — check per-worker positions/h against the laptop's ~12k. Don't write the same dataset name on two machines at once (shard names and game ids would collide).

Plan once it's done: `DATASET = "gen1-10k"` in train_pair.py → `pair1024_gen1-10k`, match vs the release net and vs `pair1024_hce-10k-part1_e6` (4.6M HCE positions — the closer comparison at this data size).

### Update (same day) — laptop gen1-10k stopped, duplication check, moving to AWS

Stopped the laptop run at 22:22 after ~2½ h (moving the bulk to AWS): **352,837 positions, 19,515 games**, all shards end on complete rows.

- Unique boards **96.3%** (hce-10k-part1 was 92.9%). Duplicates are almost all start boards: move 0 is 37.6% unique (7,335 of the 8,100 possible starts used), move 1 98.8%, moves 2+ ~100%.
- Whole games: 19,514 distinct sequences out of 19,515 — the random openers do their job.
- Of boards seen more than once (6,112, mostly starts), 66% carry both W and L labels — expected for near-50/50 start positions.
- Game length mean 18.1 / median 18 (hce-10k: 17.3 / 17), W/L 51.6/48.4 by row, ~1% draws dropped.

AWS run goes under its own name **`gen1-10k-aws`** (same config) so shard names/game ids don't collide with the laptop's `gen1-10k`. Cargo.toml now pins `gyges` to the v2.0.0 git tag (no Windows path), and `gyges-tools datagen <dataset> <workers> <hours>` runs data gen without editing code. Plan: c8a.48xlarge spot in us-east-1a (~$3.74/h vs $10.35 on-demand), 192 workers.

### Update (same day) — gen1-10k-aws running

Launched **`gen1-10k-aws`** on a c8a.48xlarge spot instance (AMD EPYC 9R45, 192 cores, no SMT, 2 NUMA nodes, 370 GB) in us-east-1d at ~$3.78/h (1a had no spot capacity). Amazon Linux, root disk grown 8 → 30 GB. Same config as `gen1-10k`, 190 workers, 10 h budget.

**Build gotcha: v2.0.0 panics when built with current Rust.** First test on the box crashed at ply 2 with `user-provided comparison function does not correctly implement a total order`. The move-ordering comparator in `gyges_engine/src/search/mod.rs:865` (score if >3% apart, else history) isn't transitive, and Rust ≥1.81's sort detects that and panics. The laptop toolchain is 1.80.0 (the engine exe carries rustc commit 051478957), which just sorts silently. Pinned the server to 1.80.0 so its engine matches the laptop one exactly; the comparator needs a real fix in GygesRust before anyone builds the release on modern Rust.

Rates: 4-worker smoke test **~30k positions/h per worker** (2.5× the laptop). At 190 workers **~20k/h per worker, ~3.8M/h total** — the drop is most likely shared L3 (each engine reads its own 24 MB of f32 weights; 8 cores share 32 MB). Fewer workers can't beat the total (96 × 30k = 2.9M/h), so 190 stays. ~20M positions in ~5¼ h; running the full 10 h for ~38M so a matched 20M subset and a data-volume curve both come out of it.

### Update (same day) — small-TT engine test (is the per-move TT realloc slowing data gen?)

v2.0.0 frees and re-allocates its 400 MB TT (`alloc_zeroed`, 2^22 clusters) before every search (`ugi.rs:336` → `tt().reset()`). Suspected the page faults/zeroing were a big part of the 30k → 20k/h per-worker drop at 190 workers. Added `setoption hash <MB>` on a GygesRust branch `tt-size` (off v2.0.0, local only; 2 MB → 16,384 clusters = 1.6 MB) and benched on the laptop: same positions sampled from gen1-10k, 10k nodes, randomize off, stock exe vs new build.

| config | 1 engine (300 pos) | 12 parallel (1200 pos) | same move as stock |
|---|---|---|---|
| stock v2.0.0, 400 MB | 7.0 pos/s | 35.9 pos/s | — |
| new build, 400 MB (control) | 6.6 | 36.2 | 100% |
| new build, 16 MB | 5.7 | 36.9 | 100% |
| new build, 2 MB | 7.3 | **38.8 (+8%)** | 100% |

Moves are **identical** at every size, so the TT size has no effect on the data at 10k nodes. But the speed gain is small — ~8% at 12 parallel, single-engine numbers are within run-to-run noise (±10%). The realloc is not the main cost; the NN eval itself is (full-refresh f32, ~78 × 1024 adds per eval). Windows only — Linux page-fault cost could differ, untested on the AWS box. Not worth restarting the AWS run for. Real speedups would need eval work (incremental accumulator, int16 weights), which can change moves.

### Update (same day) — why AWS workers are slower than the smoke test

Profiled the running 190-worker job (no restart): 190 engines at 99% CPU each, `sy` 2% (kernel), 2.2M page faults/s (~2k per move — the TT realloc, but only ~2% of CPU on Linux), run queue ~205 on 192 cores (slight oversubscription), and `perf stat -a`: 5.46e12 cycles / 10 s / 192 cores = **~2.85 GHz per core, IPC 2.58**.

So nothing is gridlocked and the cores aren't stalling on memory (IPC 2.58 = compute-bound). The 30k → 20k/h per-worker drop is the clock: the 4-worker smoke test ran near single-core boost (~4.4 GHz), the full box runs all-core at ~2.85 GHz (ratio ~1.5, matches). Fewer workers won't raise the total; the small TT is worth ~2% here. The only real lever is fewer instructions per move — NN eval work (incremental accumulator, wider SIMD).

### Update (same day) — where the duplicates come from (hce-10k-part1 vs gen1-10k)

Same breakdown as gen1-10k, on `hce-10k-part1` (4.57M rows, 263,738 games): 92.9% unique, 323,877 duplicate rows. **79% of those are start boards** — move 0 is 263,738 rows but only 8,100 distinct boards (all 90 × 90 used). Move 1: 84.8% unique, move 2+: ~99.6%, moves 20+: 97.5%.

Leaving out move 0: hce-10k-part1 is 98.4% unique, gen1-10k 99.7%. So the headline dup rate is mostly "how many games vs 8,100 starts", and it rises with dataset size for any data source — gen1's 96.3% at 19.5k games will drop toward ~93% at hce-10k-part1's size. The real difference is mid/late game: HCE repeats more there (97.5% unique at moves 20+ vs gen1's 99.7%), plausibly the old rules (board-unchanged moves) plus more forcing HCE lines.

Training keeps duplicates (no dedup in `train_pair.py`; `stats()` only counts them). With MSE on W/L, a repeated board trains toward its average outcome, which is the right target. Side effects: start boards are ~5.5-5.8% of rows with labels close to coin flips, and the random split puts copies of the same board in both train and val (val slightly optimistic, same for both datasets). Raw rows are a fair size measure between these two sets since the dup structure matches.

### Update (same day) — do start boards favour a side? (hce-10k, all 1.14M games)

Per start board, first-mover win rate (move-0 label): 8,100 boards, 94-188 games each (mean 141). **First mover wins 54.4% overall.** Spread of per-board win rates: sd 4.6 pts observed vs 4.2 pts expected from binomial noise alone → real board-to-board sd only **~1.8 pts**. So beyond the first-move edge, start boards barely matter, and the "best/worst" boards (73% / 37%) are what noise at n≈140 gives across 8,100 boards — not real favourable setups. Caveat: both sides' first two moves are random, which washes out any start-board effect; under strong play from move 1 it could be bigger. The net learns roughly "side to move ≈ 54%" on start boards plus small per-board tweaks.

### Update (2026-10-24) — first gen1 net: pair1024_gen1-10k+gen1-10k-aws_e6 vs pair1024_hce-10k-part1_e6 @10k

Trained `pair1024_gen1-10k+gen1-10k-aws` with the release recipe (h1024, dropout 0.5, lr 1e-3, batch 4096, mirror on, no filters) on 4,704,430 rows (12 laptop + 190 AWS shards): 93.0% unique, 261,317 games — matched to hce-10k-part1 (4,571,787 rows, 92.9% unique, 263,738 games). Val 0.3689 → 0.3542 (e5) → 0.3539 (e6) → 0.3538 (e7) → 0.3544 (e8, back up); train still falling (0.3259 @ e6, 0.3096 @ e8). Took **e6** (same epoch as the HCE net). Val is far below the release run's 0.3901 but it's a different val set — gen1 outcomes may just be more predictable (one player, self-consistent), so loss says nothing about strength here.

Match: same 2.0.0 engine, both nets via `weightsPath`, 10k nodes, 1 thread, randomize 0, independent boards, 4 games at once, 200 pairs (7 min).

| | |
|---|---|
| pair score (gen1) | **0.559**, 95% CI [0.508, 0.609], LOS 98.9% |
| pentanomial LL / LD / DD / WD / WW | 41 / 6 / 83 / 5 / 65 |
| games gen1 – hce – draw | 217 – 170 – 13 |
| think time / depth | gen1 201 ms, 2.61 · hce 227 ms, 2.79 |

SPRT inconclusive against the 0.52 bar (LLR +1.46), but the CI clears 0.5: **at equal data size, NN self-play data gives a stronger net than HCE data** — modest, ~56% pair score. It wins with slightly shallower search, so the gain is in the eval. Caveats: part1_e6's dropout was never logged (may not be 0.5), and the HCE data is pre-2.0.0 rules (~1% board-unchanged moves). A clean baseline would retrain hce-10k-part1 with today's script. Next: same match at 100k nodes, then gen1 vs the release net.

### Update (same day) — gen1 e6 vs the release net @10k

Same setup, B = `pair1024_hce-10k_e9` (embedded), 8 games at once, 200 pairs (4m 42s).

| | |
|---|---|
| pair score (gen1) | **0.419**, 95% CI [0.370, 0.468], LOS 0.06% |
| pentanomial LL / LD / DD / WD / WW | 69 / 2 / 88 / 7 / 34 |
| games gen1 – release – draw | 163 – 228 – 9 |
| think time / depth | gen1 263 ms, 2.57 · release 276 ms, 2.73 |

The release net (4.2× the data) is clearly stronger: it takes 67% of the decisive pairs (69 LL vs 34 WW). Ordering at 10k: **release (hce-10k, 19.8M) > gen1 (4.7M) > hce-10k-part1_e6 (4.6M)**. Better data closes only part of a 4× size gap, so data volume is still the bigger lever at this scale. Plan: grow gen1 to ~20M (resume `gen1-10k-aws`) for a same-size gen1 vs release comparison, which also gives a second point on the size curve. Before that: gen1 vs part1_e6 again at 50k nodes.

### Update (same day) — gen1 e6 vs hce-10k-part1_e6 @50k

Same pair as the 10k match, 50k nodes, 8 games at once, 200 pairs (25m 07s).

| | |
|---|---|
| pair score (gen1) | **0.591**, 95% CI [0.543, 0.640], LOS 99.99% |
| pentanomial LL / LD / DD / WD / WW | 33 / 4 / 90 / 3 / 70 |
| games gen1 – hce – draw | 233 – 160 – 7 |
| think time / depth | gen1 1344 ms, 3.61 · hce 1441 ms, 3.71 |

Up from 0.559 @10k (WW/LL 65/41 → 70/33, 61% → 68% of decisive pairs). The +0.033 change is inside the ~±0.07 noise on a difference of two 200-pair scores, so it's a hint that gen1's edge grows with search, not a result. Still searching ~0.1 ply shallower and winning. Next: gen1 vs the release net @50k — if gen1 loses *worse* there than at 10k (0.419), deeper search amplifies whichever eval is better overall; if it closes the gap, gen1's eval is relatively better in what deeper search reaches.

### Update (same day) — gen1 e6 vs the release net @50k: the gap closes

Same setup as the 10k release match, 50k nodes, 8 games at once, 200 pairs (25m 57s).

| | |
|---|---|
| pair score (gen1) | **0.534**, 95% CI [0.485, 0.582], LOS 91.4% |
| pentanomial LL / LD / DD / WD / WW | 40 / 8 / 92 / 5 / 55 |
| games gen1 – release – draw | 207 – 180 – 13 |
| think time / depth | gen1 1320 ms, 3.65 · release 1471 ms, 3.74 |

**At 10k the release net beat gen1 0.419; at 50k gen1 is level or slightly ahead (0.534).** +0.115 is ~3 sd, well outside noise. Gen1 was trained on 4.7M rows vs the release net's 19.8M.

| gen1 e6 vs | @10k | @50k | change |
|---|---|---|---|
| hce-10k-part1_e6 (4.6M HCE) | 0.559 | 0.591 | +0.033 (noise-level) |
| hce-10k_e9 release (19.8M HCE) | 0.419 | 0.534 | **+0.115** |

Reading: the release net's edge at 10k is breadth (4× the positions); with deeper search gen1's eval takes over. Plausible mechanism: HCE-data nets learned what wins when a weak player (HCE @10k) plays the position out, gen1 learned what wins under strong play — and deeper search makes both engines play more like the strong player, which is the regime gen1's labels describe. So gen1's eval gets relatively more accurate as search deepens. Next: 100k nodes (does gen1 pull clearly ahead?), optionally 400 pairs @50k to firm up "≥ release", then gen1 at ~20M.
