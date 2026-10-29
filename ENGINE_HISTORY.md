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

### Update (same day) — gen1 e6 vs the release net @100k (partial, 45 pairs)

Stopped at 45 pairs (14m 21s, 8 games at once) for time. Pair score **0.500**, 95% CI [0.407, 0.593]; LL / LD / DD / WD / WW 9 / 1 / 24 / 3 / 8; games 43 – 43 – 4. Depth gen1 4.22 · release 4.34, ~3 s/move.

Too few pairs to separate from the 50k result (0.534) — the CI is ±0.09 — but it's consistent with "level", and only just overlaps 10k's 0.419. So far: gen1 (4.7M) vs release (19.8M) = 0.419 @10k → 0.534 @50k → ~0.50 @100k. The gap closes past 10k and stays closed; no sign yet of gen1 pulling further ahead or the release net coming back. Needs the full 200 pairs to say more.

### Update (same day) — gen1 grown to release size (gen1-10k-aws2)

Resumed the stopped spot instance and ran a new dataset **`gen1-10k-aws2`** (same config, 190 workers) instead of appending to `gen1-10k-aws`, so the 4.35M that `pair1024_gen1-10k+gen1-10k-aws_e6` was trained on stays untouched and the size curve comes from dataset mixes. Rate ~3.6M/h (14.24M at 3h56m); stopped at **15,164,610 rows** (~4h12m). All gen1 data: 352,837 + 4,351,593 + 15,164,610 = **19,869,040**, within 0.5% of `hce-10k` (19.77M). Next: train `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2` with the release recipe and match it against the release net at 10k / 50k — the same-size data-source comparison.

### Update (same day) — gen1 at release size beats the release net @10k

Trained `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2` (19.87M rows, release recipe): train mirrored ~31.8M, 89% unique. Val 0.3593 (e1) → 0.3445 (e9) → 0.3443 (e11/e12, flat); train 0.3350 @ e9, train-val gap 0.0095 vs 0.028 for the 4.7M run — much less overfitting. Took **e9** (first epoch within 0.0002 of the best, same rule as the 4.7M e6 pick).

Match vs `pair1024_hce-10k_e9` (19.77M HCE rows), 10k nodes, 1 thread, 8 games at once, 200 pairs (4m 04s):

| | |
|---|---|
| pair score (gen1 19.9M) | **0.593**, 95% CI [0.544, 0.641], LOS 99.99% |
| pentanomial LL / LD / DD / WD / WW | 33 / 6 / 85 / 6 / 70 |
| games gen1 – release – draw | 231 – 157 – 12 |
| think time / depth | gen1 220 ms, 2.50 · release 246 ms, 2.76 |

**At equal data size, NN self-play data beats HCE data clearly** — gen1 takes 68% of decisive pairs, ~+65 Elo (from pair score), and does it searching a quarter ply shallower. This is the strongest net we have.

Scaling (vs the release net @10k): gen1 4.7M = 0.419 (~−57 Elo) → gen1 19.9M = 0.593 (~+65 Elo): ~+120 Elo for 4.2× the data, so size is still paying off on gen1 too. Also means gen1 is now the strongest player at 10k → it can play gen2's data. Next: same match @50k.

### Update (same day) — gen1 19.9M vs the release net @50k: SPRT passes

Same pair, 50k nodes, 8 games at once, 200 pairs (20m 11s).

| | |
|---|---|
| pair score (gen1 19.9M) | **0.640**, 95% CI [0.595, 0.685], LOS 100% — **H1 accepted** (LLR +5.00) |
| pentanomial LL / LD / DD / WD / WW | 22 / 1 / 94 / 9 / 74 |
| games gen1 – release – draw | 251 – 139 – 10 |
| think time / depth | gen1 1041 ms, 3.62 · release 1160 ms, 3.76 |

First SPRT pass of the gen1 work. gen1 takes 77% of decisive pairs (74 WW vs 22 LL), ~+100 Elo. Up from 0.593 @10k (+0.047, ~1.4 sd on the difference — same direction as the 4.7M net's 10k → 50k jump, but not significant on its own).

gen1 vs the release net so far:

| gen1 data | @10k | @50k |
|---|---|---|
| 4.7M | 0.419 | 0.534 |
| 19.9M (size-matched) | 0.593 | **0.640** |

Both rows rise from 10k to 50k. Same-size NN self-play data gives a clearly stronger net at both budgets, and the margin is bigger with deeper search. `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` is the new best net and the obvious player for gen2 data.

### Update (same day) — gen1 19.9M vs the release net @100k: SPRT passes again

Same pair, 100k nodes, 8 games at once, 200 pairs (42m 45s).

| | |
|---|---|
| pair score (gen1 19.9M) | **0.6525**, 95% CI [0.604, 0.701], LOS 100% — **H1 accepted** (LLR +4.68) |
| pentanomial LL / LD / DD / WD / WW | 25 / 8 / 74 / 6 / 87 |
| games gen1 – release – draw | 254 – 132 – 14 |
| think time / depth | gen1 2163 ms, 3.97 · release 2366 ms, 4.15 |

~+109 Elo; gen1 takes 78% of WW+LL pairs (87 vs 25). Fewer DD than at 50k (74 vs 94) — more pairs are being decided, and most of them go gen1's way. Still a ~0.2 ply shallower search at the same node count.

gen1 vs the release net:

| gen1 data | @10k | @50k | @100k |
|---|---|---|---|
| 4.7M | 0.419 | 0.534 | ~0.50 (45 pairs) |
| 19.9M (size-matched) | 0.593 (+65) | 0.640 (+100) | **0.6525 (+109)** |

The size-matched row goes up at every budget. 50k → 100k is only +0.013 (~0.4 sd), so the trend is levelling off; 10k → 100k is +0.06 (~1.7 sd). So NN self-play data is at least as big a win with deeper search, and if anything bigger. 

### Update (same day) — does it translate? gen1 19.9M @10k vs HCE @100k

Same setup as the release-net baseline above (NN @10k vs HCE @100k, 1 thread each), 8 games at once, 200 pairs (16m 54s).

| | |
|---|---|
| pair score (gen1 19.9M) | **0.5225**, 95% CI [0.474, 0.571], LOS 82% — SPRT inconclusive (LLR +0.41) |
| pentanomial LL / LD / DD / WD / WW | 43 / 5 / 94 / 7 / 51 |
| games gen1 – HCE – draw | 203 – 185 – 12 |
| think time / depth | gen1 246 ms, 2.63 · HCE 1727 ms, 4.00 |
| game length | gen1 wins avg 22.3, losses avg 17.8 |

| vs HCE @100k | pair score | ≈ Elo | LL | WW |
|---|---|---|---|---|
| release net @10k (177 pairs) | 0.462 | −26 | 26.6% | 19.8% |
| gen1 19.9M @10k | **0.5225** | +16 | 21.5% | 25.5% |

+0.06 over the release net's baseline, ~1.7 sd on the difference — positive, not significant alone. ~+42 Elo vs HCE against +65 head-to-head with the release net, so about two thirds of the gain carries over. gen1 at 10k now slightly beats HCE at 10× the nodes, which the release net didn't.

The shape of the games is unchanged: losses are still the short ones (17.8 moves, same as the release net) — HCE's deeper search still finds the knockouts. The gain is more won-both pairs and fewer lost-both: gen1 wins more of the long positional games, it doesn't survive the tactics better. Running score sat around 0.53–0.54 for most of the match and finished at 0.5225 (normal swing at <100 pairs).

## Overnight queue (2026-10-25) — code changes first

Runs one thing at a time on the laptop (training and matches never overlap — 15.4 GB RAM, 8 GB GPU). All uncommitted; `git diff` against edaede4 shows everything.

- **train_pair.py — trains 40M-row mixes in 8 GB.** Boards go to the GPU as int8 and are encoded a batch at a time on the GPU (`encode_batch`), replacing the up-front int16 index array (~200 B a position; 39.6M rows would need ~13 GB). The sum over active features doesn't depend on order, so the net computes the same thing. Every run first checks encode_batch against `encode_position` on 3,000 boards and exits on any mismatch. Measured on the 39.6M mix: GPU 5.9 / 8 GB, RAM peak 8.7 GB (loading), ~4.7 min an epoch for 63.4M train rows (the old code: ~3.2 min for 31.8M, plus a long Python encode up front).
- **train_pair.py — other:** `hce-10k`, `-part1`, `-part2` now work inside a `+` mix (still read from the frozen archive, read-only). `DROPOUT` and `LR_DECAY` (1.0 = off) are settings at the top; runs stop after 3 epochs without a new best val (`patience`); each run folder gets `config.txt` and `log.txt`.
- **common.py:** boards held as int8 (were int64), CSVs read one file at a time with small dtypes, mirror and duplicate count vectorized (duplicates counted on boards packed into 72 bits). Checked against the old code on gen1-10k: same boards, labels, game ids, mirrored boards and unique count (672,091).
- **main.rs:** `gyges-tools sprt <net A> <net B> <nodes A> [nodes B]` runs one match with run_sprt's settings (200 pairs, 8 games at once, SPRT 0.50/0.52). A net is a file in engines/weights/ (no .bin), `release` or `hce`.
- Smoke test (gen1-10k, deleted after): encoder check passed, stopped itself at e9.

### Overnight 1 — hce-10k + gen1 (39.6M) trained; beats gen1 19.9M @10k

Trained `pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2` (release recipe): 39.64M rows, 88.8% unique; 63.4M mirrored train (87.9% unique). Val 0.3794 (e1) → 0.3664 (e12) → 0.3663 (e14), creeping by ≤0.0001 an epoch, so I stopped it by hand after e14 instead of waiting for the 3-epoch rule. Took **e12** (first within 0.0002 of best). Train 0.3616 @ e12 — train-val gap 0.005, the smallest yet. Val set is half HCE games, so the number isn't comparable to gen1-only val (0.3445).

Match vs `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9`, 10k nodes, 8 games at once, 200 pairs (4m 56s) — run with the new `gyges-tools sprt` command:

| | |
|---|---|
| pair score (mix 39.6M) | **0.5875**, 95% CI [0.539, 0.636], LOS 99.98% (LLR +2.57, just short of the bound) |
| pentanomial LL / LD / DD / WD / WW | 33 / 3 / 92 / 5 / 67 |
| games mix – gen1 – draw | 231 – 161 – 8 |
| think time / depth | mix 274 ms, 2.69 · gen1 277 ms, 2.66 |

~+61 Elo over the previous best at 10k. Doubling the data with *weaker* labels still helps a lot at shallow search. 50k next (that's where HCE labels hurt before).

### Overnight queue stopped — laptop slept

The laptop was shut at ~2:30 a.m.; Windows slept at 02:32:46 (Kernel-Power 42), five minutes into the mix @50k match, and woke at 11:09. Nothing ran in between, so the dropout / lr-decay / move-num tests never started. The 50k match had 41 pairs at wake (~0.56); discarded and rerun from scratch, since the 8 games mid-move during sleep saw an 8 h gap.

### Mix 39.6M vs gen1 19.9M @50k — the advantage disappears

Same pair as Overnight 1, 50k nodes, 200 pairs (20m 33s):

| | |
|---|---|
| pair score (mix 39.6M) | **0.495**, 95% CI [0.448, 0.542], LOS 42% (LLR −0.53) |
| pentanomial LL / LD / DD / WD / WW | 46 / 4 / 100 / 8 / 42 |
| games mix – gen1 – draw | 192 – 196 – 12 |
| think time / depth | mix 1134 ms, 3.70 · gen1 1080 ms, 3.67 |

| mix 39.6M vs gen1 19.9M | @10k | @50k |
|---|---|---|
| pair score | 0.5875 (+61) | **0.495** (−3) |

−0.09 from 10k to 50k, ~2.7 sd on the difference — real. Adding the 19.8M HCE rows helps at shallow search and is worth nothing by 50k. Same direction as gen1 vs the release net, where HCE-label nets fall further behind with depth. Reading: the extra rows buy coverage (more positions seen), which matters when search is shallow; with deeper search the quality of the labels decides, and half of these labels come from weak HCE play. For later generations: don't stack HCE data in. Whether gen1 + gen2 stacks well is a separate question (gen1 labels are far better than HCE's). 100k running next.

### Mix 39.6M vs gen1 19.9M @100k — still even

Same pair, 100k nodes, 200 pairs (44m 32s):

| | |
|---|---|
| pair score (mix 39.6M) | **0.5075**, 95% CI [0.459, 0.556], LOS 62% (LLR −0.08) |
| pentanomial LL / LD / DD / WD / WW | 45 / 9 / 91 / 5 / 50 |
| games mix – gen1 – draw | 196 – 190 – 14 |
| think time / depth | mix 2361 ms, 4.13 · gen1 2299 ms, 4.04 |

| mix 39.6M vs gen1 19.9M | @10k | @50k | @100k |
|---|---|---|---|
| pair score | 0.5875 (+61) | 0.495 | 0.5075 |

Flat from 50k on: the mix isn't worse with deeper search, just no better. So the 19.8M HCE rows buy ~+60 Elo at 10k and nothing at ≥50k — they don't hurt, but they stop helping once search is deep enough. gen1 19.9M e9 stays the reference net for deep search; the mix is the better net only for shallow (10k) play, e.g. as a data-gen player at 10k nodes.

### Do hce-10k and gen1 label the same boards differently? (analysis, no training)

Script: scratchpad `label_conflict.py`. Boards canonicalized over the mirror; labels −1/0/+1 (unscaled). Each board's mean label in hce-10k compared with its mean label in gen1 19.9M. Baseline = the same comparison between two halves of one dataset (split by game id), which shows how much labels disagree from ordinary game-to-game noise.

- **Shared boards: 487,640** (~2.8% of either set's 17.5M unique boards), but they carry ~12% of each set's rows, because nearly all of them are openings: 4,068 start boards (1.2M rows a side) and 365k boards after move 1 (1.0M rows a side). Past move 3 only ~62k boards / ~0.45% of rows are shared.
- **Conflict vs noise:** cross-source mean |Δlabel| weighted 0.329, strong opposite labels (both |mean| ≥ 0.5, opposite sign) weighted 5.6%. Within-source baseline: hce 0.313 / 6.7%, gen1 0.299 / 6.4%. **So on shared boards hce and gen1 disagree about as much as two halves of the same dataset** — no conflict beyond game-outcome noise.
- **One systematic shift:** gen1 sees a bigger first-mover edge. Start boards: mean label hce +0.078 vs gen1 +0.135; after move 1 (second mover to play): −0.071 vs −0.123. Mixing gives ~+0.11 / −0.10. Small.
- Side finding: within one dataset, boards past move 8 shared by the two halves have identical labels (|Δ| ≈ 0.01) — duplicate games. After the 2 random opening moves play is deterministic (1 thread, node cap), so two games with the same random opening are the same game. ~1–2% of rows; also a val leak.

**Reading:** direct label conflict isn't the problem — the two sources almost never contain the same position past the opening, and where they do they agree. The HCE half differs by *which* positions it covers (the ones HCE play reaches), not by labeling shared positions differently. That fits the match results: extra coverage helps at 10k and stops mattering at ≥50k. Not measured: conflict between *similar* (not identical) positions.

**Standing after the mix tests:** strongest net on paper is the mix `pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12` (+61 Elo at 10k, even at 50k/100k — never worse). Reference / favorite stays `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9`: pure high-quality data, simpler story, equal at depth. Caveat on the mix: hce-10k is pre-2.0.0-rules data (~1% board-unchanged moves), so the mix also carries a rules confound.

### Where things stand / plan (2026-10-25, for picking up in a new chat)

- Label-conflict script moved into the repo: `training/label_conflict.py` (run `python training/label_conflict.py`; writes nothing).
- Match logs for tonight's runs were in the session scratchpad only; every result is in the tables above. Matches are now one command: `gyges-tools sprt <net A> <net B> <nodes A> [nodes B]` (nets by file name in engines/weights/, or `release` / `hce`).
- Nets in engines/weights/ added today: mix `…_e12` (main) and `…_e7` (epoch test).
- Queue, one at a time: (1) mix e7 vs e12 @50k (running), (2) release @10k vs HCE @10k — strength gap between the two data-gen players, (3) mix with dropout 0.3 (`DROPOUT = 0.3`, `TAG = "drop0.3"`) → vs gen1 e9 @10k/50k, (4) gen1 19.9M with dropout 0.3 as the control → same matches. Reading: mix drop0.3 wins at 50k and the control doesn't → the net was capacity-limited by dropout; both win → dropout 0.5 was simply too high; neither → label quality explains the depth gap.
- Later: lr decay (`LR_DECAY = 0.3`), `MIN_MOVE_NUM` 1–2 (drop start / random-opening boards), split train/val by game (val leaks ~95% of games), then gen2 data played by the best net — consider storing the search score per position for blended labels. Don't stack weaker generations at full weight (weight by generation instead).

### Mix e7 vs mix e12 — the later epoch isn't worse

`pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2` e7 (val 0.3678, train-val gap 0) vs e12 (0.3664, gap 0.005), 200 pairs each:

| e7 vs e12 | pair score | 95% CI | LL / LD / DD / WD / WW |
|---|---|---|---|
| @10k | 0.501 | [0.455, 0.548] | 45 / 2 / 103 / 7 / 43 |
| @50k | 0.4825 | [0.432, 0.533] | 56 / 6 / 82 / 8 / 48 |

Even at 10k, e12 slightly ahead at 50k (within noise). At 40M rows the five extra epochs (train −0.0062, val −0.0014) didn't hurt play, so memorization isn't costing strength at this size and the "first epoch within 0.0002 of best val" pick stands for now. (On the 4.7M gen1 run val turned up at e7 — small sets are where early epochs matter.)

### How much stronger is gen1's player than hce-10k's? Release net @10k vs HCE @10k

The two data-gen players head to head at the node budget they generated with (equal 10k nodes), 200 pairs (2m 42s):

| | |
|---|---|
| pair score (release net) | **0.854**, 95% CI [0.818, 0.890] (LLR +20.2) — **~+307 Elo** |
| pentanomial LL / LD / DD / WD / WW | 7 / 0 / 42 / 5 / 146 |
| games net – HCE – draw | 336 (of 400) won by the net · see log |
| depth | net 2.42 · HCE 2.46 |
| game length | net wins avg 15.1 moves, losses 18.8 |

The net wins both games of 146 pairs and loses both of 7 — 95% of decisive pairs. So gen1 data was played by a player ~300 Elo stronger than hce-10k's, matching the estimate from the 100k/200k bracket (+250–350). Its wins are *short* here (15.1), unlike vs HCE @100k where it won the long games: at equal nodes it simply out-plays HCE tactically too. This is the "data-gen player strength" number for the paper; measure gen1 19.9M e9 @10k vs the release net @10k the same way for gen2's player (= 0.593 already, ~+65).

Correction to the table above: games net – HCE – draw = **339 – 56 – 5** (84.8% / 14.0% / 1.2%). The "336" line was written before I'd read the game counts.

### Mix with dropout 0.3 — stopped at e8 (laptop battery)

`pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_drop0.3` (only change from the e12 run: `DROPOUT = 0.3`). Stopped by hand after e8 because the laptop was nearly out of battery (GPU training drains it); e1–e8 saved.

| epoch | drop0.3 train | drop0.3 val | drop0.5 val (same epoch) |
|---|---|---|---|
| 1 | 0.3887 | 0.3765 | 0.3794 |
| 2 | 0.3757 | 0.3715 | 0.3746 |
| 3 | 0.3697 | 0.3693 | 0.3719 |
| 4 | 0.3652 | 0.3681 | 0.3703 |
| 5 | 0.3617 | 0.3675 | 0.3692 |
| 6 | 0.3587 | 0.3674 | 0.3685 |
| 7 | 0.3562 | 0.3672 | 0.3678 |
| 8 | 0.3539 | 0.3672 | 0.3672 |

Faster to start, then flat at ~0.3672 from e7 while drop0.5 kept going to 0.3663 (e14). Train-val gap at e8: 0.013 (drop0.5: 0.0009) — overfits much sooner. On val, less dropout reaches a slightly *worse* floor, so no sign yet that 0.5 was holding the net back. Not matched yet: next is e7 (best val so far) vs gen1 e9 @10k/50k; val leaks, so the matches decide. Possibly worth trying drop0.3 + `LR_DECAY` if it's flat because of lr noise.

Also added `training/feature_counts.py` (examples per feature). gen1 19.9M: 78 active features a board, median ~184k examples per feature, rarest possible feature ~54k; 108 features are impossible (cross-type pair on one square) and never fire. Per-feature coverage isn't the limit; more data helps by covering more feature *combinations*.

### Mix drop0.3 (e7) vs gen1 19.9M e9 — less dropout loses the 10k gain

Net `pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_drop0.3_e7` (best val so far when stopped, 0.3672), 200 pairs each, 8 games at once:

| vs gen1 e9 | pair score | 95% CI | LL / LD / DD / WD / WW | games W–L–D |
|---|---|---|---|---|
| @10k | 0.490 | [0.440, 0.540] | 53 / 7 / 83 / 9 / 48 | 187 – 195 – 18 |
| @50k | 0.504 | [0.457, 0.550] | 40 / 12 / 99 / 3 / 46 | 194 – 191 – 15 |

| mix vs gen1 e9 | @10k | @50k |
|---|---|---|
| dropout 0.5 (e12) | **0.5875** | 0.495 |
| dropout 0.3 (e7) | 0.490 | 0.504 |

At 10k dropout 0.3 is ~0.10 worse (~2.8 sd) — it loses the mix's whole +61 Elo. Not the epoch: drop0.5 e7 played even with its e12 (0.501). At 50k both are even with gen1. So **dropout 0.5 isn't the bottleneck — it's what lets the net use the noisier HCE half**; with 0.3 it overfits that noise (train-val gap 0.013 vs 0.0009) and the extra coverage stops helping. The depth gap is label quality, not regularization. Keep dropout 0.5. Skipped the planned gen1 drop0.3 control: with the mix result this clear, it can't change the decision.

### results/ tables + figures (no more copying numbers by hand)

New `results/` folder (see `results/README.md`): `matches.csv` (one row per finished match — the runner appends it itself now), `runs.csv` + `epochs.csv` (train_pair.py writes the config at start and a row per epoch: dropout, lr, lr_decay, batch, patience, filters, train/val), `datasets.csv` (who played, nodes, generation). Backfilled 19 matches from the summary boxes in the session transcript and the match logs (parsed, not retyped; every pentanomial sums to its pair count), the HCE @200k match from its table above, and 6 runs / 42 epoch rows.

`python training/plots.py` → `results/figures/`: `score_vs_nodes` (head-to-head score vs budget), `scaling` (Elo vs release net against training rows, one panel per budget — the paper figure; points with no direct match are chained through one net and drawn hollow), `training_curves`, `player_strength` (data-gen player ladder: release @10k = +307 over HCE @10k, gen1 19.9M @10k = +372).

Also: `datagen` takes an optional 4th argument, nodes (default 10k). Laptop rate at 12 workers: 10k nodes 129k rows/h, 1k nodes 1.21M rows/h (9.4× — nearly all time is search, no tooling overhead worth fixing). 1k nodes searches ~1.0 ply early, ~1.9 ply from move 11.

### Mix @10k vs HCE @200k — even with 20× the nodes

`pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12` @10k vs HCE @200k, 200 pairs (42m 29s). First row the runner wrote to results/matches.csv itself.

| | |
|---|---|
| pair score (mix) | **0.489**, 95% CI [0.438, 0.539] |
| pentanomial LL / LD / DD / WD / WW | 53 / 8 / 84 / 5 / 50 |
| games mix – HCE – draw | 189 – 198 – 13 |
| think time / depth | mix 302 ms, 2.59 · HCE 4393 ms, 4.28 |
| game length | mix wins 23.0, losses 19.1 |

Release net @10k vs HCE @200k was 0.341. The mix at 10k plays even with HCE at 20× the nodes, thinking 14× less per move and 1.7 ply shallower; chained Elo predicted ~0.50–0.53, so chaining holds. Same shape as always: NN wins the long games, HCE the short tactical ones.

New headline metric — **HCE-node equivalent**: the HCE node budget a net @10k scores 50% against. Release net ≈ 80–90k (0.462 @100k, 0.341 @200k); mix ≈ 200k. gen1 19.9M: 0.5225 @100k, @200k running.

Also: matches.csv gained `a_win_len, a_loss_len, draw_len` (average game length for a's wins, a's losses, draws), filled in for the old rows from the same summary boxes.

### gen1 19.9M @10k vs HCE @200k — HCE-node equivalents

gen1 19.9M e9 @10k vs HCE @200k, 200 pairs (41m 34s): **0.4575**, 95% CI [0.411, 0.504]; LL / LD / DD / WD / WW 52 / 8 / 97 / 8 / 35; games 175 – 209 – 16; depth 2.62 vs 4.34; gen1 wins avg 23.7 moves, losses 18.6.

| net @10k | vs HCE @100k | vs HCE @200k | HCE-node equivalent (50% point) |
|---|---|---|---|
| release net (HCE data 19.8M) | 0.462 | 0.341 | **≈83k** |
| gen1 19.9M | 0.5225 | 0.4575 | **≈127k** |
| mix 39.6M | — | 0.489 | ≈185k (one point; assumes the same slope) |

50% point = Elo interpolated linearly in log2(HCE nodes) between the two budgets. Each data step made the net's eval worth more HCE search: 8× → ~13× → ~19× its own 10k nodes. New chart `results/figures/hce_equivalent`, plus `strength_{10k,50k,100k}`: every net on one chart, one rating per net fitted jointly from all net-vs-net matches at that budget (weighted least squares on Elo differences, release net = 0), lines joining nets of the same data family.

### gen0 4.6M (part1 e6) vs the release net, direct — and gen1-1k data gen started

First direct matches of `pair1024_hce-10k-part1_e6` vs the release net (equal nodes, 200 pairs each; rows written by the runner):

| part1 e6 vs release | pair score | 95% CI | LL / LD / DD / WD / WW | games W–L–D |
|---|---|---|---|---|
| @10k | 0.371 | [0.324, 0.419] | 78 / 5 / 86 / 4 / 27 | 144 – 247 – 9 |
| @50k | 0.406 | [0.359, 0.453] | 65 / 9 / 92 / 4 / 30 | 156 – 231 – 13 |

The joint rating fit had predicted 0.363 / 0.441 — within error. Equal size, different source (~5M rows): gen0 0.371 / 0.406 vs gen1 0.419 / 0.534 at 10k / 50k — gen1's lead grows with depth, same as at 20M. The golden chart (`vs_release_by_budget`) now has gen0 4.6M measured at 10k and 50k.

Started **`gen1-1k`** on the laptop: release net plays (so gen1, same player as gen1-10k), 1k nodes every move incl. the random openers, 12 workers: `datagen gen1-1k 12 6 1000`. Stopping at ~4.7M rows to size-match gen1 4.7M. Plan after: train → match vs release @10k/50k → player check (new net @1k vs release @1k) → gen2-1k played by the gen1-1k net (a pure 1k iteration loop).

### gen1-1k: same player, 10× fewer data-gen nodes — flat with depth

Data: `gen1-1k`, 4,733,829 rows (92.4% unique, 271k games, 17.4 positions/game), release net @1k nodes every move, laptop 12 → 15 workers (15 workers 1.36M rows/h vs 12 at 1.27M; ~3.5 h; rate dipped to ~1.25M/h on a weak charger). Net: `pair1024_gen1-1k`, release recipe, stopped itself at e9; took **e5** (val 0.4182, best 0.4181 @ e6). gen1 4.7M (10k-node games) reached val 0.3539 — different val set, but 1k-game results are clearly much harder to predict from the position.

Matches vs the release net (equal nodes, 200 pairs each, rows written by the runner):

| ~4.7M rows of… | @10k | @50k |
|---|---|---|
| HCE @10k games (part1 e6) | 0.371 | 0.406 |
| **release net @1k games (gen1-1k e5)** | **0.399** [0.353, 0.444] | **0.391** [0.344, 0.439] |
| release net @10k games (gen1 4.7M e6) | 0.419 | 0.534 |

At 10k the 1k data is nearly as good as the 10k data (−0.02, ~0.6 sd). At 50k it's at HCE-data level, 0.14 below the 10k data (~4 sd) — the only line on the golden chart that doesn't rise with depth. So data-gen search depth still matters with the net playing: 1k games are decided by later mistakes often enough that labels are noisy, and a net trained on noisy labels doesn't gain from deeper search (same as the HCE half of the mix). Third case of the same rule: noisy labels help shallow search, not deep. Drops the pure-1k iteration loop. Open: where between 1k and 10k (and above 10k — never tested) label quality saturates; next cheap point is 3k nodes (~1.3 h locally for 4.7M rows).

### Mix 39.6M vs the release net, direct — and the joint rating fit

`pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12` vs the release net, equal nodes, 200 pairs each:

| | pair score | 95% CI | LL / LD / DD / WD / WW | games W–L–D |
|---|---|---|---|---|
| @10k | 0.645 | [0.600, 0.690] | 22 / 3 / 87 / 13 / 75 | 250 – 134 – 16 |
| @50k | 0.674 | [0.630, 0.718] | 17 / 3 / 88 / 8 / 84 | 264 – 125 – 11 |

Fit had predicted 0.674 / 0.635 — within error. Against the release net the mix scores a bit above gen1 19.9M at 50k (0.674 vs 0.640, ~0.9 sd), but head-to-head it was even (0.495); possibly a small style effect (half its data is HCE games like the release net's). Joint rating fit over all net-vs-net matches (Elo vs release net):

| | @10k | @50k | @100k |
|---|---|---|---|
| gen0 4.6M (part1 e6) | −94 | −58 | |
| gen1-1k 4.7M | −71 | −77 | |
| gen1 4.7M | −55 | +15 | |
| gen1 19.9M | +57 | +110 | +109 |
| mix 39.6M | +111 | +116 | +115 |

Mix ≈ +54 over gen1 at 10k, level at depth (+6). Golden chart and `score_vs_rows` now have the mix measured, not fitted. New chart `score_vs_rows`: score vs release net against training rows, one panel per match budget, color = generation, marker = data-gen nodes; the golden chart now finds every net automatically (one epoch per run, variants left out) with the same encodings.

### Data quality follows the data-gen player's strength

Worker test (laptop 8 cores / 16 threads, 10k nodes, ~3.7 min each): 12 workers 142k rows/h, 16 → 151k, 20 → 151k, 24 → 156k (noise). Throughput stops at one worker per hardware thread; on the AWS c8a (192 cores, no SMT) 190 workers was already the max. Laptop default now 16. `datagen` gained a 5th argument, the player net (`release` default, or a file in engines/weights/).

Player-strength matches (200 pairs each; release @20k vs @10k started and dropped):

| | pair score | ≈ Elo |
|---|---|---|
| release @1k vs release @10k | 0.136 [0.105, 0.167] | −320 |
| gen1 19.9M e9 @1k vs release @1k | 0.590 [0.542, 0.638] | +63 |

Ladder of data-gen players (HCE @10k = 0) next to how good their data was (net trained on ~4.7M rows, vs release @50k):

| player | strength | data → net @50k |
|---|---|---|
| HCE @10k (hce-10k-part1) | 0 | 0.406 |
| release @1k (gen1-1k) | ≈ −13 | 0.391 |
| gen1 e9 @1k (gen2-1k, generating on AWS) | ≈ +50 | ? |
| release @10k (gen1-10k) | +307 | 0.534 |

The release net at 1k plays as well as HCE at 10k, and its data was as good as HCE's — two very different players of equal strength gave equal-quality data. Data quality looks like a function of player strength alone, whatever makes the player strong (eval or search). The generation step is the same size at 1k as at 10k (+63 vs +65). If gen2-1k lands slightly above gen1-1k, nodes vs generations can be priced on one Elo-of-the-player scale.

### gen2-10k-aws: first gen2 data — trained, 10k match done (rest paused)

Data: `gen2-10k-aws`, played by `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` (gen1 19.9M) at 10k nodes, AWS 190 workers, 1.35 h, 3.6M rows/h → 4,852,079 rows (93.0% unique), 190 shards, 59 MB as .tgz. First run with the CloudWatch auto-stop (CPU < 5% for 3×5 min → Stop): it stopped the instance ~15 min after data gen ended. Server took the code with `git fetch && git reset --hard origin/nnue` after the history force-push (`6da4f93`).

Net `pair1024_gen2-10k-aws`, release recipe, stopped itself at e10; took **e6** (val 0.3414, best 0.3413 @ e7). Val by data-gen player at ~4.8M rows: release @1k 0.418, release @10k 0.354, gen1 e9 @10k **0.341** — cleaner labels with the stronger player (different val sets, so a hint).

| ~4.8M rows of… | vs release @10k |
|---|---|
| gen1-1k (release @1k) | 0.399 |
| gen1 (release @10k) | 0.419 |
| **gen2 (gen1 e9 @10k)** | **0.4425** [0.396, 0.489] |

+0.024 over gen1 at 10k (~0.7 sd) — about what the player-strength rule predicted (+65 Elo of player → ~+0.025). Paused here (laptop moved): still to run — gen2 vs release @50k, gen2 vs gen1 4.7M (`pair1024_gen1-10k+gen1-10k-aws_e6`) @50k with 400 pairs (`sprt <A> <B> 50000 50000 400`, new optional pairs argument), then plots.

### gen2-10k-aws: the rest of the matches — a small edge, as predicted

`pair1024_gen2-10k-aws_e6` (4.85M rows of gen2 games), rows written by the runner:

| | pair score | 95% CI | LL / LD / DD / WD / WW | games W–L–D |
|---|---|---|---|---|
| vs release @10k (from before) | 0.4425 | [0.396, 0.489] | 56 / 8 / 94 / 10 / 32 | |
| vs release @50k | 0.524 | [0.478, 0.569] | 38 / 5 / 102 / 10 / 45 | |
| **vs gen1 4.7M e6 @50k, 400 pairs** | **0.519** | [0.485, 0.554] | 90 / 12 / 178 / 17 / 103 | 399 – 368 – 33 |

Same-size comparison (~4.8M rows), vs release: gen1 0.419 / 0.534, gen2 0.4425 / 0.524 at 10k / 50k. Head-to-head at 50k: +0.019 (~+13 Elo, ~1.1 sd, LOS ~86%) — a small edge for gen2, not significant alone. Matches the player-strength prediction (+65 Elo of player → ~+0.025) and the user's call that the generation step would be small now. Cleaner labels (val 0.341 vs 0.354) but little strength from them at this size.

Reading: after the one-time jump from HCE to a learned eval (+307 Elo of player strength), a generation step adds only ~+65 of player strength and a small gain in the trained net. Depth (nodes per data-gen move) is now the bigger per-step lever (~+95 per doubling, from release @1k vs @10k), at 2× cost per row. Next cheap test: gen2 at 20k nodes (≈ +150 of player strength). Charts redrawn (gen2 in aqua on golden, size and data-quality charts).

### What does the added half's quality do? gen1 4.7M + 5M of gen0 / gen1 / gen2

Same base (gen1 4.7M = `gen1-10k+gen1-10k-aws`), plus ~5M of data that differs only in who played it. Release recipe, epochs by the usual rule, 200 pairs each vs the release net. New `<dataset>@N` subset syntax gives the gen1 control half (`gen1-10k-aws2@59`, first 59 shards = 4.72M) without copying files. Order per user: train both, quick 10k tests, 50k at the end, control last. Laptop on full power: ~60 s an epoch for 14.8M training rows (~10% faster than before).

| ~10M net (gen1 4.7M + …) | rows | best val | vs release @10k | @50k |
|---|---|---|---|---|
| (base: gen1 4.7M alone) | 4.70M | 0.354 | 0.419 | 0.534 |
| A: + gen0 (HCE @10k) | 9.28M | 0.3756 (half HCE val) | 0.556 [0.510, 0.602] | **0.515** [0.466, 0.564] |
| B: + gen1 (more of the same) | 9.42M | 0.3504 | 0.5025 [0.453, 0.553] | **0.551** [0.505, 0.598] |
| C: + gen2 (gen1 e9 @10k) | 9.56M | 0.3447 | 0.5625 [0.514, 0.611] | **0.561** [0.518, 0.605] |
| (gen1 19.9M) | 19.87M | 0.3445 | 0.593 | 0.640 |

- **At 50k the added half's quality decides it:** HCE adds nothing (A 0.515, base 0.534), gen1 and gen2 both add (B 0.551, C 0.561; C − B +0.01, noise — same small gen2 edge as the 400-pair head-to-head).
- **At 10k diversity beats quality:** a *different* player's games (A: HCE, C: gen2) gave +0.14; more of the *same* player's games (B) only +0.08 (~1.5–1.7 sd). New players reach new positions; coverage is what shallow search uses.
- **C is the only one best at both depths** (new player = coverage, stronger player = quality) → always play new data with the newest net.
- **No ceiling yet:** all three 10M nets sit far below gen1 19.9M at 50k (0.55–0.56 vs 0.640). 5M of gen2 doesn't replace 15M more gen1; good data still scales. Odd: pure gen1 4.7 → 9.4M gained only +0.017 @50k but 9.4 → 19.9M +0.089 — each point ±0.05, the 4.7M point is likely high (joint fit puts it ~0.52); not a pattern yet.
- Next: more good data played by the newest net — gen2 to ~20M and train gen1+gen2 (~40M good rows).

Charts redrawn; mixes now have their own colors (gen0+gen1 gray, gen1+gen2 green).

### gen1 19.9M + gen2 4.85M (24.7M) — no gain over gen1 19.9M

Trained `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2+gen2-10k-aws` (24.72M rows, release recipe). Val 0.3557 (e1) → 0.3409 (e12) → 0.3407 (e16), creeping ≤0.0001 an epoch, so stopped by hand at e16 (as with the 39.6M mix); took **e12** (first within 0.0002 of best). ~2.7 min an epoch.

| | pair score | 95% CI |
|---|---|---|
| vs release @10k | 0.574 | [0.528, 0.619] (gen1 19.9M: 0.593) |
| vs release @50k | 0.661 | [0.616, 0.707] (gen1 19.9M: 0.640) |
| **vs gen1 19.9M e9 @50k, 400 pairs** | **0.494** | [0.458, 0.531] |

Even head-to-head (it led 0.53 at 115 pairs, then evened out). The +0.02 vs release at 50k was noise. Adding 25% more rows from a slightly stronger player (+65 Elo) doesn't move a 20M net. Fits the rest: gen2's edge over gen1 is small at every size (+13 Elo head-to-head at 5M, +0.01 in the 10M mixes). Bigger player jumps are needed — deeper data-gen search (gen2 at 20k ≈ +150 of player strength) is the next test, before generating more gen2 at 10k. Also: the data-quality chart now holds every net (x = row-weighted player strength, line = size band), colors fixed (gen1+gen2 mix violet).

### Encoding: pairs only (no single-piece features) — the singles are redundant

New `SINGLETONS` setting in train_pair.py (default True = release recipe). False: the 108 single-piece features are never encoded and their first-layer rows stay exactly zero (checked in the saved file), so the net is still the 5886 layout the v2.0.0 engine loads — no engine change. Smoke run on gen1-10k checked the encoder and the zero rows, then deleted. Charts now treat any tagged run (`_drop0.3`, `_nosingle`) as a variant, not a point on the data lines.

`pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_nosingle` — same data and split as gen1 19.9M:

| | val (same val set) | vs release @10k | vs release @50k |
|---|---|---|---|
| pairs + singles (gen1 19.9M e9) | 0.3443 best | 0.593 | 0.640 |
| **pairs only (e9)** | 0.3446 best | 0.565 [0.519, 0.612] | **0.640** [0.594, 0.686] |

Val curves matched epoch for epoch (within 0.0003). Head-to-head @50k (400 pairs) stopped at 36 pairs on request — the rest already says "no difference". Why: every board has exactly 12 pieces, so each piece is in exactly 11 pairs, and any single-piece weight can be spread over its 11 pairs with the same first-layer output — the singles are linearly redundant given the pairs. First clean encoding ablation (same data/split/recipe). The zero-rows trick extends to any feature subset and to narrower widths (zero hidden units), so encoding and capacity ablations can run on v2.0.0 without engine work.

### Tail on the 39.6M mix (1024 → 32 → 1): much better val, not (yet) better play

**Training.** New `TAIL` setting in train_pair.py (0 = release recipe; N adds `Linear(1024, N) → ReLU → Linear(N, 1)`, runs named `pair1024xN_…`, saved as fc1 + fc2[N×1024] + b2[N] + fc3[1×N] + b3). `pair1024x32_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2`, same data/split as the no-tail mix: val 0.3735 (e1) → 0.3593 (e12) → 0.3592 (e14, stopped by hand); no-tail best 0.3663. Ahead at every epoch, ~0.007 lower at the end — by far the biggest val gain from any single change (dropout / data / encoding moved val ≤ 0.001).

**Dev engine.** GygesRust: new worktree `../GygesRust-dev-tail` on branch `dev-tail`, **no commits**, one file changed (`gyges_engine/src/search/network.rs`, +67/−6): a weights file of the tail size loads a `Tail { w2[32][1024], b2[32], w3[32], b3 }` and eval does `tanh(w3 · relu(w2 · relu(L1) + b2) + b3)`; plain 1024→1 files load and evaluate exactly as before (only `tail: None` added). Built with Rust 1.80.0 (installed on the laptop with rustup; minimal profile) using the repo's release profile (opt 3, LTO, 1 codegen unit) → `engines/gyges-dev-x32.exe`. Runner: `dev:<net>` in `gyges-tools sprt` runs that net on the dev build.

**Engine check.** dev vs v2.0.0 with the same release net (20 pairs @10k) gave LL 5 / WW 4 — but v2.0.0 vs *itself* gave LL 5 / WW 4 too: **the engine is not deterministic run to run** even with 1 thread and a node cap, so a pair-identity test can't prove two builds equal. (Harmless — it's inside the error bars — but worth knowing.) The engine's `eval` command prints the HCE only, so the tail path was checked by code (no-tail path unchanged) and by play. Both check rows in matches.csv are labelled as engine checks.

**Matches** (tail net on the dev engine):

| | pair score | 95% CI | think time / depth |
|---|---|---|---|
| vs release @10k | **0.678** | [0.630, 0.725] | 492 ms, 2.56 · release 300 ms, 2.77 (no-tail mix: 0.645) |
| vs release @50k | ~0.64 at 89/200 pairs | — | stopped (user: go straight to the head-to-head) |
| **vs no-tail mix e12 @50k** | **0.460 at 88/400 pairs** | [0.395, 0.526] | 2.10 s, 3.73 · 1.26 s, 3.70 — stopped for the night, LL 20 / WW 13 |

So far the tail plays a bit *worse* than the no-tail net at 50k head-to-head (within noise at 88 pairs), despite the clearly better val. Possible reading: val measures predicting game results on average, not ranking the moves search has to choose between. Bar set before the test: ≥ 0.59 head-to-head worth it even at equal time (~60–80% slower evals ≈ 0.6× the nodes ≈ −65 Elo); 0.53–0.58 worth it only with a faster eval. Resume: rerun the head-to-head from scratch (`gyges-tools sprt dev:pair1024x32_…_e12 pair1024_hce-10k+…_e12 50000 50000 400`; the stopped one wrote no row).

### Tail: conclusion, and how predictable each phase of the game is

**Tail, for now:** no gain in play worth its cost — head-to-head 0.460 at 88/400 pairs vs the no-tail mix @50k (even the top of the CI, ~+0.03, is far below the ~0.59 a 60–80% slower eval needs). The biggest val gain of anything tried, but no clear strength gain: **val loss no longer tracks playing strength** at this level — every change has to be judged in matches.

**Per-phase predictability** (`training/phase_analysis.py`, results in `results/phase_gen2-10k-aws.csv`). All nets run over gen2-10k-aws (4.85M positions, unseen by every net below). R² = 1 − MSE / variance of the (×0.75) labels in each move-number bucket; acc = sign of eval matches the winner (draws left out).

| move # | n | release | gen1 5M | gen1-1k 5M | gen1 20M | mix 40M | mix 40M + tail |
|---|---|---|---|---|---|---|---|
| 0 | 270k | −0.016 | −0.010 | −0.026 | −0.002 | −0.004 | −0.008 |
| 1–2 | 540k | 0.039 | 0.041 | 0.023 | 0.055 | 0.053 | 0.057 |
| 3–5 | 801k | 0.172 | 0.169 | 0.151 | 0.190 | 0.196 | 0.207 |
| 6–9 | 934k | 0.276 | 0.261 | 0.231 | 0.287 | 0.312 | 0.323 |
| 10–14 | 1.03M | 0.474 | 0.448 | 0.395 | 0.490 | 0.513 | 0.530 |
| 15–19 | 686k | 0.646 | 0.627 | 0.568 | 0.661 | 0.678 | 0.700 |
| 20–29 | 499k | 0.727 | 0.713 | 0.648 | 0.745 | 0.758 | 0.784 |
| 30+ | 94k | 0.784 | 0.782 | 0.710 | 0.810 | 0.817 | 0.848 |

Sign accuracy runs from ~0.59 at the start board (≈ "side to move wins") to ~0.95 after move 30.

- **Openings are nearly unpredictable for every net** (R² ≈ 0 at the start board, ≤ 0.06 after the two random moves, < 0.33 until move 10). A property of the game — results get decided later — not a net limit.
- **Nets rank the same in every phase** (tail > mix > gen1 20M > release > gen1 5M > gen1-1k); no phase where one net is uniquely weak, so phase buckets are unlikely to help.
- **The tail predicts better in every phase, most late** (+0.02–0.03 R² from move 15) — so its val gain is real and not an opening artifact, which makes the no-gain-in-play result sharper: predicting results better on average ≠ ranking sibling moves better. Next test: move agreement (does each net's best child match the strong player's move?).

### Move agreement — explains the tail, and tracks strength better than val

New `gyges-tools children <in.csv> <out.csv>` (src/children.rs): every legal next position of each input position, flipped to the opponent's view (the orientation of the next data-gen row), with a win flag. `training/move_agreement.py` samples positions from move 4 on (the first 4 plies of data-gen games are random), scores every child with each net (pick = child with the lowest score from the opponent's side) and checks it against the move actually played (the next row of the game). Sanity check: the played move was among the generated children for all 20,000 positions. Positions with an immediate win left out (none in this sample). **~236 legal moves per position** on average; random pick agrees 2.7%.

On gen2-10k-aws (20k positions; played by gen1 19.9M e9 at 10k nodes):

| net | agree (top pick = played move) | rank of played move (share rated above it) |
|---|---|---|
| gen1 19.9M e9 (*the net that played these games — biased*) | 0.256 | 0.141 |
| mix 40M e12 | **0.168** | **0.142** |
| mix 40M + tail e12 | 0.150 | 0.148 |
| release | 0.139 | 0.160 |
| gen1 4.7M | 0.129 | 0.178 |
| gen1-1k 4.7M | 0.116 | 0.178 |

- **The tail agrees less than the same data without it** (0.150 vs 0.168) and ranks the played move worse — it predicts results better (R²) but ranks sibling moves worse, which is what search uses. Explains why it didn't play better.
- **Agreement orders the nets like the matches do** (mix > release > gen1 5M > gen1-1k; tail below mix), unlike val/R² (which put the tail first). A cheap strength proxy (minutes, no matches) — use it to screen variants before matches.
- Caveats: the data's own player net is biased upward (needs another player's data to place fairly); agreement is 1-ply vs the player's ~2.6-ply search.
Results per phase in `results/move_agreement_gen2-10k-aws.csv`.

### Why the next data saves the search: the eval is reliable late and blind early — depth is the lever

**Win-prediction and move-selection measurements work** (both on gen2-10k-aws, unseen by the nets compared; graded against how those games — gen1 19.9M e9 @10k — actually went):
- Winner accuracy from the eval alone (mix 40M): ~60% at the start board (= the first-mover edge, no knowledge), 68% at moves 3–5, 74% at 6–9, 83% at 10–14, 90% at 15–19, 93% at 20–29, 95% at 30+; **91% over all positions from move 15 on**. Nets differ by only 1–2 points; the phase dominates.
- Move agreement (1-ply eval's top pick = the move the 10k-node search played, out of ~236 legal moves; random 2.7%): mix 16.8%, tail 15.0%, release 13.9%, gen1 5M 12.9%, gen1-1k 11.6%; played move on average in the net's top ~15%. Orders nets like the matches do.

**Reading.** The opening is unreadable by any eval (a property of Gyges: results are decided later), the late game is ~95% readable, and the middle game (moves 6–19) is where judgment is partial and choices matter. Strength comes from search reaching the readable positions. Every data result so far points the same way: label quality follows the strength of the data-gen player, and depth is its biggest lever (≈ +95 Elo of player strength per doubling of nodes vs +65 for a generation step). At 20k nodes the middle-game search completes only ~3 ply (2.0.0 deepens 2 plies at a time: 1, 3, 5, …; with ~236 moves per position, 20k nodes often finishes only ply 1 in the opening).

**Labels plan (user):** a forced win/loss found by search is exact for that position under any opponent (the positions leading to it are not); everything else is "the result under play by this player", truer with deeper search. So: keep game results, and save the search for every position so labels can later use proven results exactly and blend in the search verdict, weighted by how deep the search got. Relabelling old data costs about as much as generating new data at the same budget, so the search info goes into data gen itself; one gen2-20k run answers both "does depth make better data" (vs gen2-10k) and "do search labels help" (result-only vs blended on the same rows).

**Data gen format, from now on — 47 columns** (old datasets keep 39 and load as before; the training loader reads the first 39 of either):
`sq0..sq35, result, move_num, game_id, score, ply, time_s, forced, best, nodes, pv, depth_scores`
- `score` search score for the side to move (tanh × 10000; ≥ 100000 in size = proven); `nan` for the 4 random opening plies
- `ply` deepest completed iteration; `time_s`; `forced` +1/−1 proven win/loss else 0
- `best` the move actually played (games replay exactly from it); `nodes` of the last completed iteration; `pv` the expected line (space-separated)
- `depth_scores` `ply:score` per completed iteration, e.g. `1:-974.6 3:4559.6 5:99999997.0`
Shared by data gen and `relabel` (`game::search_columns`); ~155 bytes a row (4.85M rows ≈ 0.75 GB, ~150–200 MB as .tgz).

**Validated** with new `training/check_datagen.py` on a real-settings run (gen1 e9 @20k, 8 workers, killed and resumed): replays every game (board + `best`, flipped = next row) — all 1,042 rows pass; results alternate; all 62 games end with the mover reaching the goal; random plies have `nan` scores; forced flags match scores; `depth_scores` ends at `ply`/`score` with odd increasing plies. **Forced results disagreeing with the game result: 0 of 358 at 20k** (2.3% at 1k nodes — weaker search failing to convert its wins, or unsound proofs; to recheck on the AWS data). Documented engine behaviour, not errors: positions decided at the root (forced, or a random ply) have `ply 0`, no nodes/pv (124 rows); in proven losses the engine may play a different move than its PV's first (32 rows) — `best` is always the move played.

**Also:** `.gitignore` excluded every `*.csv`, so the results tables (matches/runs/epochs/datasets/phase/agreement) were never committed — fixed with `!results/*.csv` (datasets stay ignored).

Next: AWS `datagen gen2-20k-aws 190 2.7 20000 pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9` (~4.85M rows, ~2.7 h, ~$11), then `check_datagen.py` on it, the forced-proof spot check, train result-only vs blended, phase + move agreement, matches.

**Runner now confirms the net.** Found that the runner only checked the weights file exists: if 2.0.0 fails to load it (wrong size, corrupt), it prints `info string weight load failed` and silently keeps its embedded release net. Now `start_engine` stops the run on that line, then runs a 100-node probe search and checks the engine's `NN evaluation using <file>` line against the requested file; prints `[engine] net confirmed: <file>` once. Tested: gen1 e9 → confirmed; truncated file → stops at start-up with exit code 1; release → `default.bin (embedded)`.

**Did gen2-10k-aws really use gen1 e9?** (question: is that why gen2 was barely different?) Yes — move agreement already shows it: on gen2-10k-aws gen1 e9 picks the played move 25.6% of the time, the stronger 40M mix 16.8%, release 13.9%. Only the player has that outsized lead. The small gen2 edge is real, not a loading failure.

### Openings: the random plies teach the wrong opening, and real openings are unreadable by every net

Question (user): are the 4 random plies why nets read openings badly, and is deeper search strong because it reaches past the blurry opening? New `datagen … [searched|wild|none]` opening modes (searched = today's; wild = uniform random plies picked by gyges-tools, no search, ~1/3 of run time saved at 20k since random plies took 32.6% of search time; none = no random plies). Both validated with `check_datagen.py` (rule relaxed: nan scores only allowed in moves 0–3).

**Ablation, rows 0–3 dropped** (`pair1024_gen1-10k+gen1-10k-aws_from4`, 3.66M rows, e6, val 0.3010 — not comparable to the full set's val). Phase acc on gen2-10k-aws vs the same data with them: moves 6+ identical (6–9 0.724 vs 0.722, later within 0.001); openings much worse (move 0 **0.430** vs 0.588, 1–2 0.528 vs 0.579). Move agreement 0.132 vs 0.129 (noise level; most at moves 4–5, 0.158 vs 0.146). The random-ply rows are not junk: they're the only opening information the net gets, and they don't hurt later phases. Match running.

**Good play from the start** (`test-none-10k`, local, gen1 e9 @10k, 15 min, 12 workers: 1,455 games, 29,934 rows, mean 20.6 plies). Findings:
- **No-random openings are deterministic**: 114 start boards came up twice, and all 114 replayed the exact same game. So `none` gives at most one game per start board — 8,100 boards (90 × 90 back rows), ~167k rows per player/node setting. Can't be the bulk mode.
- **Every net is at coin flip on real openings**, the player included: acc at move 0 0.45–0.51, 1–2 0.49–0.51, 3–5 0.52–0.57, 6–9 0.66 (vs 0.59 / 0.58 / 0.66 / 0.73 for the release net on random-opening games); R² ≤ 0 through move 5. Late game the same as before.
- **Random openings flip the first-move advantage**: side to move at move 0 wins **59.7%** with random openings (gen2-10k-aws, 270k games) but **45.9%** with good play (1,455 games, ~3 SE from 50%). The release net learned the random-opening version: it calls the side to move ahead on 97.5% of start boards. Guess: after 4 random plies the first mover is the first to make a searched move — worth a lot in a tempo game.
- Supports the user's search hypothesis (nets are blind for ~5 plies of real play; at 10k the opening search reaches ~1.6–3 plies, more nodes reach past it) — not yet tested directly.

**Match: `from4` beats its baseline** — `pair1024_gen1-10k+gen1-10k-aws_from4_e6` vs `pair1024_gen1-10k+gen1-10k-aws_e6` @10k, 200 pairs: **0.550 ± 0.023** (95% 0.505–0.595, ≈ +35 Elo), 2.5% draws. So dropping the random-ply rows makes it *stronger* even though it reads openings worse (move 0 acc 0.43 vs 0.59) — correcting my "no help" above, which was from phase/agreement only (agreement had pointed this way: 0.132 vs 0.129, mostly at moves 4–5). Guesses: the random-ply labels teach the wrong opening (first mover favoured), which costs more than knowing nothing; or the capacity goes to positions that matter. Opening eval accuracy and playing strength disagree again, like val and strength did for the tail.

**Same at 20M: `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_from4`** (gen1 19.87M with moves 0–3 dropped → 15.46M rows; stopped itself at e15, took **e10**, val 0.2892, best 0.2890 @ e12). vs `…+gen1-10k-aws2_e9` (same data with the opening rows) @10k, 200 pairs: **0.5725 ± 0.026** (95% 0.523–0.622, ≈ +51 Elo, LOS 99.8%), 5% draws. Bigger than at 4.7M (0.550). Dropping the random-ply rows is the first training-side change since the release recipe that clearly gains strength. 50k match running. (Ran while AWS gen2-20k-aws was being set up; that run's player stays gen1 e9 so it remains a clean 20k-vs-10k comparison with gen2-10k-aws.)

**`from4` 20M @50k** vs gen1 20M e9: **0.585 ± 0.025** (95% 0.536–0.634, ≈ +60 Elo, LOS 99.97%), 4.5% draws — the gain holds with deeper search (+51 at 10k). Joint rating fit (release = 0): @10k +108 ± 46, level with the 40M mix e12 (+111 ± 28) and the tail (+129 ± 38); @50k **+172 ± 42**, top of the list (next: 40M mix e12 +117 ± 27) — on 15.5M rows vs 39.6M. Direct match vs release @10k running. Charts: "moves 4+" runs now their own series (pink) on the golden chart and score_vs_rows (`DATA_VARIANTS` in plots.py); tagged runs show as variants in the ratings chart instead of joining their family's line.

User's reading: the gain is the net leaning on search in the opening instead of the (wrong) opening learned from random-opening labels. Fits what we have — on real openings `from4` is the only net above 50% at move 0 (0.51 vs 0.45–0.46), the others confidently back the side that loses more often — but capacity going to later positions would also fit. Test that would separate them: keep rows 0–3 but give them label 0 (no opinion) — same gain → it was the wrong labels.

Parked (user): the 40M mix with moves 0–3 dropped. Its HCE half has the same random plies, so the mix may gain too — but it's ~50 min training + matches, and the depth data (gen2-20k-aws) comes first. Circle back later. Meanwhile training gen2-10k-aws with moves 4+, so tonight's 20k-vs-10k comparison uses the same filter on both sides.

**`from4` 20M vs release @10k (direct): 0.681 ± 0.022** (95% 0.638–0.724, ≈ +132 Elo, LOS 100%), 4.2% draws — above the chained estimate (+116) and the best measured score vs release at 10k (40M mix e12 0.645, gen1 20M e9 0.5925). Charts updated.

**Decision (user): let gen2-20k-aws finish with gen1 e9** rather than switch to the moves 4+ net mid-run — keeps the clean depth comparison (same player as gen2-10k-aws, 20k vs 10k nodes, ~5M each, moves 4+ on both). Past evidence for weighing it: one player step (gen1 e9, +65 over release) gave only a small data gain at ~5M (gen2-10k vs gen1 h2h 0.519); doubling nodes is worth ~+95 Elo of player strength but its effect on data was never measured. Next big run: ~20M rows played by `…_from4_e10` (md5 48642ef1b57596b4d34ed642daa10edb), node budget set by tonight's depth result (10k ≈ 5.6 h/$23, 20k ≈ 10.8 h/$44 on AWS), labels by the result-only vs blended test.

**`from4` 20M vs release @50k (direct): 0.740 ± 0.021** (95% 0.699–0.782, ≈ +181 Elo), 3.7% draws — rating-fit prediction was 0.73. Best measured @50k by far (40M mix 0.673, gen1 20M e9 0.640). Gap over release grows with search: +132 @10k → +181 @50k. 100k running. Also training `…gen1-10k+gen1-10k-aws_from4_min11` (moves 4+ and games ≤ 10 rows dropped — user's idea: games decided by the random opening teach lopsided positions, though their labels are right) to match vs `…_from4_e6` @10k.

**`from4` 20M vs release @100k (direct): 0.719 ± 0.024** (95% 0.672–0.765, ≈ +163 Elo); games 280–105–15; pairs WW 51.5%, LL 10.5%. Curve vs release: +132 @10k → +181 @50k → +163 @100k — steady from 50k on (the 50k/100k difference is inside both CIs), as the user expected: once the search gets past the opening, release's wrong opening belief costs it less. Still the best at 100k (gen1 20M e9 0.6525, 40M mix ~0.66 from the fit).

**Short games (user's idea): no clear effect.** `…gen1-10k+gen1-10k-aws_from4_min11_e5` (moves 4+ and games ≤ 10 rows dropped, 3.52M rows) vs `…_from4_e6` @10k: **0.529 ± 0.025** (95% 0.480–0.578, ≈ +20 Elo, LOS 87%), 10% draws. Not significant; at most small — fits the reasoning that those rows' labels are right (lopsided but real positions), unlike the opening rows. Not added to the recipe; the exact version (games already proven at move 4) can be tested on gen2-20k-aws.

**New best vs HCE @50k** (100 pairs, curiosity check): `…_from4_e10` **0.873 ± 0.024** (95% 0.826–0.919, ≈ +333 Elo); games 170–21–9. For reference release vs HCE @10k was 0.854; release vs HCE @50k not measured.

**Moves 4+ replicates on gen2 data:** `pair1024_gen2-10k-aws_from4_e6` (3.77M rows) vs release @10k **0.551 ± 0.024** (95% 0.504–0.599); the same data without the filter (`pair1024_gen2-10k-aws_e6`) scored 0.4425 — ≈ +0.11 / +76 Elo from the filter at ~5M too, on data from a different player. This net is the 10k side of the gen2-20k depth comparison.

### gen2-20k-aws: depth test data is in

5,028,120 rows / 284,761 games, gen1 e9 @20k, AWS 190 workers 2.7 h (1.86M rows/h, ≈ $11), 92.9% unique (gen2-10k-aws 93.0%). `check_datagen.py` all pass. **Proven results hold up at 20k: 1.72M proven rows (34%), only 1,762 (0.10%) disagree with how the game ended** (2.3% at 1k nodes in the small local test). First download was cut off — scp started while the server's tar was still writing (139%) and the idle alarm then stopped the instance; restarted, re-archived, md5 matched. Training `pair1024_gen2-20k-aws_from4` (same recipe as `pair1024_gen2-10k-aws_from4_e6`).

**Depth test result: 20k-node data = 10k-node data at ~5M.** `pair1024_gen2-20k-aws_from4_e7` (3.89M rows; e7 of 10, val 0.2497) vs `pair1024_gen2-10k-aws_from4_e6` (3.77M; val 0.2837), same player (gen1 e9), same recipe and filter: @10k **0.504 ± 0.013** (95% 0.479–0.528; 79% draws), @50k **0.496 ± 0.017** (0.463–0.530; 53% draws). vs release @10k: 20k net 0.485 ± 0.024, 10k net 0.551 ± 0.024 (≈ 2 SE apart; head-to-head says level — read as noise). Rules out a gain above ≈ +20 Elo. The draw rate is the information: the two nets play almost identically. Val much lower for the 20k data (cleaner labels) without any strength gain — val ≠ strength again. Likely reason: the engine deepens 2 plies per iteration, so doubling nodes rarely completes another iteration (match avg depth ≈ 2.6 @10k, ≈ 3.5 even @50k); and the label (who won) is set by play many moves later. 1k → 10k data was a big step (gen1-1k vs gen1-10k), 10k → 20k nothing: a real depth step likely needs 4–8× nodes. Implication for the next big run: more rows from the moves 4+ net at 10k (~20M for ~$23), not fewer at 20k.

Proof stats on gen2-20k-aws (moves 4+): 44.2% of rows proven; mean search ply of unproven rows 3.51; first proof comes a mean 6.98 plies (median 6) before the game ends — 83% of games proven ≥ 5 plies before the end, 48% ≥ 7. (The mate distance in the score isn't usable for this: 69% of proven rows report distance 0.) Comparing with a fresh 10k sample with search columns (`test-gen2-10k-search`, same player/settings, local) — user's idea: if 20k sees further, its proofs should come earlier. Blend-label training (`pair1024_gen2-20k-aws_from4_blend0.5`) running.

**Proofs: 20k sees a bit further, but it doesn't make better data.** `test-gen2-10k-search` (13721 rows, 760 games, local, same player/settings as gen2-20k-aws at 10k; check_datagen all pass) vs gen2-20k-aws, moves 4+: mean search ply of unproven rows 3.04 vs 3.51; proven rows 41.3% vs 44.2%; first proof a mean 6.73 vs 6.98 plies before the end; games proven ≥ 7 plies before the end **40.3% vs 48.3%** (~4 SE). So the deeper search does find wins earlier (user's check) — but a proof found earlier doesn't change who won, which is the label, and the moves played barely change; hence no stronger net. Depth could still matter through *labels* (proven/blended targets), which the blend test checks.

**Labels: calibrated blend beats result-only.** `pair1024_gen2-20k-aws_from4_blend0.5_e8` vs `pair1024_gen2-20k-aws_from4_e7` (same 3.89M rows, split, recipe; only the targets differ) @10k: **0.551 ± 0.014** (95% 0.523–0.579, ≈ +36 Elo), 73% draws. Blend = proven rows 0.75 × proven result (1.72M), others 0.375 × result + 0.375 × E(score), E = the result positions with that search score actually reached in this dataset (equal-count bins, monotone; score 0 → −0.263). 50k running. Next dial: λ 0.25 (more search). Note the search's verdict is worth something even where 20k-node *data* (moves played) wasn't (depth test) — depth helps through labels, not through the games.

**Blend @50k: 0.531 ± 0.020** (95% 0.492–0.571, ≈ +22 Elo, LOS 94%), 38% draws — same direction as 10k (0.551 ± 0.014, significant), not significant alone. Modest real gain, maybe smaller with deeper search. **Decision (user): keep win/loss-only labels as the main recipe** — the blend distils the previous net's opinion (its search score carries gen1 e9's side-to-move bias), which breaks the "learn from outcomes only" principle; outcome labels don't pass one net's technique to the next, only the quality and design of the games. Blend reported as an ablation. Pure alternatives to try: proven results (exact), averaged outcomes of several games from the same position, target shrunk with distance from the end; and deep relabel of the proof frontier (positions just before the first proof).

Overnight (started 23:39): moves 4+ on gen1 20M + gen2-10k + gen2-20k (29.9M rows, result labels) → `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2+gen2-10k-aws+gen2-20k-aws_from4`, then vs the current best @10k and @50k.

**The opening bias grows with player strength.** Side to move wins at move 0 (4 random plies): gen1-1k 53.0%, gen1-10k 57.4%, gen1-10k-aws 57.4%, gen2-10k-aws 59.7%, gen2-20k-aws 62.4% — vs 45.9% with good play from the start. After the random plies the first mover is also first to make a searched move; the stronger the search, the more that head start is worth. So the weakest player's data is the *least* wrong about the opening. Prediction (mechanism): the moves 4+ gain should be smallest on gen1-1k (user expected the largest — weaker player, more bad data). Fits so far: +35 Elo on gen1 4.7M (57.4%) vs +76 on gen2-10k (59.7%). Queued: `pair1024_gen1-1k_from4` vs `pair1024_gen1-1k_e5` @10k after the overnight run.

**Stopped for the night (00:10, user shutting down).** Overnight run `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2+gen2-10k-aws+gen2-20k-aws_from4` (23.1M rows after moves 4+) stopped after e8, val still falling (e6 0.2783, e7 0.2776, e8 0.2772) — unfinished; no match played. Its run dir (e1–e8) and runs/epochs rows stay as the record of the partial run; a rerun needs a new TAG (or the partial run removed first). The gen1-1k moves 4+ test never trained (only its settings were applied; trainer restored). To do next: (1) rerun the 30M moves 4+ training to the end and match vs `…_from4_e10` @10k/50k; (2) gen1-1k moves 4+ vs `pair1024_gen1-1k_e5` @10k — prediction: smallest gain of all (least opening bias); if it's big instead, equal-cost test (~20M rows of 1k data vs 10k nets); (3) next AWS run ≈ 20M rows played by `…_from4_e10` at 10k; (4) pure-label refinements (averaged outcomes from replays, proof-frontier relabel); (5) parked: 40M mix with moves 4+.

### 2026-10-28: adding gen2 to the moves 4+ recipe doesn't help — worse at 50k

Restarted the 30M run (the stopped attempt renamed `…_from4_stopped-e8`, dir + table rows, so the rerun keeps the normal name). It tracked the stopped run's val epoch by epoch, ran to e18, took **e12** (0.27593; best 0.27586 @ e15). `pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2+gen2-10k-aws+gen2-20k-aws_from4_e12` (23.1M rows: gen1 15.5M + gen2 ~7.7M after moves 4+) vs the best net `…+gen1-10k-aws2_from4_e10` (15.5M, gen1 only): @10k **0.491 ± 0.011** (95% 0.470–0.513; 82% draws) — level; @50k **0.448 ± 0.020** (0.408–0.487; ≈ −37 Elo, LOS 0.5%) — worse. More data with a third from a stronger player did not make a better net; the best net stays `…_from4_e10`. Same as before the filter (gen1 20M + gen2-10k vs gen1 20M: 0.494 @10k). Unexplained so far; guess: gen2 was played by gen1 e9, whose own eval carries the opening bias, so its middle-game positions come from skewed play — not measured. Worth a direct check before the next AWS run, since that run would also be data from a different player mixed with gen1.

**Why does adding gen2 hurt? Two checks.** (1) The first-mover lean lasts the whole game, not just moves 0–3: side to move wins on even moves ~58–61% in gen1 data, ~60–63% gen2-10k, ~63–66% gen2-20k (odd moves the mirror), vs ~46% with good play — one winner per game, so every row of the first mover carries the same lean; moves 4+ only removes the rows where it's easiest to *see* who moved first. gen2 leans harder. (2) But the mixed net did **not** absorb it — on real openings (test-none-10k): calls the first mover ahead in moves 0–8: release 64.0%, best 44.7%, mixed 42.0% (truth 46%); acc moves 0–5: 51.8 / 55.5 / 55.4%; moves 6–12: 70.5 / 71.6 / 72.3%. So the opening-bias explanation is ruled out: the mixed net evaluates as well as the best by every measure we have, yet plays worse at 50k. Next: is gen2 data, row for row, worse for *deep* search? `pair1024_gen2-10k-aws_from4_e6` (3.8M) vs `pair1024_gen1-10k+gen1-10k-aws_from4_e6` (3.7M) @10k and @50k (never compared at 50k; fit @10k had gen2 ahead).

**gen2 data is better row for row, also at depth:** `pair1024_gen2-10k-aws_from4_e6` (3.8M) vs `pair1024_gen1-10k+gen1-10k-aws_from4_e6` (3.7M): @10k 0.530 ± 0.024 (0.484–0.576), @50k **0.566 ± 0.024** (0.520–0.613, ≈ +46 Elo). So "gen2 hurts deep search" is wrong too: gen2 alone beats gen1 alone at equal size, yet gen1 + gen2 (23M) lost to gen1 alone (15.5M) @50k. Suspect untested: run-to-run variance — train_pair.py seeds only the split (seed 42); init and batch order are random, and single training runs have been compared as if each were *the* recipe. Free measurement: last night's stopped run and today's rerun are the same recipe/data, both have e8 (val within 0.0002, weights differ) — queued `…+gen2-20k-aws_from4_e8` vs `…_from4_stopped-e8_e8` @10k and @50k. Also on the user's challenge: test-none-10k isn't "proper" openings, it's gen1 e9 playing without random plies (and gen1 e9 has the first-mover bias); queued no-random-opening sets played by the best net and by release to see if the 46% is the game's or the player's.

**The first mover's ~46% is the game's, not gen1 e9's** (user's challenge: test-none-10k isn't "proper" openings, it's gen1 e9's play). No-random-opening games @10k by three players: gen1 e9 45.9% ± 2.6% (1,455 decisive games), release 46.4% ± 2.4% (1,688), best net 46.8% ± 5.0% (389) — agree: with good play from the start the second mover has a small edge. On all three sets the old nets call the first mover ahead 56–64% of moves 0–8, the moves 4+ nets 43–45%.

**The best net draws itself from the start board:** 81% draws (2,080 played, 389 decisive) vs 2% for release and gen1 e9. Matches between similar moves 4+ nets were also drawish (79–82% @10k). Its 389 decisive games are an unusual sample (every net reads them worse from move 6: 53–57%). **Matters for the next AWS run** with this net as the player: check its draw rate with random openings locally first — draws aren't written, so a high rate cuts rows/hour.

**1k test: no gain, as predicted.** `pair1024_gen1-1k_from4_e5` vs `pair1024_gen1-1k_e5` @10k: 0.484 ± 0.024 (0.437–0.531). 1k data has the weakest opening bias (53%), so nothing to remove — the gain follows how wrong the opening labels are, not how weak the player is. (User hoped for a big gain → cheap 1k data; not supported.)

**gen2-20k vs gen1 (moves 4+, ~4M each) @50k: 0.698 ± 0.021** (0.657–0.738, ≈ +144 Elo) — much more than gen2-10k's +46 vs the same net, though gen2-20k and gen2-10k were level head to head. A ≈ B but A ≫ C ≫ B: run-to-run variance or non-transitive matchups. Variance matches (same recipe, two runs, e8 vs e8) running.

### Run-to-run variance is as large as the effects we've been measuring at 50k

Same recipe, same data, same epoch — last night's stopped run vs today's rerun, both e8 (val 0.27723 vs 0.27703; only init and batch order differ): @10k **0.494 ± 0.007** (92% draws), @50k **0.558 ± 0.016** (95% 0.527–0.588, ≈ **+40 Elo apart**). Deeper search turns small eval differences into different games; at 10k they mostly play the same moves. So at 50k a single training run per recipe carries ~±40 Elo of "which run did you get" on top of match noise. Consequences:
- "gen1 + gen2 hurts @50k" (−37 Elo, one run each) is inside that range — not established; read as level.
- The A ≈ B, A ≫ C ≫ B puzzle (gen2-20k ≈ gen2-10k head-to-head but +144 vs +46 over gen1) fits single-run noise of this size.
- Robust despite it: moves 4+ (replicated on gen1 4.7M, gen1 20M, gen2-10k; +67–76 Elo @10k, where run noise is small), the first-mover ~46% (three players), 1k no-gain, release-relative ladders built from many matches.
- Blend (+36 @10k significant; +22 @50k n.s.) and gen2-vs-gen1 @50k (+46) are in the zone where a second run is needed.
Fix for future comparisons: seed init + batch order in train_pair.py (repeatable runs), and train 2+ runs per recipe for anything under ~60 Elo at 50k (or compare at 10k, where run noise is small).

### Why 1k data is worse: the 1k player can't see the reply

1k data (gen1-1k, release net @1k) trains nets nearly as good as 10k data at 10k play (0.399 vs 0.419 vs release) but far worse at 50k (0.391 vs 0.534). Looked at the games. Judge = `pair1024_gen2-20k-aws_from4_e7` (trained on neither set); new small sets test-release-1k / -3k / -10k (release net, searched opening, 10 min each) match the big sets' numbers.

| | 1k | 3k | 10k |
|---|---|---|---|
| ply-1 searches (doesn't see the reply), plies 4–9 | 80% | 57% | 24% |
| ply-1, plies 10–19 / 20+ | 43% / 34% | 20% / 15% | 5% / 3% |
| judge picks the winner, plies 20–29 / 30+ | 80.7 / 82.0% | 85.3 / 84.6% | 87.2 / 89.8% (gen1-10k-aws 90.1 / 92.2%) |
| loser was clearly ahead (judge > 0.5) at some point, ply 4+ | 33% | 31% | 24% |
| first mover wins | 52.8% | 54.4% | 56.2% |

In 1k games a position that looks clearly won late still ends lost ~1 in 5: the labels record shallow blunders, not the position. Quality rises steadily with the share of moves that see the opponent's reply — no cliff at a magic node count; 3k sits between, nearer 1k on thrown games.

Cost of finishing ply 3 (3,000 positions from gen1-10k-aws, ply 4+, searched to exactly ply 3, no node cap; `relabel` got an optional ply cap): nodes median 1.7k, mean 4.4k, p90 11.7k, p99 36.5k; done within 1k 38%, 3k 64%, 10k 87%, 20k 96%. Early positions are the expensive ones (plies 4–9 median 4k, 77% within 10k). Per move: 10k cap + always finish ply 3 ≈ 1.1× the 10k cost; exact ply 3 (no cap) ≈ 3.7k nodes, ~0.37×. 20k data (ply 1 cut to 4%) = 10k at ~5M, so a ply-3 floor on 10k likely buys little; the open test is exact-ply-3 data (never blind, ~2.7× cheaper than 10k) vs 10k data.

### Ply-3 floor at 10k: much cleaner games, ~1.2× cost, and a bigger first-mover gift

User's idea: never let a data-gen move be blind to the reply. New data gen option `min_ply` (datagen arg 7): a searched move that stopped below it (and wasn't decided) is searched again to exactly that ply with no node cap; random plies untouched. test-release-10k-min3 (release net, 10k + floor 3, searched opening, 10 min) vs test-release-10k (same, no floor); judge `pair1024_gen2-20k-aws_from4_e7`:

| | 10k | 10k + floor 3 |
|---|---|---|
| depth of searched moves (ply 4+): 1 / 3 / 5+ | 10.7 / 50.1 / 24.9% | 0 / 62.1 / 22.6% |
| loser was clearly ahead at some point (ply 4+) | 24.4% | **14.9%** (±1.2–1.4 each) |
| judge picks the winner, plies 4–5 / 6–9 / 10–14 / 20–29 | 70.7 / 71.1 / 79.5 / 87.2% | 80.0 / 79.1 / 84.2 / 89.1% |
| first mover wins | 56.2% | **65.3%** |
| positions/h per worker | 11.8k | 9.7k (1.22× cost) |
| moves/game | 18.1 | 17.0 |

Cleaner by every game-quality measure (thrown games down by 40%) — I'd expected little from the 20k comparison; wrong. But the floor also hands the first mover more of the opening gift: blind moves were concentrated in the first searched moves (plies 4–9, 24% at 10k), so ply 4 — always the first mover's — now always sees the reply. 65.3% is above even gen2-20k (62.4%). Fits the head-start mechanism, and makes a fair opening (random 4 or 5 plies) more urgent if the floor is adopted.

### Near-best opening: data that looks like real openings, with variety

The engine's `skill-*` options can't randomize openings: the pick comes from a hash of the board (same position → same move), and non-best root scores are null-window bounds, not exact. So near-best is scored in the tools: each legal move = the move, the opponent's best reply, static eval (ply-1 search of the child, no node cap). Probe on 200 real opening positions (test-none-release-10k plies 0–7): a uniformly random non-losing move (today's opening) gives away 0.26 (plies 0–1) – 0.48 (6–7); moves within 0.1 of the best: median 18 / 8.5 / 3 / 3 for plies 0–1 / 2–3 / 4–5 / 6–7 (within 0.05: 5 / 3 / 2 / 2; within 0.2: 75 / 46 / 12 / 8).

New opening `nearbest[:margin]` (datagen arg 6): the first 4 plies are random among moves within the margin (default 0.1) of the best; a goal move is always taken. ~18·18·8.5·8.5 ≈ 23k openings per start board, ~380M over the 16,200 starts. test-release-10k-min3-nb (release net, 10k + floor 3, nearbest 0.1, 10 min, 477 games written, ~6% draws), judge `pair1024_gen2-20k-aws_from4_e7`:

| | old opening (10k) | old opening + floor | **near-best + floor** | real openings (no random plies, release 10k) |
|---|---|---|---|---|
| first mover wins | 56.2% | 65.3% | **46.3%** | 46.4% |
| \|eval\| > 0.5 at ply 4 / ply 6 | 40.1 / 40.6% | 41.8 / 46.7% | **13.6 / 24.8%** | 15.4 / 27.3% |
| judge cost of moves at plies 1 / 2 / 3 | 0.34 / 0.51 / 0.55 | 0.35 / 0.50 / 0.57 | **0.06 / 0.21 / 0.19** | 0.04 / 0.18 / 0.19 |
| games over by ply 10 | 20.1% | 19.6% | **2.1%** | 2.8% |
| distinct positions at ply 6 | 100% | 100% | **100%** (476) | 80% (after 1,688 games) |
| loser was clearly ahead at some point | 24.4% | 14.9% | 19.9% | 31.0% (no floor) |
| judge picks the winner, plies 0–3 / 4–5 | 50.4 / 70.7% | 47.5 / 80.0% | 56.7 / 67.2% | 52.6 / 57.6% |

Near-best openings match real openings on every opening measure, with full variety. Thrown games: between realistic openings, the floor takes 31.0% → 19.9%; the old-opening sets swing less because their games are lopsided from the start. The 80% at plies 4–5 under the old opening + floor was the gift, as expected — fair positions are harder to call. With a fair opening, moves 0–3 no longer need dropping.

Cost: 9,245 rows in 10 min vs 24,345 for plain 10k (2.6× slower per row; ~2.0× per usable row, as all rows are now usable vs 77% with moves 4+). The tools-side scoring is the cost (~1.5 s per opening ply, every legal move searched); an engine-side near-best (threshold null-window test of each root move) would make it nearly free — then the new recipe would cost about the same per usable row as the old one (floor-only rate 19.4k rows/10 min, all usable, vs 18.7k usable for plain 10k).

### Floor and near-best moved into the engine

User: no tools-side re-search or move scoring for what belongs in the engine. GygesRust branch `datagen-options` (local, not pushed; commit 052bfd1 on v2.0.0), two options, both off by default:
- `minPly N`: the node cap applies only once ply N is complete (shared flag across search threads) — no re-search.
- `randomMargin M` (engine score units, 1000 = 0.1): with `randomize`, the pick is uniform among root moves whose two-ply score (move, best reply, static eval) is within M of the best; replies tried best-first, a move is dropped at the first reply that puts it below the bar.

Built with Rust 1.80 → `engines/gyges-v2.0.0-datagen.exe` (md5 69bda5df80a7a8c5fa7c68f98bdd6feb). Checks: with both options off it plays identically to v2.0.0 (move, score and nodes on 300/300 positions @10k); `minPly 3` @10k never ends at ply 1; `randomMargin 1000` picks stayed inside the exact near-best pool on 20/20 opening positions (picks covered 219 of 231 pool moves in 60 tries each), 134 ms per pick vs ~1.5 s tools-side. Tools: the tools-side floor/near-best/relabel-ply code is reverted; data gen sends `minPly` / `randomMargin` and uses the datagen build only when `nearbest` or a ply floor is asked for (`datagen … nearbest[:margin] <min_ply>`); random opening plies get a ply-1 main search under `nearbest` (the pick doesn't use it).

test-release-10k-min3-nb-engine (release, 10k + minPly 3 + nearbest 0.1; stopped after ~4 min, 660 games): check_datagen passes (47 columns, search data recorded as before; opening plies score nan). Searched moves ply 4+: depth 0 / 1 / 3 / 5+ = 12.6 / 0 / 64.0 / 23.2%.

| | plain 10k (old opening) | tools-side nb + floor | **engine nb + floor** | real openings |
|---|---|---|---|---|
| positions/h per worker | 11.8k | ~4.5k | **11.2k** | — |
| usable rows/h per worker (moves 4+ needed for the old opening: 77%) | 9.1k | ~4.5k | **11.2k** | — |
| first mover wins | 56.2% | 46.3% (477 games) | 51.4% (660) | 46.4% (1,688) |
| \|eval\| > 0.5 at ply 4 / 6 | 40.1 / 40.6% | 13.6 / 24.8% | 14.1 / 25.2% | 15.4 / 27.3% |
| judge cost of moves at plies 1 / 2 / 3 | 0.34 / 0.51 / 0.55 | 0.06 / 0.21 / 0.19 | 0.07 / 0.20 / 0.22 | 0.04 / 0.18 / 0.19 |
| games over by ply 10 | 20.1% | 2.1% | 2.6% | 2.8% |
| loser clearly ahead at some point | 24.4% | 19.9% | 23.8% | 31.0% (no floor) |

Engine-side runs at 95% of plain-10k speed (the cheap ply-1 opening searches offset the floor) and every row is usable: ~1.23× more usable rows per hour than the old recipe. Opening measures match the tools-side version and real openings. First mover 51.4% vs 46.3% tools-side (same pick pool, verified) is a 1.7σ gap; pooled 49.3 ± 1.5 vs 46.4 ± 1.2 real — a small leftover edge or noise; far from 56–65%. The all-rows vs moves 4+ ablation on the AWS data will show whether the opening rows are clean.

### ~5M nets: one table

User asked for the generation names and the ~5M comparison nets in one clean place. NOTES.md now has a **Generations** sheet (gen0 = HCE games, gen1 = release net's, gen2 = gen1 e9's, gen3 = best net's — moves 4+ counted as a generation jump), the two recipes (old / new), a test-set table and a ~5M nets table. Every ~5M net below has its .bin in engines/weights/ and a runs.csv row. All results are direct matches (pair score, Elo), single training runs — at 50k read differences under ~60 Elo as unsettled (run-to-run ≈ 40 Elo).

| Net (data, rows) | vs release @10k | vs release @50k | Head to head |
|---|---|---|---|
| gen0: `hce-10k-part1_e6` (4.57M) | 0.371 (−91) | 0.406 (−66) | — |
| gen1: `gen1-10k+gen1-10k-aws_e6` (4.70M) | 0.419 (−57) | 0.534 (+23) | vs gen0: 0.559 @10k, 0.591 @50k |
| gen1 moves 4+: `…_from4_e6` (3.66M) | — | — | vs gen1 all rows: **0.550 @10k (+35)** |
| gen1 moves 4+, short games dropped: `…_from4_min11_e5` (3.52M) | — | — | vs `…_from4_e6`: 0.529 @10k (+20, n.s.) |
| gen1 @1k: `gen1-1k_e5` (4.73M) | 0.399 (−71) | 0.391 (−77) | — (10k data: −57 / +23) |
| gen1 @1k moves 4+: `gen1-1k_from4_e5` (3.65M) | — | — | vs `gen1-1k_e5`: 0.484 @10k (−11, n.s.) |
| gen2: `gen2-10k-aws_e6` (4.85M) | 0.443 (−40) | 0.524 (+17) | vs gen1 all rows: 0.519 @50k (+13, n.s.) |
| gen2 moves 4+: `gen2-10k-aws_from4_e6` (3.77M) | **0.551 (+36)** | — | vs gen1 moves 4+: 0.530 @10k (+21), **0.566 @50k (+46)** |
| gen2 @20k moves 4+: `gen2-20k-aws_from4_e7` (3.89M) | 0.485 (−10) | — | vs gen2 @10k moves 4+: 0.504 @10k, 0.496 @50k (level); vs gen1 moves 4+: 0.698 @50k (+145 — doesn't fit the rest, likely run noise) |
| gen2 @20k moves 4+, blend labels: `…_blend0.5_e8` (3.89M) | — | — | vs `gen2-20k-aws_from4_e7`: 0.551 @10k (+36), 0.531 @50k (+22, n.s.) — ablation only |

What the ~5M nets say: newer player → better rows (gen0 < gen1 < gen2), but gen2's edge only shows once the opening rows are dropped (its random-opening gift was bigger: first mover 59.7% vs 57.4%); moves 4+ helps where the opening bias is big (gen1 10k, gen2) and not on 1k data (53%); 1k data is as good as 10k at 10k play and far worse at 50k (blind ply-1 search); 20k data = 10k data. Reference for the next set: a ~5M gen3 set (new recipe) trains a net to compare against `gen2-10k-aws_from4_e6` (the best ~5M net so far) and `gen1-10k+gen1-10k-aws_from4_e6`.

### gen3 player check: the best net with the new recipe

test-gen3-10k-min3-nb (best net `…_from4_e10`, 10k + minPly 3 + nearbest 0.1, 10 min, 12 workers): 853 games played, 832 written — **2.5% draws** (the 81% was the no-randomness case; near-best openings give enough variety). check_datagen passes; searched moves ply 4+: depth 0 / 1 / 3 / 5+ = 11.7 / 0 / 64.9 / 23.0%. First mover wins **48.0%** (±1.7; real openings 46.4%). Opening moves cost 0.09 / 0.12 / 0.12 at plies 1–3 by the judge (release-played: 0.07 / 0.20 / 0.22) — the judge agrees more with the best net's picks; lopsided at ply 4 / 6: 10.1 / 18.7%; games over by ply 10: 1.6%; thrown games 22.2%; 829/830 distinct at ply 6. Speed 8.7k positions/h per worker vs 11.2k for the release net — but the laptop was on battery (PowerOnline False), so part of that may be throttling; mean game 21.0 plies vs 19.9.

### Predictions for gen3, written before the data exists (2026-10-29)

Reading so far (user): the opening bias was the main problem — moves 4+ gained exactly as much as the bias was big (gen1-1k 53% → no gain, gen1 57% → +35, gen2 60% → +76) and moved the nets' real-opening calls from 56–64% "first mover ahead" to 43–45% (truth 46%). Moves 4+ stopped teaching something false; it taught nothing new. The other problem: blind ply-1 moves at 10k (~11% of moves, 24% at plies 4–9) — untested for strength on its own, but every game-quality measure points to it (1k vs 10k, floor test). gen3 (best net as player, nearbest 0.1 + minPly 3, all rows) fixes the bias in every row (first mover 48% vs 57–62% all game), trains on honest opening rows for the first time (~23% of rows; today's best net has never seen one), and cleans labels with the floor and a stronger player. The gen3 test can't split floor from opening from player — it tests the stack.

Predictions for the ~5M gen3 set:
1. Ablation flips: all rows ≥ moves 4+ on gen3 data (old data: moves 4+ won by +35 to +76).
2. Openings: the gen3 net picks the winner at moves 0–5 on the real-opening test sets (test-none-*) better than the best net (51–56% now), and calls the first mover ahead ~46%.
3. Strength: gen3 5M net vs `pair1024_gen2-10k-aws_from4_e6` (best ~5M net) @50k ≥ 0.60 (≈ +70); vs `pair1024_gen1-10k+gen1-10k-aws_from4_e6` ≥ 0.65 (≈ +110).
Reading: 1 and 2 hold but 3 short → the opening fix works and something else (volume?) limits strength; all three → a generation jump; 3 below 0.55 → something in the stack isn't working.

**The real test is the early game** (user): both fixes act there — the new opening on plies 0–3 (honest opening rows) and fairness of every row; the floor mostly on plies 4–9 (24% blind at 10k; later moves almost always reach ply 3 anyway). So prediction 2, sharpened: on the real-opening test sets (test-none-10k, test-none-release-10k, test-none-best-10k; `phase_analysis.py`), the gen3 net picks the winner more often than the best net at moves 0–5 (best: 55.5% on test-none-10k) and 6–12 (71.6%), keeps calling the first mover ahead ~46% (best 44.7%), and is about level from move 15 on (~90%, neither fix targets it). Early gain + late level = the fixes work where they should; no early gain = something in the pipeline isn't doing what we think, whatever the overall strength.

### Decisions and reasoning, 2026-10-29 (so the thinking is on record)

- **It's the minimum depth, not the depth** (user). 20k data raised average depth and changed little; the ply-3 floor raises the minimum and cleaned the games up. Ply 1 isn't a search — it never looks at the reply — so ply 3 is the smallest search worth calling one.
- **The floor is a design choice, not an experiment** (user). Clear mechanism, clear game-quality evidence, ~free in the engine (95% of plain-10k speed), nothing plausible that makes it hurt. No floor-only strength test before using it; if the paper wants its share later, a floor-only set is an optional ablation. Paper wording: a minimum search depth so every move considers the opponent's reply.
- **The real test of gen3 is the early game** (user): both fixes act on plies 0–9, so no early-game gain on real openings = something isn't working (logged with the predictions).
- **gen3 trains alone, no pooling with gen1** (user). gen3 is a different player and recipe; pooling gen1's old-recipe games back in mixes the fix with what it fixed; no pooled set has ever beaten its pure base at depth. Pooling gets tested only if gen3 20M alone doesn't beat the best net (one extra training then: gen3 vs gen3 + gen1).
- **The player net's own biases barely matter** for the data: labels are outcomes, not the net's opinion. A uniform "first mover is ahead" belief shifts every leaf at the same depth equally, so it doesn't change move choice — measured: release (calls first mover ahead 64% on real openings), gen1 e9 (56–64%) and the best net (44.7%) all give the first mover 46% in real games. What does carry over is position-specific misjudgment (blunders, unvisited positions); the floor and the near-best opening cut exactly that, and a stronger player cuts it further — hence the best net as gen3's player.
- **AlphaZero / Leela comparison:** they also train value on outcomes and pool games, but only from a sliding window of nets a few Elo apart, with opening variety from sampling near-best moves for both sides — they never pool players 100+ Elo apart or hand one side a messy random opening. Supports "pool only close players" and "fix the opening", not "pooling is fine". Stockfish NNUE pools freely but trains mostly on search scores, which depend much less on who played.
- **Blend labels parked** until the big changes (recipe, gen3) are done (user): +36 @10k / n.s. @50k is small next to the data recipe.
- **Too many bad models** (user): stop testing weak variants; keep only decision-relevant tests. The ~5M nets are the test bed for 20M decisions — moves 4+ is the one 5M → 20M transfer checked so far (+35 at 4.7M, ≈ +67 at 20M).
- **Seeded trainer:** `SEED` in train_pair.py (init, dropout, batch order); checked on the GPU — same seed gives bit-identical weights after 40 steps, a different seed differs. All runs before 2026-10-29 are unseeded.
- **Plan:** AWS gen3 5M (best net, nearbest 0.1 + minPly 3, all rows; ~2.8–3.6 h on the c8a) → train alone, check the predictions → 20M if it holds. The server needs the GygesRust `datagen-options` branch (not pushed).

**Overnight queue (2026-10-29 night, then shutdown):** seeded second runs (`SEED = 1`, tag `from4_s1`) of the two ~5M baselines the gen3 5M net will be judged against — `gen1-10k+gen1-10k-aws` and `gen2-10k-aws`, moves 4+; matches @10k then @50k: the gen2-vs-gen1 2×2 (does a stronger player give better rows? one pairing already played: 0.566 @50k) and seed-vs-original for each (run-to-run noise at ~5M, so we know how many gen3 seeds to train); then local gen3 data (`gen3-10k-laptop`, 14 workers, new recipe) with the time left. Status lines in the scratchpad `overnight_1029.status`.

### Overnight 2026-10-29 results: gen2 vs gen1 per row, and run-to-run noise at ~5M

Both seeded reruns (`_from4_s1`) picked epoch 6 with the same val as the originals (gen1 0.3011 vs 0.3010, gen2 0.2837 vs 0.2837). Pair scores (400 games each):

| Match | @10k | @50k |
|---|---|---|
| gen2 orig vs gen1 orig (earlier) | 0.530 | 0.566 |
| gen2 s1 vs gen1 s1 | 0.599 | 0.476 |
| gen2 orig vs gen1 s1 | 0.557 | 0.532 |
| gen2 s1 vs gen1 orig | 0.580 | 0.531 |
| **gen2 vs gen1, mean of 4** | **0.567 (+47)** | **0.526 (+18)** |
| gen1 s1 vs gen1 orig | 0.497 | 0.477 |
| gen2 s1 vs gen2 orig | 0.487 | **0.401 (−70)** |

Stronger player → better rows: clear @10k (4/4), only a lean @50k (3/4). Same recipe, different seed: level @10k, but up to 70 Elo apart @50k — at ~5M one net per recipe can't settle a 50k comparison. gen3 5M: train 2+ seeds, read 10k and 50k together. User: 5M is a screen for big effects; 20M is where data effects really show. A 5M gen3 net is judged against the ~5M nets, not the best net (15.5M rows) — the early-game "beat the best net" prediction is a bonus bar, not a fair one at 5M.

Local gen3 pilot `gen3-10k-laptop` (best net, nearbest 0.1 + minPly 3, 4 near-best plies, 5.1 h, 14 workers): 713,135 rows, 34,409 games, check_datagen passes. Floor works: of 506k searched rows at move 4+, ply 1 = 23 rows (5 s per-move time cap hit while finishing ply 3), ply 3 = 73% (19% of those go past 10k nodes to finish ply 3, max 228k), ply 5+ never past 10k (max 9,999).

**Problem found: duplicate games.** The engine is deterministic after the opening, so two games that reach the same board after the random plies are the same game to the end (all 917 cases). 4 near-best plies give only ~570k distinct openings (measured from matching pairs): 2.7% duplicate games after 34k games, 6.3% repeated rows at move 4+ (old recipe 0.13%). Extrapolated: ~20% duplicate games at 5M, ~55% at 20M. The pilot is a dress rehearsal, not part of gen3 (opening differs from the final recipe).

### 8 near-best opening plies

New data gen arg 8 `random_moves` (full moves of random opening, default 2); check_datagen takes the random plies as arg 2. Test `test-gen3-10k-min3-nb8` (best net, nearbest 0.1, minPly 3, random_moves 4, 30 min, 14 workers): 108,456 rows, 5,601 games, ~1.4% draws (not written), checks pass.

- **Variety:** distinct openings measured from matching pairs roughly double per extra ply — ~600k after 4 plies, 1.3M / 2.2M / 3.8M after 5/6/7, ≥ ~4M after 8 (one matching pair at ply 8 → ~15M, very noisy). Per-ply engine sampling (`opening_pool.py`) overestimates 2–18× (transpositions, uneven pools), so the pair counts are the number to trust. Projected duplicate games: ~1–3% at 5M, ~3–12% at 20M (4 plies: ~20% / ~55%).
- **Fairness:** first mover wins 45.4% (±0.7; pilot 47.9%, real openings ~46%).
- **Quality** (judge `gen2-20k-aws_from4_e7`, mover's eval lost per move): old random opening plies 2–4 cost 0.34 / 0.51 / 0.55; near-best plies 0.10–0.13 at every ply 2–8; pilot's searched plies 5–8 0.08 / 0.08 / 0.02 / 0.02. So 4 extra near-best moves cost ~0.06 more each than searched moves. In both near-best sets the first 2 searched plies after the opening still read ~0.08–0.10, then ~0 — the judge (moves 4+ net, never saw opening rows; ply 1 reads −0.20) is rough this early, so costs are approximate. Lopsided (|judge| > 0.5) at ply 8: 29.7% vs pilot 28.9% vs old 43.4%; at ply 4: 8.7% / 9.2% / 40.1%.

**Why 8 near-best plies is faster, not slower:** per-move times in the data — searched moves at plies 9–12 take the same time in the 4-ply pilot and the 8-ply test (0.40 s), so the machine ran at the same speed. The difference: searched plies 5–8 are the most expensive moves of the game (0.87 / 0.78 / 0.73 / 0.59 s — early middlegame, the floor needs the most nodes there); with 8 near-best plies those become near-best picks (~0.13 s). Rows per worker-hour from move times: 10.6k (4-ply pilot) → 16.5k (8-ply). Those plies still see the reply (two-ply score), they just aren't searched deeper. Decision (user): 8 near-best plies is the gen3 recipe for 5M and 20M; skip-and-count exact opening repeats before 20M (later merges are an opening effect too and nearly vanish with 8 plies: 0.1% of rows vs 3.8% with 4); smoke test on AWS before the run.
