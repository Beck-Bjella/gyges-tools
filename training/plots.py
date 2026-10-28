"""Figures from the results/ tables. Re-run after any new match or training run:

    python training/plots.py

Writes results/figures/*.png and *.pdf. Every number comes from
results/matches.csv, runs.csv, epochs.csv and datasets.csv; nothing is typed in.

Elo vs the release net: from a direct match when there is one, otherwise
chained through one intermediate net matched at the same node budget (chained
points are drawn hollow). Elo = -400 log10(1/s - 1) on the pair score.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
RELEASE = "pair1024_hce-10k_e9"

# Reference palette, light mode (validated categorical order; first three
# pass all-pairs, so scatter-style charts stay within them).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 8,
})


# ── Data ────────────────────────────────────────────────────────────────────

def load():
    m = pd.read_csv(os.path.join(RESULTS_DIR, "matches.csv"))
    runs = pd.read_csv(os.path.join(RESULTS_DIR, "runs.csv"))
    epochs = pd.read_csv(os.path.join(RESULTS_DIR, "epochs.csv"))
    data = pd.read_csv(os.path.join(RESULTS_DIR, "datasets.csv"))
    # Only equal-budget matches with at least 100 pairs count as points.
    m["nodes_a"] = m["nodes_a"].astype(str)
    m["nodes_b"] = m["nodes_b"].astype(str)
    return m, runs, epochs, data


def elo(s):
    s = min(max(s, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / s - 1)


def elo_se(s, se):
    """Pair-score standard error -> Elo standard error (delta method)."""
    s = min(max(s, 1e-4), 1 - 1e-4)
    return 400 / math.log(10) / (s * (1 - s)) * se


def direct(m, a, b, nodes, min_pairs=100):
    """(score of a vs b, stderr) at equal nodes, from either side, or None."""
    n = str(nodes)
    for x, y, flip in ((a, b, False), (b, a, True)):
        r = m[(m.net_a == x) & (m.net_b == y) & (m.nodes_a == n) & (m.nodes_b == n) & (m.pairs >= min_pairs)]
        if len(r):
            r = r.iloc[-1]                       # latest if repeated
            s = 1 - r.pair_score if flip else r.pair_score
            return s, r.stderr
    return None


def elo_vs(m, net, ref, nodes):
    """(elo, se, chained) of net vs ref at nodes, chaining through one net if needed."""
    if net == ref:
        return 0.0, 0.0, False
    d = direct(m, net, ref, nodes)
    if d:
        return elo(d[0]), elo_se(*d), False
    others = set(m.net_a) | set(m.net_b)
    for mid in sorted(others - {net, ref}):
        d1, d2 = direct(m, net, mid, nodes), direct(m, mid, ref, nodes)
        if d1 and d2:
            e = elo(d1[0]) + elo(d2[0])
            return e, math.hypot(elo_se(*d1), elo_se(*d2)), True
    return None


def net_info(net, runs, data):
    """Training rows and data-source label for a net name `{run}_e{N}`."""
    run = net.rsplit("_e", 1)[0]
    r = runs[runs.run == run]
    if not len(r):
        return None
    r = r.iloc[0]
    parts = [p.split("@")[0] for p in r.dataset.split("+")]   # "name@N" = subset of name
    gens = sorted({int(data[data.dataset == p].generation.iloc[0]) if p in set(data.dataset)
                   else (0 if p.startswith("hce") else 1) for p in parts})
    if gens == [0]:
        source = "HCE data (gen0)"
    elif len(gens) == 1:
        source = f"gen{gens[0]} self-play"
    else:
        source = ("HCE + " if 0 in gens else "") + " + ".join(f"gen{g}" for g in gens if g) + " mix"
    # Data-gen node budget of every part (hce-10k itself is the two archive parts).
    known = dict(zip(data.dataset, data.player_nodes))
    nodes = sorted({int(known.get(p, known.get(p + "-part1", 0))) for p in parts})
    family = source + (f", {nodes[0] // 1000}k-node games" if len(nodes) == 1 and nodes[0] else "")
    # Run names are pair{H}[x{TAIL}]_{dataset}[_{TAG}]: anything besides pair1024_{dataset} is a variant.
    arch, _, tag = run.partition("_" + r.dataset)
    variant = ", ".join(v for v in (arch[len("pair1024"):] if arch.startswith("pair1024") else arch,
                                    tag.lstrip("_")) if v)
    return dict(run=run, rows=int(r.rows), dropout=r.dropout, source=source, family=family, variant=variant)


def fit_ratings(m, nodes, anchor=RELEASE, min_pairs=100):
    """One Elo rating per net from every equal-budget net-vs-net match at `nodes`.

    Weighted least squares on each match's Elo difference (weight 1/se^2), with
    `anchor` fixed at 0 — the same idea as engine rating lists. Returns
    {net: (elo, se)} for nets connected to the anchor through matches.
    """
    import numpy as np
    n = str(nodes)
    g = m[(m.nodes_a == n) & (m.nodes_b == n) & (m.pairs >= min_pairs)
          & m.net_a.str.startswith("pair") & m.net_b.str.startswith("pair")]
    # Keep only nets linked to the anchor.
    linked, frontier = {anchor}, [anchor]
    while frontier:
        x = frontier.pop()
        for a, b in zip(g.net_a, g.net_b):
            for p, q in ((a, b), (b, a)):
                if p == x and q not in linked:
                    linked.add(q)
                    frontier.append(q)
    g = g[g.net_a.isin(linked) & g.net_b.isin(linked)]
    nets = sorted(linked - {anchor})
    if not nets or not len(g):
        return {anchor: (0.0, 0.0)} if len(g) else {}
    idx = {x: i for i, x in enumerate(nets)}
    A = np.zeros((len(g), len(nets)))
    d = np.zeros(len(g))
    w = np.zeros(len(g))
    for k, r in enumerate(g.itertuples()):
        if r.net_a in idx: A[k, idx[r.net_a]] += 1
        if r.net_b in idx: A[k, idx[r.net_b]] -= 1
        d[k] = elo(r.pair_score)
        w[k] = 1 / max(elo_se(r.pair_score, r.stderr), 1e-6) ** 2
    AtW = A.T * w
    cov = np.linalg.pinv(AtW @ A)
    est = cov @ (AtW @ d)
    out = {anchor: (0.0, 0.0)}
    for x, i in idx.items():
        out[x] = (float(est[i]), float(math.sqrt(max(cov[i, i], 0))))
    return out


# ── Figures ─────────────────────────────────────────────────────────────────

def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_score_vs_nodes(m):
    """Pair score vs node budget for each head-to-head comparison."""
    panels = [
        ("vs the release net (pair1024_hce-10k_e9)", RELEASE, [
            ("pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9", "gen1 19.9M"),
            ("pair1024_gen1-10k+gen1-10k-aws_e6", "gen1 4.7M"),
        ]),
        ("vs gen1 19.9M (…aws2_e9)", "pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9", [
            ("pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12", "mix 39.6M, dropout 0.5"),
            ("pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_drop0.3_e7", "mix 39.6M, dropout 0.3"),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    budgets = [10_000, 50_000, 100_000]
    for ax, (title, ref, series) in zip(axes, panels):
        for k, (net, label) in enumerate(series):
            pts = [(n, direct(m, net, ref, n, min_pairs=40)) for n in budgets]
            pts = [(n, d) for n, d in pts if d]
            if not pts:
                continue
            # Dodge the two series sideways so overlapping CIs stay readable.
            xs = [n / 1000 * (1 + 0.04 * (k - 0.5)) for n, _ in pts]
            ys = [d[0] for _, d in pts]
            es = [1.96 * d[1] for _, d in pts]
            ax.plot(xs, ys, color=SERIES[k], label=label)
            for (n, _), x, y, e in zip(pts, xs, ys, es):
                partial = direct(m, net, ref, n, min_pairs=100) is None
                ax.errorbar(x, y, yerr=e, color=SERIES[k], marker="o", capsize=4, elinewidth=1.2,
                            markerfacecolor=SURFACE if partial else SERIES[k], markeredgewidth=2)
            # Direct label at the 10k point, where the series are furthest apart.
            above = ys[0] >= 0.5
            ax.annotate(label, (xs[0], ys[0]), xytext=(10, 22 if above else -20), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_xticks([b / 1000 for b in budgets], [f"{b // 1000}k" for b in budgets])
        ax.set_xlim(7, 190)
        ax.set_title(title, loc="left")
        ax.set_xlabel("nodes per move (both sides)")
        ax.set_ylim(0.36, 0.76)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("pair score (95% CI)")
    fig.text(0.06, -0.04, "Hollow = stopped early (fewer than 100 pairs). Dashed line = even.",
             fontsize=8.5, color=INK2)
    fig.suptitle("Head-to-head score vs search budget", x=0.06, ha="left", fontweight="bold")
    save(fig, "score_vs_nodes")


def fig_scaling(m, runs, data):
    """Elo vs the release net against training rows, one panel per match budget."""
    nets = sorted({n for n in set(m.net_a) | set(m.net_b) if n.startswith("pair1024_")})
    budgets = [10_000, 50_000, 100_000]
    sources = ["HCE data (gen0)", "gen1 self-play", "HCE + gen1 mix"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, n in zip(axes, budgets):
        for k, src in enumerate(sources):
            pts = []
            for net in nets:
                info = net_info(net, runs, data)
                if not info or info["source"] != src or info["dropout"] == 0.3:
                    continue
                e = elo_vs(m, net, RELEASE, n)
                if e:
                    pts.append((info["rows"], *e))
            if not pts:
                continue
            pts.sort()
            xs = [p[0] / 1e6 for p in pts]
            ax.plot(xs, [p[1] for p in pts], color=SERIES[k], label=src, zorder=2)
            for x, (_, y, se, chained) in zip(xs, pts):
                ax.errorbar(x, y, yerr=1.96 * se if se else None, color=SERIES[k], capsize=3, elinewidth=1, zorder=3,
                            marker="o", markerfacecolor=SURFACE if chained else SERIES[k], markeredgewidth=2)
        ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 40], ["5M", "10M", "20M", "40M"])
        ax.set_xlim(3, 55)
        ax.set_title(f"match budget {n // 1000}k nodes", loc="left")
        ax.set_xlabel("training rows (log)")
    axes[0].set_ylabel("Elo vs release net (95% CI)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.text(0.06, -0.04, "Hollow = chained through another net at the same budget. "
             "Dropout-0.5 runs only (part1: dropout not recorded). Release net = HCE data at 19.8M (0 by definition).",
             fontsize=8.5, color=INK2)
    fig.suptitle("Strength vs training data", x=0.06, ha="left", fontweight="bold")
    save(fig, "scaling")


# Families keep their color as new ones are added (color follows the family).
FAMILIES = [
    "HCE data (gen0), 10k-node games",
    "gen1 self-play, 10k-node games",
    "HCE + gen1 mix, 10k-node games",
    "gen1 self-play, 1k-node games",
    "gen2 self-play, 10k-node games",
    "gen2 self-play, 1k-node games",
]


def fig_strength(m, runs, data, nodes):
    """Every net on one chart: fitted rating vs training rows, one line per data family."""
    ratings = fit_ratings(m, nodes)
    # One point per run on the lines: the epoch with the most matches (the one in use).
    n_matches = lambda net: int(((m.net_a == net) | (m.net_b == net)).sum())
    main_net = {}
    for net in ratings:
        run = net.rsplit("_e", 1)[0]
        if run not in main_net or n_matches(net) > n_matches(main_net[run]):
            main_net[run] = net
    by_family, variants = {}, []
    for net, (e, se) in ratings.items():
        info = net_info(net, runs, data)
        if not info:
            continue
        pt = (info["rows"] / 1e6, e, se, net)
        if info["dropout"] == info["dropout"] and info["dropout"] != 0.5:     # tagged variant (NaN = unknown)
            variants.append((info["family"], f"dropout {info['dropout']:g}", *pt))
        elif info["variant"]:
            variants.append((info["family"], info["variant"], *pt))
        elif main_net[info["run"]] != net:
            variants.append((info["family"], "e" + net.rsplit("_e", 1)[1], *pt))
        else:
            by_family.setdefault(info["family"], []).append(pt)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for fam, pts in sorted(by_family.items(), key=lambda kv: FAMILIES.index(kv[0]) if kv[0] in FAMILIES else 99):
        color = SERIES[FAMILIES.index(fam)] if fam in FAMILIES else INK2
        pts.sort()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, label=fam, zorder=2)
        ax.errorbar(xs, ys, yerr=[1.96 * p[2] for p in pts], fmt="o", color=color, capsize=3,
                    elinewidth=1, zorder=3)
        for x, y in zip(xs, ys):                  # training rows above each point
            ax.annotate(f"{x:.1f}M", (x, y), xytext=(0, 9), textcoords="offset points",
                        fontsize=8.5, color=INK2, ha="center")
    for i, (fam, label, x, y, se, net) in enumerate(sorted(variants, key=lambda v: v[1])):
        color = SERIES[FAMILIES.index(fam)] if fam in FAMILIES else INK2
        x = x * (1.08 + 0.09 * i)                 # step sideways off the main point
        ax.errorbar(x, y, yerr=1.96 * se, fmt="D", color=color, markerfacecolor=SURFACE,
                    markeredgewidth=2, capsize=3, elinewidth=1, zorder=3)
        ax.annotate(label, (x, y), xytext=(9, 0), textcoords="offset points", fontsize=8.5,
                    color=INK2, va="center")
    ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("release net (v2.0.0)", (19.77, 0), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks, [f"{t}M" for t in ticks])
    ax.set_xlim(3, 60)
    ax.set_xlabel("training rows (log)")
    ax.set_ylabel("strength: Elo vs release net (95% CI)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"All nets, matches at {nodes // 1000}k nodes", loc="left")
    fig.text(0.02, -0.03, "One rating per net, fitted jointly from every net-vs-net match at this budget "
             "(release net = 0). +100 Elo = 64% expected score. Hollow diamond = training variant.",
             fontsize=8.5, color=INK2)
    save(fig, f"strength_{nodes // 1000}k")


# Golden chart encodings: color = generation of the data, marker = data-gen nodes,
# line style = data size. Colors stay fixed per generation as new ones are added.
GEN_COLORS = {"gen0": SERIES[0], "gen1": SERIES[1], "gen2": SERIES[2], "gen3": SERIES[7], "gen4": SERIES[3],
              "mix gen0+gen1": INK2, "mix gen1+gen2": SERIES[6], "gen1, moves 4+": SERIES[4],
              "gen2, moves 4+": SERIES[5], "mix gen1+gen2, moves 4+": SERIES[7]}
# Run TAGs that change which rows are trained on (not how): shown as their own series.
DATA_VARIANTS = {"from4": ", moves 4+"}     # moves 0-3 (the random opening plies) dropped
NODE_MARKERS = {1_000: "^", 3_000: "s", 10_000: "o", 20_000: "D", 30_000: "P"}


def size_style(rows):
    if rows < 10e6:
        return (0, (5, 3)), "~5M"
    if rows < 30e6:
        return "-", "~20M"
    return (0, (7, 2, 1, 2)), "~40M"


def golden_nets(m, runs, data):
    """Every net with matches, one epoch per run (the one with the most matches),
    training variants (a run TAG, e.g. _drop0.3 or _nosingle) left out — except
    DATA_VARIANTS, which change which rows are trained on and get their own series."""
    nets = {n for n in set(m.net_a) | set(m.net_b) if n.startswith("pair")}
    n_matches = lambda net: int(((m.net_a == net) | (m.net_b == net)).sum())
    dataset_of = dict(zip(runs.run, runs.dataset))
    best = {}
    for net in nets:
        info = net_info(net, runs, data)
        if not info or (info["variant"] and info["variant"] not in DATA_VARIANTS):
            continue
        if info["run"] not in best or n_matches(net) > n_matches(best[info["run"]]):
            best[info["run"]] = net
    out = []
    known = data.set_index("dataset")
    for run, net in best.items():
        info = net_info(net, runs, data)
        parts = [p.split("@")[0] for p in runs[runs.run == run].iloc[0].dataset.split("+")]
        parts = sum((["hce-10k-part1", "hce-10k-part2"] if p == "hce-10k" else [p] for p in parts), [])
        gens = sorted({int(known.generation[p]) for p in parts if p in known.index})
        nodes = sorted({int(known.player_nodes[p]) for p in parts if p in known.index})
        gen = f"gen{gens[0]}" if len(gens) == 1 else "mix " + "+".join(f"gen{g}" for g in gens)
        if info["variant"]:
            gen += DATA_VARIANTS[info["variant"]]
        out.append(dict(net=net, gen=gen, nodes=nodes[0] if len(nodes) == 1 else None, rows=info["rows"]))
    return sorted(out, key=lambda d: (d["gen"], -(d["nodes"] or 0), d["rows"]))


def fig_vs_release_by_budget(m, runs, data):
    """The golden chart: each net against the release net at equal node budgets.
    Color = generation, marker = data-gen nodes, line style = data size. Measured
    points are filled; points with no direct match come from the rating fit (hollow)."""
    budgets = [10_000, 50_000, 100_000]
    fits = {n: fit_ratings(m, n) for n in budgets}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for d in golden_nets(m, runs, data):
        net = d["net"]
        color = GEN_COLORS.get(d["gen"], INK2)
        marker = NODE_MARKERS.get(d["nodes"], "o")
        style, size = size_style(d["rows"])
        xs, ys, es, measured = [], [], [], []
        for n in budgets:
            if net == RELEASE:
                s, se, direct_pt = 0.5, 0.0, True
            else:
                dd = direct(m, net, RELEASE, n, min_pairs=100)
                if dd:
                    (s, se), direct_pt = dd, True
                elif net in fits[n]:
                    r, rse = fits[n][net]
                    s = 1 / (1 + 10 ** (-r / 400))
                    se = rse * math.log(10) / 400 * s * (1 - s)
                    direct_pt = False
                else:
                    continue
            xs.append(n / 1000); ys.append(s); es.append(1.96 * se); measured.append(direct_pt)
        if not xs:
            continue
        nodes = f"{d['nodes'] // 1000}k-node games" if d["nodes"] else "mixed nodes"
        label = f"{d['gen']}, {nodes}, {d['rows'] / 1e6:.1f}M" + (" = release net" if net == RELEASE else "")
        ax.plot(xs, ys, color=color, linestyle=style, marker=marker, markersize=0, label=label, zorder=2)
        for x, y, e, meas in zip(xs, ys, es, measured):
            ax.errorbar(x, y, yerr=e if e else None, fmt=marker, color=color, capsize=3, elinewidth=1,
                        markersize=8, markerfacecolor=color if meas else SURFACE, markeredgewidth=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([b / 1000 for b in budgets], [f"{b // 1000}k" for b in budgets])
    ax.set_xlim(7, 140)
    ax.set_xlabel("nodes per move, both sides (log)")
    ax.set_ylabel("pair score vs release net (95% CI)")
    # Legend markers match the data markers (the line handles carry the style).
    handles, labels = ax.get_legend_handles_labels()
    for h in handles:
        h.set_markersize(7)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    ax.set_title("Each training set vs the release net, as search gets deeper", loc="left")
    fig.text(0.02, -0.04, "Color = generation of the data · marker = data-gen nodes (● 10k, ◆ 20k, ▲ 1k) · "
             "line = data size (dashed ~5M, solid ~20M, dash-dot ~40M).\n"
             "Filled = measured head-to-head (≥100 pairs); hollow = from the joint rating fit. "
             "Release net = 0.5 by definition.", fontsize=8.5, color=INK2)
    save(fig, "vs_release_by_budget")


def fig_score_vs_rows(m, runs, data):
    """Data size on x: pair score vs the release net against training rows, one panel
    per match budget. Color = generation, marker = data-gen nodes; a line joins the
    same recipe (generation + nodes) across sizes."""
    budgets = [10_000, 50_000, 100_000]
    fits = {n: fit_ratings(m, n) for n in budgets}
    nets = golden_nets(m, runs, data)
    recipes = {}
    for d in nets:
        recipes.setdefault((d["gen"], d["nodes"]), []).append(d)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, n in zip(axes, budgets):
        for (gen, nodes), ds in recipes.items():
            color = GEN_COLORS.get(gen, INK2)
            marker = NODE_MARKERS.get(nodes, "o")
            pts = []
            for d in ds:
                net = d["net"]
                if net == RELEASE:
                    pts.append((d["rows"], 0.5, 0.0, True))
                    continue
                dd = direct(m, net, RELEASE, n, min_pairs=100)
                if dd:
                    pts.append((d["rows"], dd[0], dd[1], True))
                elif net in fits[n]:
                    r, rse = fits[n][net]
                    s = 1 / (1 + 10 ** (-r / 400))
                    pts.append((d["rows"], s, rse * math.log(10) / 400 * s * (1 - s), False))
            if not pts:
                continue
            pts.sort()
            xs = [p[0] / 1e6 for p in pts]
            label = f"{gen}, {nodes // 1000}k-node games" if nodes else f"{gen}, mixed nodes"
            ax.plot(xs, [p[1] for p in pts], color=color, marker=marker, markersize=0, label=label, zorder=2)
            for x, (rows, s, se, meas) in zip(xs, pts):
                ax.errorbar(x, s, yerr=1.96 * se if se else None, fmt=marker, color=color, capsize=3,
                            elinewidth=1, markersize=8, markerfacecolor=color if meas else SURFACE,
                            markeredgewidth=2, zorder=3)
        ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ticks = [2, 5, 10, 20, 40, 100]
        ax.set_xticks(ticks, [f"{t}M" for t in ticks])
        ax.set_xlim(3, 60)
        ax.set_title(f"matches at {n // 1000}k nodes", loc="left")
        ax.set_xlabel("training rows (log)")
    axes[0].set_ylabel("pair score vs release net (95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    for h in handles:
        h.set_markersize(7)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, -0.02), ncol=4, fontsize=9)
    fig.text(0.06, -0.22, "Color = generation of the data · marker = data-gen nodes (● 10k, ◆ 20k, ▲ 1k). "
             "Filled = measured head-to-head (≥100 pairs); hollow = from the joint rating fit. "
             "Release net = 19.8M gen0 at 0.5.", fontsize=8.5, color=INK2)
    fig.suptitle("Strength vs training data size, by search budget", x=0.06, ha="left", fontweight="bold")
    save(fig, "score_vs_rows")


# Data-gen cost in laptop-hours (12 workers) per million rows, by (player kind, nodes).
# Measured on the laptop 2026-10-25: NN @10k 129k rows/h, NN @1k 1.27M rows/h.
# HCE @10k not timed here: estimated from think time vs the net at 10k (188 vs 214 ms/move).
COST_H_PER_M = {("nn", 10_000): 1e6 / 129_000, ("nn", 1_000): 1e6 / 1_270_000,
                ("hce", 10_000): 1e6 / 129_000 * 188 / 214}


def fig_cost_frontier(m, runs, data):
    """Quantity vs quality at equal cost: strength vs what the training data cost to make."""
    recipes = [  # (label, family index for color, [(net, dataset parts)], cost key, cost estimated?)
        ("HCE games @10k (gen0)", 0, [("pair1024_hce-10k-part1_e6", ["hce-10k-part1"]),
                                      (RELEASE, ["hce-10k-part1", "hce-10k-part2"])], ("hce", 10_000), True),
        ("gen1 games @10k", 1, [("pair1024_gen1-10k+gen1-10k-aws_e6", ["gen1-10k", "gen1-10k-aws"]),
                                ("pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9",
                                 ["gen1-10k", "gen1-10k-aws", "gen1-10k-aws2"])], ("nn", 10_000), False),
        ("gen1 games @1k", 3, [("pair1024_gen1-1k_e5", ["gen1-1k"])], ("nn", 1_000), False),
    ]
    rows_of = dict(zip(data.dataset, data.rows))
    budgets = [10_000, 50_000]
    fits = {n: fit_ratings(m, n) for n in budgets}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, n in zip(axes, budgets):
        for label, fam, nets, key, est in recipes:
            pts = []
            for net, parts in nets:
                rows = sum(int(rows_of[p]) for p in parts)
                cost = rows / 1e6 * COST_H_PER_M[key]
                if net == RELEASE:
                    s, se, meas = 0.5, 0.0, True
                else:
                    d = direct(m, net, RELEASE, n)
                    if d:
                        (s, se), meas = d, True
                    elif net in fits[n]:
                        r, rse = fits[n][net]
                        s = 1 / (1 + 10 ** (-r / 400))
                        se, meas = rse * math.log(10) / 400 * s * (1 - s), False
                    else:
                        continue
                pts.append((cost, s, se, meas, rows))
            if not pts:
                continue
            pts.sort()
            color = SERIES[fam]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, label=label,
                    linestyle=(0, (5, 3)) if est else "-", zorder=2)
            for cost, s, se, meas, rows in pts:
                ax.errorbar(cost, s, yerr=1.96 * se if se else None, fmt="o", color=color, capsize=3,
                            elinewidth=1, markerfacecolor=color if meas else SURFACE, markeredgewidth=2, zorder=3)
                ax.annotate(f"{rows / 1e6:.1f}M", (cost, s), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK2)
        ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ticks = [1, 3, 10, 30, 100, 300]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlim(1.5, 400)
        ax.set_title(f"matches at {n // 1000}k nodes", loc="left")
        ax.set_xlabel("data-gen cost (laptop-hours at 12 workers, log)")
    axes[0].set_ylabel("pair score vs release net (95% CI)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.text(0.02, -0.05, "Labels = training rows. At any cost (vertical line) the highest recipe gives the "
             "strongest net. HCE cost estimated from think time (dashed line). Hollow = from the rating fit.",
             fontsize=8.5, color=INK2)
    fig.suptitle("Quantity vs quality: strength for the same data-gen cost", x=0.02, ha="left", fontweight="bold")
    save(fig, "cost_frontier")


def fig_hce_equivalent(m):
    """Each net @10k against HCE at several node budgets. Where a line crosses
    50% is the HCE search that net's eval is worth (log-node interpolation)."""
    g = m[(m.net_b.str.startswith("HCE")) & (m.nodes_a == "10000") & (m.pairs >= 60)]
    order = [RELEASE, "pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9",
             "pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12"]
    labels = {order[0]: "release net (HCE data 19.8M)", order[1]: "gen1 self-play 19.9M",
              order[2]: "HCE + gen1 mix 39.6M"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, net in enumerate(order):
        r = g[g.net_a == net].assign(hn=lambda d: d.nodes_b.astype(int)).sort_values("hn")
        r = r.drop_duplicates("hn", keep="last")
        if not len(r):
            continue
        xs = list(r.hn / 1000)
        ys = list(r.pair_score)
        ax.plot(xs, ys, color=SERIES[k], label=labels[net], zorder=2)
        ax.errorbar(xs, ys, yerr=list(1.96 * r.stderr), fmt="o", color=SERIES[k], capsize=3, elinewidth=1, zorder=3)
        # 50% crossing: interpolate Elo linearly in log2(nodes) between bracketing budgets,
        # or between the single point and nothing (then no estimate).
        es = [elo(s) for s in ys]
        for (x0, e0), (x1, e1) in zip(zip(xs, es), zip(xs[1:], es[1:])):
            if e0 >= 0 >= e1 and e0 != e1:
                xc = x0 * 2 ** (math.log2(x1 / x0) * e0 / (e0 - e1))
                ax.plot([xc], [0.5], marker="v", color=SERIES[k], markersize=9, zorder=4)
                ax.annotate(f"≈{xc:.0f}k", (xc, 0.5), xytext=(0, -16), textcoords="offset points",
                            ha="center", fontsize=9, color=INK2)
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ticks = [10, 25, 50, 100, 200, 400]
    ax.set_xticks(ticks, [f"{t}k" for t in ticks])
    ax.set_xlim(8, 300)
    ax.set_xlabel("HCE nodes per move (log) — the net always searches 10k")
    ax.set_ylabel("net's pair score vs HCE (95% CI)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("How much HCE search is each net's eval worth?", loc="left")
    fig.text(0.02, -0.03, "Triangle = where the net scores 50%: the HCE node budget its 10k-node search equals.",
             fontsize=8.5, color=INK2)
    save(fig, "hce_equivalent")


def fig_training_curves(epochs):
    """Val loss per epoch. Panels only compare runs that share a val set."""
    panels = [
        ("gen1 self-play: 4.7M vs 19.9M rows", [
            ("pair1024_gen1-10k+gen1-10k-aws", "gen1 4.7M"),
            ("pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2", "gen1 19.9M"),
        ]),
        ("HCE + gen1 mix 39.6M: dropout 0.5 vs 0.3 (same val set)", [
            ("pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2", "dropout 0.5"),
            ("pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_drop0.3", "dropout 0.3"),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (title, series) in zip(axes, panels):
        for k, (run, label) in enumerate(series):
            e = epochs[epochs.run == run].sort_values("epoch")
            if not len(e):
                continue
            ax.plot(e.epoch, e.val, color=SERIES[k], marker="o", markersize=5, label=f"{label} val")
            tr = e.dropna(subset=["train"])
            ax.plot(tr.epoch, tr.train, color=SERIES[k], linestyle=(0, (3, 2)), linewidth=1.5, label=f"{label} train")
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8.5, ncol=2)
    axes[0].set_ylabel("MSE loss")
    fig.text(0.06, -0.04, "Train loss is measured with dropout on. Val sets differ between datasets, "
             "so compare val only within a panel.", fontsize=8.5, color=INK2)
    fig.suptitle("Training curves", x=0.06, ha="left", fontweight="bold")
    save(fig, "training_curves")


GEN1_E9 = "pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9"


def player_ladder(m):
    """Elo of each (player, nodes) vs HCE @10k, chaining match by match.

    A player is `HCE` or a net name, at a node budget. Each rung is measured
    against an earlier rung. Returns {(player, nodes): (elo, se)}.
    """
    rungs = [  # (player, nodes) measured against (ref player, ref nodes)
        ((RELEASE, 10_000), ("HCE", 10_000)),
        (("HCE", 100_000), (RELEASE, 10_000)),
        (("HCE", 200_000), (RELEASE, 10_000)),
        ((GEN1_E9, 10_000), (RELEASE, 10_000)),
        ((RELEASE, 1_000), (RELEASE, 10_000)),
        ((GEN1_E9, 1_000), (RELEASE, 1_000)),
    ]
    name = lambda p, n: f"HCE @{n // 1000}k" if p == "HCE" else p

    def score(a, na, b, nb):
        for x, nx, y, ny, flip in ((a, na, b, nb, False), (b, nb, a, na, True)):
            r = m[(m.net_a == name(x, nx)) & (m.net_b == name(y, ny))
                  & (m.nodes_a == str(nx)) & (m.nodes_b == str(ny))]
            if len(r):
                r = r.iloc[-1]
                return (1 - r.pair_score if flip else r.pair_score), r.stderr
        return None

    out = {("HCE", 10_000): (0.0, 0.0)}
    for (p, n), (rp, rn) in rungs:
        d = score(p, n, rp, rn)
        if d and (rp, rn) in out:
            base, bse = out[(rp, rn)]
            out[(p, n)] = (base + elo(d[0]), math.hypot(bse, elo_se(*d)))
    return out


def player_label(p, n):
    who = {RELEASE: "release net", GEN1_E9: "gen1 19.9M net", "HCE": "HCE"}.get(p, p)
    return f"{who} @{n // 1000}k"


def fig_player_strength(m):
    """Data-gen players on one Elo ladder, HCE @10k = 0 (chained where needed)."""
    ladder = player_ladder(m)
    roles = {(RELEASE, 10_000): "gen1-10k's player", (RELEASE, 1_000): "gen1-1k's player",
             (GEN1_E9, 10_000): "gen2-10k's player", (GEN1_E9, 1_000): "gen2-1k's player",
             ("HCE", 10_000): "gen0 data's player"}
    pts = sorted(((player_label(*k) + (f"\n({roles[k]})" if k in roles else ""), e, se)
                  for k, (e, se) in ladder.items()), key=lambda p: p[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ys = list(range(len(pts)))
    ax.set_ylim(-0.6, len(pts) - 0.4)
    for y, (label, e, se) in zip(ys, pts):
        ax.errorbar(e, y, xerr=1.96 * se if se else None, color=SERIES[0], marker="o", capsize=4, elinewidth=1.2)
        ax.annotate(f"{e:+.0f}", (e, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    ax.set_yticks(ys, [p[0] for p in pts])
    ax.set_xlabel("Elo vs HCE @10k (95% CI; each rung chained from the match against an earlier rung)")
    ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.grid(axis="y", visible=False)
    ax.set_title("Data-gen player strength", loc="left")
    save(fig, "player_strength")


SIZE_BANDS = [  # (upper rows, label, line style) — nets of similar size are joined by a line
    (7e6, "~5M", (0, (1, 2))),
    (15e6, "~10M", (0, (5, 3))),
    (30e6, "~20–25M", "-"),
    (1e12, "~40M", (0, (7, 2, 1, 2))),
]


def data_quality(m, runs, data, net, rows):
    """Row-weighted average strength of the players that made a net's data, or None."""
    ladder = player_ladder(m)
    known = data.set_index("dataset")
    parts = runs[runs.run == net.rsplit("_e", 1)[0]].iloc[0].dataset.split("+")
    parts = sum((["hce-10k-part1", "hce-10k-part2"] if p == "hce-10k" else [p] for p in parts), [])
    full = {p: int(known.rows[p]) for p in parts if "@" not in p and p in known.index}
    subset = [p for p in parts if "@" in p]
    weights = dict(full)
    for p in subset:                        # "name@N": whatever rows the full parts don't explain
        weights[p] = max(rows - sum(full.values()), 0) // len(subset)
    tot, elo_sum, se_sum = 0, 0.0, 0.0
    for p, w in weights.items():
        base = p.split("@")[0]
        if base not in known.index:
            return None
        key = (known.player[base], int(known.player_nodes[base]))
        if key not in ladder:
            return None
        e, se = ladder[key]
        tot += w
        elo_sum += w * e
        se_sum += w * se
    return (elo_sum / tot, se_sum / tot) if tot else None


def fig_data_quality(m, runs, data):
    """Size and quality on one chart: x = how strong the player(s) behind a net's data were
    (row-weighted for mixes), y = the net vs the release net. Color = generation, marker =
    data-gen nodes, and nets of similar size are joined by a line (style = size band)."""
    pts = []
    for d in golden_nets(m, runs, data):
        q = data_quality(m, runs, data, d["net"], d["rows"])
        if q:
            pts.append((d, *q))
    budgets = [10_000, 50_000]
    fits = {n: fit_ratings(m, n) for n in budgets}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for ax, n in zip(axes, budgets):
        placed = []
        for d, qe, qse in pts:
            if d["net"] == RELEASE:
                s, se, meas = 0.5, 0.0, True
            else:
                dd = direct(m, d["net"], RELEASE, n)
                if dd:
                    (s, se), meas = dd, True
                elif d["net"] in fits[n]:
                    r, rse = fits[n][d["net"]]
                    s = 1 / (1 + 10 ** (-r / 400))
                    se, meas = rse * math.log(10) / 400 * s * (1 - s), False
                else:
                    continue
            placed.append((d, qe, qse, s, se, meas))
        for upper, label, style in SIZE_BANDS:
            lo = max([0] + [u for u, _, _ in SIZE_BANDS if u < upper])
            band = sorted((p for p in placed if lo <= p[0]["rows"] < upper), key=lambda p: p[1])
            if len(band) > 1:
                ax.plot([p[1] for p in band], [p[3] for p in band], color=INK2, linewidth=1.2,
                        linestyle=style, zorder=1, label=f"{label} rows")
            elif band:
                ax.plot([], [], color=INK2, linewidth=1.2, linestyle=style, label=f"{label} rows")
        for d, qe, qse, s, se, meas in placed:
            color = GEN_COLORS.get(d["gen"], INK2)
            marker = NODE_MARKERS.get(d["nodes"], "o")
            ax.errorbar(qe, s, xerr=1.96 * qse if qse else None, yerr=1.96 * se if se else None,
                        fmt=marker, color=color, markersize=9, capsize=3, elinewidth=1, zorder=3,
                        markerfacecolor=color if meas else SURFACE, markeredgewidth=2)
            ax.annotate(f"{d['rows'] / 1e6:.1f}M", (qe, s), xytext=(7, 5), textcoords="offset points",
                        fontsize=8, color=INK2)
        ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(f"net matched at {n // 1000}k nodes", loc="left")
        ax.set_xlabel("data-gen player strength (Elo vs HCE @10k)")
        ax.set_xlim(-60, 420)
    axes[0].set_ylabel("trained net vs release net (pair score)")
    # Two legends: size bands (lines) and what the colors / markers mean.
    size_handles, size_labels = axes[1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    gens = sorted({d["gen"] for d, *_ in pts})
    keys = [Line2D([], [], color=GEN_COLORS.get(g, INK2), marker="o", linestyle="", markersize=8, label=g)
            for g in gens]
    keys += [Line2D([], [], color=INK, marker=NODE_MARKERS[k], linestyle="", markersize=8, markerfacecolor=SURFACE,
                    markeredgewidth=1.5, label=f"{k // 1000}k-node games") for k in (10_000, 1_000)]
    fig.legend(handles=keys + size_handles, labels=[k.get_label() for k in keys] + [f"line: {l}" for l in size_labels],
               loc="upper left", bbox_to_anchor=(0.02, -0.02), ncol=6, fontsize=8.5)
    fig.text(0.02, -0.14, "Moving along a line = better data at the same size; moving between lines = more data. "
             "Labels = training rows. Mixes use the row-weighted average player. Hollow = from the rating fit.",
             fontsize=8.5, color=INK2)
    fig.suptitle("Data quality and data size together", x=0.02, ha="left", fontweight="bold")
    save(fig, "data_quality_vs_player")


def fig_predict_vs_rank(dataset="gen2-10k-aws"):
    """Predicting results (R², phase_analysis.py) vs ranking moves (move agreement,
    move_agreement.py) by game phase, for the same nets on the same unseen data."""
    p_path = os.path.join(RESULTS_DIR, f"phase_{dataset}.csv")
    a_path = os.path.join(RESULTS_DIR, f"move_agreement_{dataset}.csv")
    if not (os.path.exists(p_path) and os.path.exists(a_path)):
        return
    ph, ag = pd.read_csv(p_path), pd.read_csv(a_path)
    ag = ag[ag.phase != "all"]
    labels = {  # net -> (label, color, line style); the data's own player net is dashed (biased)
        "pair1024_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12": ("mix 40M", SERIES[0], "-"),
        "pair1024x32_hce-10k+gen1-10k+gen1-10k-aws+gen1-10k-aws2_e12": ("mix 40M + tail", SERIES[1], "-"),
        "pair1024_hce-10k_e9": ("release net", SERIES[2], "-"),
        "pair1024_gen1-10k+gen1-10k-aws_e6": ("gen1 5M", SERIES[3], "-"),
        "pair1024_gen1-1k_e5": ("gen1-1k 5M", SERIES[4], "-"),
        "pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9": ("gen1 20M (played this data)", INK2, (0, (4, 3))),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, df, col, title, ylabel in (
        (axes[0], ph, "r2", "Predicting the result", "R² vs the phase's average result"),
        (axes[1], ag, "agree", "Ranking moves", "top pick = the move played"),
    ):
        order = list(dict.fromkeys(df.phase))
        for net, (lab, color, style) in labels.items():
            d = df[df.net == net].set_index("phase").reindex(order)
            if d[col].isna().all():
                continue
            ax.plot(range(len(order)), d[col], color=color, linestyle=style, marker="o", markersize=5, label=lab)
        if col == "agree":
            base = df[df.net == df.net.iloc[0]].set_index("phase").reindex(order)
            ax.plot(range(len(order)), base["random"], color=INK2, linestyle=(0, (1, 2)), linewidth=1.2,
                    label="random legal move")
        ax.set_xticks(range(len(order)), order)
        ax.set_xlabel("move number")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    handles, labs = axes[1].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper left", bbox_to_anchor=(0.02, -0.02), ncol=4, fontsize=9)
    fig.text(0.02, -0.17, f"Both on {dataset} (unseen by every net except the dashed one, which played these games). "
             "~236 legal moves per position. The tail predicts results best but ranks moves worse than the "
             "same net without it.", fontsize=8.5, color=INK2)
    fig.suptitle("Predicting results vs ranking moves", x=0.02, ha="left", fontweight="bold")
    save(fig, "predict_vs_rank")


if __name__ == "__main__":
    m, runs, epochs, data = load()
    print(f"{len(m)} matches, {len(runs)} runs, {len(epochs)} epoch rows -> {FIG_DIR}")
    fig_score_vs_nodes(m)
    fig_scaling(m, runs, data)
    for n in (10_000, 50_000, 100_000):
        fig_strength(m, runs, data, n)
    fig_vs_release_by_budget(m, runs, data)
    fig_cost_frontier(m, runs, data)
    fig_score_vs_rows(m, runs, data)
    fig_hce_equivalent(m)
    fig_training_curves(epochs)
    fig_player_strength(m)
    fig_data_quality(m, runs, data)
    fig_predict_vs_rank()
