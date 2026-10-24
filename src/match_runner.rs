//! SPRT match orchestration and live summary display.
//!
//! Spawns a pool of workers that each own one engine pair and play pairs of
//! games (same random starting board, each engine moving first once). Completed
//! pairs are fed into an `SprtState` and a full summary box is printed after
//! every pair. Sets a shared atomic stop flag once SPRT reaches a terminal verdict.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crate::EngineConfig;
use crate::game::{gen_board_independent, play_match_game, print_board_from_starting_perspective, start_engine, Schedules};
use crate::sprt::{Decision, PairData, SprtConfig, SprtState};

#[derive(Debug)]
pub struct ThinkStats {
    pub move_count: usize,
    pub total_ms: u128,
    pub min_ms: u128,
    pub max_ms: u128,
    pub total_avg_ply: f64,
}

impl ThinkStats {
    fn new() -> Self {
        ThinkStats { move_count: 0, total_ms: 0, min_ms: u128::MAX, max_ms: 0, total_avg_ply: 0.0 }
    }

    fn record(&mut self, ms: u128, avg_ply: f64) {
        self.move_count += 1;
        self.total_ms += ms;
        if ms < self.min_ms { self.min_ms = ms; }
        if ms > self.max_ms { self.max_ms = ms; }
        self.total_avg_ply += avg_ply;
    }

    pub fn avg_ms(&self) -> f64 {
        if self.move_count == 0 { return 0.0; }
        self.total_ms as f64 / self.move_count as f64
    }

    pub fn avg_depth(&self) -> f64 {
        if self.move_count == 0 { return 0.0; }
        self.total_avg_ply / self.move_count as f64
    }
}

/// One game pair's worth of outcomes, sent from `pair_worker` back to the
/// main thread for aggregation in `run_sprt_match`.
#[derive(Debug)]
struct PairResult {
    e1_wins: usize,
    e2_wins: usize,
    draws: usize,
    e1_win_moves: Vec<usize>,
    e2_win_moves: Vec<usize>,
    draw_moves: Vec<usize>,
    /// (elapsed_ms, final_depth) per E1 move
    e1_thinks: Vec<(u128, f64)>,
    /// (elapsed_ms, final_depth) per E2 move
    e2_thinks: Vec<(u128, f64)>,
}

pub struct MatchSummary {
    pub sprt: SprtState,
    pub e1_name: String,
    pub e2_name: String,
    /// One-line description of the search limits, printed in the header.
    pub limits: String,
    pub e1_think: ThinkStats,
    pub e2_think: ThinkStats,
    pub e1_win_moves: Vec<usize>,
    pub e2_win_moves: Vec<usize>,
    pub draw_moves: Vec<usize>,
}

/// Runs an SPRT match between `e1_cfg` (the candidate) and `e2_cfg`, each
/// searching with its own schedules.
///
/// `thread_count` is the number of games played at once; each one runs two
/// engine processes. `watch` prints every move of worker 0's games.
pub fn run_sprt_match(
    e1_cfg: EngineConfig,
    e2_cfg: EngineConfig,
    sprt_cfg: SprtConfig,
    e1_schedules: Schedules,
    e2_schedules: Schedules,
    thread_count: usize,
    watch: bool,
) -> MatchSummary {
    let stop = Arc::new(AtomicBool::new(false));
    let next_pair = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<PairResult>();

    let mut handles = Vec::with_capacity(thread_count);
    for worker_id in 0..thread_count {
        let next_pair = next_pair.clone();
        let stop = stop.clone();
        let tx = tx.clone();
        let e1_cfg = e1_cfg.clone();
        let e2_cfg = e2_cfg.clone();
        let e1_schedules = e1_schedules.clone();
        let e2_schedules = e2_schedules.clone();
        let watch = watch && worker_id == 0;

        handles.push(thread::spawn(move || {
            pair_worker(worker_id, sprt_cfg.max_pairs, next_pair, tx, stop, e1_cfg, e2_cfg, e1_schedules, e2_schedules, watch);
        }));
    }
    drop(tx);

    let mut summary = MatchSummary {
        sprt: SprtState::new(sprt_cfg),
        e1_name: e1_cfg.name.clone(),
        e2_name: e2_cfg.name.clone(),
        limits: if e1_schedules.nodes == e2_schedules.nodes {
            format!("{}  |  {} games at once", describe(&e1_schedules), thread_count)
        } else {
            format!(
                "{}: {}
                 {}: {}
                 {} games at once",
                e1_cfg.name, describe(&e1_schedules), e2_cfg.name, describe(&e2_schedules), thread_count,
            )
        },
        e1_think: ThinkStats::new(),
        e2_think: ThinkStats::new(),
        e1_win_moves: vec![],
        e2_win_moves: vec![],
        draw_moves: vec![],
    };

    while let Ok(sr) = rx.recv() {
        let pair = PairData {
            e1_wins: sr.e1_wins,
            e2_wins: sr.e2_wins,
            draws: sr.draws,
        };
        summary.sprt.update(&pair);
        for (ms, d) in &sr.e1_thinks { summary.e1_think.record(*ms, *d); }
        for (ms, d) in &sr.e2_thinks { summary.e2_think.record(*ms, *d); }
        summary.e1_win_moves.extend(sr.e1_win_moves);
        summary.e2_win_moves.extend(sr.e2_win_moves);
        summary.draw_moves.extend(sr.draw_moves);

        print_summary_box(&summary);

        if sprt_cfg.auto_stop_on_decision && summary.sprt.decision().is_terminal() {
            stop.store(true, Ordering::SeqCst);
            break;
        }
    }

    if sprt_cfg.auto_stop_on_decision {
        println!();
        println!("  Decision reached. Waiting for in-flight pairs to finish...");
        println!();
    }
    drop(rx);
    for h in handles { let _ = h.join(); }

    println!();
    println!("  ── FINAL ─────────────────────────────────────────────────────────");
    print_summary_box(&summary);

    summary
}

/// Pulls pair indices off the shared counter and plays each pair: two games on
/// the same random starting board, each engine moving first once. Engines that
/// crash or get killed for blowing the time limit are restarted between games.
fn pair_worker(
    worker_id: usize,
    pair_count: usize,
    next_pair: Arc<AtomicUsize>,
    tx: mpsc::Sender<PairResult>,
    stop: Arc<AtomicBool>,
    e1_cfg: EngineConfig,
    e2_cfg: EngineConfig,
    e1_schedules: Schedules,
    e2_schedules: Schedules,
    watch: bool,
) {
    let mut engine_1 = start_engine(&e1_cfg.path, &e1_cfg.init_commands);
    let mut engine_2 = start_engine(&e2_cfg.path, &e2_cfg.init_commands);

    loop {
        if stop.load(Ordering::SeqCst) { break; }
        let pair_num = next_pair.fetch_add(1, Ordering::SeqCst);
        if pair_num >= pair_count { break; }

        let board = gen_board_independent();

        let mut pr = PairResult {
            e1_wins: 0, e2_wins: 0, draws: 0,
            e1_win_moves: vec![], e2_win_moves: vec![], draw_moves: vec![],
            e1_thinks: vec![], e2_thinks: vec![],
        };

        for e1_goes_first in [true, false] {
            if watch {
                let starting_engine = if e1_goes_first { &e1_cfg.name } else { &e2_cfg.name };
                println!();
                println!("--------------------------------");
                println!("  [w{worker_id}] PAIR {}  —  {} goes first", pair_num + 1, starting_engine);
                println!("--------------------------------");
                println!("Starting board:");
                print_board_from_starting_perspective(&board, !e1_goes_first);
            }

            let game = play_match_game(
                &mut engine_1, &mut engine_2, &e1_cfg.name, &e2_cfg.name,
                e1_goes_first, board, &e1_schedules, &e2_schedules, watch, false,
            );

            if !engine_1.is_alive() { engine_1 = start_engine(&e1_cfg.path, &e1_cfg.init_commands); }
            if !engine_2.is_alive() { engine_2 = start_engine(&e2_cfg.path, &e2_cfg.init_commands); }

            pr.e1_thinks.extend(game.e1_thinks);
            pr.e2_thinks.extend(game.e2_thinks);
            if game.outcome == 1.0 {
                pr.e1_wins += 1;
                pr.e1_win_moves.push(game.moves);
            } else if game.outcome == -1.0 {
                pr.e2_wins += 1;
                pr.e2_win_moves.push(game.moves);
            } else {
                pr.draws += 1;
                pr.draw_moves.push(game.moves);
            }
        }

        if tx.send(pr).is_err() {
            break; // main thread dropped the receiver
        }
    }

    engine_1.send("quit");
    engine_2.send("quit");
}

fn describe(s: &Schedules) -> String {
    format!("nodes {:?}  time {:?}s  ply {:?}  randomize {:?}", s.nodes, s.time, s.depth, s.randomize)
}

fn print_summary_box(s: &MatchSummary) {
    let width = 82;
    let line: String = "═".repeat(width);

    let cfg = &s.sprt.cfg;
    let n_pairs = s.sprt.n_pairs();
    let n_games = s.sprt.n_games();
    let mean = s.sprt.mean();
    let se = s.sprt.stderr();
    let (lo, hi) = s.sprt.score_ci_95();
    let llr = s.sprt.llr();
    let lb = s.sprt.lower_bound();
    let ub = s.sprt.upper_bound();
    let los = s.sprt.los();
    let dec = s.sprt.decision();

    let elapsed = s.sprt.elapsed();
    let elapsed_str = format_duration(elapsed);

    println!();
    println!("{line}");
    println!("  SPRT MATCH  —  {}   vs   {}", s.e1_name, s.e2_name);
    println!("{line}");
    println!(
        "  Hypothesis:    H0: score ≤ {:.3}     H1: score ≥ {:.3}     α = {:.2}  β = {:.2}",
        cfg.h0_score, cfg.h1_score, cfg.alpha, cfg.beta,
    );
    println!("  Limits:        {}", s.limits);
    println!(
        "  Pairs played:  {} / {} max        Games:  {}          Wall time:  {}",
        n_pairs, cfg.max_pairs, n_games, elapsed_str,
    );

    println!();
    println!("  ──── DECISION {}", "─".repeat(width.saturating_sub(17)));
    if se.is_finite() {
        println!("    LLR:        {:+.3}    bounds: [{:+.3}, {:+.3}]", llr, lb, ub);
    } else {
        println!("    LLR:        n/a (need ≥ 2 pairs)    bounds: [{:+.3}, {:+.3}]", lb, ub);
    }
    print_verdict(dec, cfg, &s.e1_name, &s.e2_name);

    println!();
    println!("  ──── SCORE {}", "─".repeat(width.saturating_sub(14)));
    if se.is_finite() {
        println!(
            "    Pair score:    {:.4}    stderr: {:.4}    95% CI: [{:.4}, {:.4}]",
            mean, se, lo, hi,
        );
        println!(
            "    LOS:           {:.2}%    (probability {} is genuinely stronger)",
            los * 100.0, s.e1_name,
        );
    } else {
        println!("    Pair score:    {:.4}    (need ≥ 2 pairs for stderr / CI)", mean);
    }
    println!();
    println!("  ──── PENTANOMIAL  ({} pairs) {}", n_pairs, "─".repeat(width.saturating_sub(28 + count_digits(n_pairs))));
    print_penta_row("LL  lose both        ", s.sprt.penta.ll, n_pairs);
    print_penta_row("LD  lose + draw      ", s.sprt.penta.ld, n_pairs);
    print_penta_row("DD  split / both draw", s.sprt.penta.dd, n_pairs);
    print_penta_row("WD  win + draw       ", s.sprt.penta.wd, n_pairs);
    print_penta_row("WW  win both         ", s.sprt.penta.ww, n_pairs);

    println!();
    println!("  ──── GAME RESULTS  ({} games) {}", n_games, "─".repeat(width.saturating_sub(26 + count_digits(n_games))));
    let ng = n_games.max(1);
    let name_w = s.e1_name.len().max(s.e2_name.len()).max(5);
    println!(
        "    {:<w$}  |  {:>4} wins   ({:>4.1}%)",
        s.e1_name, s.sprt.e1_total_wins, s.sprt.e1_total_wins as f64 / ng as f64 * 100.0, w = name_w,
    );
    println!(
        "    {:<w$}  |  {:>4} wins   ({:>4.1}%)",
        s.e2_name, s.sprt.e2_total_wins, s.sprt.e2_total_wins as f64 / ng as f64 * 100.0, w = name_w,
    );
    println!(
        "    {:<w$}  |  {:>4} draws  ({:>4.1}%)",
        "Draws", s.sprt.draws_total, s.sprt.draws_total as f64 / ng as f64 * 100.0, w = name_w,
    );

    println!();
    println!("  ──── THINK TIMES {}", "─".repeat(width.saturating_sub(20)));
    print_think_row(&s.e1_name, &s.e1_think, name_w);
    print_think_row(&s.e2_name, &s.e2_think, name_w);

    println!();
    println!("  ──── GAME LENGTH (moves) {}", "─".repeat(width.saturating_sub(28)));
    println!(
        "    {:<w$}  |  wins avg {}   losses avg {}",
        s.e1_name, fmt_avg(&s.e1_win_moves), fmt_avg(&s.e2_win_moves), w = name_w,
    );
    println!(
        "    {:<w$}  |  wins avg {}   losses avg {}",
        s.e2_name, fmt_avg(&s.e2_win_moves), fmt_avg(&s.e1_win_moves), w = name_w,
    );
    println!(
        "    {:<w$}  |  avg {}",
        "Draws", fmt_avg(&s.draw_moves), w = name_w,
    );
    println!("{line}");
}

fn print_verdict(dec: Decision, cfg: &SprtConfig, e1: &str, e2: &str) {
    match dec {
        Decision::WarmingUp => {
            println!(
                "    Verdict:    warming up (need ≥ {} pairs before a verdict is reliable)",
                cfg.min_pairs,
            );
        }
        Decision::Continue => {
            println!("    Verdict:    in progress  (LLR inside corridor; need more data)");
        }
        Decision::AcceptH1 => {
            println!(
                "    Verdict:    H1 ACCEPTED ✓     (95% confident: {} pair-score ≥ {:.3})",
                e1, cfg.h1_score,
            );
        }
        Decision::AcceptH0 => {
            println!(
                "    Verdict:    H0 ACCEPTED       (no improvement at the {:.3} bar — {} is NOT meaningfully ≥ {:.3})",
                cfg.h1_score, e1, cfg.h1_score,
            );
            println!(
                "                                  (does not mean {} is worse — check LOS for that)",
                e1,
            );
            let _ = e2;
        }
        Decision::Inconclusive => {
            println!(
                "    Verdict:    INCONCLUSIVE     (hit max_pairs cap = {}; difference too small to resolve at {:.3} bar)",
                cfg.max_pairs, cfg.h1_score,
            );
        }
    }
}

fn print_penta_row(label: &str, count: usize, total: usize) {
    let pct = if total == 0 { 0.0 } else { count as f64 / total as f64 * 100.0 };
    let bar_chars = if pct > 0.0 { ((pct / 100.0) * 50.0).round().max(1.0) as usize } else { 0 };
    let bar: String = "▎".repeat(bar_chars);
    println!("    {}  {:>4}  ({:>5.1}%)  {}", label, count, pct, bar);
}

fn print_think_row(name: &str, ts: &ThinkStats, w: usize) {
    let min = if ts.move_count == 0 { 0 } else { ts.min_ms };
    let max = ts.max_ms;
    println!(
        "    {:<w$}  |  avg {:>7.1}ms  min {:>5}ms  max {:>5}ms  avg depth {:>5.2}  ({} moves)",
        name, ts.avg_ms(), min, max, ts.avg_depth(), ts.move_count, w = w,
    );
}

fn fmt_avg(v: &[usize]) -> String {
    if v.is_empty() { return "  n/a".to_string(); }
    let avg = v.iter().sum::<usize>() as f64 / v.len() as f64;
    format!("{:>5.1}", avg)
}

fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 { format!("{}h {:02}m {:02}s", h, m, s) }
    else if m > 0 { format!("{}m {:02}s", m, s) }
    else { format!("{}s", s) }
}

fn count_digits(n: usize) -> usize {
    if n == 0 { 1 } else { (n as f64).log10().floor() as usize + 1 }
}
