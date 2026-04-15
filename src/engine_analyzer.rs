use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::thread;
use std::time::Instant;

use rand::*;

use crate::EngineConfig;
use crate::ugi_engine::UgiEngine;
use crate::board::BoardState;

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

#[derive(Debug)]
pub struct AnalysisResults {
    pub engine_1_name: String,
    pub engine_2_name: String,
    pub engine_1_wins: usize,
    pub engine_2_wins: usize,
    pub draws: usize,
    pub engine_1_think: ThinkStats,
    pub engine_2_think: ThinkStats,
    /// Move counts for games won by engine 1
    pub e1_win_moves: Vec<usize>,
    /// Move counts for games won by engine 2
    pub e2_win_moves: Vec<usize>,
    /// Move counts for drawn games
    pub draw_moves: Vec<usize>,
    // Set-level outcomes
    pub e1_set_full_wins: usize,
    pub e2_set_full_wins: usize,
    pub e1_set_partial_wins: usize,
    pub e2_set_partial_wins: usize,
    pub tied_sets: usize,
}

impl AnalysisResults {
    pub fn print_summary(&self) {
        let total = self.engine_1_wins + self.engine_2_wins + self.draws;
        let pct = |n: usize| if total == 0 { 0.0 } else { n as f64 / total as f64 * 100.0 };
        let total_sets = self.e1_set_full_wins + self.e2_set_full_wins + self.e1_set_partial_wins + self.e2_set_partial_wins + self.tied_sets;
        let spct = |n: usize| if total_sets == 0 { 0.0 } else { n as f64 / total_sets as f64 * 100.0 };
        let w = self.engine_1_name.len().max(self.engine_2_name.len());

        let sep = "═".repeat(w + 54);
        let avg_moves = |v: &Vec<usize>| -> String {
            if v.is_empty() { return "  n/a".to_string(); }
            let avg = v.iter().sum::<usize>() as f64 / v.len() as f64;
            format!("{:>5.1}", avg)
        };

        println!();
        println!("{sep}");
        println!("  ANALYSIS COMPLETE  —  {}  vs  {}  ({} games)", self.engine_1_name, self.engine_2_name, total);
        println!("{sep}");

        println!("  Results:");
        println!("    {:<w$}  |  {:>4} wins   ({:.1}%)", self.engine_1_name, self.engine_1_wins, pct(self.engine_1_wins), w = w);
        println!("    {:<w$}  |  {:>4} wins   ({:.1}%)", self.engine_2_name, self.engine_2_wins, pct(self.engine_2_wins), w = w);
        println!("    {:<w$}  |  {:>4} draws  ({:.1}%)", "Draws", self.draws, pct(self.draws), w = w);

        println!();
        println!("  Set results  ({} sets):", total_sets);
        println!("    {:<w$}  |  {:>4} full wins ({:.1}%)   {:>4} partial wins ({:.1}%)", self.engine_1_name, self.e1_set_full_wins, spct(self.e1_set_full_wins), self.e1_set_partial_wins, spct(self.e1_set_partial_wins), w = w);
        println!("    {:<w$}  |  {:>4} full wins ({:.1}%)   {:>4} partial wins ({:.1}%)", self.engine_2_name, self.e2_set_full_wins, spct(self.e2_set_full_wins), self.e2_set_partial_wins, spct(self.e2_set_partial_wins), w = w);
        println!("    {:<w$}  |  {:>4} tied sets ({:.1}%)", "Tied", self.tied_sets, spct(self.tied_sets), w = w);

        println!();
        println!("  Think times:");
        println!("    {:<w$}  |  avg {:>7.1}ms  min {:>5}ms  max {:>5}ms  avg depth {:>5.2}  ({} moves)",
            self.engine_1_name, self.engine_1_think.avg_ms(),
            self.engine_1_think.min_ms, self.engine_1_think.max_ms,
            self.engine_1_think.avg_depth(),
            self.engine_1_think.move_count, w = w,
        );
        println!("    {:<w$}  |  avg {:>7.1}ms  min {:>5}ms  max {:>5}ms  avg depth {:>5.2}  ({} moves)",
            self.engine_2_name, self.engine_2_think.avg_ms(),
            self.engine_2_think.min_ms, self.engine_2_think.max_ms,
            self.engine_2_think.avg_depth(),
            self.engine_2_think.move_count, w = w,
        );

        println!();
        println!("  Game length (moves):");
        println!("    {:<w$}  |  wins avg {}  losses avg {}", self.engine_1_name, avg_moves(&self.e1_win_moves), avg_moves(&self.e2_win_moves), w = w);
        println!("    {:<w$}  |  wins avg {}  losses avg {}", self.engine_2_name, avg_moves(&self.e2_win_moves), avg_moves(&self.e1_win_moves), w = w);
        println!("    {:<w$}  |  avg {}", "Draws", avg_moves(&self.draw_moves), w = w);

        println!("{sep}");
    }
}

pub struct EngineAnalyzer {
    engine_1: UgiEngine,
    engine_2: UgiEngine,
    engine_1_name: String,
    engine_2_name: String,
    engine_1_path: String,
    engine_2_path: String,
    /// Ply depth per move number (0-indexed). Last entry applies to all subsequent moves.
    /// e.g. vec![3, 3, 5] => moves 1 & 2 use ply 3, move 3+ uses ply 5.
    depth_schedule: Vec<u32>,
    /// Max think time (ms) per move number (0-indexed). Last entry applies to all subsequent moves.
    /// e.g. vec![20, 20, 100] => moves 1 & 2 use 20ms, move 3+ uses 100ms.
    time_schedule: Vec<u32>,
    /// Randomize value per move number (0-indexed). Last entry applies to all subsequent moves.
    randomize_schedule: Vec<u32>,
    /// Max node count per move number (0-indexed). Last entry applies to all subsequent moves.
    /// 0 means no node limit.
    node_schedule: Vec<u64>,
    /// When true, prints per-move board states and raw engine I/O.
    pub debug: bool,
}
impl EngineAnalyzer {
    pub fn new(engine_1_config: EngineConfig, engine_2_config: EngineConfig, depth_schedule: Vec<u32>, time_schedule: Vec<u32>, randomize_schedule: Vec<u32>, node_schedule: Vec<u64>, debug: bool) -> EngineAnalyzer {
        let mut engine_1 = UgiEngine::new(engine_1_config.path.as_str());
        let mut engine_2 = UgiEngine::new(engine_2_config.path.as_str());

        engine_1.send("ugi");
        engine_1.send("isready");

        engine_2.send("ugi");
        engine_2.send("isready");

        wait_for_readyok(&mut engine_1, 10_000);
        wait_for_readyok(&mut engine_2, 10_000);

        EngineAnalyzer {
            engine_1,
            engine_2,
            engine_1_name: engine_1_config.name,
            engine_2_name: engine_2_config.name,
            engine_1_path: engine_1_config.path,
            engine_2_path: engine_2_config.path,
            depth_schedule,
            time_schedule,
            randomize_schedule,
            node_schedule,
            debug,
        }

    }

    fn ply_for_move(&self, move_num: usize) -> u32 {
        let idx = move_num.min(self.depth_schedule.len() - 1);
        self.depth_schedule[idx]
    }

    fn time_for_move(&self, move_num: usize) -> u32 {
        let idx = move_num.min(self.time_schedule.len() - 1);
        self.time_schedule[idx]
    }

    fn randomize_for_move(&self, move_num: usize) -> u32 {
        let idx = move_num.min(self.randomize_schedule.len() - 1);
        self.randomize_schedule[idx]
    }

    fn nodes_for_move(&self, move_num: usize) -> u64 {
        let idx = move_num.min(self.node_schedule.len() - 1);
        self.node_schedule[idx]
    }

    /// Runs `set_count` sets of 4 games each (2 starting players x 2 board orientations).
    /// Prints an interim summary after every `interim_every` sets.
    pub fn analyze(&mut self, set_count: usize) -> AnalysisResults {
        let games_per_set = 2; // update if set_games changes
        let total_games = set_count * games_per_set;
        let mut results = AnalysisResults {
            engine_1_name: self.engine_1_name.clone(),
            engine_2_name: self.engine_2_name.clone(),
            engine_1_wins: 0,
            engine_2_wins: 0,
            draws: 0,
            engine_1_think: ThinkStats::new(),
            engine_2_think: ThinkStats::new(),
            e1_win_moves: vec![],
            e2_win_moves: vec![],
            draw_moves: vec![],
            e1_set_full_wins: 0,
            e2_set_full_wins: 0,
            e1_set_partial_wins: 0,
            e2_set_partial_wins: 0,
            tied_sets: 0,
        };

        for set in 0..set_count {
            println!();
            println!();
            println!("================================================================");
            println!("  SET {} / {}", set + 1, set_count);
            println!("================================================================");
            println!();
            let board = gen_board();
            let mut flipped = board.clone();
            flipped.flip();

            let set_games = [
                (1.0,  board.clone()),
                (-1.0, board.clone()),
                // (1.0,  flipped.clone()),
                // (-1.0, flipped.clone()),
            ];

            let mut set_e1_wins = 0usize;
            let mut set_e2_wins = 0usize;
            let mut set_draws = 0usize;

            for (game_in_set, (starting_player, starting_board)) in set_games.iter().enumerate() {
                let game_num = set * games_per_set + game_in_set + 1;
                let starting_engine = if *starting_player == 1.0 { &self.engine_1_name } else { &self.engine_2_name };
                println!();
                println!("--------------------------------");
                println!("  GAME {}  —  {} goes first", game_num, starting_engine);
                println!("--------------------------------");
                println!("Starting board:");
                print_board_from_starting_perspective(starting_board, *starting_player == -1.0);

                let (outcome, e1_times, e2_times, e1_depths, e2_depths) = self.sim_game(*starting_player, starting_board.clone());

                let move_count = e1_times.len() + e2_times.len();
                let e1_game_avg = avg_ms(&e1_times);
                let e2_game_avg = avg_ms(&e2_times);

                for (ms, depth) in e1_times.iter().zip(e1_depths.iter()) { results.engine_1_think.record(*ms, *depth); }
                for (ms, depth) in e2_times.iter().zip(e2_depths.iter()) { results.engine_2_think.record(*ms, *depth); }

                let winner = if outcome == 1.0 {
                    results.engine_1_wins += 1;
                    results.e1_win_moves.push(move_count);
                    set_e1_wins += 1;
                    format!("{} won", results.engine_1_name)
                } else if outcome == -1.0 {
                    results.engine_2_wins += 1;
                    results.e2_win_moves.push(move_count);
                    set_e2_wins += 1;
                    format!("{} won", results.engine_2_name)
                } else {
                    results.draws += 1;
                    results.draw_moves.push(move_count);
                    set_draws += 1;
                    "Draw".to_string()
                };

                println!(
                    "Set {:>3}/{} Game {}: {}  ({} moves)  |  {} avg {:.0}ms  {} avg {:.0}ms",
                    set + 1, set_count, game_in_set + 1, winner, move_count,
                    results.engine_1_name, e1_game_avg,
                    results.engine_2_name, e2_game_avg,
                );

            }

            // Classify set outcome
            match (set_e1_wins, set_e2_wins, set_draws) {
                (e1, 0, 0) if e1 > 0 => results.e1_set_full_wins += 1,
                (0, e2, 0) if e2 > 0 => results.e2_set_full_wins += 1,
                (1, 0, _) => results.e1_set_partial_wins += 1,
                (0, 1, _) => results.e2_set_partial_wins += 1,
                _ => results.tied_sets += 1,
            }

            results.print_summary();

        }

        return results;

    }

    pub fn sim_game(&mut self, starting_player: f64, starting_board: BoardState) -> (f64, Vec<u128>, Vec<u128>, Vec<f64>, Vec<f64>) {
        let mut board = starting_board.clone();
        let e1_name = self.engine_1_name.clone();
        let e2_name = self.engine_2_name.clone();

        // Orient the board so the first mover always sees it from their perspective.
        let e1_goes_first = starting_player == 1.0;
        if !e1_goes_first {
            board.flip();
        }

        let mut mv_history: Vec<Vec<usize>> = vec![];
        let mut p1_moves = 0usize;
        let mut p2_moves = 0usize;
        let mut e1_times: Vec<u128> = vec![];
        let mut e2_times: Vec<u128> = vec![];
        let mut e1_depths: Vec<f64> = vec![];
        let mut e2_depths: Vec<f64> = vec![];
        // First mover's turn always shows the board without a flip; second mover's with a flip.
        let mut flip_for_print = false;

        loop {
            for is_first_mover_turn in [true, false] {
                let is_e1 = is_first_mover_turn == e1_goes_first;
                let move_num = if is_e1 { p1_moves } else { p2_moves };
                let ply = self.ply_for_move(move_num);
                let time_secs = self.time_for_move(move_num);
                let randomize = self.randomize_for_move(move_num);
                let nodes = self.nodes_for_move(move_num);
                let player_sign = if is_e1 { 1.0 } else { -1.0 };
                let name = if is_e1 { &e1_name } else { &e2_name };

                let t = Instant::now();
                let mv_data = if is_e1 {
                    get_move(&mut self.engine_1, board.clone(), player_sign, ply, time_secs, randomize, nodes, self.debug)
                } else {
                    get_move(&mut self.engine_2, board.clone(), player_sign, ply, time_secs, randomize, nodes, self.debug)
                };
                let elapsed = t.elapsed().as_millis();

                if is_e1 { e1_times.push(elapsed); e1_depths.push(mv_data.3); p1_moves += 1; }
                else      { e2_times.push(elapsed); e2_depths.push(mv_data.3); p2_moves += 1; }

                // Null move from engine = draw
                let mv = match mv_data.0 {
                    Some(m) => m,
                    None => {
                        println!("  [DRAW] {} returned null move (engine-detected draw)", name);
                        return (0.0, e1_times, e2_times, e1_depths, e2_depths);
                    }
                };

                if mv_history.iter().filter(|m| **m == mv).count() > 1 {
                    println!("  [DRAW] Move repetition detected");
                    return (0.0, e1_times, e2_times, e1_depths, e2_depths);
                }
                board.make_move(mv.clone());
                mv_history.push(mv.clone());

                let display_move_num = if is_e1 { p1_moves } else { p2_moves };
                println!("After {}'s move {}:", name, display_move_num);
                print_board_from_starting_perspective(&board, flip_for_print);
                flip_for_print = !flip_for_print;

                if *mv.last().unwrap() == 37 {
                    return (if is_e1 { 1.0 } else { -1.0 }, e1_times, e2_times, e1_depths, e2_depths);
                }
                board.flip();
            }
        }
    }

    /// Runs `game_count` games between engine_1 (P1) and engine_2 (P2) and writes
    /// every position from every game to `training_data.csv`.
    ///
    /// The file is opened in **append mode**: restarting the program after a crash
    /// or stop continues the file rather than overwriting it. A header row is written
    /// only when the file does not yet exist.
    ///
    /// Positions for a single game are buffered in memory until the game ends (so we
    /// know the outcome), then the whole batch is written and flushed immediately.
    /// A crash can only lose the positions from the single in-progress game.
    ///
    /// CSV columns: sq0..sq35, score (1=win -1=loss 0=draw, from the perspective of the player to move)
    pub fn generate_data(&mut self, game_count: usize) {
        let path = "training_data.csv";

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("cannot open training_data.csv");

        // 1 MB write buffer — reduces syscall overhead during long runs.
        let mut writer = BufWriter::with_capacity(1 << 20, file);

        let mut games_written = 0usize;
        let mut draws = 0usize;
        let mut p1_wins = 0usize;
        let mut p2_wins = 0usize;
        let mut total_moves = 0usize;

        for game in 0..game_count {
            let p1_goes_first = game % 2 == 0;
            println!("[data gen] game {}/{} ({})", game + 1, game_count, if p1_goes_first { "P1 first" } else { "P2 first" });
            let board = gen_board();

            let (positions, _depths) = self.data_sim_game(board, p1_goes_first);

            // First position's score is from the first mover's perspective;
            // convert back to absolute (P1 positive) for stats.
            let first_score = positions.first().map(|(_, s)| *s).unwrap_or(0.0);
            let winner = if p1_goes_first { first_score } else { -first_score };
            if winner == 0.0 {
                draws += 1;
                continue;
            }
            total_moves += positions.len();
            if winner > 0.0 { p1_wins += 1; } else { p2_wins += 1; }
            for (board_state, score) in &positions {
                let row = encode_csv_row(board_state, *score);
                writer.write_all(row.as_bytes()).unwrap();
            }
            writer.flush().unwrap();
            games_written += 1;

            if (game + 1) % 10 == 0 {
                let avg_moves = if games_written > 0 { total_moves / games_written } else { 0 };
                println!(
                    "[{:>5}/{}] written {:>5} | P1 {:>3} ({:.0}%)  P2 {:>3} ({:.0}%)  draws {:>2} | avg {} moves",
                    game + 1, game_count, games_written,
                    p1_wins, if games_written > 0 { p1_wins as f64 / games_written as f64 * 100.0 } else { 0.0 },
                    p2_wins, if games_written > 0 { p2_wins as f64 / games_written as f64 * 100.0 } else { 0.0 },
                    draws, avg_moves,
                );
            }
        }

        println!(
            "[data gen] done — {} games written → {}",
            games_written, path
        );
    }

    /// Simulates one game with engine_1 as Player One and engine_2 as Player Two.
    /// Player One always goes first; the starting board is never flipped.
    ///
    /// Returns a Vec of `(board, score)` — one entry per half-move, plus depths.
    fn data_sim_game(&mut self, starting_board: BoardState, p1_goes_first: bool) -> (Vec<(BoardState, f64)>, Vec<f64>) {
        sim_game_for_data(
            &mut self.engine_1,
            starting_board, p1_goes_first,
            &self.depth_schedule, &self.time_schedule, &self.randomize_schedule, &self.node_schedule,
            self.debug,
        )
    }

    /// Spawns `thread_count` parallel workers, each playing `games_per_worker` games.
    /// Each worker spawns its own engine and writes to `training_data_{id}.csv`.
    pub fn generate_data_parallel(&self, games_per_worker: usize, thread_count: usize) {
        let handles: Vec<_> = (0..thread_count).map(|worker_id| {
            let worker_games = games_per_worker;
            let engine_path = self.engine_1_path.clone();
            let depth_sched = self.depth_schedule.clone();
            let time_sched = self.time_schedule.clone();
            let rand_sched = self.randomize_schedule.clone();
            let node_sched = self.node_schedule.clone();
            let debug = self.debug;

            thread::spawn(move || {
                data_gen_worker(worker_id, worker_games, engine_path, depth_sched, time_sched, rand_sched, node_sched, debug);
            })
        }).collect();

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(_) => {},
                Err(_) => eprintln!("[parallel] worker {} panicked", i),
            }
        }

        println!(
            "[parallel] all {} workers done — files: training_data_0.csv .. training_data_{}.csv",
            thread_count, thread_count - 1
        );
    }

    /// Kill the engines spawned by new() without full quit handshake.
    /// Use before generate_data_parallel which spawns its own engines.
    pub fn kill_engines(&mut self) {
        self.engine_1.kill();
        self.engine_2.kill();
    }

    pub fn quit(&mut self) {
        self.engine_1.quit();
        self.engine_2.quit();

    }

}

/// Standalone game simulation for training data. Used by both single-threaded
/// `generate_data` and parallel `data_gen_worker`.
///
/// Returns a Vec of `(board, score)` — one entry per half-move, plus a Vec of
/// depths reached per move. `board` is from the perspective of the player to move.
/// `score` is from that player's perspective: +1 if they won, -1 if they lost, 0 for a draw.
fn sim_game_for_data(
    engine: &mut UgiEngine,
    starting_board: BoardState,
    p1_goes_first: bool,
    depth_schedule: &[u32],
    time_schedule: &[u32],
    randomize_schedule: &[u32],
    node_schedule: &[u64],
    debug: bool,
) -> (Vec<(BoardState, f64)>, Vec<f64>) {
    let mut board = starting_board;
    if !p1_goes_first {
        board.flip();
    }

    let mut recorded: Vec<(BoardState, f64)> = vec![];
    let mut depths: Vec<f64> = vec![];
    let mut move_history: Vec<Vec<usize>> = vec![];
    let mut turn = 0usize;

    loop {
        let first_mover_turn = turn % 2 == 0;
        let current_player = if p1_goes_first == first_mover_turn { 0 } else { 1 };
        let player_sign = if current_player == 0 { 1.0 } else { -1.0 };

        // Record board from the mover's perspective (board is already oriented this way)
        recorded.push((board, player_sign));
        let move_num = turn / 2;
        let ply = depth_schedule[move_num.min(depth_schedule.len() - 1)];
        let time_secs = time_schedule[move_num.min(time_schedule.len() - 1)];
        let randomize = randomize_schedule[move_num.min(randomize_schedule.len() - 1)];
        let nodes = node_schedule[move_num.min(node_schedule.len() - 1)];

        let mv_data = get_move(engine, board, player_sign, ply, time_secs, randomize, nodes, debug);

        // Only record depth for non-trivial positions (no forced win/loss found)
        if mv_data.1.abs() < 100000.0 {
            depths.push(mv_data.3);
        }

        // Null move from engine = draw
        let mv = match mv_data.0 {
            Some(m) => m,
            None => return (recorded.into_iter().map(|(b, _)| (b, 0.0)).collect(), depths),
        };

        if move_history.iter().filter(|m| **m == mv).count() > 1 {
            return (recorded.into_iter().map(|(b, _)| (b, 0.0)).collect(), depths);
        }

        board.make_move(mv.clone());
        move_history.push(mv.clone());

        if debug {
            let player_label = if current_player == 0 { "P1" } else { "P2" };
            let print_board = if p1_goes_first == first_mover_turn {
                board
            } else {
                let mut b = board;
                b.flip();
                b
            };
            println!("After {}'s move {}:", player_label, turn / 2 + 1);
            println!("{}", print_board);
        }

        if *mv.last().unwrap() == 37 {
            let winner = if current_player == 0 { 1.0 } else { -1.0 };
            return (recorded.into_iter().map(|(b, ps)| (b, winner * ps)).collect(), depths);
        }

        board.flip();
        turn += 1;
    }
}

/// Worker function for parallel data generation. Each worker spawns its own
/// engine pair and writes to `training_data_{worker_id}.csv`.
fn data_gen_worker(
    worker_id: usize,
    game_count: usize,
    engine_path: String,
    depth_schedule: Vec<u32>,
    time_schedule: Vec<u32>,
    randomize_schedule: Vec<u32>,
    node_schedule: Vec<u64>,
    debug: bool,
) {
    let mut engine = UgiEngine::new(&engine_path);

    engine.send("ugi");
    engine.send("isready");
    wait_for_readyok(&mut engine, 10_000);

    let path = format!("training_data_{}.csv", worker_id);

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("[w{}] cannot open {}: {}", worker_id, path, e));

    let mut writer = BufWriter::with_capacity(1 << 20, file);

    let mut games_written = 0usize;
    let mut draws = 0usize;
    let mut p1_wins = 0usize;
    let mut p2_wins = 0usize;
    let mut total_moves = 0usize;
    let mut total_positions = 0usize;
    let mut unique_boards: HashSet<[usize; 38]> = HashSet::new();
    // Depth buckets: moves 1-5, 6-10, 11+
    let mut depth_buckets: [(f64, usize); 3] = [(0.0, 0); 3];  // (sum, count)

    for game in 0..game_count {
        let p1_goes_first = game % 2 == 0;

        let board = gen_board();

        let (positions, game_depths) = sim_game_for_data(
            &mut engine,
            board, p1_goes_first,
            &depth_schedule, &time_schedule, &randomize_schedule, &node_schedule,
            debug,
        );

        let first_score = positions.first().map(|(_, s)| *s).unwrap_or(0.0);
        let winner = if p1_goes_first { first_score } else { -first_score };
        if winner == 0.0 {
            draws += 1;
            println!("[w{}] game {} — draw ({} moves)", worker_id, game + 1, positions.len());
        } else {
            let outcome = if winner > 0.0 { p1_wins += 1; "P1 win" } else { p2_wins += 1; "P2 win" };
            println!("[w{}] game {} — {} ({} moves)", worker_id, game + 1, outcome, positions.len());
            total_moves += positions.len();
            for (board_state, score) in &positions {
                unique_boards.insert(board_state.data);
                total_positions += 1;
                let row = encode_csv_row(board_state, *score);
                writer.write_all(row.as_bytes()).unwrap();
            }
            writer.flush().unwrap();
            games_written += 1;
        }

        for (i, d) in game_depths.iter().enumerate() {
            let bucket = if i < 5 { 0 } else if i < 10 { 1 } else { 2 };
            depth_buckets[bucket].0 += d;
            depth_buckets[bucket].1 += 1;
        }

        if (game + 1) % 10 == 0 {
            let avg_moves = if games_written > 0 { total_moves as f64 / games_written as f64 } else { 0.0 };
            let bucket_avg = |b: usize| -> f64 {
                if depth_buckets[b].1 > 0 { depth_buckets[b].0 / depth_buckets[b].1 as f64 } else { 0.0 }
            };
            let unique_pct = if total_positions > 0 { unique_boards.len() as f64 / total_positions as f64 * 100.0 } else { 0.0 };
            println!("[w{}] --- {}/{} | written {} | P1 {} P2 {} | draws {} | avg {:.1} moves | depth: mv1-5 {:.2}, mv6-10 {:.2}, mv11+ {:.2} | unique: {}/{} ({:.1}%) ---",
                worker_id, game + 1, game_count, games_written, p1_wins, p2_wins, draws, avg_moves,
                bucket_avg(0), bucket_avg(1), bucket_avg(2),
                unique_boards.len(), total_positions, unique_pct);
        }
    }

    engine.send("quit");

    println!(
        "[w{}] done — {} games written → {}",
        worker_id, games_written, path
    );
}

/// Encodes one board position as a CSV row (no trailing newline is added by caller —
/// this function includes the `\n`).
/// Format: sq0,sq1,...,sq35,score\n
fn encode_csv_row(board: &BoardState, score: f64) -> String {
    let mut s = String::with_capacity(90);
    for i in 0..36 {
        s.push_str(&board.data[i].to_string());
        s.push(',');
    }
    // Write score as integer: 1, -1, or 0
    let score_int = if score > 0.0 { 1i32 } else if score < 0.0 { -1i32 } else { 0i32 };
    s.push_str(&score_int.to_string());
    s.push('\n');
    s
}

fn print_board_from_starting_perspective(board: &BoardState, flip: bool) {
    if flip {
        let mut b = *board;
        b.flip();
        println!("{}", b);
    } else {
        println!("{}", board);
    }
}

fn avg_ms(times: &[u128]) -> f64 {
    if times.is_empty() { return 0.0; }
    times.iter().sum::<u128>() as f64 / times.len() as f64
}

/// Returns `(move, score, time_secs, final_depth)` where `final_depth` is the
/// deepest ply completed before the engine returned `bestmove`.
/// A `None` move means the engine returned `bestmove null` (draw).
pub fn get_move(engine: &mut UgiEngine, board: BoardState, player: f64, ply: u32, time_secs: u32, randomize: u32, nodes: u64, debug: bool) -> (Option<Vec<usize>>, f64, f64, f64) {
    let mut s: String = String::new();
    for i in 0..38 {
        s.push_str(&board.data[i].to_string());
    }
    let cmd: String = format!("setpos data {}", s);
    if debug { println!("  [{}]", cmd); }

    engine.send(cmd.as_str());
    engine.send(format!("setoption maxPly {}", ply).as_str());
    engine.send(format!("setoption maxTime {}", time_secs).as_str());
    engine.send(format!("setoption randomize {}", randomize).as_str());
    engine.send(format!("setoption maxNodes {}", nodes).as_str());
    engine.send("go");

    let mut final_depth = 0u32;

    loop {
        if let Some(r) = engine.recive() {
            if debug { println!("engine {}: {}", player, r); }

            let raw_cmds: Vec<&str> = r.split_whitespace().collect();
            if raw_cmds.is_empty() { continue; }
            if raw_cmds[0] == "info" && raw_cmds.contains(&"bestmove") {
                if let Some(ply_pos) = raw_cmds.iter().position(|&t| t == "ply") {
                    if let Some(n) = raw_cmds.get(ply_pos + 1).and_then(|v| v.parse::<u32>().ok()) {
                        final_depth = n;
                    }
                }
            } else if raw_cmds[0] == "bestmove" {
                if raw_cmds[1] == "null" {
                    return (None, 0.0, 0.0, final_depth as f64);
                }
                return (
                    Some(parse_move(raw_cmds[1])),
                    raw_cmds[3].parse::<f64>().unwrap(),
                    raw_cmds[5].parse::<f64>().unwrap(),
                    final_depth as f64,
                );
            }
        }
    }
}

pub fn parse_move(raw_move: &str) -> Vec<usize> {
    let raw_mv_data: Vec<&str> = raw_move.split("|").collect();

    let mut mv = vec![];
    for i in 0..raw_mv_data.len() {
        mv.push(raw_mv_data[i].parse::<usize>().unwrap());

    }

    mv

}

/// Drains engine output until a `readyok` line is seen or `timeout_ms` elapses.
fn wait_for_readyok(engine: &mut UgiEngine, timeout_ms: u64) {
    let deadline = Instant::now();
    while deadline.elapsed().as_millis() < timeout_ms as u128 {
        if let Some(line) = engine.recive() {
            if line.trim() == "readyok" {
                return;
            }
        }
    }
    eprintln!("[data gen] warning: engine did not send readyok within {}ms", timeout_ms);
}

pub fn gen_board() -> BoardState {
    let mut p1_pieces: Vec<usize> = vec![3, 3, 2, 2, 1, 1];

    let mut board = BoardState::new();
    let mut rng = rand::thread_rng();

    for i in 0..6 {
        let p1_piece = p1_pieces.remove(rng.gen_range(0..p1_pieces.len()));
        board.data[i] = p1_piece;
    }

    // Mirror P1's row for P2 (reversed) so the board looks identical from both perspectives.
    // After a 180° flip, data[30+i] maps to column 5-i of the bottom row.
    for i in 0..6 {
        board.data[30 + i] = board.data[5 - i];
    }

    board

}
