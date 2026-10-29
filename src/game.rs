//! Engine I/O and single-game simulation, shared by the match runner and
//! training-data generation.
//!
//! The runner keeps its own board and never trusts an engine to follow the
//! rules: every move is checked against the `gyges` library's move generator
//! before it is applied (see `BoardState::is_legal`). An illegal move, a crash,
//! a blown time limit, or a 1.1.0-style resignation loses the game for the
//! engine responsible, and is logged to stderr as an `[EVENT]` line.

use std::time::{Duration, Instant};

use rand::seq::SliceRandom;
use rand::Rng;

use crate::board::BoardState;
use crate::ugi_engine::UgiEngine;

/// Per-move search limits. Each vector is indexed by the mover's own move
/// number (0-indexed); the last entry repeats for all later moves.
/// e.g. `nodes: vec![0, 0, 5_000]` => no node cap on moves 1-2, 5k from move 3 on.
#[derive(Clone, Debug)]
pub struct Schedules {
    /// Ply cap per move.
    pub depth: Vec<u32>,
    /// Seconds per move. The runner's hard limit is twice this plus 5 seconds.
    pub time: Vec<u32>,
    /// Engine-side randomization: 1 = play a random non-losing move, 0 = off.
    pub randomize: Vec<u32>,
    /// Node cap per move. 0 = no cap.
    pub nodes: Vec<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct MoveLimits {
    pub ply: u32,
    pub time_secs: u32,
    pub randomize: u32,
    pub nodes: u64,
}

impl Schedules {
    pub fn for_move(&self, move_num: usize) -> MoveLimits {
        fn pick<T: Copy>(v: &[T], i: usize) -> T { v[i.min(v.len() - 1)] }
        MoveLimits {
            ply: pick(&self.depth, move_num),
            time_secs: pick(&self.time, move_num),
            randomize: pick(&self.randomize, move_num),
            nodes: pick(&self.nodes, move_num),
        }
    }
}

/// What an engine replied to `go`.
pub enum Reply {
    /// `nodes` and `pv` come from the last completed iteration's `info` line;
    /// `depth_scores` holds (ply, score) after each completed iteration. 2.0.0 deepens
    /// two plies at a time, so these are plies 1, 3, 5, …
    Move { mv: Vec<usize>, score: f64, time_secs: f64, depth: f64, nodes: u64, pv: String, depth_scores: Vec<(u32, f64)> },
    /// `bestmove null`: the engine declares the position drawn.
    Draw { depth: f64 },
    /// No `bestmove` before the hard limit, or the engine process died.
    NoReply { depth: f64 },
}

impl Reply {
    fn depth(&self) -> f64 {
        match self {
            Reply::Move { depth, .. } | Reply::Draw { depth } | Reply::NoReply { depth } => *depth,
        }
    }
}

/// Spawns an engine and runs the `ugi` / init / `isready` handshake, then confirms
/// the engine evaluates with the net the init commands asked for (2.0.0 falls back to
/// its embedded net when a weights file fails to load, so that must stop the run).
pub fn start_engine(path: &str, init: &[String]) -> UgiEngine {
    let mut engine = UgiEngine::new(path);
    engine.send("ugi");
    for cmd in init { engine.send(cmd); }
    engine.send("isready");
    wait_for_readyok(&mut engine, 10_000);
    confirm_net(&mut engine, init);
    engine
}

/// Prints the confirmed net once per process (not once per engine).
static NET_CONFIRMED: std::sync::Once = std::sync::Once::new();

/// Runs a tiny search and checks the engine's `info string NN evaluation using <file>`
/// line against the requested weights file (or the embedded net). Panics on a mismatch.
fn confirm_net(engine: &mut UgiEngine, init: &[String]) {
    if !init.iter().any(|c| c.trim() == "setoption nn true") {
        return;
    }
    let expected = init.iter()
        .find_map(|c| c.strip_prefix("setoption weightsPath "))
        .map(|p| std::path::Path::new(p.trim()).file_name().unwrap().to_string_lossy().into_owned())
        .unwrap_or_else(|| String::from("default.bin (embedded)"));

    let layout: String = gen_board_independent().data.iter().map(|p| p.to_string()).collect();
    engine.send(format!("setpos data {}", layout).as_str());
    engine.send("setoption maxNodes 100");
    engine.send("go");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut reported = None;
    loop {
        let Some(line) = engine.recive_with_timeout(deadline.saturating_duration_since(Instant::now())) else {
            panic!("engine gave no reply to the net check (expected net {})", expected);
        };
        if let Some(rest) = line.trim().strip_prefix("info string NN evaluation using ") {
            reported = Some(rest.trim_end_matches(" enabled").to_string());
        }
        if line.starts_with("bestmove") {
            break;
        }
    }
    match reported {
        Some(name) if name == expected => {
            NET_CONFIRMED.call_once(|| println!("[engine] net confirmed: {}", name));
        }
        other => panic!("engine is not using the requested net: expected {}, engine reports {:?}", expected, other),
    }
}

/// Drains engine output until a `readyok` line is seen or `timeout_ms` elapses.
/// A failed weights load stops the run here instead of silently using another net.
fn wait_for_readyok(engine: &mut UgiEngine, timeout_ms: u64) {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            eprintln!("[warn] engine did not send readyok within {}ms", timeout_ms);
            return;
        }
        match engine.recive_with_timeout(remaining) {
            Some(line) if line.trim() == "readyok" => return,
            Some(line) if line.contains("weight load failed") => panic!("engine: {}", line.trim()),
            _ => {}
        }
    }
}

/// Sends the position and limits, then waits for `bestmove`.
/// An engine that blows the hard limit (2x move time + 5s) is killed.
pub fn get_move(engine: &mut UgiEngine, board: BoardState, limits: MoveLimits, debug: bool) -> Reply {
    let layout: String = board.data.iter().map(|p| p.to_string()).collect();
    let cmd = format!("setpos data {}", layout);
    if debug { println!("  [{}]", cmd); }

    // 2.0.0 rejects `maxNodes 0` and keeps its previous limit, so "no cap" has
    // to be sent as a huge cap instead.
    let nodes = if limits.nodes == 0 { u64::MAX } else { limits.nodes };

    engine.send(cmd.as_str());
    engine.send(format!("setoption maxPly {}", limits.ply).as_str());
    engine.send(format!("setoption maxTime {}", limits.time_secs).as_str());
    engine.send(format!("setoption randomize {}", limits.randomize).as_str());
    engine.send(format!("setoption maxNodes {}", nodes).as_str());
    engine.send("go");

    let mut final_depth = 0u32;
    let mut final_nodes = 0u64;
    let mut final_pv = String::new();
    let mut depth_scores: Vec<(u32, f64)> = vec![];
    let deadline = Instant::now() + Duration::from_secs(limits.time_secs as u64 * 2 + 5);

    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let Some(r) = engine.recive_with_timeout(left) else {
            if Instant::now() >= deadline {
                eprintln!("[EVENT] engine timed out (> {}s), killed", limits.time_secs * 2 + 5);
                engine.kill();
            }
            return Reply::NoReply { depth: final_depth as f64 };
        };
        if debug { println!("  engine: {}", r); }

        let raw_cmds: Vec<&str> = r.split_whitespace().collect();
        if raw_cmds.is_empty() { continue; }

        if raw_cmds[0] == "info" && raw_cmds.contains(&"bestmove") {
            if let Some(ply_pos) = raw_cmds.iter().position(|&t| t == "ply") {
                if let Some(n) = raw_cmds.get(ply_pos + 1).and_then(|v| v.parse::<u32>().ok()) {
                    final_depth = n;
                }
            }
            if let Some(p) = raw_cmds.iter().position(|&t| t == "nodes") {
                final_nodes = raw_cmds.get(p + 1).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            }
            if let Some(p) = raw_cmds.iter().position(|&t| t == "pv") {
                final_pv = raw_cmds[p + 1..].join(" ");
            }
            if let Some(p) = raw_cmds.iter().position(|&t| t == "score") {
                if let Some(s) = raw_cmds.get(p + 1).and_then(|v| v.parse::<f64>().ok()) {
                    depth_scores.push((final_depth, s));
                }
            }
        } else if raw_cmds[0] == "bestmove" {
            if raw_cmds.get(1) == Some(&"null") {
                return Reply::Draw { depth: final_depth as f64 };
            }
            let Some(mv) = raw_cmds.get(1).and_then(|m| parse_move(m)) else {
                eprintln!("[EVENT] unparseable bestmove line: '{}'", r);
                return Reply::NoReply { depth: final_depth as f64 };
            };
            // 1.1.0 prints only `bestmove <move>`, with no score or time.
            return Reply::Move {
                mv,
                score: raw_cmds.get(3).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0),
                time_secs: raw_cmds.get(5).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0),
                depth: final_depth as f64,
                nodes: final_nodes,
                pv: final_pv,
                depth_scores,
            };
        }
    }
}

pub fn parse_move(raw_move: &str) -> Option<Vec<usize>> {
    raw_move.split('|').map(|s| s.parse::<usize>().ok()).collect()
}

/// Why a game ended without a normal win/draw. The engine responsible loses.
fn forfeit_reason(reply: &Reply, board: &BoardState) -> Option<String> {
    match reply {
        Reply::NoReply { .. } => Some("crashed or timed out".to_string()),
        // 1.1.0 replies 100|100|100 when every move loses.
        Reply::Move { mv, .. } if mv.contains(&100) => Some("resigned".to_string()),
        Reply::Move { mv, .. } if !board.is_legal(mv) => Some(format!("played illegal move {:?}", mv)),
        _ => None,
    }
}

/// Result of one engine-vs-engine game, from engine 1's perspective.
pub struct GameRecord {
    /// +1 engine 1 won, -1 engine 2 won, 0 draw.
    pub outcome: f64,
    pub moves: usize,
    /// (elapsed_ms, final_depth) per engine-1 move.
    pub e1_thinks: Vec<(u128, f64)>,
    /// (elapsed_ms, final_depth) per engine-2 move.
    pub e2_thinks: Vec<(u128, f64)>,
}

/// Plays one game between two engines from `starting_board`. If engine 2
/// moves first, the board is flipped first so it starts from the other side.
/// Each engine searches with its own schedules (e.g. different node budgets).
///
/// `print_moves` prints the board after every move — for watching one game live.
pub fn play_match_game(
    engine_1: &mut UgiEngine,
    engine_2: &mut UgiEngine,
    e1_name: &str,
    e2_name: &str,
    e1_goes_first: bool,
    starting_board: BoardState,
    e1_schedules: &Schedules,
    e2_schedules: &Schedules,
    print_moves: bool,
    debug: bool,
) -> GameRecord {
    let mut board = starting_board;
    if !e1_goes_first {
        board.flip();
    }

    let mut rec = GameRecord { outcome: 0.0, moves: 0, e1_thinks: vec![], e2_thinks: vec![] };
    let mut mv_history: Vec<Vec<usize>> = vec![];
    let mut flip_for_print = false;

    loop {
        for is_first_mover_turn in [true, false] {
            let is_e1 = is_first_mover_turn == e1_goes_first;
            let move_num = if is_e1 { rec.e1_thinks.len() } else { rec.e2_thinks.len() };
            let limits = if is_e1 { e1_schedules } else { e2_schedules }.for_move(move_num);
            let name = if is_e1 { e1_name } else { e2_name };
            let mover_wins = if is_e1 { 1.0 } else { -1.0 };

            let t = Instant::now();
            let reply = if is_e1 {
                get_move(engine_1, board, limits, debug)
            } else {
                get_move(engine_2, board, limits, debug)
            };
            let elapsed = t.elapsed().as_millis();

            if is_e1 { rec.e1_thinks.push((elapsed, reply.depth())); }
            else     { rec.e2_thinks.push((elapsed, reply.depth())); }
            rec.moves += 1;

            if let Some(reason) = forfeit_reason(&reply, &board) {
                eprintln!("[EVENT] {} {} — forfeits", name, reason);
                rec.outcome = -mover_wins;
                return rec;
            }
            let mv = match reply {
                Reply::Move { mv, .. } => mv,
                _ => {
                    if print_moves { println!("  [DRAW] {} returned null move (engine-detected draw)", name); }
                    return rec;
                }
            };

            if mv_history.iter().filter(|m| **m == mv).count() > 1 {
                if print_moves { println!("  [DRAW] Move repetition detected"); }
                return rec;
            }
            board.make_move(&mv);
            mv_history.push(mv.clone());

            if print_moves {
                println!("After {}'s move {}:", name, move_num + 1);
                print_board_from_starting_perspective(&board, flip_for_print);
            }
            flip_for_print = !flip_for_print;

            if *mv.last().unwrap() == 37 {
                rec.outcome = mover_wins;
                return rec;
            }
            board.flip();
        }
    }
}

/// Result of one self-play game for training data.
pub struct DataGame {
    /// Every position, from the mover's perspective, labelled +1 if the mover
    /// went on to win, -1 if it lost, 0 for a draw.
    pub positions: Vec<(BoardState, f64)>,
    /// Final search depth per move, skipping moves where a forced win/loss was found.
    pub depths: Vec<f64>,
    /// Moves that used at least 95% of their time limit.
    pub time_cap_hits: usize,
    /// The engine forfeited (illegal move, crash, timeout). Don't train on it.
    pub forfeit: bool,
    /// The position after the random opening plies was already played: stopped there, not written.
    pub repeat: bool,
    /// Per position, the search that chose its move, as `search_columns` text.
    pub search: Vec<String>,
}

/// Scores at or beyond this size are proven wins/losses.
pub const FORCED_SCORE: f64 = 100000.0;

/// The search result columns appended to data rows:
/// `score,ply,time_s,forced,best,nodes,pv,depth_scores`.
/// `best` is the chosen move (`null` for a draw); `nodes` and `pv` (moves separated by
/// spaces) are from the last completed iteration; `depth_scores` is `ply:score` after each
/// completed iteration, separated by spaces (e.g. `1:-974.6 3:4559.6 5:99999997.0`).
/// A randomized move's score doesn't describe the move played, so it is written as `nan`.
pub fn search_columns(score: f64, depth: f64, time_secs: f64, mv: Option<&[usize]>, random: bool,
                      nodes: u64, pv: &str, depth_scores: &[(u32, f64)]) -> String {
    let best = mv.map_or(String::from("null"), |m| m.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("|"));
    let by_depth = depth_scores.iter().map(|(p, s)| format!("{}:{:.1}", p, s)).collect::<Vec<_>>().join(" ");
    if random {
        return format!("nan,{},{:.3},0,{},{},{},{}", depth, time_secs, best, nodes, pv, by_depth);
    }
    let forced = if score >= FORCED_SCORE { 1 } else if score <= -FORCED_SCORE { -1 } else { 0 };
    format!("{:.1},{},{:.3},{},{},{},{},{}", score, depth, time_secs, forced, best, nodes, pv, by_depth)
}

/// Plays one self-play game with a single engine on both sides. With `wild_random`, the
/// randomized moves are picked here (`random_move`) instead of by the engine after a search.
pub fn play_data_game(
    engine: &mut UgiEngine,
    starting_board: BoardState,
    p1_goes_first: bool,
    schedules: &Schedules,
    wild_random: bool,
    random_plies: usize,
    opening_seen: &mut dyn FnMut(&BoardState) -> bool,
    debug: bool,
) -> DataGame {
    let mut board = starting_board;
    if !p1_goes_first {
        board.flip();
    }

    let mut recorded: Vec<(BoardState, f64)> = vec![];
    let mut game = DataGame { positions: vec![], depths: vec![], time_cap_hits: 0, forfeit: false, repeat: false, search: vec![] };
    let mut move_history: Vec<Vec<usize>> = vec![];
    let mut turn = 0usize;

    loop {
        // The engine is deterministic after the random plies, so a game reaching an opening
        // position that was already played would be an exact copy: stop it before any search.
        if random_plies > 0 && turn == random_plies && opening_seen(&board) {
            game.repeat = true;
            return game;
        }
        let first_mover_turn = turn % 2 == 0;
        let current_player = if p1_goes_first == first_mover_turn { 0 } else { 1 };
        let player_sign = if current_player == 0 { 1.0 } else { -1.0 };

        // Record board from the mover's perspective (board is already oriented this way)
        recorded.push((board, player_sign));
        let limits = schedules.for_move(turn / 2);

        // A wild random move is picked here with no search, and recorded like a searched
        // random move (score nan) with no ply, time, nodes or pv.
        let wild = wild_random && limits.randomize != 0;
        let reply = if wild {
            random_move(&board).map_or(Reply::Draw { depth: 0.0 }, |mv| Reply::Move {
                mv, score: 0.0, time_secs: 0.0, depth: 0.0, nodes: 0, pv: String::new(), depth_scores: vec![],
            })
        } else {
            get_move(engine, board, limits, debug)
        };

        if let Some(reason) = forfeit_reason(&reply, &board) {
            eprintln!("[EVENT] data gen engine {} — game discarded", reason);
            game.forfeit = true;
            return game;
        }
        let (mv, score, time_secs, depth) = match reply {
            Reply::Move { mv, score, time_secs, depth, nodes, pv, depth_scores } => {
                game.search.push(search_columns(score, depth, time_secs, Some(&mv), limits.randomize != 0,
                                                nodes, &pv, &depth_scores));
                (mv, score, time_secs, depth)
            }
            // Null move from engine = draw
            Reply::Draw { depth } | Reply::NoReply { depth } => {
                game.search.push(search_columns(0.0, depth, 0.0, None, false, 0, "", &[]));
                game.positions = recorded.into_iter().map(|(b, _)| (b, 0.0)).collect();
                return game;
            }
        };

        if time_secs >= limits.time_secs as f64 * 0.95 {
            game.time_cap_hits += 1;
        }
        // Only record depth for searched, non-trivial positions (no forced win/loss found)
        if !wild && score.abs() < 100000.0 {
            game.depths.push(depth);
        }

        if move_history.iter().filter(|m| **m == mv).count() > 1 {
            game.positions = recorded.into_iter().map(|(b, _)| (b, 0.0)).collect();
            return game;
        }

        board.make_move(&mv);
        move_history.push(mv.clone());

        if debug {
            let player_label = if current_player == 0 { "P1" } else { "P2" };
            println!("After {}'s move {}:", player_label, turn / 2 + 1);
            print_board_from_starting_perspective(&board, p1_goes_first != first_mover_turn);
        }

        if *mv.last().unwrap() == 37 {
            let winner = if current_player == 0 { 1.0 } else { -1.0 };
            game.positions = recorded.into_iter().map(|(b, ps)| (b, winner * ps)).collect();
            return game;
        }

        board.flip();
        turn += 1;
    }
}

pub fn print_board_from_starting_perspective(board: &BoardState, flip: bool) {
    if flip {
        let mut b = *board;
        b.flip();
        println!("{}", b);
    } else {
        println!("{}", board);
    }
}

/// Random starting board: each home row is an independent shuffle of 3,3,2,2,1,1.
/// A uniformly random legal move for the side to move (on the bottom); a move that
/// reaches the goal is always taken, as the engine does. None when there is no move.
fn random_move(board: &BoardState) -> Option<Vec<usize>> {
    let mut gboard = gyges::BoardState::from(board.data);
    let mut move_gen = gyges::MoveGen::default();
    let moves = unsafe { move_gen.gen::<gyges::GenMoves, gyges::NoQuit>(&mut gboard, gyges::Player::One) }
        .move_list.moves(&gboard);
    let mv = moves.iter().find(|m| m.is_win()).or_else(|| moves.choose(&mut rand::thread_rng()))?;
    Some(mv.to_string().split('|').map(|s| s.parse().unwrap()).collect())
}

pub fn gen_board_independent() -> BoardState {
    let mut p1_pieces: Vec<usize> = vec![3, 3, 2, 2, 1, 1];
    let mut p2_pieces: Vec<usize> = vec![3, 3, 2, 2, 1, 1];

    let mut board = BoardState::new();
    let mut rng = rand::thread_rng();

    for i in 0..6 {
        let p1_piece = p1_pieces.remove(rng.gen_range(0..p1_pieces.len()));
        board.data[i] = p1_piece;
    }

    for i in 0..6 {
        let p2_piece = p2_pieces.remove(rng.gen_range(0..p2_pieces.len()));
        board.data[30 + i] = p2_piece;
    }

    board
}
