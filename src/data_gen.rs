//! Self-play training-data generation. Each worker owns one engine and appends
//! every position of every decisive game to `training/data/<dataset>/w<NN>.csv`.
//!
//! CSV row: sq0..sq35, label (+1 mover won / -1 mover lost), move_num, game_id.
//! Drawn games and games where the engine forfeited are not written.
//!
//! `dataset` is the registry name from NOTES.md, e.g. `gen1-5k`. Restarting a
//! run with the same name appends to the same shards and carries on the game
//! numbering where each worker left off. Each finished game is flushed to disk
//! right away, so stopping the program (or a time budget running out) loses at
//! most the games in progress.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use crate::EngineConfig;
use crate::board::BoardState;
use crate::game::{gen_board_independent, play_data_game, start_engine, Schedules};

/// Game IDs are `(dataset_key * 1000 + worker_id) * GAMES_PER_WORKER_MAX + local_game_idx`,
/// where `dataset_key` is a hash of the dataset name: unique across workers and
/// (barring a 1-in-100k hash clash) across datasets, with no shared counter.
/// 10M per worker is well beyond any plausible run.
const GAMES_PER_WORKER_MAX: usize = 10_000_000;

/// FNV-1a hash of the dataset name, reduced to 5 digits.
fn dataset_key(name: &str) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h % 100_000) as usize
}

/// Spawns `thread_count` workers, each playing until it has played
/// `games_per_worker` games or `max_hours` have passed, whichever comes first.
pub fn generate_data_parallel(
    dataset: &str,
    engine: EngineConfig,
    schedules: Schedules,
    games_per_worker: usize,
    max_hours: Option<f64>,
    thread_count: usize,
    debug: bool,
) {
    let out_dir = Path::new("training").join("data").join(dataset);
    fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("cannot create {}: {}", out_dir.display(), e));
    let deadline = max_hours.map(|h| Instant::now() + Duration::from_secs_f64(h * 3600.0));
    println!(
        "[data gen] {} — {} workers x {} games{} → {}",
        dataset, thread_count, games_per_worker,
        max_hours.map_or(String::new(), |h| format!(" (or {} h)", h)),
        out_dir.display(),
    );

    let key = dataset_key(dataset);
    let handles: Vec<_> = (0..thread_count).map(|worker_id| {
        let engine = engine.clone();
        let schedules = schedules.clone();
        let path = out_dir.join(format!("w{:02}.csv", worker_id));
        let id_base = (key * 1000 + worker_id) * GAMES_PER_WORKER_MAX;
        thread::spawn(move || data_gen_worker(worker_id, &path, id_base, games_per_worker, deadline, engine, schedules, debug))
    }).collect();

    for (i, handle) in handles.into_iter().enumerate() {
        if handle.join().is_err() {
            eprintln!("[parallel] worker {} panicked", i);
        }
    }

    println!("[parallel] all {} workers done — shards in {}", thread_count, out_dir.display());
}

/// Cuts a half-written last row (from the program being killed mid-write) off
/// an existing shard, so the file always ends on a complete line.
fn trim_partial_line(path: &Path) {
    let Ok(mut file) = OpenOptions::new().read(true).write(true).open(path) else { return };
    let Ok(len) = file.metadata().map(|m| m.len()) else { return };
    if len == 0 { return; }
    let start = len.saturating_sub(512);
    let mut tail = Vec::new();
    if file.seek(SeekFrom::Start(start)).is_err() || file.read_to_end(&mut tail).is_err() { return; }
    if tail.last() == Some(&b'\n') { return; }
    let keep = tail.iter().rposition(|&b| b == b'\n').map_or(0, |i| start + i as u64 + 1);
    eprintln!("[warn] {} ended mid-row; trimmed {} bytes", path.display(), len - keep);
    let _ = file.set_len(keep);
}

/// Local index of the last game written to an existing shard (from the last
/// row's game_id), so a restarted worker continues instead of reusing ids.
fn last_local_game(path: &Path, id_base: usize) -> Option<usize> {
    let mut file = fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    file.seek(SeekFrom::Start(len.saturating_sub(512))).ok()?;
    let mut tail = String::new();
    file.read_to_string(&mut tail).ok()?;
    let game_id: usize = tail.lines().last()?.rsplit(',').next()?.trim().parse().ok()?;
    game_id.checked_sub(id_base)
}

fn data_gen_worker(
    worker_id: usize,
    path: &Path,
    id_base: usize,
    game_count: usize,
    deadline: Option<Instant>,
    engine_cfg: EngineConfig,
    schedules: Schedules,
    debug: bool,
) {
    let mut engine = start_engine(&engine_cfg.path, &engine_cfg.init_commands);

    trim_partial_line(path);
    let first_game = last_local_game(path, id_base).map_or(0, |g| g + 1);
    if first_game > 0 {
        println!("[w{}] resuming {} at game {}", worker_id, path.display(), first_game);
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("[w{}] cannot open {}: {}", worker_id, path.display(), e));

    let started = Instant::now();
    let mut games_played = 0usize;
    let mut games_written = 0usize;
    let mut draws = 0usize;
    let mut forfeits = 0usize;
    let mut p1_wins = 0usize;
    let mut p2_wins = 0usize;
    let mut total_moves = 0usize;
    let mut total_positions = 0usize;
    let mut total_time_caps = 0usize;
    // Depth buckets: moves 1-5, 6-10, 11+
    let mut depth_buckets: [(f64, usize); 3] = [(0.0, 0); 3];  // (sum, count)

    for game in first_game..game_count {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            println!("[w{}] time budget reached", worker_id);
            break;
        }
        let p1_goes_first = game % 2 == 0;

        let board = gen_board_independent();
        let result = play_data_game(&mut engine, board, p1_goes_first, &schedules, debug);
        games_played += 1;

        total_time_caps += result.time_cap_hits;
        if result.time_cap_hits > 0 {
            println!("[w{}] game {} — hit time cap {} time(s)", worker_id, game + 1, result.time_cap_hits);
        }

        if result.forfeit {
            forfeits += 1;
            if !engine.is_alive() {
                engine = start_engine(&engine_cfg.path, &engine_cfg.init_commands);
            }
        } else {
            let first_score = result.positions.first().map(|(_, s)| *s).unwrap_or(0.0);
            let winner = if p1_goes_first { first_score } else { -first_score };
            if winner == 0.0 {
                draws += 1;
            } else {
                if winner > 0.0 { p1_wins += 1; } else { p2_wins += 1; }
                total_moves += result.positions.len();
                // Write the whole game in one go and flush it, so a stopped run
                // only ever loses the game in progress.
                let game_id = id_base + game;
                let mut rows = String::with_capacity(110 * result.positions.len());
                for (move_num, (board_state, score)) in result.positions.iter().enumerate() {
                    rows.push_str(&encode_csv_row(board_state, *score, move_num, game_id));
                }
                file.write_all(rows.as_bytes()).unwrap();
                file.flush().unwrap();
                total_positions += result.positions.len();
                games_written += 1;
            }
        }

        for (i, d) in result.depths.iter().enumerate() {
            let bucket = if i < 5 { 0 } else if i < 10 { 1 } else { 2 };
            depth_buckets[bucket].0 += d;
            depth_buckets[bucket].1 += 1;
        }

        if (game + 1) % 50 == 0 {
            let avg_moves = if games_written > 0 { total_moves as f64 / games_written as f64 } else { 0.0 };
            let bucket_avg = |b: usize| -> f64 {
                if depth_buckets[b].1 > 0 { depth_buckets[b].0 / depth_buckets[b].1 as f64 } else { 0.0 }
            };
            let hours = started.elapsed().as_secs_f64() / 3600.0;
            println!("[w{}] --- {}/{} | written {} | P1 {} P2 {} | draws {} | forfeits {} | avg {:.1} moves | depth: mv1-5 {:.2}, mv6-10 {:.2}, mv11+ {:.2} | positions: {} ({:.0}/h) | time caps: {} ---",
                worker_id, game + 1, game_count, games_written, p1_wins, p2_wins, draws, forfeits, avg_moves,
                bucket_avg(0), bucket_avg(1), bucket_avg(2),
                total_positions, total_positions as f64 / hours.max(1e-9), total_time_caps);
        }
    }

    engine.send("quit");

    println!(
        "[w{}] done — {} games played, {} written ({} positions) → {}",
        worker_id, games_played, games_written, total_positions, path.display()
    );
}

/// Encodes one board position as a CSV row.
/// Format: sq0,sq1,...,sq35,score,move_num,game_id\n
///
/// `move_num` is the 0-indexed position within the game (0 = initial board,
/// 1 = after the first move, ...). `game_id` is globally unique across workers
/// and datasets — see GAMES_PER_WORKER_MAX for the encoding scheme.
fn encode_csv_row(board: &BoardState, score: f64, move_num: usize, game_id: usize) -> String {
    let mut s = String::with_capacity(110);
    for i in 0..36 {
        s.push_str(&board.data[i].to_string());
        s.push(',');
    }
    // Score as integer: 1, -1, or 0
    let score_int = if score > 0.0 { 1i32 } else if score < 0.0 { -1i32 } else { 0i32 };
    s.push_str(&score_int.to_string());
    s.push(',');
    s.push_str(&move_num.to_string());
    s.push(',');
    s.push_str(&game_id.to_string());
    s.push('\n');
    s
}
