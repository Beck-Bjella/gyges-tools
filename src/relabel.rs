//! `gyges-tools relabel <dataset> <out_dataset> <workers> <nodes> [net]`
//!
//! Searches every position of an existing dataset and writes a copy with the search
//! result appended. The original dataset is only read. Output rows are the original
//! row (36 squares, final game result, move_num, game_id) plus:
//!
//!   score   engine score of the position for the side to move (tanh × 10000; a forced
//!           win/loss is ≥ 100000 in size)
//!   ply     deepest iteration the search completed
//!   time_s  search time
//!   forced  +1 / −1 when the search proved a win / loss, else 0
//!   best    the move the search chose (`from|to` or `from|via|to`), `null` for a draw
//!   nodes   nodes of the last completed iteration
//!   pv      the line the search expects (moves separated by spaces)
//!   depth_scores  `ply:score` after each completed iteration (2.0.0 deepens 2 plies at a
//!           time: 1, 3, 5, …), separated by spaces
//!
//! One engine per worker for the whole run; shards are taken from a shared queue, so
//! every core stays busy to the end. Boards seen before by a worker are not searched
//! again. Output is flushed regularly and a restart resumes each shard where it stopped.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::board::BoardState;
use crate::data_gen::trim_partial_line;
use crate::game::{get_move, search_columns, start_engine, MoveLimits, Reply};

pub fn run(args: &[String]) {
    let usage = "usage: gyges-tools relabel <dataset> <out_dataset> <workers> <nodes> [net]";
    let dataset = args.first().expect(usage);
    let out_name = args.get(1).expect(usage);
    let workers: usize = args.get(2).expect(usage).parse().expect("workers must be a number");
    let nodes: u64 = args.get(3).expect(usage).parse().expect("nodes must be a number");
    let net = args.get(4).map_or("release", String::as_str);
    assert!(out_name != dataset, "out_dataset must differ from dataset (the original is never written)");

    let in_dir = Path::new("training").join("data").join(dataset);
    let out_dir = Path::new("training").join("data").join(out_name);
    let mut shards: Vec<PathBuf> = fs::read_dir(&in_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", in_dir.display(), e))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "csv")
                 && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('w')))
        .collect();
    shards.sort();
    assert!(!shards.is_empty(), "no shards in {}", in_dir.display());
    fs::create_dir_all(&out_dir).expect("cannot create output folder");

    println!("[relabel] {} → {} — {} shards, {} workers, {} nodes per position, net {}",
             dataset, out_name, shards.len(), workers, nodes, net);
    let engine_cfg = crate::engine_for(net, nodes);
    let limits = MoveLimits { ply: 60, time_secs: 60, randomize: 0, nodes };
    let next = Arc::new(AtomicUsize::new(0));
    let shards = Arc::new(shards);
    let started = Instant::now();

    let handles: Vec<_> = (0..workers).map(|w| {
        let (next, shards, out_dir, cfg) = (next.clone(), shards.clone(), out_dir.clone(), engine_cfg.clone());
        thread::spawn(move || {
            let mut engine = start_engine(&cfg.path, &cfg.init_commands);
            let mut cache: HashMap<[u8; 36], String> = HashMap::new();
            let (mut searched, mut reused) = (0usize, 0usize);
            loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(input) = shards.get(i) else { break };
                let output = out_dir.join(input.file_name().unwrap());
                trim_partial_line(&output);
                let done = File::open(&output).map(|f| BufReader::new(f).lines().count()).unwrap_or(0);
                let mut writer = BufWriter::new(OpenOptions::new().create(true).append(true).open(&output)
                    .unwrap_or_else(|e| panic!("cannot open {}: {}", output.display(), e)));
                let reader = BufReader::new(File::open(input).expect("cannot open shard"));
                for (n, line) in reader.lines().enumerate().skip(done) {
                    let line = line.expect("read error");
                    let mut key = [0u8; 36];
                    let mut board = BoardState::new();
                    for (sq, v) in line.split(',').take(36).enumerate() {
                        let p: usize = v.trim().parse().expect("bad square");
                        key[sq] = p as u8;
                        board.data[sq] = p;
                    }
                    let extra = match cache.get(&key) {
                        Some(e) => { reused += 1; e.clone() }
                        None => {
                            let mut reply = get_move(&mut engine, board, limits, false);
                            if matches!(reply, Reply::NoReply { .. }) {
                                // Engine died or hung: restart it and try once more.
                                engine = start_engine(&cfg.path, &cfg.init_commands);
                                reply = get_move(&mut engine, board, limits, false);
                            }
                            let e = match reply {
                                Reply::Move { mv, score, time_secs, depth, nodes, pv, depth_scores } =>
                                    search_columns(score, depth, time_secs, Some(&mv), false, nodes, &pv, &depth_scores),
                                Reply::Draw { depth } => search_columns(0.0, depth, 0.0, None, false, 0, "", &[]),
                                Reply::NoReply { depth } => format!("nan,{},nan,0,none,0,,", depth),
                            };
                            searched += 1;
                            cache.insert(key, e.clone());
                            e
                        }
                    };
                    writeln!(writer, "{},{}", line, extra).unwrap();
                    if n % 500 == 499 {
                        writer.flush().unwrap();
                    }
                }
                writer.flush().unwrap();
                let hours = started.elapsed().as_secs_f64() / 3600.0;
                println!("[w{}] {} done — {} searched, {} cached so far ({:.0} rows/h for this worker)",
                         w, output.display(), searched, reused, (searched + reused) as f64 / hours.max(1e-9));
            }
            engine.send("quit");
            (searched, reused)
        })
    }).collect();

    let (mut searched, mut reused) = (0usize, 0usize);
    for h in handles {
        let (s, r) = h.join().unwrap();
        searched += s;
        reused += r;
    }
    let hours = started.elapsed().as_secs_f64() / 3600.0;
    println!("[relabel] done — {} rows ({} searched, {} reused from cache) in {:.2} h, {:.0} rows/h → {}",
             searched + reused, searched, reused, hours, (searched + reused) as f64 / hours.max(1e-9),
             out_dir.display());
}
