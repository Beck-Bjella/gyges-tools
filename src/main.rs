mod board;
mod data_gen;
mod game;
mod match_runner;
mod sprt;
mod ugi_engine;

use game::Schedules;
use match_runner::run_sprt_match;
use sprt::SprtConfig;

#[derive(Clone)]
pub struct EngineConfig {
    pub name: String,
    pub path: String,
    pub init_commands: Vec<String>,
}

/// GygesRust v2.0.0 release build. Its embedded default network is
/// pair1024_hce-10k_e9 (pair encoding, 5886 -> 1024 -> 1; see the NOTES.md registry).
/// Build it from the GygesRust repo and copy `target/release/gyges_engine(.exe)` here.
const ENGINE_V2: &str = if cfg!(windows) { "./engines/gyges-v2.0.0.exe" } else { "./engines/gyges-v2.0.0" };

fn main() {
    // `gyges-tools datagen [dataset] [workers] [hours] [nodes] [net]` runs data gen with
    // those overrides (for servers, no code edits needed).
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("datagen") {
        run_data_gen(&args[1..]);
        return;
    }
    // `gyges-tools sprt <net A> <net B> <nodes A> [nodes B]` runs one match
    // with run_sprt's settings but these engines and node caps (for queued runs).
    if args.first().map(String::as_str) == Some("sprt") {
        run_sprt_cli(&args[1..]);
        return;
    }

    // Pick one. SPRT match is the default workflow.
    run_sprt();
    // run_data_gen(&[]);
}

// =====================================================================
//                        SPRT MATCH (default)
// =====================================================================
//
// Plays paired games (same opening, swapped first mover) between ENGINE_A
// and ENGINE_B. Prints a full summary box every pair.
//
// To change what's tested: edit the blocks below. Nets are named
// {arch}_{dataset}_e{epoch} (NOTES.md registry) and live in engines/weights/.
// Any net loaded with weightsPath must match the width compiled into the
// engine (5886 -> 1024 -> 1); the engine rejects other sizes.

#[allow(dead_code)]
fn run_sprt() {
    // ---- Engines under test ------------------------------------------
    // ENGINE_A is the "candidate" — H1 asks whether it scores ≥ h1_score.
    let engine_a = EngineConfig {
        name: String::from("pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9"),
        path: String::from(ENGINE_V2),
        init_commands: vec![
            String::from("setoption nn true"),
            String::from("setoption weightsPath engines/weights/pair1024_gen1-10k+gen1-10k-aws+gen1-10k-aws2_e9.bin"),
            String::from("setoption threads 1"),
        ],
    };
    // HCE on the same engine. Baseline: the release net @10k vs HCE @100k = 0.462.
    let engine_b = EngineConfig {
        name: String::from("HCE @100k"),
        path: String::from(ENGINE_V2),
        init_commands: vec![
            String::from("setoption nn false"),
            String::from("setoption threads 1"),
        ],
    };
    // The release net, embedded in 2.0.0 (no weightsPath needed):
    // let engine_b = EngineConfig {
    //     name: String::from("pair1024_hce-10k_e9"),
    //     path: String::from(ENGINE_V2),
    //     init_commands: vec![
    //         String::from("setoption nn true"),
    //         String::from("setoption threads 1"),
    //     ],
    // };
    // Size-matched HCE-data net:
    // let engine_b = EngineConfig {
    //     name: String::from("pair1024_hce-10k-part1_e6"),
    //     path: String::from(ENGINE_V2),
    //     init_commands: vec![
    //         String::from("setoption nn true"),
    //         String::from("setoption weightsPath engines/weights/pair1024_hce-10k-part1_e6.bin"),
    //         String::from("setoption threads 1"),
    //     ],
    // };

    // ---- Schedules ---------------------------------------------------
    // One set per engine. Node-budget matches (fixed nodes, generous time cap)
    // compare evals on a known search workload; equal budgets on both sides is
    // how all NN-vs-NN tests have been run.
    let a_schedules = Schedules {
        depth:     vec![30],       // ply cap per move
        time:      vec![120],      // seconds per move (hard cap; runner kills at 2x + 5s)
        randomize: vec![0],        // engine-side move randomization
        nodes:     vec![10_000],   // node cap per move (0 = no cap)
    };
    // HCE gets 10x the nodes (same setup as the release-net baseline).
    let b_schedules = Schedules { nodes: vec![100_000], ..a_schedules.clone() };

    // ---- SPRT parameters ---------------------------------------------
    // h1_score is the bar for "engine_a is meaningfully better."
    //   0.505 → ~3.5 chess-Elo equiv  (very fine tuning)
    //   0.510 → ~7    "               (small edge)
    //   0.520 → ~14   "               (clear edge — default)
    //   0.550 → ~35   "               (strong improvement)
    let sprt_cfg = SprtConfig {
        h0_score: 0.500,
        h1_score: 0.520,
        alpha: 0.05,
        beta:  0.05,
        min_pairs: 16,
        max_pairs: 200,                 // 200 pairs = 400 games; raise (or Ctrl-C) as needed
        auto_stop_on_decision: false,   // keep running past verdict so CIs keep tightening
    };

    // ---- Parallelism -------------------------------------------------
    // Games played at once. Each runs two engines, and each 2.0.0 engine
    // allocates a ~400 MB transposition table.
    let thread_count = 8;

    // ---- Run ---------------------------------------------------------
    let _summary = run_sprt_match(engine_a, engine_b, sprt_cfg, a_schedules, b_schedules, thread_count, false);
}

/// An engine for `net`: a file name in engines/weights/ (no `.bin`),
/// `release` (the net embedded in 2.0.0) or `hce` (no net).
fn engine_for(net: &str, nodes: u64) -> EngineConfig {
    let (name, nn) = match net {
        "release" => (String::from("pair1024_hce-10k_e9"), vec![String::from("setoption nn true")]),
        "hce" => (format!("HCE @{}k", nodes / 1000), vec![String::from("setoption nn false")]),
        _ => (String::from(net), vec![
            String::from("setoption nn true"),
            format!("setoption weightsPath engines/weights/{net}.bin"),
        ]),
    };
    if net != "release" && net != "hce" {
        assert!(std::path::Path::new(&format!("engines/weights/{net}.bin")).exists(), "no net engines/weights/{net}.bin");
    }
    let mut init_commands = nn;
    init_commands.push(String::from("setoption threads 1"));
    EngineConfig { name, path: String::from(ENGINE_V2), init_commands }
}

fn run_sprt_cli(args: &[String]) {
    let usage = "usage: gyges-tools sprt <net A> <net B> <nodes A> [nodes B]";
    let (a, b) = (args.first().expect(usage), args.get(1).expect(usage));
    let nodes_a: u64 = args.get(2).expect(usage).parse().expect("nodes must be a number");
    let nodes_b: u64 = args.get(3).map_or(nodes_a, |n| n.parse().expect("nodes must be a number"));

    // Same settings as run_sprt.
    let schedules = |nodes| Schedules { depth: vec![30], time: vec![120], randomize: vec![0], nodes: vec![nodes] };
    let sprt_cfg = SprtConfig {
        h0_score: 0.500,
        h1_score: 0.520,
        alpha: 0.05,
        beta:  0.05,
        min_pairs: 16,
        max_pairs: 200,
        auto_stop_on_decision: false,
    };
    let _summary = run_sprt_match(engine_for(a, nodes_a), engine_for(b, nodes_b), sprt_cfg,
                                  schedules(nodes_a), schedules(nodes_b), 8, false);
}

// =====================================================================
//                         TRAINING DATA GENERATION
// =====================================================================
//
// Self-play games using a single engine config. Shards go to
// training/data/<dataset>/w<NN>.csv; add a row for the dataset to the NOTES.md
// registry. Re-running with the same name resumes where each worker stopped.
// Each game is flushed as it finishes, so stopping early is safe.
//
// Two machines must not share a dataset name: shard names and game ids come
// from the name + worker number, so give each machine its own (gen1-10k-aws).

#[allow(dead_code)]
fn run_data_gen(overrides: &[String]) {
    // {who played}-{nodes per move}: hce-…, or gen{N}-… for NN self-play
    // (gen1 = played by the release net pair1024_hce-10k_e9).
    let dataset = overrides.first().map_or("gen1-10k", String::as_str);

    // ---- Engine -------------------------------------------------------
    // The player: 2.0.0 with its embedded net (pair1024_hce-10k_e9) unless a
    // net from engines/weights/ is given (gen2 = played by a gen1-trained net).
    let player = overrides.get(4).map_or("release", String::as_str);
    let engine = engine_for(player, 0);

    // ---- Schedules ----------------------------------------------------
    // The node cap applies to the randomized opening moves too: uncapped,
    // 2.0.0 spends the full maxTime searching each random move.
    let schedules = Schedules {
        depth:     vec![60],         // effectively no depth cap
        time:      vec![5],          // 5 sec per move (safety cap)
        randomize: vec![1, 1, 0],    // randomize first two moves only
        nodes:     vec![overrides.get(3).map_or(10_000, |n| n.parse().expect("nodes must be a number"))],  // default 10k nodes on every move
    };

    // ---- Run length ---------------------------------------------------
    // Workers stop at whichever comes first.
    let games_per_worker = 1_000_000;
    let max_hours        = Some(overrides.get(2).map_or(10.0, |h| h.parse().expect("hours must be a number")));

    // ---- Parallelism --------------------------------------------------
    // 8 cores / 16 threads: 14 workers for max output, 8 to keep the machine usable.
    let thread_count     = overrides.get(1).map_or(12, |w| w.parse().expect("workers must be a number"));

    data_gen::generate_data_parallel(dataset, engine, schedules, games_per_worker, max_hours, thread_count, false);
}
