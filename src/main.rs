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
/// Build it from the GygesRust repo and copy `target/release/gyges_engine.exe` here.
const ENGINE_V2: &str = "./engines/gyges-v2.0.0.exe";

fn main() {
    // Pick one. SPRT match is the default workflow.
    run_sprt();
    // run_data_gen();
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
        name: String::from("pair1024_hce-10k_e9 @10k"),
        path: String::from(ENGINE_V2),
        init_commands: vec![
            String::from("setoption nn true"),
            String::from("setoption threads 1"),
        ],
    };
    let engine_b = EngineConfig {
        name: String::from("HCE @200k"),
        path: String::from(ENGINE_V2),
        init_commands: vec![
            String::from("setoption nn false"),
            String::from("setoption threads 1"),
        ],
    };
    // A different net on the same engine:
    // let engine_a = EngineConfig {
    //     name: String::from("pair1024_gen1-5k_e6"),
    //     path: String::from(ENGINE_V2),
    //     init_commands: vec![
    //         String::from("setoption nn true"),
    //         String::from("setoption weightsPath engines\\weights\\pair1024_gen1-5k_e6.bin"),
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
    let b_schedules = Schedules {
        nodes: vec![200_000],
        ..a_schedules.clone()
    };

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
    let thread_count = 4;

    // ---- Run ---------------------------------------------------------
    let _summary = run_sprt_match(engine_a, engine_b, sprt_cfg, a_schedules, b_schedules, thread_count, false);
}

// =====================================================================
//                         TRAINING DATA GENERATION
// =====================================================================
//
// Self-play games using a single engine config. Shards go to
// training/data/<dataset>/w<NN>.csv; add a row for the dataset to the NOTES.md
// registry. Re-running with the same name resumes where each worker stopped.
// Each game is flushed as it finishes, so stopping early is safe.

#[allow(dead_code)]
fn run_data_gen() {
    // {who played}-{nodes per move}: hce-…, or gen{N}-… for NN self-play
    // (gen1 = played by the release net pair1024_hce-10k_e9).
    let dataset = "gen1-10k";

    // ---- Engine -------------------------------------------------------
    // 2.0.0 with its embedded net (pair1024_hce-10k_e9).
    let engine = EngineConfig {
        name: String::from("pair1024_hce-10k_e9"),
        path: String::from(ENGINE_V2),
        init_commands: vec![
            String::from("setoption nn true"),
            String::from("setoption threads 1"),
        ],
    };

    // ---- Schedules ----------------------------------------------------
    // The node cap applies to the randomized opening moves too: uncapped,
    // 2.0.0 spends the full maxTime searching each random move.
    let schedules = Schedules {
        depth:     vec![60],         // effectively no depth cap
        time:      vec![5],          // 5 sec per move (safety cap)
        randomize: vec![1, 1, 0],    // randomize first two moves only
        nodes:     vec![10_000],     // 10k nodes on every move
    };

    // ---- Run length ---------------------------------------------------
    // Workers stop at whichever comes first.
    let games_per_worker = 1_000_000;
    let max_hours        = Some(3.0);

    // ---- Parallelism --------------------------------------------------
    // 8 cores / 16 threads: 14 workers for max output, 8 to keep the machine usable.
    let thread_count     = 14;

    data_gen::generate_data_parallel(dataset, engine, schedules, games_per_worker, max_hours, thread_count, false);
}
