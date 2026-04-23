mod board;
mod engine_analyzer;
mod ugi_engine;

use std::env;
use engine_analyzer::EngineAnalyzer;

#[derive(Clone)]
pub struct EngineConfig {
    pub name: String,
    pub path: String,
    pub init_commands: Vec<String>,
}

// Main entry point for generating training data
fn main() {
    // Analysis
    // run_analysis();

    // ========== DATA CONVERSION ==========
    // for i in 0..1 {
    //     let input = format!("training\\data\\hce_100kn.csv");
    //     let output = format!("training\\data\\hce_100kn_converted.csv");
    //     println!("Converting file {} of 24: {} -> {}", i + 1, input, output);
    //     convert_dataset(&input, &output);
    // }

    // ========== DATA GENERATION ==========
    let args: Vec<String> = env::args().collect();

    let thread_count: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            println!("Usage: gyges-tools <thread_count> [games_per_worker]");
            std::process::exit(1);
        });

    let games_per_worker: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000);

    println!("Starting with {} workers, {} games per worker", thread_count, games_per_worker);

    // Engines
    let hce_baseline = EngineConfig {
        name: String::from("HCE"),
        path: String::from("./engines/gyges_144x256"),
        init_commands: vec![
            String::from("setoption nn false"),
            String::from("setoption nnAcc off"),
        ],
    };

    // Setup
    let debug = false;
    let depth_schedule = vec![60]; // NO DEPTH LIMIT
    let time_schedule = vec![240]; // 4 min max
    let randomize_schedule = vec![1, 1, 0];
    let node_schedule = vec![1000, 1000, 100_000]; // 100k node limit

    let mut engine_analyser = EngineAnalyzer::new(
        hce_baseline.clone(),
        hce_baseline.clone(),
        depth_schedule,
        time_schedule,
        randomize_schedule,
        node_schedule,
        debug,
    );

    // Kill the engines spawned by new() — parallel workers spawn their own
    engine_analyser.kill_engines();

    // Generate training data
    engine_analyser.generate_data_parallel(games_per_worker, thread_count);

}


/// Runs engine vs engine analysis for testing/comparison.
#[allow(dead_code)]
fn run_analysis() {
    let hce = EngineConfig {
        name: String::from("HCE"),
        path: String::from("./engines/gyges_144x256"),
        init_commands: vec![String::from("setoption nn false")],
    };
    let nn_144x128 = EngineConfig {
        name: String::from("NN 128"),
        path: String::from("./engines/gyges_144x128"),
        init_commands: vec![
            String::from("setoption nn true"),
            String::from("setoption nnAcc off"),
            String::from("setoption weightsPath engines\\weights\\e100_144x128.bin"),
        ],
    };
    let nn_144x256 = EngineConfig {
        name: String::from("NN 256"),
        path: String::from("./engines/gyges_144x256"),
        init_commands: vec![
            String::from("setoption nn true"),
            String::from("setoption nnAcc off"),
            String::from("setoption weightsPath engines\\weights\\e100_144x256.bin"),
        ],
    };
    let nn_144x256x32 = EngineConfig {
        name: String::from("NN 256x32"),
        path: String::from("./engines/gyges_144x256x32"),
        init_commands: vec![
            String::from("setoption nn true"),
            String::from("setoption nnAcc off"),
            String::from("setoption weightsPath engines\\weights\\e50_144x256x32.bin"),
        ],
    };
    let nn_large = EngineConfig {
        name: String::from("NN LARGE"),
        path: String::from("./engines/gyges_large"),
        init_commands: vec![
            String::from("setoption nn true"),
            String::from("setoption nnAcc off"),
            String::from("setoption weightsPath engines\\weights\\e10_large_new.bin"),
        ],
    };


    let debug = true;
    let depth_schedule = vec![30]; // NO DEPTH LIMIT 
    let time_schedule = vec![120]; // 4 min max
    let randomize_schedule = vec![0]; // NO RANDOMIZATION
    let node_schedule = vec![100_000]; // NO NODE LIMIT

    let mut analyser = EngineAnalyzer::new(
        nn_144x256,
        nn_large,
        depth_schedule,
        time_schedule,
        randomize_schedule,
        node_schedule,
        debug,
    );

    let data = analyser.analyze(1000);
    data.print_summary();

    analyser.quit();

}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////

/// Converts an existing position CSV to include per-square control features.
/// The board is already in "my perspective" — current player's home rank at rank 0.
///
/// Input CSV columns:  sq0..sq35 (piece type 0-3), outcome (-1/1)  — no header
/// Output CSV columns: sq0_piece, sq0_ctrl, ..., sq35_piece, sq35_ctrl, outcome
///
/// Control values: +1 = my unique (P1), 0 = shared/empty, -1 = opponent unique (P2)
///
fn convert_dataset(input_path: &str, output_path: &str) {
    use std::fs::File;
    use std::io::{BufRead, BufReader, BufWriter, Write};
    use gyges::board::BoardState;
    use gyges::moves::movegen::MoveGen;
    use gyges_engine::search::evaluation::EvaluationContext;

    let reader = BufReader::new(File::open(input_path).expect("Cannot open input CSV"));
    let mut writer = BufWriter::new(File::create(output_path).expect("Cannot create output CSV"));

    let mut mg = MoveGen::default();
    let mut count = 0;

    for line in reader.lines() {
        let line = line.unwrap();

        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 37 {
            continue;
        }

        // Reconstruct board string from the 36 square values
        let board_str: String = cols[..36].iter().map(|s| s.trim()).collect();
        let outcome = cols[36].trim();

        let mut board = BoardState::from(board_str.as_str());

        // Compute piece control bitboards — P1 = current player (at rank 0)
        let ctx = EvaluationContext::new(&mut board, &mut mg);

        // Build output row: piece_type + control scalar per square
        let mut row: Vec<String> = Vec::with_capacity(73);
        let mut control_grid = [0i32; 36];
        for sq in 0..36usize {
            let piece_type: i32 = cols[sq].trim().parse().unwrap_or(0);

            let control = if piece_type == 0 {
                0

            } else {
                let bit = 1u64 << sq;
                if ctx.unique_piece_control[0].0 & bit != 0 {
                    1 // my unique
                } else if ctx.unique_piece_control[1].0 & bit != 0 {
                    -1 // opponent unique
                } else {
                    0 // shared
                }

            };

            control_grid[sq] = control;
            row.push(piece_type.to_string());
            row.push(control.to_string());

        }

        row.push(outcome.to_string());

        writeln!(writer, "{}", row.join(",")).unwrap();

        count += 1;

        // Print first 5 boards for verification
        if count <= 5 {
            println!("── Row {} ── outcome: {}", count, outcome);
            println!("  {}", board);
            for row_idx in 0..6 {
                print!("  ");
                for col_idx in 0..6 {
                    let sq = row_idx * 6 + col_idx;
                    let piece: i32 = cols[sq].trim().parse().unwrap_or(0);
                    let ctrl_str = match control_grid[sq] {
                        1 => "me",
                        -1 => "op",
                        _ => " =",
                    };
                    print!("[{} {}] ", piece, ctrl_str);

                }
                println!();
                
            }
            println!();
            
        }

        if count % 5000 == 0 {
            println!("Processed {} rows...", count);
        }

    }

    println!("Done. {} rows written to {}", count, output_path);

}
