mod board;
mod engine_analyzer;
mod ugi_engine;

use engine_analyzer::EngineAnalyzer;

#[derive(Clone)]
pub struct EngineConfig {
    pub name: String,
    pub path: String,
}

fn main() {
    // Engines
    let hce_baseline = EngineConfig {
        name: String::from("UPDATED BASELINE"),
        path: String::from("./engines/baseline"),
    };

    // Setup
    let debug = false;

    // First 2 moves at ply 3 (/w randomization) then 1 second per move
    let depth_schedule = vec![3, 3, 30];
    let time_schedule = vec![5, 5, 1];
    let randomize_schedule = vec![1, 1, 0];

    let mut engine_analyser = EngineAnalyzer::new(
        hce_baseline.clone(),
        hce_baseline.clone(),
        depth_schedule,
        time_schedule,
        randomize_schedule,
        debug,
    );

    // Run analysis
    // let data = engine_analyser.analyze(1000);
    // data.print_summary();

    // Generate training data
    engine_analyser.generate_data_parallel(50000, 16);

    engine_analyser.quit();

}
