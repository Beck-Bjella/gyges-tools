//! Sequential Probability Ratio Test (SPRT) for engine vs engine matches.
//!
//! Pure score-space implementation — no Elo conversion. Operates on per-pair
//! scores in [0, 1] from paired games (same opening, swapped first mover).
//!
//! Hypothesis form:
//!   H0: pair_score ≤ h0_score  (default 0.500 — no advantage)
//!   H1: pair_score ≥ h1_score  (default 0.520 — meaningful edge)

use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct SprtConfig {
    pub h0_score: f64,
    pub h1_score: f64,
    pub alpha: f64,
    pub beta: f64,
    pub min_pairs: usize,
    pub max_pairs: usize,
    /// When true, the match stops the moment SPRT reaches a terminal verdict
    /// (AcceptH0 / AcceptH1 / Inconclusive). When false, the match keeps running
    /// until `max_pairs` or Ctrl-C — useful when you want to leave it going and
    /// watch the verdict line update / CIs tighten over time.
    pub auto_stop_on_decision: bool,
}

impl Default for SprtConfig {
    fn default() -> Self {
        SprtConfig {
            h0_score: 0.500,
            h1_score: 0.520,
            alpha: 0.05,
            beta: 0.05,
            min_pairs: 16,
            max_pairs: 100_000,
            auto_stop_on_decision: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    WarmingUp,
    Continue,
    AcceptH0,
    AcceptH1,
    Inconclusive,
}

impl Decision {
    pub fn is_terminal(self) -> bool {
        matches!(self, Decision::AcceptH0 | Decision::AcceptH1 | Decision::Inconclusive)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PentaCounts {
    pub ll: usize,
    pub ld: usize,
    pub dd: usize,
    pub wd: usize,
    pub ww: usize,
}

impl PentaCounts {
    pub fn total(&self) -> usize {
        self.ll + self.ld + self.dd + self.wd + self.ww
    }
}

/// One pair's worth of game outcomes from engine 1's perspective.
/// Each pair contains exactly 2 games, so e1_wins + e2_wins + draws == 2.
#[derive(Clone, Copy, Debug)]
pub struct PairData {
    pub e1_wins: usize,
    pub e2_wins: usize,
    pub draws: usize,
}

impl PairData {
    /// Pair score for engine 1, in [0, 1]. Wins=1, draws=0.5, losses=0, averaged.
    pub fn score(&self) -> f64 {
        (self.e1_wins as f64 + 0.5 * self.draws as f64) / 2.0
    }

    /// Pentanomial bucket for this pair from E1's perspective.
    pub fn bucket(&self) -> Bucket {
        match (self.e1_wins, self.e2_wins, self.draws) {
            (2, 0, 0) => Bucket::WW,
            (1, 0, 1) => Bucket::WD,
            (0, 1, 1) => Bucket::LD,
            (0, 2, 0) => Bucket::LL,
            // (1,1,0) split + (0,0,2) both-draw collapse to the middle bucket
            _ => Bucket::DD,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Bucket { LL, LD, DD, WD, WW }

pub struct SprtState {
    pub cfg: SprtConfig,
    pub pair_scores: Vec<f64>,
    pub penta: PentaCounts,
    pub e1_total_wins: usize,
    pub e2_total_wins: usize,
    pub draws_total: usize,
    pub start_time: Instant,
}

impl SprtState {
    pub fn new(cfg: SprtConfig) -> Self {
        Self {
            cfg,
            pair_scores: Vec::with_capacity(cfg.max_pairs),
            penta: PentaCounts::default(),
            e1_total_wins: 0,
            e2_total_wins: 0,
            draws_total: 0,
            start_time: Instant::now(),
        }
    }

    pub fn update(&mut self, pair: &PairData) {
        self.pair_scores.push(pair.score());
        match pair.bucket() {
            Bucket::LL => self.penta.ll += 1,
            Bucket::LD => self.penta.ld += 1,
            Bucket::DD => self.penta.dd += 1,
            Bucket::WD => self.penta.wd += 1,
            Bucket::WW => self.penta.ww += 1,
        }
        self.e1_total_wins += pair.e1_wins;
        self.e2_total_wins += pair.e2_wins;
        self.draws_total += pair.draws;
    }

    pub fn n_pairs(&self) -> usize { self.pair_scores.len() }
    pub fn n_games(&self) -> usize { self.n_pairs() * 2 }

    pub fn mean(&self) -> f64 {
        if self.pair_scores.is_empty() { return 0.5; }
        self.pair_scores.iter().sum::<f64>() / self.pair_scores.len() as f64
    }

    /// Sample variance (n-1 denominator).
    pub fn variance(&self) -> f64 {
        let n = self.pair_scores.len();
        if n < 2 { return 0.0; }
        let mean = self.mean();
        let sum_sq: f64 = self.pair_scores.iter().map(|s| (s - mean).powi(2)).sum();
        sum_sq / (n - 1) as f64
    }

    pub fn stderr(&self) -> f64 {
        let n = self.pair_scores.len();
        if n < 2 { return f64::INFINITY; }
        let v = self.variance();
        if v <= 0.0 { return 0.0; }
        (v / n as f64).sqrt()
    }

    /// Wald LLR (matches the formula used by cutechess-cli / fastchess SPRT).
    pub fn llr(&self) -> f64 {
        let n = self.pair_scores.len();
        if n < 2 { return 0.0; }
        let var = self.variance();
        if var <= 0.0 { return 0.0; }
        let mean = self.mean();
        let s0 = self.cfg.h0_score;
        let s1 = self.cfg.h1_score;
        (n as f64 / (2.0 * var)) * ((mean - s0).powi(2) - (mean - s1).powi(2))
    }

    pub fn lower_bound(&self) -> f64 { (self.cfg.beta / (1.0 - self.cfg.alpha)).ln() }
    pub fn upper_bound(&self) -> f64 { ((1.0 - self.cfg.beta) / self.cfg.alpha).ln() }

    /// 95% CI on pair score via normal approx.
    pub fn score_ci_95(&self) -> (f64, f64) {
        let mean = self.mean();
        let se = self.stderr();
        if !se.is_finite() { return (0.0, 1.0); }
        ((mean - 1.96 * se).max(0.0), (mean + 1.96 * se).min(1.0))
    }

    /// Likelihood of Superiority — P(true_score > 0.5).
    pub fn los(&self) -> f64 {
        let se = self.stderr();
        if !se.is_finite() || se <= 0.0 {
            return if self.mean() > 0.5 { 1.0 } else if self.mean() < 0.5 { 0.0 } else { 0.5 };
        }
        let z = (self.mean() - 0.5) / se;
        normal_cdf(z)
    }

    pub fn decision(&self) -> Decision {
        let n = self.n_pairs();
        if n < self.cfg.min_pairs { return Decision::WarmingUp; }
        let llr = self.llr();
        if llr >= self.upper_bound() { return Decision::AcceptH1; }
        if llr <= self.lower_bound() { return Decision::AcceptH0; }
        if n >= self.cfg.max_pairs { return Decision::Inconclusive; }
        Decision::Continue
    }

    pub fn elapsed(&self) -> std::time::Duration { self.start_time.elapsed() }
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz–Stegun 7.1.26 — accurate to ~1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t)
        + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}
