//! Benchmarks: `cargo run --release --bin benchmark`.
//! Compares dense, cached Z-branch and auto (pair-decomposed) engines on layered Clifford+T circuits.
use polyq::dense;
use polyq::rng::{random_layered, Rng};
use polyq::{SimOptions, Simulator, Strategy};
use std::time::Instant;

fn main() {
    println!("{:>3} {:>5} {:>3} | {:>9} {:>10} {:>10} | {:>4} {:>4} {:>5}", "n", "rounds", "T", "dense ms", "cached ms", "auto ms", "d", "p", "m");
    for &(n, rounds) in &[(4usize, 2usize), (6, 2), (8, 2), (10, 2), (10, 3), (12, 2)] {
        let mut rng = Rng::new(1000 + n as u64);
        let c = random_layered(n, rounds, &mut rng);
        let input = vec![false; n];
        let t0 = Instant::now();
        let want = dense::statevector(&c, &input);
        let td = t0.elapsed().as_secs_f64() * 1e3;
        let mut res = Vec::new();
        for strat in [Strategy::Cached, Strategy::Auto] {
            let t = Instant::now();
            let sim = Simulator::with_options(&c, SimOptions { strategy: strat, ..Default::default() });
            let sv = sim.statevector(&input);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let err = sv.iter().zip(&want).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-7, "mismatch {err}");
            res.push((ms, sim.stats(), sim.pairs()));
        }
        println!("{:>3} {:>5} {:>3} | {:>9.2} {:>10.2} {:>10.2} | {:>4} {:>4} {:>5}", n, rounds, c.t_count(), td, res[0].0, res[1].0, res[0].1.branch_dim, res[1].2, res[0].1.free_vars);
    }
}
