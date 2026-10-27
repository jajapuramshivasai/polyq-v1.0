//! Exact vs pair-decomposed vs Monte-Carlo on larger T-heavy circuits (dense reference).
use polyq::dense;
use polyq::rng::{random_layered, Rng};
use polyq::{SimOptions, Simulator, Strategy};
use std::time::Instant;
fn main() {
    println!("{:>3} {:>3} {:>3} | {:>10} {:>6} {:>10} {:>5} | {:>10} {:>12} {:>9}", "n", "rnd", "T", "cached ms", "d", "auto ms", "pairs", "MC 1e5 ms", "MC |err|", "stderr");
    for &(n, rounds) in &[(8usize, 3usize), (10, 3), (12, 3), (14, 3), (16, 3)] {
        let mut rng = Rng::new(7 + n as u64);
        let c = random_layered(n, rounds, &mut rng);
        let want = dense::statevector(&c, &vec![false; n]);
        let y = 5u64;
        let t = Instant::now();
        let s1 = Simulator::with_options(&c, SimOptions { strategy: Strategy::Cached, ..Default::default() });
        let a1 = s1.amplitude_index(0, y);
        let t1 = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let s2 = Simulator::new(&c);
        let a2 = s2.amplitude_index(0, y);
        let t2 = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let (e, se) = s1.amplitude_estimate(0, y, 100_000, 1);
        let t3 = t.elapsed().as_secs_f64() * 1e3;
        assert!((a1 - want[y as usize]).norm() < 1e-8 && (a2 - want[y as usize]).norm() < 1e-8);
        println!("{:>3} {:>3} {:>3} | {:>10.2} {:>6} {:>10.2} {:>5} | {:>10.2} {:>12.2e} {:>9.2e}", n, rounds, c.t_count(), t1, s1.stats().branch_dim, t2, s2.pairs(), t3, (e - want[y as usize]).norm(), se);
    }
}
