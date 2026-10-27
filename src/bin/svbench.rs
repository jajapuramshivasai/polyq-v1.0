//! Statevector scaling: `svbench <n> <rounds>`; run with RAYON_NUM_THREADS=1/2 to see threading.
use polyq::rng::{random_layered, Rng};
use polyq::{Simulator, dense};
use std::time::Instant;
fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|x| x.parse().unwrap()).collect();
    let (n, rounds) = (a[0], a[1]);
    let mut rng = Rng::new(42);
    let c = random_layered(n, rounds, &mut rng);
    let t = Instant::now();
    let sim = Simulator::new(&c);
    let tb = t.elapsed();
    let t = Instant::now();
    let sv = sim.statevector(&vec![false; n]);
    let ts = t.elapsed();
    let st = sim.stats();
    println!("n={n} T={} build {:?} statevector {:?} ({:.2} us/amp) d={} pairs={} threads={}", c.t_count(), tb, ts, ts.as_secs_f64() * 1e6 / sv.len() as f64, st.branch_dim, sim.pairs(), rayon::current_num_threads());
    if n <= 20 {
        let t = Instant::now();
        let d = dense::statevector(&c, &vec![false; n]);
        let err = sv.iter().zip(&d).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max);
        println!("dense {:?} max err {:.2e}", t.elapsed(), err);
    }
}
