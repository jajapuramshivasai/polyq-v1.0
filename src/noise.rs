//! Pauli-noise Monte Carlo on top of a **cached** normal form.
//!
//! Because `X` is a free wire-flag in the compiler (see [`crate::pathsum`]), a Pauli
//! fault changes only *values* — Z-type linear terms, constants and rotation signs — never
//! the matrix `B`, the parity of `v`, or the variable numbering.  A faulty shot therefore
//! re-uses the fault-free [`Engine`] via [`Engine::rebind`]; no elimination is repeated.
//!
//! The estimator targets `P_noisy(y|x) = Σ_f Pr(f)·|<y|U_f|x>|²`, an exact mixture over
//! Pauli fault patterns `f`, by sampling `f` and evaluating one strong-simulation
//! amplitude per shot (unbiased; standard error is reported).

use crate::circuit::{Circuit, Gate};
use crate::engine::{Engine, EngineOptions};
use crate::pathsum::{Fault, Pauli, PathSum};
use crate::rng::Rng;
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Sample depolarising faults: after every gate, each touched qubit independently suffers
/// X, Y or Z with total probability `p` (each `p/3`).
pub fn sample_faults(c: &Circuit, p: f64, rng: &mut Rng) -> Vec<Fault> {
    let mut out = Vec::new();
    for (i, g) in c.gates.iter().enumerate() {
        for q in g.qubits() {
            if rng.next_f64() < p {
                let pauli = [Pauli::X, Pauli::Y, Pauli::Z][rng.below(3)];
                out.push(Fault { position: i + 1, qubit: q, pauli });
            }
        }
    }
    out
}

/// The circuit with explicit Pauli gates inserted (reference semantics of a fault list).
pub fn circuit_with_faults(c: &Circuit, faults: &[Fault]) -> Circuit {
    let mut fs: Vec<&Fault> = faults.iter().collect();
    fs.sort_by_key(|f| f.position);
    let mut out = Circuit::new(c.num_qubits);
    let mut fi = 0;
    for (pos, g) in c.gates.iter().enumerate() {
        while fi < fs.len() && fs[fi].position <= pos {
            out.push(pauli_gate(fs[fi]));
            fi += 1;
        }
        out.push(g.clone());
    }
    while fi < fs.len() {
        out.push(pauli_gate(fs[fi]));
        fi += 1;
    }
    out
}

fn pauli_gate(f: &Fault) -> Gate {
    match f.pauli {
        Pauli::X => Gate::X(f.qubit),
        Pauli::Y => Gate::Y(f.qubit),
        Pauli::Z => Gate::Z(f.qubit),
    }
}

/// Monte-Carlo estimate with standard error.
#[derive(Copy, Clone, Debug)]
pub struct Estimate {
    /// Sample mean of `|amplitude|²`.
    pub mean: f64,
    /// Standard error of the mean.
    pub stderr: f64,
    /// Shots for which the cached structure could not be reused (expected: 0).
    pub rebuilds: usize,
}

/// Noisy-outcome estimator with a cached fault-free engine.
pub struct NoisyEstimator {
    circuit: Circuit,
    base: Engine,
}

impl NoisyEstimator {
    /// Build (one elimination).
    pub fn new(circuit: &Circuit) -> Self {
        let ps = PathSum::from_circuit(circuit);
        let base = Engine::new(&ps, &ps.io_vars(), EngineOptions::default());
        Self { circuit: circuit.clone(), base }
    }

    /// `<output| U_f |input>` for an explicit fault list; `rebuilt` is set if the cached
    /// structure had to be abandoned.
    pub fn faulted_amplitude(&self, input: &[bool], output: &[bool], faults: &[Fault], rebuilt: Option<&AtomicUsize>) -> Complex64 {
        let ps = PathSum::from_circuit_faulted(&self.circuit, faults);
        let Some(bd) = ps.boundary(input, output) else { return Complex64::new(0.0, 0.0) };
        match self.base.rebind(&ps) {
            Some(e) => e.eval_assignment(&bd),
            None => {
                if let Some(r) = rebuilt {
                    r.fetch_add(1, Ordering::Relaxed);
                }
                Engine::new(&ps, &ps.io_vars(), EngineOptions::default()).eval_assignment(&bd)
            }
        }
    }

    /// Estimate `P_noisy(output | input)` under depolarising noise `p` with `shots` samples.
    pub fn estimate_probability(&self, input: &[bool], output: &[bool], p: f64, shots: usize, seed: u64) -> Estimate {
        let rebuilt = AtomicUsize::new(0);
        let (s1, s2) = (0..shots)
            .into_par_iter()
            .map(|i| {
                let mut rng = Rng::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let f = sample_faults(&self.circuit, p, &mut rng);
                let a = self.faulted_amplitude(input, output, &f, Some(&rebuilt)).norm_sqr();
                (a, a * a)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let n = shots as f64;
        let mean = s1 / n;
        let var = (s2 / n - mean * mean).max(0.0);
        Estimate { mean, stderr: (var / n).sqrt(), rebuilds: rebuilt.load(Ordering::Relaxed) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense;
    use crate::rng::random_layered;
    use crate::sim::bits_of;

    #[test]
    fn faulted_amplitudes_match_dense_and_reuse_structure() {
        let mut rng = Rng::new(8);
        for _ in 0..25 {
            let n = 2 + rng.below(3);
            let mut c = random_layered(n, 2, &mut rng);
            c.x(0).y(1 % n).swap(0, n - 1);
            let est = NoisyEstimator::new(&c);
            let rebuilt = AtomicUsize::new(0);
            for _ in 0..6 {
                let f = sample_faults(&c, 0.15, &mut rng);
                let cf = circuit_with_faults(&c, &f);
                let input = bits_of(n, rng.below(1 << n) as u64);
                let sv = dense::statevector(&cf, &input);
                for y in 0..(1u64 << n) {
                    let a = est.faulted_amplitude(&input, &bits_of(n, y), &f, Some(&rebuilt));
                    assert!((a - sv[y as usize]).norm() < 1e-8, "{a} vs {}", sv[y as usize]);
                }
            }
            assert_eq!(rebuilt.load(Ordering::Relaxed), 0, "Pauli faults must reuse the cached structure");
        }
    }

    #[test]
    fn monte_carlo_matches_exact_mixture_small_case() {
        // 1 qubit, H T H, depolarising p after each gate: compare with exact enumeration
        let mut c = Circuit::new(1);
        c.h(0).t(0).h(0);
        let p = 0.2;
        let est = NoisyEstimator::new(&c);
        let e = est.estimate_probability(&[false], &[false], p, 60_000, 5);
        // exact: enumerate fault patterns (4 options after each of 3 gates)
        let mut exact = 0.0;
        for code in 0..64usize {
            let mut prob = 1.0;
            let mut faults = Vec::new();
            for g in 0..3 {
                match (code >> (2 * g)) & 3 {
                    0 => prob *= 1.0 - p,
                    k => {
                        prob *= p / 3.0;
                        faults.push(Fault { position: g + 1, qubit: 0, pauli: [Pauli::X, Pauli::Y, Pauli::Z][k - 1] });
                    }
                }
            }
            let sv = dense::statevector(&circuit_with_faults(&c, &faults), &[false]);
            exact += prob * sv[0].norm_sqr();
        }
        assert!((e.mean - exact).abs() < 4.0 * e.stderr + 1e-3, "mc {} ± {} vs exact {}", e.mean, e.stderr, exact);
        assert_eq!(e.rebuilds, 0);
    }
}
