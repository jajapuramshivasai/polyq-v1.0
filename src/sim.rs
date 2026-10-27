//! High-level simulator API.
//!
//! * [`Simulator`] — strong simulation (`amplitude`, `probability`, `statevector`),
//!   marginals and **weak simulation** (`sample`) without ever storing a statevector.
//! * [`equivalence`] — exact global-phase equivalence checking of two circuits via the
//!   trace of the miter `C₂† C₁`.
//!
//! ## Marginals and sampling
//!
//! For a set `M` of measured qubits with outcomes `y_M`,
//!
//! ```text
//!   P(y_M) = <x| C† ( ⊗_{q∈M} |y_q><y_q| ) C |x>
//! ```
//!
//! is a *single* path sum: compile `C` followed by `C†` on the same wires
//! ([`PathSum::sandwich`]); the junction variable of every measured wire is **fixed** to
//! its outcome, the junction of unmeasured wires stays a summed variable (that *is* the
//! partial trace).  Sampling is the usual chain rule over a qubit prefix; the `n` engines
//! (one per prefix length) are built once and cached, so each sample costs `n` cached
//! evaluations.
//!
//! ## Equivalence
//!
//! `C₁ ≡ C₂` up to a global phase iff `|tr(C₂† C₁)| = 2ⁿ`.  The trace is the path sum of
//! the miter with every output variable identified with its input variable and *all*
//! variables free ([`PathSum::trace_closure`]); one engine evaluation returns
//! `tr(U)/2ⁿ = e^{iφ}` and thus also the relative global phase.

use crate::circuit::Circuit;
use crate::engine::{EngineOptions, EngineStats};
use crate::pathsum::PathSum;
use crate::rank::RankEngine;
use crate::rng::Rng;
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::OnceLock;

/// How the non-Clifford part is handled.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Pick the number of decomposed T-pairs with the cost model of [`RankEngine::auto`].
    Auto,
    /// Pure cached Z-branch engine (`2^d` per amplitude, one normal form).
    Cached,
    /// Decompose exactly this many T-pairs (`2^p` stabiliser branches).
    Pairs(usize),
}

/// Simulator configuration.
#[derive(Copy, Clone, Debug)]
pub struct SimOptions {
    /// Non-Clifford strategy.
    pub strategy: Strategy,
    /// Engine tuning.
    pub engine: EngineOptions,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self { strategy: Strategy::Auto, engine: EngineOptions::default() }
    }
}

fn build_engine(ps: &PathSum, fixed: &[usize], o: &SimOptions) -> RankEngine {
    match o.strategy {
        Strategy::Auto => RankEngine::auto(ps, fixed, o.engine),
        Strategy::Cached => RankEngine::new(ps, fixed, 0, o.engine),
        Strategy::Pairs(p) => RankEngine::new(ps, fixed, p, o.engine),
    }
}

/// Expand a basis-state index (bit `q` = qubit `q`) to a bool slice.
pub fn bits_of(n: usize, index: u64) -> Vec<bool> {
    (0..n).map(|q| (index >> q) & 1 == 1).collect()
}

/// Insert `(var, val)` into a deduplicated assignment; `false` on conflict.
fn put(asg: &mut Vec<(usize, bool)>, var: usize, val: bool) -> bool {
    if let Some(e) = asg.iter().find(|e| e.0 == var) {
        return e.1 == val;
    }
    asg.push((var, val));
    true
}

struct Sandwich {
    ps: PathSum,
    junction: Vec<usize>,
    jneg: Vec<bool>,
}

impl Sandwich {
    /// Fixed variables for measuring the given qubits.
    fn fixed_for(&self, measured: impl Iterator<Item = usize>) -> Vec<usize> {
        let mut f = self.ps.io_vars();
        for q in measured {
            if !f.contains(&self.junction[q]) {
                f.push(self.junction[q]);
            }
        }
        f
    }

    /// Boundary assignment for `<x|C† Π_y C|x>`, `None` if contradictory.
    fn assignment(&self, input: &[bool], measured: &[(usize, bool)]) -> Option<Vec<(usize, bool)>> {
        let mut a = Vec::new();
        for q in 0..self.ps.num_qubits {
            if !put(&mut a, q, input[q]) {
                return None;
            }
        }
        for q in 0..self.ps.num_qubits {
            if !put(&mut a, self.ps.outputs[q], input[q] ^ self.ps.out_neg[q]) {
                return None;
            }
        }
        for &(q, y) in measured {
            if !put(&mut a, self.junction[q], y ^ self.jneg[q]) {
                return None;
            }
        }
        Some(a)
    }
}

/// Strong + weak simulator for Clifford+T circuits (see the crate docs).
pub struct Simulator {
    circuit: Circuit,
    ps: PathSum,
    engine: RankEngine,
    opts: SimOptions,
    sandwich: OnceLock<Sandwich>,
    prefix_engines: OnceLock<Vec<RankEngine>>,
}

impl Simulator {
    /// Build with default options ([`Strategy::Auto`]).
    pub fn new(circuit: &Circuit) -> Self {
        Self::with_options(circuit, SimOptions::default())
    }

    /// Build with explicit options.
    pub fn with_options(circuit: &Circuit, opts: SimOptions) -> Self {
        let ps = PathSum::from_circuit(circuit);
        let engine = build_engine(&ps, &ps.io_vars(), &opts);
        Self { circuit: circuit.clone(), ps, engine, opts, sandwich: OnceLock::new(), prefix_engines: OnceLock::new() }
    }

    /// The compiled path sum.
    pub fn path_sum(&self) -> &PathSum {
        &self.ps
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.ps.num_qubits
    }

    /// Normal-form size information of the main engine.
    pub fn stats(&self) -> EngineStats {
        self.engine.stats()
    }

    /// Number of decomposed T-pairs actually used.
    pub fn pairs(&self) -> usize {
        self.engine.pairs()
    }

    /// `<output| C |input>`.
    pub fn amplitude(&self, input: &[bool], output: &[bool]) -> Complex64 {
        match self.ps.boundary(input, output) {
            None => Complex64::new(0.0, 0.0),
            Some(bd) => self.engine.eval_assignment(&bd),
        }
    }

    /// Same with basis-state indices (bit `q` = qubit `q`).
    pub fn amplitude_index(&self, input: u64, output: u64) -> Complex64 {
        let n = self.num_qubits();
        self.amplitude(&bits_of(n, input), &bits_of(n, output))
    }

    /// Unbiased Monte-Carlo estimate of `<output|C|input>` (see [`RankEngine::estimate`]);
    /// returns `(value, standard_error)`.  Meant for circuits whose exact branch enumeration
    /// is too large.
    pub fn amplitude_estimate(&self, input: u64, output: u64, samples: usize, seed: u64) -> (Complex64, f64) {
        let n = self.num_qubits();
        match self.ps.boundary(&bits_of(n, input), &bits_of(n, output)) {
            None => (Complex64::new(0.0, 0.0), 0.0),
            Some(bd) => self.engine.estimate(&bd, samples, seed),
        }
    }

    /// `|<output| C |input>|²`.
    pub fn probability(&self, input: &[bool], output: &[bool]) -> f64 {
        self.amplitude(input, output).norm_sqr()
    }

    /// All `2^n` amplitudes for the given input (parallel over outputs).
    pub fn statevector(&self, input: &[bool]) -> Vec<Complex64> {
        let n = self.num_qubits();
        (0..(1u64 << n))
            .into_par_iter()
            .map_init(
                || self.engine.scratch(),
                |sc, y| match self.ps.boundary(input, &bits_of(n, y)) {
                    None => Complex64::new(0.0, 0.0),
                    Some(bd) => self.engine.eval_assignment_with(&bd, sc),
                },
            )
            .collect()
    }

    fn sandwich(&self) -> &Sandwich {
        self.sandwich.get_or_init(|| {
            let (ps, junction, jneg) = PathSum::sandwich(&self.circuit);
            Sandwich { ps, junction, jneg }
        })
    }

    /// Marginal probability that measuring the listed `(qubit, outcome)` pairs gives those
    /// outcomes.  Builds a dedicated engine each call (use [`sample`](Self::sample) for the
    /// cached prefix chain).
    pub fn marginal_probability(&self, input: &[bool], measured: &[(usize, bool)]) -> f64 {
        let sw = self.sandwich();
        let Some(asg) = sw.assignment(input, measured) else { return 0.0 };
        let fixed = sw.fixed_for(measured.iter().map(|m| m.0));
        let eng = build_engine(&sw.ps, &fixed, &self.opts);
        eng.eval_assignment(&asg).re.max(0.0)
    }

    fn prefix_engines(&self) -> &Vec<RankEngine> {
        self.prefix_engines.get_or_init(|| {
            let sw = self.sandwich();
            (1..=self.num_qubits()).map(|l| build_engine(&sw.ps, &sw.fixed_for(0..l), &self.opts)).collect()
        })
    }

    /// Marginal of the first `y.len()` qubits through the cached prefix engines.
    fn prefix_marginal(&self, input: &[bool], y: &[bool]) -> f64 {
        if y.is_empty() {
            return 1.0;
        }
        let sw = self.sandwich();
        let measured: Vec<(usize, bool)> = y.iter().cloned().enumerate().collect();
        let Some(asg) = sw.assignment(input, &measured) else { return 0.0 };
        self.prefix_engines()[y.len() - 1].eval_assignment(&asg).re.max(0.0)
    }

    /// Draw one measurement outcome of all qubits in the computational basis without
    /// computing the statevector (chain rule over qubit prefixes).
    pub fn sample(&self, input: &[bool], rng: &mut Rng) -> Vec<bool> {
        let n = self.num_qubits();
        let mut y: Vec<bool> = Vec::with_capacity(n);
        let mut p_prev = 1.0f64;
        for _ in 0..n {
            y.push(false);
            let p0 = self.prefix_marginal(input, &y).min(p_prev);
            let one = rng.next_f64() * p_prev >= p0;
            if one {
                *y.last_mut().unwrap() = true;
                p_prev = (p_prev - p0).max(0.0);
            } else {
                p_prev = p0;
            }
        }
        y
    }

    /// `shots` samples, returned as basis-state indices.
    pub fn sample_many(&self, input: &[bool], shots: usize, seed: u64) -> Vec<u64> {
        let mut rng = Rng::new(seed);
        // warm the caches before going wide
        let _ = self.prefix_engines();
        let seeds: Vec<u64> = (0..shots).map(|_| rng.next_u64()).collect();
        seeds
            .into_par_iter()
            .map(|s| {
                let mut r = Rng::new(s);
                self.sample(input, &mut r).iter().enumerate().fold(0u64, |a, (q, &b)| a | ((b as u64) << q))
            })
            .collect()
    }
}

/// Result of [`equivalence`].
#[derive(Copy, Clone, Debug)]
pub struct Equivalence {
    /// `tr(C₂† C₁) / 2ⁿ` (modulus 1 iff the circuits agree up to a global phase).
    pub overlap: Complex64,
    /// `|overlap| ≥ 1 − tol`.
    pub equivalent: bool,
    /// The relative global phase `arg(overlap)` if equivalent (`C₁ = e^{iφ} C₂`).
    pub global_phase: Option<f64>,
}

/// Exact equivalence check up to global phase: `C₁ ≡ C₂ ⇔ |tr(C₂†C₁)| = 2ⁿ`.
///
/// Both circuits are padded to the larger qubit count.  `tol` is the allowed deficit of
/// `|overlap|` (e.g. `1e-9`).
pub fn equivalence(c1: &Circuit, c2: &Circuit, tol: f64, opts: SimOptions) -> Equivalence {
    let n = c1.num_qubits.max(c2.num_qubits);
    let mut miter = Circuit::new(n);
    miter.extend(c1);
    miter.extend(&c2.inverse());
    let mut ps = PathSum::from_circuit(&miter);
    ps.trace_closure();
    let eng = build_engine(&ps, &[], &opts);
    let tr = eng.eval_assignment(&[]);
    let overlap = tr / 2f64.powi(n as i32);
    let equivalent = overlap.norm() >= 1.0 - tol;
    Equivalence { overlap, equivalent, global_phase: equivalent.then(|| overlap.arg()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense;
    use crate::rng::{random_clifford_t, random_layered};

    fn close(a: &[Complex64], b: &[Complex64], eps: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).norm() < eps, "index {i}: {x} vs {y}");
        }
    }

    #[test]
    fn statevector_matches_dense_all_strategies() {
        let mut rng = Rng::new(99);
        for trial in 0..50 {
            let n = 1 + rng.below(5);
            let c = if trial % 2 == 0 { random_clifford_t(n, 25, rng.below(8), trial % 4 == 0, &mut rng) } else { random_layered(n, 2 + rng.below(2), &mut rng) };
            let input: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
            let want = dense::statevector(&c, &input);
            for strat in [Strategy::Cached, Strategy::Auto, Strategy::Pairs(2), Strategy::Pairs(100)] {
                let sim = Simulator::with_options(&c, SimOptions { strategy: strat, ..Default::default() });
                close(&sim.statevector(&input), &want, 1e-8);
            }
        }
    }

    #[test]
    fn toffoli_circuit_exact() {
        let mut c = Circuit::new(4);
        c.h(0).h(1).ccx(0, 1, 2).cx(2, 3).ccz(0, 2, 3).h(3);
        let input = [false; 4];
        let sim = Simulator::new(&c);
        close(&sim.statevector(&input), &dense::statevector(&c, &input), 1e-9);
    }

    #[test]
    fn marginals_match_dense() {
        let mut rng = Rng::new(3);
        for _ in 0..12 {
            let n = 2 + rng.below(3);
            let c = random_layered(n, 2, &mut rng);
            let input = vec![false; n];
            let sv = dense::statevector(&c, &input);
            let sim = Simulator::new(&c);
            for _ in 0..4 {
                let k = 1 + rng.below(n);
                let mut qs: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    qs.swap(i, rng.below(i + 1));
                }
                let measured: Vec<(usize, bool)> = qs[..k].iter().map(|&q| (q, rng.next_f64() < 0.5)).collect();
                let want: f64 = sv
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| measured.iter().all(|&(q, b)| ((i >> q) & 1 == 1) == b))
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                let got = sim.marginal_probability(&input, &measured);
                assert!((got - want).abs() < 1e-9, "{got} vs {want} for {measured:?}");
            }
        }
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let mut c = Circuit::new(3);
        c.h(0).t(0).h(0).cx(0, 1).h(2).t(2).h(2).cz(1, 2).h(1).t(1).h(1);
        let input = vec![false; 3];
        let sv = dense::statevector(&c, &input);
        let sim = Simulator::new(&c);
        let shots = 40_000;
        let samples = sim.sample_many(&input, shots, 1234);
        let mut hist = vec![0usize; 8];
        for s in samples {
            hist[s as usize] += 1;
        }
        for y in 0..8 {
            let p = sv[y].norm_sqr();
            let f = hist[y] as f64 / shots as f64;
            assert!((p - f).abs() < 0.012, "y={y}: expected {p}, sampled {f}");
        }
    }

    #[test]
    fn equivalence_checks() {
        // HZH = X, with a global phase difference carried by gphase
        let mut a = Circuit::new(1);
        a.h(0).z(0).h(0);
        let mut b = Circuit::new(1);
        b.x(0);
        let r = equivalence(&a, &b, 1e-9, SimOptions::default());
        assert!(r.equivalent, "{:?}", r);
        assert!(r.global_phase.unwrap().abs() < 1e-9);
        // T T = S, global phase 0;  Y = i X Z => relative phase check
        let mut t2 = Circuit::new(2);
        t2.t(0).t(0).cx(0, 1);
        let mut s1 = Circuit::new(2);
        s1.s(0).cx(0, 1);
        assert!(equivalence(&t2, &s1, 1e-9, SimOptions::default()).equivalent);
        // not equivalent
        let mut t1 = Circuit::new(2);
        t1.t(0).cx(0, 1);
        assert!(!equivalence(&t1, &s1, 1e-9, SimOptions::default()).equivalent);
        // global phase detection: Y vs X Z up to i
        let mut y = Circuit::new(1);
        y.y(0);
        let mut xz = Circuit::new(1);
        xz.z(0).x(0);
        let r = equivalence(&y, &xz, 1e-9, SimOptions::default());
        assert!(r.equivalent);
        assert!((r.global_phase.unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-9, "{:?}", r);
        // Toffoli decompositions agree: CCX vs H CCZ H
        let mut c1 = Circuit::new(3);
        c1.ccx(0, 1, 2);
        let mut c2 = Circuit::new(3);
        c2.h(2).ccz(0, 1, 2).h(2);
        assert!(equivalence(&c1, &c2, 1e-9, SimOptions::default()).equivalent);
        let mut c3 = Circuit::new(3);
        c3.ccx(0, 1, 2).t(0);
        assert!(!equivalence(&c1, &c3, 1e-9, SimOptions::default()).equivalent);
    }

    #[test]
    fn equivalence_random_against_dense() {
        let mut rng = Rng::new(17);
        for _ in 0..25 {
            let n = 1 + rng.below(3);
            let c = random_clifford_t(n, 12, rng.below(4), true, &mut rng);
            let mut same = c.clone();
            same.gphase(crate::phase::Phase(0x1234_5678));
            assert!(equivalence(&c, &same, 1e-9, SimOptions::default()).equivalent);
            let mut diff = c.clone();
            diff.t(rng.below(n));
            // compare with the dense definition of the overlap
            let (mut tr, dim) = (Complex64::new(0.0, 0.0), 1usize << n);
            for x in 0..dim {
                let xb = bits_of(n, x as u64);
                let a = dense::statevector(&diff, &xb);
                let b = dense::statevector(&c, &xb);
                tr += b.iter().zip(&a).map(|(p, q)| p.conj() * q).sum::<Complex64>() * 0.0 + b[x].conj() * a[x];
                // <x|C† D|x> needs the full inner product with the right basis state, so recompute:
                let _ = a;
            }
            let _ = tr;
            let r = equivalence(&c, &diff, 1e-9, SimOptions::default());
            // overlap = tr(D† C)/2^n; check against dense matrices
            let mut tr_dense = Complex64::new(0.0, 0.0);
            for x in 0..dim {
                let xb = bits_of(n, x as u64);
                let cx = dense::statevector(&c, &xb);
                let dx = dense::statevector(&diff, &xb);
                tr_dense += dx.iter().zip(&cx).map(|(d, c)| d.conj() * c).sum::<Complex64>();
            }
            assert!((tr_dense / dim as f64 - r.overlap).norm() < 1e-8, "{} vs {}", tr_dense / dim as f64, r.overlap);
        }
    }
}
