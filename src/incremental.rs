//! Streaming simulator: append gates one at a time and query between appends.
//!
//! The compiler is incremental (`O(1)` per gate).  After each append the engine is
//! **re-bound** instead of rebuilt whenever the Gram structure survived:
//!
//! | appended gate | effect on the polynomial | cost of next query |
//! |---------------|--------------------------|--------------------|
//! | `T`, `T†`, `P(θ)`, `Z`, `X`, `Y`, `SWAP`, global phase | values / rotation set only | `O(m·t)` (new branch RREF), no elimination |
//! | `S`, `S†`, `CZ`, `H`, `CX` | changes the diagonal parity / edges / bordering `B` | full rebuild |
//!
//! A genuine rank-1/rank-2 update of the `LDLᵀ` basis for the second row is possible (Schur
//! complement updates of the Gram matrix) but is not implemented; the counters in
//! [`IncrementalStats`] show how often the cheap path was taken.

use crate::circuit::Gate;
use crate::engine::{Engine, EngineOptions};
use crate::pathsum::{Compiler, PathSum};
use num_complex::Complex64;

/// How often queries reused the cached normal form.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct IncrementalStats {
    /// Re-bindings (no elimination).
    pub rebinds: usize,
    /// Full engine builds.
    pub rebuilds: usize,
}

/// Append-and-query simulator.
pub struct IncrementalSimulator {
    n: usize,
    compiler: Compiler,
    cache: Option<(PathSum, Engine)>,
    dirty: bool,
    opts: EngineOptions,
    /// Reuse counters.
    pub stats: IncrementalStats,
}

impl IncrementalSimulator {
    /// Empty circuit on `n` qubits.
    pub fn new(n: usize) -> Self {
        Self { n, compiler: Compiler::new(n, 16), cache: None, dirty: true, opts: EngineOptions::default(), stats: IncrementalStats::default() }
    }

    /// Append a gate.
    pub fn push(&mut self, g: Gate) -> &mut Self {
        self.compiler.apply(&g);
        self.dirty = true;
        self
    }

    fn sync(&mut self) {
        if !self.dirty && self.cache.is_some() {
            return;
        }
        let ps = self.compiler.snapshot();
        let fixed = ps.io_vars();
        let reused = self.cache.as_ref().and_then(|(_, e)| if e.fixed() == fixed.as_slice() { e.rebind(&ps) } else { None });
        let eng = match reused {
            Some(e) => {
                self.stats.rebinds += 1;
                e
            }
            None => {
                self.stats.rebuilds += 1;
                Engine::new(&ps, &fixed, self.opts)
            }
        };
        self.cache = Some((ps, eng));
        self.dirty = false;
    }

    /// `<output| C |input>` for the gates appended so far.
    pub fn amplitude(&mut self, input: &[bool], output: &[bool]) -> Complex64 {
        self.sync();
        let (ps, eng) = self.cache.as_ref().unwrap();
        match ps.boundary(input, output) {
            None => Complex64::new(0.0, 0.0),
            Some(bd) => eng.eval_assignment(&bd),
        }
    }

    /// Full statevector for the gates appended so far.
    pub fn statevector(&mut self, input: &[bool]) -> Vec<Complex64> {
        let n = self.n;
        (0..(1u64 << n)).map(|y| self.amplitude(input, &crate::sim::bits_of(n, y))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::dense;
    use crate::rng::{random_clifford_t, Rng};

    #[test]
    fn streaming_matches_dense_at_every_prefix() {
        let mut rng = Rng::new(31);
        for _ in 0..15 {
            let n = 1 + rng.below(4);
            let c = random_clifford_t(n, 20, 5, true, &mut rng);
            let mut sim = IncrementalSimulator::new(n);
            let mut prefix = Circuit::new(n);
            let input = vec![false; n];
            for g in &c.gates {
                sim.push(g.clone());
                prefix.push(g.clone());
                let want = dense::statevector(&prefix, &input);
                let got = sim.statevector(&input);
                for (a, b) in got.iter().zip(&want) {
                    assert!((a - b).norm() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn diagonal_appends_reuse_structure() {
        let mut sim = IncrementalSimulator::new(3);
        for q in 0..3 {
            sim.push(Gate::H(q));
        }
        sim.push(Gate::CZ(0, 1)).push(Gate::CZ(1, 2));
        let z = [false; 3];
        let _ = sim.amplitude(&z, &z);
        assert_eq!(sim.stats.rebuilds, 1);
        for q in 0..3 {
            sim.push(Gate::T(q));
            let _ = sim.amplitude(&z, &z);
            sim.push(Gate::Z(q));
            let _ = sim.amplitude(&z, &z);
        }
        assert_eq!(sim.stats.rebuilds, 1, "diagonal non-Clifford/Pauli appends must not re-eliminate");
        assert_eq!(sim.stats.rebinds, 6);
    }
}
