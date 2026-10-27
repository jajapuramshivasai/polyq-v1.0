//! # polyq 2 — phase-polynomial simulation of Clifford+T circuits
//!
//! Amplitudes of a Clifford+T circuit are exponential sums of a **phase polynomial**
//! (a *path sum*).  The Clifford part is a Z4 quadratic form whose Gauss sum has a closed
//! form once the form is brought to normal form by a congruence.  This crate organises the
//! computation around that normal form:
//!
//! | module | role |
//! |--------|------|
//! | [`bits`] | word-packed GF(2) vectors/matrices |
//! | [`phase`] | dyadic phase ring `Z_{2^32}` (angles mod 2π; fixes the v1 π-wrap bug) |
//! | [`circuit`], [`qasm`] | gate set, builders (`ccx`, `ccz`, …), OpenQASM-2 subset |
//! | [`pathsum`] | compiler: free `X`/`SWAP`, rotation merging, variable substitution, trace closure, Pauli faults |
//! | [`normal_form`] | **Dumas–Pernet-style symmetric `LDLᵀ`** over F2 with 1×1/2×2 pivots, Z4 diagonals via popcounts, closed-form Gauss sum |
//! | [`engine`] | cached normal form; boundary conditions and `T`-branches become *shifts*; **affine-subspace branch enumeration** |
//! | [`approx`] | rigorous pruning/rounding of rotations, Monte-Carlo estimators with standard error |
//! | [`clifford`] | exact weak sampling of Clifford circuits from the normal form (affine support) |
//! | [`rank`] | exact **rank-2 stabiliser decomposition of T-pairs** + cost model |
//! | [`sim`] | strong/weak simulation, marginals, sampling, **equivalence checking** |
//! | [`noise`] | **Pauli-noise Monte Carlo** reusing the cached structure |
//! | [`incremental`] | streaming simulator with structure re-use |
//! | [`dense`], [`rng`] | reference simulator and utilities for tests/benchmarks |
//!
//! ```
//! use polyq::{Circuit, Simulator};
//! let mut c = Circuit::new(2);
//! c.h(0).cx(0, 1).t(1).h(1);
//! let sim = Simulator::new(&c);
//! let a = sim.amplitude(&[false, false], &[true, true]);
//! assert!((a.norm_sqr() - sim.probability(&[false, false], &[true, true])).abs() < 1e-12);
//! ```

pub mod approx;
pub mod bits;
pub mod circuit;
pub mod clifford;
pub mod dense;
pub mod engine;
pub mod incremental;
pub mod noise;
pub mod normal_form;
pub mod pathsum;
pub mod phase;
pub mod qasm;
pub mod rank;
pub mod rng;
pub mod sim;

pub use circuit::{Circuit, Gate};
pub use incremental::IncrementalSimulator;
pub use noise::NoisyEstimator;
pub use pathsum::{Fault, Pauli, PathSum};
pub use phase::Phase;
pub use sim::{equivalence, SimOptions, Simulator, Strategy};

#[cfg(feature = "python")]
mod pybind;
