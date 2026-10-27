//! Gate set and in-memory circuits.
//!
//! The supported gate set is Clifford+T extended with arbitrary dyadic
//! diagonal rotations:
//!
//! | gate | matrix / action |
//! |------|-----------------|
//! | `H`  | Hadamard |
//! | `X`, `Y`, `Z` | Paulis |
//! | `S`, `Sdg` | `diag(1, ±i)` |
//! | `T`, `Tdg` | `diag(1, e^{±iπ/4})` |
//! | `P(q, θ)` | `diag(1, e^{iθ})` with `θ` a [`Phase`] |
//! | `CZ`, `CX`, `Swap` | two-qubit Cliffords |
//! | `GPhase(θ)` | global phase `e^{iθ}` |
//!
//! Toffoli / CCZ are provided as *builders* that expand into 7 T-type gates.
//!
//! Qubit `q` is bit `q` (little endian) of every basis-state index used by the
//! public API.

use crate::phase::Phase;

/// A single gate.  See the module docs for the matrices.
#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),
    /// `diag(1, e^{iθ})`.
    P(usize, Phase),
    CZ(usize, usize),
    /// Control, target.
    CX(usize, usize),
    Swap(usize, usize),
    /// Global phase `e^{iθ}`.
    GPhase(Phase),
}

impl Gate {
    /// The inverse gate.
    pub fn inverse(&self) -> Gate {
        match *self {
            Gate::S(q) => Gate::Sdg(q),
            Gate::Sdg(q) => Gate::S(q),
            Gate::T(q) => Gate::Tdg(q),
            Gate::Tdg(q) => Gate::T(q),
            Gate::P(q, p) => Gate::P(q, -p),
            Gate::GPhase(p) => Gate::GPhase(-p),
            ref g => g.clone(), // H, X, Y, Z, CZ, CX, Swap are involutions
        }
    }

    /// Qubits touched by the gate.
    pub fn qubits(&self) -> Vec<usize> {
        match *self {
            Gate::H(q) | Gate::X(q) | Gate::Y(q) | Gate::Z(q) | Gate::S(q) | Gate::Sdg(q) | Gate::T(q) | Gate::Tdg(q) | Gate::P(q, _) => {
                vec![q]
            }
            Gate::CZ(a, b) | Gate::CX(a, b) | Gate::Swap(a, b) => vec![a, b],
            Gate::GPhase(_) => vec![],
        }
    }

    /// `true` iff the gate is *not* Clifford (a `P(θ)` with `θ` not a multiple of `π/2`).
    pub fn is_non_clifford(&self) -> bool {
        match *self {
            Gate::T(_) | Gate::Tdg(_) => true,
            Gate::P(_, p) => p.clifford_level().is_none(),
            _ => false,
        }
    }
}

/// An ordered list of gates on `num_qubits` wires.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Gates in time order.
    pub gates: Vec<Gate>,
}

macro_rules! single {
    ($(#[$m:meta])* $name:ident, $variant:ident) => {
        $(#[$m])*
        pub fn $name(&mut self, q: usize) -> &mut Self {
            self.push(Gate::$variant(q))
        }
    };
}

impl Circuit {
    /// Empty circuit on `num_qubits` wires.
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits, gates: Vec::new() }
    }

    /// Append a gate, growing `num_qubits` if the gate touches a new wire.
    pub fn push(&mut self, g: Gate) -> &mut Self {
        for q in g.qubits() {
            self.num_qubits = self.num_qubits.max(q + 1);
        }
        self.gates.push(g);
        self
    }

    single!(/// Hadamard.
        h, H);
    single!(/// Pauli X.
        x, X);
    single!(/// Pauli Y.
        y, Y);
    single!(/// Pauli Z.
        z, Z);
    single!(/// Phase gate `S`.
        s, S);
    single!(/// `S†`.
        sdg, Sdg);
    single!(/// `T` gate.
        t, T);
    single!(/// `T†`.
        tdg, Tdg);

    /// `diag(1, e^{iθ})` with an exact dyadic [`Phase`].
    pub fn p(&mut self, q: usize, theta: Phase) -> &mut Self {
        self.push(Gate::P(q, theta))
    }

    /// Qiskit-convention `RZ(θ) = diag(e^{-iθ/2}, e^{iθ/2})` for `θ` in radians
    /// (implemented as `P(θ)` plus a global phase `e^{-iθ/2}`).
    pub fn rz(&mut self, q: usize, radians: f64) -> &mut Self {
        self.push(Gate::P(q, Phase::from_radians(radians)));
        self.push(Gate::GPhase(Phase::from_radians(-radians / 2.0)))
    }

    /// Controlled-Z.
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.push(Gate::CZ(a, b))
    }

    /// Controlled-X (control `c`, target `t`).
    pub fn cx(&mut self, c: usize, t: usize) -> &mut Self {
        self.push(Gate::CX(c, t))
    }

    /// SWAP (free in the simulator: it only relabels wires).
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.push(Gate::Swap(a, b))
    }

    /// Global phase.
    pub fn gphase(&mut self, theta: Phase) -> &mut Self {
        self.push(Gate::GPhase(theta))
    }

    /// Toffoli `CCX(a, b -> c)` expanded into Clifford + 7 T/T† gates
    /// (Nielsen & Chuang, Fig. 4.9).
    pub fn ccx(&mut self, a: usize, b: usize, c: usize) -> &mut Self {
        self.h(c);
        self.cx(b, c);
        self.tdg(c);
        self.cx(a, c);
        self.t(c);
        self.cx(b, c);
        self.tdg(c);
        self.cx(a, c);
        self.t(b);
        self.t(c);
        self.h(c);
        self.cx(a, b);
        self.t(a);
        self.tdg(b);
        self.cx(a, b)
    }

    /// CCZ expanded into Clifford + 7 T/T† gates (`H(c) · CCX · H(c)`).
    pub fn ccz(&mut self, a: usize, b: usize, c: usize) -> &mut Self {
        self.h(c);
        self.ccx(a, b, c);
        self.h(c)
    }

    /// The inverse circuit (reverse order, each gate inverted).
    pub fn inverse(&self) -> Circuit {
        Circuit { num_qubits: self.num_qubits, gates: self.gates.iter().rev().map(Gate::inverse).collect() }
    }

    /// Append all gates of `other`.
    pub fn extend(&mut self, other: &Circuit) -> &mut Self {
        self.num_qubits = self.num_qubits.max(other.num_qubits);
        self.gates.extend(other.gates.iter().cloned());
        self
    }

    /// Number of non-Clifford rotations (`T`, `T†`, generic `P`).
    pub fn t_count(&self) -> usize {
        self.gates.iter().filter(|g| g.is_non_clifford()).count()
    }

    /// Number of Hadamards, counting `CX` as two (it is `H CZ H`).
    pub fn h_count(&self) -> usize {
        self.gates
            .iter()
            .map(|g| match g {
                Gate::H(_) => 1,
                Gate::CX(_, _) => 2,
                _ => 0,
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense;

    #[test]
    fn inverse_is_inverse() {
        let mut c = Circuit::new(3);
        c.h(0).t(0).cx(0, 1).s(1).tdg(2).cz(1, 2).swap(0, 2).x(1).y(0).p(2, Phase::from_pi_fraction(3, 5));
        let mut full = c.clone();
        full.extend(&c.inverse());
        let sv = dense::statevector(&full, &[false, false, false]);
        assert!((sv[0].norm() - 1.0).abs() < 1e-9, "|<0|U†U|0>| = {}", sv[0].norm());
    }

    #[test]
    fn toffoli_matches_truth_table() {
        for input in 0..8u32 {
            let bits: Vec<bool> = (0..3).map(|i| (input >> i) & 1 == 1).collect();
            let mut c = Circuit::new(3);
            c.ccx(0, 1, 2);
            let sv = dense::statevector(&c, &bits);
            let expect = if bits[0] && bits[1] { input ^ 4 } else { input } as usize;
            assert!((sv[expect].norm() - 1.0).abs() < 1e-9, "input {input}");
            assert_eq!(c.t_count(), 7);
        }
    }
}
