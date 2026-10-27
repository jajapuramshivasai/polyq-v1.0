//! Circuit → phase polynomial ("path sum") compiler.
//!
//! A Clifford+T circuit `C` on `n` qubits has amplitudes
//!
//! ```text
//!   <y| C |x>  =  2^{-h/2} · e^{iφ} · Σ_{z ∈ F2^V}  i^{ Q(z) } · e^{ i Σ_k θ_k z_k }
//! ```
//!
//! over `V = n + h` Boolean *path variables* (`h` = number of Hadamards), where
//!
//! * `Q(z) = Σ_k v_k z_k + 2 Σ_{j<k} B_{jk} z_j z_k  (mod 4)` is the **Z4
//!   quadratic form** (Clifford part): `B` is a symmetric zero-diagonal F2 matrix
//!   and `v ∈ Z4^V`;
//! * `θ_k` are single-variable **non-Clifford rotations** (`T`, `T†`, `P(θ)`);
//! * `z_0..z_{n-1}` are the input variables (fixed to `x`), and the *last*
//!   variable of every wire (`outputs[q]`) is fixed to `y_q ⊕ out_neg[q]`.
//!
//! ## Exact tricks used by the compiler
//!
//! * **Free X.** `X` never creates a variable: each wire carries a negation flag
//!   (`wire value = z ⊕ neg`).  All later gates are rewritten using
//!   `z⊕1 = 1−z`, which only produces *Z-type* (even) linear terms, constants and
//!   rotation sign flips.  Hence Pauli faults never change `B` nor the parity of
//!   `v`, which is what makes the Pauli-noise Monte Carlo cheap (see
//!   [`crate::noise`]).
//! * **Free SWAP.** wires are relabelled.
//! * **Rotation merging.** all `T`/`P` gates acting on the same path variable are
//!   summed into one `θ_k`; if the sum is a multiple of `π/2` it is folded into
//!   the Z4 form (this is exactly phase-polynomial T-count merging).
//! * `CX = H·CZ·H`.

use crate::bits::BitMat;
use crate::circuit::{Circuit, Gate};
use crate::phase::{z4_phase, Phase};

/// Single-qubit Pauli used for fault injection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pauli {
    X,
    Y,
    Z,
}

/// A Pauli fault inserted *before* `gates[position]` (`position == gates.len()`
/// means after the last gate).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    /// Insert before this gate index.
    pub position: usize,
    /// Faulty qubit.
    pub qubit: usize,
    /// Pauli error.
    pub pauli: Pauli,
}

/// A compiled phase polynomial.  See the module docs for the meaning of fields.
#[derive(Clone, Debug)]
pub struct PathSum {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Number of path variables (including dead ones).
    pub num_vars: usize,
    /// Number of Hadamard factors `h` (normalisation `2^{-h/2}`).
    pub num_h: usize,
    /// Symmetric zero-diagonal adjacency of the quadratic part (`V × V`).
    pub b: BitMat,
    /// Z4 linear coefficients.
    pub v: Vec<u8>,
    /// Non-Clifford single-variable rotations `e^{iθ_k z_k}`.
    pub theta: Vec<Phase>,
    /// Global phase `e^{iφ}`.
    pub global: Phase,
    /// Eliminated variables (see [`PathSum::substitute`]); they are ignored everywhere.
    pub dead: Vec<bool>,
    /// Variable that holds the final value of each wire.
    pub outputs: Vec<usize>,
    /// Wire value `= z_{outputs[q]} ⊕ out_neg[q]`.
    pub out_neg: Vec<bool>,
    /// `true` if the sum is known to vanish identically (e.g. trace of a bare `X`).
    pub zero: bool,
}

/// Incremental circuit compiler (also used by the streaming simulator).
#[derive(Clone, Debug)]
pub struct Compiler {
    n: usize,
    wire: Vec<usize>,
    neg: Vec<bool>,
    b: BitMat,
    v: Vec<u8>,
    theta: Vec<Phase>,
    nvars: usize,
    num_h: usize,
    global: Phase,
}

impl Compiler {
    /// Start a compiler for `n` qubits; `extra_capacity` is a hint for the number of
    /// Hadamard-created variables (the matrix grows geometrically if exceeded).
    pub fn new(n: usize, extra_capacity: usize) -> Self {
        let cap = n + extra_capacity;
        Self {
            n,
            wire: (0..n).collect(),
            neg: vec![false; n],
            b: BitMat::new(cap, cap),
            v: vec![0; n],
            theta: vec![Phase::ZERO; n],
            nvars: n,
            num_h: 0,
            global: Phase::ZERO,
        }
    }

    /// Current variable of each wire and its negation flag.
    pub fn wires(&self) -> (Vec<usize>, Vec<bool>) {
        (self.wire.clone(), self.neg.clone())
    }

    fn new_var(&mut self) -> usize {
        if self.nvars == self.b.nrows() {
            let newcap = (self.b.nrows() * 2).max(8);
            self.b.grow_square(newcap);
        }
        self.v.push(0);
        self.theta.push(Phase::ZERO);
        self.nvars += 1;
        self.nvars - 1
    }

    fn add_v(&mut self, z: usize, k: u8) {
        self.v[z] = (self.v[z] + k) & 3;
    }

    /// Apply one gate.  Wires beyond `n` are a logic error and panic.
    pub fn apply(&mut self, g: &Gate) {
        match *g {
            Gate::H(q) => {
                let z = self.wire[q];
                let zp = self.new_var();
                self.b.flip_sym(z, zp);
                self.num_h += 1;
                if self.neg[q] {
                    // (-1)^{(z⊕1) z'} = (-1)^{z z'} (-1)^{z'}
                    self.add_v(zp, 2);
                    self.neg[q] = false;
                }
                self.wire[q] = zp;
            }
            Gate::X(q) => self.neg[q] = !self.neg[q],
            Gate::Z(q) => {
                let z = self.wire[q];
                self.add_v(z, 2);
                if self.neg[q] {
                    self.global += Phase::PI;
                }
            }
            Gate::S(q) => {
                let z = self.wire[q];
                if self.neg[q] {
                    // i^{1-z} = i · i^{-z}
                    self.add_v(z, 3);
                    self.global += Phase::HALF_PI;
                } else {
                    self.add_v(z, 1);
                }
            }
            Gate::Sdg(q) => {
                let z = self.wire[q];
                if self.neg[q] {
                    // (-i)^{1-z} = (-i) · i^{z}
                    self.add_v(z, 1);
                    self.global -= Phase::HALF_PI;
                } else {
                    self.add_v(z, 3);
                }
            }
            Gate::T(q) => self.rotate(q, Phase::QUARTER_PI),
            Gate::Tdg(q) => self.rotate(q, -Phase::QUARTER_PI),
            Gate::P(q, th) => self.rotate(q, th),
            Gate::Y(q) => {
                // Y = i · X · Z
                self.apply(&Gate::Z(q));
                self.apply(&Gate::X(q));
                self.global += Phase::HALF_PI;
            }
            Gate::CZ(a, b) => {
                let (za, zb) = (self.wire[a], self.wire[b]);
                debug_assert_ne!(za, zb);
                self.b.flip_sym(za, zb);
                // (-1)^{(za⊕na)(zb⊕nb)} = (-1)^{za zb + nb za + na zb + na nb}
                if self.neg[b] {
                    self.add_v(za, 2);
                }
                if self.neg[a] {
                    self.add_v(zb, 2);
                }
                if self.neg[a] && self.neg[b] {
                    self.global += Phase::PI;
                }
            }
            Gate::CX(c, t) => {
                self.apply(&Gate::H(t));
                self.apply(&Gate::CZ(c, t));
                self.apply(&Gate::H(t));
            }
            Gate::Swap(a, b) => {
                self.wire.swap(a, b);
                self.neg.swap(a, b);
            }
            Gate::GPhase(p) => self.global += p,
        }
    }

    fn rotate(&mut self, q: usize, th: Phase) {
        let z = self.wire[q];
        if self.neg[q] {
            // e^{iθ(1-z)} = e^{iθ} e^{-iθ z}
            self.theta[z] -= th;
            self.global += th;
        } else {
            self.theta[z] += th;
        }
    }

    /// Apply a Pauli fault on `q`.
    pub fn apply_pauli(&mut self, q: usize, p: Pauli) {
        match p {
            Pauli::X => self.apply(&Gate::X(q)),
            Pauli::Y => self.apply(&Gate::Y(q)),
            Pauli::Z => self.apply(&Gate::Z(q)),
        }
    }

    /// Produce a [`PathSum`] for the gates applied so far (the compiler stays usable).
    pub fn snapshot(&self) -> PathSum {
        let nv = self.nvars;
        let mut b = BitMat::new(nv, nv);
        for i in 0..nv {
            for j in self.b.row(i).iter_ones() {
                if j < nv {
                    b.set(i, j, true);
                }
            }
        }
        let mut ps = PathSum {
            num_qubits: self.n,
            num_vars: nv,
            num_h: self.num_h,
            b,
            v: self.v.clone(),
            theta: self.theta.clone(),
            global: self.global,
            dead: vec![false; nv],
            outputs: self.wire.clone(),
            out_neg: self.neg.clone(),
            zero: false,
        };
        ps.normalize();
        ps
    }

    /// Consume the compiler.
    pub fn finish(self) -> PathSum {
        self.snapshot()
    }
}

/// `(path sum of C† Π C, junction variable of each wire, junction negation flag)`.
pub type Sandwich = (PathSum, Vec<usize>, Vec<bool>);

impl PathSum {
    /// Compile a circuit.
    pub fn from_circuit(c: &Circuit) -> PathSum {
        Self::from_circuit_faulted(c, &[])
    }

    /// Compile a circuit with Pauli faults inserted (faults are applied in the order
    /// of `position`, ties in slice order).
    pub fn from_circuit_faulted(c: &Circuit, faults: &[Fault]) -> PathSum {
        let mut comp = Compiler::new(c.num_qubits, c.h_count());
        let mut fs: Vec<&Fault> = faults.iter().collect();
        fs.sort_by_key(|f| f.position);
        let mut fi = 0;
        for (pos, g) in c.gates.iter().enumerate() {
            while fi < fs.len() && fs[fi].position <= pos {
                comp.apply_pauli(fs[fi].qubit, fs[fi].pauli);
                fi += 1;
            }
            comp.apply(g);
        }
        while fi < fs.len() {
            comp.apply_pauli(fs[fi].qubit, fs[fi].pauli);
            fi += 1;
        }
        comp.finish()
    }

    /// Compile `C` followed by `C†` on the same wires and report the junction
    /// variables.  Fixing the junction variable of a measured qubit `q` to the
    /// outcome `y_q ⊕ junction_neg[q]` and leaving the others free yields
    /// `<x| C† (⊗_{q∈M} |y_q><y_q|) C |x>` — the marginal probability of the outcomes
    /// on `M` — as a single path sum.
    pub fn sandwich(c: &Circuit) -> Sandwich {
        let mut comp = Compiler::new(c.num_qubits, 2 * c.h_count());
        for g in &c.gates {
            comp.apply(g);
        }
        let (junction, junction_neg) = comp.wires();
        for g in c.inverse().gates.iter() {
            comp.apply(g);
        }
        (comp.finish(), junction, junction_neg)
    }

    /// Fold Clifford rotations into the Z4 form.
    pub fn normalize(&mut self) {
        for k in 0..self.num_vars {
            if let Some(l) = self.theta[k].clifford_level() {
                self.v[k] = (self.v[k] + l) & 3;
                self.theta[k] = Phase::ZERO;
            }
        }
    }

    /// The input variables followed by the (deduplicated) output variables: the standard
    /// set of variables fixed by `<y|C|x>`.
    pub fn io_vars(&self) -> Vec<usize> {
        let mut f: Vec<usize> = (0..self.num_qubits).collect();
        for &o in &self.outputs {
            if !f.contains(&o) {
                f.push(o);
            }
        }
        f
    }

    /// Indices of live variables.
    pub fn live_vars(&self) -> Vec<usize> {
        (0..self.num_vars).filter(|&i| !self.dead[i]).collect()
    }

    /// Number of live non-Clifford rotations (the T-count of the path sum).
    pub fn rotation_count(&self) -> usize {
        (0..self.num_vars).filter(|&k| !self.dead[k] && self.theta[k] != Phase::ZERO).count()
    }

    /// Eliminate variable `b` by imposing `z_b = z_a ⊕ c`.
    ///
    /// This is the algebraic core of the exact rank-2 decomposition of a pair of
    /// T-type rotations (see [`crate::rank`]) and of trace closure.  Derivation: after
    /// substituting `z_b = z_a + c − 2 c z_a`:
    ///
    /// * `c = 0`: `v_a += v_b + 2 B_ab`, `θ_a += θ_b`, `B_a· ⊕= B_b·`.
    /// * `c = 1`: `v_a -= v_b`, `θ_a -= θ_b`, constant `i^{v_b} e^{iθ_b}`, `B_a· ⊕= B_b·`
    ///   and every neighbour `k` of `b` gets `v_k += 2`.  (The `2 B_ab z_a z_b` term
    ///   becomes `2 B_ab z_a (1 - z_a) = 0`.)
    ///
    /// Both variables must be live and distinct.
    pub fn substitute(&mut self, a: usize, b: usize, c: bool) {
        assert!(a != b && !self.dead[a] && !self.dead[b]);
        let bab = self.b.get(a, b);
        let (ta, tb) = (self.theta[a], self.theta[b]);
        let vb = self.v[b];
        let mut nb = self.b.row(b).clone();
        nb.set(a, false);
        if !c {
            self.v[a] = (self.v[a] + vb + 2 * (bab as u8)) & 3;
            self.theta[a] = ta + tb;
        } else {
            self.v[a] = (self.v[a] + 4 - vb) & 3;
            self.theta[a] = ta - tb;
            self.global += z4_phase(vb) + tb;
            for k in nb.iter_ones() {
                self.v[k] = (self.v[k] + 2) & 3;
            }
        }
        for k in nb.iter_ones() {
            self.b.flip_sym(a, k);
        }
        // detach b
        let nbrs = self.b.row(b).ones();
        for k in nbrs {
            self.b.row_mut(k).set(b, false);
        }
        self.b.row_mut(b).clear();
        self.v[b] = 0;
        self.theta[b] = Phase::ZERO;
        self.dead[b] = true;
        if let Some(l) = self.theta[a].clifford_level() {
            self.v[a] = (self.v[a] + l) & 3;
            self.theta[a] = Phase::ZERO;
        }
    }

    /// Close the circuit into a trace: identify the output variable of every wire with
    /// its input variable and leave **all** variables free, so that evaluating the
    /// (unfixed) sum gives `tr(U)`.
    pub fn trace_closure(&mut self) {
        let nv = self.num_vars;
        let mut rep: Vec<usize> = (0..nv).collect();
        let mut par = vec![false; nv];
        fn find(rep: &[usize], par: &[bool], mut x: usize) -> (usize, bool) {
            let mut p = false;
            while rep[x] != x {
                p ^= par[x];
                x = rep[x];
            }
            (x, p)
        }
        for q in 0..self.num_qubits {
            let (ra, pa) = find(&rep, &par, q);
            let (rb, pb) = find(&rep, &par, self.outputs[q]);
            let c = pa ^ pb ^ self.out_neg[q];
            if ra == rb {
                if c {
                    self.zero = true;
                }
                continue;
            }
            self.substitute(ra, rb, c);
            rep[rb] = ra;
            par[rb] = c;
        }
    }

    /// Boundary conditions for the amplitude `<output|C|input>`: pairs `(variable, value)`
    /// for every input and output variable, or `None` if two conditions contradict (the
    /// amplitude is then exactly 0).
    pub fn boundary(&self, input: &[bool], output: &[bool]) -> Option<Vec<(usize, bool)>> {
        let mut map: Vec<(usize, bool)> = Vec::with_capacity(2 * self.num_qubits);
        let mut put = |var: usize, val: bool| -> bool {
            if let Some(e) = map.iter().find(|e| e.0 == var) {
                return e.1 == val;
            }
            map.push((var, val));
            true
        };
        for q in 0..self.num_qubits {
            if !put(q, input[q]) {
                return None;
            }
        }
        for q in 0..self.num_qubits {
            if !put(self.outputs[q], output[q] ^ self.out_neg[q]) {
                return None;
            }
        }
        Some(map)
    }

    /// **Exponential reference evaluator** (tests only): sum the path polynomial over all
    /// free variables by brute force.  `fixed` gives `(var, value)` pairs.
    pub fn brute_force(&self, fixed: &[(usize, bool)]) -> num_complex::Complex64 {
        use num_complex::Complex64;
        if self.zero {
            return Complex64::new(0.0, 0.0);
        }
        let mut val = vec![false; self.num_vars];
        let mut is_fixed = vec![false; self.num_vars];
        for &(v, x) in fixed {
            val[v] = x;
            is_fixed[v] = true;
        }
        let free: Vec<usize> = (0..self.num_vars).filter(|&i| !self.dead[i] && !is_fixed[i]).collect();
        assert!(free.len() <= 26, "brute force over {} variables", free.len());
        let live = self.live_vars();
        let mut total = Complex64::new(0.0, 0.0);
        for mask in 0u64..(1u64 << free.len()) {
            for (bi, &f) in free.iter().enumerate() {
                val[f] = (mask >> bi) & 1 == 1;
            }
            let mut z4: u32 = 0;
            let mut th = Phase::ZERO;
            for &j in &live {
                if val[j] {
                    z4 += self.v[j] as u32;
                    th += self.theta[j];
                    for k in self.b.row(j).iter_ones() {
                        if k > j && val[k] {
                            z4 += 2;
                        }
                    }
                }
            }
            total += (z4_phase((z4 & 3) as u8) + th).to_complex();
        }
        total * self.global.to_complex() * 2f64.powf(-(self.num_h as f64) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense;
    use crate::rng::{random_clifford_t, Rng};
    use num_complex::Complex64;

    fn bits(n: usize, idx: usize) -> Vec<bool> {
        (0..n).map(|q| (idx >> q) & 1 == 1).collect()
    }

    /// Path-sum brute force must agree with the dense simulator for every gate type.
    #[test]
    fn pathsum_matches_dense_random() {
        let mut rng = Rng::new(7);
        let mut checked = 0;
        for trial in 0..60 {
            let n = 1 + rng.below(3);
            let c = random_clifford_t(n, 6 + rng.below(8), rng.below(4), trial % 2 == 0, &mut rng);
            let ps = PathSum::from_circuit(&c);
            if ps.live_vars().len() > 22 {
                continue;
            }
            checked += 1;
            for xin in 0..(1usize << n) {
                let x = bits(n, xin);
                let sv = dense::statevector(&c, &x);
                for y in 0..(1usize << n) {
                    let yb = bits(n, y);
                    let amp = match ps.boundary(&x, &yb) {
                        None => Complex64::new(0.0, 0.0),
                        Some(bd) => ps.brute_force(&bd),
                    };
                    assert!((amp - sv[y]).norm() < 1e-9, "trial {trial} n={n} x={xin} y={y}: {amp} vs {}", sv[y]);
                }
            }
        }
        assert!(checked >= 40, "only {checked} circuits checked");
    }

    #[test]
    fn substitution_is_exact() {
        // sum over (a,b) of e^{iθa a + iθb b} f(a,b) == branch c=0 + branch c=1
        let mut rng = Rng::new(11);
        let mut checked = 0;
        for _ in 0..80 {
            let n = 3;
            let c = random_clifford_t(n, 14, 3, true, &mut rng);
            let ps = PathSum::from_circuit(&c);
            let live: Vec<usize> = ps.live_vars().into_iter().filter(|&i| i >= n && !ps.outputs.contains(&i)).collect();
            if live.len() < 2 || live.len() > 18 {
                continue;
            }
            let x = vec![false; n];
            let y = vec![false; n];
            let Some(bd) = ps.boundary(&x, &y) else { continue };
            let full = ps.brute_force(&bd);
            let (a, b) = (live[0], live[1]);
            // Decomposition: e^{iθa za + iθb zb} = [za=zb] e^{i(θa+θb)za} + [za≠zb] e^{iθb} e^{i(θa-θb)za}
            // which is only an identity of the *rotation* factor, so move both rotations into
            // the substituted branches and strip them from the base polynomial.
            let mut base = ps.clone();
            base.theta[a] = Phase::ZERO;
            base.theta[b] = Phase::ZERO;
            let mut s0 = base.clone();
            s0.theta[a] = ps.theta[a];
            s0.theta[b] = ps.theta[b];
            s0.substitute(a, b, false);
            let mut s1 = base.clone();
            s1.theta[a] = ps.theta[a];
            s1.theta[b] = ps.theta[b];
            s1.substitute(a, b, true);
            checked += 1;
            let part = s0.brute_force(&bd) + s1.brute_force(&bd);
            // Only the *rotation* part is decomposed; the Clifford coupling between a and b stays,
            // so the full identity holds for the whole polynomial.
            assert!((part - full).norm() < 1e-9, "{part} vs {full}");
        }
        assert!(checked >= 20, "only {checked} substitutions checked");
    }
}
