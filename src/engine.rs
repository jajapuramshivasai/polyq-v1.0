//! The cached evaluation engine: normal form + affine-subspace branch summation.
//!
//! An [`Engine`] is built once from a [`PathSum`] and a set of **fixed** variables
//! (inputs/outputs, measured junctions, ...).  It then evaluates the fixed-variable
//! assignments in `O(|fixed|·m/64 + 2^d · m/64)` time without ever re-running the
//! elimination.
//!
//! ## Why the elimination can be cached
//!
//! Fix the set `F` of boundary variables; the remaining *free* variables carry the
//! form `Q_free` whose matrix `B_ff` and linear parity do **not** depend on the values
//! assigned to `F`.  The values only produce
//!
//! * a constant `i^{...}` (from `F`–`F` terms), and
//! * a **Z-type shift** `(-1)^{b·z}` with `b = B_{free,F} x_F`.
//!
//! A shift is a vector `u = Pᵀb` in the normal-form basis ([`Structure::transform`]),
//! `u = ⊕_{f: x_f=1} ucol_f`, so evaluating a new boundary condition is a few word XORs.
//!
//! ## Z-decomposition of the non-Clifford part (the affine-subspace trick)
//!
//! Every remaining free rotation is split as
//!
//! ```text
//!   e^{iθ z} = α + β (-1)^z ,     α = (1+e^{iθ})/2 ,  β = (1-e^{iθ})/2 .
//! ```
//!
//! Branch `s ∈ F2^t` therefore multiplies the Clifford sum by `∏_k (s_k ? β_k : α_k)` and
//! adds the *linear Z-type* term `(-1)^{Σ s_k z_{v_k}}`.  The matrix `B` never changes
//! between branches, so the normal form is shared by **all** `2^t` branches, and branch `s`
//! is the shift `u(s) = u_0 ⊕ M s` where `M` has columns `mcol_k = Pᵀ e_{v_k}`.
//!
//! Radical (even) basis vectors force `(u(s))_e = w_e`, i.e. the **linear system**
//! `K s = κ` with `K = M|_even`.  Only the affine subspace `s ∈ s_0 + ker K` of
//! dimension `d = t − rank K` contributes; we precompute an RREF of `K` once, and each
//! evaluation only solves for a particular `s_0` and Gray-code-enumerates `ker K` with
//! `O(m/64)` work per step.  Worst case is still `2^t` (the problem is #P-hard), but in
//! practice many branches die in bulk.

use crate::bits::{BitMat, BitVec};
use crate::normal_form::{Diag, PivotRule, Structure, UVec};
use crate::pathsum::PathSum;
use crate::phase::{z4_phase, Phase, ROOT8};
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::Arc;

/// Tuning knobs.
#[derive(Copy, Clone, Debug)]
pub struct EngineOptions {
    /// Pivot rule of the symmetric elimination.
    pub pivot: PivotRule,
    /// Enumerate the branch subspace in parallel once its dimension is at least this big.
    pub parallel_min_dim: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self { pivot: PivotRule::MinWeight, parallel_min_dim: 16 }
    }
}

/// Size information about a built engine.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EngineStats {
    /// Free (summed) variables.
    pub free_vars: usize,
    /// Hyperbolic planes in the normal form.
    pub planes: usize,
    /// Non-isotropic lines.
    pub lines: usize,
    /// Radical vectors.
    pub radical: usize,
    /// Non-Clifford branch terms `t`.
    pub terms: usize,
    /// Dimension `d = t − rank K` of the surviving affine branch subspace
    /// (the work per amplitude is `2^d`).
    pub branch_dim: usize,
}

/// Per-thread work buffers (see [`Engine::scratch`]).
pub struct Scratch {
    u0: UVec,
    u: UVec,
    kappa: BitVec,
    s0: BitVec,
    #[allow(dead_code)]
    sv: BitVec,
    ones: Vec<usize>,
}

/// An affine subspace `origin + span(basis)` of `F2^k`.
#[derive(Clone, Debug)]
pub struct AffineSpace {
    /// A point of the subspace.
    pub origin: BitVec,
    /// Basis of the direction space.
    pub basis: Vec<BitVec>,
}

const NONE: usize = usize::MAX;
const LOW_BITS: usize = 14;
const RECOMPUTE_EVERY: usize = 2048;

/// RREF of the constraint matrix `K` (radical rows × branch columns).
#[derive(Clone, Debug)]
struct Elim {
    t: usize,
    /// `(pivot column, R-row over radical indices)`.
    pivots: Vec<(usize, BitVec)>,
    /// Rows with zero `K` part: `<r, κ>` must vanish for a solution to exist.
    consistency: Vec<BitVec>,
    kernel: Vec<BitVec>,
    kernel_supp: Vec<Vec<usize>>,
    kernel_u: Vec<UVec>,
}

impl Elim {
    fn new(structure: &Structure, mcols: &[UVec]) -> Elim {
        let t = mcols.len();
        let ne = structure.even.len();
        // rows: (K row over t, R row over ne)
        let mut rows: Vec<(BitVec, BitVec)> = (0..ne)
            .map(|e| {
                let mut k = BitVec::new(t);
                for (c, m) in mcols.iter().enumerate() {
                    if m.e.get(e) {
                        k.set(c, true);
                    }
                }
                (k, BitVec::from_ones(ne, [e]))
            })
            .collect();
        let mut pivot_cols: Vec<usize> = Vec::new();
        let mut rank = 0;
        for col in 0..t {
            let Some(r) = (rank..rows.len()).find(|&r| rows[r].0.get(col)) else { continue };
            rows.swap(rank, r);
            let (pk, pr) = (rows[rank].0.clone(), rows[rank].1.clone());
            for r2 in 0..rows.len() {
                if r2 != rank && rows[r2].0.get(col) {
                    rows[r2].0.xor_assign(&pk);
                    rows[r2].1.xor_assign(&pr);
                }
            }
            pivot_cols.push(col);
            rank += 1;
        }
        let pivots: Vec<(usize, BitVec)> = (0..rank).map(|i| (pivot_cols[i], rows[i].1.clone())).collect();
        let consistency: Vec<BitVec> = (rank..rows.len()).map(|i| rows[i].1.clone()).collect();
        let is_pivot: Vec<bool> = {
            let mut v = vec![false; t];
            for &c in &pivot_cols {
                v[c] = true;
            }
            v
        };
        let mut kernel = Vec::new();
        for j in 0..t {
            if is_pivot[j] {
                continue;
            }
            let mut n = BitVec::new(t);
            n.set(j, true);
            for i in 0..rank {
                if rows[i].0.get(j) {
                    n.set(pivot_cols[i], true);
                }
            }
            kernel.push(n);
        }
        let kernel_supp: Vec<Vec<usize>> = kernel.iter().map(|n| n.ones()).collect();
        let kernel_u: Vec<UVec> = kernel_supp
            .iter()
            .map(|supp| {
                let mut u = UVec::zero(structure);
                for &k in supp {
                    u.xor_assign(&mcols[k]);
                }
                u
            })
            .collect();
        Elim { t, pivots, consistency, kernel, kernel_supp, kernel_u }
    }

    /// Allocation-free variant of [`solve`](Self::solve): writes the particular solution
    /// into `s` and returns `false` if the system is inconsistent.
    fn solve_into(&self, kappa: &BitVec, s: &mut BitVec) -> bool {
        for r in &self.consistency {
            if r.dot(kappa) {
                return false;
            }
        }
        s.clear();
        for (col, r) in &self.pivots {
            if r.dot(kappa) {
                s.set(*col, true);
            }
        }
        true
    }

    /// Particular solution of `K s = κ`, if any.
    #[allow(dead_code)]
    fn solve(&self, kappa: &BitVec) -> Option<BitVec> {
        for r in &self.consistency {
            if r.dot(kappa) {
                return None;
            }
        }
        let mut s = BitVec::new(self.t);
        for (col, r) in &self.pivots {
            if r.dot(kappa) {
                s.set(*col, true);
            }
        }
        Some(s)
    }
}

/// One non-Clifford branch term `e^{iθ z} = a0 + a1 (-1)^z`.
#[derive(Copy, Clone, Debug)]
struct Term {
    a: [Complex64; 2],
    /// `a1/a0` (multiplier when `s_k: 0 → 1`).
    up: Complex64,
    /// `a0/a1` (multiplier when `s_k: 1 → 0`).
    down: Complex64,
}

impl Term {
    fn new(theta: Phase) -> Term {
        let e = theta.to_complex();
        let a0 = (Complex64::new(1.0, 0.0) + e) * 0.5;
        let a1 = (Complex64::new(1.0, 0.0) - e) * 0.5;
        Term { a: [a0, a1], up: a1 / a0, down: a0 / a1 }
    }
}

/// The expensive, value-independent part: free/fixed split and the normal form.
#[derive(Debug)]
struct Core {
    fixed: Vec<usize>,
    fixed_pos: Vec<usize>,
    free: Vec<usize>,
    b_full: BitMat,
    dead: Vec<bool>,
    parity: BitVec,
    /// Adjacency among fixed variables.
    b_fixed: BitMat,
    /// Structure of the free form.
    structure: Structure,
    b_ff: BitMat,
    /// `Pᵀ (column of B between free vars and fixed var f)` for each fixed `f`.
    ucols: Vec<UVec>,
    /// Per-variable column vectors of the basis (see `Structure::column_vectors`).
    tcols: Vec<UVec>,
    /// Which entries of `tcols` were computed.
    tcols_valid: BitVec,
    opts: EngineOptions,
}

/// The cheap branch bookkeeping (depends on *which* variables carry rotations).
#[derive(Debug)]
struct Branching {
    /// Free-variable index of each branch term.
    term_vars: Vec<usize>,
    mcols: Vec<UVec>,
    elim: Elim,
}

/// A path sum with a cached normal form, ready to evaluate boundary assignments.
#[derive(Clone, Debug)]
pub struct Engine {
    core: Arc<Core>,
    br: Arc<Branching>,
    diag: Diag,
    fixed_v: Vec<u8>,
    fixed_theta: Vec<Phase>,
    terms: Vec<Term>,
    num_h: usize,
    global: Phase,
    zero: bool,
}

impl Engine {
    /// Build an engine for `ps` with the given fixed variables (each must be live and
    /// listed once).
    pub fn new(ps: &PathSum, fixed: &[usize], opts: EngineOptions) -> Engine {
        let nv = ps.num_vars;
        let mut fixed_pos = vec![NONE; nv];
        for (i, &f) in fixed.iter().enumerate() {
            assert!(f < nv && !ps.dead[f], "fixed variable {f} is not live");
            assert_eq!(fixed_pos[f], NONE, "duplicate fixed variable {f}");
            fixed_pos[f] = i;
        }
        let free: Vec<usize> = (0..nv).filter(|&i| !ps.dead[i] && fixed_pos[i] == NONE).collect();
        let m = free.len();
        let mut fidx = vec![NONE; nv];
        for (i, &f) in free.iter().enumerate() {
            fidx[f] = i;
        }
        let mut b_ff = BitMat::new(m, m);
        for (i, &f) in free.iter().enumerate() {
            for j in ps.b.row(f).iter_ones() {
                if fidx[j] != NONE {
                    b_ff.set(i, fidx[j], true);
                }
            }
        }
        let parity = BitVec::from_ones(m, (0..m).filter(|&i| ps.v[free[i]] & 1 == 1));
        let nf = fixed.len();
        let mut b_fixed = BitMat::new(nf, nf);
        for (i, &f) in fixed.iter().enumerate() {
            for j in ps.b.row(f).iter_ones() {
                if fixed_pos[j] != NONE {
                    b_fixed.set(i, fixed_pos[j], true);
                }
            }
        }
        let t0 = std::time::Instant::now();
        let structure = Structure::new(&b_ff, &parity, opts.pivot);
        let t_struct = t0.elapsed();
        let mut needed = BitVec::new(m);
        for &f in fixed {
            for j in ps.b.row(f).iter_ones() {
                if fidx[j] != NONE {
                    needed.set(fidx[j], true);
                }
            }
        }
        for i in 0..m {
            if ps.theta[free[i]] != Phase::ZERO {
                needed.set(i, true);
            }
        }
        let tcols = structure.column_vectors_for(&needed);
        let ucols: Vec<UVec> = fixed
            .iter()
            .map(|&f| {
                let mut col = BitVec::new(m);
                for j in ps.b.row(f).iter_ones() {
                    if fidx[j] != NONE {
                        col.set(fidx[j], true);
                    }
                }
                structure.transform_sparse(&tcols, &col)
            })
            .collect();
        if std::env::var_os("POLYQ_PROFILE").is_some() {
            eprintln!("[engine] m={} fixed={} structure {:?} ucols {:?} (g={} odd={} even={})", m, nf, t_struct, t0.elapsed() - t_struct, structure.g(), structure.odd.len(), structure.even.len());
        }
        let core = Core {
            fixed: fixed.to_vec(),
            fixed_pos,
            free,
            b_full: ps.b.clone(),
            dead: ps.dead.clone(),
            parity,
            b_fixed,
            structure,
            b_ff,
            ucols,
            tcols,
            tcols_valid: needed,
            opts,
        };
        let br = Self::build_branching(&core, ps);
        Self::bind_values(Arc::new(core), Arc::new(br), ps)
    }

    fn build_branching(core: &Core, ps: &PathSum) -> Branching {
        let m = core.free.len();
        let term_vars: Vec<usize> = (0..m).filter(|&i| ps.theta[core.free[i]] != Phase::ZERO).collect();
        let mcols: Vec<UVec> = term_vars.iter().map(|&i| if core.tcols_valid.get(i) { core.tcols[i].clone() } else { core.structure.column_of(i) }).collect();
        let elim = Elim::new(&core.structure, &mcols);
        Branching { term_vars, mcols, elim }
    }

    /// Attach the numerical data (`v`, `θ`, constants) of `ps` to an existing structure.
    fn bind_values(core: Arc<Core>, br: Arc<Branching>, ps: &PathSum) -> Engine {
        let m = core.free.len();
        let v_free: Vec<u8> = core.free.iter().map(|&f| ps.v[f]).collect();
        debug_assert_eq!(v_free.len(), m);
        let diag = core.structure.diagonal(&core.b_ff, &v_free);
        let terms: Vec<Term> = br.term_vars.iter().map(|&i| Term::new(ps.theta[core.free[i]])).collect();
        let fixed_v = core.fixed.iter().map(|&f| ps.v[f]).collect();
        let fixed_theta = core.fixed.iter().map(|&f| ps.theta[f]).collect();
        Engine { core, br, diag, fixed_v, fixed_theta, terms, num_h: ps.num_h, global: ps.global, zero: ps.zero }
    }

    /// Re-use this engine's expensive structure for another path sum that differs only in
    /// *values*: same `B`, same live set and same parity of `v` (so the normal form is
    /// unchanged).  The set of rotated variables may differ — then only the cheap branch
    /// bookkeeping (`M`, RREF of `K`: `O(m·t)`) is rebuilt.  Examples: Pauli-faulted
    /// copies of a circuit; appending `T`, `P(θ)`, `Z`, `X`, `Y`, `SWAP` to a circuit.
    /// Returns `None` if the structure differs (build a fresh engine then).
    pub fn rebind(&self, ps: &PathSum) -> Option<Engine> {
        let c = &self.core;
        if ps.num_vars != c.dead.len() || ps.dead != c.dead || ps.b != c.b_full {
            return None;
        }
        let m = c.free.len();
        for i in 0..m {
            if (ps.v[c.free[i]] & 1 == 1) != c.parity.get(i) {
                return None;
            }
        }
        let tv: Vec<usize> = (0..m).filter(|&i| ps.theta[c.free[i]] != Phase::ZERO).collect();
        let br = if tv == self.br.term_vars { Arc::clone(&self.br) } else { Arc::new(Self::build_branching(c, ps)) };
        Some(Self::bind_values(Arc::clone(&self.core), br, ps))
    }

    /// Size information.
    pub fn stats(&self) -> EngineStats {
        let (c, b) = (&self.core, &self.br);
        EngineStats {
            free_vars: c.free.len(),
            planes: c.structure.g(),
            lines: c.structure.odd.len(),
            radical: c.structure.even.len(),
            terms: b.term_vars.len(),
            branch_dim: b.elim.kernel.len(),
        }
    }

    /// The fixed variables, in the order expected by [`eval`](Self::eval).
    pub fn fixed(&self) -> &[usize] {
        &self.core.fixed
    }

    /// For an engine **without rotations** (pure Clifford), the set of values of the
    /// `unknown` fixed variables (given the `known` ones) for which the amplitude is
    /// non-zero is an **affine subspace** of `F2^{|unknown|}`, and `|amplitude|²` is
    /// *constant* on it.  Returns that subspace, or `None` if it is empty or the engine has
    /// rotations.
    ///
    /// Derivation: a fixed variable with value 1 contributes the shift column
    /// `ucol_f`; survival requires `(⊕_{f=1} ucol_f).even == w_even`, a linear system in the
    /// unknown bits.  This is the "weak simulation without a statevector" primitive.
    pub fn affine_support(&self, known: &[(usize, bool)], unknown: &[usize]) -> Option<AffineSpace> {
        if !self.br.term_vars.is_empty() || self.zero {
            return None;
        }
        let c = &self.core;
        let ne = c.structure.even.len();
        let mut rhs = self.diag.we.clone();
        for &(var, val) in known {
            if val {
                rhs.xor_assign(&c.ucols[c.fixed_pos[var]].e);
            }
        }
        let k = unknown.len();
        // rows: (A row over k unknown bits, R row over ne)
        let mut rows: Vec<(BitVec, BitVec)> = (0..ne)
            .map(|e| {
                let mut a = BitVec::new(k);
                for (j, &o) in unknown.iter().enumerate() {
                    if c.ucols[c.fixed_pos[o]].e.get(e) {
                        a.set(j, true);
                    }
                }
                (a, BitVec::from_ones(ne, [e]))
            })
            .collect();
        let mut pivots = Vec::new();
        let mut rank = 0;
        for col in 0..k {
            let Some(r) = (rank..rows.len()).find(|&r| rows[r].0.get(col)) else { continue };
            rows.swap(rank, r);
            let (pa, pr) = (rows[rank].0.clone(), rows[rank].1.clone());
            for r2 in 0..rows.len() {
                if r2 != rank && rows[r2].0.get(col) {
                    rows[r2].0.xor_assign(&pa);
                    rows[r2].1.xor_assign(&pr);
                }
            }
            pivots.push(col);
            rank += 1;
        }
        for r in &rows[rank..] {
            if r.1.dot(&rhs) {
                return None;
            }
        }
        let mut origin = BitVec::new(k);
        for (i, &col) in pivots.iter().enumerate() {
            if rows[i].1.dot(&rhs) {
                origin.set(col, true);
            }
        }
        let mut is_piv = vec![false; k];
        for &p in &pivots {
            is_piv[p] = true;
        }
        let mut basis = Vec::new();
        for j in 0..k {
            if is_piv[j] {
                continue;
            }
            let mut v = BitVec::new(k);
            v.set(j, true);
            for (i, &col) in pivots.iter().enumerate() {
                if rows[i].0.get(j) {
                    v.set(col, true);
                }
            }
            basis.push(v);
        }
        Some(AffineSpace { origin, basis })
    }

    /// Evaluate with an assignment given as `(variable, value)` pairs covering every fixed
    /// variable.
    pub fn eval_assignment(&self, assignment: &[(usize, bool)]) -> Complex64 {
        let mut sc = self.scratch();
        self.eval_assignment_with(assignment, &mut sc)
    }

    /// [`eval_assignment`](Self::eval_assignment) re-using work buffers.
    pub fn eval_assignment_with(&self, assignment: &[(usize, bool)], sc: &mut Scratch) -> Complex64 {
        let s = &self.core;
        let mut vals = vec![false; s.fixed.len()];
        let mut seen = 0;
        for &(var, x) in assignment {
            let p = s.fixed_pos.get(var).copied().unwrap_or(NONE);
            assert!(p != NONE, "variable {var} is not fixed in this engine");
            vals[p] = x;
            seen += 1;
        }
        assert!(seen >= s.fixed.len(), "assignment does not cover all fixed variables");
        self.eval_with(&vals, sc)
    }

    /// Reusable work buffers for [`eval_with`](Self::eval_with) (one per thread).
    pub fn scratch(&self) -> Scratch {
        let st = &self.core.structure;
        Scratch {
            u0: UVec::zero(st),
            u: UVec::zero(st),
            kappa: BitVec::new(st.even.len()),
            s0: BitVec::new(self.terms.len()),
            sv: BitVec::new(self.terms.len()),
            ones: Vec::new(),
        }
    }

    /// Evaluate the amplitude-normalised sum
    /// `2^{-h/2} e^{iφ} Σ_{free z} i^{Q(z,x_F)} e^{iΘ(z,x_F)}` for fixed values `values`
    /// (aligned with [`fixed`](Self::fixed)).
    pub fn eval(&self, values: &[bool]) -> Complex64 {
        let mut sc = self.scratch();
        self.eval_with(values, &mut sc)
    }

    /// Same as [`eval`](Self::eval) but without per-call allocations (hot loops).
    pub fn eval_with(&self, values: &[bool], sc: &mut Scratch) -> Complex64 {
        let Some((phase, c_start)) = self.prepare(values, sc) else { return Complex64::new(0.0, 0.0) };
        let total = self.sum_subspace(&sc.u, &sc.s0, c_start);
        phase.to_complex() * total * self.magnitude()
    }

    fn magnitude(&self) -> f64 {
        2f64.powf(self.core.structure.log2_magnitude() - self.num_h as f64 / 2.0)
    }

    /// Boundary constants, shift and start point of the surviving affine subspace.
    /// Leaves `sc.u`/`sc.s0` at the start point; `None` if the sum vanishes identically.
    fn prepare(&self, values: &[bool], sc: &mut Scratch) -> Option<(Phase, Complex64)> {
        if self.zero {
            return None;
        }
        let (s, br) = (&self.core, &self.br);
        debug_assert_eq!(values.len(), s.fixed.len());
        sc.ones.clear();
        sc.ones.extend((0..values.len()).filter(|&i| values[i]));
        let mut z4: u32 = 0;
        let mut th = self.global;
        for (ii, &f) in sc.ones.iter().enumerate() {
            z4 += self.fixed_v[f] as u32;
            th += self.fixed_theta[f];
            for &g in &sc.ones[ii + 1..] {
                if s.b_fixed.get(f, g) {
                    z4 += 2;
                }
            }
        }
        let phase = th + z4_phase((z4 & 3) as u8);
        sc.u0.clear();
        for &f in &sc.ones {
            sc.u0.xor_assign(&s.ucols[f]);
        }
        sc.kappa.copy_from(&self.diag.we);
        sc.kappa.xor_assign(&sc.u0.e);
        if !br.elim.solve_into(&sc.kappa, &mut sc.s0) {
            return None;
        }
        sc.u.copy_from(&sc.u0);
        let mut c_start = Complex64::new(1.0, 0.0);
        for k in 0..self.terms.len() {
            if sc.s0.get(k) {
                sc.u.xor_assign(&br.mcols[k]);
                c_start *= self.terms[k].a[1];
            } else {
                c_start *= self.terms[k].a[0];
            }
        }
        debug_assert!(Structure::survives(&self.diag, &sc.u));
        Some((phase, c_start))
    }

    /// **Monte-Carlo estimate** of [`eval`](Self::eval): average `2^d · c(s) · G(u(s))` over
    /// `samples` points `s` drawn uniformly from the surviving affine branch subspace
    /// (dimension `d`).  Unbiased; returns `(estimate, standard_error)`.  Falls back to the exact
    /// sum when `d ≤ log2(samples)`.  Use when `d` is too large for exact enumeration.
    pub fn estimate(&self, values: &[bool], samples: usize, seed: u64) -> (Complex64, f64) {
        let mut sc = self.scratch();
        let Some((phase, c_start)) = self.prepare(values, &mut sc) else { return (Complex64::new(0.0, 0.0), 0.0) };
        let e = &self.br.elim;
        let d = e.kernel.len();
        let norm = phase.to_complex() * self.magnitude();
        if d < 63 && (1usize << d) <= samples {
            return (norm * self.sum_subspace(&sc.u, &sc.s0, c_start), 0.0);
        }
        let (u_start, s_start) = (sc.u.clone(), sc.s0.clone());
        let chunks = (samples + 1023) / 1024;
        let (sum, sq, cnt) = (0..chunks)
            .into_par_iter()
            .map(|ci| {
                let mut rng = crate::rng::Rng::new(seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let n_here = 1024.min(samples - ci * 1024);
                let (mut sum, mut sq) = (Complex64::new(0.0, 0.0), 0.0f64);
                let (mut u, mut sv) = (u_start.clone(), s_start.clone());
                for _ in 0..n_here {
                    u.copy_from(&u_start);
                    sv.copy_from(&s_start);
                    let mut word = 0u64;
                    for j in 0..d {
                        if j % 64 == 0 {
                            word = rng.next_u64();
                        }
                        if (word >> (j % 64)) & 1 == 1 {
                            u.xor_assign(&e.kernel_u[j]);
                            sv.xor_assign(&e.kernel[j]);
                        }
                    }
                    let x = self.weight(&sv) * ROOT8[Structure::phase8(&self.diag, &u) as usize];
                    sum += x;
                    sq += x.norm_sqr();
                }
                (sum, sq, n_here)
            })
            .reduce(|| (Complex64::new(0.0, 0.0), 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let n = cnt as f64;
        let mean = sum / n;
        let var = (sq / n - mean.norm_sqr()).max(0.0);
        let scale = 2f64.powi(d as i32);
        (norm * mean * scale, norm.norm() * scale * (var / n).sqrt())
    }

    fn weight(&self, svec: &BitVec) -> Complex64 {
        let mut c = Complex64::new(1.0, 0.0);
        for (k, t) in self.terms.iter().enumerate() {
            c *= t.a[svec.get(k) as usize];
        }
        c
    }

    fn sum_subspace(&self, u_start: &UVec, s_start: &BitVec, c_start: Complex64) -> Complex64 {
        let e = &self.br.elim;
        let d = e.kernel.len();
        let dl = d.min(LOW_BITS);
        let nhi = 1usize << (d - dl);
        let chunk = |hi: usize| -> Complex64 {
            let (mut u, mut sv) = (u_start.clone(), s_start.clone());
            let mut c = c_start;
            if hi != 0 {
                for j in 0..(d - dl) {
                    if (hi >> j) & 1 == 1 {
                        u.xor_assign(&e.kernel_u[dl + j]);
                        sv.xor_assign(&e.kernel[dl + j]);
                    }
                }
                c = self.weight(&sv);
            }
            self.gray(u, sv, c, dl)
        };
        if d >= self.core.opts.parallel_min_dim && nhi > 1 {
            (0..nhi).into_par_iter().map(chunk).sum()
        } else {
            (0..nhi).map(chunk).sum()
        }
    }

    /// Gray-code walk over the first `dl` kernel vectors starting at `(u, s, c)`.
    fn gray(&self, mut u: UVec, mut sv: BitVec, mut c: Complex64, dl: usize) -> Complex64 {
        let e = &self.br.elim;
        let mut acc = Complex64::new(0.0, 0.0);
        let steps = 1usize << dl;
        for i in 0..steps {
            let k8 = Structure::phase8(&self.diag, &u);
            acc += c * ROOT8[k8 as usize];
            if i + 1 < steps {
                let j = (i + 1).trailing_zeros() as usize;
                u.xor_assign(&e.kernel_u[j]);
                if (i + 1) % RECOMPUTE_EVERY == 0 {
                    sv.xor_assign(&e.kernel[j]);
                    c = self.weight(&sv);
                } else {
                    for &k in &e.kernel_supp[j] {
                        if sv.get(k) {
                            c *= self.terms[k].down;
                        } else {
                            c *= self.terms[k].up;
                        }
                        sv.flip(k);
                    }
                }
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::rng::{random_clifford_t, Rng};

    fn bits(n: usize, idx: usize) -> Vec<bool> {
        (0..n).map(|q| (idx >> q) & 1 == 1).collect()
    }

    fn std_fixed(ps: &PathSum) -> Vec<usize> {
        let mut f: Vec<usize> = (0..ps.num_qubits).collect();
        for &o in &ps.outputs {
            if !f.contains(&o) {
                f.push(o);
            }
        }
        f
    }

    #[test]
    fn engine_matches_brute_force() {
        let mut rng = Rng::new(21);
        let mut checked = 0;
        for trial in 0..120 {
            let n = 1 + rng.below(4);
            let c = random_clifford_t(n, 8 + rng.below(14), rng.below(6), trial % 3 == 0, &mut rng);
            let ps = PathSum::from_circuit(&c);
            if ps.live_vars().len() > 20 {
                continue;
            }
            let eng = Engine::new(&ps, &std_fixed(&ps), EngineOptions::default());
            checked += 1;
            for xin in 0..(1usize << n) {
                for y in 0..(1usize << n) {
                    let (x, yb) = (bits(n, xin), bits(n, y));
                    let want = match ps.boundary(&x, &yb) {
                        None => Complex64::new(0.0, 0.0),
                        Some(bd) => ps.brute_force(&bd),
                    };
                    let got = match ps.boundary(&x, &yb) {
                        None => Complex64::new(0.0, 0.0),
                        Some(bd) => eng.eval_assignment(&bd),
                    };
                    assert!((got - want).norm() < 1e-8, "trial {trial} n={n} x={xin} y={y}: {got} vs {want}");
                }
            }
        }
        assert!(checked >= 60, "only {checked}");
    }

    #[test]
    fn bell_and_ghz() {
        let mut c = Circuit::new(3);
        c.h(0).cx(0, 1).cx(1, 2);
        let ps = PathSum::from_circuit(&c);
        let eng = Engine::new(&ps, &std_fixed(&ps), EngineOptions::default());
        let amp = |y: usize| eng.eval_assignment(&ps.boundary(&[false; 3], &bits(3, y)).unwrap());
        for y in 0..8 {
            let want = if y == 0 || y == 7 { std::f64::consts::FRAC_1_SQRT_2 } else { 0.0 };
            assert!((amp(y) - Complex64::new(want, 0.0)).norm() < 1e-12, "y={y}");
        }
    }

    #[test]
    fn old_v1_bug_case_is_fixed() {
        // H T Z H |0> = 1/2[(1-w)|0> + (1+w)|1>]; v1 returned the wrong sign on |0>.
        let mut c = Circuit::new(1);
        c.h(0).t(0).z(0).h(0);
        let ps = PathSum::from_circuit(&c);
        let eng = Engine::new(&ps, &std_fixed(&ps), EngineOptions::default());
        let w = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        let one = Complex64::new(1.0, 0.0);
        let a0 = eng.eval_assignment(&ps.boundary(&[false], &[false]).unwrap());
        let a1 = eng.eval_assignment(&ps.boundary(&[false], &[true]).unwrap());
        assert!((a0 - (one - w) * 0.5).norm() < 1e-12, "{a0}");
        assert!((a1 - (one + w) * 0.5).norm() < 1e-12, "{a1}");
    }
}
