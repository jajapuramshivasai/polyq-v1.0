//! Exact **low-rank stabiliser decomposition** of T-type rotation pairs, and the cost
//! model that decides how many pairs to decompose.
//!
//! ## The decomposition
//!
//! For two path variables `x, y` carrying rotations `e^{iθ_a x}` and `e^{iθ_b y}` the
//! function `f(x,y) = e^{iθ_a x + iθ_b y}` splits **exactly** into two terms supported on
//! complementary affine subspaces:
//!
//! ```text
//!   f(x,y) = [x = y]·e^{i(θ_a+θ_b) x}  +  [x ≠ y]·e^{iθ_b}·e^{i(θ_a−θ_b) x}.
//! ```
//!
//! For `θ_a, θ_b` odd multiples of `π/4` (any mix of `T` and `T†`), both `θ_a ± θ_b`
//! are multiples of `π/2`, so *each branch is Clifford*: the pair costs 2 branches instead
//! of the 4 of the plain Z-decomposition.  In state language this is the classic fact
//! `χ(|T⟩^{⊗2}) = 2`:
//!
//! ```text
//!   |T⟩|T⟩ = √2 (|00⟩ + i|11⟩)/√2 + (1+i) (|01⟩ + |10⟩)/√2 .
//! ```
//!
//! (we found it by exhaustive search over the 60 two-qubit stabiliser states).  The
//! indicator `[x = y]` is realised by the variable substitution
//! [`PathSum::substitute`], so a branch is an ordinary path sum with one variable fewer.
//!
//! **Generalisation.** The same identity is exact for *any* angles; it is useful whenever one of
//! `θ_a ± θ_b` is a multiple of `π/2`, i.e. for every pair of equal or opposite angles modulo
//! `π/2` (not just `T`/`T†`).  See [`pair_candidates`].
//!
//! Decomposing `p` disjoint pairs yields `2^p` branches that each have `t − 2p` rotations
//! left (handled by the cheap Z-decomposition inside [`Engine`]): total work
//! `2^p · (LDL + 2^{d(t−2p)})`.  `p = 0` is the pure cached Z-branch engine.
//!
//! [`RankEngine::auto`] picks `p` with a simple cost model (documented below), so tiny
//! `T`-counts stay on the cached fast path while T-heavy circuits halve their exponent.

use crate::engine::{Engine, EngineOptions, Scratch};
use crate::pathsum::PathSum;
use num_complex::Complex64;
use rayon::prelude::*;

/// Maximum number of decomposed pairs (`2^MAX_PAIRS` branch engines are materialised).
pub const MAX_PAIRS: usize = 14;

/// Sum of `2^p` cached engines (one per stabiliser branch).
#[derive(Debug, Clone)]
pub struct RankEngine {
    branches: Vec<Engine>,
    pairs: usize,
}

const MASK30: u32 = (1 << 30) - 1;

/// Disjoint pairs of free rotated variables `(a, b)` for which at least one of
/// `θ_a ± θ_b` is a multiple of `π/2` (so at least one branch of the pair decomposition
/// removes both rotations).  Pairs are ordered by usefulness: T-class pairs (both branches
/// Clifford) first, then equal/opposite-angle pairs of generic angles (one clean branch, the
/// other keeps a single rotation `θ_a ± θ_b`).
///
/// Returns `(pairs, number_of_T_class_pairs)`.
pub fn pair_candidates(ps: &PathSum, fixed: &[usize]) -> (Vec<(usize, usize)>, usize) {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for k in 0..ps.num_vars {
        if ps.dead[k] || ps.theta[k] == crate::phase::Phase::ZERO || fixed.contains(&k) {
            continue;
        }
        let r = ps.theta[k].0 & MASK30;
        let canon = r.min(((MASK30 as u64 + 1 - r as u64) & MASK30 as u64) as u32);
        groups.entry(canon).or_default().push(k);
    }
    let mut tpairs = Vec::new();
    let mut gpairs = Vec::new();
    for (canon, vars) in groups {
        let target = if canon == 1 << 29 { &mut tpairs } else { &mut gpairs };
        for ch in vars.chunks_exact(2) {
            target.push((ch[0], ch[1]));
        }
    }
    let nt = tpairs.len();
    tpairs.extend(gpairs);
    (tpairs, nt)
}

impl RankEngine {
    /// Decompose `pairs` disjoint T-class pairs (clamped to what is available).
    pub fn new(ps: &PathSum, fixed: &[usize], pairs: usize, opts: EngineOptions) -> RankEngine {
        let (cands, _) = pair_candidates(ps, fixed);
        let p = pairs.min(cands.len()).min(MAX_PAIRS);
        let chosen: Vec<(usize, usize)> = cands[..p].to_vec();
        let branches: Vec<Engine> = (0usize..(1usize << p))
            .into_par_iter()
            .map(|pattern| {
                let mut b = ps.clone();
                for (i, &(a, bb)) in chosen.iter().enumerate() {
                    b.substitute(a, bb, (pattern >> i) & 1 == 1);
                }
                Engine::new(&b, fixed, opts)
            })
            .collect();
        RankEngine { branches, pairs: p }
    }

    /// Choose the number of pairs with the cost model below and build the engine.
    ///
    /// Let `m` be the number of free variables, `d₀` the branch-subspace dimension of the
    /// plain engine and `A` the number of available pairs.  The model (in word operations) is
    ///
    /// ```text
    ///   cost(p) = 2^p · ( 2m³/64 + 50m + 2000  +  2^{max(0, d₀−2p)} · (20 + 4⌈m/64⌉ + t/2) )
    /// ```
    ///
    /// and `p = argmin_{0 ≤ p ≤ min(A, MAX_PAIRS)} cost(p)`.
    pub fn auto(ps: &PathSum, fixed: &[usize], opts: EngineOptions) -> RankEngine {
        let base = Engine::new(ps, fixed, opts);
        let (cands, nt) = pair_candidates(ps, fixed);
        let p = best_pairs(&base, cands.len(), nt);
        if p == 0 {
            return RankEngine { branches: vec![base], pairs: 0 };
        }
        Self::new(ps, fixed, p, opts)
    }

    /// Number of decomposed pairs `p` (there are `2^p` branches).
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// Number of stabiliser branches.
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// One scratch buffer per branch (see [`eval_assignment_with`](Self::eval_assignment_with)).
    pub fn scratch(&self) -> Vec<Scratch> {
        self.branches.iter().map(|e| e.scratch()).collect()
    }

    /// Sequential, allocation-free evaluation for hot loops that are already parallel
    /// over assignments (statevector, sampling).
    pub fn eval_assignment_with(&self, assignment: &[(usize, bool)], sc: &mut [Scratch]) -> Complex64 {
        self.branches.iter().zip(sc.iter_mut()).map(|(e, s)| e.eval_assignment_with(assignment, s)).sum()
    }

    /// Unbiased Monte-Carlo estimate of an amplitude: with a single engine, samples its branch
    /// subspace ([`Engine::estimate`]); with `2^p` stabiliser branches, draws `samples` branches
    /// uniformly and evaluates them exactly (`2^p · mean`).  Returns `(value, stderr)`.
    pub fn estimate(&self, assignment: &[(usize, bool)], samples: usize, seed: u64) -> (Complex64, f64) {
        if self.branches.len() == 1 {
            let e = &self.branches[0];
            let mut vals = vec![false; e.fixed().len()];
            for &(v, x) in assignment {
                vals[e.fixed().iter().position(|&f| f == v).unwrap()] = x;
            }
            return e.estimate(&vals, samples, seed);
        }
        let nb = self.branches.len();
        let (sum, sq) = (0..samples)
            .into_par_iter()
            .map(|i| {
                let mut rng = crate::rng::Rng::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let a = self.branches[rng.below(nb)].eval_assignment(assignment);
                (a, a.norm_sqr())
            })
            .reduce(|| (Complex64::new(0.0, 0.0), 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let n = samples as f64;
        let mean = sum / n;
        let var = (sq / n - mean.norm_sqr()).max(0.0);
        (mean * nb as f64, nb as f64 * (var / n).sqrt())
    }

    /// Statistics of the first branch (representative).
    pub fn stats(&self) -> crate::engine::EngineStats {
        self.branches[0].stats()
    }

    /// Fixed variables expected by [`eval_assignment`](Self::eval_assignment).
    pub fn fixed(&self) -> &[usize] {
        self.branches[0].fixed()
    }

    /// Evaluate a boundary assignment (sum over all branches).
    pub fn eval_assignment(&self, assignment: &[(usize, bool)]) -> Complex64 {
        if self.branches.len() >= 8 {
            self.branches.par_iter().map(|e| e.eval_assignment(assignment)).sum()
        } else {
            self.branches.iter().map(|e| e.eval_assignment(assignment)).sum()
        }
    }

    /// Re-bind all branch engines to a value-only variant of the path sum, if possible
    /// (see [`Engine::rebind`]).  Not supported when pairs are decomposed (the branches
    /// depend on the values of the rotations); returns `None` then.
    pub fn rebind(&self, ps: &PathSum) -> Option<RankEngine> {
        if self.pairs == 0 {
            self.branches[0].rebind(ps).map(|e| RankEngine { branches: vec![e], pairs: 0 })
        } else {
            None
        }
    }
}

/// Cost model, see [`RankEngine::auto`].
pub fn best_pairs(base: &Engine, available_pairs: usize, t_class_pairs: usize) -> usize {
    let st = base.stats();
    let m = st.free_vars as f64;
    let words = (st.free_vars / 64 + 1) as f64;
    let ldl = 2.0 * m * m * m / 64.0 + 50.0 * m + 2000.0;
    let step = 20.0 + 4.0 * words + st.terms as f64 / 2.0;
    let mut best = (0usize, f64::INFINITY);
    for p in 0..=available_pairs.min(MAX_PAIRS) {
        // a T-class pair removes 2 branch dimensions, a generic pair about 1 (one clean branch)
        let removed = 2 * p.min(t_class_pairs) + p.saturating_sub(t_class_pairs);
        let d = st.branch_dim.saturating_sub(removed) as i32;
        let cost = 2f64.powi(p as i32) * (if p == 0 { 0.0 } else { ldl } + 2f64.powi(d) * step);
        if cost < best.1 {
            best = (p, cost);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn rank_engine_matches_plain_engine() {
        let mut rng = Rng::new(5);
        let mut with_pairs = 0;
        for trial in 0..60 {
            let n = 2 + rng.below(3);
            let c = if trial % 2 == 0 {
                random_clifford_t(n, 20, 4 + rng.below(5), trial % 4 == 0, &mut rng)
            } else {
                crate::rng::random_layered(n, 3, &mut rng)
            };
            let ps = PathSum::from_circuit(&c);
            if ps.live_vars().len() > 20 {
                continue;
            }
            let fixed = std_fixed(&ps);
            let plain = Engine::new(&ps, &fixed, EngineOptions::default());
            for pairs in [1usize, 2, 3, 6] {
                let r = RankEngine::new(&ps, &fixed, pairs, EngineOptions::default());
                if r.pairs() > 0 {
                    with_pairs += 1;
                }
                for xin in 0..(1usize << n) {
                    let x = bits(n, xin);
                    for y in 0..(1usize << n) {
                        let Some(bd) = ps.boundary(&x, &bits(n, y)) else { continue };
                        let (a, b) = (plain.eval_assignment(&bd), r.eval_assignment(&bd));
                        assert!((a - b).norm() < 1e-8, "trial {trial} pairs {pairs}: {a} vs {b}");
                    }
                }
            }
        }
        assert!(with_pairs >= 20, "pair decomposition barely exercised ({with_pairs})");
    }

    #[test]
    fn generic_equal_angle_pairs_are_exact() {
        use crate::phase::Phase;
        let mut rng = Rng::new(123);
        let mut used = 0;
        for _ in 0..30 {
            let n = 3;
            let mut c = crate::circuit::Circuit::new(n);
            let th = Phase::from_pi_fraction(1 + 2 * rng.below(3) as i64, 3); // odd multiples of pi/8
            for _ in 0..2 {
                for q in 0..n {
                    c.h(q);
                }
                for q in 0..n {
                    if rng.next_f64() < 0.8 {
                        c.p(q, if rng.next_f64() < 0.5 { th } else { -th });
                    }
                }
                c.cz(0, 1);
                c.cz(1, 2);
            }
            let ps = PathSum::from_circuit(&c);
            if ps.live_vars().len() > 18 {
                continue;
            }
            let fixed = std_fixed(&ps);
            let (cands, nt) = pair_candidates(&ps, &fixed);
            assert_eq!(nt, 0);
            let plain = Engine::new(&ps, &fixed, EngineOptions::default());
            let r = RankEngine::new(&ps, &fixed, 3, EngineOptions::default());
            used += r.pairs().min(cands.len());
            for xin in 0..(1usize << n) {
                for y in 0..(1usize << n) {
                    let Some(bd) = ps.boundary(&bits(n, xin), &bits(n, y)) else { continue };
                    let (a, b) = (plain.eval_assignment(&bd), r.eval_assignment(&bd));
                    assert!((a - b).norm() < 1e-8, "{a} vs {b}");
                }
            }
        }
        assert!(used >= 20, "generic pairing barely exercised ({used})");
    }

    #[test]
    fn t_pair_identity_is_clifford_in_both_branches() {
        // two T on independent wires after an H each: both branches must have no rotations left
        let mut c = crate::circuit::Circuit::new(2);
        c.h(0).h(1).t(0).t(1).h(0).h(1);
        let ps = PathSum::from_circuit(&c);
        let fixed = std_fixed(&ps);
        let r = RankEngine::new(&ps, &fixed, 1, EngineOptions::default());
        assert_eq!(r.pairs(), 1);
        assert_eq!(r.num_branches(), 2);
        assert_eq!(r.stats().terms, 0);
    }
}
