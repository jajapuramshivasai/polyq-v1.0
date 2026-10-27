//! Dumas–Pernet-style `LDLᵀ` normal form of a Z4 quadratic form, and its closed-form
//! Gauss sum.
//!
//! ## The problem
//!
//! Let `q(x) = Σ_k v_k x_k + 2 Σ_{j<k} B_{jk} x_j x_k (mod 4)` with `x ∈ F2^m`,
//! `B` symmetric over F2 with zero diagonal and `v ∈ Z4^m`.  We need
//! `G = Σ_x i^{q(x)}` (plus the same sum with an extra *Z-type shift*
//! `(-1)^{b·x}`, which is how boundary conditions and branch terms enter).
//!
//! ## Key facts
//!
//! 1. `q(x)` depends only on `x mod 2`, and for any integer matrix `P` that is
//!    invertible mod 2, `q(Py)` is again a Z4 quadratic form: with `A = diag(v) +
//!    2·offdiag(B)` (a symmetric matrix over Z4) one has `q(Py) = yᵀ(PᵀAP)y`.
//! 2. Only the **F2 Gram matrix** `Ā = B + diag(v mod 2)` controls the *structure*:
//!    off-diagonal entries of `PᵀAP` mod 2 are those of `PᵀĀP`.
//! 3. Symmetric Gaussian elimination over F2 (this file) finds a basis
//!    `p_1..p_m` in which `Ā` is **block diagonal** with blocks
//!    * `[1]`            — a *non-isotropic line* (`p^TĀp = 1`);
//!    * `[[0,1],[1,0]]`   — a *hyperbolic plane*;
//!    * `[0]`             — a *radical* vector.
//!
//!    This is a symmetric `P L D Lᵀ Pᵀ` factorisation with 1×1 and 2×2 pivots — the
//!    symmetric specialisation of the Dumas–Pernet rank-profile elimination.
//! 4. In that basis the Z4 form splits into independent factors, each given by the
//!    Z4 value `d(p) = q(p)` of the basis vector(s):
//!
//!    | block | factor `Σ_y` |
//!    |-------|--------------|
//!    | line, `d` odd     | `1 + i^d  = √2·e^{±iπ/4}` |
//!    | radical, `d` even | `1 + i^d  = 2` (`d=0`) or `0` (`d=2`) |
//!    | plane, `d_a=2w_a`, `d_b=2w_b` | `2·(-1)^{w_a w_b}` |
//!
//! 5. A Z-type shift `(-1)^{b·x}` is just `d_l → d_l + 2 (Pᵀb)_l`: **the structure
//!    never has to be recomputed** when only the shift changes.  This is the crux of
//!    the cached simulator: boundary conditions, `T`-branches (Z-decomposition) and
//!    Pauli faults are all shifts.
//!
//! ## Complexity
//!
//! The elimination is word-parallel: every pivot costs `O(|S|·m/64)` word XORs where
//! `S` is the pivot's neighbourhood, so the worst case is `O(m³/64)`; with the
//! `MinWeight` pivot rule sparse circuit graphs stay sparse.  (A *sub-cubic*
//! `O(m^ω)` variant needs fast F2 matrix multiplication / recursion à la M4RI; the
//! block structure produced here is exactly what such a recursion would produce, so
//! it can be swapped in behind [`Structure::new`].)

use crate::bits::{BitMat, BitVec};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::BTreeSet;


/// Threshold (word operations) above which neighbourhood updates run in parallel.
const PAR_WORK: usize = 1 << 15;

/// One 1×1 pivot step on alive row `a` with `Ā_aa = 1`: Schur complement
/// `Ā'_{bc} = Ā_bc ⊕ Ā_ba Ā_ac` and `p_b += p_a` for the neighbours `b`.
/// Returns the neighbour set and the pivot's basis vector.
fn single_step(gram: &mut [BitVec], basis: &mut [BitVec], alive: &mut BitVec, a: usize) -> (BitVec, BitVec) {
    let mut s = gram[a].clone();
    s.and_assign(alive);
    s.set(a, false);
    let pa = basis[a].clone();
    let words = s.words().len();
    if s.count_ones() as usize * words >= PAR_WORK {
        gram.par_iter_mut().zip(basis.par_iter_mut()).enumerate().for_each(|(i, (g, bv))| {
            if s.get(i) {
                g.xor_assign(&s);
                bv.xor_assign(&pa);
            }
        });
    } else {
        for bi in s.iter_ones() {
            gram[bi].xor_assign(&s);
            basis[bi].xor_assign(&pa);
        }
    }
    alive.set(a, false);
    (s, pa)
}

/// One 2×2 (hyperbolic) pivot step for alive row `a` with `Ā_aa = 0` and a neighbour `b`:
/// `Ā'_{kc} = Ā_kc ⊕ Ā_ka Ā_bc ⊕ Ā_kb Ā_ac`, `p_c += Ā_cb p_a + Ā_ca p_b`.
/// Returns the touched rows and the plane's two basis vectors.
fn hyper_step(gram: &mut [BitVec], basis: &mut [BitVec], alive: &mut BitVec, a: usize) -> (BitVec, BitVec, BitVec, usize) {
    let mut ra = gram[a].clone();
    ra.and_assign(alive);
    let b_idx = ra.first_one().expect("row has a neighbour");
    let mut sa = ra;
    sa.set(b_idx, false);
    let mut sb = gram[b_idx].clone();
    sb.and_assign(alive);
    sb.set(a, false);
    let (pa, pb) = (basis[a].clone(), basis[b_idx].clone());
    let mut touched = sa.clone();
    let mut both = sa.clone();
    both.and_assign(&sb);
    touched.xor_assign(&sb);
    touched.xor_assign(&both); // union of sa and sb
    let words = sa.words().len();
    if touched.count_ones() as usize * words * 2 >= PAR_WORK {
        gram.par_iter_mut().zip(basis.par_iter_mut()).enumerate().for_each(|(k, (g, bv))| {
            if sa.get(k) {
                g.xor_assign(&sb);
                bv.xor_assign(&pb);
            }
            if sb.get(k) {
                g.xor_assign(&sa);
                bv.xor_assign(&pa);
            }
        });
    } else {
        for k in touched.iter_ones() {
            if sa.get(k) {
                gram[k].xor_assign(&sb);
                basis[k].xor_assign(&pb);
            }
            if sb.get(k) {
                gram[k].xor_assign(&sa);
                basis[k].xor_assign(&pa);
            }
        }
    }
    alive.set(a, false);
    alive.set(b_idx, false);
    (touched, pa, pb, b_idx)
}

/// Pivot selection rule for the symmetric elimination.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PivotRule {
    /// Take the first admissible pivot (deterministic, cheapest to choose).
    First,
    /// Take the admissible pivot with the lightest row (Markowitz-style; keeps basis
    /// vectors and the Gram matrix sparse, which speeds up later popcounts).
    MinWeight,
}

/// Block-diagonalising basis of the F2 Gram matrix (see module docs).
///
/// Basis vectors are `BitVec`s over the `m` original variables.  They are stored by
/// block type: `pairs_a[j], pairs_b[j]` form the `j`-th hyperbolic plane.
#[derive(Clone, Debug)]
pub struct Structure {
    /// Number of variables.
    pub m: usize,
    /// First vector of each hyperbolic plane.
    pub pairs_a: Vec<BitVec>,
    /// Second vector of each hyperbolic plane.
    pub pairs_b: Vec<BitVec>,
    /// Non-isotropic lines (`d` is odd).
    pub odd: Vec<BitVec>,
    /// Radical vectors (`d` is even; a `d = 2` here kills the whole sum).
    pub even: Vec<BitVec>,
}

/// Coordinates of a vector in the normal-form basis, split by region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UVec {
    /// Coordinates along `pairs_a`.
    pub a: BitVec,
    /// Coordinates along `pairs_b`.
    pub b: BitVec,
    /// Coordinates along `odd`.
    pub o: BitVec,
    /// Coordinates along `even`.
    pub e: BitVec,
}

impl UVec {
    /// All-zero coordinates for the given structure.
    pub fn zero(s: &Structure) -> Self {
        Self { a: BitVec::new(s.pairs_a.len()), b: BitVec::new(s.pairs_b.len()), o: BitVec::new(s.odd.len()), e: BitVec::new(s.even.len()) }
    }
    /// Overwrite with `o` (same shape) without reallocating.
    #[inline(always)]
    pub fn copy_from(&mut self, o: &UVec) {
        self.a.copy_from(&o.a);
        self.b.copy_from(&o.b);
        self.o.copy_from(&o.o);
        self.e.copy_from(&o.e);
    }
    /// Zero all coordinates.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
        self.o.clear();
        self.e.clear();
    }
    /// `self ^= other`.
    #[inline(always)]
    pub fn xor_assign(&mut self, o: &UVec) {
        self.a.xor_assign(&o.a);
        self.b.xor_assign(&o.b);
        self.o.xor_assign(&o.o);
        self.e.xor_assign(&o.e);
    }
}

/// Z4 values of the basis vectors, repackaged as the masks the evaluator needs.
#[derive(Clone, Debug)]
pub struct Diag {
    /// `w_a[j] = (d(pairs_a[j]) / 2) mod 2`.
    pub wa: BitVec,
    /// `w_b[j] = (d(pairs_b[j]) / 2) mod 2`.
    pub wb: BitVec,
    /// Lines with `d = 1`.
    pub o1: BitVec,
    /// Lines with `d = 3`.
    pub o3: BitVec,
    /// `w_e[l] = (d(even[l]) / 2) mod 2` (`d(even[l]) = 2 w_e[l]`).
    pub we: BitVec,
    /// `#(d=1) − #(d=3)` among the lines.
    pub sigma: i32,
}

impl Structure {
    /// Number of hyperbolic planes `g`.
    pub fn g(&self) -> usize {
        self.pairs_a.len()
    }
    /// Rank of the Gram matrix `2g + #odd`.
    pub fn rank(&self) -> usize {
        2 * self.g() + self.odd.len()
    }

    /// Compute the block-diagonalising basis of `Ā = B + diag(parity)`.
    ///
    /// `b` must be a symmetric zero-diagonal `m × m` matrix; `parity[i] = v_i mod 2`.
    ///
    /// With [`PivotRule::MinWeight`] the pivot is the lightest admissible row, found in
    /// `O(log m)` from two ordered sets that are updated only for the rows a pivot touches.
    /// For the sparse, local graphs produced by circuits this makes the whole elimination
    /// cost `O(Σ_pivots |S|·m/64)` (proportional to *fill*), instead of the `O(m²·m/64)` of
    /// a naive pivot scan — see [`crate`] docs for measurements.  Large neighbourhoods are
    /// updated in parallel with rayon.
    pub fn new(b: &BitMat, parity: &BitVec, rule: PivotRule) -> Structure {
        let m = b.nrows();
        assert_eq!(b.ncols(), m);
        assert_eq!(parity.len(), m);
        let mut gram: Vec<BitVec> = (0..m)
            .map(|i| {
                let mut r = b.row(i).clone();
                r.set(i, parity.get(i));
                r
            })
            .collect();
        let mut basis: Vec<BitVec> = (0..m).map(|i| BitVec::from_ones(m, [i])).collect();
        let mut alive = BitVec::from_ones(m, 0..m);
        let (mut pairs_a, mut pairs_b, mut odd, mut even) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());

        match rule {
            PivotRule::First => loop {
                let a = alive.iter_ones().find(|&a| gram[a].get(a));
                if let Some(a) = a {
                    let (_, pa) = single_step(&mut gram, &mut basis, &mut alive, a);
                    odd.push(pa);
                    continue;
                }
                let a = alive.iter_ones().find(|&a| gram[a].and_count(&alive) > 0);
                let Some(a) = a else { break };
                let (_, pa, pb, _) = hyper_step(&mut gram, &mut basis, &mut alive, a);
                pairs_a.push(pa);
                pairs_b.push(pb);
            },
            PivotRule::MinWeight => {
                // class: 0 = none (radical so far), 1 = has diagonal, 2 = off-diagonal only
                let mut key: Vec<(u8, u32)> = vec![(0, 0); m];
                let mut diag_set: BTreeSet<(u32, usize)> = BTreeSet::new();
                let mut off_set: BTreeSet<(u32, usize)> = BTreeSet::new();
                let reclass = |a: usize,
                               gram: &Vec<BitVec>,
                               alive: &BitVec,
                               key: &mut Vec<(u8, u32)>,
                               diag_set: &mut BTreeSet<(u32, usize)>,
                               off_set: &mut BTreeSet<(u32, usize)>| {
                    match key[a].0 {
                        1 => {
                            diag_set.remove(&(key[a].1, a));
                        }
                        2 => {
                            off_set.remove(&(key[a].1, a));
                        }
                        _ => {}
                    }
                    if !alive.get(a) {
                        key[a] = (0, 0);
                        return;
                    }
                    let w = gram[a].and_count(alive);
                    if gram[a].get(a) {
                        key[a] = (1, w);
                        diag_set.insert((w, a));
                    } else if w > 0 {
                        key[a] = (2, w);
                        off_set.insert((w, a));
                    } else {
                        key[a] = (0, 0);
                    }
                };
                for a in 0..m {
                    reclass(a, &gram, &alive, &mut key, &mut diag_set, &mut off_set);
                }
                loop {
                    if let Some(&(_, a)) = diag_set.iter().next() {
                        let (s, pa) = single_step(&mut gram, &mut basis, &mut alive, a);
                        odd.push(pa);
                        reclass(a, &gram, &alive, &mut key, &mut diag_set, &mut off_set);
                        for bi in s.iter_ones() {
                            reclass(bi, &gram, &alive, &mut key, &mut diag_set, &mut off_set);
                        }
                        continue;
                    }
                    if let Some(&(_, a)) = off_set.iter().next() {
                        let (touched, pa, pb, partner) = hyper_step(&mut gram, &mut basis, &mut alive, a);
                        pairs_a.push(pa);
                        pairs_b.push(pb);
                        reclass(a, &gram, &alive, &mut key, &mut diag_set, &mut off_set);
                        reclass(partner, &gram, &alive, &mut key, &mut diag_set, &mut off_set);
                        for k in touched.iter_ones() {
                            reclass(k, &gram, &alive, &mut key, &mut diag_set, &mut off_set);
                        }
                        continue;
                    }
                    break;
                }
            }
        }
        for r in alive.iter_ones() {
            even.push(basis[r].clone());
        }
        Structure { m, pairs_a, pairs_b, odd, even }
    }

    /// `d(p) = q(p) mod 4` for a basis vector, given Z4 bit-planes of `v`.
    fn z4_value(b: &BitMat, v0: &BitVec, v1: &BitVec, p: &BitVec) -> u8 {
        let lin = p.and_count(v0) + 2 * p.and_count(v1);
        let mut t: u32 = 0;
        for j in p.iter_ones() {
            t += b.row(j).and_count(p);
        }
        debug_assert!(t % 2 == 0);
        let edges_parity = (t / 2) & 1;
        ((lin + 2 * edges_parity) & 3) as u8
    }

    /// Compute the Z4 diagonal `d` of the basis for the linear part `v`.
    ///
    /// `d(p) = Σ_{j∈p} v_j + 2·#{edges of B inside supp p}  (mod 4)`, evaluated with
    /// popcounts on the two bit-planes of `v`.
    pub fn diagonal(&self, b: &BitMat, v: &[u8]) -> Diag {
        assert_eq!(v.len(), self.m);
        let mut v0 = BitVec::new(self.m);
        let mut v1 = BitVec::new(self.m);
        for (i, &x) in v.iter().enumerate() {
            v0.set(i, x & 1 == 1);
            v1.set(i, x & 2 == 2);
        }
        let g = self.g();
        let (mut wa, mut wb) = (BitVec::new(g), BitVec::new(g));
        for j in 0..g {
            let da = Self::z4_value(b, &v0, &v1, &self.pairs_a[j]);
            let db = Self::z4_value(b, &v0, &v1, &self.pairs_b[j]);
            debug_assert!(da % 2 == 0 && db % 2 == 0, "hyperbolic plane with odd diagonal");
            wa.set(j, da == 2);
            wb.set(j, db == 2);
        }
        let (mut o1, mut o3) = (BitVec::new(self.odd.len()), BitVec::new(self.odd.len()));
        let mut sigma = 0i32;
        for (l, p) in self.odd.iter().enumerate() {
            let d = Self::z4_value(b, &v0, &v1, p);
            debug_assert!(d % 2 == 1, "non-isotropic line with even diagonal");
            if d == 1 {
                o1.set(l, true);
                sigma += 1;
            } else {
                o3.set(l, true);
                sigma -= 1;
            }
        }
        let mut we = BitVec::new(self.even.len());
        for (l, p) in self.even.iter().enumerate() {
            let d = Self::z4_value(b, &v0, &v1, p);
            debug_assert!(d % 2 == 0, "radical vector with odd diagonal");
            we.set(l, d == 2);
        }
        Diag { wa, wb, o1, o3, we, sigma }
    }

    /// Coordinates `u_l = <p_l, x> mod 2` of the Z-type shift vector `x` (so that the
    /// shift `(-1)^{x·z}` becomes `d_l → d_l + 2 u_l`).
    pub fn transform(&self, x: &BitVec) -> UVec {
        let dots = |ps: &[BitVec]| {
            let mut out = BitVec::new(ps.len());
            for (i, p) in ps.iter().enumerate() {
                if p.dot(x) {
                    out.set(i, true);
                }
            }
            out
        };
        UVec { a: dots(&self.pairs_a), b: dots(&self.pairs_b), o: dots(&self.odd), e: dots(&self.even) }
    }

    /// Column vectors of the basis matrix, **only for the variables in `needed`**
    /// (others get an empty `UVec`): `cols[j]` holds the coordinates `(p_l[j])_l`, i.e. the
    /// shift produced by a Z-type perturbation of the single variable `j`.  Then
    /// `Pᵀ x = ⊕_{j ∈ supp x} cols[j]`, cheap whenever `x` is sparse (boundary columns and
    /// branch terms always are).  Cost: one masked pass over the basis, `O(m · m/64 + hits)`.
    pub fn column_vectors_for(&self, needed: &BitVec) -> Vec<UVec> {
        let empty = || UVec { a: BitVec::new(0), b: BitVec::new(0), o: BitVec::new(0), e: BitVec::new(0) };
        let mut cols: Vec<UVec> = (0..self.m).map(|_| empty()).collect();
        for j in needed.iter_ones() {
            cols[j] = UVec::zero(self);
        }
        let mut scan = |ps: &[BitVec], put: &mut dyn FnMut(&mut UVec, usize)| {
            for (l, p) in ps.iter().enumerate() {
                let mut hit = p.clone();
                hit.and_assign(needed);
                for j in hit.iter_ones() {
                    put(&mut cols[j], l);
                }
            }
        };
        scan(&self.pairs_a, &mut |c, l| c.a.set(l, true));
        scan(&self.pairs_b, &mut |c, l| c.b.set(l, true));
        scan(&self.odd, &mut |c, l| c.o.set(l, true));
        scan(&self.even, &mut |c, l| c.e.set(l, true));
        cols
    }

    /// The single column `(p_l[j])_l` (cost `O(m)` bit tests); for variables not covered by
    /// [`column_vectors_for`](Self::column_vectors_for).
    pub fn column_of(&self, j: usize) -> UVec {
        let mut u = UVec::zero(self);
        for (l, p) in self.pairs_a.iter().enumerate() {
            u.a.set(l, p.get(j));
        }
        for (l, p) in self.pairs_b.iter().enumerate() {
            u.b.set(l, p.get(j));
        }
        for (l, p) in self.odd.iter().enumerate() {
            u.o.set(l, p.get(j));
        }
        for (l, p) in self.even.iter().enumerate() {
            u.e.set(l, p.get(j));
        }
        u
    }

    /// `Pᵀ x` using precomputed [`column_vectors`](Self::column_vectors).
    pub fn transform_sparse(&self, cols: &[UVec], x: &BitVec) -> UVec {
        let mut u = UVec::zero(self);
        for j in x.iter_ones() {
            u.xor_assign(&cols[j]);
        }
        u
    }

    /// `log2` of the magnitude of the Gauss sum when it does not vanish:
    /// `g + #even + #odd/2`.
    pub fn log2_magnitude(&self) -> f64 {
        self.g() as f64 + self.even.len() as f64 + self.odd.len() as f64 / 2.0
    }

    /// Does the shifted sum survive (all radical factors non-zero)?  Equivalent to
    /// `u.e == d.we`.
    #[inline(always)]
    pub fn survives(d: &Diag, u: &UVec) -> bool {
        u.e == d.we
    }

    /// Phase of the (non-vanishing) Gauss sum as an index `k` meaning `e^{iπ k/4}`,
    /// `k ∈ 0..8`.  Assumes [`survives`](Self::survives).
    #[inline(always)]
    pub fn phase8(d: &Diag, u: &UVec) -> u8 {
        let pc1 = u.o.and_count(&d.o1) as i32;
        let pc3 = u.o.and_count(&d.o3) as i32;
        let mut n = d.sigma - 2 * (pc1 - pc3);
        // sign of Π_planes (-1)^{(w_a⊕u_a)(w_b⊕u_b)}
        let mut cnt = 0u32;
        let (wa, wb, ua, ub) = (d.wa.words(), d.wb.words(), u.a.words(), u.b.words());
        for i in 0..wa.len() {
            cnt += ((wa[i] ^ ua[i]) & (wb[i] ^ ub[i])).count_ones();
        }
        if cnt & 1 == 1 {
            n += 4;
        }
        n.rem_euclid(8) as u8
    }

    /// Full value of `Σ_x i^{q(x)} (-1)^{shift·x}` given the shift coordinates `u`.
    pub fn gauss_sum(&self, d: &Diag, u: &UVec) -> Complex64 {
        if !Self::survives(d, u) {
            return Complex64::new(0.0, 0.0);
        }
        crate::phase::ROOT8[Self::phase8(d, u) as usize] * 2f64.powf(self.log2_magnitude())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random_form(m: usize, density: f64, rng: &mut Rng) -> (BitMat, Vec<u8>) {
        let mut b = BitMat::new(m, m);
        for i in 0..m {
            for j in (i + 1)..m {
                if rng.next_f64() < density {
                    b.flip_sym(i, j);
                }
            }
        }
        let v: Vec<u8> = (0..m).map(|_| rng.below(4) as u8).collect();
        (b, v)
    }

    fn brute(b: &BitMat, v: &[u8], shift: &BitVec) -> Complex64 {
        let m = v.len();
        let mut tot = Complex64::new(0.0, 0.0);
        for x in 0u32..(1u32 << m) {
            let mut q = 0u32;
            for j in 0..m {
                if (x >> j) & 1 == 1 {
                    q += v[j] as u32;
                    if shift.get(j) {
                        q += 2;
                    }
                    for k in (j + 1)..m {
                        if (x >> k) & 1 == 1 && b.get(j, k) {
                            q += 2;
                        }
                    }
                }
            }
            tot += crate::phase::ROOT8[(2 * (q & 3)) as usize];
        }
        tot
    }

    fn gram_entry(b: &BitMat, parity: &BitVec, p: &BitVec, q: &BitVec) -> bool {
        let mut acc = false;
        for j in p.iter_ones() {
            acc ^= b.row(j).dot(q);
            acc ^= parity.get(j) && q.get(j);
        }
        acc
    }

    #[test]
    fn basis_block_diagonalises_gram() {
        let mut rng = Rng::new(1);
        for trial in 0..200 {
            let m = 1 + rng.below(40);
            let (b, v) = random_form(m, [0.05, 0.2, 0.5][trial % 3], &mut rng);
            let parity = BitVec::from_ones(m, (0..m).filter(|&i| v[i] & 1 == 1));
            for rule in [PivotRule::First, PivotRule::MinWeight] {
                let s = Structure::new(&b, &parity, rule);
                assert_eq!(2 * s.g() + s.odd.len() + s.even.len(), m);
                // list all vectors with a type tag and check the Gram matrix
                let mut all: Vec<(&BitVec, u8, usize)> = Vec::new();
                for (j, p) in s.pairs_a.iter().enumerate() {
                    all.push((p, 0, j));
                }
                for (j, p) in s.pairs_b.iter().enumerate() {
                    all.push((p, 1, j));
                }
                for (j, p) in s.odd.iter().enumerate() {
                    all.push((p, 2, j));
                }
                for (j, p) in s.even.iter().enumerate() {
                    all.push((p, 3, j));
                }
                for &(p, tp, ip) in &all {
                    for &(q, tq, iq) in &all {
                        let expect = match (tp, tq) {
                            (0, 1) | (1, 0) => ip == iq,
                            (2, 2) => ip == iq,
                            _ => false,
                        };
                        assert_eq!(gram_entry(&b, &parity, p, q), expect, "types {tp},{tq} idx {ip},{iq}");
                    }
                }
                // the basis must be invertible: rank of the stacked matrix is m
                let mut rows: Vec<BitVec> = all.iter().map(|x| x.0.clone()).collect();
                let mut rank = 0;
                for c in 0..m {
                    if let Some(r) = (rank..rows.len()).find(|&r| rows[r].get(c)) {
                        rows.swap(rank, r);
                        let piv = rows[rank].clone();
                        for r2 in 0..rows.len() {
                            if r2 != rank && rows[r2].get(c) {
                                rows[r2].xor_assign(&piv);
                            }
                        }
                        rank += 1;
                    }
                }
                assert_eq!(rank, m);
            }
        }
    }

    #[test]
    fn gauss_sum_matches_brute_force() {
        let mut rng = Rng::new(2);
        for trial in 0..400 {
            let m = 1 + rng.below(11);
            let (b, v) = random_form(m, [0.1, 0.3, 0.6][trial % 3], &mut rng);
            let parity = BitVec::from_ones(m, (0..m).filter(|&i| v[i] & 1 == 1));
            let shift = BitVec::from_ones(m, (0..m).filter(|_| rng.next_f64() < 0.4));
            for rule in [PivotRule::First, PivotRule::MinWeight] {
                let s = Structure::new(&b, &parity, rule);
                let d = s.diagonal(&b, &v);
                let u = s.transform(&shift);
                let got = s.gauss_sum(&d, &u);
                let want = brute(&b, &v, &shift);
                assert!((got - want).norm() < 1e-9, "trial {trial} m={m}: got {got} want {want}");
            }
        }
    }

    #[test]
    fn alternating_and_nonalternating_cases() {
        // all-even v => alternating Gram (pure planes + radical); all-odd => lines appear
        let mut rng = Rng::new(3);
        for _ in 0..100 {
            let m = 2 + rng.below(9);
            let (b, mut v) = random_form(m, 0.4, &mut rng);
            for x in v.iter_mut() {
                *x &= 2;
            }
            let parity = BitVec::new(m);
            let s = Structure::new(&b, &parity, PivotRule::MinWeight);
            assert!(s.odd.is_empty());
            let d = s.diagonal(&b, &v);
            let z = BitVec::new(m);
            let got = s.gauss_sum(&d, &s.transform(&z));
            assert!((got - brute(&b, &v, &z)).norm() < 1e-9);
        }
    }
}
