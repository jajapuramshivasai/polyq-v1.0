//! Weak simulation of Clifford circuits **from the cached normal form**.
//!
//! The output distribution of a Clifford circuit on a basis-state input is uniform on an
//! affine subspace of `F2^n`.  With the cached normal form this subspace is the solution
//! set of one linear system ([`Engine::affine_support`]), so after a one-off
//! `O(m³/64)` analysis each shot costs `O(dim · n / 64)` word operations — the same
//! "analyse once, then sample cheaply" shape as Stim's reference-sample + frame approach,
//! but derived from the path-sum.
//!
//! Only qubits listed in `record` are reported (a marginal of a uniform affine
//! distribution is again uniform on the projected subspace, so projecting full samples is
//! exact).

use crate::bits::BitVec;
use crate::circuit::Circuit;
use crate::engine::{AffineSpace, Engine, EngineOptions};
use crate::pathsum::PathSum;
use crate::rng::Rng;

/// Sampler for the computational-basis measurement distribution of a Clifford circuit.
pub struct CliffordSampler {
    record: Vec<usize>,
    /// Output bit of wire `q` = `base[q] ⊕ <coef[q], r>` for `r` uniform on `F2^dim`.
    rows: Vec<BitVec>,
    consts: BitVec,
    dim: usize,
    /// `log2` of the support size (the entropy of the distribution in bits).
    pub support_dim: usize,
}

impl CliffordSampler {
    /// Analyse a Clifford circuit on input `input` (default all-zero when `None`).
    /// Returns `Err` if the circuit contains non-Clifford rotations.
    pub fn new(c: &Circuit, input: Option<&[bool]>, record: Option<Vec<usize>>) -> Result<Self, String> {
        let n = c.num_qubits;
        let ps = PathSum::from_circuit(c);
        if ps.rotation_count() != 0 {
            return Err("circuit contains non-Clifford rotations".into());
        }
        let zero = vec![false; n];
        let input = input.unwrap_or(&zero);
        let fixed = ps.io_vars();
        let eng = Engine::new(&ps, &fixed, EngineOptions::default());
        // known: inputs; unknown: output variables that are not inputs
        let mut known: Vec<(usize, bool)> = (0..n).map(|q| (q, input[q])).collect();
        let mut unknown: Vec<usize> = Vec::new();
        for &o in &ps.outputs {
            if o >= n && !unknown.contains(&o) {
                unknown.push(o);
            }
        }
        // outputs that coincide with an input variable are forced
        let mut forced: Vec<Option<bool>> = vec![None; n];
        for q in 0..n {
            let o = ps.outputs[q];
            if o < n {
                forced[q] = Some(input[o] ^ ps.out_neg[q]);
            }
        }
        known.sort();
        let sp: AffineSpace = eng.affine_support(&known, &unknown).ok_or("empty support (unnormalised circuit?)")?;
        let dim = sp.basis.len();
        let upos = |o: usize| unknown.iter().position(|&x| x == o).unwrap();
        // wire bit = z_out ^ neg ; z_out = origin[pos] ^ <basis columns, r>
        let mut rows = vec![BitVec::new(dim); n];
        let mut consts = BitVec::new(n);
        for q in 0..n {
            if let Some(f) = forced[q] {
                consts.set(q, f);
                continue;
            }
            let p = upos(ps.outputs[q]);
            consts.set(q, sp.origin.get(p) ^ ps.out_neg[q]);
            for (j, b) in sp.basis.iter().enumerate() {
                if b.get(p) {
                    rows[q].set(j, true);
                }
            }
        }
        let record = record.unwrap_or_else(|| (0..n).collect());
        Ok(Self { record, rows, consts, dim, support_dim: dim })
    }

    /// Number of reported bits per shot.
    pub fn num_recorded(&self) -> usize {
        self.record.len()
    }

    /// Draw `shots` samples; result is row-major `shots × record.len()` with one byte (0/1)
    /// per bit.
    pub fn sample(&self, shots: usize, seed: u64) -> Vec<u8> {
        use rayon::prelude::*;
        let k = self.record.len();
        let mut out = vec![0u8; shots * k];
        let chunk = 256usize;
        out.par_chunks_mut(chunk * k).enumerate().for_each(|(ci, buf)| {
            let mut rng = Rng::new(seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let words = (self.dim + 63) / 64;
            let mut r = BitVec::new(self.dim);
            for s in 0..(buf.len() / k) {
                // random coefficient vector
                let mut ws = vec![0u64; words.max(1)];
                for w in ws.iter_mut() {
                    *w = rng.next_u64();
                }
                if self.dim > 0 {
                    r = BitVec::from_ones(self.dim, (0..self.dim).filter(|&j| (ws[j >> 6] >> (j & 63)) & 1 == 1));
                }
                for (i, &q) in self.record.iter().enumerate() {
                    let bit = self.consts.get(q) ^ (self.dim > 0 && self.rows[q].dot(&r));
                    buf[s * k + i] = bit as u8;
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense;
    use crate::rng::{random_clifford_t, Rng};

    #[test]
    fn support_matches_dense_distribution() {
        let mut rng = Rng::new(77);
        for _ in 0..40 {
            let n = 2 + rng.below(5);
            let c = random_clifford_t(n, 30, 0, false, &mut rng);
            let input: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
            let sv = dense::statevector(&c, &input);
            let s = CliffordSampler::new(&c, Some(&input), None).unwrap();
            let shots = 3000;
            let out = s.sample(shots, 5);
            let mut hist = vec![0usize; 1 << n];
            for sh in 0..shots {
                let mut idx = 0usize;
                for q in 0..n {
                    idx |= (out[sh * n + q] as usize) << q;
                }
                hist[idx] += 1;
            }
            for y in 0..(1 << n) {
                let p = sv[y].norm_sqr();
                if p < 1e-12 {
                    assert_eq!(hist[y], 0, "sampled outside the support");
                } else {
                    let f = hist[y] as f64 / shots as f64;
                    assert!((f - p).abs() < 0.06, "y={y} p={p} f={f}");
                }
            }
        }
    }
}
