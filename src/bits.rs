//! Word-packed bit vectors and bit matrices over GF(2).
//!
//! These are the workhorse containers of the simulator.  Everything that is a
//! "row of a symmetric F2 matrix" or "a subset of path variables" is a
//! [`BitVec`]; every dense F2 matrix is a [`BitMat`] (a `Vec<BitVec>` of rows).
//!
//! All bulk operations (`xor_assign`, `and_count`, ...) run word-at-a-time, so
//! an operation on `m` bits costs `O(m / 64)`.

/// A fixed-length bit vector packed into `u64` words (little-endian bit order:
/// bit `i` lives in word `i / 64` at position `i % 64`).
///
/// Invariant: bits at positions `>= len` in the last word are always zero, so
/// popcounts and equality are exact.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct BitVec {
    words: Vec<u64>,
    len: usize,
}

/// Scalar reference implementation of `popcount(a & b)`.
#[inline]
pub fn and_count_scalar(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

/// `popcount(a & b)` with runtime-dispatched SIMD.
#[inline]
pub fn and_count_words(a: &[u64], b: &[u64]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 16 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature checked above; slices have equal length.
            return unsafe { simd::and_count_avx2(a, b) };
        }
    }
    and_count_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;

    /// AVX2 popcount of `a & b`: per-byte lookup through `pshufb`, byte sums folded with
    /// `psadbw` into four 64-bit lanes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn and_count_avx2(a: &[u64], b: &[u64]) -> u32 {
        let n = a.len().min(b.len());
        let lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        let low = _mm256_set1_epi8(0x0f);
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 4 <= n {
            let x = _mm256_and_si256(_mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i), _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i));
            let lo = _mm256_and_si256(x, low);
            let hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low);
            let cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(cnt, zero));
            i += 4;
        }
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
        let mut total = (lanes[0] + lanes[1] + lanes[2] + lanes[3]) as u32;
        while i < n {
            total += (a[i] & b[i]).count_ones();
            i += 1;
        }
        total
    }
}

#[inline(always)]
fn nwords(len: usize) -> usize {
    (len + 63) / 64
}

impl BitVec {
    /// All-zero vector of `len` bits.
    pub fn new(len: usize) -> Self {
        Self { words: vec![0; nwords(len)], len }
    }

    /// Vector with exactly the listed positions set.
    pub fn from_ones(len: usize, ones: impl IntoIterator<Item = usize>) -> Self {
        let mut v = Self::new(len);
        for i in ones {
            v.set(i, true);
        }
        v
    }

    /// Number of bits.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` iff the vector has zero bits (not "all zeros", see [`is_zero`](Self::is_zero)).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read bit `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// Write bit `i`.
    #[inline(always)]
    pub fn set(&mut self, i: usize, val: bool) {
        debug_assert!(i < self.len);
        let (w, b) = (i >> 6, i & 63);
        if val {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
    }

    /// Toggle bit `i`.
    #[inline(always)]
    pub fn flip(&mut self, i: usize) {
        debug_assert!(i < self.len);
        self.words[i >> 6] ^= 1u64 << (i & 63);
    }

    /// Raw words (read only).
    #[inline(always)]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// `self ^= other`.
    #[inline(always)]
    pub fn xor_assign(&mut self, other: &BitVec) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a ^= *b;
        }
    }

    /// `self &= other`.
    #[inline(always)]
    pub fn and_assign(&mut self, other: &BitVec) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= *b;
        }
    }

    /// `self &= !other`.
    #[inline(always)]
    pub fn andnot_assign(&mut self, other: &BitVec) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a &= !*b;
        }
    }

    /// Population count.
    #[inline(always)]
    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Population count of `self & other` without materialising the AND.
    ///
    /// Uses an AVX2 `pshufb` nibble-table popcount for long vectors when the CPU supports
    /// it (runtime detected), otherwise the scalar `popcnt` loop.
    #[inline]
    pub fn and_count(&self, other: &BitVec) -> u32 {
        debug_assert_eq!(self.len, other.len);
        and_count_words(&self.words, &other.words)
    }

    /// Parity of `<self, other>` over GF(2), i.e. `popcount(self & other) mod 2`.
    #[inline(always)]
    pub fn dot(&self, other: &BitVec) -> bool {
        debug_assert_eq!(self.len, other.len);
        if self.words.len() >= 16 {
            return and_count_words(&self.words, &other.words) & 1 == 1;
        }
        let mut acc = 0u64;
        for (a, b) in self.words.iter().zip(&other.words) {
            acc ^= a & b;
        }
        acc.count_ones() & 1 == 1
    }

    /// `true` iff every bit is zero.
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Index of the lowest set bit, if any.
    pub fn first_one(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate() {
            if w != 0 {
                return Some(wi * 64 + w.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Iterate over indices of set bits in increasing order.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    None
                } else {
                    let t = w.trailing_zeros() as usize;
                    w &= w - 1;
                    Some(wi * 64 + t)
                }
            })
        })
    }

    /// Collect set-bit indices.
    pub fn ones(&self) -> Vec<usize> {
        self.iter_ones().collect()
    }

    /// Overwrite with `other` (same length) without reallocating.
    #[inline(always)]
    pub fn copy_from(&mut self, other: &BitVec) {
        debug_assert_eq!(self.len, other.len);
        self.words.copy_from_slice(&other.words);
    }

    /// Zero every bit.
    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }
}

/// A dense `rows x cols` matrix over GF(2), stored row-major as [`BitVec`]s.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BitMat {
    rows: Vec<BitVec>,
    ncols: usize,
}

impl BitMat {
    /// Zero matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { rows: (0..nrows).map(|_| BitVec::new(ncols)).collect(), ncols }
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Read entry `(i, j)`.
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> bool {
        self.rows[i].get(j)
    }

    /// Write entry `(i, j)`.
    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, v: bool) {
        self.rows[i].set(j, v);
    }

    /// Borrow row `i`.
    #[inline(always)]
    pub fn row(&self, i: usize) -> &BitVec {
        &self.rows[i]
    }

    /// Mutably borrow row `i`.
    #[inline(always)]
    pub fn row_mut(&mut self, i: usize) -> &mut BitVec {
        &mut self.rows[i]
    }

    /// Toggle the symmetric pair `(i,j)` and `(j,i)`.
    #[inline(always)]
    pub fn flip_sym(&mut self, i: usize, j: usize) {
        self.rows[i].flip(j);
        self.rows[j].flip(i);
    }

    /// Append `extra` zero rows **and** zero columns (square/symmetric growth).
    pub fn grow_square(&mut self, new_n: usize) {
        debug_assert!(new_n >= self.rows.len());
        let old = self.rows.len();
        let mut fresh = BitMat::new(new_n, new_n);
        for i in 0..old {
            for j in self.rows[i].iter_ones() {
                fresh.rows[i].set(j, true);
            }
        }
        *self = fresh;
    }

    /// `y = A x` over GF(2) (`x` has `ncols` bits, `y` has `nrows` bits).
    pub fn mul_vec(&self, x: &BitVec) -> BitVec {
        let mut y = BitVec::new(self.nrows());
        for (i, r) in self.rows.iter().enumerate() {
            if r.dot(x) {
                y.set(i, true);
            }
        }
        y
    }

    /// Return the sub-matrix with the given row and column index lists.
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> BitMat {
        let mut out = BitMat::new(rows.len(), cols.len());
        for (ri, &r) in rows.iter().enumerate() {
            for (ci, &c) in cols.iter().enumerate() {
                if self.get(r, c) {
                    out.set(ri, ci, true);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitvec_basic() {
        let mut v = BitVec::new(130);
        v.set(0, true);
        v.set(64, true);
        v.set(129, true);
        assert_eq!(v.count_ones(), 3);
        assert_eq!(v.ones(), vec![0, 64, 129]);
        v.flip(64);
        assert_eq!(v.ones(), vec![0, 129]);
        let w = BitVec::from_ones(130, [129, 5]);
        assert!(v.dot(&w));
        assert_eq!(v.and_count(&w), 1);
        v.xor_assign(&w);
        assert_eq!(v.ones(), vec![0, 5]);
        assert_eq!(v.first_one(), Some(0));
    }

    #[test]
    fn simd_popcount_matches_scalar() {
        let mut x = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for len in [0usize, 1, 3, 4, 15, 16, 17, 63, 64, 125, 1000] {
            let a: Vec<u64> = (0..len).map(|_| next()).collect();
            let b: Vec<u64> = (0..len).map(|_| next()).collect();
            assert_eq!(and_count_words(&a, &b), and_count_scalar(&a, &b), "len {len}");
        }
    }

    #[test]
    #[ignore]
    fn simd_popcount_speed() {
        let a: Vec<u64> = (0..125u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let b: Vec<u64> = (0..125u64).map(|i| (i + 7).wrapping_mul(0xBF58_476D_1CE4_E5B9)).collect();
        for (name, f) in [("scalar", and_count_scalar as fn(&[u64], &[u64]) -> u32), ("dispatch", and_count_words)] {
            let t = std::time::Instant::now();
            let mut acc = 0u64;
            for _ in 0..2_000_000 {
                acc += f(std::hint::black_box(&a), std::hint::black_box(&b)) as u64;
            }
            println!("{name}: {:.1} ns/call (125 words) [{acc}]", t.elapsed().as_nanos() as f64 / 2e6);
        }
    }

    #[test]
    fn bitmat_mulvec() {
        let mut a = BitMat::new(3, 70);
        a.set(0, 69, true);
        a.set(1, 0, true);
        a.set(1, 69, true);
        let x = BitVec::from_ones(70, [69]);
        let y = a.mul_vec(&x);
        assert_eq!(y.ones(), vec![0, 1]);
    }
}
