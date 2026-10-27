//! Tiny deterministic RNG (SplitMix64) plus random-circuit generators used by the
//! tests, benchmarks and Monte-Carlo routines.  No external dependency.

use crate::circuit::Circuit;
use crate::phase::Phase;

/// SplitMix64 pseudo random generator.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    /// Seeded generator.
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform integer in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Random circuit over `{H, S, Z, X, CZ, CX, SWAP}` plus exactly `t_count`
/// T/T† gates (and, if `generic_angles`, a few generic dyadic rotations).
pub fn random_clifford_t(n: usize, depth: usize, t_count: usize, generic_angles: bool, rng: &mut Rng) -> Circuit {
    let mut c = Circuit::new(n);
    let mut t_left = t_count;
    let total = depth.max(t_count);
    for step in 0..total {
        // sprinkle T gates evenly
        if t_left > 0 && (total - step <= t_left || rng.next_f64() < 0.3) {
            let q = rng.below(n);
            match rng.below(if generic_angles { 4 } else { 2 }) {
                0 => c.t(q),
                1 => c.tdg(q),
                2 => c.p(q, Phase::from_pi_fraction(1 + 2 * rng.below(7) as i64, 4)),
                _ => c.p(q, Phase::from_pi_fraction(1 + 2 * rng.below(3) as i64, 3)),
            };
            t_left -= 1;
            continue;
        }
        let q = rng.below(n);
        let r = rng.below(if n > 1 { 9 } else { 5 });
        match r {
            0 | 1 => c.h(q),
            2 => c.s(q),
            3 => c.z(q),
            4 => c.x(q),
            _ => {
                let mut q2 = rng.below(n);
                while q2 == q {
                    q2 = rng.below(n);
                }
                match r {
                    5 | 6 => c.cz(q, q2),
                    7 => c.cx(q, q2),
                    _ => c.swap(q, q2),
                }
            }
        };
    }
    c
}

/// Layered "H, T-layer, CZ" circuits: every qubit gets a Hadamard, then random `T`/`T†`
/// on a random subset, then random CZs; repeated `rounds` times.  Maximises the number of
/// distinct rotated path variables (worst case for the plain Z-branch engine).
pub fn random_layered(n: usize, rounds: usize, rng: &mut Rng) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..rounds {
        for q in 0..n {
            c.h(q);
        }
        for q in 0..n {
            match rng.below(3) {
                0 => c.t(q),
                1 => c.tdg(q),
                _ => c.s(q),
            };
        }
        if n > 1 {
            for _ in 0..n {
                let (a, b) = (rng.below(n), rng.below(n));
                if a != b {
                    c.cz(a, b);
                }
            }
        }
    }
    c
}
