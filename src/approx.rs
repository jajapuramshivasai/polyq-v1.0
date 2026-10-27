//! Approximate and sampled variants of the exact decompositions.
//!
//! | tool | what it trades | error control |
//! |------|----------------|---------------|
//! | [`prune_small_rotations`] | drops gates `P(θ)` with `‖P(θ)−I‖ = 2|sin(θ/2)|` below a threshold (shrinks the T-count) | **rigorous**: `‖U−U'‖ ≤ Σ dropped` (telescoping), hence every amplitude error ≤ the bound |
//! | [`round_rotations`] | snaps each `P(θ)` to the nearest multiple of `π/2^k` (k = 2 gives Clifford+T) | **rigorous**: `‖U−U'‖ ≤ Σ 2|sin(δ/2)|` |
//! | [`Engine::estimate`](crate::engine::Engine::estimate) | samples the surviving branch subspace instead of enumerating it | unbiased; standard error reported (statistical) |
//! | [`estimate_amplitude`] | samples *pair-decomposition branches* lazily (no `2^p` engines are stored) | unbiased; standard error reported (statistical) |
//!
//! The statistical estimators are exact in expectation but their variance can be large when
//! branches interfere destructively (typical for T-heavy circuits): always look at the
//! returned standard error.

use crate::circuit::{Circuit, Gate};
use crate::engine::{Engine, EngineOptions};
use crate::pathsum::PathSum;
use crate::phase::Phase;
use crate::rank::pair_candidates;
use crate::rng::Rng;
use num_complex::Complex64;
use rayon::prelude::*;

/// A circuit approximation with a rigorous operator-norm error bound.
#[derive(Clone, Debug)]
pub struct Approximation {
    /// The modified circuit.
    pub circuit: Circuit,
    /// Upper bound on `‖U − U'‖` (operator norm); bounds every amplitude error.
    pub error_bound: f64,
    /// Number of gates changed or removed.
    pub changed: usize,
}

/// `‖diag(1,e^{iθ}) − I‖ = 2|sin(θ/2)|`.
fn dev(theta: Phase) -> f64 {
    2.0 * (theta.to_radians() / 2.0).sin().abs()
}

/// Drop every `P(θ)` whose deviation from the identity is below `eps_each`.
pub fn prune_small_rotations(c: &Circuit, eps_each: f64) -> Approximation {
    let mut out = Circuit::new(c.num_qubits);
    let (mut bound, mut changed) = (0.0, 0);
    for g in &c.gates {
        if let Gate::P(_, th) = g {
            let d = dev(*th);
            if d < eps_each {
                bound += d;
                changed += 1;
                continue;
            }
        }
        out.push(g.clone());
    }
    Approximation { circuit: out, error_bound: bound, changed }
}

/// Round every `P(θ)` to the nearest multiple of `π / 2^log2_den` (`log2_den = 2` yields a
/// Clifford+T circuit; the result has `P(θ)` on the grid only).
pub fn round_rotations(c: &Circuit, log2_den: u32) -> Approximation {
    assert!((1..=30).contains(&log2_den));
    let step = 1u64 << (31 - log2_den); // grid spacing in Phase units (π/2^k)
    let mut out = Circuit::new(c.num_qubits);
    let (mut bound, mut changed) = (0.0, 0);
    for g in &c.gates {
        if let Gate::P(q, th) = g {
            let k = (th.0 as u64 + step / 2) / step;
            let snapped = Phase(((k * step) & 0xFFFF_FFFF) as u32);
            if snapped != *th {
                bound += dev(snapped - *th);
                changed += 1;
            }
            out.push(Gate::P(*q, snapped));
        } else {
            out.push(g.clone());
        }
    }
    Approximation { circuit: out, error_bound: bound, changed }
}

/// Estimate with standard error.
#[derive(Copy, Clone, Debug)]
pub struct AmplitudeEstimate {
    /// Estimated amplitude.
    pub value: Complex64,
    /// Standard error of the estimate (modulus scale).
    pub stderr: f64,
}

/// Lazily sampled pair-decomposition estimate of the amplitude with boundary `assignment`.
///
/// The exact amplitude is `Σ_{c ∈ {0,1}^p} A_c` over the `2^p` stabiliser branches of the
/// first `p` pairs of [`pair_candidates`].  Here `samples` branches are drawn uniformly, each
/// is built **on the fly** (substitutions + one elimination — nothing of size `2^p` is stored)
/// and evaluated exactly (or by [`Engine::estimate`] if its subspace dimension exceeds
/// `exact_dim_limit`); the estimate is `2^p · mean(A_c)`.  Works for `p` up to 62.
pub fn estimate_amplitude(
    ps: &PathSum,
    fixed: &[usize],
    assignment: &[(usize, bool)],
    pairs: usize,
    samples: usize,
    seed: u64,
    exact_dim_limit: usize,
) -> AmplitudeEstimate {
    let (cands, _) = pair_candidates(ps, fixed);
    let p = pairs.min(cands.len()).min(62);
    let chosen = &cands[..p];
    let opts = EngineOptions::default();
    let (sum, sq) = (0..samples)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut b = ps.clone();
            for &(a, bb) in chosen {
                b.substitute(a, bb, rng.next_u64() & 1 == 1);
            }
            let eng = Engine::new(&b, fixed, opts);
            let amp = if eng.stats().branch_dim <= exact_dim_limit {
                eng.eval_assignment(assignment)
            } else {
                let mut vals = vec![false; fixed.len()];
                for &(v, x) in assignment {
                    vals[fixed.iter().position(|&f| f == v).unwrap()] = x;
                }
                eng.estimate(&vals, 1 << 14, rng.next_u64()).0
            };
            (amp, amp.norm_sqr())
        })
        .reduce(|| (Complex64::new(0.0, 0.0), 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let n = samples as f64;
    let mean = sum / n;
    let var = (sq / n - mean.norm_sqr()).max(0.0);
    let scale = 2f64.powi(p as i32);
    AmplitudeEstimate { value: mean * scale, stderr: scale * (var / n).sqrt() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense;
    use crate::rng::{random_layered, Rng};
    use crate::sim::{bits_of, Simulator};

    #[test]
    fn pruning_bound_is_rigorous() {
        let mut rng = Rng::new(4);
        for _ in 0..30 {
            let n = 3;
            let mut c = random_layered(n, 2, &mut rng);
            for _ in 0..6 {
                c.p(rng.below(n), Phase::from_radians(0.001 * (1.0 + rng.next_f64() * 20.0)));
                c.h(rng.below(n));
            }
            let ap = prune_small_rotations(&c, 0.05);
            assert!(ap.changed > 0);
            let x = vec![false; n];
            let (a, b) = (dense::statevector(&c, &x), dense::statevector(&ap.circuit, &x));
            let err = a.iter().zip(&b).map(|(p, q)| (p - q).norm_sqr()).sum::<f64>().sqrt();
            assert!(err <= ap.error_bound + 1e-12, "state error {err} > bound {}", ap.error_bound);
            assert!(ap.circuit.t_count() < c.t_count());
        }
    }

    #[test]
    fn rounding_bound_is_rigorous() {
        let mut rng = Rng::new(9);
        for _ in 0..30 {
            let n = 3;
            let mut c = random_layered(n, 2, &mut rng);
            for _ in 0..5 {
                c.p(rng.below(n), Phase::from_radians(rng.next_f64() * 6.28));
                c.h(rng.below(n));
            }
            let ap = round_rotations(&c, 2);
            let x = vec![false; n];
            let (a, b) = (dense::statevector(&c, &x), dense::statevector(&ap.circuit, &x));
            let err = a.iter().zip(&b).map(|(p, q)| (p - q).norm_sqr()).sum::<f64>().sqrt();
            assert!(err <= ap.error_bound + 1e-12, "{err} > {}", ap.error_bound);
            // result is on the pi/4 grid
            for g in &ap.circuit.gates {
                if let Gate::P(_, th) = g {
                    assert!(th.eighth_root_index().is_some());
                }
            }
        }
    }

    #[test]
    fn subspace_monte_carlo_is_unbiased() {
        let mut rng = Rng::new(61);
        let mut tested = 0;
        for _ in 0..40 {
            let n = 4;
            let c = random_layered(n, 3, &mut rng);
            let sim = Simulator::new(&c);
            if sim.stats().branch_dim < 6 || sim.stats().branch_dim > 12 || sim.pairs() != 0 {
                continue;
            }
            tested += 1;
            let want = dense::statevector(&c, &vec![false; n]);
            let y = rng.below(1 << n) as u64;
            let (v, se) = sim.amplitude_estimate(0, y, 200_000, 3);
            assert!((v - want[y as usize]).norm() < 6.0 * se + 5e-3, "{v} ± {se} vs {}", want[y as usize]);
        }
        assert!(tested >= 3, "tested {tested}");
        let _ = bits_of;
    }

    #[test]
    fn lazy_pair_sampling_is_unbiased() {
        let mut rng = Rng::new(62);
        let mut tested = 0;
        for _ in 0..30 {
            let n = 3;
            let c = random_layered(n, 3, &mut rng);
            let ps = PathSum::from_circuit(&c);
            let fixed = ps.io_vars();
            let (cands, _) = pair_candidates(&ps, &fixed);
            if cands.len() < 3 || ps.live_vars().len() > 20 {
                continue;
            }
            tested += 1;
            let want = dense::statevector(&c, &vec![false; n]);
            let y = rng.below(1 << n);
            let bd = ps.boundary(&vec![false; n], &bits_of(n, y as u64)).unwrap();
            let est = estimate_amplitude(&ps, &fixed, &bd, 3, 4000, 5, 20);
            assert!((est.value - want[y]).norm() < 6.0 * est.stderr + 5e-3, "{} ± {} vs {}", est.value, est.stderr, want[y]);
        }
        assert!(tested >= 3);
    }
}
