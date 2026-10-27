//! Dense statevector reference simulator.
//!
//! This is the ground truth used by the test-suite and benchmarks.  It applies
//! every gate of the crate's gate set directly and costs `O(2^n)` memory, so it
//! is only meant for `n <= ~24`.

use crate::circuit::{Circuit, Gate};
use num_complex::Complex64;

/// Run `circuit` on the computational basis state `input` (`input[q]` is qubit `q`)
/// and return all `2^n` amplitudes; index bit `q` is qubit `q`.
pub fn statevector(circuit: &Circuit, input: &[bool]) -> Vec<Complex64> {
    let n = circuit.num_qubits;
    assert!(input.len() >= n);
    let mut sv = vec![Complex64::new(0.0, 0.0); 1usize << n];
    let mut idx = 0usize;
    for q in 0..n {
        if input[q] {
            idx |= 1 << q;
        }
    }
    sv[idx] = Complex64::new(1.0, 0.0);
    for g in &circuit.gates {
        apply(&mut sv, n, g);
    }
    sv
}

fn apply(sv: &mut [Complex64], n: usize, g: &Gate) {
    let dim = 1usize << n;
    let i_ = Complex64::new(0.0, 1.0);
    match *g {
        Gate::H(q) => {
            let m = 1 << q;
            let r = std::f64::consts::FRAC_1_SQRT_2;
            for i in 0..dim {
                if i & m == 0 {
                    let (a, b) = (sv[i], sv[i | m]);
                    sv[i] = (a + b) * r;
                    sv[i | m] = (a - b) * r;
                }
            }
        }
        Gate::X(q) => {
            let m = 1 << q;
            for i in 0..dim {
                if i & m == 0 {
                    sv.swap(i, i | m);
                }
            }
        }
        Gate::Y(q) => {
            // Y|0> = i|1>, Y|1> = -i|0>
            let m = 1 << q;
            for i in 0..dim {
                if i & m == 0 {
                    let (a, b) = (sv[i], sv[i | m]);
                    sv[i] = -i_ * b;
                    sv[i | m] = i_ * a;
                }
            }
        }
        Gate::Z(q) => diag1(sv, q, Complex64::new(-1.0, 0.0)),
        Gate::S(q) => diag1(sv, q, i_),
        Gate::Sdg(q) => diag1(sv, q, -i_),
        Gate::T(q) => diag1(sv, q, Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4)),
        Gate::Tdg(q) => diag1(sv, q, Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4)),
        Gate::P(q, p) => diag1(sv, q, Complex64::from_polar(1.0, p.to_radians())),
        Gate::CZ(a, b) => {
            let (ma, mb) = (1 << a, 1 << b);
            for i in 0..dim {
                if i & ma != 0 && i & mb != 0 {
                    sv[i] = -sv[i];
                }
            }
        }
        Gate::CX(c, t) => {
            let (mc, mt) = (1 << c, 1 << t);
            for i in 0..dim {
                if i & mc != 0 && i & mt == 0 {
                    sv.swap(i, i | mt);
                }
            }
        }
        Gate::Swap(a, b) => {
            let (ma, mb) = (1 << a, 1 << b);
            for i in 0..dim {
                if i & ma != 0 && i & mb == 0 {
                    sv.swap(i, (i ^ ma) | mb);
                }
            }
        }
        Gate::GPhase(p) => {
            let f = Complex64::from_polar(1.0, p.to_radians());
            sv.iter_mut().for_each(|a| *a *= f);
        }
    }
}

fn diag1(sv: &mut [Complex64], q: usize, ph: Complex64) {
    let m = 1 << q;
    for (i, a) in sv.iter_mut().enumerate() {
        if i & m != 0 {
            *a *= ph;
        }
    }
}
