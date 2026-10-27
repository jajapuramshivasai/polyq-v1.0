//! Python bindings (feature `python`).  Bulk data crosses the boundary as raw bytes
//! (`uint8` bit matrices, `complex128` vectors) so NumPy can wrap it with `frombuffer`
//! without extra copies on the Rust side.

use crate::circuit::{Circuit as RCircuit, Gate};
use crate::clifford::CliffordSampler as RSampler;
use crate::phase::Phase;
use crate::qasm::parse_qasm;
use crate::rng::Rng;
use crate::sim::{bits_of, equivalence as r_equiv, SimOptions, Simulator as RSim, Strategy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(name = "Circuit")]
#[derive(Clone)]
struct PyCircuit {
    inner: RCircuit,
}

#[pymethods]
impl PyCircuit {
    #[new]
    #[pyo3(signature = (num_qubits=0))]
    fn new(num_qubits: usize) -> Self {
        Self { inner: RCircuit::new(num_qubits) }
    }
    #[staticmethod]
    fn from_qasm(src: &str) -> PyResult<Self> {
        parse_qasm(src).map(|c| Self { inner: c }).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits
    }
    /// Drop `p(theta)` gates with `2|sin(theta/2)| < eps_each`; returns (circuit, error_bound).
    fn prune_small_rotations(&self, eps_each: f64) -> (PyCircuit, f64) {
        let a = crate::approx::prune_small_rotations(&self.inner, eps_each);
        (PyCircuit { inner: a.circuit }, a.error_bound)
    }
    /// Snap every `p(theta)` to a multiple of pi/2^log2_den; returns (circuit, error_bound).
    fn round_rotations(&self, log2_den: u32) -> (PyCircuit, f64) {
        let a = crate::approx::round_rotations(&self.inner, log2_den);
        (PyCircuit { inner: a.circuit }, a.error_bound)
    }
    fn to_qasm(&self) -> String {
        crate::qasm::write_qasm_string(&self.inner)
    }
    fn t_count(&self) -> usize {
        self.inner.t_count()
    }
    fn __len__(&self) -> usize {
        self.inner.gates.len()
    }
    fn h(&mut self, q: usize) {
        self.inner.h(q);
    }
    fn x(&mut self, q: usize) {
        self.inner.x(q);
    }
    fn y(&mut self, q: usize) {
        self.inner.y(q);
    }
    fn z(&mut self, q: usize) {
        self.inner.z(q);
    }
    fn s(&mut self, q: usize) {
        self.inner.s(q);
    }
    fn sdg(&mut self, q: usize) {
        self.inner.sdg(q);
    }
    fn t(&mut self, q: usize) {
        self.inner.t(q);
    }
    fn tdg(&mut self, q: usize) {
        self.inner.tdg(q);
    }
    /// `diag(1, e^{i theta})`, theta in radians (rounded to the 2^32 grid).
    fn p(&mut self, q: usize, theta: f64) {
        self.inner.p(q, Phase::from_radians(theta));
    }
    fn rz(&mut self, q: usize, theta: f64) {
        self.inner.rz(q, theta);
    }
    fn cx(&mut self, c: usize, t: usize) {
        self.inner.cx(c, t);
    }
    fn cz(&mut self, a: usize, b: usize) {
        self.inner.cz(a, b);
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.inner.swap(a, b);
    }
    fn ccx(&mut self, a: usize, b: usize, c: usize) {
        self.inner.ccx(a, b, c);
    }
    fn ccz(&mut self, a: usize, b: usize, c: usize) {
        self.inner.ccz(a, b, c);
    }
    /// Append gates from a flat list: `(name, qubits...)` encoded as `name` + int tuple.
    fn extend(&mut self, ops: Vec<(String, Vec<usize>)>) -> PyResult<()> {
        for (name, q) in ops {
            let g = match (name.as_str(), q.as_slice()) {
                ("H", [a]) => Gate::H(*a),
                ("X", [a]) => Gate::X(*a),
                ("Y", [a]) => Gate::Y(*a),
                ("Z", [a]) => Gate::Z(*a),
                ("S", [a]) => Gate::S(*a),
                ("SDG", [a]) => Gate::Sdg(*a),
                ("T", [a]) => Gate::T(*a),
                ("TDG", [a]) => Gate::Tdg(*a),
                ("CX", [a, b]) => Gate::CX(*a, *b),
                ("CZ", [a, b]) => Gate::CZ(*a, *b),
                ("SWAP", [a, b]) => Gate::Swap(*a, *b),
                _ => return Err(PyValueError::new_err(format!("bad op {name} {q:?}"))),
            };
            self.inner.push(g);
        }
        Ok(())
    }
}

fn parse_strategy(s: &str, pairs: Option<usize>) -> PyResult<Strategy> {
    match (s, pairs) {
        (_, Some(p)) => Ok(Strategy::Pairs(p)),
        ("auto", _) => Ok(Strategy::Auto),
        ("cached", _) => Ok(Strategy::Cached),
        _ => Err(PyValueError::new_err("strategy must be 'auto' or 'cached'")),
    }
}

#[pyclass(name = "Simulator")]
struct PySim {
    inner: RSim,
}

#[pymethods]
impl PySim {
    #[new]
    #[pyo3(signature = (circuit, strategy="auto", pairs=None))]
    fn new(circuit: &PyCircuit, strategy: &str, pairs: Option<usize>) -> PyResult<Self> {
        let st = parse_strategy(strategy, pairs)?;
        Ok(Self { inner: RSim::with_options(&circuit.inner, SimOptions { strategy: st, ..Default::default() }) })
    }
    fn amplitude(&self, input: u64, output: u64) -> (f64, f64) {
        let a = self.inner.amplitude_index(input, output);
        (a.re, a.im)
    }
    /// Unbiased Monte-Carlo amplitude estimate: (re, im, standard_error).
    fn amplitude_estimate(&self, py: Python<'_>, input: u64, output: u64, samples: usize, seed: u64) -> (f64, f64, f64) {
        let (v, se) = py.allow_threads(|| self.inner.amplitude_estimate(input, output, samples, seed));
        (v.re, v.im, se)
    }
    fn probability(&self, input: u64, output: u64) -> f64 {
        self.inner.amplitude_index(input, output).norm_sqr()
    }
    /// Statevector as raw little-endian `complex128` bytes.
    fn statevector<'py>(&self, py: Python<'py>, input: u64) -> Bound<'py, PyBytes> {
        let n = self.inner.num_qubits();
        let sv = py.allow_threads(|| self.inner.statevector(&bits_of(n, input)));
        let mut buf = Vec::with_capacity(sv.len() * 16);
        for a in sv {
            buf.extend_from_slice(&a.re.to_le_bytes());
            buf.extend_from_slice(&a.im.to_le_bytes());
        }
        PyBytes::new(py, &buf)
    }
    fn sample_many(&self, py: Python<'_>, input: u64, shots: usize, seed: u64) -> Vec<u64> {
        let n = self.inner.num_qubits();
        py.allow_threads(|| self.inner.sample_many(&bits_of(n, input), shots, seed))
    }
    fn marginal(&self, input: u64, measured: Vec<(usize, bool)>) -> f64 {
        let n = self.inner.num_qubits();
        self.inner.marginal_probability(&bits_of(n, input), &measured)
    }
    fn stats(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let s = self.inner.stats();
        (s.free_vars, s.planes, s.lines, s.radical, s.terms, s.branch_dim, self.inner.pairs())
    }
}

#[pyclass(name = "CliffordSampler")]
struct PySampler {
    inner: RSampler,
}

#[pymethods]
impl PySampler {
    #[new]
    #[pyo3(signature = (circuit, input=None, record=None))]
    fn new(py: Python<'_>, circuit: &PyCircuit, input: Option<u64>, record: Option<Vec<usize>>) -> PyResult<Self> {
        let n = circuit.inner.num_qubits;
        let inp = input.map(|i| bits_of(n, i));
        let c = circuit.inner.clone();
        let s = py.allow_threads(|| RSampler::new(&c, inp.as_deref(), record));
        s.map(|inner| Self { inner }).map_err(PyValueError::new_err)
    }
    #[getter]
    fn support_dim(&self) -> usize {
        self.inner.support_dim
    }
    #[getter]
    fn num_recorded(&self) -> usize {
        self.inner.num_recorded()
    }
    /// `shots * num_recorded` bytes (0/1), row-major.
    fn sample<'py>(&self, py: Python<'py>, shots: usize, seed: u64) -> Bound<'py, PyBytes> {
        let out = py.allow_threads(|| self.inner.sample(shots, seed));
        PyBytes::new(py, &out)
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, tol=1e-9))]
fn equivalent(a: &PyCircuit, b: &PyCircuit, tol: f64) -> (bool, f64, f64) {
    let r = r_equiv(&a.inner, &b.inner, tol, SimOptions::default());
    (r.equivalent, r.overlap.re, r.overlap.im)
}

#[pyfunction]
fn rng_u64(seed: u64) -> u64 {
    Rng::new(seed).next_u64()
}

#[pymodule]
fn polyq(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCircuit>()?;
    m.add_class::<PySim>()?;
    m.add_class::<PySampler>()?;
    m.add_function(wrap_pyfunction!(equivalent, m)?)?;
    m.add_function(wrap_pyfunction!(rng_u64, m)?)?;
    Ok(())
}
