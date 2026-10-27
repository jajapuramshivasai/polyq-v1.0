# polyq

Phase-polynomial (path-sum) simulator for Clifford+T circuits in Rust. `cargo test --release` runs 31 tests plus 1 doc-test; every engine is checked against a dense statevector or a brute-force path sum.

## Setup and quick start

### Rust

Install the stable Rust toolchain from [rustup](https://rustup.rs/), then build and test
the library from the repository root:

```bash
cargo build --release
cargo test --release
```

Run the included benchmark binary:

```bash
cargo run --release --bin benchmark
```

Other useful binaries are available with the same pattern:

```bash
cargo run --release --bin approxbench
cargo run --release --bin svbench
cargo run --release --bin profile
```

The Rust API can be used from another Rust project by adding this repository as a
dependency, or directly from a small program:

```rust
use polyq::{Circuit, Simulator};

fn main() {
    let mut circuit = Circuit::new(2);
    circuit.h(0).cx(0, 1).t(1);

    let simulator = Simulator::new(&circuit);
    let amplitude = simulator.amplitude(&[false, false], &[true, true]);
    println!("{amplitude}");
}
```

### Python

Python support is provided through a PyO3 extension built by maturin. Python 3.9 or
newer and a Rust toolchain are required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install maturin numpy
python -m maturin develop --release
```

The repository also provides a Makefile for building both libraries and removing
repository-local build outputs and Python caches:

```bash
make build
make clean
```

If maturin is installed in a virtual environment that is not activated, select
its interpreter explicitly with `make PYTHON=.venv/bin/python build`.

The optional `stim` dependency is needed only for the Stim conversion benchmark and
its corresponding tests:

```bash
python -m pip install pytest stim
python -m pytest python/test_polyq.py
```

Example Python usage:

```python
import polyq

circuit = polyq.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.t(1)

simulator = polyq.Simulator(circuit, strategy="auto")
print(simulator.amplitude(0, 3))       # (real, imaginary)
print(simulator.probability(0, 3))
```

The Python module exposes `Circuit`, `Simulator`, `CliffordSampler`, and
`equivalent`. See [python/test_polyq.py](python/test_polyq.py) for complete examples.

## How the library works

PolyQ is a path-sum simulator. It does not primarily evolve a `2^n`-element
statevector gate by gate. Instead, it compiles a circuit into a Boolean exponential
sum, factorizes the reusable quadratic part once, and evaluates amplitudes or
probabilities from that cached representation.

### Repository structure

| Path | Role |
|---|---|
| [src/circuit.rs](src/circuit.rs) | Gate enum, circuit builder, inverse circuits, and CCX/CCZ expansions |
| [src/pathsum.rs](src/pathsum.rs) | Circuit-to-path-sum compiler and Pauli-fault handling |
| [src/phase.rs](src/phase.rs) | `Phase`, a `Z_2^32` dyadic phase ring |
| [src/normal_form.rs](src/normal_form.rs) | Symmetric F2 elimination and closed-form Z4 Gauss sums |
| [src/engine.rs](src/engine.rs) | Cached normal-form evaluation and affine branch enumeration |
| [src/rank.rs](src/rank.rs) | Exact pairwise stabilizer decomposition and strategy selection |
| [src/sim.rs](src/sim.rs) | Strong simulation, statevectors, marginals, sampling, and equivalence |
| [src/clifford.rs](src/clifford.rs) | Weak sampling from a Clifford affine subspace |
| [src/incremental.rs](src/incremental.rs) | Streaming compilation and engine reuse |
| [src/noise.rs](src/noise.rs) | Pauli-noise Monte Carlo using cached structure |
| [src/dense.rs](src/dense.rs) | Direct statevector reference implementation for tests and comparisons |
| [src/qasm.rs](src/qasm.rs) | Supported OpenQASM-2-style parser and writer |
| [src/pybind.rs](src/pybind.rs) | Optional Python bindings |

### The simulation pipeline

For an amplitude `⟨y|C|x⟩`, the compiler in [src/pathsum.rs](src/pathsum.rs)
creates path variables for the circuit. Hadamards introduce variables; diagonal
gates add phase terms; `CZ` adds quadratic interactions; `CX` is represented as
`H-CZ-H`; and `X`/`SWAP` are handled as cheap wire transformations.

The resulting expression has the form:

```text
⟨y|C|x⟩ =
2^(-h/2) · e^(iφ)
· Σ_z i^(Q(z)) · e^(i Σ_k θ_k z_k)
```

Here `Q(z)` is a Z4 quadratic form:

```text
Q(z) = Σ_k v_k z_k + 2 Σ_{j<k} B_jk z_j z_k  (mod 4)
```

`B` is a symmetric binary matrix, `v` contains the Z4 linear coefficients, and
`theta` contains the remaining non-Clifford rotations.

1. **Compile.** `PathSum::from_circuit` records the quadratic matrix, linear phases,
   rotation terms, global phase, live variables, and input/output variables.
2. **Separate fixed and free variables.** Input and output values are boundary
   variables. Internal path variables remain summed.
3. **Factorize once.** `Structure::new` in [src/normal_form.rs](src/normal_form.rs)
   block-diagonalizes the free-variable Gram matrix using word-parallel symmetric
   elimination. It produces odd one-dimensional blocks, hyperbolic two-dimensional
   blocks, and radical vectors.
4. **Evaluate the Clifford part.** The block form turns the exponential sum into
   independent factors. Their product is a closed-form Gauss sum computed with
   popcounts and eighth roots of unity; no full enumeration of free variables is
   needed.
5. **Reuse boundary assignments.** Changing `x` or `y` changes only a constant phase
   and a Z-type linear shift. The cached basis is transformed once, so the expensive
   elimination is not repeated for every amplitude.
6. **Handle non-Clifford rotations.** Each remaining factor is decomposed as
   `e^(iθz) = α + β(-1)^z`. The radical coordinates impose a binary linear system,
   so [src/engine.rs](src/engine.rs) enumerates only the surviving affine subspace
   of branches. Gray-code updates and Rayon parallelism reduce the per-branch cost.
7. **Optionally pair T rotations.** [src/rank.rs](src/rank.rs) can replace selected
   pairs of compatible rotations with two exact stabilizer branches. `strategy="auto"`
   or `Strategy::Auto` chooses the number of pairs with a cost model.

The main high-level entry point is `Simulator` in [src/sim.rs](src/sim.rs). It
provides amplitudes, probabilities, full statevectors, marginals, weak sampling,
and equivalence checking. `statevector` is parallel over output basis states, but
each output is evaluated through the cached path-sum engine.

### Specialized execution paths

* **Clifford sampling:** [src/clifford.rs](src/clifford.rs) analyzes a Clifford
  circuit once and samples uniformly from its resulting affine output subspace.
* **Streaming:** [src/incremental.rs](src/incremental.rs) rebinds the existing
  structure after diagonal gates, Paulis, `SWAP`, and global phase; gates that
  change the quadratic graph trigger a rebuild.
* **Pauli noise:** [src/noise.rs](src/noise.rs) samples fault patterns and reuses
  the fault-free structure whenever the path-sum structure is unchanged.
* **Reference simulation:** [src/dense.rs](src/dense.rs) applies gates directly to
  a statevector and is intended for validation and small circuits.

## What is new vs `polyq-core` v1
| Area | v1 | v2 |
|---|---|---|
| Phase ring | `u16`, units of π/2^16, so angles wrap at **π** (bug: `H·T·Z·H\|0⟩` gets the wrong sign; `T†` unrepresentable) | `u32`, units of 2π/2^32, exact for π·j/2^k with k ≤ 31 |
| Reduction | per-amplitude Dickson on Z4 vectors | one cached symmetric `LDLᵀ` over F2 (1×1 and 2×2 pivots, rank-profile style); Z4 diagonals by popcount; closed-form Gauss sum |
| Boundary / T branches | re-reduced per amplitude, 2^nv branches by conditioning | both are *shifts* of the cached form; Z-decomposition `e^{iθz}=α+β(-1)^z`; only the affine subspace `ker K` (dim `t−rank K`) is enumerated |
| Low-rank stabiliser | none | exact rank-2 decomposition of T-pairs (`[x=y]·i^x + [x≠y]·ω`), cost-model chooser |
| Gates | H Z S T RZ CZ | + X Y S† T† P(θ) CX SWAP CCX CCZ GPhase; X and SWAP are free |
| Extras | – | marginals, weak sampling, equivalence checking (`tr(C₂†C₁)`), Pauli-noise MC, streaming |

## Honest limits
* The elimination is word-parallel `O(m³/64)`, not a sub-cubic `O(m^ω)` recursion. The block structure is what such a recursion would produce, so it could be swapped in behind `Structure::new`.
* Worst case is still exponential in the non-Clifford count (the problem is #P-hard). On the small layered circuits in the benchmark, dense simulation is faster; the path-sum engine pays off for larger n with a modest effective branch dimension.
* Streaming only avoids re-elimination for diagonal non-Clifford gates, Paulis, SWAP and global phase. S, CZ, H and CX trigger a rebuild; a rank-1/2 Schur update is future work.
* Rotation angles are rounded to the 2^32 grid (error ≤ 7.3e-10 rad). Angles that are dyadic multiples of π are exact.
* The old `rz <int> q[i]` QASM syntax is not accepted. Use `p(expr)`, `rz(expr)`, `t`, and so on.

Run `cargo run --release --bin benchmark` for timings. Results from the sandbox:

| n | T | dense ms | cached ms | auto ms | d | pairs |
|---|---|---|---|---|---|---|
| 10 | 20 | 0.07 | 62.0 | 18.4 | 12 | 3 |
| 12 | 16 | 0.27 | 8.6 | 7.5 | 7 | 0 |

---
# v2.1 additions

## Python
```
pip install maturin numpy stim        # stim only for the benchmark
maturin develop --release             # builds the `polyq` module (feature "python")
python -m pytest python/test_polyq.py
```
`polyq.Circuit`, `polyq.Simulator(circuit, strategy="auto"|"cached", pairs=None)` (`amplitude`, `probability`, `statevector`, `sample_many`, `marginal`, `amplitude_estimate`), `polyq.CliffordSampler`, `polyq.equivalent`. `python/stim_convert.py` turns a noiseless Stim circuit into a polyq circuit by deferred measurement.

## What changed
* **Elimination.** Pivot choice is now an ordered set updated only for the rows a pivot touches. The old scan cost `O(m²)` popcounts per elimination; now the cost is proportional to fill. On the d=9 surface code (m=7984) the elimination went from 2.1 s to 23 ms. Large neighbourhoods are updated with rayon.
  **Not delivered: a worst-case ω<3 elimination.** Dense forms are still `O(m³/64)`. A recursive or Four-Russians version was not written; circuit forms are sparse, so it was not the bottleneck here.
* **Boundary/branch columns** use per-variable basis columns computed only for the variables that are needed (rep-code d=61: 1.4 s → 0.07 s).
* **SIMD.** AVX2 `pshufb` popcount with runtime detection: 108 → 37 ns for 125-word vectors, tested against scalar.
* **Threads.** Statevector is parallel over outputs with per-thread scratch buffers (1.97× on 2 cores). The amplitude sum is parallel over branch chunks.
* **Generalized pairing.** The exact pair identity is applied to any pair with `θ_a ± θ_b ≡ 0 (mod π/2)`, not only T/T†. A cost model picks the number of pairs.
* **Approximate tools.** `prune_small_rotations` and `round_rotations` have rigorous operator-norm bounds. `Engine::estimate`, `RankEngine::estimate` and `approx::estimate_amplitude` are unbiased Monte-Carlo estimators with standard errors. The pair-sampling estimator builds branches lazily, so nothing of size 2^p is stored. The standard error is a sample estimate and can understate heavy-tailed variance (one test run showed a 4.7σ miss).
* **Clifford weak sampler** (`clifford.rs`): analyse once, then each shot is a random point of an affine subspace.

## Benchmark vs Stim 1.16 (noiseless memory circuits, measurement records; 2-core sandbox)
Circuits come from `stim.Circuit.generated` (surface-code memory circuits are the Stim paper's Fig. 1 family; the repetition code is included as a second family). Measurements are converted by deferred measurement, so polyq's qubit count is larger. Every polyq sample was checked against Stim's detectors and observable (all satisfied).

| circuit | d | stim first sample | stim 100k (unpacked / bit-packed) | polyq analyse | polyq 100k (1 byte per bit) |
|---|---|---|---|---|---|
| rotated surface | 5 | 0.07 ms | 123 / 43 ms | 1.8 ms | 62 ms |
| rotated surface | 11 | 0.67 ms | 1700 / 478 ms | 168 ms | 554 ms |
| unrotated surface | 9 | 0.67 ms | 1731 / 415 ms | 169 ms | 622 ms |
| repetition | 31 | 0.15 ms | 1044 / 256 ms | 44 ms | 173 ms |
| repetition | 61 | 0.69 ms | 4165 / 1173 ms | 831 ms | 632 ms |

Reading: Stim is 100–1000× faster to *analyse* a circuit (about 1 ms vs 0.05–0.8 s) and so wins at small shot counts. At 100k shots polyq's sampling is 2–3× faster than Stim's unpacked output and roughly level with Stim's bit-packed output, but polyq's output is unpacked bytes. This benchmark covers noiseless circuits only: Stim also handles noise, detector sampling and mid-circuit measurement natively, and polyq does not (it relies on deferred measurement and has no noisy sampler beyond `NoisyEstimator`). polyq's strengths are strong simulation (exact amplitudes), non-Clifford gates, marginals and equivalence checking, none of which Stim offers. Reproduce with `python python/bench_stim2.py`.
