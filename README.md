# polyq 

Phase-polynomial (path-sum) simulator for Clifford+T circuits in Rust. `cargo test --release` runs 31 tests plus 1 doc-test; every engine is checked against a dense statevector or a brute-force path sum.

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
