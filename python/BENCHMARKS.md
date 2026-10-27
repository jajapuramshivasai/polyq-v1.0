# Application benchmarks

These scripts benchmark concrete quantities, not the broad labels “simulation” or
“surface-code performance.” Run the Rust release tests first, then install/build the
Python extension and optional comparison packages:

```sh
cargo test --release
pip install maturin numpy qiskit qiskit-aer
maturin develop --release
```

## 15-qubit RM projection primitive (not full distillation)

```sh
python python/bench_rm15_projection.py --reps 5
```

`bench_rm15_projection.py` creates the [[15,1,3]] punctured Reed–Muller code’s 10 Z- and 4
X-type stabilizer checks, prepares 15 ideal `T|+>` states, coherently extracts all
syndrome bits, and times the probability of the all-zero syndrome. It reports setup,
query time, and process peak RSS for PolyQ and Aer’s matrix-product-state backend.
The check supports are from the Error Correction Zoo’s `stab_15_1_3` entry; the X
and Z checks are verified to commute by the script’s construction test.

This is explicitly a **code-projection microbenchmark**, not a complete 15-to-1
factory benchmark. Its ideal-state projection probability is not the factory's
acceptance rate. A proper protocol benchmark still needs the encoding/decoding or
gate-teleportation network, declared noisy input states, and output logical-frame
fidelity calculation. Stim is not timed for this non-stabilizer input task. Use it
for the separate stabilizer-only sampling tests.

## Surface-code logical-state overlap

```sh
python python/bench_logical_overlap.py \
  --logical-prep 'circuits/logical_zero_{n}.qasm' \
  --target-prep 'circuits/target_{n}.qasm' \
  --sizes 50 98 162 242 --reps 3 --output overlap.jsonl
```

Each pair of QASM files must describe unitary state preparations on the same `N`
qubits. The script composes `U_logical† U_prep` and evaluates
`|<0^N|U_logical† U_prep|0^N>|² = |<0_L|U_prep|0^N>|²`. It records PolyQ analysis
and query times, peak RSS, and (when within the configured cutoff) an exact Aer
statevector cross-check. Aer is skipped above 28 qubits by default to avoid an
unintentional exponential allocation; raise the limit only when sufficient memory
is available. For N above 64, PolyQ’s Python integer basis-index interface still
supports the all-zero input/output used here.

This is **not** the same quantity as Stim’s shot-sampling performance. Stim has no
matching arbitrary complex-overlap timing in this benchmark. Use `bench_stim2.py`
for its separate detector/sample workload, and report the two tasks separately.
Circuit assets should state the code family, distance, qubit-count convention, and
gate-synthesis method. In particular, the suggested sizes 50, 98, 162, and 242
should be checked against whether `N` counts data qubits, ancillas, or both before
publishing results.

## Reproducibility notes

- Keep circuit generation and simulator setup separate from repeated query timings.
- Report versions, CPU, memory, thread settings, and the exact QASM assets.
- Cross-check small instances against Aer and an independent statevector or exact
  projection calculation before trusting large-instance timings.
- A missing optional backend is an environment/setup limitation, not a timing result.
