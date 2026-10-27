#!/usr/bin/env python3
"""Benchmark a zero-syndrome projection primitive for the [[15,1,3]] RM code.

The benchmark prepares 15 |A> = T|+> inputs and coherently projects onto the 14
stabilizer generators of the punctured Reed-Muller code. It reports the probability
that all syndrome ancillas are zero. **This is only a code-projection microbenchmark,
not a complete Bravyi-Kitaev 15-to-1 factory circuit.** A protocol-level benchmark
must add the correct encoding/decoding or gate-teleportation network, input noise
model, and output-frame convention before results can be called distillation results.

The CSS checks below are the [[15,1,3]] tableau published by the Error Correction
Zoo (stab_15_1_3).
Stim is not timed because its stabilizer simulator cannot represent the 15 injected
non-stabilizer |A> states. Qiskit Aer uses its matrix-product-state method so this
29-qubit check circuit does not require a 2^29 statevector allocation.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import resource
import sys
import time

import numpy as np

DATA = 15
Z_CHECKS = [
    (0, 1, 3, 6), (0, 1, 4, 5), (0, 1, 7, 10), (0, 1, 8, 9),
    (0, 1, 11, 14), (0, 1, 12, 13), (0, 2, 3, 5), (0, 2, 7, 9),
    (0, 2, 11, 13), (0, 3, 7, 12),
]
X_CHECKS = [
    (7, 8, 9, 10, 11, 12, 13, 14),
    (3, 4, 5, 6, 11, 12, 13, 14),
    (1, 2, 5, 6, 9, 10, 13, 14),
    (0, 2, 4, 6, 8, 10, 12, 14),
]


def make_qiskit_circuit():
    from qiskit import QuantumCircuit

    checks = Z_CHECKS + X_CHECKS
    qc = QuantumCircuit(DATA + len(checks))
    for q in range(DATA):
        qc.h(q)
        qc.t(q)
    for j, support in enumerate(Z_CHECKS):
        anc = DATA + j
        for q in support:
            qc.cx(q, anc)
    for j, support in enumerate(X_CHECKS, start=len(Z_CHECKS)):
        anc = DATA + j
        qc.h(anc)
        for q in support:
            qc.cx(anc, q)
        qc.h(anc)
    return qc


def projection_oracle() -> float:
    """Small independent 15-qubit oracle applying the 14 code projectors directly."""
    dim = 1 << DATA
    basis = np.arange(dim, dtype=np.uint32)
    # |A>^15, with |A> = T|+>.
    weights = np.fromiter((int(x).bit_count() for x in basis), dtype=np.int16, count=dim)
    state = np.exp(1j * np.pi / 4 * weights) / np.sqrt(dim)
    for checks, pauli in ((Z_CHECKS, "z"), (X_CHECKS, "x")):
        for support in checks:
            mask = sum(1 << q for q in support)
            if pauli == "z":
                signs = 1 - 2 * np.fromiter(
                    ((int(x & mask).bit_count() & 1) for x in basis), dtype=np.int8, count=dim
                )
                acted = signs * state
            else:
                acted = state[basis ^ mask]
            state = (state + acted) * 0.5
    return float(np.vdot(state, state).real)


def worker(name: str, reps: int, queue) -> None:
    try:
        qc = make_qiskit_circuit()
        ancillas = list(range(DATA, DATA + len(Z_CHECKS) + len(X_CHECKS)))
        if name == "polyq":
            import polyq
            from qiskit import transpile

            basis = ["h", "cx", "t"]
            lowered = transpile(qc, basis_gates=basis, optimization_level=0)
            circuit = polyq.Circuit(lowered.num_qubits)
            for item in lowered.data:
                op = item.operation.name.lower()
                qs = [lowered.find_bit(q).index for q in item.qubits]
                if op == "h":
                    circuit.h(qs[0])
                elif op == "t":
                    circuit.t(qs[0])
                elif op == "cx":
                    circuit.cx(qs[0], qs[1])
                else:
                    raise ValueError(f"unexpected gate after lowering: {op}")
            t0 = time.perf_counter()
            sim = polyq.Simulator(circuit, strategy="auto")
            setup_s = time.perf_counter() - t0
            times = []
            p_accept = None
            fixed = [(q, False) for q in ancillas]
            for _ in range(reps):
                t0 = time.perf_counter()
                p_accept = sim.marginal(0, fixed)
                times.append(time.perf_counter() - t0)
        elif name == "qiskit_aer_mps":
            from qiskit_aer import AerSimulator
            from qiskit import transpile

            t0 = time.perf_counter()
            backend = AerSimulator(method="matrix_product_state")
            compiled = transpile(qc, backend, optimization_level=0)
            compiled.save_probabilities(ancillas, label="syndrome_probabilities")
            setup_s = time.perf_counter() - t0
            times = []
            p_accept = None
            for _ in range(reps):
                t0 = time.perf_counter()
                result = backend.run(compiled, shots=1).result()
                times.append(time.perf_counter() - t0)
                p_accept = float(result.data(0)["syndrome_probabilities"][0])
        else:
            raise ValueError(name)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mib = rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
        queue.put({"backend": name, "setup_s": setup_s, "query_s_min": min(times),
                   "acceptance_probability": p_accept, "peak_rss_mib": rss_mib})
    except BaseException as exc:
        queue.put({"backend": name, "error": f"{type(exc).__name__}: {exc}"})


def run(name: str, reps: int) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=worker, args=(name, reps, queue))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        return {"backend": name, "error": f"worker exited with status {proc.exitcode}"}
    return queue.get()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be positive")
    expected = projection_oracle()
    for backend in ("polyq", "qiskit_aer_mps"):
        result = run(backend, args.reps)
        if "acceptance_probability" in result:
            result["oracle_probability"] = expected
            result["oracle_abs_error"] = abs(result["acceptance_probability"] - expected)
            if result["oracle_abs_error"] > 1e-8:
                raise AssertionError(f"{backend} disagrees with projection oracle: {result}")
        print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
