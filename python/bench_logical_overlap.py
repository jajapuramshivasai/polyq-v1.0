#!/usr/bin/env python3
"""Benchmark logical-state overlap from two unitary state-preparation QASM files.

The requested overlap is <0_L|U_prep|0^N>, where ``logical-prep.qasm`` prepares
|0_L> and ``target-prep.qasm`` is U_prep|0^N>.  We evaluate it as
<0^N|U_logical^dagger U_prep|0^N>, reporting its squared magnitude.  PolyQ and
Qiskit Aer receive the same composed circuit.  Stim is deliberately not timed
here: it is a stabilizer sampler/ simulator, not an equivalent general complex
overlap API.  Use the existing ``bench_stim2.py`` for its sampling workload.

Files may use ``{n}`` in their paths, e.g. ``--logical-prep circuits/logical_{n}.qasm``.
Qiskit is used only to parse/compose and lower the circuits.  Qiskit Aer is optional;
its statevector method is skipped above ``--aer-max-qubits`` to avoid accidental
exponential memory allocation.  PolyQ and Aer run in separate child processes so
their peak-RSS measurements are independent.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import load as load_qasm2

BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cz", "swap", "p", "rz"]


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSDs used by CI generally report KiB.
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def lower_to_polyq(qc: QuantumCircuit):
    import polyq

    lowered = transpile(qc, basis_gates=BASIS, optimization_level=0)
    out = polyq.Circuit(lowered.num_qubits)
    for item in lowered.data:
        op = item.operation
        name = op.name.lower()
        qs = [lowered.find_bit(q).index for q in item.qubits]
        if name in ("barrier", "id"):
            continue
        if name in ("h", "x", "y", "z", "s", "sdg", "t", "tdg"):
            getattr(out, name)(qs[0])
        elif name in ("cx", "cz", "swap"):
            getattr(out, name)(qs[0], qs[1])
        elif name in ("p", "rz"):
            getattr(out, name)(qs[0], float(op.params[0]))
        else:
            raise ValueError(f"PolyQ QASM lowering left unsupported gate {op.name!r}")
    # Global phase does not affect the reported squared overlap.
    return out


def worker(backend: str, qc: QuantumCircuit, reps: int, result_queue) -> None:
    try:
        if backend == "polyq":
            circuit = lower_to_polyq(qc)
            t0 = time.perf_counter()
            sim = __import__("polyq").Simulator(circuit, strategy="auto")
            setup = time.perf_counter() - t0
            times = []
            amp = None
            for _ in range(reps):
                t0 = time.perf_counter()
                amp = sim.amplitude(0, 0)
                times.append(time.perf_counter() - t0)
            value = complex(*amp)
        elif backend == "aer":
            from qiskit_aer import AerSimulator

            t0 = time.perf_counter()
            aer = AerSimulator(method="statevector")
            compiled = transpile(qc, aer, optimization_level=0)
            compiled.save_amplitudes([0], label="overlap")
            setup = time.perf_counter() - t0
            times = []
            value = None
            for _ in range(reps):
                t0 = time.perf_counter()
                result = aer.run(compiled, shots=1).result()
                times.append(time.perf_counter() - t0)
                value = complex(result.data(0)["overlap"][0])
        else:
            raise ValueError(backend)
        result_queue.put({
            "backend": backend,
            "setup_s": setup,
            "query_s_min": min(times),
            "amplitude": [value.real, value.imag],
            "overlap_probability": abs(value) ** 2,
            "peak_rss_mib": peak_rss_mib(),
        })
    except BaseException as exc:
        result_queue.put({"backend": backend, "error": f"{type(exc).__name__}: {exc}"})


def run_worker(backend: str, qc: QuantumCircuit, reps: int) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=worker, args=(backend, qc, reps, queue))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        return {"backend": backend, "error": f"worker exited with status {proc.exitcode}"}
    return queue.get()


def resolve(pattern: str, n: int) -> Path:
    return Path(pattern.format(n=n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logical-prep", required=True, help="QASM path/pattern preparing |0_L>")
    ap.add_argument("--target-prep", required=True, help="QASM path/pattern for U_prep|0^N>")
    ap.add_argument("--sizes", nargs="+", type=int, help="substitute each N into {n} in paths")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--aer-max-qubits", type=int, default=28,
                    help="skip Aer above this size; statevector memory is exponential")
    ap.add_argument("--output", type=Path, help="optional JSONL output path")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be positive")

    ns = args.sizes or [None]
    sink = args.output.open("w") if args.output else None
    try:
        for requested_n in ns:
            # A literal path is supported without --sizes; otherwise {n} is required.
            if requested_n is None and ("{n}" in args.logical_prep or "{n}" in args.target_prep):
                ap.error("--sizes is required when either input path contains {n}")
            n_hint = requested_n or 0
            logical_path = resolve(args.logical_prep, n_hint)
            target_path = resolve(args.target_prep, n_hint)
            logical = load_qasm2(str(logical_path))
            target = load_qasm2(str(target_path))
            if logical.num_qubits != target.num_qubits:
                raise ValueError(f"qubit count mismatch: {logical.num_qubits} vs {target.num_qubits}")
            n = logical.num_qubits
            if requested_n is not None and n != requested_n:
                raise ValueError(f"requested N={requested_n}, QASM declares N={n}")
            combined = logical.inverse().compose(target)
            result = {
                "N": n,
                "logical_prep": str(logical_path),
                "target_prep": str(target_path),
                "task": "abs(<0^N|U_logical^dagger U_prep|0^N>)^2",
                "polyq": run_worker("polyq", combined, args.reps),
            }
            if n <= args.aer_max_qubits:
                result["qiskit_aer"] = run_worker("aer", combined, args.reps)
            else:
                result["qiskit_aer"] = {"skipped": f"N>{args.aer_max_qubits} statevector limit"}
            p = result["polyq"].get("overlap_probability")
            a = result["qiskit_aer"].get("overlap_probability")
            if p is not None and a is not None:
                result["agreement_abs"] = abs(p - a)
                if result["agreement_abs"] > 1e-8:
                    raise AssertionError(f"PolyQ/Aer overlap mismatch at N={n}: {p} vs {a}")
            line = json.dumps(result, sort_keys=True)
            print(line, flush=True)
            if sink:
                sink.write(line + "\n")
                sink.flush()
    finally:
        if sink:
            sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
