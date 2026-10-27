"""polyq CliffordSampler vs Stim on Stim-paper circuit families (noiseless, measurement records)."""
import time, sys, numpy as np, stim, polyq
sys.path.insert(0, '.')
from stim_convert import convert

FAMILIES = {
    "surface_code:rotated_memory_z": (3, 5, 7, 9, 11),
    "surface_code:unrotated_memory_z": (3, 5, 7, 9),
    "repetition_code:memory": (3, 11, 31, 61, 101),
}

def best(f, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)

def run(name, d):
    sc = stim.Circuit.generated(name, distance=d, rounds=d)
    t_stim_first = best(lambda: sc.compile_sampler().sample(1))
    sp = sc.compile_sampler()
    t_stim_100k = best(lambda: sp.sample(100_000), 2)
    t_stim_packed = best(lambda: sp.sample(100_000, bit_packed=True), 2)
    c, rec = convert(sc)
    t_an = best(lambda: polyq.CliffordSampler(c, record=rec), 2)
    s = polyq.CliffordSampler(c, record=rec)
    t_pq_100k = best(lambda: np.frombuffer(s.sample(100_000, 3), dtype=np.uint8).reshape(100_000, -1).view(bool), 2)
    # correctness: all detectors must be satisfied
    m = np.frombuffer(s.sample(4000, 9), dtype=np.uint8).reshape(4000, -1).astype(bool)
    det, obs = sc.compile_m2d_converter().convert(measurements=m, separate_observables=True)
    ok = (not det.any()) and (not obs.any())
    return sc.num_qubits, c.num_qubits, len(c), t_stim_first, t_stim_100k, t_an, t_pq_100k, ok, t_stim_packed

if __name__ == "__main__":
    print(f"{'circuit':34} {'d':>4} {'stimQ':>6} {'pqQ':>6} {'gates':>6} | {'stim 1st':>9} {'stim 100k':>10} {'stim packed':>11} | {'pq analyse':>10} {'pq 100k':>9} | ok")
    for name, ds in FAMILIES.items():
        for d in ds:
            try:
                r = run(name, d)
            except Exception as e:
                print(name, d, "ERR", e); continue
            print(f"{name:34} {d:>4} {r[0]:>6} {r[1]:>6} {r[2]:>6} | {r[3]*1e3:>7.2f}ms {r[4]*1e3:>8.1f}ms {r[8]*1e3:>9.1f}ms | {r[5]*1e3:>8.1f}ms {r[6]*1e3:>7.1f}ms | {r[7]}")
            sys.stdout.flush()
