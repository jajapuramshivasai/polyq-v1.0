import time, sys, numpy as np, stim, polyq
sys.path.insert(0, '.')
from stim_convert import convert

def check(d, shots=2000):
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d)
    c, rec = convert(sc)
    s = polyq.CliffordSampler(c, record=rec)
    m = np.frombuffer(s.sample(shots, 1), dtype=np.uint8).reshape(shots, -1).astype(bool)
    det, obs = sc.compile_m2d_converter().convert(measurements=m, separate_observables=True)
    assert not det.any() and not obs.any(), "detector/observable violated"
    ref = sc.compile_sampler().sample(shots)
    # compare per-bit means and a few pairwise correlations
    dm = np.abs(m.mean(0) - ref.mean(0)).max()
    return dm, s.support_dim

def bench(d, shots_list=(1, 1000)):
    sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d)
    out = {}
    for shots in shots_list:
        t = time.perf_counter(); sp = sc.compile_sampler(); sp.sample(shots); out[('stim', shots)] = time.perf_counter() - t
    t = time.perf_counter(); c, rec = convert(sc); tconv = time.perf_counter() - t
    t = time.perf_counter(); s = polyq.CliffordSampler(c, record=rec); tan = time.perf_counter() - t
    for shots in shots_list:
        t = time.perf_counter(); s.sample(shots, 7); out[('polyq_sample', shots)] = time.perf_counter() - t
        out[('polyq', shots)] = tconv + tan + out[('polyq_sample', shots)]
    return out, c.num_qubits, len(c), s.support_dim

if __name__ == "__main__":
    for d in (3, 5):
        print("check d=%d: max |mean diff| %.3f, support dim %d" % ((d,) + check(d)))
    print(f"{'d':>3} {'qubits':>7} {'gates':>7} | {'stim1':>9} {'stim1000':>9} | {'pq analyse':>10} {'pq 1':>9} {'pq 1000':>9}")
    for d in (3, 5, 7, 9, 11):
        o, nq, ng, sd = bench(d)
        print(f"{d:>3} {nq:>7} {ng:>7} | {o[('stim',1)]*1e3:>8.1f}ms {o[('stim',1000)]*1e3:>8.1f}ms | {(o[('polyq',1)]-o[('polyq_sample',1)])*1e3:>8.0f}ms {o[('polyq',1)]*1e3:>7.0f}ms {o[('polyq',1000)]*1e3:>7.0f}ms")
        sys.stdout.flush()
