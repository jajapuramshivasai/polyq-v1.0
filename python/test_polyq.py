"""Python-level tests: polyq vs an independent NumPy statevector simulator, and vs Stim."""
import numpy as np, random, stim, polyq, sys
sys.path.insert(0, ".")
from stim_convert import convert

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
GATES1 = {"H": H, "X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1, -1]), "S": np.diag([1, 1j]),
          "T": np.diag([1, np.exp(1j * np.pi / 4)]), "TDG": np.diag([1, np.exp(-1j * np.pi / 4)])}

def apply1(sv, n, q, U):
    sv = sv.reshape([2] * n)               # axis i <-> qubit n-1-i
    ax = n - 1 - q
    sv = np.moveaxis(np.tensordot(U, sv, axes=([1], [ax])), 0, ax)
    return sv.reshape(-1)

def applycx(sv, n, c, t):
    idx = np.arange(1 << n)
    flip = ((idx >> c) & 1) == 1
    out = sv.copy(); out[idx[flip]] = sv[idx[flip] ^ (1 << t)]
    return out

def random_circuit(n, depth, seed):
    rng = random.Random(seed)
    pc = polyq.Circuit(n); ops = []
    for _ in range(depth):
        if rng.random() < 0.3 and n > 1:
            a, b = rng.sample(range(n), 2); pc.cx(a, b); ops.append(("CX", a, b))
        else:
            g = rng.choice(list(GATES1)); q = rng.randrange(n)
            getattr(pc, g.lower())(q); ops.append((g, q))
    return pc, ops

def dense(n, ops):
    sv = np.zeros(1 << n, complex); sv[0] = 1
    for op in ops:
        sv = applycx(sv, n, op[1], op[2]) if op[0] == "CX" else apply1(sv, n, op[1], GATES1[op[0]])
    return sv

def test_statevector_matches_numpy():
    for seed in range(25):
        n = 2 + seed % 4
        pc, ops = random_circuit(n, 25, seed)
        for strategy in ("auto", "cached"):
            sim = polyq.Simulator(pc, strategy=strategy)
            got = np.frombuffer(sim.statevector(0), dtype=np.complex128)
            assert np.allclose(got, dense(n, ops), atol=1e-8), (seed, strategy)

def test_equivalence_and_estimate():
    a = polyq.Circuit(3); a.ccx(0, 1, 2)
    b = polyq.Circuit(3); b.h(2); b.ccz(0, 1, 2); b.h(2)
    assert polyq.equivalent(a, b)[0]
    b.t(0)
    assert not polyq.equivalent(a, b)[0]
    pc, ops = random_circuit(4, 40, 3)
    sim = polyq.Simulator(pc, strategy="cached")
    want = dense(4, ops)[5]
    re, im, se = sim.amplitude_estimate(0, 5, 100000, 1)
    assert abs(complex(re, im) - want) < 6 * se + 1e-6

def test_pruning_bound():
    c = polyq.Circuit(2); c.h(0); c.p(0, 0.01); c.cx(0, 1); c.p(1, 0.02); c.t(1)
    pruned, bound = c.prune_small_rotations(0.05)
    assert len(pruned) == len(c) - 2 and bound > 0

def test_clifford_sampler_vs_stim_detectors():
    for name, d in (("surface_code:rotated_memory_z", 3), ("repetition_code:memory", 5)):
        sc = stim.Circuit.generated(name, distance=d, rounds=d)
        c, rec = convert(sc)
        s = polyq.CliffordSampler(c, record=rec)
        m = np.frombuffer(s.sample(500, 1), dtype=np.uint8).reshape(500, -1).astype(bool)
        det, obs = sc.compile_m2d_converter().convert(measurements=m, separate_observables=True)
        assert not det.any() and not obs.any()
