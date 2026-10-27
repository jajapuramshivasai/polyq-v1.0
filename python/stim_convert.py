"""Convert a *noiseless* Stim circuit (Clifford gates + M/MR/R + annotations) into a polyq
Circuit using the deferred-measurement principle.

* `M q`   : copy the Z value of q onto a fresh record qubit (CX q -> rec).
* `MR q`  : same, then uncompute q with CX rec -> q (the qubit is back in |0>).
* `R q`   : only supported when q is untouched (initial reset) or right after M/MR.
Records are returned in measurement order, so polyq samples projected on the record qubits
have the same marginal distribution as stim.compile_sampler().sample().
"""
import stim
import polyq

_ONE = {"H": "H", "X": "X", "Y": "Y", "Z": "Z", "S": "S", "S_DAG": "SDG", "I": None}
_TWO = {"CX": "CX", "CNOT": "CX", "CZ": "CZ", "SWAP": "SWAP"}


def convert(circ: stim.Circuit):
    flat = circ.flattened()
    nq = flat.num_qubits
    ops = []
    records = []
    next_q = nq
    touched = [False] * nq
    for ins in flat:
        name = ins.name
        t = [x.value for x in ins.targets_copy()]
        if name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if name in _ONE:
            if _ONE[name]:
                for q in t:
                    ops.append((_ONE[name], [q])); touched[q] = True
        elif name in _TWO:
            for a, b in zip(t[0::2], t[1::2]):
                ops.append((_TWO[name], [a, b])); touched[a] = touched[b] = True
        elif name == "R":
            for q in t:
                if touched[q]:
                    raise ValueError("mid-circuit reset of a used qubit needs MR")
        elif name in ("M", "MR"):
            for q in t:
                rec = next_q; next_q += 1
                ops.append(("CX", [q, rec])); touched[q] = True
                if name == "MR":
                    ops.append(("CX", [rec, q]))
                records.append(rec)
        else:
            raise ValueError(f"unsupported instruction {name}")
    c = polyq.Circuit(next_q)
    c.extend(ops)
    return c, records
