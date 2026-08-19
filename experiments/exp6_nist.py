import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bellek.costmodel import default_dim4free, gate_count_log2
from bellek.geometry import delta_bkz
from bellek.listsize import bytes_per_vector, list_size_agps, list_size_ball, list_size_g6k

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SCHEMES = {
    "ML-KEM-512": {"n": 512, "q": 3329, "sigma": math.sqrt(1.5)},
    "ML-KEM-768": {"n": 768, "q": 3329, "sigma": 1.0},
    "ML-KEM-1024": {"n": 1024, "q": 3329, "sigma": 1.0},
    "ML-DSA-44": {"n": 1024, "q": 8380417, "sigma": math.sqrt(2.0)},
    "ML-DSA-65": {"n": 1280, "q": 8380417, "sigma": math.sqrt(20.0 / 3.0)},
    "ML-DSA-87": {"n": 1792, "q": 8380417, "sigma": math.sqrt(2.0)},
}

LAWS = {
    "AGPS20": list_size_agps,
    "AGPS20-1": lambda n: 0.5 * list_size_agps(n),
    "G6K": list_size_g6k,
    "Ours": list_size_ball,
}


def vec_bits_full(n):
    return bytes_per_vector(n) * 8.0


def vec_bits_min(n):
    return math.log2(n) * n


def vec_bits_agps(n):
    return math.log2(n)


def max_dimension(budget_bits, law, vecbits, dmax=2000):
    best = 0
    for n in range(40, dmax):
        if law(n) * vecbits(n) <= budget_bits:
            best = n
        else:
            break
    return best


def primal_blocksize(n, q, sigma, dmax_extra=2):
    best = None
    for m in range(n // 2, dmax_extra * n + 1, 8):
        d = m + n + 1
        logvol = m * math.log(q)
        for beta in range(50, d):
            delta = delta_bkz(beta)
            lhs = math.log(sigma) + 0.5 * math.log(beta)
            rhs = (2.0 * beta - d - 1) * math.log(delta) + logvol / d
            if lhs <= rhs:
                if best is None or beta < best[0]:
                    best = (beta, m, d)
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="nist.json")
    args = ap.parse_args()
    out = {"budgets": [], "budgets_agps_convention": [], "schemes": []}

    for bits in [64, 80, 100, 120, 140, 160]:
        rec = {"log2_bits": bits}
        for name, law in LAWS.items():
            n = max_dimension(2.0 ** bits, law, vec_bits_full)
            rec[name] = {"dim": n, "gates_log2": gate_count_log2(n)}
        out["budgets"].append(rec)
        print(json.dumps(rec), flush=True)

    for bits in [140]:
        rec = {"log2_bits": bits}
        for name, law in LAWS.items():
            rec[name] = max_dimension(2.0 ** bits, law, vec_bits_agps)
        out["budgets_agps_convention"].append(rec)
        print("agps-convention", json.dumps(rec), flush=True)

    for name, par in SCHEMES.items():
        n, q, sigma = par["n"], par["q"], par["sigma"]
        beta, m, d = primal_blocksize(n, q, sigma)
        nsieve = beta - default_dim4free(beta)
        rec = {
            "scheme": name,
            "n": n,
            "q": q,
            "sigma": sigma,
            "m": m,
            "d": d,
            "beta": beta,
            "sieve_dim": nsieve,
            "gates_log2": gate_count_log2(nsieve),
        }
        for lbl, law in LAWS.items():
            rec[lbl + "_mem_bits"] = math.log2(law(nsieve) * vec_bits_full(nsieve))
        out["schemes"].append(rec)
        print(json.dumps(rec), flush=True)

    with open(os.path.join(RESULTS, args.out), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
