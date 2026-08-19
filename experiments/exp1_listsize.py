import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpylll import FPLLL, IntegerMatrix, LLL
from g6k import Siever, SieverParams
from g6k.siever import SaturationError

from bellek.listsize import list_size_agps, list_size_ball, list_size_g6k

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def saturates(n, factor, seed, threads, alg, sat_ratio):
    FPLLL.set_random_seed(seed)
    A = LLL.reduction(IntegerMatrix.random(n, "qary", k=n // 2, bits=30))
    params = SieverParams(
        threads=threads,
        db_size_factor=factor,
        default_sieve=alg,
        saturation_ratio=sat_ratio,
        saturation_radius=4.0 / 3.0,
    )
    g = Siever(A, params)
    g.initialize_local(0, 0, n)
    t0 = time.time()
    try:
        g(alg=alg)
    except SaturationError:
        return False, time.time() - t0, len(g)
    return True, time.time() - t0, len(g)


def minimal_factor(n, seeds, threads, alg, sat_ratio, lo=0.6, hi=6.0, steps=7):
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        ok = 0
        for s in seeds:
            r, _, _ = saturates(n, mid, s, threads, alg, sat_ratio)
            ok += 1 if r else 0
        if ok * 2 >= len(seeds) + 1:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="46,50,54,58,62,66,70,74,78,82")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--alg", type=str, default="bgj1")
    ap.add_argument("--sat", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.6)
    ap.add_argument("--hi", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--out", type=str, default="listsize.json")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    seeds = list(range(1, args.seeds + 1))
    out = []
    for n in dims:
        t0 = time.time()
        c = minimal_factor(n, seeds, args.threads, args.alg, args.sat, args.lo, args.hi, args.steps)
        rec = {
            "n": n,
            "min_factor": c,
            "ball": list_size_ball(n) / (4.0 / 3.0) ** (n / 2.0),
            "agps": list_size_agps(n) / (4.0 / 3.0) ** (n / 2.0),
            "g6k": list_size_g6k(n) / (4.0 / 3.0) ** (n / 2.0),
            "seconds": time.time() - t0,
        }
        out.append(rec)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(RESULTS, args.out), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
