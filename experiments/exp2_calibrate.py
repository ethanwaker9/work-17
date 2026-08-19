import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fpylll import FPLLL, IntegerMatrix, LLL
from g6k import Siever, SieverParams
from g6k.algorithms.pump import pump
from g6k.utils.stats import SieveTreeTracer

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def measure(n, threads, alg, seed=1):
    FPLLL.set_random_seed(seed)
    A = LLL.reduction(IntegerMatrix.random(n, "qary", k=n // 2, bits=30))
    params = SieverParams(threads=threads, default_sieve=alg)
    g = Siever(A, params)
    base = rss_bytes()
    g.initialize_local(0, 0, n)
    t0 = time.time()
    g(alg=alg)
    dt = time.time() - t0
    size = len(g)
    mem = rss_bytes() - base
    del g
    return dt, size, mem


def measure_pump(d, n, threads, alg, seed=1):
    FPLLL.set_random_seed(seed)
    A = LLL.reduction(IntegerMatrix.random(d, "qary", k=d // 2, bits=30))
    params = SieverParams(threads=threads, default_sieve=alg)
    g = Siever(A, params)
    tracer = SieveTreeTracer(g, root_label="pump", start_clocks=True)
    g.initialize_local(0, max(d - 40, 0), d)
    g(alg=alg)
    g.shrink_db(0)
    t0 = time.time()
    pump(g, tracer, 0, d, d - n)
    dt = time.time() - t0
    size = g.db_size()
    del g
    return dt, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="50,55,60,65,70,75,80,85")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--alg", type=str, default="bgj1")
    args = ap.parse_args()

    out = {"sieve": [], "pump": [], "threads": args.threads, "alg": args.alg}
    for n in [int(x) for x in args.dims.split(",")][:4]:
        dt, size, mem = measure(n, args.threads, args.alg)
        out["sieve"].append({"n": n, "time": dt, "db": size, "rss": mem})
        print("sieve", n, round(dt, 3), size, mem, flush=True)
    for n in [int(x) for x in args.dims.split(",")]:
        d = min(n + int(11.5 + 0.075 * n), n + 30)
        dt, size = measure_pump(d, n, args.threads, args.alg)
        out["pump"].append({"d": d, "n": n, "time": dt, "db": size})
        print("pump", d, n, round(dt, 3), size, flush=True)

    with open(os.path.join(RESULTS, "calibration_raw.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("raw calibration written; run fit_calibration.py to fit the model", flush=True)


if __name__ == "__main__":
    main()
