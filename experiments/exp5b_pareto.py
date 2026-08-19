import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bellek import strategies as S
from bellek.costmodel import CostModel
from bellek.geometry import gsa_profile
from bellek.search import bellek_search
from bellek.simulators import bellek_sim
from bellek.svpdim import svp_dim_bellek, target_norm_svp_challenge

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def score(prof, log_target, cost, steps, dim4free_fun):
    d = len(prof)
    l = list(prof)
    t = 0.0
    m = 0.0
    for (beta, jump, tours) in steps:
        f = dim4free_fun(beta)
        for _ in range(tours):
            l = bellek_sim(l, beta, jump=jump, tours=1, dim4free=f)
        t += tours * cost.pnjbkz_time(d, beta, jump, f)
        m = max(m, cost.sieve_memory(beta - f))
    n = svp_dim_bellek(l, log_target)
    if n > d:
        return None
    t += cost.pump_time(n, d - n)
    m = max(m, cost.sieve_memory(n))
    return t, m, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=140)
    ap.add_argument("--cap", type=float, default=180.0)
    ap.add_argument("--nmin", type=int, default=56)
    ap.add_argument("--nmax", type=int, default=74)
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()
    d = args.dim
    prof = gsa_profile(d, 2, 0.0)
    lt = target_norm_svp_challenge(prof)
    cost = CostModel.load()
    bmax = min(d - 1, 160)
    curve = []
    for n in range(args.nmin, args.nmax + 1, args.step):
        b = cost.sieve_memory(n)
        best, _, _ = bellek_search(prof, lt, cost, b, beta_min=50, beta_max=bmax,
                                   jumps=(1, 2, 3, 4, 6, 8, 12), tours=(1,), eps=0.008)
        if best is not None:
            curve.append([b, best[0]])
        print("  budget n=%d  %s" % (n, ("%.4g s" % best[0]) if best else "infeasible"), flush=True)
    pts = {}
    plans = [
        ("G6K-default", S.g6k_default(prof, lt, cost)),
        ("pBKZ", S.pbkz(prof, lt, cost)),
        ("BKZ2.0", S.bkz2(prof, lt, cost)),
        ("AsiaCCS23", S.asiaccs_tradeoff(prof, lt, cost)),
        ("PSSearch", S.pssearch(prof, lt, cost, beta_min=50, beta_max=bmax,
                                jumps=(1, 2, 3, 4, 6, 8, 12), time_cap=args.cap)),
    ]
    for name, p in plans:
        if name == "G6K-default":
            pts[name] = [p.time, p.memory]
            continue
        sc = score(prof, lt, cost, p.steps, __import__("bellek.costmodel", fromlist=["x"]).default_dim4free)
        if sc:
            pts[name] = [sc[0], sc[1]]
    sp = os.path.join(RESULTS, "search.json")
    if os.path.isfile(sp):
        rows = [r for r in json.load(open(sp)) if r["dim"] == d]
        if rows and "PSSearch" in rows[0]:
            pts["PSSearch"] = [rows[0]["PSSearch"]["pred_time"], rows[0]["PSSearch"]["pred_mem"]]
    with open(os.path.join(RESULTS, "pareto.json"), "w") as f:
        json.dump({"dim": d, "curve": curve, "points": pts}, f, indent=1)
    print("curve points:", len(curve))
    for p in curve:
        print("  mem=%.4g GiB  time=%.4g s" % (p[0] / 1024.0 ** 3, p[1]))
    for k, v in pts.items():
        print("  %-12s time=%.4g mem=%.4g GiB" % (k, v[0], v[1] / 1024.0 ** 3))


if __name__ == "__main__":
    main()
