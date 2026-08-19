import argparse
import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bellek import strategies as S
from bellek.costmodel import CostModel
from bellek.geometry import gsa_profile
from bellek.search import bellek_search
from bellek.simulators import bellek_sim, pnjbkz_sim
from bellek.simulators import _apply_pump
from bellek.svpdim import reachable_head_norm, svp_dim_bellek, target_norm_svp_challenge

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def timed(fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    tracemalloc.start()
    fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="100,120,140,160,180")
    ap.add_argument("--budget", type=float, default=6e9)
    ap.add_argument("--own", action="store_true")
    ap.add_argument("--cap", type=float, default=900.0)
    ap.add_argument("--sweepdim", type=int, default=140)
    ap.add_argument("--out", type=str, default="search.json")
    args = ap.parse_args()

    cost = CostModel.load()
    out = []
    for d in [int(x) for x in args.dims.split(",")]:
        prof = gsa_profile(d, 2, 0.0)
        lt = target_norm_svp_challenge(prof)
        bmax = min(d - 1, 160)
        l = list(prof)
        n_prog = d
        for n in range(max(40, d - 60), d + 1):
            if reachable_head_norm(l, 0, d - n) <= lt:
                n_prog = n
                break
            _apply_pump(l, 0, d, d - n, down_stop_extra=0)
        budget = cost.sieve_memory(n_prog)
        rec = {"dim": d, "budget": budget, "cap": n_prog}

        for name, fn in [
            ("pBKZ", lambda: S.pbkz(prof, lt, cost)),
            ("BKZ2.0", lambda: S.bkz2(prof, lt, cost)),
            ("AsiaCCS23", lambda: S.asiaccs_tradeoff(prof, lt, cost)),
        ]:
            p, dt, peak = timed(fn)
            rec[name] = {
                "walltime": dt,
                "peak_bytes": peak,
                "pred_time": p.time,
                "pred_mem": p.memory,
                "dsvp": p.dsvp,
                "steps": len(p.steps),
                "sims": p.stats.get("sims", 0),
                "dom_tests": p.stats.get("dom_tests", 0),
                "frontier": p.stats.get("frontier_final", 1),
            }

        kw = {"simulator": bellek_sim, "dsvp_fun": svp_dim_bellek}
        if args.own:
            kw = {}
        p, dt, peak = timed(
            lambda: S.pssearch(prof, lt, cost, beta_min=50, beta_max=bmax,
                               jumps=(1, 2, 3, 4, 6, 8, 12), time_cap=args.cap, **kw)
        )
        rec["PSSearch"] = {
            "walltime": dt,
            "peak_bytes": peak,
            "pred_time": p.time,
            "pred_mem": p.memory,
            "dsvp": p.dsvp,
            "steps": len(p.steps),
            "sims": p.stats["sims"],
            "dom_tests": p.stats["dom_tests"],
            "frontier": p.stats["frontier_final"],
            "completed": p.stats.get("completed", 1),
        }

        res, dt, peak = timed(
            lambda: bellek_search(prof, lt, cost, budget, beta_min=50, beta_max=bmax,
                                  jumps=(1, 2, 3, 4, 6, 8, 12), tours=(1,), eps=0.008)
        )
        best, st, _ = res
        rec["Bellek"] = {
            "walltime": dt,
            "peak_bytes": peak,
            "pred_time": best[0],
            "pred_mem": best[1],
            "dsvp": best[2],
            "steps": len(best[3]),
            "sims": st["sims"],
            "dom_tests": st["dom_tests"],
            "frontier": st["frontier_final"],
        }
        out.append(rec)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(RESULTS, args.out), "w") as fh:
            json.dump(out, fh, indent=1)

        if d == args.sweepdim:
            sweep = []
            for eps in (0.032, 0.016, 0.008, 0.004, 0.0):
                res2, dt2, pk2 = timed(
                    lambda e=eps: bellek_search(prof, lt, cost, budget, beta_min=50,
                                                beta_max=bmax, jumps=(1, 2, 3, 4, 6, 8, 12),
                                                tours=(1,), eps=e, time_cap=args.cap)
                )
                b2, s2, _ = res2
                sweep.append({"eps": eps, "walltime": dt2, "peak_bytes": pk2,
                              "pred_time": b2[0], "dsvp": b2[2], "frontier": s2["frontier_final"],
                              "sims": s2["sims"], "steps": len(b2[3]),
                              "completed": s2.get("completed", 1)})
                print("eps", eps, "t=%.4g" % b2[0], "wall=%.1f" % dt2, "front", s2["frontier_final"], flush=True)
            with open(os.path.join(RESULTS, "epssweep.json"), "w") as fh:
                json.dump({"dim": d, "sweep": sweep}, fh, indent=1)
            _, _, curve = bellek_search(prof, lt, cost, budget, beta_min=50, beta_max=bmax,
                                        jumps=(1, 2, 3, 4, 6, 8, 12), tours=(1,), eps=0.008,
                                        record_curve=True)
            pts = dict((k, [rec[k]["pred_time"], rec[k]["pred_mem"]]) for k in
                       ["pBKZ", "BKZ2.0", "AsiaCCS23", "PSSearch"])
            g = S.g6k_default(prof, lt, cost)
            pts["G6K-default"] = [g.time, g.memory]
            with open(os.path.join(RESULTS, "pareto.json"), "w") as fh:
                json.dump({"curve": [[c[0], c[1]] for c in curve], "points": pts}, fh, indent=1)


if __name__ == "__main__":
    main()
