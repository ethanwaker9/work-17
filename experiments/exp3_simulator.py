import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.utils.stats import SieveTreeTracer

from bellek.challenges import load_challenge, log_profile
from bellek.simulators import bellek_sim, cn11, pnjbkz_sim, pnjbkz_sim_s5, pnjbkz_sim_s6

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SIMS = {
    "CN11": cn11,
    "PnJBKZ": pnjbkz_sim,
    "S5": pnjbkz_sim_s5,
    "S6": pnjbkz_sim_s6,
    "Bellek": bellek_sim,
}


def errors(real, sim, head=40):
    d = len(real)
    e = [(real[i] - sim[i]) ** 2 for i in range(d)]
    return {
        "head": sum(e[:head]),
        "body": sum(e[head : d - 45]),
        "tail": sum(e[d - 45 :]),
        "total": sum(e),
        "head_max": max(abs(real[i] - sim[i]) for i in range(head)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=110)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=int, default=60)
    ap.add_argument("--jumps", type=str, default="1,4,8,12")
    ap.add_argument("--tours", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=45)
    ap.add_argument("--out", type=str, default="simaccuracy.json")
    args = ap.parse_args()

    d = args.dim
    out = []
    for jump in [int(x) for x in args.jumps.split(",")]:
        A = load_challenge(d, args.seed, randomize_seed=args.seed + 1)
        params = SieverParams(threads=args.threads, default_sieve="bgj1")
        g = Siever(A, params)
        tracer = SieveTreeTracer(g, root_label="sim", start_clocks=True)
        for b in range(40, args.warmup + 1, 5):
            pump_n_jump_bkz_tour(g, tracer, b, jump=1)
        start = log_profile(g, d)
        sims = dict((k, list(start)) for k in SIMS)
        startprof = list(start)
        for t in range(args.tours):
            t0 = time.time()
            pump_n_jump_bkz_tour(g, tracer, args.beta, jump=jump)
            dt = time.time() - t0
            real = log_profile(g, d)
            rec = {"jump": jump, "tour": t + 1, "beta": args.beta, "dim": d, "seconds": dt,
                   "real": real, "start": startprof}
            for k, fn in SIMS.items():
                sims[k] = fn(sims[k], args.beta, jump=jump, tours=1)
                rec[k] = errors(real, sims[k])
                rec[k + "_prof"] = list(sims[k])
            out.append(rec)
            print(
                "jump",
                jump,
                "tour",
                t + 1,
                dict((k, round(rec[k]["head"], 4)) for k in SIMS),
                flush=True,
            )
            with open(os.path.join(RESULTS, args.out), "w") as fh:
                json.dump(out, fh)
        del g


if __name__ == "__main__":
    main()
