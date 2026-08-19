import argparse
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpylll.util import gaussian_heuristic
from g6k import Siever, SieverParams
from g6k.algorithms.workout import workout
from g6k.utils.stats import SieveTreeTracer

from bellek.challenges import load_challenge, log_profile

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


class DimSampler(threading.Thread):
    def __init__(self, g, period=0.004):
        threading.Thread.__init__(self)
        self.daemon = True
        self.g = g
        self.period = period
        self.peak_dim = 0
        self.peak_db = 0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                m = self.g.r - self.g.l
                n = self.g.db_size()
            except Exception:
                m, n = 0, 0
            if 0 <= m <= 4096 and m > self.peak_dim:
                self.peak_dim = m
            if 0 <= n < 10 ** 9 and n > self.peak_db:
                self.peak_db = n
            time.sleep(self.period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--alg", type=str, default="bgj1")
    args = ap.parse_args()
    d = args.dim
    A = load_challenge(d, args.seed, randomize_seed=args.seed + 1)
    params = SieverParams(threads=args.threads, default_sieve=args.alg)
    g = Siever(A, params)
    tracer = SieveTreeTracer(g, root_label="svp", start_clocks=True)
    prof = log_profile(g, d)
    gh = gaussian_heuristic([g.M.get_r(i, i) for i in range(d)])
    goal = (1.05 ** 2) * gh
    s = DimSampler(g)
    s.start()
    t0 = time.time()
    workout(g, tracer, 0, d, goal_r0=goal, pump_params={"down_sieve": False})
    wall = time.time() - t0
    s.stop = True
    s.join(timeout=1.0)
    r0 = g.M.get_r(0, 0)
    rec = {
        "dim": d,
        "seed": args.seed,
        "exit_dim": s.peak_dim,
        "peak_db": s.peak_db,
        "time": wall,
        "r0_over_gh": math.sqrt(r0 / gh),
        "solved": bool(r0 <= goal),
        "profile": prof,
    }
    with open(os.path.join(RESULTS, "exitdim.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(dict((k, v) for k, v in rec.items() if k != "profile")))


if __name__ == "__main__":
    main()
