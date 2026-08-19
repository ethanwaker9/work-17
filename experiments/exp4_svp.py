import argparse
import json
import math
import os
import resource
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpylll.util import gaussian_heuristic
from g6k import Siever, SieverParams
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.algorithms.pump import pump
from g6k.algorithms.workout import workout
from g6k.utils.stats import SieveTreeTracer

from bellek import strategies as S
from bellek.challenges import load_challenge, log_profile
from bellek.costmodel import CostModel
from bellek.search import bellek_search
from bellek.svpdim import target_norm_svp_challenge

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def plan(name, prof, log_target, cost, budget):
    d = len(prof)
    bmax = min(d - 1, 100)
    if name == "g6k":
        return None, {}
    if name == "pbkz":
        p = S.pbkz(prof, log_target, cost)
    elif name == "bkz2":
        p = S.bkz2(prof, log_target, cost)
    elif name == "asiaccs":
        p = S.asiaccs_tradeoff(prof, log_target, cost)
    elif name == "pssearch":
        p = S.pssearch(prof, log_target, cost, beta_min=50, beta_max=bmax, time_cap=300.0)
    elif name in ("bellek", "bellekt"):
        b = budget
        if name == "bellekt":
            b = S.g6k_default(prof, log_target, cost).memory
        best, st, _ = bellek_search(
            prof, log_target, cost, b, beta_min=50, beta_max=bmax,
            jumps=(1, 2, 3, 4, 6, 8, 12), tours=(1,), eps=0.008,
        )
        return (best[3], best[2]), st
    else:
        raise ValueError(name)
    return (p.steps, p.dsvp), p.stats


class DbSampler(threading.Thread):
    def __init__(self, g, period=0.005):
        threading.Thread.__init__(self)
        self.daemon = True
        self.g = g
        self.period = period
        self.peak_db = 0
        self.peak_dim = 0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                n = self.g.db_size()
                m = self.g.r - self.g.l
            except Exception:
                n, m = 0, 0
            if 0 <= n < 10 ** 9:
                if n > self.peak_db:
                    self.peak_db = n
            if 0 <= m <= 4096 and m > self.peak_dim:
                self.peak_dim = m
            time.sleep(self.period)


def run(name, d, seed, threads, budget, alg):
    A = load_challenge(d, seed, randomize_seed=seed + 1)
    params = SieverParams(threads=threads, default_sieve=alg)
    g = Siever(A, params)
    tracer = SieveTreeTracer(g, root_label="svp", start_clocks=True)
    prof = log_profile(g, d)
    gh = gaussian_heuristic([g.M.get_r(i, i) for i in range(d)])
    goal = (1.05 ** 2) * gh
    log_target = target_norm_svp_challenge(prof)
    cost = CostModel.load()
    t_plan = time.time()
    p, stats = plan(name, prof, log_target, cost, budget)
    t_plan = time.time() - t_plan
    sampler = DbSampler(g)
    sampler.start()
    t0 = time.time()
    if name == "g6k":
        workout(g, tracer, 0, d, goal_r0=goal, pump_params={"down_sieve": False})
        used = "workout"
        dsvp = g.r - g.l
    else:
        steps, dsvp = p
        for (beta, jump, tours) in steps:
            for _ in range(tours):
                pump_n_jump_bkz_tour(g, tracer, beta, jump=jump)
        n = dsvp
        while n <= d and g.M.get_r(0, 0) > goal:
            pump(g, tracer, 0, d, d - n, goal_r0=goal, down_sieve=False)
            n += 1
        dsvp = n - 1
        used = [list(x) for x in steps]
    wall = time.time() - t0
    sampler.stop = True
    sampler.join(timeout=1.0)
    peak_db = sampler.peak_db
    peak_dim = sampler.peak_dim
    r0 = g.M.get_r(0, 0)
    rec = {
        "strategy": name,
        "dim": d,
        "seed": seed,
        "plan_time": t_plan,
        "solve_time": wall,
        "rss": rss_bytes(),
        "peak_db": peak_db,
        "peak_dim": peak_dim,
        "dsvp": dsvp,
        "steps": used,
        "r0_over_gh": math.sqrt(r0 / gh),
        "solved": bool(r0 <= goal),
        "search_stats": dict((k, v) for k, v in stats.items() if isinstance(v, (int, float))),
    }
    del g
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", type=str, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=float, default=6e9)
    ap.add_argument("--alg", type=str, default="bgj1")
    ap.add_argument("--out", type=str, default="svp.jsonl")
    args = ap.parse_args()
    rec = run(args.strategy, args.dim, args.seed, args.threads, args.budget, args.alg)
    print(json.dumps(rec), flush=True)
    with open(os.path.join(RESULTS, args.out), "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
