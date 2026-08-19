import math
import time as _time

from .costmodel import default_dim4free
from .geometry import prefix_sums, profile_slope
from .simulators import _apply_pump, bellek_sim, cn11, pnjbkz_sim, pnjbkz_sim_s6
from .svpdim import svp_dim_asiaccs, svp_dim_bellek, svp_dim_ducas

FREE_ENVELOPE = 60


class Plan(object):
    def __init__(self, name, steps, dsvp, time, memory, stats=None):
        self.name = name
        self.steps = steps
        self.dsvp = dsvp
        self.time = time
        self.memory = memory
        self.stats = stats or {}

    def __repr__(self):
        return "%s(steps=%s, dsvp=%d, t=%.3g, m=%.3g)" % (
            self.name,
            self.steps,
            self.dsvp,
            self.time,
            self.memory,
        )


def peak_memory(cost, steps, dsvp, dim4free_fun=default_dim4free):
    m = cost.sieve_memory(dsvp)
    for (b, j, t) in steps:
        m = max(m, cost.sieve_memory(b - dim4free_fun(b)))
    return m


def total_time(cost, d, steps, dsvp, dim4free_fun=default_dim4free):
    t = 0.0
    for (b, j, tt) in steps:
        t += tt * cost.pnjbkz_time(d, b, j, dim4free_fun(b))
    t += cost.pump_time(dsvp, d - dsvp)
    return t


def g6k_default(profile, log_target, cost, start_n=40, dim4free_dec=1):
    d = len(profile)
    l = list(profile)
    t0 = _time.time()
    total = 0.0
    peak = 0.0
    n = max(start_n, d - FREE_ENVELOPE)
    reached = False
    while n <= d:
        f = d - n
        _apply_pump(l, 0, d, f, down_stop_extra=0)
        total += cost.pump_time(n, f)
        peak = max(peak, cost.sieve_memory(n))
        if l[0] <= log_target:
            reached = True
            break
        n += dim4free_dec
    p = Plan("G6K-default", [], n, total, peak, {"walltime": _time.time() - t0, "reached": reached})
    p.profile = l
    return p


def _progressive_schedule(profile, log_target, cost, simulator, dsvp_fun, jump=1,
                          beta_min=50, beta_step=1, dim4free_fun=default_dim4free):
    d = len(profile)
    l = list(profile)
    steps = []
    best = None
    acc = 0.0
    beta = beta_min
    while beta < d:
        f = dim4free_fun(beta)
        l = simulator(l, beta, jump=jump, tours=1)
        acc += cost.pnjbkz_time(d, beta, jump, f)
        steps = steps + [(beta, jump, 1)]
        n = dsvp_fun(l, log_target)
        tot = acc + cost.pump_time(n, d - n)
        mem = peak_memory(cost, steps, n, dim4free_fun)
        if best is None or tot < best[0]:
            best = (tot, mem, n, list(steps))
        elif tot > 3.0 * best[0]:
            break
        beta += beta_step
    return best


def pbkz(profile, log_target, cost, dim4free_fun=default_dim4free):
    t0 = _time.time()
    best = _progressive_schedule(profile, log_target, cost, cn11, svp_dim_ducas, jump=1)
    return Plan("pBKZ", best[3], best[2], best[0], best[1], {"walltime": _time.time() - t0})


def bkz2(profile, log_target, cost, dim4free_fun=default_dim4free, tours=8):
    d = len(profile)
    t0 = _time.time()
    best = None
    for beta in range(50, min(d - 1, 200)):
        l = cn11(list(profile), beta, tours=tours)
        n = svp_dim_ducas(l, log_target)
        steps = [(beta, 1, tours)]
        tot = total_time(cost, d, steps, n, dim4free_fun)
        mem = peak_memory(cost, steps, n, dim4free_fun)
        if best is None or tot < best[0]:
            best = (tot, mem, n, steps)
        elif tot > 5.0 * best[0]:
            break
    return Plan("BKZ2.0", best[3], best[2], best[0], best[1], {"walltime": _time.time() - t0})


def asiaccs_tradeoff(profile, log_target, cost, jumps=(1, 2, 3, 4, 6, 8, 10, 12, 15, 18),
                     dim4free_fun=default_dim4free):
    d = len(profile)
    t0 = _time.time()
    ref = list(profile)
    cur = list(profile)
    steps = []
    acc = 0.0
    best = None
    beta = 50
    while beta < d:
        ref = cn11(ref, beta, tours=1)
        target_pre = prefix_sums(ref)
        pick = None
        for b2 in range(beta, min(d - 1, beta + 40)):
            f2 = dim4free_fun(b2)
            for j in jumps:
                if j > max(f2, 1) + 3:
                    continue
                cand = pnjbkz_sim_s6(cur, b2, jump=j, tours=1)
                cp = prefix_sums(cand)
                if all(cp[i] <= target_pre[i] + 1e-9 for i in range(len(cp))):
                    c = cost.pnjbkz_time(d, b2, j, f2)
                    if pick is None or c < pick[0]:
                        pick = (c, b2, j, cand)
            if pick is not None:
                break
        if pick is None:
            f2 = dim4free_fun(beta)
            pick = (cost.pnjbkz_time(d, beta, 1, f2), beta, 1, pnjbkz_sim_s6(cur, beta, jump=1, tours=1))
        acc += pick[0]
        cur = pick[3]
        steps = steps + [(pick[1], pick[2], 1)]
        n = svp_dim_asiaccs(cur, log_target, profile_slope(cur))
        tot = acc + cost.pump_time(n, d - n)
        mem = peak_memory(cost, steps, n, dim4free_fun)
        if best is None or tot < best[0]:
            best = (tot, mem, n, list(steps))
        elif tot > 3.0 * best[0]:
            break
        beta += 1
    return Plan("AsiaCCS23", best[3], best[2], best[0], best[1], {"walltime": _time.time() - t0})


def pssearch(profile, log_target, cost, beta_min=50, beta_max=None,
             jumps=(1, 2, 3, 4, 6, 8, 10, 12), simulator=pnjbkz_sim,
             dsvp_fun=svp_dim_ducas, dim4free_fun=default_dim4free, node_cap=200000,
             time_cap=1800.0):
    d = len(profile)
    t0 = _time.time()
    if beta_max is None:
        beta_max = d - 1
    root_prof = list(profile)
    root_n = dsvp_fun(root_prof, log_target)
    bs = [(0.0, root_prof, prefix_sums(root_prof), [], root_n)]
    k = 0
    stats = {"expansions": 0, "sims": 0, "dom_tests": 0, "frontier_max": 1, "completed": 1}
    while k < len(bs):
        if _time.time() - t0 > time_cap:
            stats["completed"] = 0
            break
        item = bs[k]
        for beta in range(beta_min, beta_max + 1):
            f = dim4free_fun(beta)
            for j in jumps:
                if j > max(f, 1) + 3:
                    continue
                prof = simulator(item[1], beta, jump=j, tours=1)
                stats["sims"] += 1
                pre = prefix_sums(prof)
                if all(pre[i] >= item[2][i] - 1e-9 for i in range(len(pre))):
                    continue
                tred = item[0] + cost.pnjbkz_time(d, beta, j, f)
                steps = item[3] + [(beta, j, 1)]
                n = dsvp_fun(prof, log_target)
                stats["expansions"] += 1
                dominated = False
                for other in bs:
                    stats["dom_tests"] += 1
                    if other[0] <= tred and all(other[2][i] <= pre[i] + 1e-9 for i in range(len(pre))):
                        dominated = True
                        break
                if dominated:
                    continue
                newbs = []
                for other in bs:
                    stats["dom_tests"] += 1
                    if tred <= other[0] and all(pre[i] <= other[2][i] + 1e-9 for i in range(len(pre))):
                        continue
                    newbs.append(other)
                bs = newbs + [(tred, prof, pre, steps, n)]
                stats["frontier_max"] = max(stats["frontier_max"], len(bs))
                if len(bs) > node_cap:
                    stats["completed"] = 0
                    k = len(bs)
                    break
        k += 1
    best = None
    for item in bs:
        n = item[4]
        tot = item[0] + cost.pump_time(n, d - n)
        mem = peak_memory(cost, item[3], n, dim4free_fun)
        if best is None or tot < best[0]:
            best = (tot, mem, n, item[3])
    stats["walltime"] = _time.time() - t0
    stats["frontier_final"] = len(bs)
    return Plan("PSSearch", best[3], best[2], best[0], best[1], stats)
