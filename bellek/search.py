import bisect
import math
import time as _time

from .costmodel import default_dim4free
from .geometry import prefix_sums
from .simulators import _apply_pump, bellek_sim
from .svpdim import reachable_head_norm, svp_dim_bellek

FREE_ENVELOPE = 60


class State(object):
    __slots__ = ("time", "profile", "steps", "level", "pre", "dsvp")

    def __init__(self, time, profile, steps, level, dsvp):
        self.time = time
        self.profile = profile
        self.steps = steps
        self.level = level
        self.pre = prefix_sums(profile)
        self.dsvp = dsvp


def dominates(a, b, eps=0.0):
    if a.time > b.time:
        return False
    pa, pb = a.pre, b.pre
    for i in range(len(pa)):
        if pa[i] > pb[i] + i * eps + 1e-9:
            return False
    return True


class Frontier(object):
    def __init__(self, eps=1e-9):
        self.items = []
        self.keys = []
        self.tests = 0
        self.eps = eps

    def _dom(self, a, b):
        return dominates(a, b, self.eps)

    def insert(self, s):
        hi = bisect.bisect_right(self.keys, s.dsvp)
        for i in range(hi):
            self.tests += 1
            if self._dom(self.items[i], s):
                return False
        keep = []
        keepk = []
        for i, t in enumerate(self.items):
            self.tests += 1
            if not self._dom(s, t):
                keep.append(t)
                keepk.append(self.keys[i])
        pos = bisect.bisect_left(keepk, s.dsvp)
        keep.insert(pos, s)
        keepk.insert(pos, s.dsvp)
        self.items = keep
        self.keys = keepk
        return True

    def __len__(self):
        return len(self.items)


def memory_cap_dimension(cost, budget_bytes, dmax=1024):
    best = 0
    for n in range(30, dmax):
        if cost.sieve_memory(n) <= budget_bytes:
            best = n
        else:
            break
    return best


def bellek_search(
    profile,
    log_target,
    cost,
    budget_bytes,
    beta_min=50,
    beta_max=None,
    jumps=(1, 2, 3, 4, 6, 8, 10, 12),
    tours=(1, 2),
    simulator=bellek_sim,
    dsvp_fun=svp_dim_bellek,
    dim4free_fun=default_dim4free,
    record_curve=False,
    eps=4e-3,
    time_cap=None,
    start_pump=40,
    lookback=1000,
):
    d = len(profile)
    t0 = _time.time()
    nmax = memory_cap_dimension(cost, budget_bytes)
    if beta_max is None:
        beta_max = d - 1
    betas = [b for b in range(beta_min, beta_max + 1) if b - dim4free_fun(b) <= nmax]
    stats = {"expansions": 0, "sims": 0, "frontier_max": 1, "dom_tests": 0, "completed": 1}

    def base_peak(st):
        if not st.steps:
            return 0.0
        b = st.steps[-1][0]
        return cost.sieve_memory(b - dim4free_fun(b))

    def evaluate(st):
        n = dsvp_fun(st.profile, log_target, nmax=nmax)
        if n > nmax:
            return None
        total = st.time + cost.pump_time(n, d - n)
        peak = max(cost.sieve_memory(n), base_peak(st))
        if peak > budget_bytes:
            return None
        return (total, peak, n, st.steps)

    def evaluate_progressive(st, n_hint):
        l = list(st.profile)
        acc = st.time
        peak = base_peak(st)
        n0 = max(start_pump, n_hint - lookback, d - FREE_ENVELOPE)
        n = n0
        for _ in range(n, nmax + 1):
            f = d - n
            hit = reachable_head_norm(l, 0, f) <= log_target
            acc += cost.pump_time(n, f)
            peak = max(peak, cost.sieve_memory(n))
            if peak > budget_bytes:
                return None
            if hit:
                return (acc, peak, n0, st.steps)
            _apply_pump(l, 0, d, f, down_stop_extra=0)
            n += 1
        return None

    root = State(0.0, list(profile), [], 0, dsvp_fun(list(profile), log_target, nmax=nmax))
    front = Frontier(eps=eps)
    front.insert(root)
    best = evaluate(root)
    alt = evaluate_progressive(root, root.dsvp)
    if alt is not None and (best is None or alt[0] < best[0]):
        best = alt
    curve = []
    hkz = simulator(list(profile), min(d - 1, nmax + dim4free_fun(nmax)), jump=1, tours=12)
    n_lb = dsvp_fun(hkz, log_target, nmax=nmax)
    floor_cost = cost.pump_time(n_lb, d - n_lb)

    for beta in betas:
        if time_cap is not None and _time.time() - t0 > time_cap:
            stats["completed"] = 0
            break
        f = dim4free_fun(beta)
        base = list(front.items)
        fresh = []
        for st in base:
            if st.level >= beta:
                continue
            if best is not None and st.time + floor_cost >= best[0]:
                continue
            for j in jumps:
                if j > max(f, 1) + 3:
                    continue
                prof = st.profile
                acc = st.time
                for t in tours:
                    step_t = cost.pnjbkz_time(d, beta, j, f)
                    prof = simulator(prof, beta, jump=j, tours=1, dim4free=f)
                    stats["sims"] += 1
                    acc = acc + step_t
                    if best is not None and acc >= best[0]:
                        break
                    steps = st.steps + [(beta, j, t)]
                    ns = State(acc, list(prof), steps, beta, dsvp_fun(prof, log_target, nmax=nmax))
                    fresh.append(ns)
                    stats["expansions"] += 1
        for ns in fresh:
            kept = front.insert(ns)
            if not kept:
                continue
            cand = evaluate(ns)
            if cand is None:
                continue
            if best is None or cand[0] < best[0]:
                best = cand
            if record_curve:
                curve.append((cand[1], cand[0], cand[3], cand[2]))
            if best is None or ns.time + floor_cost < best[0]:
                alt = evaluate_progressive(ns, cand[2])
                if alt is not None:
                    if alt[0] < best[0]:
                        best = alt
                    if record_curve:
                        curve.append((alt[1], alt[0], alt[3], alt[2]))
        stats["frontier_max"] = max(stats["frontier_max"], len(front))
    stats["dom_tests"] = front.tests
    stats["walltime"] = _time.time() - t0
    stats["frontier_final"] = len(front)
    if record_curve:
        bym = {}
        for pt in curve:
            k = round(pt[0])
            if k not in bym or pt[1] < bym[k][1]:
                bym[k] = pt
        env = []
        bestt = None
        for k in sorted(bym):
            pt = bym[k]
            if bestt is None or pt[1] < bestt:
                bestt = pt[1]
                env.append(pt)
        curve = env
    return best, stats, curve


def bellek_curve(profile, log_target, cost, budgets, **kw):
    out = []
    big = max(budgets)
    best, stats, curve = bellek_search(profile, log_target, cost, big, record_curve=True, **kw)
    for b in budgets:
        feasible = [c for c in curve if c[0] <= b]
        out.append((b, min(feasible, key=lambda x: x[1]) if feasible else None))
    return out, stats
