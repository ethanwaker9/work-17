import math

from .costmodel import default_dim4free
from .geometry import log_unit_gh
from .simulators import HALF_LOG43, LOG_DB, MU, MU_EXIT, MU_LIFT, lift_floor_log, reachable_min_norm, unit_gh

LOG43 = math.log(4.0 / 3.0)


def log_gh_block(profile, start, end):
    m = end - start
    if m <= 0:
        return float("-inf")
    return math.fsum(profile[start:end]) / m + unit_gh(m)


def reachable_head_norm(profile, kappa, f, mu=None):
    d = len(profile)
    m = d - kappa
    if m <= 1:
        return profile[kappa] if kappa < d else float("inf")
    log_covol = math.fsum(profile[kappa:d])
    split = max(f, 0)
    n_s = m - split
    if n_s <= 0:
        return float("inf")
    log_rdb = log_gh_block(profile, d - n_s, d) + 0.5 * LOG43 + LOG_DB / n_s
    mu_ = MU_EXIT if mu is None else mu
    val = reachable_min_norm(m, split, log_covol, log_rdb, target=mu_)
    if split > 0:
        g = (2.0 * (profile[kappa] - profile[kappa + split - 1]) / (split - 1)
             if split > 1 else 0.0)
        fl = lift_floor_log(
            n_s * HALF_LOG43 + LOG_DB, math.log(mu_ * MU_LIFT), split,
            2.0 * profile[kappa], g if g > 0.0 else 0.0, val,
        )
        if fl is not None and fl > val:
            return fl
    return val


def svp_dim_bellek(profile, log_target, kappa=0, nmax=None, mu=None):
    d = len(profile)
    m = d - kappa
    if nmax is None:
        nmax = m
    hi_n = min(nmax, m)
    if hi_n < 30 or m <= 1:
        return nmax + 1
    mu_ = MU_EXIT if mu is None else mu
    suf = [0.0] * (m + 1)
    acc = 0.0
    for j in range(1, m + 1):
        acc += profile[d - j]
        suf[j] = acc
    log_covol = suf[m]

    def ok(n):
        return reachable_min_norm(
            m, m - n, log_covol,
            suf[n] / n + unit_gh(n) + 0.5 * LOG43 + LOG_DB / n,
            target=mu_,
        ) <= log_target

    if not ok(hi_n):
        return nmax + 1
    lo, hi = 30, hi_n
    if ok(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    while hi > 30 and ok(hi - 1):
        hi -= 1
    return hi


def svp_dim_ducas(profile, log_target, kappa=0, optimistic=True, nmax=None):
    d = len(profile)
    for n in range(30, d - kappa + 1):
        f = d - kappa - n
        lhs = log_target
        if optimistic:
            lhs += 0.5 * math.log(float(n) / (d - kappa))
        if lhs <= log_gh_block(profile, kappa + f, d) + 0.5 * LOG43:
            return n
    return d - kappa


def svp_dim_asiaccs(profile, log_target, slope=None, kappa=0, nmax=None):
    if slope is None:
        from .geometry import profile_slope

        slope = profile_slope(profile)
    return _svp_dim_asiaccs(profile, log_target, slope, kappa)


def _svp_dim_asiaccs(profile, log_target, slope, kappa=0):
    d = len(profile)
    lndelta = -slope / 2.0
    if lndelta <= 1e-12:
        return d - kappa
    f1 = 0.5 * LOG43 / lndelta
    f2 = 0.0
    for _ in range(60):
        val = 0.5 * math.log(4.0 * (d - f2) / (3.0 * d))
        nf = max(val / lndelta, 0.0)
        if abs(nf - f2) < 1e-6:
            break
        f2 = nf
    f = int(0.5 * (f1 + f2))
    return max(d - kappa - f, 30)


def usvp_dim(profile, sigma, kappa=0, prob=0.999):
    d = len(profile)
    for n in range(30, d - kappa + 1):
        start = d - n
        lhs = 2.0 * math.log(sigma) + math.log(n)
        rhs = 2.0 * log_gh_block(profile, start, d) + LOG43
        if lhs <= rhs:
            return n
    return d - kappa


def target_norm_svp_challenge(profile, factor=1.05):
    d = len(profile)
    return math.log(factor) + math.fsum(profile) / d + unit_gh(d)


def head_gap(profile, log_target):
    return profile[0] - log_target
