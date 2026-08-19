import math

import numpy as np

from .costmodel import default_dim4free
from .geometry import HKZ_HEAD, log_ball_volume, log_unit_gh, profile_slope

LOG43 = math.log(4.0 / 3.0)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_GL_T = 0.5 * (_GL_X + 1.0)
_GL_WT = 0.5 * _GL_W
_UNIT_GH = [log_unit_gh(i) for i in range(0, 4096)]


def unit_gh(m):
    if m < len(_UNIT_GH):
        return _UNIT_GH[m]
    return log_unit_gh(m)


def cn11(profile, beta, jump=1, tours=1):
    l = list(profile)
    n = len(l)
    b = min(beta, n)
    for _ in range(tours):
        logvol = sum(l[:b])
        updated = False
        for k in range(n - 1):
            end = min(k + b, n)
            m = end - k
            cand = logvol / m + unit_gh(m)
            if updated or cand < l[k]:
                l[k] = cand
                updated = True
            logvol -= l[k]
            if end < n:
                logvol += l[end]
        l[-1] = logvol
        if not updated:
            break
    return l


def pnjbkz_sim(profile, beta, jump=1, tours=1, sim_beta=None):
    b = sim_beta if sim_beta is not None else beta
    l = list(profile)
    d = len(l)
    b = min(b, d)
    total = sum(l)
    for _ in range(tours):
        pre_old = np.cumsum([0.0] + l)
        lp = [0.0] * d
        acc = 0.0
        updated = False
        for k in range(d - 45):
            j = k % jump
            if k < d - b:
                end = min(k - j + b, d)
            else:
                end = d
            m = end - k
            if m <= 0:
                lp[k] = l[k]
                continue
            logv = pre_old[end] - acc
            cand = logv / m + unit_gh(m)
            if updated:
                lp[k] = cand
            elif cand < l[k]:
                lp[k] = cand
                updated = True
            else:
                lp[k] = l[k]
            acc += lp[k]
        k0 = max(d - 45, 0)
        logv = total - acc
        for k in range(k0, d):
            lp[k] = logv / (d - k0) + HKZ_HEAD[k - k0]
        l = lp
        if not updated:
            break
    return l


def sieving_dim(beta):
    return beta - default_dim4free(beta)


def d4f_asymptotic_optimistic(beta):
    if beta <= 20:
        return 0
    return int(beta * LOG43 / math.log(beta / (2.0 * math.pi * math.e)))


def d4f_asymptotic_conservative(beta):
    if beta <= 10:
        return 0
    return int(beta * LOG43 / math.log(beta / (2.0 * math.pi)))


def d4f_gsa_optimistic(slope):
    lndelta = -slope / 2.0
    if lndelta <= 1e-9:
        return 0
    return int(0.5 * LOG43 / lndelta)


def d4f_gsa_conservative(slope, d, jump=1):
    lndelta = -slope / 2.0
    if lndelta <= 1e-9:
        return 0
    g = max(1.0 - 0.01 * jump, 0.05)
    f = 0.0
    for _ in range(60):
        val = math.log(g) + 0.5 * math.log(4.0 * (d - f) / (3.0 * d))
        nf = max(val / lndelta, 0.0)
        if abs(nf - f) < 1e-6:
            break
        f = nf
    return int(f)


def sim_blocksize(profile, beta, jump, strategy):
    if strategy == 2:
        return beta
    if strategy == 3:
        return sieving_dim(beta) + d4f_asymptotic_optimistic(beta)
    if strategy == 4:
        return sieving_dim(beta) + d4f_asymptotic_conservative(beta)
    s = profile_slope(profile)
    if strategy == 5:
        return sieving_dim(beta) + d4f_gsa_optimistic(s)
    return sieving_dim(beta) + d4f_gsa_conservative(s, len(profile), jump)


def pnjbkz_sim_d4f(profile, beta, jump=1, tours=1, strategy=6):
    l = list(profile)
    for _ in range(tours):
        b = min(sim_blocksize(l, beta, jump, strategy), len(l))
        l = pnjbkz_sim(l, beta, jump=jump, tours=1, sim_beta=b)
    return l


def pnjbkz_sim_s5(profile, beta, jump=1, tours=1):
    return pnjbkz_sim_d4f(profile, beta, jump, tours, strategy=5)


def pnjbkz_sim_s6(profile, beta, jump=1, tours=1):
    return pnjbkz_sim_d4f(profile, beta, jump, tours, strategy=6)


_GLT_POW = {}


def _glt_pow(n):
    a = _GLT_POW.get(n)
    if a is None:
        a = np.power(_GL_T, 2.0 / n)
        _GLT_POW[n] = a
    return a


def _lift_count_log(m, f, log_covol, log_r, log_rdb):
    if f <= 0 or log_r <= log_rdb:
        return m * log_r + log_ball_volume(m) - log_covol
    n = m - f
    u = math.exp(log_rdb - log_r)
    inner = 1.0 - (u * u) * _glt_pow(n)
    np.maximum(inner, 1e-300, out=inner)
    integral = float(np.dot(_GL_WT, np.power(inner, f / 2.0)))
    if integral <= 0.0:
        return float("-inf")
    return (
        log_ball_volume(f)
        + log_ball_volume(n)
        + f * log_r
        + n * log_rdb
        + math.log(integral)
        - log_covol
    )


_LIFT_CACHE = {}


def _lift_gap(m, f, gap, logt):
    cv = -m * unit_gh(m)
    lo = 0.0
    if _lift_count_log(m, f, cv, lo, gap) >= logt:
        return lo
    hi = 10.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if _lift_count_log(m, f, cv, mid, gap) < logt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


MU = 0.25
MU_LIFT = 100.0
MU_EXIT = 0.005
MU_REC = 1000.0
_LOG_MU = math.log(MU)
SQRT2 = math.sqrt(2.0)
LOG_PI = math.log(math.pi)
_PSI_LO, _PSI_HI, _PSI_N = -25.0, 60.0, 34000
_PSI_X = np.linspace(_PSI_LO, _PSI_HI, _PSI_N)
_PSI_H = _PSI_X[1] - _PSI_X[0]


def _phi_log(x):
    s = np.exp(x)
    out = np.empty_like(s)
    small = s < 1e-8
    out[small] = -s[small] / 12.0
    b = ~small
    sb = s[b]
    out[b] = 0.5 * (LOG_PI - np.log(sb)) + np.log(np.maximum(_erf_vec(np.sqrt(sb) / 2.0), 1e-300))
    return out


def _erf_vec(a):
    return np.vectorize(math.erf)(a)


_PSI_Y = _phi_log(_PSI_X)
_PSI = np.concatenate(([0.0], np.cumsum(0.5 * (_PSI_Y[1:] + _PSI_Y[:-1]) * _PSI_H)))
_PSI = _PSI - _PSI[0]


def _psi(x):
    if x <= _PSI_LO:
        return 0.0
    if x >= _PSI_HI:
        return _PSI[-1] + 0.5 * (
            LOG_PI * (x - _PSI_HI) - 0.5 * (x * x - _PSI_HI * _PSI_HI)
        )
    k = (x - _PSI_LO) / _PSI_H
    i = int(k)
    if i >= _PSI_N - 1:
        return _PSI[-1]
    w = k - i
    return _PSI[i] * (1.0 - w) + _PSI[i + 1] * w


def _phi_scalar(x):
    if x < -18.4:
        return -math.exp(x) / 12.0
    s = math.exp(x)
    e = math.erf(math.sqrt(s) / 2.0)
    if e <= 0.0:
        return -700.0
    return 0.5 * (LOG_PI - x) + math.log(e)


_GMIN = 1e-3


def _cgf(x, f, g):
    return (_psi(x) - _psi(x - g * f)) / g


def _cgf_d(x, f, g):
    return (_phi_scalar(x) - _phi_scalar(x - g * f)) / g


def _chernoff_log(tau, f, g):
    if tau <= 0.0:
        return -700.0
    c = math.log(f / (2.0 * tau))
    lo, hi = c - 22.0, c + 22.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if math.exp(mid) * tau + _cgf_d(mid, f, g) < 0.0:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    return min(0.0, math.exp(x) * tau + _cgf(x, f, g)), x


_CH_CACHE = {}


def _chernoff_mean(f, g):
    return (1.0 - math.exp(-g * f)) / (12.0 * (1.0 - math.exp(-g)))


def _chernoff_tau(f, g, target):
    key = (f, int(g * 500.0), int(target * 25.0))
    val = _CH_CACHE.get(key)
    if val is not None:
        return val
    gq = max(key[1] / 500.0, _GMIN)
    tq = key[2] / 25.0
    mean = _chernoff_mean(f, gq)
    lo, hi = 0.0, mean
    v, _ = _chernoff_log(hi, f, gq)
    if v < tq:
        _CH_CACHE[key] = mean
        return mean
    tau = 0.5 * mean
    for _ in range(40):
        v, x = _chernoff_log(tau, f, gq)
        if v < tq:
            lo = tau
        else:
            hi = tau
        if abs(v - tq) < 1e-4:
            break
        step = (tq - v) / math.exp(x)
        nt = tau + step
        if not (lo < nt < hi):
            nt = 0.5 * (lo + hi)
        if abs(nt - tau) < 1e-12 * max(1.0, tau):
            tau = nt
            break
        tau = nt
    _CH_CACHE[key] = tau
    return tau


def lift_floor_log(log_ndb, log_mu, f, log_a0, g, log_r=None):
    if f <= 0:
        return None
    target = log_mu + math.log(MU_REC) - log_ndb
    if target >= 0.0:
        return None
    gq = max(g, _GMIN)
    if log_r is not None and 2.0 * log_r - log_a0 >= math.log(_chernoff_mean(f, gq)):
        return None
    tau = _chernoff_tau(f, gq, target)
    if tau <= 0.0:
        return None
    return 0.5 * (log_a0 + math.log(tau))


def _lift_lookup(m, f, gap, mub):
    key = (m, f, int(gap * 2000.0), int(mub * 100.0))
    val = _LIFT_CACHE.get(key)
    if val is None:
        val = _lift_gap(m, f, key[2] / 2000.0, math.log(mub))
        _LIFT_CACHE[key] = val
    return val


def reachable_min_norm(m, f, log_covol, log_rdb, target=None):
    mu = MU if target is None else target
    ref = log_covol / m + unit_gh(m)
    base = ref + math.log(mu) / m
    if f <= 0 or base <= log_rdb:
        return base
    return ref + _lift_lookup(m, f, log_rdb - ref, mu * MU_LIFT)


HALF_LOG43 = 0.5 * math.log(4.0 / 3.0)
DB_FACTOR = 3.2
LOG_DB = math.log(2.0 * DB_FACTOR)


def _apply_pump(l, kappa, blocksize, f, down_stop_extra=3):
    d = len(l)
    end = min(kappa + blocksize, d)
    if end - kappa <= 1:
        return
    last = min(kappa + f + down_stop_extra, end - 2)
    if last < kappa:
        return
    ugh = _UNIT_GH
    mub = MU * MU_LIFT
    log_mub = math.log(mub)
    volume = math.fsum(l[kappa:end])
    sieved = math.fsum(l[kappa + f : end]) if 0 < f < end - kappa else 0.0
    off = 0.0
    for i in range(kappa, last + 1):
        m = end - i
        if m <= 1:
            break
        cur = l[i] + off
        raw_i = l[i]
        k = i - kappa
        split = f if f <= m - 1 else m - 1
        n_s = m - split
        if n_s <= 1:
            break
        ref = volume / m + ugh[m]
        if m <= 45:
            cand = ref
        else:
            base = ref + _LOG_MU / m
            head = LOG_DB - k * HALF_LOG43
            log_rdb = (sieved + n_s * off) / n_s + ugh[n_s] + HALF_LOG43 + (
                head / n_s if head > 0.0 else 0.0
            )
            if split <= 0 or base <= log_rdb:
                cand = base
            else:
                cand = ref + _lift_lookup(m, split, log_rdb - ref, mub)
            if split > 0:
                g = 2.0 * (raw_i - l[i + split - 1]) / (split - 1) if split > 1 else 0.0
                fl = lift_floor_log(
                    n_s * HALF_LOG43 + (head if head > 0.0 else 0.0),
                    log_mub, split, 2.0 * cur, g if g > 0.0 else 0.0, cand,
                )
                if fl is not None and fl > cand:
                    cand = fl
        if cand < cur - 1e-12:
            off += (cur - cand) / (m - 1)
            l[i] = cand
        else:
            l[i] = cur
        volume -= l[i]
        if f > 0:
            sieved -= l[i + f] if i + f < end else 0.0
    if off != 0.0:
        for j in range(last + 1, end):
            l[j] += off


def _hkz_tail(l, m=45):
    d = len(l)
    if d <= m:
        return
    k0 = d - m
    logv = math.fsum(l[k0:])
    for k in range(k0, d):
        l[k] = logv / m + HKZ_HEAD[k - k0]


def tour_indices(d, beta, jump, f):
    idx = [(0, beta - f + i, i) for i in range(0, f, jump)]
    idx += [(i, beta, f) for i in range(0, max(d - beta, 0), jump)]
    idx += [(d - beta + i, beta - i, f - i) for i in range(0, f, jump)]
    return idx


def bellek_sim(profile, beta, jump=1, tours=1, dim4free=None, down_stop_extra=3):
    l = list(profile)
    d = len(l)
    if beta > d:
        beta = d
    f = default_dim4free(beta) if dim4free is None else dim4free
    for _ in range(tours):
        before = list(l)
        for (kappa, b, ff) in tour_indices(d, beta, jump, f):
            if kappa < 0 or b <= 2 or kappa + b > d:
                continue
            _apply_pump(l, kappa, b, max(ff, 0), down_stop_extra)
        _hkz_tail(l)
        if max(abs(l[i] - before[i]) for i in range(d)) < 1e-9:
            break
    return l


def apply_strategy(profile, steps, simulator, **kw):
    l = list(profile)
    for step in steps:
        beta, jump, tours = step[0], step[1], step[2]
        l = simulator(l, beta, jump=jump, tours=tours, **kw)
    return l
