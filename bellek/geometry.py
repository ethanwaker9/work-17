import math

from scipy import integrate, special

RK = (
    0.789527997160000,
    0.780003183804613,
    0.750872218594458,
    0.706520454592593,
    0.696345241018901,
    0.660533841808400,
    0.626274718790505,
    0.581480717333169,
    0.553171463433503,
    0.520811087419712,
    0.487994338534253,
    0.459541470573431,
    0.414638319529319,
    0.392811729940846,
    0.339090376264829,
    0.306561491936042,
    0.276041187709516,
    0.236698863270441,
    0.196186341673080,
    0.161214212092249,
    0.110895134828114,
    0.0678261623920553,
    0.0272807162335610,
    -0.0234609979600137,
    -0.0320527224746912,
    -0.0940331032784437,
    -0.129109087817554,
    -0.176965384290173,
    -0.209405754915959,
    -0.265867993276493,
    -0.299031324494802,
    -0.349338597048432,
    -0.380428160303508,
    -0.427399405474537,
    -0.474944677694975,
    -0.530140672818150,
    -0.561625221138784,
    -0.612008793872032,
    -0.669011014635905,
    -0.713766731570930,
    -0.754041787011810,
    -0.808609696192079,
    -0.859933249032210,
    -0.884479963601658,
    -0.886666930030433,
)

LN2 = math.log(2.0)
HKZ_HEAD = tuple(x * LN2 for x in RK)


def log_ball_volume(d, r=1.0):
    if d <= 0:
        return 0.0
    out = (d / 2.0) * math.log(math.pi) - special.gammaln(d / 2.0 + 1.0)
    if r != 1.0:
        out += d * math.log(r)
    return out


def ball_volume(d, r=1.0):
    return math.exp(log_ball_volume(d, r))


def log_unit_gh(d):
    if d <= 0:
        return 0.0
    if d <= 45:
        return HKZ_HEAD[-d] - sum(HKZ_HEAD[-d:]) / d
    return -log_ball_volume(d) / d


def half_space_measure(d, x):
    if x >= 1.0:
        return 0.0
    if x <= -1.0:
        return 1.0
    if x >= 0.0:
        return 0.5 * special.betainc((d - 1) / 2.0, 0.5, 1.0 - x * x)
    return 1.0 - 0.5 * special.betainc((d - 1) / 2.0, 0.5, 1.0 - x * x)


def cap_measure(d, theta):
    return half_space_measure(d, math.cos(theta))


def log_gh(log_profile):
    n = len(log_profile)
    if n == 0:
        return float("-inf")
    return sum(log_profile) / n + log_unit_gh(n)


def gh(log_profile):
    return math.exp(log_gh(log_profile))


def log_gh_from_logvol(log_vol, n):
    return log_vol / n + log_unit_gh(n)


def delta_bkz(beta):
    if beta <= 2:
        return 1.0219
    return ((beta / (2.0 * math.pi * math.e)) * (math.pi * beta) ** (1.0 / beta)) ** (
        1.0 / (2.0 * (beta - 1.0))
    )


def gsa_slope(beta):
    return -2.0 * math.log(delta_bkz(beta))


def gsa_profile(d, beta, log_vol):
    slope = gsa_slope(beta)
    centre = log_vol / d
    return [centre + slope * (i - (d - 1) / 2.0) for i in range(d)]


def profile_slope(log_profile):
    d = len(log_profile)
    xbar = (d - 1) / 2.0
    ybar = sum(log_profile) / d
    num = sum((i - xbar) * (log_profile[i] - ybar) for i in range(d))
    den = sum((i - xbar) ** 2 for i in range(d))
    return num / den


def prefix_sums(log_profile):
    out = [0.0]
    acc = 0.0
    for x in log_profile:
        acc += x
        out.append(acc)
    return out


def quality_leq(a, b, tol=1e-9):
    pa = prefix_sums(a)
    pb = prefix_sums(b)
    return all(pa[i] <= pb[i] + tol for i in range(len(pa)))


def root_hermite(log_profile):
    d = len(log_profile)
    return math.exp((log_profile[0] - sum(log_profile) / d) / d)


def log_cylinder_count(block_profile, split, log_r, log_rdb):
    f = split
    n = len(block_profile) - split
    log_covol = sum(block_profile)
    r = math.exp(log_r)
    rdb = math.exp(log_rdb)
    if f <= 0:
        return log_ball_volume(n, min(r, rdb)) - log_covol
    if n <= 0:
        return log_ball_volume(f, r) - log_covol
    ub = min(r, rdb)
    if ub <= 0.0:
        return float("-inf")
    log_surface = log_ball_volume(n) + math.log(n)

    def integrand(t):
        if t >= r or t <= 0.0:
            return 0.0
        return math.exp(
            log_surface
            + (n - 1) * math.log(t)
            + log_ball_volume(f, math.sqrt(max(r * r - t * t, 0.0)))
            - log_covol
        )

    val, _ = integrate.quad(integrand, 0.0, ub, limit=200)
    if val <= 0.0:
        return float("-inf")
    return math.log(val)


def solve_radius(block_profile, split, log_rdb, target_count=1.0, lo=None, hi=None):
    m = len(block_profile)
    base = log_gh(block_profile)
    if lo is None:
        lo = base - 3.0
    if hi is None:
        hi = base + 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c = log_cylinder_count(block_profile, split, mid, log_rdb)
        if c < math.log(target_count):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
