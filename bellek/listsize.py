import math

from scipy import integrate

from .geometry import cap_measure, half_space_measure

SQRT43 = math.sqrt(4.0 / 3.0)
G6K_FACTOR = 3.2


def reduction_probability(d):
    if d < 2:
        return 1.0

    def inner(g):
        if g <= 0.0:
            return 0.0
        return g ** (d - 1) * 2.0 * half_space_measure(d, g / 2.0)

    val, _ = integrate.quad(inner, 0.0, 1.0, limit=400)
    return d * val


def reduction_probability_asymptotic(d):
    return (6.0 / math.sqrt(2.0 * math.pi * d)) * (math.sqrt(3.0) / 2.0) ** (d - 1)


def list_size_ball(d, partners=1.0):
    return partners / reduction_probability(d)


def list_size_ball_asymptotic(d, partners=1.0):
    return partners * math.sqrt(6.0 * math.pi * d) / 12.0 * (4.0 / 3.0) ** (d / 2.0)


def list_size_constant(d, partners=1.0):
    return list_size_ball(d, partners) / (4.0 / 3.0) ** (d / 2.0)


def list_size_agps(d):
    return 2.0 / cap_measure(d, math.pi / 3.0)


def list_size_g6k(d, factor=G6K_FACTOR):
    return factor * (4.0 / 3.0) ** (d / 2.0)


def saturation_target(d, ratio=0.5, radius_sq=4.0 / 3.0):
    return 0.5 * ratio * radius_sq ** (d / 2.0)


ENTRY_SLOPE = 4.0
ENTRY_CONST = 0.0


def bytes_per_vector(d, slope=ENTRY_SLOPE, const=ENTRY_CONST):
    return slope * d + const


def memory_bytes(d, partners=1.0, factor=None):
    n = list_size_g6k(d, factor) if factor is not None else list_size_ball(d, partners)
    return n * bytes_per_vector(d)


def max_sieve_dimension(budget_bytes, partners=1.0, factor=None, dmax=1200):
    best = 0
    for d in range(2, dmax):
        if memory_bytes(d, partners, factor) <= budget_bytes:
            best = d
        else:
            break
    return best


def log2_memory_bits(d, partners=1.0, factor=None):
    return math.log2(memory_bytes(d, partners, factor) * 8.0)
