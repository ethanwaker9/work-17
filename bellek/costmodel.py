import json
import math
import os

from .listsize import bytes_per_vector, list_size_ball, list_size_g6k

DEFAULT_PARAMS = {
    "a0s": 0.0,
    "c_sieve": 0.349,
    "c_mem": 0.2075,
    "a1": 3.1e-9,
    "a2": 6.4e-10,
    "start_n": 30,
    "gauss_crossover": 50,
    "lift_cost": 1.0e-9,
}

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "calibration.json")


class CostModel(object):
    def __init__(self, params=None, partners=1.0):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.partners = partners
        self._sieve_cache = {}
        self._pump_cache = {}

    @classmethod
    def load(cls, path=None, partners=1.0):
        path = path or CALIBRATION_FILE
        if os.path.isfile(path):
            with open(path) as fh:
                return cls(json.load(fh), partners=partners)
        return cls(partners=partners)

    def sieve_time(self, n):
        if n <= 0:
            return 0.0
        v = self._sieve_cache.get(n)
        if v is not None:
            return v
        p = self.p
        v = p.get("a0s", 0.0) + p["a1"] * 2.0 ** (p["c_sieve"] * n) + p["a2"] * n * 2.0 ** (p["c_mem"] * n)
        self._sieve_cache[n] = v
        return v

    def pump_time(self, n_sieve, f=0, down=True):
        if n_sieve <= 0:
            return 0.0
        key = (n_sieve, f, down)
        v = self._pump_cache.get(key)
        if v is not None:
            return v
        start = min(self.p["start_n"], n_sieve)
        total = 0.0
        for j in range(start, n_sieve + 1):
            total += self.sieve_time(j)
        if down:
            total *= 2.0
        total += self.p["lift_cost"] * (f + 1) * self.list_size(n_sieve)
        self._pump_cache[key] = total
        return total

    def list_size(self, n):
        return list_size_ball(n, self.partners)

    def sieve_memory(self, n):
        return self.list_size(n) * bytes_per_vector(n)

    def g6k_memory(self, n, factor=3.2):
        return list_size_g6k(n, factor) * bytes_per_vector(n)

    def pnjbkz_time(self, d, beta, jump, f):
        n = beta - f
        blocks = 0
        blocks += len(range(0, f, jump))
        blocks += len(range(0, max(d - beta, 0), jump))
        blocks += len(range(0, f, jump))
        return blocks * self.pump_time(n, f)

    def pnjbkz_memory(self, beta, f):
        return self.sieve_memory(beta - f)

    def strategy_cost(self, d, steps, dim4free_fun):
        t = 0.0
        m = 0.0
        for (beta, jump, tours) in steps:
            f = dim4free_fun(beta)
            t += tours * self.pnjbkz_time(d, beta, jump, f)
            m = max(m, self.pnjbkz_memory(beta, f))
        return t, m


def default_dim4free(beta):
    if beta < 40:
        return 0
    return int(min((beta - 40) / 2.0, int(11.5 + 0.075 * beta)))


def sieve_dim_from_blocksize(beta):
    return beta - default_dim4free(beta)


def blocksize_from_sieve_dim(n):
    beta = n
    while sieve_dim_from_blocksize(beta + 1) <= n:
        beta += 1
    return beta


def gate_count_log2(n, quantum=False):
    if quantum:
        return 0.265 * n + 16.4 + math.log2(8 * n)
    return 0.292 * n + 16.4 + math.log2(8 * n)
