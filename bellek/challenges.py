import math
import os

from fpylll import GSO, LLL, FPLLL, IntegerMatrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE_DIR = os.path.join(ROOT, "svpchallenge")
URL = "https://www.latticechallenge.org/svp-challenge/generator.php"


def challenge_path(d, seed=0):
    os.makedirs(CHALLENGE_DIR, exist_ok=True)
    return os.path.join(CHALLENGE_DIR, "svpchallenge-dim-%03d-seed-%02d.txt" % (d, seed))


def download(d, seed=0):
    import requests

    path = challenge_path(d, seed)
    if os.path.isfile(path):
        return path
    r = requests.post(URL, data={"dimension": d, "seed": seed, "sent": "True"})
    with open(path, "w") as fh:
        fh.write(r.text)
    return path


def load_challenge(d, seed=0, randomize_seed=None, float_type="double"):
    path = download(d, seed)
    A = IntegerMatrix.from_file(path)
    A = LLL.reduction(A)
    A = IntegerMatrix.from_matrix(A, int_type="long")
    if randomize_seed is not None:
        from fpylll.algorithms.bkz2 import BKZReduction

        FPLLL.set_random_seed(randomize_seed)
        M = GSO.Mat(A, float_type=float_type)
        bkz = BKZReduction(M)
        bkz.randomize_block(0, A.nrows, density=A.ncols // 4)
        LLL.reduction(A)
    return A


def log_profile(g6k, d):
    return [0.5 * math.log(g6k.M.get_r(i, i)) for i in range(d)]


def log_profile_gso(M, d):
    return [0.5 * math.log(M.get_r(i, i)) for i in range(d)]
