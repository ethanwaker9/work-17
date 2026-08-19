import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CHILD = r"""
import os, resource, sys, json
from fpylll import FPLLL, IntegerMatrix, LLL
from g6k import Siever, SieverParams
n = int(sys.argv[1]); threads = int(sys.argv[2])
FPLLL.set_random_seed(1)
A = LLL.reduction(IntegerMatrix.random(n, "qary", k=n // 2, bits=30))
g = Siever(A, SieverParams(threads=threads, default_sieve="bgj1"))
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
g.initialize_local(0, 0, n)
g(alg="bgj1")
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
unit = 1 if sys.platform == "darwin" else 1024
print(json.dumps({"n": n, "db": len(g), "base": base * unit, "peak": peak * unit}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="55,60,65,70,75")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out = []
    for n in [int(x) for x in args.dims.split(",")]:
        r = subprocess.run([sys.executable, "-c", CHILD, str(n), str(args.threads)],
                           capture_output=True, text=True)
        line = [x for x in r.stdout.strip().split("\n") if x.startswith("{")]
        if not line:
            continue
        rec = json.loads(line[-1])
        rec["bytes_per_entry"] = (rec["peak"] - rec["base"]) / max(rec["db"], 1)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    with open(os.path.join(RESULTS, "entrysize.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
