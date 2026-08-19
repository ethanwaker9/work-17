import argparse
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bellek import simulators as SIM

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def prefix(p, k):
    return math.fsum(p[:k])


def score(pred, real, kmax=41):
    return math.fsum((prefix(pred, k) - prefix(real, k)) ** 2 for k in range(kmax))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="simaccuracy.json")
    ap.add_argument("--out", type=str, default="mucalib.json")
    args = ap.parse_args()
    data = json.load(open(os.path.join(RESULTS, args.src)))
    beta = data[0]["beta"]
    by = {}
    for r in data:
        by[(r["jump"], r["tour"])] = r["real"]
    pairs = [(j, t) for (j, t) in by if (j, t - 1) in by]
    mus = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    lifts = [10.0, 30.0, 100.0]
    recs = [100.0, 1000.0, 10000.0]
    best = None
    grid = {}
    for mu, ml, mr in itertools.product(mus, lifts, recs):
        SIM.MU = mu
        SIM._LOG_MU = math.log(mu)
        SIM.MU_LIFT = ml
        SIM.MU_REC = mr
        SIM._LIFT_CACHE.clear()
        SIM._CH_CACHE.clear()
        tot = 0.0
        for (j, t) in pairs:
            pred = SIM.bellek_sim(list(by[(j, t - 1)]), beta, jump=j, tours=1)
            tot += score(pred, by[(j, t)])
        grid[(mu, ml, mr)] = tot
        if best is None or tot < best[1]:
            best = ((mu, ml, mr), tot)
        print("mu=%-6g lift=%-5g rec=%-8g err=%.4f" % (mu, ml, mr, tot), flush=True)
    print("best", best)
    rows = []
    for mu in mus:
        rows.append([mu] + [grid[(mu, ml, best[0][2])] for ml in lifts])
    with open(os.path.join(RESULTS, args.out), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(RESULTS, "mubest.json"), "w") as f:
        json.dump({"mu": best[0][0], "lift": best[0][1], "rec": best[0][2],
                   "err": best[1], "pairs": len(pairs), "beta": beta}, f)


if __name__ == "__main__":
    main()
