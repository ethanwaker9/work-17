import json
import os
import statistics
import sys

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ORDER = ["g6k", "pbkz", "bkz2", "asiaccs", "pssearch", "bellek", "bellekt"]


def main():
    path = os.path.join(RESULTS, "svp.jsonl")
    if not os.path.isfile(path):
        print("no svp.jsonl")
        return
    rows = [json.loads(x) for x in open(path) if x.strip()]
    dims = sorted(set(r["dim"] for r in rows))
    for d in dims:
        print("d =", d)
        agg = {}
        for k in ORDER:
            rs = [r for r in rows if r["dim"] == d and r["strategy"] == k]
            if not rs:
                continue
            agg[k] = (
                statistics.median([r["solve_time"] for r in rs]),
                statistics.median([r["peak_db"] for r in rs]),
                statistics.median([r["plan_time"] for r in rs]),
                statistics.median([r["dsvp"] for r in rs]),
                len(rs),
                sum(1 for r in rs if r["solved"]),
            )
        b = agg.get("bellek")
        for k in ORDER:
            if k not in agg:
                continue
            t, m, pt, ds, n, ok = agg[k]
            sp = t / b[0] if b else 0
            mp = m / b[1] if b else 0
            print(
                "  %-9s n=%d plan=%7.1f solve=%9.2f dsvp=%4.1f db=%9d solved=%d/%d "
                " x%.2f time x%.2f mem" % (k, n, pt, t, ds, m, ok, n, sp, mp)
            )


if __name__ == "__main__":
    main()
