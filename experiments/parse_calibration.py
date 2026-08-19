import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    sieve, pump = [], []
    with open(os.path.join(RESULTS, "exp2.log")) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 5 and p[0] == "sieve":
                sieve.append({"n": int(p[1]), "time": float(p[2]), "db": int(p[3]), "rss": int(p[4])})
            elif len(p) == 5 and p[0] == "pump":
                pump.append({"d": int(p[1]), "n": int(p[2]), "time": float(p[3]), "db": int(p[4])})
    out = {"sieve": sieve, "pump": pump, "threads": 8, "alg": "bgj1"}
    with open(os.path.join(RESULTS, "calibration_raw.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(len(sieve), "sieve rows,", len(pump), "pump rows")


if __name__ == "__main__":
    main()
