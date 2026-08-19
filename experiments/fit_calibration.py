import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import least_squares

from bellek.listsize import list_size_ball

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LIFT = 1.0e-9


def model(p, d, n):
    a0s, a1, c = p
    s = sum(a0s + a1 * 2.0 ** (c * j) for j in range(30, n + 1))
    return 2.0 * s + LIFT * (d - n + 1) * list_size_ball(n)


def main():
    raw = json.load(open(os.path.join(RESULTS, "calibration_raw.json")))
    pts = [(r["d"], r["n"], r["time"]) for r in raw["pump"]]
    sol = least_squares(
        lambda p: [model(p, d, n) / t - 1.0 for (d, n, t) in pts],
        [0.005, 3e-8, 0.345],
        bounds=([0.0, 1e-12, 0.20], [1.0, 1e-3, 0.50]),
    )
    a0s, a1, c = [float(x) for x in sol.x]
    entry = json.load(open(os.path.join(RESULTS, "entrysize.json"))) if os.path.isfile(
        os.path.join(RESULTS, "entrysize.json")) else {}
    out = {
        "a0s": a0s,
        "a1": a1,
        "a2": 0.0,
        "c_sieve": c,
        "c_mem": 0.2075,
        "start_n": 30,
        "lift_cost": LIFT,
        "fit_points": len(pts),
        "fit_cost": float(sol.cost),
    }
    for k in ("entry_slope", "entry_const"):
        if k in entry:
            out[k] = entry[k]
    with open(os.path.join(RESULTS, "calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    for (d, n, t) in pts:
        print("  d=%3d n=%2d meas=%9.3f pred=%9.3f ratio=%.3f"
              % (d, n, t, model(sol.x, d, n), model(sol.x, d, n) / t))


if __name__ == "__main__":
    main()
