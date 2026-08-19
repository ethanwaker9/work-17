import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpylll import GSO

from bellek.challenges import load_challenge, log_profile_gso
from bellek.simulators import MU_EXIT, _apply_pump
from bellek.svpdim import reachable_head_norm, target_norm_svp_challenge

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def model_exit(prof, mu, n0=40):
    d = len(prof)
    lt = target_norm_svp_challenge(prof)
    l = list(prof)
    for n in range(n0, d + 1):
        if reachable_head_norm(l, 0, d - n, mu=mu) <= lt:
            return n
        _apply_pump(l, 0, d, d - n, down_stop_extra=0)
    return d + 1


def main():
    rows = [json.loads(x) for x in open(os.path.join(RESULTS, "svp.jsonl")) if x.strip()]
    out = []
    for r in rows:
        if r["strategy"] != "g6k":
            continue
        d, seed = r["dim"], r["seed"]
        A = load_challenge(d, seed, randomize_seed=seed + 1)
        M = GSO.Mat(A)
        M.update_gso()
        prof = log_profile_gso(M, d)
        out.append({"dim": d, "seed": seed, "exit_dim": r["peak_dim"],
                    "model": model_exit(prof, MU_EXIT)})
        print(out[-1])
    with open(os.path.join(RESULTS, "exitcalval.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
