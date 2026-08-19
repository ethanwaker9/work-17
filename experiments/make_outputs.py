import json
import math
import statistics
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bellek.listsize import list_size_agps, list_size_ball, list_size_g6k

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PAPER = os.path.abspath(os.path.join(ROOT, "..", "final_paper"))
GEN = os.path.join(PAPER, "gen")
FIG = os.path.join(PAPER, "figures")

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 5.8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
    }
)

STYLE = {
    "Bellek-tight": ("#555555", "P", "-"),
    "G6K-default": ("#1b9e77", "o", "-"),
    "pBKZ": ("#d95f02", "s", "--"),
    "BKZ2.0": ("#7570b3", "^", "-."),
    "AsiaCCS23": ("#e7298a", "v", ":"),
    "PSSearch": ("#66a61e", "D", "--"),
    "Bellek": ("#111111", "*", "-"),
}
NAMEMAP = {
    "g6k": "G6K-default",
    "pbkz": "pBKZ",
    "bkz2": "BKZ2.0",
    "asiaccs": "AsiaCCS23",
    "pssearch": "PSSearch",
    "bellek": "Bellek",
    "bellekt": "Bellek-tight",
}


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        if name.endswith(".jsonl"):
            return [json.loads(x) for x in fh if x.strip()]
        return json.load(fh)


def save(fig, name):
    fig.tight_layout(pad=0.35)
    fig.savefig(os.path.join(FIG, name + ".eps"), format="eps", bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def w(name, text):
    with open(os.path.join(GEN, name), "w") as fh:
        fh.write(text)


def fmt(x, k=2):
    if x is None:
        return "---"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        e = int(math.floor(math.log10(abs(x))))
        return "$%.*f\\!\\cdot\\!10^{%d}$" % (k, x / 10.0 ** e, e)
    return ("%." + str(k) + "f") % x


def numbers_add(store, key, value):
    store[key] = value


def do_listsize(nums):
    data = load("listsize.json") or []
    extra = load("listsize_hi.json") or []
    seen = set()
    rows = []
    for r in sorted(list(data) + list(extra), key=lambda x: x["n"]):
        if r["n"] < 50 or r["n"] in seen:
            continue
        seen.add(r["n"])
        rows.append(r)
    if not rows:
        w("tab_listsize_main.tex", "")
        w("tab_listsize.tex", "")
        return
    ns = np.array([r["n"] for r in rows], dtype=float)
    meas = np.array([r["min_factor"] for r in rows])
    ours = np.array([r["ball"] for r in rows])
    agps = np.array([r["agps"] for r in rows])
    c_meas = float(np.mean(meas / np.sqrt(ns)))
    numbers_add(nums, "cmeas", "%.3f" % c_meas)
    numbers_add(nums, "ctheory", "%.3f" % (math.sqrt(6 * math.pi) / 12.0))
    numbers_add(nums, "listrelerr", "%.1f" % (100 * float(np.mean(np.abs(meas - ours) / ours))))
    ct = math.sqrt(6 * math.pi) / 12.0
    numbers_add(nums, "cerr", "%.1f" % (100 * abs(c_meas - ct) / ct))
    numbers_add(nums, "listagpsratio", "%.1f" % float(np.mean(agps / meas)))
    numbers_add(nums, "listnmin", str(int(ns[0])))
    numbers_add(nums, "listnmax", str(int(ns[-1])))

    body = []
    for r in rows:
        body.append(
            "%d & %.2f & %.2f & %.2f & %.2f & %.3f \\\\"
            % (r["n"], r["min_factor"], r["ball"], r["g6k"], r["agps"], r["min_factor"] / math.sqrt(r["n"]))
        )
    tab = (
        "\\begin{tabular}{rrrrrr}\n\\toprule\n$d$ & measured & $\\Nstar$ & G6K & AGPS20 & measured$/\\sqrt d$\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    w("tab_listsize_main.tex", tab)
    w("tab_listsize.tex", "\\begin{table}[H]\n\\centering\n\\caption{Least database size factor at which the \\texttt{bgj1} sieve of G6K still reaches its saturation target, against the three predictions, all divided by $(4/3)^{d/2}$.}\n\\label{tab:listfull}\n\\small\n" + tab + "\\end{table}\n")

    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.5))
    ax = axes[0]
    grid = np.arange(46, 128)
    ax.plot(grid, [list_size_agps(int(x)) / (4.0 / 3.0) ** (x / 2.0) for x in grid], color="#d95f02", ls="--", label="AGPS20 $2/C_d(\\pi/3)$")
    ax.plot(grid, [list_size_ball(int(x)) / (4.0 / 3.0) ** (x / 2.0) for x in grid], color="#111111", ls="-", label="$N^\\star$ (this work)")
    ax.plot(grid, [list_size_g6k(int(x)) / (4.0 / 3.0) ** (x / 2.0) for x in grid], color="#7570b3", ls="-.", label="G6K constant $3.2$")
    ax.plot(ns, meas, "o", color="#1b9e77", label="measured threshold")
    ax.set_yscale("log")
    ax.set_xlabel("sieving dimension $d$")
    ax.set_ylabel("list size $/\\,(4/3)^{d/2}$")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    ax.set_title("(a)", fontsize=7)

    ax = axes[1]
    ax.plot(ns, meas / ours, "o-", color="#111111", label="measured $/\\,N^\\star$")
    ax.plot(ns, meas / agps, "s--", color="#d95f02", label="measured $/\\,$AGPS20")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("sieving dimension $d$")
    ax.set_ylabel("ratio")
    ax.legend(loc="center right", frameon=False, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    ax.set_title("(b)", fontsize=7)
    save(fig, "fig_listsize")


def do_calib(nums):
    data = load("calibration_raw.json")
    ent = load("entrysize.json")
    if not data:
        w("tab_calib.tex", "")
        return
    body = []
    for r in data["pump"]:
        body.append("%d & %d & %.3f & %d \\\\" % (r["n"], r["d"], r["time"], r["db"]))
    tab = (
        "\\begin{table}[H]\n\\centering\n\\caption{Calibration of the cost model on the machine of Section~\\ref{sec:eval}. For each sieving dimension $n$ the table gives the rank $d$ of the lattice the pump was called on, the wall time of the pump and the size of the database it ended with.}\n\\label{tab:calib}\n\\small\n\\begin{tabular}{rrrr}\n\\toprule\n$n$ & $d$ & pump time (s) & entries\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    if ent:
        db = np.array([r["db"] for r in ent], dtype=float)
        by = np.array([r["peak"] - r["base"] for r in ent], dtype=float)
        A = np.vstack([db, np.ones_like(db)]).T
        sol, *_ = np.linalg.lstsq(A, by, rcond=None)
        numbers_add(nums, "measentry", "%.0f" % sol[0])
        numbers_add(nums, "measoverhead", "%.0f" % (sol[1] / 1024.0 ** 2))
        eb = []
        for r in ent:
            eb.append("%d & %d & %.1f \\\\" % (r["n"], r["db"], (r["peak"] - r["base"]) / 1024.0 ** 2))
        tab += (
            "\n\\begin{table}[H]\n\\centering\n\\caption{Resident memory of one sieve, measured in a fresh process for each dimension. A least squares fit of the last column against the number of entries gives $%s$ bytes per entry with a fixed overhead of $%s$ mebibytes.}\n\\label{tab:entry}\n\\small\n\\begin{tabular}{rrr}\n\\toprule\n$n$ & entries & resident (MiB)\\\\\n\\midrule\n"
            % (nums["measentry"], nums["measoverhead"])
            + "\n".join(eb)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
    w("tab_calib.tex", tab)
    cal = load("calibration.json")
    if cal:
        numbers_add(nums, "fitc", "%.3f" % cal["c_sieve"])


def _prefix(x):
    out = [0.0]
    a = 0.0
    for v in x:
        a += v
        out.append(a)
    return out


def do_sim(nums):
    import bellek.simulators as BS

    data = load("simaccuracy.json")
    if not data:
        w("tab_sim_main.tex", "")
        w("tab_simfull.tex", "")
        return
    fns = {
        "CN11": BS.cn11,
        "PnJBKZ": BS.pnjbkz_sim,
        "S5": BS.pnjbkz_sim_s5,
        "S6": BS.pnjbkz_sim_s6,
        "Bellek": BS.bellek_sim,
    }
    sims = list(fns)
    beta = data[0]["beta"]
    dim = data[0]["dim"]
    jumps = sorted(set(r["jump"] for r in data))
    per = {}
    noise = {}
    for j in jumps:
        rows = sorted([r for r in data if r["jump"] == j], key=lambda r: r["tour"])
        acc = dict((k, [0.0, 0.0]) for k in sims)
        nz = [0.0, 0.0]
        for t in range(1, len(rows)):
            prev, cur = rows[t - 1]["real"], rows[t]["real"]
            Pc, Pp = _prefix(cur), _prefix(prev)
            nz[0] += sum((Pc[i] - Pp[i]) ** 2 for i in range(41))
            nz[1] += sum((cur[i] - prev[i]) ** 2 for i in range(40))
            for k, fn in fns.items():
                p = fn(list(prev), beta, jump=j, tours=1)
                Pq = _prefix(p)
                acc[k][0] += sum((Pc[i] - Pq[i]) ** 2 for i in range(41))
                acc[k][1] += sum((cur[i] - p[i]) ** 2 for i in range(40))
        per[j] = acc
        noise[j] = nz

    body = []
    ratios = {}
    for j in jumps:
        vals = [per[j][k][0] for k in sims]
        body.append("%d & " % j + " & ".join("%.1f" % v for v in vals) + " & %.1f \\\\" % noise[j][0])
        for i, k in enumerate(sims[:-1]):
            ratios.setdefault(k, []).append(vals[i] / vals[-1])
    tab = (
        "\\begin{tabular}{rrrrrrr}\n\\toprule\n$J$ & CN11 & PnJBKZ & S5 & S6 & \\Bellek & no change\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    w("tab_sim_main.tex", tab)
    tot = dict((k, sum(per[j][k][0] for j in jumps)) for k in sims)
    tot2 = dict((k, sum(per[j][k][1] for j in jumps)) for k in sims)
    numbers_add(nums, "simbeta", str(beta))
    numbers_add(nums, "simdim", str(dim))
    numbers_add(nums, "simtours", str(max(r["tour"] for r in data)))
    numbers_add(nums, "simgaincn", "%.0f" % (tot["CN11"] / tot["Bellek"]))
    numbers_add(nums, "simgainpnj", "%.0f" % (tot["PnJBKZ"] / tot["Bellek"]))
    numbers_add(nums, "simgainsfive", "%.1f" % (tot["S5"] / tot["Bellek"]))
    numbers_add(nums, "simgainssix", "%.2f" % (tot["S6"] / tot["Bellek"]))
    numbers_add(nums, "simgainmin", "%.2f" % min(tot[k] / tot["Bellek"] for k in sims[:-1]))
    numbers_add(nums, "simgainmax", "%.0f" % max(tot[k] / tot["Bellek"] for k in sims[:-1]))
    numbers_add(nums, "simnoise", "%.2f" % sum(noise[j][1] for j in jumps))
    numbers_add(nums, "simptbest", "%.2f" % min(tot2.values()))

    fb = []
    for j in jumps:
        fb.append(
            "%d & " % j
            + " & ".join("%.1f" % per[j][k][0] for k in sims)
            + " & "
            + " & ".join("%.3f" % per[j][k][1] for k in sims)
            + " \\\\"
        )
    w(
        "tab_simfull.tex",
        "\\begin{table*}[t]\n\\centering\n\\caption{One-tour prediction error of five simulators on a challenge basis of rank %d reduced with $\\text{PnJBKZ}(%d,J)$, summed over the tours. The left group is the squared error on the partial log-volumes $P_k$ for $k\\le40$, the right group the squared error on the individual Gram--Schmidt norms for $i<40$.}\n\\label{tab:simfull}\n\\small\n\\begin{tabular}{r|rrrrr|rrrrr}\n\\toprule\n & \\multicolumn{5}{c|}{partial log-volumes} & \\multicolumn{5}{c}{Gram--Schmidt norms}\\\\\n$J$ & CN11 & PnJ & S5 & S6 & \\Bellek & CN11 & PnJ & S5 & S6 & \\Bellek\\\\\n\\midrule\n"
        % (dim, beta)
        + "\n".join(fb)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
    )

    jsel = jumps[-1]
    rows = sorted([r for r in data if r["jump"] == jsel], key=lambda r: r["tour"])
    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.55))
    ax = axes[0]
    if len(rows) >= 2:
        prev, cur = rows[-2]["real"], rows[-1]["real"]
        head = min(45, len(cur))
        idx = np.arange(head)
        ax.plot(idx, cur[:head], color="#111111", lw=1.4, label="measured")
        for nm, col, ls in [("CN11", "#7570b3", "-."), ("S6", "#d95f02", "--"), ("Bellek", "#1b9e77", ":")]:
            p = fns[nm](list(prev), beta, jump=jsel, tours=1)
            ax.plot(idx, p[:head], color=col, ls=ls,
                    label=nm)
        ax.set_xlabel("index $i$")
        ax.set_ylabel("$\\log\\|\\mathbf{b}^*_i\\|$")
        ax.legend(frameon=False, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    ax.set_title("(a)", fontsize=7)

    ax = axes[1]
    for nm, (col, mk, ls) in zip(sims, [("#7570b3", "^", "-."), ("#66a61e", "s", "--"),
                                        ("#e7298a", "v", ":"), ("#d95f02", "D", "--"),
                                        ("#111111", "*", "-")]):
        ax.plot(jumps, [per[j][nm][0] for j in jumps], marker=mk, ls=ls, color=col,
                label=nm)
    ax.set_yscale("log")
    ax.set_xlabel("jump $J$")
    ax.set_ylabel("head error on $P_k$")
    ax.legend(frameon=False, ncol=2, handlelength=1.1, borderpad=0.1, labelspacing=0.2,
              columnspacing=0.7, loc="best")
    ax.set_title("(b)", fontsize=7)
    save(fig, "fig_sim")


def do_search(nums):
    data = load("search.json")
    if not data:
        w("tab_search.tex", "")
        w("tab_searchfull.tex", "")
        return
    names = ["pBKZ", "BKZ2.0", "AsiaCCS23", "PSSearch", "Bellek"]
    body = []
    tr, mr = [], []
    for rec in data:
        cells = []
        for nm in names:
            r = rec.get(nm)
            if r is None:
                cells.append("--- & ---")
            else:
                mark = "$^{\\dagger}$" if r.get("completed", 1) == 0 else ""
                cells.append("%.2f%s & %.0f" % (r["walltime"], mark, r["peak_bytes"] / 1024.0))
        body.append("%d & " % rec["dim"] + " & ".join(cells) + " \\\\")
        if rec.get("PSSearch") and rec.get("Bellek"):
            tr.append(rec["PSSearch"]["walltime"] / max(rec["Bellek"]["walltime"], 1e-9))
            mr.append(rec["PSSearch"]["peak_bytes"] / max(rec["Bellek"]["peak_bytes"], 1))
    tab = (
        "\\begin{tabular}{r" + "rr" * len(names) + "}\n\\toprule\n & "
        + " & ".join("\\multicolumn{2}{c}{%s}" % (n if n != "Bellek" else "\\Bellek") for n in names)
        + "\\\\\n$d$ & "
        + " & ".join(["s & KiB"] * len(names))
        + "\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    w("tab_search.tex", tab)
    fb = []
    for rec in data:
        for nm in names:
            r = rec.get(nm)
            if r is None:
                continue
            fb.append(
                "%d & %s & %d & %d & %d & %.4g & %d & %s \\\\"
                % (rec["dim"], nm if nm != "Bellek" else "\\Bellek", r["frontier"],
                   r["sims"], r["dom_tests"], r["pred_time"], r["dsvp"],
                   "yes" if r.get("completed", 1) else "no")
            )
    w(
        "tab_searchfull.tex",
        "\\begin{table}[H]\n\\centering\n\\caption{Structure of the search. ``frontier'' is the number of states still stored when the search ends, ``sim.'' the number of simulator calls, ``dom.'' the number of profile comparisons, ``predicted'' the total time of the returned schedule under the calibrated model, $\\dsvp$ its final sieve dimension, and the last column says whether the search exhausted its space within the fifteen minute limit.}\n\\label{tab:searchfull}\n\\scriptsize\n\\begin{tabular}{rlrrrrrl}\n\\toprule\n$d$ & selector & frontier & sim. & dom. & predicted (s) & $\\dsvp$ & done\\\\\n\\midrule\n"
        + "\n".join(fb)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )
    fr = [(rec["PSSearch"]["frontier"], rec["Bellek"]["frontier"]) for rec in data
          if "PSSearch" in rec and "Bellek" in rec]
    dt2 = [(rec["PSSearch"]["dom_tests"], rec["Bellek"]["dom_tests"]) for rec in data
           if "PSSearch" in rec and "Bellek" in rec]
    bw = [rec["Bellek"]["walltime"] for rec in data if "Bellek" in rec]
    if bw:
        numbers_add(nums, "searchwallmax", "%.0f" % max(bw))
        numbers_add(nums, "searchwallmin", "%.0f" % min(bw))
    if tr:
        numbers_add(nums, "searchspeedmin", "%.1f" % min(tr))
        numbers_add(nums, "searchspeedmax", "%.0f" % max(tr))
    if fr:
        numbers_add(nums, "frontratio", "%.0f" % min(a / max(b, 1) for a, b in fr))
        numbers_add(nums, "domratio", "%.0f" % min(a / max(b, 1) for a, b in dt2))
        sg = [rec["PSSearch"]["pred_time"] / rec["Bellek"]["pred_time"] for rec in data
              if "PSSearch" in rec and "Bellek" in rec]
        numbers_add(nums, "schedgainmin", "%.0f" % (100 * (min(sg) - 1.0)))
        numbers_add(nums, "schedgainmax", "%.0f" % (100 * (max(sg) - 1.0)))
    if tr:
        numbers_add(nums, "searchspeedmin", "%.1f" % min(tr))
        numbers_add(nums, "searchspeedmax", "%.1f" % max(tr))
        numbers_add(nums, "searchmemmin", "%.1f" % min(mr))
        numbers_add(nums, "searchmemmax", "%.1f" % max(mr))

    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.5))
    ax = axes[0]
    dims = [rec["dim"] for rec in data]
    for nm in names:
        ys = [rec[nm]["walltime"] for rec in data if nm in rec]
        if len(ys) == len(dims):
            col, mk, ls = STYLE.get(nm, ("#000", "o", "-"))
            ax.plot(dims, ys, marker=mk, ls=ls, color=col, label=nm)
    ax.set_yscale("log")
    ax.set_xlabel("lattice rank $d$")
    ax.set_ylabel("search time (s)")
    ax.set_title("(a)", fontsize=7)
    handles, labels = ax.get_legend_handles_labels()
    ax = axes[1]
    for nm in names:
        ys = [rec[nm]["peak_bytes"] / 1024.0 for rec in data if nm in rec]
        if len(ys) == len(dims):
            col, mk, ls = STYLE.get(nm, ("#000", "o", "-"))
            ax.plot(dims, ys, marker=mk, ls=ls, color=col, label=nm)
    ax.set_yscale("log")
    ax.set_xlabel("lattice rank $d$")
    ax.set_ylabel("search memory (KiB)")
    ax.set_title("(b)", fontsize=7)
    fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), handlelength=1.1, columnspacing=0.9,
               handletextpad=0.35)
    save(fig, "fig_search")


def do_svp(nums):
    data = load("svp.jsonl")
    if not data:
        w("tab_svp.tex", "")
        w("tab_svpfull.tex", "")
        w("tab_schedules.tex", "")
        return
    dims = sorted(set(r["dim"] for r in data))
    order = ["g6k", "pbkz", "bkz2", "asiaccs", "pssearch", "bellek", "bellekt"]
    nseed = max(len(set(r["seed"] for r in data if r["dim"] == d)) for d in dims)

    def med(d, k, field):
        rs = [r[field] for r in data if r["dim"] == d and r["strategy"] == k]
        return statistics.median(rs) if rs else None

    def cnt(d, k):
        rs = [r for r in data if r["dim"] == d and r["strategy"] == k]
        return len(rs), sum(1 for r in rs if r["solved"])

    body = []
    fullbody = []
    speed, memr = [], []
    for d in dims:
        cells = []
        bt = med(d, "bellek", "solve_time")
        bm = med(d, "bellek", "peak_db")
        for k in order:
            t = med(d, k, "solve_time")
            if t is None:
                cells.append("--- & ---")
                continue
            m = med(d, k, "peak_db")
            cells.append("%.1f & %.1f" % (t, m / 1e3))
            n, ok = cnt(d, k)
            fullbody.append(
                "%d & %s & %.2f & %.1f & %.0f & %d & %.2f & %.4f & %d/%d \\\\"
                % (d, NAMEMAP[k], med(d, k, "plan_time"), t, med(d, k, "peak_dim"), m,
                   med(d, k, "rss") / 1024.0 ** 3, med(d, k, "r0_over_gh"), ok, n)
            )
            if k not in ("bellek", "bellekt") and bt:
                if d >= 80:
                    speed.append(t / bt)
                if d == dims[-1]:
                    memr.append(m / max(bm, 1))
        body.append("%d & " % d + " & ".join(cells) + " \\\\")
    tab = (
        "\\begin{tabular}{r" + "rr" * len(order) + "}\n\\toprule\n & "
        + " & ".join("\\multicolumn{2}{c}{%s}" % (NAMEMAP[k] if k != "bellek" else "\\Bellek") for k in order)
        + "\\\\\n$d$ & "
        + " & ".join(["s & $10^{3}$ vec."] * len(order))
        + "\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    w("tab_svp.tex", tab)
    sb = []
    for d in dims:
        for k in order:
            rs = [r for r in data if r["dim"] == d and r["strategy"] == k and r["seed"] == 0]
            if not rs:
                continue
            st = rs[0]["steps"]
            if st == "workout" or not st:
                sb.append("%d & %s & \\multicolumn{1}{l}{no reduction tour, pumps of sieving dimension $40,\\dots,%d$} \\\\"
                          % (d, NAMEMAP[k], rs[0]["dsvp"]))
                continue
            txt = ", ".join("$(%d,%d)$" % (x[0], x[1]) for x in st[:9])
            if len(st) > 9:
                txt += ", \\dots"
            sb.append("%d & %s & %s \\\\" % (d, NAMEMAP[k], txt))
    w(
        "tab_schedules.tex",
        "\\begin{table*}[t]\n\\centering\n\\caption{Schedules returned by the selectors on the first of the five bases of each rank, as sequences of block size and jump pairs, followed in every case by the final pump phase. A selector that returns no reduction tour is described by the range of its pump dimensions.}\n\\label{tab:schedules}\n\\small\n\\begin{tabular}{rll}\n\\toprule\n$d$ & selector & schedule\\\\\n\\midrule\n"
        + "\n".join(sb)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
    )
    w(
        "tab_svpfull.tex",
        "\\begin{table}[H]\n\\centering\n\\caption{Full record of the solving runs, as medians over the %d bases of each rank. ``plan'' is the wall time of the selection, ``solve'' the wall time of the schedule, $n_{\\max}$ the largest sieving dimension the run opened, ``vectors'' the largest database held, ``resident'' the peak resident set size, the next column the ratio of the norm found to the Gaussian heuristic of the lattice and the last one the number of bases on which the target was met.}\n\\label{tab:svpfull}\n\\scriptsize\n\\begin{tabular}{rlrrrrrrr}\n\\toprule\n$d$ & selector & plan (s) & solve (s) & $n_{\\max}$ & vectors & resident (GiB) & $\\|\\mathbf{b}_0\\|/\\gh$ & solved\\\\\n\\midrule\n"
        % nseed
        + "\n".join(fullbody)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )
    if speed:
        base = [k for k in order if k not in ("bellek", "bellekt")]
        gaps, plans = [], []
        for d in dims:
            bt = med(d, "bellek", "solve_time")
            bp = med(d, "bellek", "plan_time")
            bestt = min(med(d, k, "solve_time") for k in base if med(d, k, "solve_time"))
            if d >= 80:
                gaps.append(100.0 * (bt / bestt - 1.0))
            plans.append(med(d, "pssearch", "plan_time"))
        numbers_add(nums, "svpspeedmin", "%.1f" % min(speed))
        numbers_add(nums, "svpspeedmax", "%.1f" % max(speed))
        numbers_add(nums, "svpmemmin", "%.1f" % min(memr))
        numbers_add(nums, "svpmemmax", "%.1f" % max(memr))
        numbers_add(nums, "svpdmin", str(dims[0]))
        numbers_add(nums, "svpdmax", str(dims[-1]))
        numbers_add(nums, "svpseeds", str(nseed))
        numbers_add(nums, "svpgapmax", "%.0f" % max(gaps))
        numbers_add(nums, "svpgainbest", "%.0f" % abs(min(gaps)))
        numbers_add(nums, "svpplanpss", "%.0f" % plans[-1])
        numbers_add(nums, "svpplanours", "%.0f" % med(dims[-1], "bellek", "plan_time"))

    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.55))
    for idx, field in enumerate(["solve_time", "peak_db"]):
        ax = axes[idx]
        for k in order:
            xs = [d for d in dims if med(d, k, field) is not None]
            ys = [med(d, k, field) for d in xs]
            if not xs:
                continue
            col, mk, ls = STYLE[NAMEMAP[k]]
            ax.plot(xs, ys, marker=mk, ls=ls, color=col, label=NAMEMAP[k])
        ax.set_yscale("log")
        ax.set_xlabel("SVP challenge dimension $d$")
        ax.set_ylabel("wall time (s)" if idx == 0 else "peak database entries")
        ax.set_title("(a)" if idx == 0 else "(b)", fontsize=7)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), handlelength=1.1, columnspacing=0.9,
               handletextpad=0.35)
    save(fig, "fig_svp")


def do_nist(nums):
    data = load("nist.json")
    if not data:
        w("tab_nist_main.tex", "")
        w("tab_nist.tex", "")
        return
    body = []
    for r in data["budgets"]:
        body.append(
            "$2^{%d}$ & %d & %d & %d & %.1f & %.1f \\\\"
            % (
                r["log2_bits"],
                r["AGPS20"]["dim"],
                r["G6K"]["dim"],
                r["Ours"]["dim"],
                r["AGPS20"]["gates_log2"],
                r["Ours"]["gates_log2"],
            )
        )
    tab = (
        "\\begin{tabular}{rrrrrr}\n\\toprule\nmemory & \\multicolumn{3}{c}{largest sieving dimension} & \\multicolumn{2}{c}{$\\log_2$ gates}\\\\\n(bits) & AGPS20 & G6K & ours & AGPS20 & ours\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    w("tab_nist_main.tex", tab)
    sb = []
    for r in data["schemes"]:
        sb.append(
            "%s & %d & %d & %d & %.1f & %.1f & %.1f \\\\"
            % (
                r["scheme"],
                r["d"],
                r["beta"],
                r["sieve_dim"],
                r["gates_log2"],
                r["AGPS20_mem_bits"],
                r["Ours_mem_bits"],
            )
        )
    w(
        "tab_nist.tex",
        "\\begin{table}[H]\n\\centering\n\\caption{Primal attack on the deployed standards. $d$ is the rank of the embedding lattice, $\\beta$ the least block size that meets the standard success condition, and the last two columns give the $\\log_2$ of the memory in bits required by the final sieve under the two list-size laws.}\n\\label{tab:nist}\n\\small\n\\begin{tabular}{lrrrrrr}\n\\toprule\nscheme & $d$ & $\\beta$ & sieve dim & $\\log_2$ gates & AGPS20 & ours\\\\\n\\midrule\n"
        + "\n".join(sb)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )
    b140 = [r for r in data["budgets"] if r["log2_bits"] == 140]
    if b140:
        numbers_add(nums, "dimagps", str(b140[0]["AGPS20"]["dim"]))
        numbers_add(nums, "dimours", str(b140[0]["Ours"]["dim"]))
        numbers_add(nums, "gatesagps", "%.1f" % b140[0]["AGPS20"]["gates_log2"])
        numbers_add(nums, "gatesours", "%.1f" % b140[0]["Ours"]["gates_log2"])
    if data["schemes"]:
        r = data["schemes"][0]
        numbers_add(nums, "kemmemagps", "%.1f" % r["AGPS20_mem_bits"])
        numbers_add(nums, "kemmemours", "%.1f" % r["Ours_mem_bits"])

    fig, ax = plt.subplots(figsize=(3.15, 1.45))
    bits = [r["log2_bits"] for r in data["budgets"]]
    ax.plot(bits, [r["AGPS20"]["dim"] for r in data["budgets"]], "s--", color="#d95f02", label="AGPS20")
    ax.plot(bits, [r["G6K"]["dim"] for r in data["budgets"]], "^-.", color="#7570b3", label="G6K")
    ax.plot(bits, [r["Ours"]["dim"] for r in data["budgets"]], "*-", color="#111111", label="$N^\\star$ (this work)")
    ax.set_xlabel("memory budget ($\\log_2$ bits)")
    ax.set_ylabel("largest sieving dimension")
    ax.legend(frameon=False)
    save(fig, "fig_memdim")


def do_mu(nums):
    import bellek.simulators as BS

    numbers_add(nums, "muval", ("%g" % BS.MU))
    numbers_add(nums, "muliftval", ("%g" % BS.MU_LIFT))
    numbers_add(nums, "murecval", ("%g" % BS.MU_REC))
    data = load("mucalib.json")
    best = load("mubest.json") or {}
    if not data:
        w("tab_mu.tex", "")
        return
    body = []
    for row in data:
        body.append(("%g & " % row[0]) + " & ".join("%.1f" % v for v in row[1:]) + " \\\\")
    w(
        "tab_mu.tex",
        "\\begin{table}[H]\n\\centering\n\\caption{Calibration of the constants of Remark~\\ref{rem:mu}. Each entry is the squared one-tour prediction error on the partial log-volumes of the first $40$ indices, summed over the four jumps and the tours, for the recording efficiency $\\mu$ in the rows and the lift correction $\\mu_{\\mathrm{lift}}$ in the columns, at the value $\\mu_{\\mathrm{rec}}=\\nummurecval$ that minimises the error.}\n\\label{tab:mu}\n\\small\n\\begin{tabular}{rrrr}\n\\toprule\n & \\multicolumn{3}{c}{$\\mu_{\\mathrm{lift}}$}\\\\\n$\\mu$ & $10$ & $30$ & $100$\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )


def do_eps(nums):
    data = load("epssweep.json")
    if not data:
        w("tab_eps.tex", "")
        return
    done = [r for r in data["sweep"] if r.get("completed", 1)]
    base = sorted(done, key=lambda r: r["eps"])[:1]
    t0 = base[0]["pred_time"] if base else data["sweep"][0]["pred_time"]
    body = []
    for r in sorted(data["sweep"], key=lambda x: -x["eps"]):
        mark = "" if r.get("completed", 1) else "$^{\\dagger}$"
        body.append(
            "%.3f & %.1f%s & %d & %d & %.4g & %+.2f \\\\"
            % (r["eps"], r["walltime"], mark, r["frontier"], r["sims"], r["pred_time"],
               100.0 * (r["pred_time"] / t0 - 1.0))
        )
    w(
        "tab_eps.tex",
        "\\begin{table}[H]\n\\centering\n\\caption{Effect of the domination tolerance $\\varepsilon$ on the search at rank $d=%d$. The columns are the wall time, the size of the stored frontier, the number of simulator calls, the predicted total time of the returned schedule and its excess in percent over the reference, which is the smallest tolerance whose search terminates. A dagger marks a search stopped at the fifteen minute limit.}\n\\label{tab:eps}\n\\small\n\\begin{tabular}{rrrrrr}\n\\toprule\n$\\varepsilon$ & time (s) & $W$ & sim.\\ calls & predicted (s) & excess (\\%%)\\\\\n\\midrule\n"
        % data["dim"]
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )
    zero = [r for r in data["sweep"] if r["eps"] == 0.0]
    if zero:
        numbers_add(nums, "epsfrontexact", str(zero[0]["frontier"]))
        numbers_add(nums, "epsexactexcess", "%.0f" % (100.0 * (zero[0]["pred_time"] / t0 - 1.0)))
    sel = [r for r in data["sweep"] if abs(r["eps"] - 0.008) < 1e-9]
    if base and sel:
        numbers_add(nums, "epsexcess", "%.2f" % abs(100 * (sel[0]["pred_time"] / t0 - 1.0)))
        numbers_add(nums, "epsspeed", "%.1f" % (base[0]["walltime"] / max(sel[0]["walltime"], 1e-9)))
        numbers_add(nums, "frontierW", str(sel[0]["frontier"]))
        numbers_add(nums, "epsused", "%.3f" % sel[0]["eps"])


def _model_exit(prof, mu, n0=40):
    from bellek.simulators import _apply_pump
    from bellek.svpdim import reachable_head_norm, target_norm_svp_challenge

    d = len(prof)
    lt = target_norm_svp_challenge(prof)
    l = list(prof)
    for n in range(n0, d + 1):
        if reachable_head_norm(l, 0, d - n, mu=mu) <= lt:
            return n
        _apply_pump(l, 0, d, d - n, down_stop_extra=0)
    return d + 1


def do_budget(nums):
    data = load("budget.jsonl")
    if not data:
        w("tab_budget.tex", "")
        return
    dims = sorted(set(r["dim"] for r in data))
    body = []
    slow = []
    for d in dims:
        ref = [r for r in data if r["dim"] == d and r["strategy"] == "g6k"]
        if not ref:
            continue
        rt = statistics.median([r["solve_time"] for r in ref])
        rm = statistics.median([r["peak_db"] for r in ref])
        body.append("%d & G6K-default & --- & %.1f & %d & --- \\\\" % (d, rt, rm))
        for ratio in sorted(set(r["ratio"] for r in data if r["strategy"] == "bellek"), reverse=True):
            rs = [r for r in data if r["dim"] == d and r["strategy"] == "bellek"
                  and abs(r["ratio"] - ratio) < 1e-9 and r["solve_time"] is not None]
            if not rs:
                continue
            t = statistics.median([r["solve_time"] for r in rs])
            m = statistics.median([r["peak_db"] for r in rs])
            ok = sum(1 for r in rs if r["within"] and r["solved"])
            body.append("%d & \\Bellek & %.2f & %.1f & %d & %d/%d \\\\"
                        % (d, ratio, t, m, ok, len(rs)))
            if ratio < 0.9:
                slow.append(t / rt)
    w(
        "tab_budget.tex",
        "\\begin{table}[H]\n\\centering\n\\caption{Solving under a prescribed memory budget. The budget is a fraction of the peak database that the default strategy of G6K needs on the same basis, the columns are that fraction, the median wall time, the median peak database and the number of runs that met both the budget and the target, and the first row of each rank is the unconstrained reference.}\n\\label{tab:budget}\n\\small\n\\begin{tabular}{rlrrrr}\n\\toprule\n$d$ & selector & budget & time (s) & vectors & within\\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )
    if slow:
        numbers_add(nums, "budgetslowmin", "%.1f" % min(slow))
        numbers_add(nums, "budgetslowmax", "%.1f" % max(slow))


def do_exitcal(nums):
    from bellek.simulators import MU_EXIT

    cal = load("exitdim.jsonl") or []
    if not cal:
        w("tab_exitcal.tex", "")
        return
    val = load("exitcalval.json") or []
    rowsc = []
    for r in sorted(cal, key=lambda x: (x["dim"], x["seed"])):
        m = _model_exit(r["profile"], MU_EXIT)
        rowsc.append((r["dim"], r["seed"], r["exit_dim"], m))
    sgn = lambda v: ("%+d" % v) if v else "0"
    body = ["%d & %d & %d & %d & %s \\\\" % (a, b, c, e, sgn(e - c)) for (a, b, c, e) in rowsc]
    body.append("\\midrule")
    rowsv = []
    for r in sorted(val, key=lambda x: x["dim"]):
        rowsv.append((r["dim"], r["seed"], r["exit_dim"], r["model"]))
        body.append("%d & %d & %d & %d & %s \\\\" % (r["dim"], r["seed"], r["exit_dim"], r["model"], sgn(r["model"] - r["exit_dim"])))
    rc = math.sqrt(sum((e - c) ** 2 for (_, _, c, e) in rowsc) / len(rowsc))
    w(
        "tab_exitcal.tex",
        "\\begin{table}[H]\n\\centering\n\\caption{Calibration and validation of the exit constant $\\mu_{\\mathrm{exit}}$. Each row is one challenge basis, and the columns are the rank, the randomisation seed, the largest sieving dimension the reference progressive sequence actually reached before the target was met, the dimension the exit test of Theorem~\\ref{thm:lift} predicts with $\\mu_{\\mathrm{exit}}=%g$, and their difference. The rows above the rule are the ranks on which the constant was fitted and the rows below it are the ranks of Section~\\ref{sec:svpsolve}, which were not used in the fit.}\n\\label{tab:exitcal}\n\\small\n\\begin{tabular}{rrrrr}\n\\toprule\n$d$ & seed & measured & predicted & difference\\\\\n\\midrule\n"
        % MU_EXIT
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )
    numbers_add(nums, "exitmu", "%g" % MU_EXIT)
    numbers_add(nums, "exitrms", "%.1f" % rc)
    if rowsv:
        rv = math.sqrt(sum((e - c) ** 2 for (_, _, c, e) in rowsv) / len(rowsv))
        numbers_add(nums, "exitvalrms", "%.1f" % rv)
        numbers_add(nums, "exitvalmax", "%d" % max(abs(e - c) for (_, _, c, e) in rowsv))


def do_pareto(nums):
    data = load("pareto.json")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(3.15, 1.5))
    xs = [p[0] / 1024.0 ** 3 for p in data["curve"]]
    ys = [p[1] for p in data["curve"]]
    ax.step(xs, ys, where="post", color="#111111", lw=1.4, label="Bellek frontier")
    ax.plot(xs, ys, marker="o", ms=3.0, ls="none", color="#111111")
    for nm, pt in sorted(data["points"].items()):
        col, mk, _ = STYLE.get(nm, ("#000", "o", "-"))
        ax.plot([pt[1] / 1024.0 ** 3], [pt[0]], marker=mk, ls="none", ms=5, color=col,
                markeredgecolor="black", markeredgewidth=0.4, label=nm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("peak sieve memory (GiB)")
    ax.set_ylabel("predicted solving time (s)")
    ax.grid(True, which="major", ls=":", lw=0.4, color="#999999")
    ax.margins(x=0.12, y=0.18)
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.16),
               handlelength=1.2, columnspacing=0.9, handletextpad=0.35)
    save(fig, "fig_pareto")
    if data.get("dim"):
        numbers_add(nums, "paretodim", str(data["dim"]))
    if len(xs) > 1:
        numbers_add(nums, "paretorange", "%.0f" % (max(ys) / min(ys)))
        numbers_add(nums, "paretomemmin", "%.0f" % min(xs))
        numbers_add(nums, "paretomemmax", "%.0f" % max(xs))


def main():
    os.makedirs(GEN, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    nums = {}
    do_listsize(nums)
    do_calib(nums)
    do_sim(nums)
    do_search(nums)
    do_svp(nums)
    do_nist(nums)
    do_mu(nums)
    do_eps(nums)
    do_exitcal(nums)
    do_pareto(nums)
    expected = [
        "cmeas", "ctheory", "listrelerr", "cerr", "listagpsratio", "listnmin", "listnmax",
        "fitc",
        "simgainmin", "simgainmax", "simgaincn", "simgainpnj", "simgainsfive", "simgainssix",
        "simbeta", "simdim", "simtours", "simnoise", "simptbest",
        "searchspeedmin", "searchspeedmax", "searchmemmin", "searchmemmax",
        "searchwallmax", "searchwallmin",
        "svpspeedmin", "svpspeedmax", "svpmemmin", "svpmemmax", "svpdmin", "svpdmax",
        "dimagps", "dimours", "gatesagps", "gatesours", "kemmemagps", "kemmemours",
        "epsexcess", "epsspeed", "frontierW", "epsused", "epsfrontexact", "epsexactexcess", "measentry", "measoverhead",
        "frontratio", "domratio", "schedgainmin", "schedgainmax",
        "exitmu", "exitrms", "exitvalrms", "exitvalmax", "svpseeds",
        "muval", "muliftval", "murecval", "paretodim", "paretorange", "paretomemmin", "paretomemmax",
        "svpgapmax", "svpgainbest", "svpplanpss", "svpplanours",
    ]
    for k in expected:
        nums.setdefault(k, "0")
    lines = ["\\newcommand{\\num%s}{%s\\xspace}" % (k, v) for k, v in sorted(nums.items())]
    w("numbers.tex", "\n".join(lines) + "\n")
    print("generated", len(nums), "numbers")


if __name__ == "__main__":
    main()
