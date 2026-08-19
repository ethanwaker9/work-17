# Efficient Sieving Strategies Under the Practical Memory Budget

This repository contains the ball model list size law for lattice sieves, a head accurate simulator for pump and jump BKZ, our `Bellek` as a memory budgeted optimal strategy search, and implementations of core of strategy selector techniques that we compare against, inside one framework.

## Requirements
Python 3.12, numpy 2.3, scipy 1.16, matplotlib 3.10, fpylll 0.6.4, g6k 0.1.2. In fact, `fpylll` and `g6k` are needed only for the experiments that touch real lattices (`exp1`, `exp2`, `exp3`, `exp4`). The library itself, the simulators, the cost model and `Bellek` depend only on numpy and scipy, so `exp5` and `exp6` run without G6K.

### Installing G6K

```
pip install cython numpy requests
git clone https://github.com/fplll/g6k
cd g6k
autoreconf -i && ./configure && make -j8
python setup.py build_ext --inplace -j8
pip install .
```

On Apple silicon add `-I$(xcrun --show-sdk-path)/usr/include/c++/v1` to `CXXFLAGS`,
export `ARCHFLAGS="-arch arm64"`, and guard the two `#include <immintrin.h>` lines of
`kernel/simd.h` and `kernel/bdgl_sieve.cpp` with `#if defined(__x86_64__)`.

## Implementation content

`geometry.py` contains  ball volumes, cap measures, Gaussian heuristic, profiles; `listsize.py` has reducing pair probability, N*, AGPS20 and G6K laws, memory; `costmodel.py` consists of sieve, pump and PnJBKZ time and memory models; `simulators.py` contains CN11, PnJBKZ, the two AsiaCCS'23 (WWW23) variants, and ours; `svpdim.py` has exit dimension estimators, including the cylinder rule; `strategies.py` G6K default, pBKZ, BKZ 2.0, AsiaCCS'23 trade-off, PSSearch; `challenges.py` has Darmstadt SVP challenge download and loading.

## Running the experiments

`run_all.sh` performs the whole sequence in the right order; the individual drivers are

```
python experiments/exp1_listsize.py  --dims 50,54,58,62,66 --seeds 3 --threads 8
python experiments/exp2b_entry.py    --dims 55,60,65,70 --threads 8
python experiments/exp2_calibrate.py --dims 50,55,60,65,70,75,80,85 --threads 8
python experiments/fit_calibration.py
python experiments/exp3_simulator.py --dim 120 --beta 75 --jumps 1,4,8,12 --tours 4 --warmup 50
python experiments/fit_mu.py
python experiments/exp4c_exit.py     --dim 85 --seed 0 --threads 8
python experiments/exp4_svp.py       --strategy bellek --dim 90 --seed 0 --threads 8 --budget 4e9
python experiments/exp4d_val.py
python experiments/exp5_search.py    --dims 100,120,140,160 --sweepdim 140 --cap 900
python experiments/exp5b_pareto.py   --dim 140 --nmin 104 --nmax 120 --step 2
python experiments/exp6_nist.py
python experiments/make_outputs.py
```

`parse_calibration.py` rebuilds `results/calibration_raw.json` from the log of a calibration run
that was interrupted, and `fit_calibration.py` refits the cost model from that file.
`make_outputs.py` writes the LaTeX tables into `../final_paper/gen` and the figures, in EPS , into `../final_paper/figures`. `summarise.py` prints the solving results as medians.

## Our experiments measurements

**exp1** binary searches, for every dimension and three random lattices, the least database
size factor at which the `bgj1` sieve still reaches its saturation target, and shows the
median. This is the measurement of the list size law.

**exp2** times one pump for a range of sieving dimensions and fits the exponent of the cost
model; **exp2b** measures the resident memory of one sieve in a fresh process per dimension and
divides by the database size to obtain the bytes per entry. The fit is written to
`results/calibration.json` and is picked up automatically by `CostModel.load()`.

**exp3** reduces a challenge basis with several tours of pump and jump BKZ and shows the
Gram--Schmidt profile after each tour. `make_outputs.py` then scores every simulator on the
one tour prediction, that is on its ability to map the measured profile after one tour to the
measured profile after the next, using the squared error on the partial log volumes.
**fit_mu** rescores the same predictions over a grid of the three constants of the insertion
rule and writes the grid and its minimiser.

**exp4c** runs the progressive pump sequence of G6K on challenge bases of ranks disjoint from
those of exp4 and records the largest sieving dimension it opens before meeting the target,
which is the measurement the exit constant is fitted to; **exp4d** replays the same prediction
on the exp4 bases as a validation set.

**exp4** runs one selector on one challenge dimension in a fresh process, executes the schedule
it returns inside G6K, and records the wall time, the largest database held at any moment and
the peak resident set size. A background thread samples the database size every five
milliseconds, so the memory figure is a true peak and not a value read after the fact. If the
planned final pump misses the target the run continues with pumps of increasing dimension, so
an optimistic selector pays for its optimism. `bellekt` is the same search given the memory
that the default strategy of G6K needs on that instance.

**exp5** calls every selector on the same profile with the same cost model and records the wall
time and the peak allocation of the selection itself, using `tracemalloc` in a separate pass so
that the tracer does not inflate the timing. The budget given to the search is the memory the
unconstrained schedule of that rank needs, so the constraint is active without making the
instance infeasible.

**exp5b** sweeps the memory budget at one rank and records the least predicted time the search
returns for each of them, which is the frontier of the trade-off figure, and scores the schedule
that every other selector returns under the same cost model, simulator and exit rule so that the
points and the frontier are commensurable.

**exp6** converts memory budgets into the largest sieving dimension that fits, under the three
list-size laws, and applies the same computation to the six deployed NIST parameter sets.

## Reproducibility notes

Lattices come from the TU Darmstadt SVP challenge generator and are cached under `svpchallenge/`.
Randomization uses the seed passed on the command line, so a run is reproducible up to the
nondeterminism of the multithreaded sieve. Timings depend on the machine; rerun `exp2` before
`exp4` and `exp5` so that all selectors share a calibrated model.
