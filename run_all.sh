#!/bin/bash
set -e
cd "$(dirname "$0")"
THREADS=${THREADS:-8}
SEEDS=${SEEDS:-"0 1 2 3 4"}

python3 -u experiments/exp1_listsize.py --dims 50,54,58,62,66 --seeds 3 --threads $THREADS
python3 -u experiments/exp2b_entry.py --dims 55,60,65,70 --threads $THREADS
python3 -u experiments/exp2_calibrate.py --dims 50,55,60,65,70,75,80,85 --threads $THREADS
python3 -u experiments/fit_calibration.py
python3 -u experiments/exp3_simulator.py --dim 120 --beta 75 --jumps 1,4,8,12 --tours 4 --warmup 50 --threads $THREADS
python3 -u experiments/fit_mu.py

rm -f results/exitdim.jsonl
for d in 60 65 75 85 95; do
  for s in 0 1; do
    python3 -u experiments/exp4c_exit.py --dim $d --seed $s --threads $THREADS
  done
done

rm -f results/svp.jsonl
for seed in $SEEDS; do
  for d in 70 80 90 100; do
    for s in bellek bellekt g6k asiaccs pssearch bkz2 pbkz; do
      python3 -u experiments/exp4_svp.py --strategy $s --dim $d --seed $seed --threads $THREADS --budget 4e9
    done
  done
done
python3 -u experiments/exp4d_val.py

python3 -u experiments/exp5_search.py --dims 100,120,140,160 --sweepdim 140 --cap 900
python3 -u experiments/exp5b_pareto.py --dim 140 --cap 120 --nmin 104 --nmax 120 --step 2
python3 -u experiments/exp6_nist.py
python3 -u experiments/make_outputs.py
