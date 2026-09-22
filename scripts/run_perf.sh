#!/bin/sh
# Quick throughput scan: fixed number of steps, consistent vs. lumped mass.
#   usage: ./run_perf.sh [path/to/cgdbe] [mpirun -np N ...]
exe=${1:-./cgdbe}; [ $# -gt 0 ] && shift
steps=200

printf "%-11s %8s %9s %10s %9s %8s\n" mass grid nodes ms/step MNUPS CG-its
for mass in consistent lumped; do
  [ $mass = lumped ] && flag="--mass lumped" || flag=
  for r in 5 6 7 8; do
    "$@" "$exe" --refine $r --steps $steps --no-output $flag > perf_${mass}_$r.log
    awk -v m=$mass -v n=$((1 << r)) '
      /nodes x/       { nodes = $5 }
      /time stepping :/ { ms = $(NF-1); sub(/\(/, "", ms) }
      /throughput/    { mnups = $3 }
      /CG iterations/ { its = $4 }
      END { printf "%-11s %4dx%-4d %8d %10.3f %9.3f %8s\n", m, n, n, nodes, ms, mnups, (its == "" ? "-" : its) }
    ' perf_${mass}_$r.log
  done
done
