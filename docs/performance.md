# Performance notes of the lbfem library

Measurements from the modularization of `cgdbe` into `include/lbfem` (PR #2),
on one core of a Xeon with AVX-512 (VM, 260 MB L3). deal.II 9.5.1 from Ubuntu
24.04 has SIMD width 1; the `dealii/dealii:v9.7.1-noble` container used by the
CI has width 2 (SSE2). Numbers like these go stale; the scripts and programs
that produced them (`benchmarks/collision_simd`, `tools/mass_preconditioners`,
`tools/cg_fused_hooks`) are what to rerun.

## Library against the original single-file solver

- The advection kernels, written as lambdas at a quadrature point inlined into
  generic matrix-free loops, execute within 0.12 % of the instruction count
  (valgrind) of the original hand-written kernels; the results are
  byte-identical until the nodal loop was vectorized.
- The nodal loop (`nodal_map`) vectorized across nodes makes the collision
  1.7-2.4 times faster (4-lane AVX2, which GCC prefers here: 7 ns per node in
  cache, 9 ns and 16 GB/s out of cache, against 17 and 15 ns scalar). With the
  wall nodes computed apart, the loop reads no wall data and runs within 6 % of
  a hand-written `omp simd` loop of the same algorithm
  (`benchmarks/collision_simd`).
- After the vectorization the populations agree with the scalar original to
  ~1e-15 relative with lumped mass and to within the CG tolerance otherwise;
  the printed diagnostics except the ~1e-13 mass drift are identical.
- Stage timers local to each rank, without the `TimerOutput` sections that
  synchronized all ranks at every stage of every step, and the wall-node
  fix-up made a step 8-14 % faster (Q1, 128^2 and 512^2, one rank).

## Fused vector updates (`--fused`)

With `--fused`, the vector updates and reductions of CG (deal.II's `SolverCG`
detects the `vmult` of `StreamingMatrix` that takes operations on ranges of
the vectors) and of the Richardson passes run inside the cell loop, on each
range of entries just before the loop first touches it and just after it
last does, with $`Ap`$ or $`Mx`$ still in cache. Two findings:

- The preconditioner must not offer a per-entry `apply()`. `DiagonalMatrix`
  does, and then `SolverCG` takes a path that preconditions lane by lane
  into a SIMD register; with SIMD width 2 (deal.II 9.7.1) its vector
  updates cost ~9 ns per entry and iteration, in or out of cache and
  whatever the degree, against 2-7 ns unfused (at width 1 there is no
  penalty). The fused CG therefore gets a Jacobi with `apply_to_subrange()`
  only (`RangeJacobi`), which preconditions blocks of 128 entries; its
  updates then cost 2.4-5.3 ns. Details and a reproducer:
  [docs/dealii-fused-cg.md](docs/dealii-fused-cg.md).
- Fusing saves the memory traffic of the vector updates, which is a small
  part of an iteration unless the operator is cheap per dof and the vectors
  are out of cache. A $`Q_1`$ mass `vmult` costs 25-35 ns per dof with SSE2
  (~60 ns without SIMD), a $`Q_4`$ one 5-10 ns.

`lbfem::MassSolver`, ns per dof and iteration, unfused → fused, in the
`dealii/dealii:v9.7.1-noble` container (SSE2):

| | CG with $`M`$ | CG with TG3 | Richardson |
|---|---|---|---|
| $`Q_1`$, $`1.7 \cdot 10^4`$ dofs | 28.2 → 31.8 | 47.1 → 50.4 | 26.7 → 26.7 |
| $`Q_1`$, $`2.6 \cdot 10^5`$ dofs | 32.3 → 32.8 | 51.1 → 52.9 | 29.7 → 28.6 |
| $`Q_1`$, $`4.2 \cdot 10^6`$ dofs | 36.6 → 35.6 | 57.2 → 55.8 | 33.4 → 31.5 |
| $`Q_2`$, $`4.2 \cdot 10^6`$ dofs | 22.0 → 18.5 | 27.8 → 24.2 | 15.8 → 13.6 |
| $`Q_4`$, $`2.6 \cdot 10^5`$ dofs | 9.6 → 9.4 | 14.6 → 15.3 | 7.2 → 6.3 |
| $`Q_4`$, $`4.2 \cdot 10^6`$ dofs | 15.7 → 11.5 | 20.2 → 16.7 | 10.9 → 8.1 |

Without SIMD (the Ubuntu package, where both preconditioner paths are
equally fast) $`Q_1`$ is within ±4 %, $`Q_2`$ at
$`4.2 \cdot 10^6`$ dofs 9-13 % faster. In cgdbe runs, whose warm-started CG
takes 1.5-17 iterations per solve, the step time changes by -6 % to +12 %
up to $`10^6`$ dofs, so `--fused` is off by default; it is for higher degrees
and meshes well beyond the cache.

## Mass-matrix preconditioners

`tools/mass_preconditioners`: the Jacobi-preconditioned mass matrix has a
condition number of about 8.8 on all meshes tried (Wathen's bounds give
eigenvalues in [1/4, 9/4] for Q1), so CG needs ~25 iterations from zero at a
relative tolerance of 1e-8 and 2-16 from the extrapolated start, independently
of the mesh size; Chebyshev polynomials in the Jacobi-preconditioned operator
reduce the iterations but not the time.
