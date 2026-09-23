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
  [dealii-fused-cg.md](dealii-fused-cg.md).
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

## Profiling of the $`Q_1`$ kernels (original single-file solver)

The measurements below predate the library split: the single-file `cgdbe`,
Ubuntu package of deal.II 9.5.1 (`VectorizedArray` width 1), GCC 13.3
`-O3 -march=native`, one core of a 2.1 GHz Xeon VM of a coding-assistant
sandbox. They are what the traffic and flop models of the end-of-run summary
([method.md](method.md#implementation-notes)) were read against; the models
are unchanged, the numbers are those of that machine. On that VM a numpy copy
ran at 14 GB/s, a triad at 8 GB/s.

The roofline itself is measured, not assumed (`benchmarks/roofline.cc`, one core):

| | working set of $`256^2`$ (0.5 MB/array, in L3) | DRAM-sized (32 MB/array) |
|---|---|---|
| scalar FMA peak, 8 independent chains | 10.1 GFlop/s | – |
| STREAM-like, 9 × (in → out) | 27 GB/s | 11 GB/s |
| collision pattern, 9 arrays in, 9 out (gather per node) | 20 GB/s | 12 GB/s |
| collision pattern, in place (as `bardow`) | 31 GB/s | 19 GB/s |

(The L3 of this VM is reported as 260 MB, so all cgdbe runs shown here are
cache-resident; the DRAM column is what a $`2000^2`$ mesh would see. A numpy copy
gives 14 GB/s, i.e. the DRAM figure.)

$`256^2`$ elements, 50 steps, one core, SIMD width 1:

| scheme, mass | stage | time | share | GB/s | GFlop/s | flop/byte |
|---|---|---|---|---|---|---|
| bardow, lumped | collision | 0.09 s | 5.0 % | 12.2 | 5.4 | 0.44 |
| | advection | 1.72 s | 93.5 % | 0.25 | 2.8 | 11.4 |
| | mass (lumped) | 0.03 s | 1.5 % | 15.2 | 1.0 | 0.06 |
| bardow, consistent | collision | 0.10 s | 1.8 % | 10.8 | 4.8 | 0.44 |
| | advection | 1.74 s | 30.5 % | 0.24 | 2.8 | 11.4 |
| | mass (CG, 1.3 it.) | 3.85 s | 67.6 % | 0.90 | 0.67 | 0.75 |
| leelin, consistent | collision | 0.15 s | 2.2 % | 9.3 | 6.7 | 0.71 |
| | advection | 2.74 s | 40.1 % | 0.23 | 2.5 | 10.7 |
| | mass (CG, 1.3 it.) | 3.93 s | 57.6 % | 0.89 | 0.67 | 0.75 |

Reading the two columns against the measured roofline: the in-cache ridge point
is 10.1 GFlop/s / 31 GB/s = 0.33 flop/byte, so the `bardow` collision (0.44
flop/byte in the single-pass model, 1.0 counted in place) is on the compute
side; it reaches 5.4 GFlop/s, 54 % of scalar peak, with the two divisions of
the moments and the dependency chains of the equilibrium accounting for the
rest. Its 12 GB/s is 40 % of the in-place pattern roofline. Vectorising the
nodal loop across nodes (AVX-512, 8 lanes) would raise the ceiling to about
80 GFlop/s and make the collision bandwidth-bound at ≈ 31 GB/s / 144 B =
0.2 GNUPS (since done, see the top of this page). The advection loop has an intensity of 11 flop/byte — far on the
compute side — and still reaches only 2.8 GFlop/s, 28 % of scalar peak, which
confirms that its cost is instruction overhead around the arithmetic (see
below), not flops and not bytes.

The 11 flop/byte itself is an artefact of the element-by-element formulation,
not a property of the operator. Per population and cell the model counts 184
flops: sum-factorised gradient at the 4 Gauss points (2 directions × 2 stages
× 8 multiply-adds) 64, inverse-Jacobian scaling 8, quadrature 24, integration
of the values 16 and of the gradients 64 + 8. The explicit $`4 \times 4`$ element
matrix would do the same action in 32 flops, and the assembled 9-point stencil
in 17 per node — the operator applied once per node instead of once per cell
with each node touched by four cells. So the numerator carries a factor of
about 10 of redundant arithmetic, while the denominator counts the minimal
traffic of one read and one write of the eight arrays; within the cache the
loop actually moves four reads and four read-modify-writes per cell and
population, i.e. its in-cache intensity is 184/96 ≈ 2 flop/byte. The
sum-factorised evaluation costs $`O(p)`$ per DoF and direction against
$`O(p^d)`$ for the element matrix, which is why the matrix-free action pays off
for $`p \gtrsim 3`$ and loses for $`Q_1`$. The CG mass solves are as slow in both
measures because each `vmult` is a cell loop with the same overhead per cell as
the advection loop but only 68 useful flops per cell. The stencil reference
(`benchmarks/stencil_bench.cc`) does 17 flops per node and direction, 3.8 GFlop/s at 3.5 GB/s:
the same discretisation at one tenth of the model flops and 15× the speed.

The nodal loops run at the memory bandwidth of the machine. The advection loop
does not: 0.54 µs per cell for 8 populations, about 70 ns per population per
cell. It is not a memory problem — the whole working set ($`f`$ and $`r`$, 8.5 MB
at $`256^2`$) sits in cache, and the `distribute_local_to_global` scatter is a
read-modify-write of four entries per cell whichever way the loop is organised.
Two experiments pin it down:

* `--per-direction` runs eight scalar cell loops (one destination array at a
  time, contiguous scatter) instead of the single 8-component loop: 1.5× *slower*
  (2.69 vs 1.75 s for 50 steps), because the per-cell work of `reinit`, index
  lookup and constraint handling is then paid eight times. The 8-component loop
  already amortises it.
* `benchmarks/stencil_bench.cc` applies the *same* discretisation ($`Q_1`$, lumped
  mass, uniform periodic grid) as the assembled 9-point stencil, direction by
  direction, structure of arrays, no FEM machinery: 5 ns per cell and direction,
  2.4 ms per step at $`256^2`$, 28 MNUPS — 15× the deal.II loop, on the same core
  and compiler.

A `gprof` build shows where the instructions go: `read_dof_values` and
`distribute_local_to_global` walk the DoFs one by one through the index map
(42 million `local_element` calls for 20 steps), with a vector-compatibility
check per component and cell; the sum-factorised arithmetic of a $`Q_1`$ cell is
a small part. This is the known trade-off of `FEEvaluation`: its gather/scatter
cost per DoF is meant to be amortised over vectorised cell batches and over the
arithmetic of higher degrees, and this build has neither (width-1
`VectorizedArray`, $`p = 1`$). Accordingly $`Q_2`$ already does 4.4 MNUPS against
1.8 for $`Q_1`$ at the same node count. With AVX-512 the loop processes 8 cells per
instruction stream and the index work is shared by the batch; that, and
$`p \geq 2`$, is where the factor to gain is. On a structured grid the stencil
form remains unbeatable, which is the point of the `dugks-mwe` code.

## Throughput (Taylor–Green, `scripts/run_perf.sh`, 200 steps, `leelin`, modes 1,1)

MNUPS = $`10^6`$ node updates (all 9 populations) per second. These numbers were
taken before the wall-face loop was added.

| mass | grid | nodes | ms/step | MNUPS | CG its/solve |
|---|---|---|---|---|---|
| consistent | $`32^2`$  | 1 089  | 3.47   | 0.31 | 3.6 |
| consistent | $`64^2`$  | 4 225  | 10.96  | 0.39 | 2.5 |
| consistent | $`128^2`$ | 16 641 | 39.05  | 0.43 | 2.0 |
| consistent | $`256^2`$ | 66 049 | 158.21 | 0.42 | 2.0 |
| lumped     | $`32^2`$  | 1 089  | 0.90   | 1.21 | – |
| lumped     | $`64^2`$  | 4 225  | 3.63   | 1.17 | – |
| lumped     | $`128^2`$ | 16 641 | 14.86  | 1.12 | – |
| lumped     | $`256^2`$ | 66 049 | 59.29  | 1.11 | – |

At $`256^2`$ (consistent): advection loop 35 %, mass solves 63 %, nodal loops 2 %.
The `bardow` scheme needs one instead of two gradient evaluations per population
in the advection loop (about 0.6 of the `leelin` loop time). Obvious next steps:
a SIMD-enabled deal.II build; a single block CG on an 8-component mass kernel.
