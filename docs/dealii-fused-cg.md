# deal.II: fused `SolverCG` is ~3x slower in its vector updates with `DiagonalMatrix`

## Summary

`SolverCG` runs its vector updates and reductions inside the matrix-vector
product (a "fused" iteration) when the matrix has a `vmult(dst, src,
operation_before, operation_after)` overload. It then has two code paths,
chosen by what the preconditioner offers:

- `apply(index, value)`: preconditions entry by entry, one SIMD lane at a time,
  inside loops over `VectorizedArray` (the path the documentation calls "more
  optimized", and the one taken when both functions exist);
- `apply_to_subrange(begin, end, src, dst)`: preconditions blocks of 128
  entries, then runs plain loops.

`DiagonalMatrix`, the natural Jacobi preconditioner, has both, so it gets the
first path. With `VectorizedArray<double>` of width 2 (deal.II 9.7.1, SSE2)
that path is 2.5-4x slower per entry than the second one, in or out of cache.
In a full fused CG, whose point is to save a few ns per entry of vector
traffic, the extra cost exceeds the saving: the fused solve becomes slower than
the unfused one. At width 1 both paths are equally fast.

## Reproducer

[`tools/cg_fused_hooks.cc`](../tools/cg_fused_hooks.cc) (built with
`-DLBFEM_BUILD_EXTRAS=ON`, or as a plain deal.II program). It creates
`internal::SolverCG::IterationWorker` for a dummy matrix and calls its
`operation_before_loop` / `operation_after_loop` on 64-entry chunks, as
`MatrixFree::cell_loop` does, for

- `DiagonalMatrix<Vector>` (per-entry `apply()` path), and
- a wrapper around the same `DiagonalMatrix` exposing only `vmult()` and
  `apply_to_subrange()`.

No matrix-vector product is involved; the difference is the preconditioner
path alone.

## Results

ns per vector entry, one call per 64-entry chunk, best of 20 (one core, Xeon
with AVX-512; the container's deal.II is built for SSE2):

**deal.II 9.7.1** (`dealii/dealii:v9.7.1-noble`), `VectorizedArray<double>`
width 2, compiled `-O2 -fopenmp-simd`:

| entries | path | before, odd iteration | before, even iteration | after (7 sums) |
|---|---|---|---|---|
| 4 096 | `apply()` (DiagonalMatrix) | 6.80 | 4.98 | 6.21 |
| | `apply_to_subrange()` | 1.78 | 1.12 | 3.41 |
| 262 144 | `apply()` | 5.80 | 4.31 | 5.40 |
| | `apply_to_subrange()` | 1.56 | 1.17 | 2.33 |
| 4 194 304 | `apply()` | 6.08 | 4.47 | 5.77 |
| | `apply_to_subrange()` | 3.46 | 2.02 | 2.72 |

**deal.II 9.5.1** (Ubuntu 24.04 package), width 1: both paths 1.2-3.8 ns per
entry, `apply()` slightly faster.

In a complete fused CG with a matrix-free Q_p mass operator (square mesh, 15
iterations), the vector updates per iteration cost, in ns per dof (9.7.1,
width 2):

| | unfused | fused, `DiagonalMatrix` | fused, `apply_to_subrange()` only |
|---|---|---|---|
| Q1, 66 049 dofs | 3.6 | 9.4 | 2.4 |
| Q1, 1 050 625 dofs | 6.5 | 8.5 | 5.3 |
| Q4, 1 050 625 dofs | 7.1 | 8.9 | 5.3 |

The fused `DiagonalMatrix` cost is flat at ~9 ns whatever the degree or size,
and even at 4 225 dofs (all in L1/L2), so it is not memory traffic. The hooks
themselves, the renumbering `DoFRenumbering::matrix_free_data_locality` and a
raw-pointer `apply()` make no difference. valgrind counts only ~5 % more
instructions for the fused CG than for the unfused one, so the time goes into
stalls, not extra work.

## Likely cause

In the `apply()` path, every `VectorizedArray` step of the loops fills the
preconditioned values lane by lane,

```cpp
for (unsigned int l = 0; l < n_lanes; ++l)
  prec_rj[l] = this->preconditioner.apply(j + l, rj[l]);
...
pj = beta * pj + prec_rj;   // reads prec_rj as a whole vector
```

and then uses `prec_rj` as a whole vector. Writing the lanes separately and
reading them back as one register is the classic pattern for a failed
store-to-load forwarding (10-15 cycles each), 2-3 times per `VectorizedArray`
step here, which fits both the size of the penalty and its absence at width 1.
This has not been confirmed with hardware counters.

## Workaround

Pass a preconditioner that has `apply_to_subrange()` but no `apply()`:

```cpp
struct RangeJacobi
{
  void vmult(VectorType &dst, const VectorType &src) const { D.vmult(dst, src); }
  void apply_to_subrange(const unsigned int begin, const unsigned int end,
                         const Number *src, Number *dst) const
  { D.apply_to_subrange(begin, end, src, dst); }
  const DiagonalMatrix<VectorType> &D;
};
```

(`include/lbfem/mass.h`). With it, fused CG with a matrix-free mass operator
is 13-27 % faster than unfused for Q2/Q4 at 4.2M dofs, and within a few % for
Q1 (see the README, *Profiling*).

## Possible fixes in deal.II

- Prefer the `apply_to_subrange()` path when both exist (at least for
  `VectorizedArray` width > 1), or
- in the `apply()` path, build `prec_rj` with a vector operation for diagonal
  preconditioners (e.g. load the diagonal as a `VectorizedArray`), instead of
  lane-wise scalar calls.
