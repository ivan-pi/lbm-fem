// Solves with the matrix of the Taylor-Galerkin streaming step, the mass
// matrix M or the left-hand side M + dt^2/6 K_e of the third-order scheme
// (TG3): consistent (CG with Jacobi preconditioner), row-sum lumped, or lumped
// with k Richardson passes (Donea's iterated lumping). With
// MassSettings::fused, the vector operations of CG and of the Richardson
// passes run inside the cell loop of the operator, on each range of entries
// just before the loop first touches it and just after it last does, so that
// an iteration passes through memory about once.
#pragma once

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <lbfem/discretization.h>

#include <cstddef>
#include <functional>
#include <type_traits>

namespace lbfem
{
  enum class Mass
  {
    cg,        // consistent, CG with Jacobi preconditioner
    lumped,    // row-sum lumped
    richardson // lumped plus k Richardson passes
  };

  struct MassSettings
  {
    Mass         type         = Mass::cg;
    unsigned int richardson   = 2;    // passes for Mass::richardson
    double       cg_tolerance = 1e-8; // relative to |r|
    // A safety cap: CG throws SolverControl::NoConvergence when it is hit.
    // The Jacobi-preconditioned mass matrix has a condition number bounded
    // independently of h (at most 9 for Q1 on a uniform grid), and a cold
    // start needs about 25 iterations at the default tolerance and 38 at
    // 1e-12, from 32^2 to 512^2 cells (the extrapolated start, 2-16).
    unsigned int cg_max_iterations = 200;
    // CG and Richardson: vector updates inside the cell loop. This saves the
    // memory traffic of the updates, which matters where the operator is cheap
    // per dof and the vectors are out of cache: Q2 and Q4 at 4M dofs solve
    // 13-27 % faster. Q1, whose vmult costs 25-35 ns per dof with SSE2, is
    // within a few % either way (see README).
    bool fused = false;
  };

  // What deal.II's solvers and preconditioners keep a checked pointer to.
#if DEAL_II_VERSION_GTE(9, 7, 0)
  using Observable = EnableObserverPointer;
#else
  using Observable = Subscriptor;
#endif

  // The matrix A = M + c K_e of the streaming step: the mass matrix for c = 0,
  // and for c = dt^2/6 the left-hand side of the third-order Taylor-Galerkin
  // step (Donea 1984)
  //   (M + dt^2/6 K_e) (g^{n+1} - g*) = -dt C_e g* - dt^2/2 K_e g*,
  // K_e = int (e.grad N)(e.grad N^T): the dt^3/6 (e.grad)^3 term of the Taylor
  // series along the characteristic, with (e.grad)^3 g ~ -(e.grad)^2 dg/dt.
  template <int fe_degree>
  class StreamingMatrix : public Observable
  {
  public:
    using FEEval    = FEEvaluation<2, fe_degree, fe_degree + 1, 1, Number>;
    using RangeFunc = std::function<void(const unsigned int, const unsigned int)>;

    explicit StreamingMatrix(const MatrixFree<2, Number> &mf,
                             const Number                 coefficient = 0.,
                             const Direction             &e           = {{0., 0.}})
      : data(mf)
      , coef(coefficient)
      , e(e)
    {}

    // dst = A src
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      data.cell_loop(&StreamingMatrix::local_apply, this, dst, src, /*zero dst*/ true);
    }

    // dst = A src, with before(i, j) run on the locally owned entries [i, j)
    // before the loop first touches them (it must zero dst there) and after(i,
    // j) once it no longer does (dst[i, j) is final). Constrained rows are the
    // identity. SolverCG and PreconditionRelaxation detect this overload and
    // move their vector updates into it.
    void
    vmult(VectorType &dst, const VectorType &src, const RangeFunc &before, const RangeFunc &after) const
    {
      data.cell_loop(&StreamingMatrix::local_apply, this, dst, src, before, after);
    }

    // The diagonal of the operator (for a Jacobi preconditioner).
    void
    compute_diagonal(VectorType &diagonal) const
    {
      data.initialize_dof_vector(diagonal);
      MatrixFreeTools::compute_diagonal(data, diagonal, &StreamingMatrix::cell_integral, this);
    }

  private:
    DEAL_II_ALWAYS_INLINE void
    cell_integral(FEEval &phi) const
    {
      if (coef == 0.) // M
        {
          phi.evaluate(EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(phi.get_value(q), q);
          phi.integrate(EvaluationFlags::values);
          return;
        }
      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto g  = phi.get_gradient(q);
          const auto eg = e[0] * g[0] + e[1] * g[1];
          Tensor<1, 2, VectorizedArray<Number>> flux;
          flux[0] = (coef * e[0]) * eg;
          flux[1] = (coef * e[1]) * eg;
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(flux, q);
        }
      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    }

    void
    local_apply(const MatrixFree<2, Number> &,
                VectorType                                  &dst,
                const VectorType                            &src,
                const std::pair<unsigned int, unsigned int> &range) const
    {
      FEEval phi(data);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          cell_integral(phi);
          phi.distribute_local_to_global(dst);
        }
    }

    const MatrixFree<2, Number> &data;
    const Number                 coef;
    const Direction              e;
  };



  // The Jacobi preconditioner of CG, with apply_to_subrange() but no per-entry
  // apply(). Given the apply() of DiagonalMatrix, the fused SolverCG
  // preconditions lane by lane into a SIMD register, which with SIMD width 2
  // (deal.II 9.7.1) makes its updates 2.5-4x slower than on ranges of 128
  // entries, and the fused CG slower than the unfused one (see
  // docs/dealii-fused-cg.md).
  struct RangeJacobi
  {
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      D.vmult(dst, src);
    }
    void
    apply_to_subrange(const unsigned int begin, const unsigned int end, const Number *src, Number *dst) const
    {
      D.apply_to_subrange(begin, end, src, dst);
    }
    const DiagonalMatrix<VectorType> &D;
  };

  // A matrix without the range operations of its vmult: deal.II then runs
  // the classic, unfused iterations.
  template <typename Matrix>
  class Unfused : public Observable
  {
  public:
    explicit Unfused(const Matrix &A)
      : A(A)
    {}
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      A.vmult(dst, src);
    }

  private:
    const Matrix &A;
  };



  template <int fe_degree>
  class MassSolver
  {
  public:
    using Disc     = Discretization<fe_degree>;
    using Settings = MassSettings;

    MassSolver(const Disc &disc, const Settings &settings)
      : disc(disc)
      , settings(settings)
    {}

    // x <- A^{-1} r, A = M + tg3_coefficient K_e(e). CG starts from the given
    // x; the lumped variants overwrite it.
    void
    solve(VectorType &x, const VectorType &r, const Direction &e, const Number tg3_coefficient)
    {
      switch (settings.type)
        {
          case Mass::lumped:
            disc.mass.get_matrix_lumped_diagonal_inverse()->vmult(x, r);
            vector_passes += 2;
            flops_per_node += 1;
            break;

          case Mass::richardson: // x_0 = M_L^{-1} r, x_{j+1} = x_j + M_L^{-1} (r - A x_j)
            with_matrix(e, tg3_coefficient, [&](const auto &A) {
              using Relaxation = PreconditionRelaxation<std::decay_t<decltype(A)>, DiagonalMatrix<VectorType>>;
              typename Relaxation::AdditionalData data(1., settings.richardson + 1);
              data.preconditioner = disc.mass.get_matrix_lumped_diagonal_inverse();
              Relaxation richardson;
              richardson.initialize(A, data);
              richardson.vmult(x, r);
            });
            // x_0: r, M_L^{-1}, x; per pass x, r, M_L^{-1}, A x (written, read)
            // and the update, or (fused) with A x in cache
            vector_passes += 3 + (settings.fused ? 5. : 7.) * settings.richardson;
            flops_per_node += 1 + (flops_vmult + 3.) * settings.richardson;
            break;

          case Mass::cg:
            {
              SolverControl control(settings.cg_max_iterations, settings.cg_tolerance * r.l2_norm());
              with_matrix(e, tg3_coefficient, [&](const auto &A) {
                SolverCG<VectorType>(control).solve(A, x, r, RangeJacobi{*disc.mass.get_matrix_diagonal_inverse()});
              });
              cg_iterations += control.last_step();
              // per iteration: vmult 2, Jacobi 2, updates of x, r, p 6 vector
              // passes; fused, with A p in cache: x, r, p (read, write), A p,
              // the Jacobi diagonal (and 7 reductions instead of 2)
              vector_passes += (settings.fused ? 8. : 10.) * control.last_step() + 4;
              flops_per_node += (flops_vmult + (settings.fused ? 21. : 11.)) * control.last_step();
            }
        }
    }

    // Arithmetic of one mass vmult per node (Q1, see README), and the work
    // counters of all solves so far: CG iterations, vector passes per node
    // (single-pass traffic model) and flops per node.
    static constexpr double flops_vmult = 68;
    std::size_t             cg_iterations  = 0;
    double                  vector_passes  = 0;
    double                  flops_per_node = 0;

  private:
    // f(A) with the matrix A = M + c K_e(e) as the solvers should see it:
    // fused, the StreamingMatrix; unfused, deal.II's MassOperator for M (a
    // little faster than StreamingMatrix) or StreamingMatrix without its range
    // operations.
    template <typename F>
    void
    with_matrix(const Direction &e, const Number c, F &&f) const
    {
      const StreamingMatrix<fe_degree> A(*disc.matrix_free, c, e);
      if (settings.fused)
        f(A);
      else if (c == 0.)
        f(disc.mass);
      else
        f(Unfused<StreamingMatrix<fe_degree>>(A));
    }

    const Disc    &disc;
    const Settings settings;
  };
} // namespace lbfem
