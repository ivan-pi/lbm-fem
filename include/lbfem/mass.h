// Solves with the matrix of the Taylor-Galerkin streaming step, the mass
// matrix M or the left-hand side M + dt^2/6 K_e of the third-order scheme
// (TG3): consistent (CG with Jacobi preconditioner), row-sum lumped, or lumped
// with k Richardson passes (Donea's iterated lumping). With
// MassSettings::fused, the vector operations of CG and of the Richardson
// passes run inside the cell loop of the operator (deal.II's SolverCG and
// PreconditionRelaxation do that when the matrix offers a vmult with
// operations on ranges of the vectors).
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
    // Vector updates of CG and Richardson inside the cell loop: pays off where
    // the operator is memory-bound (higher degree, large meshes), see
    // docs/performance.md.
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
  // With fused, vmult also comes with operations on ranges of the vectors.
  template <bool fused = true>
  class StreamingMatrix : public Observable
  {
  public:
    using RangeOperation = std::function<void(const unsigned int, const unsigned int)>;

    explicit StreamingMatrix(const MatrixFree<dim, Number> &mf,
                             const Number                   coefficient = 0.,
                             const Direction               &e           = {{0., 0.}})
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
    // identity.
    void
    vmult(VectorType &dst, const VectorType &src, const RangeOperation &before, const RangeOperation &after) const
      requires fused
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
    cell_integral(FEEval<1> &phi) const
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
          const auto                              g  = phi.get_gradient(q);
          const auto                              eg = e[0] * g[0] + e[1] * g[1];
          Tensor<1, dim, VectorizedArray<Number>> flux;
          flux[0] = (coef * e[0]) * eg;
          flux[1] = (coef * e[1]) * eg;
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(flux, q);
        }
      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    }

    void
    local_apply(const MatrixFree<dim, Number> &,
                VectorType                                  &dst,
                const VectorType                            &src,
                const std::pair<unsigned int, unsigned int> &range) const
    {
      FEEval<1> phi(data);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          cell_integral(phi);
          phi.distribute_local_to_global(dst);
        }
    }

    const MatrixFree<dim, Number> &data;
    const Number                   coef;
    const Direction                e;
  };



  // The Jacobi preconditioner of CG, with apply_to_subrange() but no per-entry
  // apply(): given the apply() of DiagonalMatrix, the fused SolverCG
  // preconditions lane by lane, which is slower than the updates it fuses
  // (docs/dealii-fused-cg.md).
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



  class MassSolver
  {
  public:
    MassSolver(const Discretization &disc, const MassSettings &settings)
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
            break;

          case Mass::cg:
            {
              SolverControl control(settings.cg_max_iterations, settings.cg_tolerance * r.l2_norm());
              with_matrix(e, tg3_coefficient, [&](const auto &A) {
                SolverCG<VectorType>(control).solve(A, x, r, RangeJacobi{*disc.mass.get_matrix_diagonal_inverse()});
              });
              cg_iterations += control.last_step();
            }
        }
    }

    std::size_t cg_iterations = 0; // of all solves so far

  private:
    // f(A) with the matrix A = M + c K_e(e) as the solvers should see it:
    // fused, the StreamingMatrix; unfused, deal.II's MassOperator for M (a
    // little faster) or the StreamingMatrix without its range operations.
    template <typename F>
    void
    with_matrix(const Direction &e, const Number c, F &&f) const
    {
      if (settings.fused)
        f(StreamingMatrix<true>(*disc.matrix_free, c, e));
      else if (c == 0.)
        f(disc.mass);
      else
        f(StreamingMatrix<false>(*disc.matrix_free, c, e));
    }

    const Discretization &disc;
    const MassSettings    settings;
  };
} // namespace lbfem
