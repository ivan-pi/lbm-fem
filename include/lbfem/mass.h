// Solves with the matrix of the Taylor-Galerkin streaming step, the mass
// matrix M or the left-hand side M + dt^2/6 K_e of the third-order scheme
// (TG3): consistent (CG with Jacobi preconditioner), row-sum lumped, or lumped
// with k Richardson passes (Donea's iterated lumping). With
// MassSettings::fused, the vector operations of CG and of the Richardson
// passes run inside the cell loop of the operator, on each range of entries
// just before the loop first touches it and just after it last does, so that
// an iteration passes through memory about once.
#pragma once

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <lbfem/discretization.h>

#include <algorithm>
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
    // CG and Richardson: vector updates inside the cell loop. This saves memory
    // traffic, and pays off where the operator is memory-bound: out of cache,
    // higher degree, wide SIMD (Q2 at 4M dofs: 5-10 % faster). The Q1 operator
    // is not (a vmult costs ~40 ns per dof with SSE2), and there fusing is up
    // to a third slower (see README).
    bool fused = false;
  };

  // The matrix A = M + c K_e of the streaming step: the mass matrix for c = 0,
  // and for c = dt^2/6 the left-hand side of the third-order Taylor-Galerkin
  // step (Donea 1984)
  //   (M + dt^2/6 K_e) (g^{n+1} - g*) = -dt C_e g* - dt^2/2 K_e g*,
  // K_e = int (e.grad N)(e.grad N^T): the dt^3/6 (e.grad)^3 term of the Taylor
  // series along the characteristic, with (e.grad)^3 g ~ -(e.grad)^2 dg/dt.
  template <int fe_degree>
  class StreamingMatrix
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
    // identity. SolverCG detects this overload and moves its vector updates
    // and reductions into it.
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



  // A matrix without the range operations of its vmult: SolverCG then runs
  // its classic iteration.
  template <typename Matrix>
  struct Unfused
  {
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      A.vmult(dst, src);
    }
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
    {
      disc.matrix_free->initialize_dof_vector(Mx);
    }

    // x <- A^{-1} r, A = M, or M + tg3_coefficient K_e(e) if tg3_coefficient > 0
    // (CG only). CG starts from the given x; the lumped variants overwrite it.
    void
    solve(VectorType &x, const VectorType &r, const Direction &e, const Number tg3_coefficient)
    {
      const auto &DL = *disc.mass.get_matrix_lumped_diagonal_inverse();
      switch (settings.type)
        {
          case Mass::lumped:
            DL.vmult(x, r);
            vector_passes += 2;
            flops_per_node += 1;
            break;

          case Mass::richardson: // x_0 = M_L^{-1} r, x_{j+1} = x_j + M_L^{-1} (r - M x_j)
            DL.vmult(x, r);
            if (settings.fused)
              {
                const StreamingMatrix<fe_degree> M(*disc.matrix_free);
                const Number *const              dl = DL.get_vector().begin();
                const Number *const              rr = r.begin();
                Number *const                    xx = x.begin();
                Number *const                    mx = Mx.begin();
                for (unsigned int j = 0; j < settings.richardson; ++j)
                  M.vmult(
                    Mx,
                    x,
                    [&](const unsigned int begin, const unsigned int end) { std::fill(mx + begin, mx + end, 0.); },
                    [&](const unsigned int begin, const unsigned int end) {
                      for (unsigned int i = begin; i < end; ++i)
                        xx[i] += dl[i] * (rr[i] - mx[i]);
                    });
              }
            else
              for (unsigned int j = 0; j < settings.richardson; ++j)
                {
                  disc.mass.vmult(Mx, x);
                  Mx.sadd(-1., 1., r);
                  DL.vmult(Mx, Mx);
                  x += Mx;
                }
            // x_0: r, M_L^{-1}, x; per pass x (read, write), r, M_L^{-1} and M x:
            // written, read and written three times, or (fused) once, in cache
            vector_passes += 3 + (settings.fused ? 5. : 11.) * settings.richardson;
            flops_per_node += 1 + (flops_vmult + 3.) * settings.richardson;
            break;

          case Mass::cg:
            {
              SolverControl                    control(settings.cg_max_iterations, settings.cg_tolerance * r.l2_norm());
              SolverCG<VectorType>             cg(control);
              const auto                      &jacobi = *disc.mass.get_matrix_diagonal_inverse();
              const StreamingMatrix<fe_degree> A(*disc.matrix_free, tg3_coefficient, e);
              if (settings.fused)
                cg.solve(A, x, r, jacobi);
              else if (tg3_coefficient == 0.)
                cg.solve(disc.mass, x, r, jacobi);
              else
                cg.solve(Unfused<StreamingMatrix<fe_degree>>{A}, x, r, jacobi);
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
    const Disc    &disc;
    const Settings settings;
    VectorType     Mx; // Richardson passes
  };
} // namespace lbfem
