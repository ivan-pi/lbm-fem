// Advection operators: the right-hand side r_alpha of the streaming step of
// every moving population alpha = 1..Q-1,
//
//   A (f_alpha^{n+1} - f_alpha^n) = r_alpha,
//
// with A = M or M + dt^2/6 K_e (see streaming.h). The element matrices of
// Lee & Lin (19)-(22) / Bardow (13), never assembled, are
//
//   C_e = int N (e.grad N^T),  K_e = int (e.grad N)(e.grad N^T) = 2 D_e,
//   Q_e = -C_e / (2 lambda).
//
// AdvectionOperator is the interface a scheme streams through; a driver can
// pass its own. The implementations are built from generic matrix-free loops
// that take the weak form at a quadrature point as a lambda (a "point
// operation", in the style of the flux functions of the deal.II tutorials).
// One loop over cells and wall faces treats all moving populations at once
// (a (Q-1)-component FEEvaluation on the scalar DoFHandler, blocks 1..Q-1
// of the block vector).
#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

#include <lbfem/collision.h>
#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>

#include <array>
#include <cstddef>
#include <utility>

namespace lbfem
{
  using Directions = std::array<Direction, Q>; // of all populations in the current sweep

  struct TimeStep
  {
    Number dt;     // time step
    Number lambda; // relaxation time nu / c_s^2
  };

  // What an advection operator may read: the populations and, for schemes
  // that use them, their nodal equilibria.
  struct AdvectionInput
  {
    const BlockVectorType &f;
    const BlockVectorType *feq = nullptr;
  };

  class AdvectionOperator
  {
  public:
    virtual ~AdvectionOperator() = default;

    // rhs_alpha <- r_alpha for alpha = 1..Q-1 (rhs has Q-1 blocks), each
    // population streamed along e[alpha].
    virtual void
    apply(BlockVectorType &rhs, const AdvectionInput &in, const Directions &e, const TimeStep &ts) const = 0;
  };



  // ------------------------- generic matrix-free loops -------------------------

  namespace weak_form
  {
    using VA       = VectorizedArray<Number>;
    using Gradient = Tensor<1, dim, VA>;
    using Values   = Tensor<1, n_moving, VA>;       // of all moving populations
    using Fluxes   = Tensor<1, n_moving, Gradient>; // of all moving populations

    // The weak-form terms of all moving populations at a quadrature point,
    //   r += int phi value + int grad(phi) . flux.
    // A point operation is called with the gradients of all moving populations
    // in each of its source fields (one or two) and returns them; a boundary
    // operation with the outward normal and the gradients, and returns values.
    struct ValueAndFlux
    {
      Values value;
      Fluxes flux;
    };

    template <std::size_t n>
    std::array<FEEval<n_moving>, n>
    evaluators(const MatrixFree<dim, Number> &data)
    {
      static_assert(n == 1 || n == 2);
      if constexpr (n == 1)
        return {{FEEval<n_moving>(data)}};
      else
        return {{FEEval<n_moving>(data), FEEval<n_moving>(data)}};
    }

    // Cell integrals of a point operation over the source fields src.
    template <std::size_t n, typename Op>
    void
    integrate_cells(const MatrixFree<dim, Number>                &data,
                    BlockVectorType                              &dst,
                    const std::array<const BlockVectorType *, n> &src,
                    const std::pair<unsigned int, unsigned int>  &range,
                    const Op                                     &op)
    {
      auto phi = evaluators<n>(data);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          for (std::size_t s = 0; s < n; ++s)
            {
              phi[s].reinit(cell);
              phi[s].read_dof_values(*src[s], 1);
              phi[s].evaluate(EvaluationFlags::gradients);
            }
          for (unsigned int q = 0; q < phi[0].n_q_points; ++q)
            {
              ValueAndFlux t;
              if constexpr (n == 1)
                t = op(phi[0].get_gradient(q));
              else
                t = op(phi[0].get_gradient(q), phi[1].get_gradient(q));
              phi[0].submit_value(t.value, q);
              phi[0].submit_gradient(t.flux, q);
            }
          phi[0].integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi[0].distribute_local_to_global(dst, 0);
        }
    }

    // Boundary-face integrals int phi value of all moving populations.
    template <typename Op>
    void
    integrate_boundary_faces(const MatrixFree<dim, Number>               &data,
                             BlockVectorType                             &dst,
                             const BlockVectorType                       &src,
                             const std::pair<unsigned int, unsigned int> &range,
                             const Op                                    &op)
    {
      FEFaceEvaluation<dim, fe_degree, fe_degree + 1, n_moving, Number> phi(data, /*interior*/ true);
      for (unsigned int face = range.first; face < range.second; ++face)
        {
          phi.reinit(face);
          phi.read_dof_values(src, 1);
          phi.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(op(phi.normal_vector(q), phi.get_gradient(q)), q);
          phi.integrate(EvaluationFlags::values);
          phi.distribute_local_to_global(dst, 0);
        }
    }

    // rhs <- cell integrals of cell_op + wall integrals of face_op, in one
    // matrix-free loop over src[0] (the populations; ghost values of the other
    // sources are exchanged here).
    template <std::size_t n, typename CellOp, typename FaceOp>
    void
    advection_loop(const MatrixFree<dim, Number>                &mf,
                   BlockVectorType                              &rhs,
                   const std::array<const BlockVectorType *, n> &src,
                   const CellOp                                 &cell_op,
                   const FaceOp                                 &face_op)
    {
      using Range = std::pair<unsigned int, unsigned int>;
      for (std::size_t s = 1; s < n; ++s)
        src[s]->update_ghost_values();
      mf.template loop<BlockVectorType, BlockVectorType>(
        [&](const auto &data, auto &dst, const auto &, const Range &range) {
          integrate_cells(data, dst, src, range, cell_op);
        },
        [](const auto &, auto &, const auto &, const Range &) {}, // continuous: no inner faces
        [&](const auto &data, auto &dst, const auto &f, const Range &range) {
          integrate_boundary_faces(data, dst, f, range, face_op);
        },
        rhs,
        *src[0],
        /*zero dst*/ true);
      for (std::size_t s = 1; s < n; ++s)
        src[s]->zero_out_ghost_values();
    }

    // Surface term S of the integration by parts of D, Eq. (23):
    //   + dt^2/2 oint phi (n.e)(e.grad f), with the known f^n on the walls.
    // Its zeroth moment, dt^2/2 n.div(Pi), is a mass flux through the wall;
    // remove_mass_flux removes it isotropically (no momentum is added),
    // otherwise the term is kept as it is (Lee & Lin, Sec. 2.3).
    inline auto
    wall_surface_term(const Directions &e, const Number dt, const bool remove_mass_flux)
    {
      return [&e, dt, remove_mass_flux](const Gradient &normal, const Fluxes &grad) {
        Values value;
        VA     mass_flux = 0.;
        for (unsigned int a = 0; a < n_moving; ++a)
          {
            const auto [ex, ey] = e[a + 1];
            value[a] = (0.5 * dt * dt) * (ex * normal[0] + ey * normal[1]) * (ex * grad[a][0] + ey * grad[a][1]);
            mass_flux += value[a];
          }
        if (remove_mass_flux)
          for (unsigned int a = 0; a < n_moving; ++a)
            value[a] -= (D2Q9::w[a + 1] / (1. - D2Q9::w[0])) * mass_flux;
        return value;
      };
    }
  } // namespace weak_form



  // --------------------------------- operators ---------------------------------

  // Bardow et al., weak form of their Eq. (9): the Taylor expansion along the
  // characteristic of the post-collision populations,
  //   r = -dt C g* - dt^2 D g*
  //     = int phi a e.grad g* + int (e.grad phi) c e.grad g*,   a = -dt, c = -dt^2/2,
  // plus the wall surface term.
  class TaylorGalerkinAdvection final : public AdvectionOperator
  {
  public:
    TaylorGalerkinAdvection(const Discretization &disc, const bool remove_wall_mass_flux)
      : disc(disc)
      , remove_wall_mass_flux(remove_wall_mass_flux)
    {}

    void
    apply(BlockVectorType &rhs, const AdvectionInput &in, const Directions &e, const TimeStep &ts) const override
    {
      using namespace weak_form;
      const Number a = -ts.dt, c = -0.5 * ts.dt * ts.dt;
      advection_loop(
        *disc.matrix_free,
        rhs,
        std::array<const BlockVectorType *, 1>{{&in.f}},
        [&e, a, c](const Fluxes &grad_g) {
          ValueAndFlux t;
          for (unsigned int alpha = 0; alpha < n_moving; ++alpha)
            {
              const auto [ex, ey] = e[alpha + 1];
              const VA e_grad_g   = ex * grad_g[alpha][0] + ey * grad_g[alpha][1];
              t.value[alpha]      = a * e_grad_g;
              t.flux[alpha][0]    = (c * ex) * e_grad_g;
              t.flux[alpha][1]    = (c * ey) * e_grad_g;
            }
          return t;
        },
        wall_surface_term(e, ts.dt, remove_wall_mass_flux));
    }

  private:
    const Discretization &disc;
    const bool            remove_wall_mass_flux;
  };



  // Lee & Lin, Eq. (17): the populations are advected together with the
  // gradients of their equilibria,
  //   r = -dt C f - dt^2 [D f + Q (f - feq)]
  //     = int phi [a e.grad f + b e.grad feq] + int (e.grad phi) c e.grad f,
  //   a = -dt + dt^2/(2 lambda),  b = -dt^2/(2 lambda),  c = -dt^2/2,
  // plus the wall surface term, kept as it is.
  class LeeLinAdvection final : public AdvectionOperator
  {
  public:
    explicit LeeLinAdvection(const Discretization &disc)
      : disc(disc)
    {}

    void
    apply(BlockVectorType &rhs, const AdvectionInput &in, const Directions &e, const TimeStep &ts) const override
    {
      using namespace weak_form;
      AssertThrow(in.feq, ExcMessage("LeeLinAdvection needs the equilibria"));
      const Number b = -0.5 * ts.dt * ts.dt / ts.lambda;
      const Number a = -ts.dt - b;
      const Number c = -0.5 * ts.dt * ts.dt;
      advection_loop(
        *disc.matrix_free,
        rhs,
        std::array<const BlockVectorType *, 2>{{&in.f, in.feq}},
        [&e, a, b, c](const Fluxes &grad_f, const Fluxes &grad_eq) {
          ValueAndFlux t;
          for (unsigned int alpha = 0; alpha < n_moving; ++alpha)
            {
              const auto [ex, ey] = e[alpha + 1];
              const VA e_grad_f   = ex * grad_f[alpha][0] + ey * grad_f[alpha][1];
              const VA e_grad_eq  = ex * grad_eq[alpha][0] + ey * grad_eq[alpha][1];
              t.value[alpha]      = a * e_grad_f + b * e_grad_eq;
              t.flux[alpha][0]    = (c * ex) * e_grad_f;
              t.flux[alpha][1]    = (c * ey) * e_grad_f;
            }
          return t;
        },
        wall_surface_term(e, ts.dt, /*remove_mass_flux*/ false));
    }

  private:
    const Discretization &disc;
  };
} // namespace lbfem
