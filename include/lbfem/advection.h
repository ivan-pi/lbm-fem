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
// AdvectionOperator is the (virtual) interface a scheme streams through; a
// driver can pass its own. It is called once per sweep, so the virtual call
// costs nothing. The implementations are built from generic matrix-free loops
// that take the weak form at a quadrature point as a lambda (a "point
// operation", in the style of the flux functions of the deal.II tutorials),
// which the compiler inlines into the loops. One loop over cells and wall faces
// treats all moving populations at once (a (Q-1)-component FEEvaluation on the
// scalar DoFHandler, blocks 1..Q-1 of the block vector).
#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

#include <lbfem/collision.h>
#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
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

  template <int fe_degree>
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
    using Gradient = Tensor<1, 2, VA>;
    using Values   = Tensor<1, n_moving, VA>;       // of all moving populations
    using Fluxes   = Tensor<1, n_moving, Gradient>; // of all moving populations

    // The weak-form terms of all moving populations at a quadrature point,
    //   r += int phi value + int grad(phi) . flux;
    // a point operation returns a type with either member or both, and only
    // what it returns is integrated.
    template <typename T>
    concept HasValue = requires(const T &t) {
      { t.value } -> std::convertible_to<Values>;
    };
    template <typename T>
    concept HasFlux = requires(const T &t) {
      { t.flux } -> std::convertible_to<Fluxes>;
    };

    struct ValueAndFlux
    {
      Values value;
      Fluxes flux;
    };

    template <std::size_t>
    using GradientsOf = Fluxes; // one set of gradients per source field

    // A point operation for n_sources fields is called as op(e, grad...), with
    // the directions of all populations and the gradients of all moving
    // populations in each field.
    template <typename Op, typename Sequence>
    struct PointCall : std::false_type
    {};
    template <typename Op, std::size_t... s>
    struct PointCall<Op, std::index_sequence<s...>>
      : std::is_invocable<const Op &, const Directions &, const GradientsOf<s> &...>
    {
      using result = std::invoke_result_t<const Op &, const Directions &, const GradientsOf<s> &...>;
    };

    template <typename Op, std::size_t n_sources>
    using PointResult = typename PointCall<Op, std::make_index_sequence<n_sources>>::result;

    template <typename Op, std::size_t n_sources>
    concept PointOperation = PointCall<Op, std::make_index_sequence<n_sources>>::value &&
                             (HasValue<PointResult<Op, n_sources>> || HasFlux<PointResult<Op, n_sources>>);

    // A boundary operation: op(e, normal, grad) with the directions of all
    // populations, the outward normal and the gradients of all moving
    // populations; returns their values.
    template <typename Op>
    concept BoundaryOperation = requires(const Op &op, const Directions &e, const Gradient &normal, const Fluxes &grad) {
      { op(e, normal, grad) } -> std::convertible_to<Values>;
    };

    // Cell integrals of a point operation, with the gradients of all moving
    // populations evaluated in each of the source fields s.
    template <int fe_degree, typename Op, std::size_t... s>
    void
    integrate_cells(const MatrixFree<2, Number>                        &data,
                    BlockVectorType                                    &dst,
                    const std::array<const BlockVectorType *, sizeof...(s)> &src,
                    const std::pair<unsigned int, unsigned int>        &range,
                    const Directions                                   &e,
                    const Op                                           &op,
                    std::index_sequence<s...>)
    {
      using FEEval = FEEvaluation<2, fe_degree, fe_degree + 1, n_moving, Number>;
      using Terms  = PointResult<Op, sizeof...(s)>;

      std::array<FEEval, sizeof...(s)> phi{{((void)s, FEEval(data))...}};
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          for (auto &phi_s : phi)
            phi_s.reinit(cell);
          ((phi[s].read_dof_values(*src[s], 1), phi[s].evaluate(EvaluationFlags::gradients)), ...);
          for (unsigned int q = 0; q < phi[0].n_q_points; ++q)
            {
              const Terms t = op(e, phi[s].get_gradient(q)...);
              if constexpr (HasValue<Terms>)
                phi[0].submit_value(t.value, q);
              if constexpr (HasFlux<Terms>)
                phi[0].submit_gradient(t.flux, q);
            }
          if constexpr (HasValue<Terms> && HasFlux<Terms>)
            phi[0].integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          else if constexpr (HasValue<Terms>)
            phi[0].integrate(EvaluationFlags::values);
          else
            phi[0].integrate(EvaluationFlags::gradients);
          phi[0].distribute_local_to_global(dst, 0);
        }
    }

    // Boundary-face integrals int phi value of all moving populations.
    template <int fe_degree, BoundaryOperation Op>
    void
    integrate_boundary_faces(const MatrixFree<2, Number>                 &data,
                             BlockVectorType                             &dst,
                             const BlockVectorType                       &src,
                             const std::pair<unsigned int, unsigned int> &range,
                             const Directions                            &e,
                             const Op                                    &op)
    {
      FEFaceEvaluation<2, fe_degree, fe_degree + 1, n_moving, Number> phi(data, /*interior*/ true);
      for (unsigned int face = range.first; face < range.second; ++face)
        {
          phi.reinit(face);
          phi.read_dof_values(src, 1);
          phi.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(op(e, phi.normal_vector(q), phi.get_gradient(q)), q);
          phi.integrate(EvaluationFlags::values);
          phi.distribute_local_to_global(dst, 0);
        }
    }

    // rhs <- cell integrals of cell_op + wall integrals of face_op, in one
    // matrix-free loop over src[0] (the populations; ghost values of the other
    // sources are exchanged here).
    template <int fe_degree, std::size_t n_sources, PointOperation<n_sources> CellOp, BoundaryOperation FaceOp>
    void
    advection_loop(const MatrixFree<2, Number>                         &mf,
                   BlockVectorType                                     &rhs,
                   const std::array<const BlockVectorType *, n_sources> &src,
                   const Directions                                    &e,
                   const CellOp                                        &cell_op,
                   const FaceOp                                        &face_op)
    {
      using Range = std::pair<unsigned int, unsigned int>;
      for (std::size_t s = 1; s < n_sources; ++s)
        src[s]->update_ghost_values();
      mf.template loop<BlockVectorType, BlockVectorType>(
        [&](const auto &data, auto &dst, const auto &, const Range &range) {
          integrate_cells<fe_degree>(data, dst, src, range, e, cell_op, std::make_index_sequence<n_sources>{});
        },
        [](const auto &, auto &, const auto &, const Range &) {}, // continuous: no inner faces
        [&](const auto &data, auto &dst, const auto &f, const Range &range) {
          integrate_boundary_faces<fe_degree>(data, dst, f, range, e, face_op);
        },
        rhs,
        *src[0],
        /*zero dst*/ true);
      for (std::size_t s = 1; s < n_sources; ++s)
        src[s]->zero_out_ghost_values();
    }

    // Surface term of the integration by parts of D, Eq. (23):
    //   + dt^2/2 oint phi (n.e)(e.grad f), with the known f^n on the walls.
    // Its zeroth moment, dt^2/2 n.div(Pi), is a mass flux through the wall;
    // remove_mass_flux removes it isotropically (no momentum is added),
    // otherwise the term is kept as it is (Lee & Lin, Sec. 2.3).
    inline auto
    wall_surface_term(const Number dt, const bool remove_mass_flux)
    {
      return [=](const Directions &e, const Gradient &normal, const Fluxes &grad) {
        Values value;
        VA                      mass_flux = 0.;
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
  //     = -int phi dt e.grad g* - int (e.grad phi) dt^2/2 e.grad g*,
  // plus the wall surface term.
  template <int fe_degree>
  class TaylorGalerkinAdvection final : public AdvectionOperator<fe_degree>
  {
  public:
    TaylorGalerkinAdvection(const Discretization<fe_degree> &disc, const bool remove_wall_mass_flux)
      : disc(disc)
      , remove_wall_mass_flux(remove_wall_mass_flux)
    {}

    void
    apply(BlockVectorType &rhs, const AdvectionInput &in, const Directions &e, const TimeStep &ts) const override
    {
      using namespace weak_form;
      const Number c_f = -ts.dt, c_btd = -0.5 * ts.dt * ts.dt;
      advection_loop<fe_degree>(
        *disc.matrix_free,
        rhs,
        std::array<const BlockVectorType *, 1>{{&in.f}},
        e,
        [=](const Directions &e, const Fluxes &grad_g) {
          ValueAndFlux t;
          for (unsigned int a = 0; a < n_moving; ++a)
            {
              const auto [ex, ey] = e[a + 1];
              const VA e_grad_g   = ex * grad_g[a][0] + ey * grad_g[a][1];
              t.value[a]          = c_f * e_grad_g;
              t.flux[a][0]        = (c_btd * ex) * e_grad_g;
              t.flux[a][1]        = (c_btd * ey) * e_grad_g;
            }
          return t;
        },
        wall_surface_term(ts.dt, remove_wall_mass_flux));
    }

  private:
    const Discretization<fe_degree> &disc;
    const bool                       remove_wall_mass_flux;
  };



  // Lee & Lin, Eq. (17): the populations are advected together with the
  // gradients of their equilibria,
  //   r = -dt C f - dt^2 [D f + Q (f - feq)]
  //     = int phi [-dt + dt^2/(2 lambda)] e.grad f - int phi dt^2/(2 lambda) e.grad feq
  //       - int (e.grad phi) dt^2/2 e.grad f,
  // plus the wall surface term, kept as it is.
  template <int fe_degree>
  class LeeLinAdvection final : public AdvectionOperator<fe_degree>
  {
  public:
    explicit LeeLinAdvection(const Discretization<fe_degree> &disc)
      : disc(disc)
    {}

    void
    apply(BlockVectorType &rhs, const AdvectionInput &in, const Directions &e, const TimeStep &ts) const override
    {
      using namespace weak_form;
      AssertThrow(in.feq, ExcMessage("LeeLinAdvection needs the equilibria"));
      const Number c_eq  = -0.5 * ts.dt * ts.dt / ts.lambda;
      const Number c_f   = -ts.dt - c_eq;
      const Number c_btd = -0.5 * ts.dt * ts.dt;
      advection_loop<fe_degree>(
        *disc.matrix_free,
        rhs,
        std::array<const BlockVectorType *, 2>{{&in.f, in.feq}},
        e,
        [=](const Directions &e, const Fluxes &grad_f, const Fluxes &grad_eq) {
          ValueAndFlux t;
          for (unsigned int a = 0; a < n_moving; ++a)
            {
              const auto [ex, ey] = e[a + 1];
              const VA e_grad_f   = ex * grad_f[a][0] + ey * grad_f[a][1];
              const VA e_grad_eq  = ex * grad_eq[a][0] + ey * grad_eq[a][1];
              t.value[a]          = c_f * e_grad_f + c_eq * e_grad_eq;
              t.flux[a][0]        = (c_btd * ex) * e_grad_f;
              t.flux[a][1]        = (c_btd * ey) * e_grad_f;
            }
          return t;
        },
        wall_surface_term(ts.dt, /*remove_mass_flux*/ false));
    }

  private:
    const Discretization<fe_degree> &disc;
  };
} // namespace lbfem
