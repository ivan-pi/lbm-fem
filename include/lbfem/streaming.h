// Characteristic Galerkin (Taylor-Galerkin) streaming of the moving
// populations. With the element matrices of Lee & Lin (19)-(22) / Bardow (13),
// never assembled,
//   C_e = int N (e.grad N^T),  K_e = int (e.grad N)(e.grad N^T) = 2 D_e,
//   Q_e = -C_e / (2 lambda),
// the right-hand side r_alpha of each moving population is one matrix-free
// loop over cells and wall faces that treats all 8 moving populations at once
// (8-component FEEvaluation on the scalar DoFHandler, blocks 1..8 of the block
// vector), and the increment is x_alpha = A^{-1} r_alpha with A = M (TG2) or
// M + dt^2/6 K_e (TG3).
#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

#include <lbfem/collision.h>
#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>
#include <lbfem/mass.h>
#include <lbfem/timers.h>

#include <array>

namespace lbfem
{
  enum class Streaming
  {
    tg2,      // Lax-Wendroff / second-order Taylor-Galerkin
    tg3,      // third-order Taylor-Galerkin: (M + dt^2/6 K_e) on the left
    tg3_split // TG3 as an x sweep followed by a y sweep (exact at CFL 1 on a lattice)
  };

  template <int fe_degree>
  class TaylorGalerkin
  {
  public:
    using Disc               = Discretization<fe_degree>;
    static constexpr int dim = Disc::dim, n_q_1d = Disc::n_q_1d;
    using VA                 = VectorizedArray<Number>;
    using Range              = std::pair<unsigned int, unsigned int>;

    struct Settings
    {
      Streaming                             streaming = Streaming::tg2;
      typename MassSolver<fe_degree>::Settings mass;
      // Remove the mass flux of the wall surface term isotropically (Bardow);
      // otherwise the term is kept as it is (Lee & Lin, Sec. 2.3).
      bool remove_wall_mass_flux = false;
    };

    TaylorGalerkin(const Disc &disc, const Settings &settings, StageTimers &timers)
      : disc(disc)
      , settings(settings)
      , is_split(settings.streaming == Streaming::tg3_split)
      , mass(disc, settings.mass)
      , timers(timers)
    {
      AssertThrow(settings.streaming == Streaming::tg2 || settings.mass.type == Mass::cg,
                  ExcMessage("TG3 streaming needs the consistent mass (Mass::cg)"));
      disc.initialize(rhs, n_moving);
      disc.initialize(incr, n_moving);
      disc.initialize(incr_prev, n_moving);
    }

    void
    set_time_step(const Number time_step, const Number relaxation_time)
    {
      dt     = time_step;
      lambda = relaxation_time;
    }

    // incr <- A^{-1} r for all moving populations. Without feq, r is the
    // Bardow right-hand side (weak form of their Eq. 9)
    //   r = -dt C f - dt^2 D f;
    // with feq, the Lee & Lin one (Eq. 17)
    //   r = -dt C f - dt^2 [D f + Q (f - feq)].
    void
    compute_increment(const BlockVectorType &f, const BlockVectorType *feq = nullptr)
    {
      assemble_rhs(f, feq);
      apply_mass_inverse(incr);
    }

    // f_alpha += A^{-1} r(f_alpha) for the moving populations (collide, then
    // stream). tg3_split: the shift by e dt is the product of the shifts by
    // (ex dt, 0) and (0, ey dt); on a tensor-product grid the 1-D sweeps commute
    // and the composition is exact at CFL 1 for all eight directions (the
    // diagonal populations get both sweeps).
    void
    stream(BlockVectorType &f)
    {
      const auto add_increment = [&](const BlockVectorType &x) {
        StageTimers::Scope t(timers, StageTimers::collision); // nodal work
        for (unsigned int a = 0; a < n_moving; ++a)
          f.block(a + 1) += x.block(a);
      };

      if (!is_split)
        {
          compute_increment(f);
          add_increment(incr);
          return;
        }
      for (unsigned int d = 0; d < dim; ++d)
        {
          for (unsigned int a = 0; a < Q; ++a)
            stream_e[a] = d == 0 ? Direction{{D2Q9::e[a][0], 0.}} : Direction{{0., D2Q9::e[a][1]}};
          auto &x = d == 0 ? incr : incr_prev;
          assemble_rhs(f, nullptr);
          apply_mass_inverse(x);
          add_increment(x);
        }
      stream_e = D2Q9::e;
    }

    const Settings &
    get_settings() const
    {
      return settings;
    }

  private:
    void
    assemble_rhs(const BlockVectorType &f, const BlockVectorType *feq)
    {
      StageTimers::Scope t(timers, StageTimers::advection);
      if (!feq)
        disc.matrix_free->loop(&TaylorGalerkin::cell_bardow, &TaylorGalerkin::no_inner_faces,
                               &TaylorGalerkin::wall_faces, this, rhs, f, /*zero dst*/ true);
      else
        {
          this->feq = feq;
          feq->update_ghost_values(); // second source vector, not seen by loop()
          disc.matrix_free->loop(&TaylorGalerkin::cell_leelin, &TaylorGalerkin::no_inner_faces,
                                 &TaylorGalerkin::wall_faces, this, rhs, f, /*zero dst*/ true);
          feq->zero_out_ghost_values();
        }
    }

    // x <- A^{-1} rhs per population. Mass::cg starts from the extrapolation
    // 2 x^{n} - x^{n-1} (the increment changes slowly in time); for the split
    // sweeps the previous sweep's solution is used instead.
    void
    apply_mass_inverse(BlockVectorType &X)
    {
      StageTimers::Scope t(timers, StageTimers::mass);
      const Number tg3 = settings.streaming == Streaming::tg2 ? 0. : dt * dt / 6.;
      for (unsigned int a = 0; a < n_moving; ++a)
        {
          auto       &x = X.block(a);
          const auto &r = rhs.block(a);
          if (stream_e[a + 1][0] == 0. && stream_e[a + 1][1] == 0.)
            {
              x = 0.; // population not streamed in this sweep
              continue;
            }
          if (settings.mass.type == Mass::cg && !is_split)
            {
              incr_prev.block(a).sadd(-1., 2., x); // 2 x^n - x^{n-1}
              x.swap(incr_prev.block(a));          // x: guess, incr_prev: x^n
            }
          mass.solve(x, r, stream_e[a + 1], tg3);
        }
    }

    // Lee & Lin, Eq. (17):  r = -dt C f - dt^2 [D f + Q (f - feq)]
    //   = int phi [-dt + dt^2/(2 lambda)] e.grad f - int phi dt^2/(2 lambda) e.grad feq
    //   - int (e.grad phi) dt^2/2 e.grad f
    void
    cell_leelin(const MatrixFree<dim, Number> &data,
                BlockVectorType               &dst,
                const BlockVectorType         &src,
                const Range                   &range) const
    {
      FEEvaluation<dim, fe_degree, n_q_1d, n_moving, Number> phi_f(data), phi_eq(data);

      const Number c_eq  = -0.5 * dt * dt / lambda;
      const Number c_f   = -dt - c_eq;
      const Number c_btd = -0.5 * dt * dt;

      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          phi_f.reinit(cell);
          phi_f.read_dof_values(src, 1);
          phi_f.evaluate(EvaluationFlags::gradients);
          phi_eq.reinit(cell);
          phi_eq.read_dof_values(*feq, 1);
          phi_eq.evaluate(EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi_f.n_q_points; ++q)
            {
              const auto grad_f  = phi_f.get_gradient(q);
              const auto grad_eq = phi_eq.get_gradient(q);

              Tensor<1, n_moving, VA>                 value;
              Tensor<1, n_moving, Tensor<1, dim, VA>> flux;
              for (unsigned int a = 0; a < n_moving; ++a)
                {
                  const auto [ex, ey] = stream_e[a + 1];
                  const VA e_grad_f   = ex * grad_f[a][0] + ey * grad_f[a][1];
                  const VA e_grad_eq  = ex * grad_eq[a][0] + ey * grad_eq[a][1];
                  value[a]            = c_f * e_grad_f + c_eq * e_grad_eq;
                  flux[a][0]          = (c_btd * ex) * e_grad_f;
                  flux[a][1]          = (c_btd * ey) * e_grad_f;
                }
              phi_f.submit_value(value, q);
              phi_f.submit_gradient(flux, q);
            }
          phi_f.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi_f.distribute_local_to_global(dst, 0);
        }
    }

    // Bardow, weak form of Eq. (9):  r = -dt C g* - dt^2 D g*
    //   = -int phi dt e.grad g* - int (e.grad phi) dt^2/2 e.grad g*
    void
    cell_bardow(const MatrixFree<dim, Number> &data,
                BlockVectorType               &dst,
                const BlockVectorType         &src,
                const Range                   &range) const
    {
      FEEvaluation<dim, fe_degree, n_q_1d, n_moving, Number> phi(data);

      const Number c_f   = -dt;
      const Number c_btd = -0.5 * dt * dt;

      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src, 1);
          phi.evaluate(EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto grad_g = phi.get_gradient(q);

              Tensor<1, n_moving, VA>                 value;
              Tensor<1, n_moving, Tensor<1, dim, VA>> flux;
              for (unsigned int a = 0; a < n_moving; ++a)
                {
                  const auto [ex, ey] = stream_e[a + 1];
                  const VA e_grad_g   = ex * grad_g[a][0] + ey * grad_g[a][1];
                  value[a]            = c_f * e_grad_g;
                  flux[a][0]          = (c_btd * ex) * e_grad_g;
                  flux[a][1]          = (c_btd * ey) * e_grad_g;
                }
              phi.submit_value(value, q);
              phi.submit_gradient(flux, q);
            }
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi.distribute_local_to_global(dst, 0);
        }
    }

    // Surface term of the integration by parts of D, Eq. (23):
    //   + dt^2/2 oint phi (n.e)(e.grad f), with the known f^n on the walls.
    // Its zeroth moment, dt^2/2 n.div(Pi), is a mass flux through the wall;
    // with remove_wall_mass_flux it is removed isotropically (no momentum is
    // added).
    void
    wall_faces(const MatrixFree<dim, Number> &data,
               BlockVectorType               &dst,
               const BlockVectorType         &src,
               const Range                   &range) const
    {
      FEFaceEvaluation<dim, fe_degree, n_q_1d, n_moving, Number> phi(data, /*interior*/ true);

      for (unsigned int face = range.first; face < range.second; ++face)
        {
          phi.reinit(face);
          phi.read_dof_values(src, 1);
          phi.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto grad_f = phi.get_gradient(q);
              const auto normal = phi.normal_vector(q);

              Tensor<1, n_moving, VA> value;
              VA                      mass_flux = 0.;
              for (unsigned int a = 0; a < n_moving; ++a)
                {
                  const auto [ex, ey] = stream_e[a + 1];
                  value[a] = (0.5 * dt * dt) * (ex * normal[0] + ey * normal[1]) *
                             (ex * grad_f[a][0] + ey * grad_f[a][1]);
                  mass_flux += value[a];
                }
              if (settings.remove_wall_mass_flux)
                for (unsigned int a = 0; a < n_moving; ++a)
                  value[a] -= (D2Q9::w[a + 1] / (1. - D2Q9::w[0])) * mass_flux;
              phi.submit_value(value, q);
            }
          phi.integrate(EvaluationFlags::values);
          phi.distribute_local_to_global(dst, 0);
        }
    }

    void
    no_inner_faces(const MatrixFree<dim, Number> &, BlockVectorType &, const BlockVectorType &, const Range &) const
    {}

    const Disc              &disc;
    const Settings           settings;
    const bool               is_split;

  public:
    MassSolver<fe_degree> mass;
    BlockVectorType       incr; // (A^{-1} r)_alpha of the last compute_increment (Q-1 blocks)

  private:
    StageTimers             &timers;
    BlockVectorType          rhs;       // advection right-hand side r_alpha      (Q-1 blocks)
    BlockVectorType          incr_prev; // previous incr (CG warm start) / second sweep
    std::array<Direction, Q> stream_e = D2Q9::e; // directions of the current sweep
    Number                   dt = 0., lambda = 0.;
    const BlockVectorType   *feq = nullptr; // Lee & Lin: equilibria of f^n
  };
} // namespace lbfem
