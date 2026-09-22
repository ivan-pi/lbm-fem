// Characteristic Galerkin (Taylor-Galerkin) streaming of the moving
// populations: the increment of population alpha is
//   x_alpha = A^{-1} r_alpha,  A = M (TG2) or M + dt^2/6 K_e (TG3),
// with r_alpha from an AdvectionOperator (advection.h) and the solve from a
// MassSolver (mass.h).
#pragma once

#include <lbfem/advection.h>
#include <lbfem/collision.h>
#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>
#include <lbfem/mass.h>
#include <lbfem/timers.h>

#include <memory>

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
    static constexpr int dim = Disc::dim;

    struct Settings
    {
      Streaming                                streaming = Streaming::tg2;
      typename MassSolver<fe_degree>::Settings mass;
    };

    TaylorGalerkin(const Disc                                  &disc,
                   const Settings                              &settings,
                   std::unique_ptr<AdvectionOperator<fe_degree>> advection,
                   StageTimers                                 &timers)
      : settings(settings)
      , is_split(settings.streaming == Streaming::tg3_split)
      , advection(std::move(advection))
      , timers(timers)
      , mass(disc, settings.mass)
    {
      AssertThrow(settings.streaming == Streaming::tg2 || settings.mass.type == Mass::cg,
                  ExcMessage("TG3 streaming needs the consistent mass (Mass::cg)"));
      disc.initialize(rhs, n_moving);
      disc.initialize(incr, n_moving);
      disc.initialize(incr_prev, n_moving);
    }

    void
    set_time_step(const TimeStep &time_step)
    {
      ts = time_step;
    }

    // incr <- A^{-1} r(in) for all moving populations.
    void
    compute_increment(const AdvectionInput &in)
    {
      assemble_rhs(in);
      apply_mass_inverse(incr);
    }

    // f_alpha += A^{-1} r(f)_alpha for the moving populations (collide, then
    // stream). tg3_split: the shift by e dt is the product of the shifts by
    // (ex dt, 0) and (0, ey dt); on a tensor-product grid the 1-D sweeps commute
    // and the composition is exact at CFL 1 for all eight directions (the
    // diagonal populations get both sweeps).
    void
    stream(BlockVectorType &f)
    {
      const auto add_increment = [&](const BlockVectorType &x) {
        timers.time(StageTimers::collision, [&] { // nodal work
          for (unsigned int a = 0; a < n_moving; ++a)
            f.block(a + 1) += x.block(a);
        });
      };

      if (!is_split)
        {
          compute_increment({.f = f});
          add_increment(incr);
          return;
        }
      for (unsigned int d = 0; d < dim; ++d)
        {
          for (unsigned int a = 0; a < Q; ++a)
            stream_e[a] = d == 0 ? Direction{{D2Q9::e[a][0], 0.}} : Direction{{0., D2Q9::e[a][1]}};
          auto &x = d == 0 ? incr : incr_prev;
          assemble_rhs({.f = f});
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
    assemble_rhs(const AdvectionInput &in)
    {
      timers.time(StageTimers::advection, [&] { advection->apply(rhs, in, stream_e, ts); });
    }

    // x <- A^{-1} rhs per population. Mass::cg starts from the extrapolation
    // 2 x^{n} - x^{n-1} (the increment changes slowly in time); for the split
    // sweeps the previous sweep's solution is used instead.
    void
    apply_mass_inverse(BlockVectorType &X)
    {
      StageTimers::Scope t(timers, StageTimers::mass);
      const Number       tg3 = settings.streaming == Streaming::tg2 ? 0. : ts.dt * ts.dt / 6.;
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

    const Settings                                settings;
    const bool                                    is_split;
    std::unique_ptr<AdvectionOperator<fe_degree>> advection;
    StageTimers                                  &timers;
    TimeStep                                      ts{0., 0.};
    Directions                                    stream_e = D2Q9::e; // directions of the current sweep
    BlockVectorType                               rhs;       // r_alpha              (Q-1 blocks)
    BlockVectorType                               incr_prev; // previous incr (CG warm start) / second sweep

  public:
    MassSolver<fe_degree> mass;
    BlockVectorType       incr; // (A^{-1} r)_alpha of the last compute_increment (Q-1 blocks)
  };
} // namespace lbfem
