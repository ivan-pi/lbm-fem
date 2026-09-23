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

  struct StreamingSettings
  {
    Streaming    streaming = Streaming::tg2;
    MassSettings mass;
  };

  template <int fe_degree>
  class TaylorGalerkin
  {
  public:
    using Disc               = Discretization<fe_degree>;
    using Settings           = StreamingSettings;
    static constexpr int dim = Disc::dim;

    TaylorGalerkin(const Disc                        &disc,
                   const Settings                    &settings,
                   std::unique_ptr<AdvectionOperator> advection,
                   StageTimers                       &timers)
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

    // The increments A^{-1} r(in) of all moving populations (Q-1 blocks).
    const BlockVectorType &
    compute_increment(const AdvectionInput &in, const TimeStep &ts)
    {
      assemble_rhs(in, ts);
      apply_mass_inverse(incr, ts);
      return incr;
    }

    // f_alpha += A^{-1} r(f)_alpha for the moving populations (collide, then
    // stream). tg3_split: the shift by e dt is the product of the shifts by
    // (ex dt, 0) and (0, ey dt); on a tensor-product grid the 1-D sweeps commute
    // and the composition is exact at CFL 1 for all eight directions (the
    // diagonal populations get both sweeps, the others one).
    void
    stream(BlockVectorType &f, const TimeStep &ts)
    {
      const auto add_increment = [&](const BlockVectorType &x) {
        timers.time(StageTimers::collision, [&] { // nodal work
          for (unsigned int a = 0; a < n_moving; ++a)
            if (streamed(a))
              f.block(a + 1) += x.block(a);
        });
      };

      if (!is_split)
        {
          add_increment(compute_increment({.f = f}, ts));
          return;
        }
      for (unsigned int d = 0; d < dim; ++d)
        {
          for (unsigned int a = 0; a < Q; ++a)
            stream_e[a] = d == 0 ? Direction{{D2Q9::e[a][0], 0.}} : Direction{{0., D2Q9::e[a][1]}};
          auto &x = d == 0 ? incr : incr_prev;
          assemble_rhs({.f = f}, ts);
          apply_mass_inverse(x, ts);
          add_increment(x);
        }
      stream_e = D2Q9::e;
    }

    // The work counters of the mass solves (for a performance model).
    const MassSolver<fe_degree> &
    mass_solver() const
    {
      return mass;
    }

    const AdvectionOperator &
    advection_operator() const
    {
      return *advection;
    }

  private:
    // Whether moving population a + 1 moves in the current sweep.
    bool
    streamed(const unsigned int a) const
    {
      return stream_e[a + 1][0] != 0. || stream_e[a + 1][1] != 0.;
    }

    void
    assemble_rhs(const AdvectionInput &in, const TimeStep &ts)
    {
      timers.time(StageTimers::advection, [&] { advection->apply(rhs, in, stream_e, ts); });
    }

    // x <- A^{-1} rhs per population. Mass::cg starts from the extrapolation
    // 2 x^{n} - x^{n-1} (the increment changes slowly in time); for the split
    // sweeps the previous sweep's solution is used instead. Populations that
    // do not move in this sweep are skipped (x is left as it is).
    void
    apply_mass_inverse(BlockVectorType &X, const TimeStep &ts)
    {
      StageTimers::Scope t(timers, StageTimers::mass);
      const Number       tg3 = settings.streaming == Streaming::tg2 ? 0. : ts.dt * ts.dt / 6.;
      for (unsigned int a = 0; a < n_moving; ++a)
        {
          if (!streamed(a))
            continue;
          auto       &x = X.block(a);
          const auto &r = rhs.block(a);
          if (settings.mass.type == Mass::cg && !is_split)
            {
              incr_prev.block(a).sadd(-1., 2., x); // 2 x^n - x^{n-1}
              x.swap(incr_prev.block(a));          // x: guess, incr_prev: x^n
            }
          mass.solve(x, r, stream_e[a + 1], tg3);
        }
    }

    const Settings                     settings;
    const bool                         is_split;
    std::unique_ptr<AdvectionOperator> advection;
    StageTimers                       &timers;
    Directions                         stream_e = D2Q9::e; // directions of the current sweep
    BlockVectorType                    rhs;       // r_alpha                      (Q-1 blocks)
    BlockVectorType                    incr;      // (A^{-1} r)_alpha              (Q-1 blocks)
    BlockVectorType                    incr_prev; // previous incr (CG warm start) / second sweep
    MassSolver<fe_degree>              mass;
  };
} // namespace lbfem
