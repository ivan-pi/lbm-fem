// Characteristic Galerkin (Taylor-Galerkin) streaming of the moving
// populations: the increment of population alpha is
//   x_alpha = A^{-1} r_alpha,  A = M (TG2) or M + dt^2/6 K_e (TG3),
// with r_alpha from an AdvectionOperator (advection.h) and the solve from a
// MassSolver (mass.h). A step is one sweep along the lattice velocities e, or
// (tg3_split) an x sweep followed by a y sweep: the shift by e dt is the
// product of the shifts by (ex dt, 0) and (0, ey dt); on a tensor-product grid
// the 1-D sweeps commute and the composition is exact at CFL 1 for all eight
// directions (the diagonal populations get both sweeps, the others one).
#pragma once

#include <lbfem/advection.h>
#include <lbfem/collision.h>
#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>
#include <lbfem/mass.h>
#include <lbfem/timers.h>

#include <array>
#include <memory>
#include <vector>

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

  class TaylorGalerkin
  {
  public:
    TaylorGalerkin(const Discretization              &disc,
                   const StreamingSettings           &settings,
                   std::unique_ptr<AdvectionOperator> advection,
                   StageTimers                       &timers)
      : advection(std::move(advection))
      , mass(disc, settings.mass)
      , tg3(settings.streaming == Streaming::tg2 ? 0. : 1. / 6.)
      , extrapolate(settings.mass.type == Mass::cg && settings.streaming != Streaming::tg3_split)
      , timers(timers)
    {
      AssertThrow(settings.streaming == Streaming::tg2 || settings.mass.type == Mass::cg,
                  ExcMessage("TG3 streaming needs the consistent mass (Mass::cg)"));
      if (settings.streaming == Streaming::tg3_split)
        sweeps = {axis(0), axis(1)};
      else
        sweeps = {D2Q9::e};
      disc.initialize(rhs, n_moving);
      for (auto &v : x)
        disc.initialize(v, n_moving);
    }

    // The increments A^{-1} r(in) of the moving populations (Q-1 blocks) along
    // the directions of sweep s; populations at rest in the sweep are skipped.
    // Mass::cg starts from the extrapolation 2 x^n - x^{n-1} of the previous
    // increments (they change slowly in time), except for the split sweeps.
    const BlockVectorType &
    increment(const AdvectionInput &in, const TimeStep &ts, const unsigned int s = 0)
    {
      const Directions &e = sweeps[s];
      timers.time(StageTimers::advection, [&] { advection->apply(rhs, in, e, ts); });
      const StageTimers::Scope t(timers, StageTimers::mass);
      for (unsigned int a = 0; a < n_moving; ++a)
        if (moves(e, a))
          {
            auto &xa = x[s].block(a);
            if (extrapolate)
              {
                x[1].block(a).sadd(-1., 2., xa); // 2 x^n - x^{n-1}
                xa.swap(x[1].block(a));          // xa: guess, x[1]: x^n
              }
            mass.solve(xa, rhs.block(a), e[a + 1], tg3 * ts.dt * ts.dt);
          }
      return x[s];
    }

    // f_alpha += A^{-1} r(f)_alpha for the moving populations (collide, then
    // stream), sweep by sweep.
    void
    stream(BlockVectorType &f, const TimeStep &ts)
    {
      for (unsigned int s = 0; s < sweeps.size(); ++s)
        {
          const auto &xs = increment({.f = f}, ts, s);
          timers.time(StageTimers::collision, [&] { // nodal work
            for (unsigned int a = 0; a < n_moving; ++a)
              if (moves(sweeps[s], a))
                f.block(a + 1) += xs.block(a);
          });
        }
    }

    std::vector<Directions>            sweeps; // the directions of all populations, per sweep
    std::unique_ptr<AdvectionOperator> advection;
    MassSolver                         mass;

  private:
    // D2Q9::e with the other component zeroed: the directions of the split sweep along axis d.
    static Directions
    axis(const unsigned int d)
    {
      Directions e{};
      for (unsigned int a = 0; a < Q; ++a)
        e[a][d] = D2Q9::e[a][d];
      return e;
    }

    // Whether moving population a + 1 moves in a sweep.
    static bool
    moves(const Directions &e, const unsigned int a)
    {
      return e[a + 1][0] != 0. || e[a + 1][1] != 0.;
    }

    const Number                   tg3;         // dt^2 coefficient of K_e: 0 (TG2) or 1/6
    const bool                     extrapolate; // CG warm start from x[0] (x^n) and x[1] (x^{n-1})
    StageTimers                   &timers;
    BlockVectorType                rhs; // r_alpha                          (Q-1 blocks)
    std::array<BlockVectorType, 2> x;   // the increment of sweep s (Q-1 blocks); with one sweep and
                                        // extrapolate, x[1] holds the previous step's
  };
} // namespace lbfem
