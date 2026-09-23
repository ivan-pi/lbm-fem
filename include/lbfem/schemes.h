// Characteristic Galerkin schemes for the discrete Boltzmann equation, with
// lambda = nu / c_s^2 the relaxation time of the continuous equation in both:
//
//   LeeLin: predictor-corrector of T. Lee, C.-L. Lin, J. Comput. Phys. 171,
//   336 (2001), for alpha = 1..8, theta = dt/lambda:
//     (1 + theta) M fhat = M (f + theta feq) - dt C f - dt^2 [D f + Q (f - feq)],
//     f^{n+1} = fhat + theta (feq(fhat) - feq(f^n))
//
//   Bardow: collide-then-stream of A. Bardow, I. V. Karlin, A. A. Gusev,
//   EPL 75, 434 (2006), in the transformed populations
//   g = f + dt/(2 lambda) (f - feq):
//     g* = g - omega (g - geq),  omega = dt / (lambda + dt/2)    (exact BGK)
//     M (g^{n+1} - g*) = -dt C g* - dt^2 D g*      (weak form of their Eq. 9)
//
// Each scheme streams through an AdvectionOperator (advection.h): by default
// the one of its paper, or any other a driver passes in. The rest population
// (e_0 = 0) needs neither loop nor solve.
#pragma once

#include <lbfem/advection.h>
#include <lbfem/collision.h>
#include <lbfem/discretization.h>
#include <lbfem/streaming.h>
#include <lbfem/test_case.h>
#include <lbfem/timers.h>

#include <memory>

namespace lbfem
{
  class Scheme
  {
  public:
    using Advection = std::unique_ptr<AdvectionOperator>;

    virtual ~Scheme() = default;

    // The time step is fixed; the relaxation time lambda = nu / c_s^2 may change
    // between runs (a Reynolds number continuation).
    void
    set_time_step(const TimeStep &time_step)
    {
      ts = time_step;
    }

    virtual void
    step() = 0;

    // The populations of the test case at t = 0, at every locally owned node.
    virtual void
    set_initial_populations(const TestCase &tc)
    {
      for (unsigned int i = 0; i < disc.n_nodes(); ++i)
        {
          const auto fi = tc.initial_populations(disc.node[i], ts.lambda);
          for (unsigned int a = 0; a < Q; ++a)
            f.block(a).local_element(i) = fi[a];
        }
    }

    StageTimers     timers;
    TaylorGalerkin  streaming;
    BlockVectorType f; // populations: f (LeeLin) or g (Bardow), Q blocks

  protected:
    Scheme(const Discretization &disc, const StreamingSettings &settings, Advection advection)
      : streaming(disc, settings, std::move(advection), timers)
      , disc(disc)
    {
      disc.initialize(f, Q);
    }

    const Discretization &disc;
    TimeStep              ts{0., 0.};
  };



  class LeeLin final : public Scheme
  {
  public:
    // TG2 streaming only; by default with the advection of Lee & Lin, which
    // keeps the wall surface term as it is. A custom advection receives the
    // equilibria in AdvectionInput::feq.
    LeeLin(const Discretization &disc, const StreamingSettings &settings, Advection advection = nullptr)
      : Scheme(disc, settings, advection ? std::move(advection) : std::make_unique<LeeLinAdvection>(disc))
    {
      AssertThrow(settings.streaming == Streaming::tg2, ExcMessage("LeeLin streams with TG2 only"));
      disc.initialize(feq, Q);
    }

    void
    step() override
    {
      timers.time(StageTimers::collision, [&] { compute_equilibrium(f, feq, disc.walls); });
      const auto &incr = streaming.increment({.f = f, .feq = &feq}, ts);
      timers.time(StageTimers::collision, [&] { predictor_corrector(f, feq, incr, disc.walls, ts.dt / ts.lambda); });
    }

  private:
    BlockVectorType feq; // nodal equilibria of f^n (Q blocks)
  };



  class Bardow final : public Scheme
  {
  public:
    // Any streaming variant; by default with the advection of Bardow et al.,
    // which removes the mass flux of the wall surface term.
    Bardow(const Discretization &disc, const StreamingSettings &settings, Advection advection = nullptr)
      : Scheme(disc,
               settings,
               advection ? std::move(advection) :
                           std::make_unique<TaylorGalerkinAdvection>(disc, /*remove_wall_mass_flux*/ true))
    {}

    void
    step() override
    {
      const auto [dt, lambda] = ts;
      timers.time(StageTimers::collision, [&] { collide_bgk(f, disc.walls, dt / (lambda + 0.5 * dt)); });
      streaming.stream(f, ts);
    }

    // The transformed populations g = f + dt/(2 lambda) (f - feq) of the
    // initial f (feq from its moments).
    void
    set_initial_populations(const TestCase &tc) override
    {
      Scheme::set_initial_populations(tc);
      const Number   scale = 0.5 * ts.dt / ts.lambda;
      const Nodal<Q> F(f);
      for (unsigned int i = 0; i < disc.n_nodes(); ++i)
        {
          const Populations fi  = F[i];
          const Populations feq = D2Q9::equilibrium(D2Q9::moments(fi));
          for (unsigned int a = 0; a < Q; ++a)
            f.block(a).local_element(i) = fi[a] + scale * (fi[a] - feq[a]);
        }
    }
  };
} // namespace lbfem
