// Characteristic Galerkin schemes for the discrete Boltzmann equation, with
// lambda = nu / c_s^2 the relaxation time of the continuous equation in both:
//
//   LeeLin: predictor-corrector of T. Lee, C.-L. Lin, J. Comput. Phys. 171,
//   336 (2001), for alpha = 1..8:
//     (1 + tau) M fhat = M (f + tau feq) - dt C f - dt^2 [D f + Q (f - feq)],
//     f^{n+1} = fhat + tau (feq(fhat) - feq(f^n)),   tau = dt/lambda
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
  template <int fe_degree>
  class Scheme
  {
  public:
    using Disc      = Discretization<fe_degree>;
    using Advection = std::unique_ptr<AdvectionOperator>;
    using Settings  = StreamingSettings;

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

    // Multiplier of the non-equilibrium part of the initial populations: 1 for
    // f, 1 + dt/(2 lambda) for the transformed g.
    virtual Number
    neq_scale() const = 0;

    // The nodal (collision) work per node and time step (see advection.h).
    virtual Work
    collision_work() const = 0;

    // Initial populations of the test case at every locally owned node.
    void
    set_initial_populations(const TestCase &tc)
    {
      const Number scale = neq_scale();
      nodal_map(f, [&](const unsigned int i) { return tc.initial_populations(disc.node[i], ts.lambda, scale); });
    }

  protected:
    Scheme(const Disc &disc, const Walls &walls, const Settings &settings, Advection advection, StageTimers &timers)
      : disc(disc)
      , walls(walls)
      , timers(timers)
      , streaming(disc, settings, std::move(advection), timers)
    {
      disc.initialize(f, Q);
    }

    const Disc  &disc;
    const Walls &walls;
    StageTimers &timers;
    TimeStep     ts{0., 0.};

  public:
    TaylorGalerkin<fe_degree> streaming;
    BlockVectorType           f; // populations: f (LeeLin) or g (Bardow), Q blocks
  };



  template <int fe_degree>
  class LeeLin final : public Scheme<fe_degree>
  {
  public:
    using Base = Scheme<fe_degree>;

    // TG2 streaming only; by default with the advection of Lee & Lin, which
    // keeps the wall surface term as it is. A custom advection receives the
    // equilibria in AdvectionInput::feq.
    LeeLin(const typename Base::Disc     &disc,
           const Walls                   &walls,
           const typename Base::Settings &settings,
           StageTimers                   &timers,
           typename Base::Advection       advection = nullptr)
      : Base(disc,
             walls,
             settings,
             advection ? std::move(advection) : std::make_unique<LeeLinAdvection<fe_degree>>(disc),
             timers)
    {
      AssertThrow(settings.streaming == Streaming::tg2, ExcMessage("LeeLin streams with TG2 only"));
      disc.initialize(feq, Q);
    }

    void
    step() override
    {
      this->timers.time(StageTimers::collision, [&] { compute_equilibrium(this->f, feq, this->walls); });
      const auto &incr = this->streaming.compute_increment({.f = this->f, .feq = &feq}, this->ts);
      this->timers.time(StageTimers::collision, [&] {
        predictor_corrector(this->f, feq, Moving{incr}, this->walls, this->ts.dt / this->ts.lambda);
      });
    }

    Number
    neq_scale() const override
    {
      return 1.;
    }

    // Reads f (9), writes feq (9), then reads f, feq, incr (9+9+8) and writes f
    // (9); equilibrium 122, predictor 32, equilibrium of fhat 122, corrector 27
    // (the wall nodes, O(sqrt N), are not counted).
    Work
    collision_work() const override
    {
      return {18 + 35, 303};
    }

  private:
    BlockVectorType feq; // nodal equilibria of f^n (Q blocks)
  };



  template <int fe_degree>
  class Bardow final : public Scheme<fe_degree>
  {
  public:
    using Base = Scheme<fe_degree>;

    // Any streaming variant; by default with the advection of Bardow et al.,
    // which removes the mass flux of the wall surface term.
    Bardow(const typename Base::Disc     &disc,
           const Walls                   &walls,
           const typename Base::Settings &settings,
           StageTimers                   &timers,
           typename Base::Advection       advection = nullptr)
      : Base(disc,
             walls,
             settings,
             advection ? std::move(advection) :
                         std::make_unique<TaylorGalerkinAdvection<fe_degree>>(disc, /*remove_wall_mass_flux*/ true),
             timers)
    {}

    void
    step() override
    {
      const auto [dt, lambda] = this->ts;
      this->timers.time(StageTimers::collision, [&] { collide_bgk(this->f, this->walls, dt / (lambda + 0.5 * dt)); });
      this->streaming.stream(this->f, this->ts);
    }

    Number
    neq_scale() const override
    {
      return 1. + 0.5 * this->ts.dt / this->ts.lambda;
    }

    // Reads and writes g (9+9) and adds incr to g (8+8 read, 8 write); moments
    // 20, equilibrium 102, relaxation 27, increment 8 (the wall nodes, O(sqrt
    // N), are not counted).
    Work
    collision_work() const override
    {
      return {18 + 24, 157};
    }
  };
} // namespace lbfem
