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
// The rest population (e_0 = 0) needs neither loop nor solve.
#pragma once

#include <lbfem/collision.h>
#include <lbfem/discretization.h>
#include <lbfem/streaming.h>
#include <lbfem/test_case.h>
#include <lbfem/timers.h>

namespace lbfem
{
  template <int fe_degree>
  class Scheme
  {
  public:
    using Disc = Discretization<fe_degree>;

    Scheme(const Disc                                      &disc,
           const Walls                                     &walls,
           const typename TaylorGalerkin<fe_degree>::Settings &streaming_settings,
           StageTimers                                     &timers)
      : disc(disc)
      , walls(walls)
      , timers(timers)
      , streaming(disc, streaming_settings, timers)
    {
      disc.initialize(f, Q);
    }

    virtual ~Scheme() = default;

    // The time step is fixed; the relaxation time lambda = nu / c_s^2 may change
    // between runs (a Reynolds number continuation).
    virtual void
    set_time_step(const Number time_step, const Number relaxation_time)
    {
      dt     = time_step;
      lambda = relaxation_time;
      streaming.set_time_step(dt, lambda);
    }

    virtual void
    step() = 0;

    // Multiplier of the non-equilibrium part of the initial populations: 1 for
    // f, 1 + dt/(2 lambda) for the transformed g.
    virtual Number
    neq_scale() const = 0;

    // Initial populations of the test case at every locally owned node.
    void
    set_initial_populations(const TestCase &tc)
    {
      const auto F = raw(f);
      for (unsigned int i = 0; i < disc.n_nodes(); ++i)
        {
          const auto f0 = tc.initial_populations(disc.node[i], lambda, neq_scale());
          for (unsigned int a = 0; a < Q; ++a)
            F[a][i] = f0[a];
        }
    }

    BlockVectorType f; // populations: f (LeeLin) or g (Bardow), Q blocks

  protected:
    const Disc  &disc;
    const Walls &walls;
    StageTimers &timers;
    Number       dt = 0., lambda = 0.;

  public:
    TaylorGalerkin<fe_degree> streaming;
  };



  template <int fe_degree>
  class LeeLin : public Scheme<fe_degree>
  {
  public:
    using Base = Scheme<fe_degree>;

    // Lee & Lin keep the wall surface term as it is and use TG2 streaming.
    LeeLin(const typename Base::Disc &disc,
           const Walls               &walls,
           const typename MassSolver<fe_degree>::Settings &mass,
           StageTimers               &timers)
      : Base(disc, walls, {Streaming::tg2, mass, /*remove_wall_mass_flux*/ false}, timers)
    {
      disc.initialize(feq, Q);
    }

    void
    step() override
    {
      {
        StageTimers::Scope t(this->timers, StageTimers::collision);
        compute_equilibrium(this->f, feq, this->walls);
      }
      this->streaming.compute_increment(this->f, &feq);
      {
        StageTimers::Scope t(this->timers, StageTimers::collision);
        predictor_corrector(this->f, feq, this->streaming.incr, this->walls, this->dt / this->lambda);
      }
    }

    Number
    neq_scale() const override
    {
      return 1.;
    }

  private:
    BlockVectorType feq; // nodal equilibria of f^n (Q blocks)
  };



  template <int fe_degree>
  class Bardow : public Scheme<fe_degree>
  {
  public:
    using Base = Scheme<fe_degree>;

    // Any streaming variant; the mass flux of the wall surface term is removed.
    Bardow(const typename Base::Disc &disc,
           const Walls               &walls,
           const Streaming            streaming,
           const typename MassSolver<fe_degree>::Settings &mass,
           StageTimers               &timers)
      : Base(disc, walls, {streaming, mass, /*remove_wall_mass_flux*/ true}, timers)
    {}

    void
    step() override
    {
      {
        StageTimers::Scope t(this->timers, StageTimers::collision);
        collide_bgk(this->f, this->walls, this->dt / (this->lambda + 0.5 * this->dt));
      }
      this->streaming.stream(this->f);
    }

    Number
    neq_scale() const override
    {
      return 1. + 0.5 * this->dt / this->lambda;
    }
  };
} // namespace lbfem
