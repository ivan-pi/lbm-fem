// The interface through which a flow is set up. Everything that distinguishes
// one flow from another is behind it: domain periodicity and mesh
// transformation, wall nodes and their velocity, initial populations, the
// reference solution and time scale. The discretization and the schemes are
// case-agnostic; a driver adds a flow by implementing this interface.
#pragma once

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>

#include <lbfem/d2q9.h>

#include <array>
#include <string>

namespace lbfem
{
  using namespace dealii;

  struct TestCase
  {
    double L = 1., U0 = 0., nu = 0., rho0 = 1.; // set by the driver

    virtual ~TestCase() = default;
    virtual std::string
    name() const = 0;
    virtual std::string
    tag(const double Re) const = 0; // stem of the output files
    virtual unsigned int
    n_periodic() const = 0; // 0, 1 (x only) or 2 periodic directions
    virtual void
    transform_mesh(Triangulation<2> &, double &) const
    {}
    virtual int
    wall_of(const Point<2> &) const // -1 interior, 0 stationary wall, 1 moving lid
    {
      return -1;
    }
    virtual std::array<double, 2>
    wall_velocity(const int wall) const
    {
      return wall == 1 ? std::array<double, 2>{{U0, 0.}} : std::array<double, 2>{{0., 0.}};
    }
    virtual double
    reference_time() const = 0;
    virtual bool
    has_exact_solution() const = 0; // else: run to a steady state
    virtual D2Q9::Moments
    exact(const Point<2> &, const double) const
    {
      return {rho0, 0., 0.};
    }
    virtual double
    energy_scale() const // normalisation of <|u|^2> in the diagnostics
    {
      return U0 * U0;
    }
    // initial populations; neq_scale multiplies the non-equilibrium part
    // (1 for f, 1 + dt/(2 lambda) for the transformed g of the bardow scheme)
    virtual std::array<double, D2Q9::Q>
    initial_populations(const Point<2> &, const double lambda, const double neq_scale) const = 0;
  };
} // namespace lbfem
