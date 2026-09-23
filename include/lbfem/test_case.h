// The interface through which a flow is set up. Everything that distinguishes
// one flow from another is behind it: domain periodicity and mesh
// transformation, the wall nodes and their velocity, the initial populations,
// the reference solution and time scale. The discretization and the schemes
// are case-agnostic; a driver adds a flow by implementing this interface.
#pragma once

#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <lbfem/d2q9.h>

#include <array>
#include <cmath>
#include <optional>
#include <string>

namespace lbfem
{
  using namespace dealii;

  struct TestCase
  {
    // Lattice units (|e_x| = 1, c_s^2 = 1/3): the velocity scale from the Mach
    // number U0 / c_s; the viscosity from the Reynolds number U0 L / nu, which
    // may change between runs (a continuation); lambda = nu / c_s^2.
    explicit TestCase(const double mach)
      : U0(mach * std::sqrt(D2Q9::cs2))
    {}
    virtual ~TestCase() = default;

    const double L = 1., rho0 = 1., U0;
    double       nu = 0.;

    void
    set_reynolds(const double Re)
    {
      nu = U0 * L / Re;
    }
    double
    relaxation_time() const
    {
      return nu / D2Q9::cs2;
    }

    virtual std::string
    name() const = 0;
    virtual std::string
    tag(const double Re) const = 0; // stem of the output files
    virtual unsigned int
    n_periodic() const = 0; // 0, 1 (x only) or 2 periodic directions
    virtual bool
    steady() const = 0; // run to a steady state; else compared with exact()
    virtual double
    reference_time() const = 0;
    // The mesh of the box [0, L]^2 after refinement, and its Courant length h_min.
    virtual void
    transform_mesh(Triangulation<2> &, double &) const
    {}
    // The velocity of the wall a node is on; nothing for an interior node.
    virtual std::optional<Direction>
    wall_velocity(const Point<2> &) const
    {
      return {};
    }
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
    // The populations at t = 0; lambda for a Chapman-Enskog non-equilibrium part.
    virtual std::array<double, D2Q9::Q>
    initial_populations(const Point<2> &p, const double) const
    {
      return D2Q9::equilibrium(exact(p, 0.));
    }
  };
} // namespace lbfem
