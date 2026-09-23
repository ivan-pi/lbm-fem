// The benchmark flows of the papers: Taylor-Green vortex (Bardow et al.),
// start-up Couette flow and lid-driven cavity (Lee & Lin).
#pragma once

#include <deal.II/grid/grid_tools.h>

#include <lbfem/d2q9.h>
#include <lbfem/test_case.h>

#include <cmath>
#include <numbers>
#include <optional>
#include <string>

namespace lbfem
{
  // Taylor-Green vortex with wave numbers k1, k2 (Bardow et al., Eqs. (14)-(16)):
  //   u = -U0 cos(k1 x) sin(k2 y) F,  v = U0 (k1/k2) sin(k1 x) cos(k2 y) F,
  //   p = p0 - rho0 U0^2/4 [cos(2 k1 x) + (k1/k2)^2 cos(2 k2 y)] F^2,
  //   F = exp(-nu (k1^2 + k2^2) t).  k1 != k2 gives a non-zero shear u_y + v_x.
  struct TaylorGreenVortex : TestCase
  {
    double n1, n2, distort;

    TaylorGreenVortex(const double mach, const unsigned int n1, const unsigned int n2, const double distort)
      : TestCase(mach)
      , n1(n1)
      , n2(n2)
      , distort(distort)
    {}

    std::string
    name() const override
    {
      return "Taylor-Green vortex, periodic, modes (" + std::to_string(int(n1)) + "," + std::to_string(int(n2)) + ")";
    }
    std::string
    tag(const double) const override
    {
      return "vortex";
    }
    unsigned int
    n_periodic() const override
    {
      return 2;
    }
    bool
    steady() const override
    {
      return false;
    }
    // smooth periodic distortion: general (non-parallelogram) quadrilaterals
    void
    transform_mesh(Triangulation<2> &tria, double &) const override
    {
      if (distort == 0.)
        return;
      GridTools::transform(
        [&](const Point<2> &p) {
          const double d =
            distort * L * std::sin(2 * std::numbers::pi * p[0] / L) * std::sin(2 * std::numbers::pi * p[1] / L);
          return Point<2>(p[0] + d, p[1] + d);
        },
        tria);
    }
    double
    k1() const
    {
      return 2. * std::numbers::pi * n1 / L;
    }
    double
    k2() const
    {
      return 2. * std::numbers::pi * n2 / L;
    }
    double
    decay(const double t) const
    {
      return std::exp(-nu * (k1() * k1() + k2() * k2()) * t);
    }
    double
    reference_time() const override // velocity decay time
    {
      return 1. / (nu * (k1() * k1() + k2() * k2()));
    }
    double
    energy_scale() const override // exact mean of |u|^2 at t = 0
    {
      const double r = k1() / k2();
      return 0.25 * U0 * U0 * (1. + r * r);
    }
    D2Q9::Moments
    exact(const Point<2> &p, const double t) const override
    {
      const double F = decay(t), x1 = k1() * p[0], y2 = k2() * p[1], r = k1() / k2();
      // p = cs^2 rho  =>  rho = rho0 + (p - p0)/cs^2
      const double dp = -0.25 * rho0 * U0 * U0 * F * F * (std::cos(2 * x1) + r * r * std::cos(2 * y2));
      return {rho0 + dp / D2Q9::cs2, -U0 * F * std::cos(x1) * std::sin(y2), U0 * r * F * std::sin(x1) * std::cos(y2)};
    }
    // equilibrium + first-order Chapman-Enskog part
    //   f1 = -lambda w rho (e e - cs^2 I) : grad u / cs^2
    //      = -lambda w rho [ (ex^2 - ey^2) u_x + ex ey (u_y + v_x) ] / cs^2   (div u = 0),
    // which removes the initial layer
    std::array<double, D2Q9::Q>
    initial_populations(const Point<2> &p, const double lambda) const override
    {
      const auto   m  = exact(p, 0.);
      auto         f  = D2Q9::equilibrium(m);
      const double x1 = k1() * p[0], y2 = k2() * p[1];
      const double ux = U0 * k1() * std::sin(x1) * std::sin(y2);                               // du/dx
      const double sh = U0 * (k1() * k1() - k2() * k2()) / k2() * std::cos(x1) * std::cos(y2); // du/dy + dv/dx
      for (unsigned int a = 0; a < D2Q9::Q; ++a)
        {
          const auto [ex, ey] = D2Q9::e[a];
          f[a] -= lambda / D2Q9::cs2 * D2Q9::w[a] * m.rho * ((ex * ex - ey * ey) * ux + ex * ey * sh);
        }
      return f;
    }
  };



  // Start-up Couette flow between y = 0 (at rest) and y = L (velocity U0),
  // periodic in x; Eq. (26) of Lee & Lin.
  struct CouetteFlow : TestCase
  {
    using TestCase::TestCase;

    std::string
    name() const override
    {
      return "start-up Couette flow";
    }
    std::string
    tag(const double) const override
    {
      return "couette";
    }
    unsigned int
    n_periodic() const override
    {
      return 1;
    }
    bool
    steady() const override
    {
      return false;
    }
    std::optional<Direction>
    wall_velocity(const Point<2> &p) const override
    {
      if (std::abs(p[1]) < 1e-12 * L)
        return Direction{{0., 0.}};
      if (std::abs(p[1] - L) < 1e-12 * L)
        return Direction{{U0, 0.}};
      return {};
    }
    double
    reference_time() const override // diffusion time
    {
      return L * L / nu;
    }
    D2Q9::Moments
    exact(const Point<2> &p, const double t) const override
    {
      const double y = p[1];
      double       u = U0 * y / L;
      for (int m = 1; m <= 400; ++m)
        {
          const double lm = m * std::numbers::pi / L;
          u += 2. * U0 * (m % 2 ? -1. : 1.) / (lm * L) * std::exp(-nu * lm * lm * t) * std::sin(lm * y);
        }
      return {rho0, u, 0.};
    }
    std::array<double, D2Q9::Q>
    initial_populations(const Point<2> &, const double) const override
    {
      return D2Q9::equilibrium({rho0, 0., 0.}); // fluid at rest (the series above is not, at t = 0)
    }
  };



  // Lid-driven cavity (Lee & Lin, Sec. 3.2) on a tanh-clustered mesh, run to a
  // steady state from rest; a list of Reynolds numbers is a continuation.
  struct LidDrivenCavity : TestCase
  {
    double stretch;

    LidDrivenCavity(const double mach, const double stretch)
      : TestCase(mach)
      , stretch(stretch)
    {}

    std::string
    name() const override
    {
      return "lid-driven cavity";
    }
    std::string
    tag(const double Re) const override
    {
      return "cavity_Re" + std::to_string(std::lround(Re));
    }
    unsigned int
    n_periodic() const override
    {
      return 0;
    }
    bool
    steady() const override
    {
      return true;
    }
    void
    transform_mesh(Triangulation<2> &tria, double &h_min) const override
    {
      if (stretch <= 0.)
        return;
      const auto cluster = [&](const double x) {
        return 0.5 * L * (1. + std::tanh(stretch * (2. * x / L - 1.)) / std::tanh(stretch));
      };
      GridTools::transform([&](const Point<2> &p) { return Point<2>(cluster(p[0]), cluster(p[1])); }, tria);
      h_min = cluster(h_min);
    }
    // The top corners belong to the side walls: a moving corner node would carry
    // momentum through the side walls (a mass source/sink pair).
    std::optional<Direction>
    wall_velocity(const Point<2> &p) const override
    {
      const auto on = [&](const double x, const double x0) {
        return std::abs(x - x0) < 1e-12 * L;
      };
      if (on(p[0], 0.) || on(p[0], L) || on(p[1], 0.))
        return Direction{{0., 0.}};
      if (on(p[1], L))
        return Direction{{U0, 0.}};
      return {};
    }
    double
    reference_time() const override // lid time
    {
      return L / U0;
    }
  };
} // namespace lbfem
