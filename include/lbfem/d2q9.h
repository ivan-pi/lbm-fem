// D2Q9 lattice: discrete velocities, weights, moments and the second-order
// equilibrium, in the numbering of Fig. 1 / Eq. (3) of Lee & Lin (2001).
// Lattice units: |e_x| = 1, c_s^2 = 1/3. No deal.II dependency.
#pragma once

#include <array>

namespace lbfem
{
  using Direction = std::array<double, 2>; // a lattice velocity, or any vector in the plane
}

namespace lbfem::D2Q9
{
  inline constexpr unsigned int Q   = 9;
  inline constexpr double       cs2 = 1. / 3.;

  inline constexpr std::array<Direction, Q> e = {
    {{{0, 0}}, {{1, 0}}, {{1, 1}}, {{0, 1}}, {{-1, 1}}, {{-1, 0}}, {{-1, -1}}, {{0, -1}}, {{1, -1}}}};

  inline constexpr std::array<double, Q> w = {
    {4. / 9., 1. / 9., 1. / 36., 1. / 9., 1. / 36., 1. / 9., 1. / 36., 1. / 9., 1. / 36.}};

  struct Moments
  {
    double rho, ux, uy;
  };

  // Eq. (5)
  constexpr Moments
  moments(const std::array<double, Q> &f)
  {
    double rho = 0, mx = 0, my = 0;
    for (unsigned int a = 0; a < Q; ++a)
      {
        rho += f[a];
        mx += e[a][0] * f[a];
        my += e[a][1] * f[a];
      }
    return {rho, mx / rho, my / rho};
  }

  // Eq. (4)
  constexpr std::array<double, Q>
  equilibrium(const Moments &m)
  {
    std::array<double, Q> feq{};
    const double          uu = m.ux * m.ux + m.uy * m.uy;
    for (unsigned int a = 0; a < Q; ++a)
      {
        const double eu = e[a][0] * m.ux + e[a][1] * m.uy;
        feq[a]          = w[a] * m.rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * uu);
      }
    return feq;
  }

  // Checked at compile time: the lattice tensors are isotropic up to 4th
  // order (as the Navier-Stokes limit needs), sum_a w_a e_a...e_a =
  //   1, 0, c_s^2 delta_ij, 0, c_s^4 (delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk),
  // and the equilibrium has the density and velocity it was built from.
  namespace detail
  {
    constexpr bool
    near(const double x, const double y)
    {
      return (x > y ? x - y : y - x) < 1e-14;
    }

    template <typename... Index>
    constexpr double
    lattice_tensor(const Index... i)
    {
      double sum = 0;
      for (unsigned int a = 0; a < Q; ++a)
        sum += (w[a] * ... * e[a][i]);
      return sum;
    }
  } // namespace detail

  static_assert(detail::near(detail::lattice_tensor(), 1.));
  static_assert(detail::near(detail::lattice_tensor(0), 0.) && detail::near(detail::lattice_tensor(1), 0.));
  static_assert(detail::near(detail::lattice_tensor(0, 0), cs2) && detail::near(detail::lattice_tensor(1, 1), cs2) &&
                detail::near(detail::lattice_tensor(0, 1), 0.));
  static_assert(detail::near(detail::lattice_tensor(0, 0, 0), 0.) && detail::near(detail::lattice_tensor(0, 1, 1), 0.));
  inline constexpr double cs4 = cs2 * cs2;
  static_assert(detail::near(detail::lattice_tensor(0, 0, 0, 0), 3 * cs4) &&
                detail::near(detail::lattice_tensor(0, 0, 1, 1), cs4) &&
                detail::near(detail::lattice_tensor(0, 0, 0, 1), 0.));
  static_assert([] {
    const Moments m = moments(equilibrium({1.1, 0.05, -0.02}));
    return detail::near(m.rho, 1.1) && detail::near(m.ux, 0.05) && detail::near(m.uy, -0.02);
  }());
} // namespace lbfem::D2Q9
