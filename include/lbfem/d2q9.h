// D2Q9 lattice: discrete velocities, weights, moments and the second-order
// equilibrium, in the numbering of Fig. 1 / Eq. (3) of Lee & Lin (2001).
// Lattice units: |e_x| = 1, c_s^2 = 1/3. No deal.II dependency.
#pragma once

#include <array>

namespace lbfem::D2Q9
{
  inline constexpr unsigned int Q   = 9;
  inline constexpr double       cs2 = 1. / 3.;

  inline constexpr std::array<std::array<double, 2>, Q> e = {{{{0, 0}},
                                                              {{1, 0}},
                                                              {{1, 1}},
                                                              {{0, 1}},
                                                              {{-1, 1}},
                                                              {{-1, 0}},
                                                              {{-1, -1}},
                                                              {{0, -1}},
                                                              {{1, -1}}}};

  inline constexpr std::array<double, Q> w = {{4. / 9.,
                                               1. / 9.,
                                               1. / 36.,
                                               1. / 9.,
                                               1. / 36.,
                                               1. / 9.,
                                               1. / 36.,
                                               1. / 9.,
                                               1. / 36.}};

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
} // namespace lbfem::D2Q9
