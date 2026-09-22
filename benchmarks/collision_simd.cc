// BGK collision (Bardow scheme, lbfem::collide_bgk) on structure-of-arrays
// populations, two ways:
//   nodewise: one node at a time through 9-element arrays, a branch for wall
//             nodes (the form of lbfem::collide_bgk);
//   simd:     a C/Fortran-style loop over nodes with the wall treatment as a
//             blend and "omp simd", which the compiler vectorizes across nodes.
// Checks that both give the same populations (to rounding) and reports
// ns per node.
//   g++ -O3 -march=native -fopenmp-simd -std=c++20 -Iinclude benchmarks/collision_simd.cc
#include <lbfem/d2q9.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace lbfem;
constexpr unsigned int Q = D2Q9::Q;

struct Populations
{
  std::array<std::vector<double>, Q> g;
  explicit Populations(const std::size_t n)
  {
    for (unsigned int a = 0; a < Q; ++a)
      g[a].resize(n);
  }
};

// As lbfem::collide_bgk.
void
collide_nodewise(Populations &p, const std::vector<int> &wall, const double uw[2][2], const double omega)
{
  const std::size_t n = p.g[0].size();
  std::array<double *, Q> G;
  for (unsigned int a = 0; a < Q; ++a)
    G[a] = p.g[a].data();
  for (std::size_t i = 0; i < n; ++i)
    {
      std::array<double, Q> gi;
      for (unsigned int a = 0; a < Q; ++a)
        gi[a] = G[a][i];
      const auto geq = D2Q9::equilibrium(D2Q9::moments(gi));
      if (wall[i] < 0)
        for (unsigned int a = 0; a < Q; ++a)
          G[a][i] = gi[a] - omega * (gi[a] - geq[a]);
      else
        {
          auto m = D2Q9::moments(gi);
          m.ux   = uw[wall[i]][0];
          m.uy   = uw[wall[i]][1];
          const auto geq_wall = D2Q9::equilibrium(m);
          for (unsigned int a = 0; a < Q; ++a)
            G[a][i] = geq_wall[a] + (1. - omega) * (gi[a] - geq[a]);
        }
    }
}

// Loop over nodes: the moments, then the equilibrium of the fluid velocity
// and of the velocity the relaxation targets (the wall velocity on wall
// nodes), blended per node instead of branched.
void
collide_simd(Populations &p, const std::vector<int> &wall, const double uw[2][2], const double omega)
{
  const std::size_t n = p.g[0].size();
  double *const     G[Q] = {p.g[0].data(), p.g[1].data(), p.g[2].data(), p.g[3].data(), p.g[4].data(),
                            p.g[5].data(), p.g[6].data(), p.g[7].data(), p.g[8].data()};
  const int *const  W    = wall.data();
#pragma omp simd
  for (std::size_t i = 0; i < n; ++i)
    {
      double gi[Q], rho = 0., mx = 0., my = 0.;
      for (unsigned int a = 0; a < Q; ++a)
        {
          gi[a] = G[a][i];
          rho += gi[a];
          mx += D2Q9::e[a][0] * gi[a];
          my += D2Q9::e[a][1] * gi[a];
        }
      const double ux = mx / rho, uy = my / rho;
      const bool   on_wall = W[i] >= 0;
      const double vx = on_wall ? uw[W[i] > 0][0] : ux, vy = on_wall ? uw[W[i] > 0][1] : uy;
      const double uu = ux * ux + uy * uy, vv = vx * vx + vy * vy;
      for (unsigned int a = 0; a < Q; ++a)
        {
          const double eu = D2Q9::e[a][0] * ux + D2Q9::e[a][1] * uy;
          const double ev = D2Q9::e[a][0] * vx + D2Q9::e[a][1] * vy;
          const double geq = D2Q9::w[a] * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * uu);
          const double gtarget = D2Q9::w[a] * rho * (1. + 3. * ev + 4.5 * ev * ev - 1.5 * vv);
          G[a][i] = on_wall ? gtarget + (1. - omega) * (gi[a] - geq) : gi[a] - omega * (gi[a] - geq);
        }
    }
}

int
main(int argc, char **argv)
{
  const std::size_t n    = argc > 1 ? std::atol(argv[1]) : 16641;
  const int         reps = argc > 2 ? std::atoi(argv[2]) : 200;
  const double      uw[2][2] = {{0., 0.}, {0.0577, 0.}}, omega = 1.6;

  Populations       init(n);
  std::vector<int>  wall(n);
  const std::size_t side = std::lround(std::sqrt(double(n)));
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t x = i % side, y = i / side; // walls on the border of a square, lid on top
      wall[i] = (y + 1 == side) ? 1 : (x == 0 || x + 1 == side || y == 0) ? 0 : -1;
      const double ux = 0.03 * std::sin(0.1 * x), uy = 0.02 * std::cos(0.07 * y), rho = 1. + 1e-3 * std::sin(0.3 * i);
      const auto   feq = D2Q9::equilibrium({rho, ux, uy});
      for (unsigned int a = 0; a < Q; ++a)
        init.g[a][i] = feq[a] * (1. + 1e-3 * std::cos(0.37 * i + a));
    }

  const auto time = [&](auto &&collide, Populations &p) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r)
      {
        p               = init;
        const auto t0   = std::chrono::steady_clock::now();
        collide(p, wall, uw, omega);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
    return best;
  };
  Populations a(n), b(n);
  const double t_node = time(collide_nodewise, a), t_simd = time(collide_simd, b);

  double diff = 0.;
  for (unsigned int q = 0; q < Q; ++q)
    for (std::size_t i = 0; i < n; ++i)
      diff = std::max(diff, std::abs(a.g[q][i] - b.g[q][i]));
  const double MB = 2. * Q * n * sizeof(double) / 1e6; // read and write every population once
  std::printf("n = %zu nodes (%.1f MB traffic per sweep), max |difference| = %.1e\n", n, MB, diff);
  std::printf("  nodewise : %6.2f ns/node  %6.2f GB/s\n", 1e9 * t_node / n, MB / 1e3 / t_node);
  std::printf("  simd     : %6.2f ns/node  %6.2f GB/s   (x%.2f)\n", 1e9 * t_simd / n, MB / 1e3 / t_simd, t_node / t_simd);
}
