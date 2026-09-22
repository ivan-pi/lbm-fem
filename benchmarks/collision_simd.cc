// BGK collision of the Bardow scheme on the populations of n nodes (a block
// vector, one block per direction), two ways:
//   library: lbfem::collide_bgk, a lambda mapped over the nodes by nodal_map
//            (vectorized across nodes by the compiler);
//   simd:    a hand-written C/Fortran-style loop over nodes with "omp simd",
//            the reference for what vectorization can give.
// Checks that both give the same populations (to rounding) and reports ns per
// node. Built with -DLBFEM_BUILD_EXTRAS=ON;  collision_simd [n] [repetitions]
// (the working set is 2 x 9 x 8 n bytes: in cache for small n).
#include <deal.II/base/mpi.h>

#include <lbfem/collision.h>
#include <lbfem/d2q9.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace lbfem;

// Moments, then the equilibria of the fluid velocity and of the velocity the
// relaxation targets (the wall velocity on wall nodes), blended per node
// instead of branched.
void
collide_simd(BlockVectorType &g, const Walls &walls, const double omega)
{
  const std::size_t n = g.block(0).locally_owned_size();
  double           *G[Q];
  for (unsigned int a = 0; a < Q; ++a)
    G[a] = g.block(a).begin();
  const int *const W = walls.of_node.data();
  const auto      *V = walls.velocity.data();
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
      const int    w       = on_wall ? W[i] : 0;
      const double vx = on_wall ? V[w][0] : ux, vy = on_wall ? V[w][1] : uy;
      const double uu = ux * ux + uy * uy, vv = vx * vx + vy * vy;
      for (unsigned int a = 0; a < Q; ++a)
        {
          const double eu      = D2Q9::e[a][0] * ux + D2Q9::e[a][1] * uy;
          const double ev      = D2Q9::e[a][0] * vx + D2Q9::e[a][1] * vy;
          const double geq     = D2Q9::w[a] * rho * (1. + 3. * eu + 4.5 * eu * eu - 1.5 * uu);
          const double gtarget = D2Q9::w[a] * rho * (1. + 3. * ev + 4.5 * ev * ev - 1.5 * vv);
          G[a][i] = on_wall ? gtarget + (1. - omega) * (gi[a] - geq) : gi[a] - omega * (gi[a] - geq);
        }
    }
}

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const std::size_t n     = argc > 1 ? std::atol(argv[1]) : 16641;
  const int         reps  = argc > 2 ? std::atoi(argv[2]) : 200;
  const double      omega = 1.6;

  // a square of nodes with walls on its border and a moving lid on top
  std::vector<int>  wall_of_node(n);
  BlockVectorType   init(Q, n);
  const std::size_t side = std::lround(std::sqrt(double(n)));
  for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t x = i % side, y = i / side;
      wall_of_node[i]     = (y + 1 == side) ? 1 : (x == 0 || x + 1 == side || y == 0) ? 0 : -1;
      const double ux = 0.03 * std::sin(0.1 * x), uy = 0.02 * std::cos(0.07 * y), rho = 1. + 1e-3 * std::sin(0.3 * i);
      const auto   feq = D2Q9::equilibrium({rho, ux, uy});
      for (unsigned int a = 0; a < Q; ++a)
        init.block(a)[i] = feq[a] * (1. + 1e-3 * std::cos(0.37 * i + a));
    }
  Walls walls;
  walls.set(std::move(wall_of_node), {{{0., 0.}}, {{0.0577, 0.}}});

  const auto time = [&](auto &&collide, BlockVectorType &g) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r)
      {
        g             = init;
        const auto t0 = std::chrono::steady_clock::now();
        collide(g, walls, omega);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
    return best;
  };
  BlockVectorType a(init), b(init);
  const double    t_lib = time(collide_bgk, a), t_simd = time(collide_simd, b);

  double diff = 0.;
  for (unsigned int q = 0; q < Q; ++q)
    for (std::size_t i = 0; i < n; ++i)
      diff = std::max(diff, std::abs(a.block(q)[i] - b.block(q)[i]));
  const double MB = 2. * Q * n * sizeof(double) / 1e6; // read and write every population once
  std::printf("n = %zu nodes (%.1f MB traffic per sweep), max |difference| = %.1e\n", n, MB, diff);
  std::printf("  library : %6.2f ns/node  %6.2f GB/s\n", 1e9 * t_lib / n, MB / 1e3 / t_lib);
  std::printf("  simd    : %6.2f ns/node  %6.2f GB/s   (x%.2f)\n", 1e9 * t_simd / n, MB / 1e3 / t_simd, t_lib / t_simd);
}
