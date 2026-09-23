// "Speed of light" for the Q1 lumped-mass Lax-Wendroff streaming on a periodic
// uniform grid: the assembled 9-point stencil (-dt C - dt^2/2 K)/h^2 applied
// direction by direction, structure-of-arrays, no FEM machinery.
#include <chrono>
#include <cstdio>
#include <vector>
int
main(int argc, char **argv)
{
  const int                        N = argc > 1 ? atoi(argv[1]) : 256, steps = 50;
  const double                     h = 1.0 / N, dt = 0.25 * h;
  const int                        ex[9] = {0, 1, 1, 0, -1, -1, -1, 0, 1}, ey[9] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
  std::vector<std::vector<double>> f(9, std::vector<double>(N * N, 1.0)), r(9, std::vector<double>(N * N));
  for (int a = 0; a < 9; ++a)
    for (int i = 0; i < N * N; ++i)
      f[a][i] = 1.0 + 1e-3 * ((i * 7919) % 13);
  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s)
    for (int a = 1; a < 9; ++a)
      {
        const double cx = ex[a], cy = ey[a];
        // element stencils (see fem_stencils_q1): C = Mass_y (x) Diff_x etc., K analogous
        double       C[3][3], K[3][3], w[3][3];
        const double m[3] = {1. / 6, 4. / 6, 1. / 6}, d[3] = {-0.5, 0., 0.5}, k[3] = {-1., 2., -1.};
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            {
              C[i][j] = cx * d[i] * m[j] + cy * m[i] * d[j]; // /h * h^2 -> h
              K[i][j] = cx * cx * k[i] * m[j] + cy * cy * m[i] * k[j] + cx * cy * d[i] * d[j] * 2.;
              w[i][j] = (-dt * h * C[i][j] - 0.5 * dt * dt * K[i][j]) / (h * h); // lumped mass M_L = h^2
            }
        const double *F = f[a].data();
        double       *R = r[a].data();
        for (int j = 0; j < N; ++j)
          {
            const int jm = (j + N - 1) % N * N, j0 = j * N, jp = (j + 1) % N * N;
            for (int i = 0; i < N; ++i)
              {
                const int im = (i + N - 1) % N, ip = (i + 1) % N;
                R[j0 + i] = w[0][0] * F[jm + im] + w[1][0] * F[jm + i] + w[2][0] * F[jm + ip] + w[0][1] * F[j0 + im] +
                            w[1][1] * F[j0 + i] + w[2][1] * F[j0 + ip] + w[0][2] * F[jp + im] + w[1][2] * F[jp + i] +
                            w[2][2] * F[jp + ip];
              }
          }
      }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double       chk = 0;
  for (int a = 1; a < 9; ++a)
    chk += r[a][N + 1];
  printf(
    "N=%d: %.3f s for %d steps, %.3f ms/step, %.1f MNUPS, %.2f GB/s (16 vector passes/step), %.2f GFlop/s (17 flops per node and direction), %.0f ns/cell/direction (chk %g)\n",
    N,
    sec,
    steps,
    1e3 * sec / steps,
    1e-6 * N * N * steps / sec,
    16.0 * 8 * N * N * steps / sec / 1e9,
    17.0 * 8 * N * N * steps / sec / 1e9,
    1e9 * sec / (steps * 8.0 * N * N),
    chk);
}
