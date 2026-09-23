// Empirical roofline for one core: scalar/vector FMA peak, STREAM-like traffic
// and the collision access pattern (9 SoA arrays in, 9 out) at the working-set
// sizes of the cgdbe runs.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
using clk = std::chrono::steady_clock;
static double
now()
{
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

__attribute__((noinline)) double
fma_peak(long iters, double x)
{
  // 8 independent chains hide the FMA latency (4 cycles); the compiler keeps
  // these scalar because each chain is a loop-carried dependency.
  double       a0 = 1, a1 = 1.1, a2 = 1.2, a3 = 1.3, a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
  const double y = 0.999999;
  for (long i = 0; i < iters; ++i)
    {
      a0 = a0 * y + x;
      a1 = a1 * y + x;
      a2 = a2 * y + x;
      a3 = a3 * y + x;
      a4 = a4 * y + x;
      a5 = a5 * y + x;
      a6 = a6 * y + x;
      a7 = a7 * y + x;
    }
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

int
main()
{
  {
    const long it = 200'000'000;
    double     t  = now();
    double     r  = fma_peak(it, 1e-9);
    t             = now() - t;
    printf("scalar FMA peak (8 chains):          %6.2f GFlop/s   (%g)\n", 16.0 * it / t / 1e9, r);
  }
  for (long n : {66049L, 4'000'000L})
    {
      const int                        Q = 9;
      std::vector<std::vector<double>> f(Q, std::vector<double>(n, 1.0)), g(Q, std::vector<double>(n, 0.0));
      double                          *F[Q], *G[Q];
      for (int a = 0; a < Q; ++a)
        {
          F[a] = f[a].data();
          G[a] = g[a].data();
        }
      const int    reps = n < 1'000'000 ? 200 : 5;
      const double MB   = n * 8.0 / 1e6;
      // (a) STREAM-like: 9 independent scale-and-add passes (out = 1.0001 in + 1e-3)
      double best = 1e9;
      for (int r = 0; r < reps; ++r)
        {
          double t = now();
          for (int a = 0; a < Q; ++a)
            for (long i = 0; i < n; ++i)
              G[a][i] = 1.0001 * F[a][i] + 1e-3;
          best = std::min(best, now() - t);
        }
      printf("n=%-8ld (%5.1f MB/array) stream 9x(in->out):          %6.2f GB/s  %5.2f GFlop/s\n",
             n,
             MB,
             18.0 * n * 8 / best / 1e9,
             2.0 * Q * n / best / 1e9);
      // (b) collision pattern: per node gather 9, rho = sum, write 9 (no feq)
      best = 1e9;
      for (int r = 0; r < reps; ++r)
        {
          double t = now();
          for (long i = 0; i < n; ++i)
            {
              double v[Q], rho = 0;
              for (int a = 0; a < Q; ++a)
                {
                  v[a] = F[a][i];
                  rho += v[a];
                }
              const double s = 1.0 / rho;
              for (int a = 0; a < Q; ++a)
                G[a][i] = v[a] * s;
            }
          best = std::min(best, now() - t);
        }
      printf("n=%-8ld (%5.1f MB/array) collision pattern 9 in, 9 out: %6.2f GB/s  %5.2f GFlop/s (19 flops/node)\n",
             n,
             MB,
             18.0 * n * 8 / best / 1e9,
             19.0 * n / best / 1e9);
      // (c) same, in place (the bardow collision): read 9, write 9 to the same arrays
      best = 1e9;
      for (int r = 0; r < reps; ++r)
        {
          double t = now();
          for (long i = 0; i < n; ++i)
            {
              double v[Q], rho = 0;
              for (int a = 0; a < Q; ++a)
                {
                  v[a] = F[a][i];
                  rho += v[a];
                }
              const double s = 1.0 + 1e-12 / rho;
              for (int a = 0; a < Q; ++a)
                F[a][i] = v[a] * s;
            }
          best = std::min(best, now() - t);
        }
      printf("n=%-8ld (%5.1f MB/array) collision pattern in place:     %6.2f GB/s\n", n, MB, 18.0 * n * 8 / best / 1e9);
    }
  return 0;
}
