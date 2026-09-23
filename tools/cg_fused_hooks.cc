// Times the two fused-CG vector-update hooks of dealii::SolverCG in isolation,
// called on 64-entry chunks (as MatrixFree::cell_loop does), for a
// DiagonalMatrix preconditioner (per-entry apply() path) and for a wrapper that
// offers apply_to_subrange() only.
#include <deal.II/base/mpi.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
using namespace dealii;
using VT = LinearAlgebra::distributed::Vector<double>;

struct Identity // never applied: only the type matters for IterationWorker
{
  void vmult(VT &d, const VT &s) const { d = s; }
  void vmult(VT &, const VT &, const std::function<void(unsigned, unsigned)> &, const std::function<void(unsigned, unsigned)> &) const {}
};
struct RangeOnly
{
  const DiagonalMatrix<VT> &D;
  void vmult(VT &d, const VT &s) const { D.vmult(d, s); }
  void apply_to_subrange(unsigned b, unsigned e, const double *s, double *d) const { D.apply_to_subrange(b, e, s, d); }
};

template <typename P>
void run(const char *name, const P &prec, const unsigned n)
{
  VT x(n), b(n);
  std::mt19937 g(1); std::uniform_real_distribution<double> u(0.5, 1.5);
  for (auto &v : b) v = u(g);
  GrowingVectorMemory<VT> mem;
  Identity A;
#if DEAL_II_VERSION_GTE(9, 6, 0)
  internal::SolverCG::IterationWorker<VT, Identity, P> w(A, prec, false, mem, x, b, true);
  w.startup();
#else
  internal::SolverCG::IterationWorker<VT, Identity, P> w(A, prec, false, mem, x);
  w.startup(b);
#endif
  w.alpha = 0.3; w.beta = 0.2; w.previous_alpha = 0.25; w.previous_beta = 0.15;
  for (auto &v : w.v) v = u(g);
  const unsigned chunk = 64;
  auto time = [&](auto &&f) {
    double t = 1e30;
    for (int r = 0; r < 20; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      f();
      t = std::min(t, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return 1e9 * t / n;
  };
  std::array<VectorizedArray<double>, 7> sums{};
  const double t_before_odd = time([&] { for (unsigned s = 0; s < n; s += chunk) w.operation_before_loop(3, s, std::min(n, s + chunk)); });
  const double t_before_even = time([&] { for (unsigned s = 0; s < n; s += chunk) w.operation_before_loop(2, s, std::min(n, s + chunk)); });
  const double t_after = time([&] { for (unsigned s = 0; s < n; s += chunk) w.operation_after_loop(s, std::min(n, s + chunk), sums); });
  std::printf("  %-26s before (odd it) %5.2f  before (even it) %5.2f  after %5.2f  ns per entry\n", name, t_before_odd, t_before_even, t_after);
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  std::printf("deal.II %s, VectorizedArray<double> width %u\n", DEAL_II_PACKAGE_VERSION, static_cast<unsigned int>(VectorizedArray<double>::size()));
  for (const unsigned n : {4096u, 262144u, 4194304u})
    {
      VT d(n);
      for (auto &v : d) v = 0.7;
      const DiagonalMatrix<VT> D(d);
      std::printf("n = %u\n", n);
      run("DiagonalMatrix (apply)", D, n);
      run("apply_to_subrange only", RangeOnly{D}, n);
    }
}
