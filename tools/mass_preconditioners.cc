// Spectrum and convergence of the mass-matrix solves of the streaming step,
// A x = r with A = M or M + dt^2/6 K_e (TG3), under different CG
// preconditioners. Built on lbfem (lbfem_add_driver); single process.
//
//   mass_preconditioners [refinements...]      (default 5 7 9)
//
// For each mesh and operator: CG iterations to |r|/|r0| < 1e-8 from x = 0 for a
// random right-hand side (all modes) and for the advection right-hand side of
// a running simulation, the extreme Ritz values of the preconditioned operator
// (the Lanczos estimates CG yields for free), and the time per solve.
#include <deal.II/base/timer.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <lbfem/advection.h>
#include <lbfem/discretization.h>
#include <lbfem/mass.h>
#include <lbfem/schemes.h>
#include <lbfem/test_cases.h>
#include <lbfem/timers.h>

#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace dealii;
using namespace lbfem;

constexpr int p = LBFEM_DEGREE;
using Disc      = Discretization<p>;
using TG3LHS    = TG3Operator<p>;
using Jacobi    = DiagonalMatrix<VectorType>;

// Counts the applications of an operator. Chebyshev keeps a pointer to it
// (hence Subscriptor) and checks its size (m(); el() is never called, since it
// gets the Jacobi preconditioner).
template <typename Operator>
struct Counted : Subscriptor
{
  Counted(const Operator &A, const types::global_dof_index size)
    : A(A)
    , size(size)
  {}
  types::global_dof_index
  m() const
  {
    return size;
  }
  double
  el(const types::global_dof_index, const types::global_dof_index) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0.;
  }
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    ++n;
    A.vmult(dst, src);
  }
  const Operator               &A;
  const types::global_dof_index size;
  mutable unsigned int          n = 0;
};

struct Outcome
{
  unsigned int its, applications;
  double       lambda_min, lambda_max, seconds;
};

// CG from x = 0; the Ritz values of the last Lanczos tridiagonal matrix.
template <typename Operator, typename Preconditioner>
Outcome
solve(const Counted<Operator> &A, const Preconditioner &P, const VectorType &b)
{
  A.n = 0;
  VectorType x(b);
  x = 0.;
  SolverControl        control(2000, 1e-8 * b.l2_norm());
  SolverCG<VectorType> cg(control);
  double               lmin = 0., lmax = 0.;
  cg.connect_eigenvalues_slot([&](const std::vector<double> &ev) {
    if (!ev.empty())
      lmin = ev.front(), lmax = ev.back();
  });
  Timer timer;
  cg.solve(A, x, b, P);
  return {control.last_step(), A.n, lmin, lmax, timer.wall_time()};
}

// Chebyshev polynomial of degree k in Jacobi-preconditioned A over the whole
// spectrum, as a fixed linear preconditioner. Its eigenvalue estimate (a CG
// run) is done here, so that the timed solves do not include it.
template <typename Operator>
auto
chebyshev(const Operator &A, const Disc &disc, const unsigned int k, const VectorType &b)
{
  using Cheb = PreconditionChebyshev<Operator, VectorType, Jacobi>;
  typename Cheb::AdditionalData data;
  data.degree              = k;
  data.smoothing_range     = 12.; // lambda_min = lambda_max / 12, below the spectrum
  data.eig_cg_n_iterations = 30;
  data.preconditioner      = disc.mass.get_matrix_diagonal_inverse();
  auto P                   = std::make_shared<Cheb>();
  P->initialize(A, data);
  P->estimate_eigenvalues(b);
  return P;
}

void
study(const std::string &label, TestCase &tc, const unsigned int refinements)
{
  Disc disc(MPI_COMM_WORLD);
  disc.reinit({.refinements = refinements}, tc);
  const Number dt = disc.time_step(1.); // CFL 1 (TG3's limit)

  // --- right-hand sides: random, and the advection of a running simulation
  VectorType random;
  disc.matrix_free->initialize_dof_vector(random);
  std::mt19937                           gen(42);
  std::uniform_real_distribution<double> uni(-1., 1.);
  for (auto &v : random)
    v = uni(gen);
  disc.constraints.set_zero(random);

  TimerOutput timer_output(MPI_COMM_WORLD, std::cout, TimerOutput::never, TimerOutput::wall_times);
  StageTimers timers(timer_output);
  Walls       walls;
  tc.set_mach(0.1);
  tc.set_reynolds(400.);
  walls.reinit(disc.node, tc);
  Bardow<p> scheme(disc, walls, {.streaming = Streaming::tg2, .mass = {.type = Mass::lumped}}, timers);
  scheme.set_time_step({.dt = disc.time_step(0.4), .lambda = tc.relaxation_time()});
  scheme.set_initial_populations(tc);
  for (unsigned int n = 0; n < 50; ++n)
    scheme.step();
  BlockVectorType rhs;
  disc.initialize(rhs, n_moving);
  scheme.streaming.advection_operator().apply(rhs, {.f = scheme.f}, D2Q9::e, {.dt = dt, .lambda = tc.relaxation_time()});
  const VectorType &advection = rhs.block(1); // population 2, e = (1, 1)

  const TG3LHS tg3(*disc.matrix_free, dt * dt / 6., {{1., 1.}});

  // the diagonal of the TG3 operator itself (the solver preconditions with diag(M))
  VectorType tg3_diagonal;
  tg3.compute_diagonal(tg3_diagonal);
  for (auto &d : tg3_diagonal)
    d = d > 0. ? 1. / d : 1.;
  const Jacobi tg3_jacobi(tg3_diagonal);

  std::printf("\n%s, Q%d, %u^2 cells\n", label.c_str(), p, 1u << refinements);
  std::printf("  %-4s %-19s %6s %6s %6s %6s %10s %10s %7s %9s %9s\n", "A", "preconditioner", "its", "A-app",
              "its", "A-app", "lambda_min", "lambda_max", "kappa", "ms rand", "ms adv");
  std::printf("  %-4s %-19s %13s %13s\n", "", "", "random rhs", "advection");
  const auto row = [&](const char *op, const std::string &name, const auto &A, const auto &P) {
    const auto r = solve(A, P, random), a = solve(A, P, advection);
    std::printf("  %-4s %-19s %6u %6u %6u %6u %10.4f %10.4f %7.2f %9.3f %9.3f\n", op, name.c_str(), r.its,
                r.applications, a.its, a.applications, r.lambda_min, r.lambda_max, r.lambda_max / r.lambda_min,
                1e3 * r.seconds, 1e3 * a.seconds);
  };
  const auto &jacobi = *disc.mass.get_matrix_diagonal_inverse();
  const auto &lumped = *disc.mass.get_matrix_lumped_diagonal_inverse();

  const auto all = [&](const char *op, const auto &A) {
    const Counted counted(A, disc.dof_handler.n_dofs());
    row(op, "none", counted, PreconditionIdentity());
    row(op, "Jacobi", counted, jacobi);
    row(op, "lumped mass", counted, lumped);
    if constexpr (std::is_same_v<std::decay_t<decltype(A)>, TG3LHS>)
      row(op, "Jacobi, own diagonal", counted, tg3_jacobi);
    for (const unsigned int k : {2u, 3u, 5u})
      {
        const auto P = chebyshev(counted, disc, k, random);
        row(op, "Chebyshev(J) k=" + std::to_string(k), counted, *P);
      }
  };
  all("M", disc.mass);
  all("TG3", tg3);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  std::vector<unsigned int>        refinements;
  for (int i = 1; i < argc; ++i)
    refinements.push_back(std::stoul(argv[i]));
  if (refinements.empty())
    refinements = {5, 7, 9};

  for (const unsigned int r : refinements)
    {
      TaylorGreenVortex uniform(1, 1, 0.), distorted(1, 1, 0.1);
      LidDrivenCavity   stretched(1.2), strongly_stretched(2.5);
      study("Taylor-Green, uniform", uniform, r);
      study("Taylor-Green, distorted (eps = 0.1)", distorted, r);
      study("cavity, stretched (gamma = 1.2)", stretched, r);
      study("cavity, stretched (gamma = 2.5)", strongly_stretched, r);
    }
}
