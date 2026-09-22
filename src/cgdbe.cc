// -----------------------------------------------------------------------------
// cgdbe -- Characteristic Galerkin schemes for the Discrete Boltzmann Equation
//
//   T. Lee, C.-L. Lin, J. Comput. Phys. 171, 336 (2001)
//   A. Bardow, I. V. Karlin, A. A. Gusev, EPL 75, 434 (2006)
//
// Matrix-free deal.II implementation (D2Q9, Q_p elements). Test cases:
//   --case tgv      Taylor-Green vortex, periodic box, wave numbers --modes n1,n2
//   --case couette  start-up Couette flow, periodic in x (Lee & Lin, Sec. 3.1)
//   --case cavity   lid-driven cavity on a wall-clustered mesh (Sec. 3.2)
//
// Two schemes, with lambda = nu / c_s^2 the relaxation time in both:
//
//   --scheme leelin   predictor-corrector of Lee & Lin, for alpha = 1..8:
//     (1 + tau) M fhat = M (f + tau feq) - dt C f - dt^2 [D f + Q (f - feq)],
//     f^{n+1} = fhat + tau (feq(fhat) - feq(f^n)),   tau = dt/lambda
//
//   --scheme bardow   collide-then-stream in the transformed populations
//     g = f + dt/(2 lambda) (f - feq):
//     g* = g - omega (g - geq),  omega = dt / (lambda + dt/2)    (exact BGK)
//     M (g^{n+1} - g*) = -dt C g* - dt^2 D g*      (weak form of their Eq. 9)
//
// with the element matrices (Lee & Lin 19-23, Bardow 13), never assembled:
//   M = int N N^T,  C_e = int N (e.grad N^T),  D_e = K_e/2,
//   K_e = int (e.grad N)(e.grad N^T),  Q_e = -C_e/(2 lambda),
// plus the wall surface term of the integration by parts of D. The advection
// right-hand sides are two distinct kernels (advection_leelin, advection_bardow),
// each a single MatrixFree::loop over all 8 moving populations; the collision
// is nodal and shares the moment/equilibrium code. The rest population (e_0 = 0)
// needs neither loop nor solve.
//
// Options that change the streaming step (bardow):
//   --mass cg|lumped|richardson[k]   consistent (CG), lumped, or lumped with k
//                                    Richardson passes (k = 2 ~ consistent accuracy)
//   --streaming tg2|tg3|tg3-split    Lax-Wendroff; third-order Taylor-Galerkin
//                                    (M + dt^2/6 K_e on the left, stable to CFL 1);
//                                    TG3 as x and y sweeps (exact at CFL 1 on a
//                                    tensor-product grid)
// -----------------------------------------------------------------------------

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <memory>
#include <string_view>
#include <vector>

using namespace dealii;

#ifndef CGDBE_DEGREE
#  define CGDBE_DEGREE 1 // polynomial degree of the FE_Q elements (cmake -DCGDBE_DEGREE=p)
#endif

// ------------------------------- D2Q9 lattice --------------------------------

namespace D2Q9
{
  inline constexpr unsigned int Q   = 9;
  inline constexpr double       cs2 = 1. / 3.;

  // numbering of Fig. 1 / Eq. (3) of the paper
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
} // namespace D2Q9

// -------------------------------- parameters ---------------------------------

enum class Case
{
  tgv,
  couette,
  cavity
};

enum class Scheme
{
  leelin, // Lee & Lin 2001: predictor-corrector, advection of f with feq gradients
  bardow  // Bardow et al. 2006: BGK collision, then pure Taylor-Galerkin streaming
};

enum class Mass
{
  cg,        // consistent, CG with Jacobi preconditioner and extrapolated start
  lumped,    // row-sum lumped
  richardson // lumped plus k Richardson passes (Donea's iterated lumping)
};

enum class Streaming // bardow only
{
  tg2,      // Lax-Wendroff / second-order Taylor-Galerkin
  tg3,      // third-order Taylor-Galerkin: (M + dt^2/6 K_e) on the left
  tg3_split // TG3 as an x sweep followed by a y sweep (exact at CFL 1 on a lattice)
};

struct Parameters
{
  Case         test_case  = Case::tgv;
  Scheme       scheme     = Scheme::bardow;
  Mass         mass       = Mass::cg;
  unsigned int richardson = 2;              // passes for Mass::richardson
  Streaming    streaming  = Streaming::tg2;

  unsigned int refinements = 6;   // 2^n x 2^n elements ...
  unsigned int n_cells     = 0;   // ... or, if > 0, n_cells x n_cells elements
  double       mach        = 0.1; // U0 / c_s
  unsigned int modes[2]    = {1, 4};  // tgv: wave numbers k_i = 2 pi n_i / L
  std::vector<double> reynolds;       // U0 L / nu (default 100 | 10 | 400); cavity: a list
                                      //   "400,1000,..." continues from each steady state
  double       cfl        = 0.;    // dt |e_x| / h_min (default 0.25; cavity 0.4 | leelin 0.5)
  double       t_end      = 0.;    // in units of t_ref (default 1 | 1 | 100):
                                   //   tgv 1/(nu k^2), couette L^2/nu, cavity L/U0
  double       stretch    = 1.2;   // cavity: tanh wall clustering, 0 = uniform
  double       distort    = 0.;    // tgv: mesh distortion x += eps sin(2 pi x) sin(2 pi y)
  double       steady_tol = 1e-4;  // cavity: stop if mean |du|/U0 per t_ref < tol
  unsigned int max_steps  = 0;     // > 0: fixed number of steps (benchmark)
  double       cg_tolerance = 1e-8; // relative to |rhs|
  unsigned int n_diagnostic = 20;   // diagnostic lines over the run
  bool         output       = true; // write the .dat files
  std::string  checkpoint;          // write populations here (per rank) at every diagnostic
  std::string  restart;             // read initial populations from this checkpoint

  static Parameters
  parse(int argc, char **argv)
  {
    Parameters prm;
    for (int i = 1; i < argc; ++i)
      {
        const std::string_view key  = argv[i];
        const auto             next = [&]() -> std::string {
          AssertThrow(i + 1 < argc, ExcMessage("missing value for " + std::string(key)));
          return argv[++i];
        };
        const auto choose = [&](const std::string &value, const std::vector<std::string> &names) {
          for (unsigned int k = 0; k < names.size(); ++k)
            if (value == names[k])
              return k;
          AssertThrow(false, ExcMessage(std::string(key) + ": unknown value " + value));
          return 0u;
        };
        if (key == "--case")
          prm.test_case = static_cast<Case>(choose(next(), {"tgv", "couette", "cavity"}));
        else if (key == "--scheme")
          prm.scheme = static_cast<Scheme>(choose(next(), {"leelin", "bardow"}));
        else if (key == "--mass")
          {
            const auto v = next(); // cg | lumped | richardson[k]
            if (v.rfind("richardson", 0) == 0)
              {
                prm.mass = Mass::richardson;
                if (v.size() > 10)
                  prm.richardson = std::stoul(v.substr(10));
              }
            else
              prm.mass = static_cast<Mass>(choose(v, {"cg", "lumped"}));
          }
        else if (key == "--streaming")
          prm.streaming = static_cast<Streaming>(choose(next(), {"tg2", "tg3", "tg3-split"}));
        else if (key == "--refine")
          prm.refinements = std::stoul(next());
        else if (key == "--cells")
          prm.n_cells = std::stoul(next());
        else if (key == "--mach")
          prm.mach = std::stod(next());
        else if (key == "--modes")
          {
            const auto v = next();
            const auto c = v.find(',');
            AssertThrow(c != std::string::npos, ExcMessage("--modes n1,n2"));
            prm.modes[0] = std::stoul(v.substr(0, c));
            prm.modes[1] = std::stoul(v.substr(c + 1));
          }
        else if (key == "--reynolds")
          {
            std::stringstream list(next());
            for (std::string item; std::getline(list, item, ',');)
              prm.reynolds.push_back(std::stod(item));
          }
        else if (key == "--cfl")
          prm.cfl = std::stod(next());
        else if (key == "--tend")
          prm.t_end = std::stod(next());
        else if (key == "--steps")
          prm.max_steps = std::stoul(next());
        else if (key == "--stretch")
          prm.stretch = std::stod(next());
        else if (key == "--distort")
          prm.distort = std::stod(next());
        else if (key == "--steady-tol")
          prm.steady_tol = std::stod(next());
        else if (key == "--cg-tol")
          prm.cg_tolerance = std::stod(next());
        else if (key == "--checkpoint")
          prm.checkpoint = next();
        else if (key == "--restart")
          prm.restart = next();
        else if (key == "--no-output")
          prm.output = false;
        else
          AssertThrow(false,
                      ExcMessage("unknown option " + std::string(key) +
                                 "\noptions: --case tgv|couette|cavity --scheme leelin|bardow "
                                 "--mass cg|lumped|richardson[k] --streaming tg2|tg3|tg3-split "
                                 "--refine n | --cells N --mach Ma --modes n1,n2 --reynolds Re[,Re..] "
                                 "--cfl c --tend t/t_ref --steps n --stretch gamma --distort eps "
                                 "--steady-tol eps --cg-tol eps --checkpoint name --restart name "
                                 "--no-output"));
      }
    const bool cavity = prm.test_case == Case::cavity;
    AssertThrow(prm.streaming == Streaming::tg2 || (prm.scheme == Scheme::bardow && prm.mass == Mass::cg),
                ExcMessage("--streaming tg3 needs --scheme bardow and --mass cg"));
    if (prm.reynolds.empty())
      prm.reynolds = {prm.test_case == Case::tgv ? 100. : cavity ? 400. : 10.};
    if (prm.cfl <= 0.)
      prm.cfl = !cavity ? 0.25 : (prm.scheme == Scheme::leelin ? 0.5 : 0.4);
    if (prm.t_end <= 0.)
      prm.t_end = cavity ? 100. : 1.;
    return prm;
  }
};

// -------------------------------- test cases ---------------------------------
//
// Everything that distinguishes the three flows is behind this interface:
// domain periodicity and mesh transformation, wall nodes and their velocity,
// initial populations, the reference solution and time scale, and what the
// diagnostics and output should report. The solver itself is case-agnostic.

struct TestCase
{
  double L = 1., U0 = 0., nu = 0., rho0 = 1.; // set by the solver

  virtual ~TestCase() = default;
  virtual std::string
  name() const = 0;
  virtual std::string
  tag(const double Re) const = 0; // stem of the output files
  virtual unsigned int
  n_periodic() const = 0; // 0, 1 (x only) or 2 periodic directions
  virtual void
  transform_mesh(Triangulation<2> &, double &) const
  {}
  virtual int
  wall_of(const Point<2> &) const // -1 interior, 0 stationary wall, 1 moving lid
  {
    return -1;
  }
  virtual std::array<double, 2>
  wall_velocity(const int wall) const
  {
    return wall == 1 ? std::array<double, 2>{{U0, 0.}} : std::array<double, 2>{{0., 0.}};
  }
  virtual double
  reference_time() const = 0;
  virtual bool
  has_exact_solution() const = 0; // else: run to a steady state
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
  // initial populations; neq_scale multiplies the non-equilibrium part
  // (1 for f, 1 + dt/(2 lambda) for the transformed g of the bardow scheme)
  virtual std::array<double, D2Q9::Q>
  initial_populations(const Point<2> &, const double lambda, const double neq_scale) const = 0;
};



// Taylor-Green vortex with wave numbers k1, k2 (Bardow et al., Eqs. (14)-(16)):
//   u = -U0 cos(k1 x) sin(k2 y) F,  v = U0 (k1/k2) sin(k1 x) cos(k2 y) F,
//   p = p0 - rho0 U0^2/4 [cos(2 k1 x) + (k1/k2)^2 cos(2 k2 y)] F^2,
//   F = exp(-nu (k1^2 + k2^2) t).  k1 != k2 gives a non-zero shear u_y + v_x.
struct TaylorGreenVortex : TestCase
{
  double n1, n2, distort;

  TaylorGreenVortex(const unsigned int n1, const unsigned int n2, const double distort)
    : n1(n1)
    , n2(n2)
    , distort(distort)
  {}

  std::string
  name() const override
  {
    return "Taylor-Green vortex, periodic, modes (" + std::to_string(int(n1)) + "," +
           std::to_string(int(n2)) + ")";
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
  // smooth periodic distortion: general (non-parallelogram) quadrilaterals
  void
  transform_mesh(Triangulation<2> &tria, double &) const override
  {
    if (distort == 0.)
      return;
    GridTools::transform(
      [&](const Point<2> &p) {
        const double d = distort * L * std::sin(2 * std::numbers::pi * p[0] / L) *
                         std::sin(2 * std::numbers::pi * p[1] / L);
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
  bool
  has_exact_solution() const override
  {
    return true;
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
  initial_populations(const Point<2> &p, const double lambda, const double neq_scale) const override
  {
    const auto   m  = exact(p, 0.);
    auto         f  = D2Q9::equilibrium(m);
    const double x1 = k1() * p[0], y2 = k2() * p[1];
    const double ux = U0 * k1() * std::sin(x1) * std::sin(y2);                                // du/dx
    const double sh = U0 * (k1() * k1() - k2() * k2()) / k2() * std::cos(x1) * std::cos(y2); // du/dy + dv/dx
    for (unsigned int a = 0; a < D2Q9::Q; ++a)
      {
        const auto [ex, ey] = D2Q9::e[a];
        f[a] -= neq_scale * lambda / D2Q9::cs2 * D2Q9::w[a] * m.rho * ((ex * ex - ey * ey) * ux + ex * ey * sh);
      }
    return f;
  }
};



// Start-up Couette flow between y = 0 (at rest) and y = L (velocity U0),
// periodic in x; Eq. (26) of Lee & Lin.
struct CouetteFlow : TestCase
{
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
  int
  wall_of(const Point<2> &p) const override
  {
    return std::abs(p[1]) < 1e-12 * L ? 0 : std::abs(p[1] - L) < 1e-12 * L ? 1 : -1;
  }
  double
  reference_time() const override // diffusion time
  {
    return L * L / nu;
  }
  bool
  has_exact_solution() const override
  {
    return true;
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
  initial_populations(const Point<2> &, const double, const double) const override
  {
    return D2Q9::equilibrium({rho0, 0., 0.}); // fluid at rest
  }
};



// Lid-driven cavity (Lee & Lin, Sec. 3.2) on a tanh-clustered mesh, run to a
// steady state; a list of Reynolds numbers is a continuation.
struct LidDrivenCavity : TestCase
{
  double stretch;

  explicit LidDrivenCavity(const double stretch)
    : stretch(stretch)
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
  int
  wall_of(const Point<2> &p) const override
  {
    const auto on = [&](const double x, const double x0) { return std::abs(x - x0) < 1e-12 * L; };
    if (on(p[0], 0.) || on(p[0], L) || on(p[1], 0.))
      return 0;
    return on(p[1], L) ? 1 : -1;
  }
  double
  reference_time() const override // lid time
  {
    return L / U0;
  }
  bool
  has_exact_solution() const override
  {
    return false;
  }
  std::array<double, D2Q9::Q>
  initial_populations(const Point<2> &, const double, const double) const override
  {
    return D2Q9::equilibrium({rho0, 0., 0.});
  }
};



// Left-hand side of the third-order Taylor-Galerkin streaming step (Donea 1984)
//   (M + dt^2/6 K_e) (g^{n+1} - g*) = -dt C_e g* - dt^2/2 K_e g*,
// K_e = int (e.grad N)(e.grad N^T): the dt^3/6 (e.grad)^3 term of the Taylor
// series along the characteristic, with (e.grad)^3 g ~ -(e.grad)^2 dg/dt.
template <int dim, int fe_degree, int n_q_1d, typename Number>
class TG3Operator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  TG3Operator(const MatrixFree<dim, Number> &mf, const Number coefficient, const std::array<double, 2> e)
    : data(mf)
    , coef(coefficient)
    , e(e)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    data.cell_loop(&TG3Operator::local_apply, this, dst, src, /*zero dst*/ true);
  }

private:
  void
  local_apply(const MatrixFree<dim, Number> &,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, fe_degree, n_q_1d, 1, Number> phi(data);
    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto g  = phi.get_gradient(q);
            const auto eg = e[0] * g[0] + e[1] * g[1];
            Tensor<1, dim, VectorizedArray<Number>> flux;
            flux[0] = (coef * e[0]) * eg;
            flux[1] = (coef * e[1]) * eg;
            phi.submit_value(phi.get_value(q), q);
            phi.submit_gradient(flux, q);
          }
        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
  }

  const MatrixFree<dim, Number> &data;
  const Number                   coef;
  const std::array<double, 2>    e;
};

// --------------------------------- solver ------------------------------------

class CGDBE
{
public:
  static constexpr int          dim       = 2;
  static constexpr int          fe_degree = CGDBE_DEGREE; // 1: bilinear, as in the papers
  static constexpr int          n_q_1d    = fe_degree + 1;
  static constexpr unsigned int Q         = D2Q9::Q;
  static constexpr unsigned int n_moving  = Q - 1;

  using Number          = double;
  using VA              = VectorizedArray<Number>;
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;
  using Range           = std::pair<unsigned int, unsigned int>;
  using Direction       = std::array<double, 2>;
  using MassOperator    = MatrixFreeOperators::MassOperator<dim, fe_degree, n_q_1d, 1, VectorType>;
  using LaplaceOperator = MatrixFreeOperators::LaplaceOperator<dim, fe_degree, n_q_1d, 1, VectorType>;
  using TG3LHS          = TG3Operator<dim, fe_degree, n_q_1d, Number>;

  explicit CGDBE(const Parameters &prm);
  bool
  run(); // false if a steady-state search ran out of time

private:
  // --- setup
  void
  setup();
  void
  set_reynolds(const double Re);

  // --- collision (nodal). macroscopic() is shared: on wall nodes it returns
  // the prescribed velocity, which is how the boundary condition enters.
  D2Q9::Moments
  macroscopic(const unsigned int i, const std::array<Number, Q> &fi) const;
  void
  compute_equilibrium();  // leelin: feq <- feq(f^n)
  void
  predictor_corrector();  // leelin: Eqs. (17) nodal part and (18), f <- f^{n+1}
  void
  collide();              // bardow: g <- g - omega (g - geq)

  // --- advection operators (matrix-free, all 8 moving populations at once)
  void
  advection_leelin(const MatrixFree<dim, Number> &, BlockVectorType &, const BlockVectorType &, const Range &) const;
  void
  advection_bardow(const MatrixFree<dim, Number> &, BlockVectorType &, const BlockVectorType &, const Range &) const;
  void
  wall_faces(const MatrixFree<dim, Number> &, BlockVectorType &, const BlockVectorType &, const Range &) const;
  void
  no_inner_faces(const MatrixFree<dim, Number> &, BlockVectorType &, const BlockVectorType &, const Range &) const
  {}
  void
  assemble_rhs();

  // --- streaming
  void
  apply_mass_inverse(BlockVectorType &x); // x <- (M or M + dt^2/6 K_e)^{-1} rhs
  void
  stream();                                // bardow: one or two Taylor-Galerkin sweeps

  // --- time steps, diagnostics, I/O
  void
  step_leelin();
  void
  step_bardow();
  bool
  diagnostics(std::ostream *file);
  void
  checkpoint(const std::string &name, const bool write);
  void
  compute_stream_function(BlockVectorType &velocity, VectorType &psi);
  void
  write_gnuplot(const std::string &name);
  void
  print_summary(const double seconds) const;

  static std::array<Number *, Q>
  raw(BlockVectorType &v)
  {
    std::array<Number *, Q> p{};
    for (unsigned int b = 0; b < v.n_blocks(); ++b)
      p[b] = v.block(b).begin();
    return p;
  }

  static std::array<Number, Q>
  gather(const std::array<Number *, Q> &F, const unsigned int i)
  {
    std::array<Number, Q> fi;
    for (unsigned int a = 0; a < Q; ++a)
      fi[a] = F[a][i];
    return fi;
  }

  // Arithmetic per node (D2Q9, a multiply-add counts 2) for the summary; see
  // README. bardow collision: moments 20, equilibrium 102, relaxation 18,
  // increment 8. leelin: equilibrium 122, predictor 32, equilibrium of fhat
  // 122, corrector 27. Advection per cell and population (Q1, cartesian):
  // gradients 72, quadrature 24, integration 88. Mass vmult per cell 68.
  static constexpr double flops_collision_bardow = 148;
  static constexpr double flops_collision_leelin = 303;
  static constexpr double flops_advection_cell   = 184. * (fe_degree == 1 ? 1. : 3.5 * fe_degree);
  static constexpr double flops_mass_vmult       = 68;

  const Parameters          prm;
  std::unique_ptr<TestCase> tc;
  const bool                is_bardow, is_split, steady; // steady: run to a steady state
  const MPI_Comm   comm = MPI_COMM_WORLD;

  ConditionalOStream pcout;
  TimerOutput        timer;

  parallel::distributed::Triangulation<dim> triangulation;
  const FE_Q<dim>                           fe;
  const MappingQ1<dim>                      mapping;
  DoFHandler<dim>                           dof_handler;
  AffineConstraints<Number>                 constraints;      // periodicity, or none
  AffineConstraints<Number>                 constraints_wall; // psi = 0 on walls (cavity)
  std::shared_ptr<MatrixFree<dim, Number>>  matrix_free;
  MassOperator                              mass;

  BlockVectorType f;         // populations (f, or g for bardow)        (Q blocks)
  BlockVectorType feq;       // leelin: nodal equilibria of f^n          (Q blocks)
  BlockVectorType rhs;       // advection right-hand side r_alpha        (Q-1 blocks)
  BlockVectorType incr;      // (M^{-1} r)_alpha                         (Q-1 blocks)
  BlockVectorType incr_prev; // previous incr (CG warm start) / second sweep
  VectorType      tmp;       // Richardson passes

  std::vector<Point<dim>>            node;          // support point per owned dof
  VectorType                         node_weight;   // int phi_i (row sums of M)
  std::vector<unsigned int>          independent;   // owned dofs that are not periodic slaves
  std::vector<int>                   wall_of_node;  // -1: interior, else index into wall_velocity
  std::vector<std::array<Number, 2>> wall_velocity; // {0,0} and {U0,0}
  std::vector<std::array<Number, 2>> u_previous;    // for the steady-state residual
  double                             time_previous = 0.;

  double                   reynolds, lambda, dt, t_ref, time = 0.;
  std::array<Direction, Q> stream_e = D2Q9::e; // directions of the current sweep

  Timer       t_collision, t_advection, t_mass; // accumulated over all steps
  double      mass_vector_passes = 0, mass_flops_per_node = 0; // traffic / flop models
  unsigned    n_steps = 0, step_no = 0, total_steps = 0;
  std::size_t cg_iterations = 0;
};



CGDBE::CGDBE(const Parameters &prm)
  : prm(prm)
  , tc(prm.test_case == Case::tgv     ? std::unique_ptr<TestCase>(new TaylorGreenVortex(prm.modes[0], prm.modes[1], prm.distort)) :
       prm.test_case == Case::couette ? std::unique_ptr<TestCase>(new CouetteFlow()) :
                                        std::unique_ptr<TestCase>(new LidDrivenCavity(prm.stretch)))
  , is_bardow(prm.scheme == Scheme::bardow)
  , is_split(prm.streaming == Streaming::tg3_split)
  , steady(!tc->has_exact_solution())
  , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
  , timer(comm, pcout, TimerOutput::never, TimerOutput::wall_times)
  , triangulation(comm)
  , fe(fe_degree)
  , dof_handler(triangulation)
{
  t_collision.stop();
  t_advection.stop();
  t_mass.stop();
}

void
CGDBE::setup()
{
  TimerOutput::Scope t(timer, "setup");

  // --- unit box: periodic (tgv) or wall-clustered (cavity)
  const double L = 1.;
  if (prm.n_cells > 0)
    GridGenerator::subdivided_hyper_cube(triangulation, prm.n_cells, 0., L, /*colorize*/ true);
  else
    GridGenerator::hyper_cube(triangulation, 0., L, /*colorize*/ true);
  const unsigned int n_per_dir = prm.n_cells > 0 ? prm.n_cells : (1u << prm.refinements);
  const unsigned int n_periodic = tc->n_periodic();
  if (n_periodic > 0)
    {
      std::vector<
        GridTools::PeriodicFacePair<parallel::distributed::Triangulation<dim>::cell_iterator>>
        periodic_faces;
      for (unsigned int d = 0; d < n_periodic; ++d)
        GridTools::collect_periodic_faces(triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
      triangulation.add_periodicity(periodic_faces);
    }
  if (prm.n_cells == 0)
    triangulation.refine_global(prm.refinements);

  double h_min = L / n_per_dir;
  tc->L         = L;
  tc->transform_mesh(triangulation, h_min);

  dof_handler.distribute_dofs(fe);

  const IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
  for (auto *c : {&constraints, &constraints_wall})
#if DEAL_II_VERSION_GTE(9, 6, 0)
    c->reinit(dof_handler.locally_owned_dofs(), relevant);
#else
    c->reinit(relevant);
#endif
  for (unsigned int d = 0; d < n_periodic; ++d)
    DoFTools::make_periodicity_constraints(dof_handler, 2 * d, 2 * d + 1, d, constraints);
  if (steady) // stream function of the steady state
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints_wall);
  constraints.close();
  constraints_wall.close();

  // --- matrix-free data. DoF index 0: populations, 1: stream function.
  MatrixFree<dim, Number>::AdditionalData data;
  data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
  data.mapping_update_flags  = update_values | update_gradients | update_JxW_values;
  if (n_periodic < dim) // walls
    data.mapping_update_flags_boundary_faces =
      update_gradients | update_JxW_values | update_normal_vectors;

  matrix_free = std::make_shared<MatrixFree<dim, Number>>();
  matrix_free->reinit(mapping,
                      std::vector<const DoFHandler<dim> *>{&dof_handler, &dof_handler},
                      std::vector<const AffineConstraints<Number> *>{&constraints,
                                                                     &constraints_wall},
                      std::vector<Quadrature<1>>{QGauss<1>(n_q_1d)},
                      data);

  {
    using GT = internal::MatrixFreeFunctions::GeometryType;
    std::array<unsigned int, 4> n_type{};
    for (unsigned int c = 0; c < matrix_free->n_cell_batches(); ++c)
      ++n_type[matrix_free->get_mapping_info().get_cell_type(c)];
    pcout << "  cell batches : " << matrix_free->n_cell_batches() << " (cartesian " << n_type[GT::cartesian]
          << ", affine " << n_type[GT::affine] << ", general " << n_type[GT::general] << ")\n";
  }

  mass.initialize(matrix_free, {0});
  mass.compute_lumped_diagonal(); // Mass::lumped, Mass::richardson
  mass.compute_diagonal();        // Jacobi preconditioner for CG

  const auto init = [&](BlockVectorType &v, const unsigned int n_blocks) {
    v.reinit(n_blocks);
    for (unsigned int b = 0; b < n_blocks; ++b)
      matrix_free->initialize_dof_vector(v.block(b));
    v.collect_sizes();
  };
  VectorType ones;
  matrix_free->initialize_dof_vector(ones);
  matrix_free->initialize_dof_vector(node_weight);
  ones = 1.;
  mass.vmult(node_weight, ones);

  matrix_free->initialize_dof_vector(tmp);

  init(f, Q);
  init(rhs, n_moving);
  init(incr, n_moving);
  init(incr_prev, n_moving);
  if (!is_bardow)
    init(feq, Q);

  // --- physical parameters (lattice units: |e_x| = 1, c_s^2 = 1/3, rho0 = 1)
  tc->U0 = prm.mach * std::sqrt(D2Q9::cs2);
  dt     = prm.cfl * h_min / (fe_degree * fe_degree);
  set_reynolds(prm.reynolds.front());
  if (!steady && prm.max_steps == 0) // hit t_end exactly
    dt = prm.t_end * t_ref / n_steps;

  // --- nodal coordinates, wall nodes, initial condition
  const auto support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
  const auto &owned         = dof_handler.locally_owned_dofs();
  node.resize(owned.n_elements());
  wall_of_node.assign(node.size(), -1);
  wall_velocity = {tc->wall_velocity(0), tc->wall_velocity(1)};

  const auto F = raw(f);
  for (unsigned int i = 0; i < node.size(); ++i)
    {
      const auto global = owned.nth_index_in_set(i);
      const auto &p = node[i] = support_points.at(global);
      if (!constraints.is_constrained(global))
        independent.push_back(i);

      wall_of_node[i] = tc->wall_of(p);
      const auto f0   = tc->initial_populations(p, lambda, is_bardow ? 1. + 0.5 * dt / lambda : 1.);
      for (unsigned int a = 0; a < Q; ++a)
        F[a][i] = f0[a];
    }
  u_previous.assign(node.size(), {{0., 0.}});
  if (!prm.restart.empty())
    checkpoint(prm.restart, /*write*/ false);

  pcout << "CGDBE (Lee & Lin 2001), deal.II " << DEAL_II_PACKAGE_VERSION << ", "
        << Utilities::MPI::n_mpi_processes(comm) << " MPI rank(s), SIMD width " << VA::size()
        << "\n  case         : " << tc->name()
        << "\n  elements     : " << triangulation.n_global_active_cells() << "  (Q" << fe_degree
        << ", " << dof_handler.n_dofs() << " nodes x " << Q << " populations), h_min = " << h_min
        << "\n  scheme       : "
        << (!is_bardow ? "Lee & Lin 2001 (predictor-corrector)" :
                         "Bardow et al. 2006 (collide, then " +
                           std::string(prm.streaming == Streaming::tg2 ? "Lax-Wendroff" :
                                       prm.streaming == Streaming::tg3 ? "TG3" : "split TG3") +
                           " streaming)")
        << "\n  mass matrix  : "
        << (prm.mass == Mass::lumped ? "lumped" :
            prm.mass == Mass::cg     ? "consistent (CG + Jacobi, extrapolated start)" :
                                       "lumped + " + std::to_string(prm.richardson) + " Richardson pass(es)")
        << "\n  Ma = " << prm.mach << ", dt = " << dt << " (CFL " << dt / h_min << ")\n";
}



// nu = lambda c_s^2; reference time: viscous decay (tgv) or L/U0 (cavity)
void
CGDBE::set_reynolds(const double Re)
{
  reynolds = Re;
  tc->nu   = tc->U0 * tc->L / Re;
  lambda   = tc->nu / D2Q9::cs2;
  t_ref    = tc->reference_time();
  n_steps  = prm.max_steps > 0 ? prm.max_steps :
                                 static_cast<unsigned int>(std::ceil(prm.t_end * t_ref / dt));
  time = time_previous = 0.;
  step_no              = 0;
}



// ============================ collision (nodal) ==============================

D2Q9::Moments
CGDBE::macroscopic(const unsigned int i, const std::array<Number, Q> &fi) const
{
  auto m = D2Q9::moments(fi);
  if (const int wall = wall_of_node[i]; wall >= 0)
    {
      m.ux = wall_velocity[wall][0];
      m.uy = wall_velocity[wall][1];
    }
  return m;
}



void
CGDBE::compute_equilibrium()
{
  TimerOutput::Scope t(timer, "1 collision (nodal)");
  t_collision.start();
  const auto F = raw(f), FEQ = raw(feq);
  for (unsigned int i = 0; i < node.size(); ++i)
    {
      const auto e = D2Q9::equilibrium(macroscopic(i, gather(F, i)));
      for (unsigned int a = 0; a < Q; ++a)
        FEQ[a][i] = e[a];
    }
  t_collision.stop();
}



// Predictor (Eq. 17) with tau = dt/lambda: (1 + tau) fhat = f + tau feq + M^{-1} r,
// corrector (Eq. 18): f^{n+1} = fhat + tau (feq(fhat) - feq(f^n)).
void
CGDBE::predictor_corrector()
{
  TimerOutput::Scope t(timer, "1 collision (nodal)");
  t_collision.start();
  const auto   F = raw(f), FEQ = raw(feq), X = raw(incr);
  const Number tau = dt / lambda, inv = 1. / (1. + tau);
  for (unsigned int i = 0; i < node.size(); ++i)
    {
      std::array<Number, Q> fhat;
      fhat[0] = inv * (F[0][i] + tau * FEQ[0][i]);
      for (unsigned int a = 1; a < Q; ++a)
        fhat[a] = inv * (F[a][i] + tau * FEQ[a][i] + X[a - 1][i]);

      const auto eq_hat = D2Q9::equilibrium(macroscopic(i, fhat));
      for (unsigned int a = 0; a < Q; ++a)
        F[a][i] = fhat[a] + tau * (eq_hat[a] - FEQ[a][i]);
    }
  t_collision.stop();
}



// BGK collision in the transformed populations, omega = dt / (lambda + dt/2).
// Wall nodes: the non-equilibrium part is relaxed as everywhere else, the
// equilibrium part is rebuilt with the wall velocity, so that the post-collision
// momentum is exactly rho u_wall before streaming.
void
CGDBE::collide()
{
  TimerOutput::Scope t(timer, "1 collision (nodal)");
  t_collision.start();
  const auto   G     = raw(f);
  const Number omega = dt / (lambda + 0.5 * dt);
  for (unsigned int i = 0; i < node.size(); ++i)
    {
      const auto g   = gather(G, i);
      const auto geq = D2Q9::equilibrium(D2Q9::moments(g));
      if (wall_of_node[i] < 0)
        for (unsigned int a = 0; a < Q; ++a)
          G[a][i] -= omega * (g[a] - geq[a]);
      else
        {
          const auto geq_wall = D2Q9::equilibrium(macroscopic(i, g));
          for (unsigned int a = 0; a < Q; ++a)
            G[a][i] = geq_wall[a] + (1. - omega) * (g[a] - geq[a]);
        }
    }
  t_collision.stop();
}



// ========================== advection operators =============================
//
// Element matrices of Lee & Lin (19)-(22) / Bardow (13), never assembled:
//   C_e = int N (e.grad N^T),  K_e = int (e.grad N)(e.grad N^T) = 2 D_e,
//   Q_e = -C_e / (2 lambda).
// Both kernels evaluate all 8 moving populations at once (8-component
// FEEvaluation on the scalar DoFHandler, blocks 1..8 of the block vector) with
// the directions of the current sweep, stream_e.

// Lee & Lin, Eq. (17):  r = -dt C f - dt^2 [D f + Q (f - feq)]
//   = int phi [-dt + dt^2/(2 lambda)] e.grad f - int phi dt^2/(2 lambda) e.grad feq
//   - int (e.grad phi) dt^2/2 e.grad f
void
CGDBE::advection_leelin(const MatrixFree<dim, Number> &data,
                        BlockVectorType               &dst,
                        const BlockVectorType         &src,
                        const Range                   &range) const
{
  FEEvaluation<dim, fe_degree, n_q_1d, n_moving, Number> phi_f(data), phi_eq(data);

  const Number c_eq  = -0.5 * dt * dt / lambda;
  const Number c_f   = -dt - c_eq;
  const Number c_btd = -0.5 * dt * dt;

  for (unsigned int cell = range.first; cell < range.second; ++cell)
    {
      phi_f.reinit(cell);
      phi_f.read_dof_values(src, 1);
      phi_f.evaluate(EvaluationFlags::gradients);
      phi_eq.reinit(cell);
      phi_eq.read_dof_values(feq, 1);
      phi_eq.evaluate(EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi_f.n_q_points; ++q)
        {
          const auto grad_f  = phi_f.get_gradient(q);
          const auto grad_eq = phi_eq.get_gradient(q);

          Tensor<1, n_moving, VA>                 value;
          Tensor<1, n_moving, Tensor<1, dim, VA>> flux;
          for (unsigned int a = 0; a < n_moving; ++a)
            {
              const auto [ex, ey] = stream_e[a + 1];
              const VA e_grad_f   = ex * grad_f[a][0] + ey * grad_f[a][1];
              const VA e_grad_eq  = ex * grad_eq[a][0] + ey * grad_eq[a][1];
              value[a]            = c_f * e_grad_f + c_eq * e_grad_eq;
              flux[a][0]          = (c_btd * ex) * e_grad_f;
              flux[a][1]          = (c_btd * ey) * e_grad_f;
            }
          phi_f.submit_value(value, q);
          phi_f.submit_gradient(flux, q);
        }
      phi_f.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi_f.distribute_local_to_global(dst, 0);
    }
}



// Bardow, weak form of Eq. (9):  r = -dt C g* - dt^2 D g*
//   = -int phi dt e.grad g* - int (e.grad phi) dt^2/2 e.grad g*
void
CGDBE::advection_bardow(const MatrixFree<dim, Number> &data,
                        BlockVectorType               &dst,
                        const BlockVectorType         &src,
                        const Range                   &range) const
{
  FEEvaluation<dim, fe_degree, n_q_1d, n_moving, Number> phi(data);

  const Number c_f   = -dt;
  const Number c_btd = -0.5 * dt * dt;

  for (unsigned int cell = range.first; cell < range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(src, 1);
      phi.evaluate(EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto grad_g = phi.get_gradient(q);

          Tensor<1, n_moving, VA>                 value;
          Tensor<1, n_moving, Tensor<1, dim, VA>> flux;
          for (unsigned int a = 0; a < n_moving; ++a)
            {
              const auto [ex, ey] = stream_e[a + 1];
              const VA e_grad_g   = ex * grad_g[a][0] + ey * grad_g[a][1];
              value[a]            = c_f * e_grad_g;
              flux[a][0]          = (c_btd * ex) * e_grad_g;
              flux[a][1]          = (c_btd * ey) * e_grad_g;
            }
          phi.submit_value(value, q);
          phi.submit_gradient(flux, q);
        }
      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst, 0);
    }
}



// Surface term of the integration by parts of D, Eq. (23):
//   + dt^2/2 oint phi (n.e)(e.grad f), with the known f^n on the walls.
// leelin: kept as it is (the "no boundary condition" of Sec. 2.3).
// bardow: its zeroth moment, dt^2/2 n.div(Pi), is a mass flux through the
// wall; it is removed isotropically (no momentum is added).
void
CGDBE::wall_faces(const MatrixFree<dim, Number> &data,
                  BlockVectorType               &dst,
                  const BlockVectorType         &src,
                  const Range                   &range) const
{
  FEFaceEvaluation<dim, fe_degree, n_q_1d, n_moving, Number> phi(data, /*interior*/ true);

  for (unsigned int face = range.first; face < range.second; ++face)
    {
      phi.reinit(face);
      phi.read_dof_values(src, 1);
      phi.evaluate(EvaluationFlags::gradients);
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto grad_f = phi.get_gradient(q);
          const auto normal = phi.normal_vector(q);

          Tensor<1, n_moving, VA> value;
          VA                      mass_flux = 0.;
          for (unsigned int a = 0; a < n_moving; ++a)
            {
              const auto [ex, ey] = stream_e[a + 1];
              value[a] = (0.5 * dt * dt) * (ex * normal[0] + ey * normal[1]) *
                         (ex * grad_f[a][0] + ey * grad_f[a][1]);
              mass_flux += value[a];
            }
          if (is_bardow)
            for (unsigned int a = 0; a < n_moving; ++a)
              value[a] -= (D2Q9::w[a + 1] / (1. - D2Q9::w[0])) * mass_flux;
          phi.submit_value(value, q);
        }
      phi.integrate(EvaluationFlags::values);
      phi.distribute_local_to_global(dst, 0);
    }
}



void
CGDBE::assemble_rhs()
{
  TimerOutput::Scope t(timer, "2 advection (cell/face loop)");
  t_advection.start();
  if (is_bardow)
    matrix_free->loop(&CGDBE::advection_bardow, &CGDBE::no_inner_faces, &CGDBE::wall_faces,
                      this, rhs, f, /*zero dst*/ true);
  else
    {
      feq.update_ghost_values(); // second source vector, not seen by loop()
      matrix_free->loop(&CGDBE::advection_leelin, &CGDBE::no_inner_faces, &CGDBE::wall_faces,
                        this, rhs, f, /*zero dst*/ true);
      feq.zero_out_ghost_values();
    }
  t_advection.stop();
}



// ================================ streaming ==================================

// x <- A^{-1} rhs per population, A = M (tg2) or M + dt^2/6 K_e (tg3).
// Mass::cg starts from the extrapolation 2 x^{n} - x^{n-1} (the increment
// changes slowly in time); for the split sweeps the previous sweep's solution
// is used instead.
void
CGDBE::apply_mass_inverse(BlockVectorType &X)
{
  TimerOutput::Scope t(timer, "3 mass solves");
  t_mass.start();
  const auto &DL = *mass.get_matrix_lumped_diagonal_inverse();
  for (unsigned int a = 0; a < n_moving; ++a)
    {
      auto &x = X.block(a);
      const auto &r = rhs.block(a);
      if (stream_e[a + 1][0] == 0. && stream_e[a + 1][1] == 0.)
        {
          x = 0.; // population not streamed in this sweep
          continue;
        }
      switch (prm.mass)
        {
          case Mass::lumped:
            DL.vmult(x, r);
            mass_vector_passes += 2;
            mass_flops_per_node += 1;
            break;

          case Mass::richardson: // x_0 = M_L^{-1} r, x_{j+1} = x_j + M_L^{-1} (r - M x_j)
            DL.vmult(x, r);
            for (unsigned int j = 0; j < prm.richardson; ++j)
              {
                mass.vmult(tmp, x);
                tmp.sadd(-1., 1., r);
                DL.vmult(tmp, tmp);
                x += tmp;
              }
            mass_vector_passes += 2 + 6. * prm.richardson;
            mass_flops_per_node += 1 + (flops_mass_vmult + 4.) * prm.richardson;
            break;

          case Mass::cg:
            {
              if (!is_split)
                {
                  incr_prev.block(a).sadd(-1., 2., x); // 2 x^n - x^{n-1}
                  x.swap(incr_prev.block(a));          // x: guess, incr_prev: x^n
                }
              SolverControl control(200, prm.cg_tolerance * r.l2_norm());
              SolverCG<VectorType> cg(control);
              if (prm.streaming == Streaming::tg2)
                cg.solve(mass, x, r, *mass.get_matrix_diagonal_inverse());
              else
                cg.solve(TG3LHS(*matrix_free, dt * dt / 6., stream_e[a + 1]), x, r,
                         *mass.get_matrix_diagonal_inverse());
              cg_iterations += control.last_step();
              // per iteration: vmult 2, Jacobi 2, updates of x, r, p 6 vector passes
              mass_vector_passes += 10. * control.last_step() + 4;
              mass_flops_per_node += (flops_mass_vmult + 11.) * control.last_step();
            }
        }
    }
  t_mass.stop();
}



// Taylor-Galerkin streaming of the post-collision populations. tg3_split: the
// shift by e dt is the product of the shifts by (ex dt, 0) and (0, ey dt); on a
// tensor-product grid the 1-D sweeps commute and the composition is exact at
// CFL 1 for all eight directions (the diagonal populations get both sweeps).
void
CGDBE::stream()
{
  const auto add_increment = [&](const BlockVectorType &x) {
    TimerOutput::Scope t(timer, "1 collision (nodal)"); // nodal work, counted with the collision
    t_collision.start();
    for (unsigned int a = 0; a < n_moving; ++a)
      f.block(a + 1) += x.block(a);
    t_collision.stop();
  };

  if (!is_split)
    {
      assemble_rhs();
      apply_mass_inverse(incr);
      add_increment(incr);
      return;
    }
  for (unsigned int d = 0; d < dim; ++d)
    {
      for (unsigned int a = 0; a < Q; ++a)
        stream_e[a] = d == 0 ? Direction{{D2Q9::e[a][0], 0.}} : Direction{{0., D2Q9::e[a][1]}};
      auto &x = d == 0 ? incr : incr_prev;
      assemble_rhs();
      apply_mass_inverse(x);
      add_increment(x);
    }
  stream_e = D2Q9::e;
}



// ================================ time steps =================================

void
CGDBE::step_leelin()
{
  compute_equilibrium();
  assemble_rhs();
  apply_mass_inverse(incr);
  predictor_corrector();
}



void
CGDBE::step_bardow()
{
  collide();
  stream();
}



// Raw dump of the locally owned populations, one file per rank. Valid for the
// same mesh and number of ranks; the transformed populations of the bardow
// scheme additionally assume the same dt.
void
CGDBE::checkpoint(const std::string &name, const bool write)
{
  const std::string file =
    name + "." + std::to_string(Utilities::MPI::this_mpi_process(comm));
  const std::size_t bytes = node.size() * sizeof(Number);
  if (write)
    {
      std::ofstream out(file + ".tmp", std::ios::binary);
      for (unsigned int a = 0; a < Q; ++a)
        out.write(reinterpret_cast<const char *>(f.block(a).begin()), bytes);
      out.close();
      std::rename((file + ".tmp").c_str(), file.c_str()); // never leave a torn file
    }
  else
    {
      std::ifstream in(file, std::ios::binary);
      for (unsigned int a = 0; a < Q; ++a)
        in.read(reinterpret_cast<char *>(f.block(a).begin()), bytes);
      AssertThrow(in.good(), ExcMessage("cannot read checkpoint " + file));
    }
}



// Monitors (nodal quadrature). tgv: kinetic energy and velocity error against the analytic
// solution; cavity: mean velocity change per reference time (returns true once
// it drops below the steady-state tolerance).
bool
CGDBE::diagnostics(std::ostream *file)
{
  TimerOutput::Scope t(timer, "diagnostics + output");

  const auto          F = raw(f);
  std::vector<double> s(6, 0.); // |u-u_ex|^2, |u_ex|^2, |u|^2, rho, area, |u-u_prev|
  for (const unsigned int i : independent)
    {
      const auto m  = D2Q9::moments(gather(F, i));
      const auto ex = tc->exact(node[i], time);
      const double w = node_weight.local_element(i); // lumped-mass quadrature
      s[0] += w * ((m.ux - ex.ux) * (m.ux - ex.ux) + (m.uy - ex.uy) * (m.uy - ex.uy));
      s[1] += w * (ex.ux * ex.ux + ex.uy * ex.uy);
      s[2] += w * (m.ux * m.ux + m.uy * m.uy);
      s[3] += w * m.rho;
      s[4] += w;
      s[5] += w * std::hypot(m.ux - u_previous[i][0], m.uy - u_previous[i][1]);
      u_previous[i] = {{m.ux, m.uy}};
    }
  Utilities::MPI::sum(s, comm, s);

  const double mass_drift = s[3] / s[4] - tc->rho0;
  const auto   line       = [&](const std::array<double, 4> &v) {
    pcout << std::setw(8) << step_no << std::setw(12) << time / t_ref << std::fixed
          << std::setw(12) << v[0] << std::setw(12) << v[1] << std::scientific << std::setw(16)
          << v[2] << std::setw(14) << v[3] << std::defaultfloat << std::endl;
    if (file)
      *file << time << ' ' << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3] << std::endl;
  };

  bool converged = false;
  if (!steady)
    {
      if (step_no == 0)
        pcout << "    step     t/t_ref        E/E0  E/E0 exact   rel. L2 err(u)    <rho>-rho0\n";
      const double E0 = tc->energy_scale();
      line({{s[2] / s[4] / E0, s[1] / s[4] / E0, std::sqrt(s[0] / std::max(s[1], 1e-300)), mass_drift}});
    }
  else
    {
      if (step_no == 0)
        pcout << "    step     t/t_ref  <u^2>/U0^2           -  <|du|>/U0/t_ref    <rho>-rho0\n";
      const double rate =
        step_no == 0 ? 1. : s[5] / s[4] / tc->U0 / ((time - time_previous) / t_ref);
      line({{s[2] / s[4] / (tc->U0 * tc->U0), 0., rate, mass_drift}});
      converged = rate < prm.steady_tol;
    }
  time_previous = time;
  return converged;
}



// Stream function with u = -dpsi/dy, v = dpsi/dx (primary cavity vortex > 0):
//   (grad phi, grad psi) = (dphi/dx, v) - (dphi/dy, u),   psi = 0 on the walls,
// i.e. the L2-best fit to the (weakly compressible) velocity field.
void
CGDBE::compute_stream_function(BlockVectorType &velocity, VectorType &psi)
{
  LaplaceOperator laplace;
  laplace.initialize(matrix_free, {1});
  laplace.compute_diagonal();

  VectorType b;
  matrix_free->initialize_dof_vector(b, 1);
  matrix_free->initialize_dof_vector(psi, 1);

  matrix_free->cell_loop<VectorType, BlockVectorType>(
    [](const auto &data, auto &dst, const auto &src, const auto &range) {
      FEEvaluation<dim, fe_degree, n_q_1d, 2, Number> u(data, 0);
      FEEvaluation<dim, fe_degree, n_q_1d, 1, Number> phi(data, 1);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          u.reinit(cell);
          u.read_dof_values(src);
          u.evaluate(EvaluationFlags::values);
          phi.reinit(cell);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto         uq = u.get_value(q);
              Tensor<1, dim, VA> flux;
              flux[0] = uq[1];
              flux[1] = -uq[0];
              phi.submit_gradient(flux, q);
            }
          phi.integrate(EvaluationFlags::gradients);
          phi.distribute_local_to_global(dst);
        }
    },
    b,
    velocity,
    true);

  SolverControl control(10000, 1e-12 * b.l2_norm());
  SolverCG<VectorType>(control).solve(laplace, psi, b, *laplace.get_matrix_diagonal_inverse());
  constraints_wall.distribute(psi);
}



// tgv:    "x y u v |u|"         (velocities / U0)
// cavity: "x y u v psi rho-1"   (psi / (U0 L)), plus the vortex data of Table I
// Nodes are sorted into blank-line separated grid rows for gnuplot.
void
CGDBE::write_gnuplot(const std::string &name)
{
  TimerOutput::Scope t(timer, "diagnostics + output");

  for (unsigned int a = 0; a < Q; ++a) // fill in the periodic slave nodes
    constraints.distribute(f.block(a));

  const auto      F = raw(f);
  BlockVectorType velocity(2);
  for (unsigned int d = 0; d < 2; ++d)
    matrix_free->initialize_dof_vector(velocity.block(d));
  velocity.collect_sizes();

  using Row = std::array<double, 6>;
  std::vector<Row> local(node.size());
  for (unsigned int i = 0; i < node.size(); ++i)
    {
      const auto m = macroscopic(i, gather(F, i));
      local[i]     = {{node[i][0], node[i][1], m.ux / tc->U0, m.uy / tc->U0,
                       std::hypot(m.ux, m.uy) / tc->U0, m.rho - tc->rho0}};
      velocity.block(0).local_element(i) = m.ux / tc->U0;
      velocity.block(1).local_element(i) = m.uy / tc->U0;
    }
  if (steady)
    {
      VectorType psi;
      compute_stream_function(velocity, psi);
      for (unsigned int i = 0; i < node.size(); ++i)
        local[i][4] = psi.local_element(i) / tc->L;
    }

  const auto gathered = Utilities::MPI::gather(comm, local, 0);
  if (Utilities::MPI::this_mpi_process(comm) != 0)
    return;

  std::vector<Row> all;
  for (const auto &part : gathered)
    all.insert(all.end(), part.begin(), part.end());
  const auto row = [&](const Row &p) { return std::llround(p[1] / tc->L * (1ll << 40)); };
  std::sort(all.begin(), all.end(), [&](const Row &a, const Row &b) {
    return row(a) != row(b) ? row(a) < row(b) : a[0] < b[0];
  });

  std::ofstream out(name);
  out << "# t/t_ref = " << time / t_ref << " (step " << step_no << "), Re = " << reynolds
      << ", Ma = " << prm.mach << ", " << std::lround(std::sqrt(triangulation.n_global_active_cells())) << "^2 Q" << fe_degree
      << " elements\n"
      << (steady ? "# x y u/U0 v/U0 psi/(U0 L) rho-rho0\n" : "# x y u/U0 v/U0 |u|/U0\n");
  for (std::size_t i = 0; i < all.size(); ++i)
    {
      if (i > 0 && row(all[i]) != row(all[i - 1]))
        out << '\n';
      for (unsigned int c = 0; c < (steady ? 6u : 5u); ++c)
        out << all[i][c] << ' ';
      out << '\n';
    }

  if (steady) // extrema of psi: primary, lower-left and lower-right vortices (Table I)
    {
      const auto extremum = [&](const char *label, const double sign, const auto &inside) {
        const Row *best = nullptr;
        for (const auto &p : all)
          if (inside(p) && (!best || sign * p[4] > sign * (*best)[4]))
            best = &p;
        pcout << "  " << std::left << std::setw(20) << label << std::right << " psi = "
              << std::setw(12) << (*best)[4] << "  at (" << (*best)[0] << ", " << (*best)[1]
              << ")\n";
        out << "# " << label << ": psi = " << (*best)[4] << " at " << (*best)[0] << ' '
            << (*best)[1] << '\n';
      };
      pcout << "\n";
      extremum("primary vortex", +1., [](const Row &) { return true; });
      extremum("lower left vortex", -1., [](const Row &p) { return p[0] < 0.5 && p[1] < 0.5; });
      extremum("lower right vortex", -1., [](const Row &p) { return p[0] > 0.5 && p[1] < 0.5; });
    }
}



bool
CGDBE::run()
{
  setup();
  bool all_steady = true;

  MPI_Barrier(comm);
  Timer wall; // pure time stepping, excluding diagnostics and output
  wall.stop();

  for (const double Re : prm.reynolds)
    {
      set_reynolds(Re);
      pcout << "\n  Re = " << Re << ", nu = " << tc->nu << ", lambda = " << lambda
            << ", dt/lambda = " << dt / lambda << ", at most " << n_steps
            << " steps, t_ref = " << t_ref << "\n\n";

      const std::string tag = tc->tag(Re);

      std::ofstream diag_file;
      if (prm.output && Utilities::MPI::this_mpi_process(comm) == 0)
        {
          diag_file.open(tag + "_history.dat");
          diag_file << (steady ? "# t  <u^2>/U0^2  -  <|du|>/U0/t_ref  mean_rho-rho0\n" :
                                 "# t  E/E0  E/E0(exact)  rel_L2_error_u  mean_rho-rho0\n");
        }
      std::ostream *diag = diag_file.is_open() ? &diag_file : nullptr;

      diagnostics(diag);
      if (prm.output && !steady)
        write_gnuplot(tag + "_initial.dat");

      // n_diagnostic lines over the run, or one per reference time for a steady-state search
      const unsigned int interval =
        !steady ? std::max(1u, n_steps / std::max(1u, prm.n_diagnostic)) :
                 std::max(1u, static_cast<unsigned int>(std::lround(t_ref / dt)));

      bool steady_state = false;
      while (step_no < n_steps)
        {
          wall.start();
          is_bardow ? step_bardow() : step_leelin();
          time += dt;
          ++step_no;
          ++total_steps;
          wall.stop();
          if (step_no % interval == 0 || step_no == n_steps)
            {
              const bool converged = diagnostics(diag);
              if (!prm.checkpoint.empty())
                checkpoint(prm.checkpoint, /*write*/ true);
              if (converged && prm.max_steps == 0)
                {
                  pcout << "  -> steady state\n";
                  steady_state = true;
                  break;
                }
            }
        }
      if (steady && !steady_state && prm.max_steps == 0)
        {
          pcout << "  -> no steady state within t/t_ref = " << prm.t_end << "\n";
          all_steady = false;
        }

      if (prm.output)
        write_gnuplot(steady ? tag + ".dat" : tag + "_final.dat");

      if (!steady) // the Reynolds list is a continuation of steady states only
        break;
    }

  print_summary(wall.wall_time());
  timer.print_wall_time_statistics(comm);
  return all_steady;
}


// Stepping time, throughput and, per stage, the effective bandwidth of a
// single-pass traffic model and the rate of a flop model (see README).
void
CGDBE::print_summary(const double seconds) const
{
  const double nodes   = dof_handler.n_dofs();
  const double GB      = 1e-9 * nodes * sizeof(Number) * total_steps; // one vector pass, all steps
  pcout << "\n  time stepping : " << seconds << " s for " << total_steps << " steps  ("
        << 1e3 * seconds / total_steps << " ms/step)\n"
        << "  throughput    : " << 1e-6 * nodes * total_steps / seconds
        << " million node updates/s (MNUPS = MDoF/s, one DoF = all " << Q << " populations)\n";
  if (prm.mass == Mass::cg)
    pcout << "  CG iterations : " << double(cg_iterations) / (double(total_steps) * n_moving)
          << " per mass solve\n";

  // Minimal traffic model: every vector is read or written once per pass, no
  // cache reuse. Collision: leelin reads f (9) and writes feq (9), then reads
  // f, feq, incr (9+9+8) and writes f (9); bardow reads/writes f (9+9) and
  // adds incr to f (8+8 read, 8 write). Advection: reads f (8, +feq 8 for
  // leelin) and writes rhs (8), the face loop re-reads a boundary layer only.
  const double passes_coll = is_bardow ? 18 + 24 : 18 + 35;
  const double passes_adv  = is_bardow ? 16 : 24;
  const double cells_per_node = double(triangulation.n_global_active_cells()) / nodes;
  const double flops_coll     = is_bardow ? flops_collision_bardow : flops_collision_leelin;
  const double flops_adv      = flops_advection_cell * n_moving * cells_per_node * (is_bardow ? 1. : 1.4);
  const auto   stage = [&](const char *name, const Timer &t, const double passes, const double flops) {
    const double sec = t.wall_time();
    pcout << "  " << std::left << std::setw(12) << name << std::right << std::setw(9) << std::fixed
          << std::setprecision(3) << sec << " s " << std::setw(5) << std::setprecision(1)
          << 100 * sec / seconds << " % " << std::setw(6) << std::setprecision(2) << passes * GB / sec
          << " GB/s " << std::setw(6) << std::setprecision(2) << 1e-9 * flops * nodes * total_steps / sec
          << " GFlop/s  (" << std::setprecision(0) << passes << " vector passes, " << flops
          << " flops per node and step, intensity " << std::setprecision(2)
          << flops / (passes * sizeof(Number)) << " flop/byte)" << std::defaultfloat << "\n";
  };
  pcout << "  breakdown of the time stepping (single-pass traffic model, flop model, see README):\n";
  stage("collision", t_collision, passes_coll, flops_coll);
  stage("advection", t_advection, passes_adv, flops_adv);
  stage("mass solves", t_mass, mass_vector_passes / total_steps, mass_flops_per_node / total_steps);
  pcout << "  other (timers, loop overhead): " << std::fixed << std::setprecision(3)
        << seconds - t_collision.wall_time() - t_advection.wall_time() - t_mass.wall_time()
        << " s" << std::defaultfloat << "\n";
}



int
main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, /*threads*/ 1);
      if (!CGDBE(Parameters::parse(argc, argv)).run())
        return 2;
    }
  catch (const std::exception &exc)
    {
      std::cerr << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
