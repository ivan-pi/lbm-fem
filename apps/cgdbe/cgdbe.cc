// -----------------------------------------------------------------------------
// cgdbe -- Characteristic Galerkin schemes for the Discrete Boltzmann Equation
//
// Driver for the lbfem library (include/lbfem): the benchmark flows of
//   T. Lee, C.-L. Lin, J. Comput. Phys. 171, 336 (2001)
//   A. Bardow, I. V. Karlin, A. A. Gusev, EPL 75, 434 (2006)
// with either scheme (see lbfem/schemes.h). Test cases:
//   --case tgv      Taylor-Green vortex, periodic box, wave numbers --modes n1,n2
//   --case couette  start-up Couette flow, periodic in x (Lee & Lin, Sec. 3.1)
//   --case cavity   lid-driven cavity on a wall-clustered mesh (Sec. 3.2)
//
//   --scheme leelin|bardow           predictor-corrector | collide-then-stream
//
// Options that change the streaming step (bardow):
//   --mass cg|lumped|richardson[k]   consistent (CG), lumped, or lumped with k
//                                    Richardson passes (k = 2 ~ consistent accuracy)
//   --streaming tg2|tg3|tg3-split    Lax-Wendroff; third-order Taylor-Galerkin
//                                    (M + dt^2/6 K_e on the left, stable to CFL 1);
//                                    TG3 as x and y sweeps (exact at CFL 1 on a
//                                    tensor-product grid)
//
// The driver owns what is specific to this application: the options, the
// Reynolds number continuation, the diagnostics, the output files and the
// performance summary.
// -----------------------------------------------------------------------------

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <lbfem/collision.h>
#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>
#include <lbfem/io.h>
#include <lbfem/schemes.h>
#include <lbfem/test_cases.h>
#include <lbfem/timers.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace dealii;
using namespace lbfem;

#ifndef LBFEM_DEGREE
#  define LBFEM_DEGREE 1 // polynomial degree of the FE_Q elements (cmake -DLBFEM_DEGREE=p)
#endif

// -------------------------------- parameters ---------------------------------

enum class Case
{
  tgv,
  couette,
  cavity
};

enum class SchemeType
{
  leelin, // lbfem::LeeLin
  bardow  // lbfem::Bardow
};

struct Parameters
{
  Case         test_case  = Case::tgv;
  SchemeType   scheme     = SchemeType::bardow;
  MassSettings mass;                       // type, Richardson passes, CG tolerance
  Streaming    streaming = Streaming::tg2; // bardow only
  MeshSettings mesh;                       // refinements or cells per direction

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
  unsigned int n_diagnostic = 20;   // diagnostic lines over the run
  bool         output       = true; // write the .dat files
  std::string  checkpoint;          // write populations here (per rank) at every diagnostic
  std::string  restart;             // read initial populations from this checkpoint

  static Parameters
  parse(int argc, char **argv);
};



// An option: its name, the form of its value ("" for a flag) and what it sets.
struct Option
{
  std::string_view                                        name, value;
  std::function<void(Parameters &, const std::string &)> set;
};

// The index of value in names, as the enum E (enumerators in the same order).
template <typename E>
  requires std::is_enum_v<E>
E
choose(const std::string_view option, const std::string &value, const std::vector<std::string_view> &names)
{
  const auto it = std::ranges::find(names, value);
  AssertThrow(it != names.end(), ExcMessage(std::string(option) + ": unknown value " + value));
  return static_cast<E>(it - names.begin());
}

const std::vector<Option> &
options()
{
  static const std::vector<Option> table = {
    {"--case", "tgv|couette|cavity",
     [](auto &p, const auto &v) { p.test_case = choose<Case>("--case", v, {"tgv", "couette", "cavity"}); }},
    {"--scheme", "leelin|bardow",
     [](auto &p, const auto &v) { p.scheme = choose<SchemeType>("--scheme", v, {"leelin", "bardow"}); }},
    {"--mass", "cg|lumped|richardson[k]",
     [](auto &p, const auto &v) {
       if (v.starts_with("richardson"))
         {
           p.mass.type = Mass::richardson;
           if (v.size() > 10)
             p.mass.richardson = std::stoul(v.substr(10));
         }
       else
         p.mass.type = choose<Mass>("--mass", v, {"cg", "lumped"});
     }},
    {"--streaming", "tg2|tg3|tg3-split",
     [](auto &p, const auto &v) { p.streaming = choose<Streaming>("--streaming", v, {"tg2", "tg3", "tg3-split"}); }},
    {"--refine", "n", [](auto &p, const auto &v) { p.mesh.refinements = std::stoul(v); }},
    {"--cells", "N", [](auto &p, const auto &v) { p.mesh.n_cells = std::stoul(v); }},
    {"--mach", "Ma", [](auto &p, const auto &v) { p.mach = std::stod(v); }},
    {"--modes", "n1,n2",
     [](auto &p, const auto &v) {
       const auto c = v.find(',');
       AssertThrow(c != std::string::npos, ExcMessage("--modes n1,n2"));
       p.modes[0] = std::stoul(v.substr(0, c));
       p.modes[1] = std::stoul(v.substr(c + 1));
     }},
    {"--reynolds", "Re[,Re..]",
     [](auto &p, const auto &v) {
       std::stringstream list(v);
       for (std::string item; std::getline(list, item, ',');)
         p.reynolds.push_back(std::stod(item));
     }},
    {"--cfl", "c", [](auto &p, const auto &v) { p.cfl = std::stod(v); }},
    {"--tend", "t/t_ref", [](auto &p, const auto &v) { p.t_end = std::stod(v); }},
    {"--steps", "n", [](auto &p, const auto &v) { p.max_steps = std::stoul(v); }},
    {"--stretch", "gamma", [](auto &p, const auto &v) { p.stretch = std::stod(v); }},
    {"--distort", "eps", [](auto &p, const auto &v) { p.distort = std::stod(v); }},
    {"--steady-tol", "eps", [](auto &p, const auto &v) { p.steady_tol = std::stod(v); }},
    {"--cg-tol", "eps", [](auto &p, const auto &v) { p.mass.cg_tolerance = std::stod(v); }},
    {"--checkpoint", "name", [](auto &p, const auto &v) { p.checkpoint = v; }},
    {"--restart", "name", [](auto &p, const auto &v) { p.restart = v; }},
    {"--no-output", "", [](auto &p, const auto &) { p.output = false; }},
  };
  return table;
}

Parameters
Parameters::parse(int argc, char **argv)
{
  const auto usage = [] {
    std::string text = "options:";
    for (const auto &[name, value, set] : options())
      text += " " + std::string(name) + (value.empty() ? "" : " " + std::string(value));
    return text;
  };

  Parameters prm;
  for (int i = 1; i < argc; ++i)
    {
      const std::string_view key = argv[i];
      const auto             opt = std::ranges::find(options(), key, &Option::name);
      AssertThrow(opt != options().end(), ExcMessage("unknown option " + std::string(key) + "\n" + usage()));
      std::string value;
      if (!opt->value.empty())
        {
          AssertThrow(i + 1 < argc, ExcMessage("missing value for " + std::string(key)));
          value = argv[++i];
        }
      opt->set(prm, value);
    }
  const bool cavity = prm.test_case == Case::cavity;
  AssertThrow(prm.streaming == Streaming::tg2 || (prm.scheme == SchemeType::bardow && prm.mass.type == Mass::cg),
              ExcMessage("--streaming tg3 needs --scheme bardow and --mass cg"));
  if (prm.reynolds.empty())
    prm.reynolds = {prm.test_case == Case::tgv ? 100. : cavity ? 400. : 10.};
  if (prm.cfl <= 0.)
    prm.cfl = !cavity ? 0.25 : (prm.scheme == SchemeType::leelin ? 0.5 : 0.4);
  if (prm.t_end <= 0.)
    prm.t_end = cavity ? 100. : 1.;
  return prm;
}

// ---------------------------------- driver -----------------------------------

class CGDBE
{
public:
  static constexpr int fe_degree = LBFEM_DEGREE; // 1: bilinear, as in the papers
  using Disc                     = Discretization<fe_degree>;

  explicit CGDBE(const Parameters &prm);
  bool
  run(); // false if a steady-state search ran out of time

private:
  void
  setup();
  void
  set_reynolds(const double Re);
  bool
  diagnostics(std::ostream *file);
  void
  write_gnuplot(const std::string &name);
  void
  print_summary(const double seconds) const;

  const Parameters          prm;
  std::unique_ptr<TestCase> tc;
  const bool                is_bardow, steady; // steady: run to a steady state
  const MPI_Comm            comm = MPI_COMM_WORLD;

  ConditionalOStream pcout;
  TimerOutput        timer;
  StageTimers        stage_timers;

  Disc                                     disc;
  Walls                                    walls;
  std::unique_ptr<lbfem::Scheme<fe_degree>> scheme;

  std::vector<std::array<Number, 2>> u_previous; // for the steady-state residual
  double                             time_previous = 0.;

  double   reynolds, lambda, dt, t_ref, time = 0.;
  unsigned n_steps = 0, step_no = 0, total_steps = 0;
};



CGDBE::CGDBE(const Parameters &prm)
  : prm(prm)
  , tc(prm.test_case == Case::tgv     ? std::unique_ptr<TestCase>(new TaylorGreenVortex(prm.modes[0], prm.modes[1], prm.distort)) :
       prm.test_case == Case::couette ? std::unique_ptr<TestCase>(new CouetteFlow()) :
                                        std::unique_ptr<TestCase>(new LidDrivenCavity(prm.stretch)))
  , is_bardow(prm.scheme == SchemeType::bardow)
  , steady(!tc->has_exact_solution())
  , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
  , timer(comm, pcout, TimerOutput::never, TimerOutput::wall_times)
  , stage_timers(timer)
  , disc(comm)
{}

void
CGDBE::setup()
{
  TimerOutput::Scope t(timer, "setup");

  // --- unit box: periodic (tgv) or wall-clustered (cavity); DoF index 1 for
  // the stream function of a steady state
  disc.reinit({.refinements = prm.mesh.refinements, .n_cells = prm.mesh.n_cells, .stream_function = steady}, *tc);
  {
    const auto n = disc.cell_batch_types();
    pcout << "  cell batches : " << disc.matrix_free->n_cell_batches() << " (cartesian " << n[0]
          << ", affine " << n[1] << ", general " << n[2] << ")\n";
  }

  if (is_bardow)
    scheme = std::make_unique<Bardow<fe_degree>>(
      disc, walls, StreamingSettings{.streaming = prm.streaming, .mass = prm.mass}, stage_timers);
  else
    scheme = std::make_unique<LeeLin<fe_degree>>(disc, walls, prm.mass, stage_timers);

  // --- physical parameters (lattice units: |e_x| = 1, c_s^2 = 1/3, rho0 = 1)
  tc->set_mach(prm.mach);
  dt = disc.time_step(prm.cfl);
  set_reynolds(prm.reynolds.front());
  if (!steady && prm.max_steps == 0) // hit t_end exactly
    {
      dt = prm.t_end * t_ref / n_steps;
      scheme->set_time_step({.dt = dt, .lambda = lambda});
    }

  // --- wall nodes, initial condition
  walls.reinit(disc.node, *tc);
  scheme->set_initial_populations(*tc);
  u_previous.assign(disc.n_nodes(), {{0., 0.}});
  if (!prm.restart.empty())
    read_checkpoint(scheme->f, prm.restart, comm);

  pcout << "CGDBE (Lee & Lin 2001), deal.II " << DEAL_II_PACKAGE_VERSION << ", "
        << Utilities::MPI::n_mpi_processes(comm) << " MPI rank(s), SIMD width "
        << VectorizedArray<Number>::size() << "\n  case         : " << tc->name()
        << "\n  elements     : " << disc.triangulation.n_global_active_cells() << "  (Q" << fe_degree
        << ", " << disc.dof_handler.n_dofs() << " nodes x " << Q << " populations), h_min = " << disc.h_min
        << "\n  scheme       : "
        << (!is_bardow ? "Lee & Lin 2001 (predictor-corrector)" :
                         "Bardow et al. 2006 (collide, then " +
                           std::string(prm.streaming == Streaming::tg2 ? "Lax-Wendroff" :
                                       prm.streaming == Streaming::tg3 ? "TG3" : "split TG3") +
                           " streaming)")
        << "\n  mass matrix  : "
        << (prm.mass.type == Mass::lumped ? "lumped" :
            prm.mass.type == Mass::cg     ? "consistent (CG + Jacobi, extrapolated start)" :
                                            "lumped + " + std::to_string(prm.mass.richardson) + " Richardson pass(es)")
        << "\n  Ma = " << prm.mach << ", dt = " << dt << " (CFL " << dt / disc.h_min << ")\n";
}



// nu = lambda c_s^2; reference time: viscous decay (tgv) or L/U0 (cavity)
void
CGDBE::set_reynolds(const double Re)
{
  reynolds = Re;
  tc->set_reynolds(Re);
  lambda   = tc->relaxation_time();
  t_ref    = tc->reference_time();
  n_steps  = prm.max_steps > 0 ? prm.max_steps :
                                 static_cast<unsigned int>(std::ceil(prm.t_end * t_ref / dt));
  time = time_previous = 0.;
  step_no              = 0;
  scheme->set_time_step({.dt = dt, .lambda = lambda});
}



// Monitors (nodal quadrature). tgv: kinetic energy and velocity error against the analytic
// solution; cavity: mean velocity change per reference time (returns true once
// it drops below the steady-state tolerance).
bool
CGDBE::diagnostics(std::ostream *file)
{
  TimerOutput::Scope t(timer, "diagnostics + output");

  const auto          F = view(scheme->f);
  std::vector<double> s(6, 0.); // |u-u_ex|^2, |u_ex|^2, |u|^2, rho, area, |u-u_prev|
  for (const unsigned int i : disc.independent)
    {
      const auto m  = D2Q9::moments(F[i]);
      const auto ex = tc->exact(disc.node[i], time);
      const double w = disc.node_weight.local_element(i); // lumped-mass quadrature
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



// tgv:    "x y u v |u|"         (velocities / U0)
// cavity: "x y u v psi rho-1"   (psi / (U0 L)), plus the vortex data of Table I
// Nodes are sorted into blank-line separated grid rows for gnuplot.
void
CGDBE::write_gnuplot(const std::string &name)
{
  TimerOutput::Scope t(timer, "diagnostics + output");

  for (unsigned int a = 0; a < Q; ++a) // fill in the periodic slave nodes
    disc.constraints.distribute(scheme->f.block(a));

  const auto      F = view(scheme->f);
  BlockVectorType velocity;
  disc.initialize(velocity, 2);

  using Row = std::array<double, 6>;
  std::vector<Row> local(disc.n_nodes());
  for (unsigned int i = 0; i < disc.n_nodes(); ++i)
    {
      const auto m = walls.macroscopic(i, F[i]);
      local[i]     = {{disc.node[i][0], disc.node[i][1], m.ux / tc->U0, m.uy / tc->U0,
                       std::hypot(m.ux, m.uy) / tc->U0, m.rho - tc->rho0}};
      velocity.block(0).local_element(i) = m.ux / tc->U0;
      velocity.block(1).local_element(i) = m.uy / tc->U0;
    }
  if (steady)
    {
      VectorType psi;
      compute_stream_function(disc, velocity, psi);
      for (unsigned int i = 0; i < disc.n_nodes(); ++i)
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
      << ", Ma = " << prm.mach << ", " << std::lround(std::sqrt(disc.triangulation.n_global_active_cells())) << "^2 Q" << fe_degree
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
          scheme->step();
          time += dt;
          ++step_no;
          ++total_steps;
          wall.stop();
          if (step_no % interval == 0 || step_no == n_steps)
            {
              const bool converged = diagnostics(diag);
              if (!prm.checkpoint.empty())
                write_checkpoint(scheme->f, prm.checkpoint, comm);
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
  const double nodes   = disc.dof_handler.n_dofs();
  const auto  &mass    = scheme->streaming.mass_solver();
  const double GB      = 1e-9 * nodes * sizeof(Number) * total_steps; // one vector pass, all steps
  pcout << "\n  time stepping : " << seconds << " s for " << total_steps << " steps  ("
        << 1e3 * seconds / total_steps << " ms/step)\n"
        << "  throughput    : " << 1e-6 * nodes * total_steps / seconds
        << " million node updates/s (MNUPS = MDoF/s, one DoF = all " << Q << " populations)\n";
  if (prm.mass.type == Mass::cg)
    pcout << "  CG iterations : " << double(mass.cg_iterations) / (double(total_steps) * n_moving)
          << " per mass solve\n";

  // Work models of the library kernels (a single-pass traffic model and a
  // flop count per node, see README).
  const double cells_per_node = double(disc.triangulation.n_global_active_cells()) / nodes;
  const Work   collision      = scheme->collision_work();
  const Work   advection      = scheme->streaming.advection_operator().work(cells_per_node);
  const auto   stage = [&](const char *name, const StageTimers::Stage s, const double passes, const double flops) {
    const double sec = stage_timers.wall_time(s);
    pcout << "  " << std::left << std::setw(12) << name << std::right << std::setw(9) << std::fixed
          << std::setprecision(3) << sec << " s " << std::setw(5) << std::setprecision(1)
          << 100 * sec / seconds << " % " << std::setw(6) << std::setprecision(2) << passes * GB / sec
          << " GB/s " << std::setw(6) << std::setprecision(2) << 1e-9 * flops * nodes * total_steps / sec
          << " GFlop/s  (" << std::setprecision(0) << passes << " vector passes, " << flops
          << " flops per node and step, intensity " << std::setprecision(2)
          << flops / (passes * sizeof(Number)) << " flop/byte)" << std::defaultfloat << "\n";
  };
  pcout << "  breakdown of the time stepping (single-pass traffic model, flop model, see README):\n";
  stage("collision", StageTimers::collision, collision.vector_passes, collision.flops);
  stage("advection", StageTimers::advection, advection.vector_passes, advection.flops);
  stage("mass solves", StageTimers::mass, mass.vector_passes / total_steps, mass.flops_per_node / total_steps);
  pcout << "  other (timers, loop overhead): " << std::fixed << std::setprecision(3)
        << seconds - stage_timers.wall_time(StageTimers::collision) -
             stage_timers.wall_time(StageTimers::advection) - stage_timers.wall_time(StageTimers::mass)
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
