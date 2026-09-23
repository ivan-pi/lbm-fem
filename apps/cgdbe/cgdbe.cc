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
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace dealii;
using namespace lbfem;

// -------------------------------- parameters ---------------------------------

enum class SchemeType
{
  leelin, // lbfem::LeeLin
  bardow  // lbfem::Bardow
};

struct Parameters
{
  std::string       test_case = "tgv"; // tgv | couette | cavity
  SchemeType        scheme    = SchemeType::bardow;
  StreamingSettings stream; // TG2/TG3 (leelin: TG2 only) and the mass solves
  MeshSettings      mesh;   // refinements or cells per direction

  double              mach     = 0.1;    // U0 / c_s
  unsigned int        modes[2] = {1, 4}; // tgv: wave numbers k_i = 2 pi n_i / L
  std::vector<double> reynolds;          // U0 L / nu (default 100 | 10 | 400); cavity: a list
                                         //   "400,1000,..." continues from each steady state
  double cfl   = 0.;                     // dt |e_x| / h_min (default 0.25; cavity 0.4 | leelin 0.5)
  double t_end = 0.;                     // in units of t_ref (default 1 | 1 | 100):
                                         //   tgv 1/(nu k^2), couette L^2/nu, cavity L/U0
  double       stretch      = 1.2;       // cavity: tanh wall clustering, 0 = uniform
  double       distort      = 0.;        // tgv: mesh distortion x += eps sin(2 pi x) sin(2 pi y)
  double       steady_tol   = 1e-4;      // cavity: stop if mean |du|/U0 per t_ref < tol
  unsigned int max_steps    = 0;         // > 0: fixed number of steps (benchmark)
  unsigned int n_diagnostic = 20;        // diagnostic lines over the run
  bool         output       = true;      // write the .dat files
  std::string  checkpoint;               // write populations here (per rank) at every diagnostic
  std::string  restart;                  // read initial populations from this checkpoint

  static Parameters
  parse(int argc, char **argv);

  std::unique_ptr<TestCase>
  make_test_case() const
  {
    if (test_case == "tgv")
      return std::make_unique<TaylorGreenVortex>(mach, modes[0], modes[1], distort);
    if (test_case == "couette")
      return std::make_unique<CouetteFlow>(mach);
    if (test_case == "cavity")
      return std::make_unique<LidDrivenCavity>(mach, stretch);
    AssertThrow(false, ExcMessage("--case: unknown value " + test_case));
    return nullptr;
  }
};



// An option: its name, the form of its value ("" for a flag) and what it sets.
struct Option
{
  std::string_view name, value;
  void (*set)(Parameters &, const std::string &);
};

// The index of value in names, as the enum E (enumerators in the same order).
template <typename E>
E
choose(const std::string_view option, const std::string &value, const std::vector<std::string_view> &names)
{
  const auto it = std::ranges::find(names, value);
  AssertThrow(it != names.end(), ExcMessage(std::string(option) + ": unknown value " + value));
  return static_cast<E>(it - names.begin());
}

const std::vector<Option> options = {
  {"--case",
   "tgv|couette|cavity",
   [](Parameters &p, const std::string &v) {
     p.test_case = v;
   }},
  {"--scheme",
   "leelin|bardow",
   [](Parameters &p, const std::string &v) {
     p.scheme = choose<SchemeType>("--scheme", v, {"leelin", "bardow"});
   }},
  {"--mass",
   "cg|lumped|richardson[k]",
   [](Parameters &p, const std::string &v) {
     if (v.starts_with("richardson"))
       {
         p.stream.mass.type = Mass::richardson;
         if (v.size() > 10)
           p.stream.mass.richardson = std::stoul(v.substr(10));
       }
     else
       p.stream.mass.type = choose<Mass>("--mass", v, {"cg", "lumped"});
   }},
  {"--streaming",
   "tg2|tg3|tg3-split",
   [](Parameters &p, const std::string &v) {
     p.stream.streaming = choose<Streaming>("--streaming", v, {"tg2", "tg3", "tg3-split"});
   }},
  {"--refine",
   "n",
   [](Parameters &p, const std::string &v) {
     p.mesh.refinements = std::stoul(v);
   }},
  {"--cells",
   "N",
   [](Parameters &p, const std::string &v) {
     p.mesh.n_cells = std::stoul(v);
   }},
  {"--mach",
   "Ma",
   [](Parameters &p, const std::string &v) {
     p.mach = std::stod(v);
   }},
  {"--modes",
   "n1,n2",
   [](Parameters &p, const std::string &v) {
     const auto n = Utilities::string_to_int(Utilities::split_string_list(v));
     AssertThrow(n.size() == 2 && n[0] >= 0 && n[1] >= 0, ExcMessage("--modes n1,n2"));
     p.modes[0] = n[0];
     p.modes[1] = n[1];
   }},
  {"--reynolds",
   "Re[,Re..]",
   [](Parameters &p, const std::string &v) {
     const auto list = Utilities::string_to_double(Utilities::split_string_list(v));
     p.reynolds.insert(p.reynolds.end(), list.begin(), list.end());
   }},
  {"--cfl",
   "c",
   [](Parameters &p, const std::string &v) {
     p.cfl = std::stod(v);
   }},
  {"--tend",
   "t/t_ref",
   [](Parameters &p, const std::string &v) {
     p.t_end = std::stod(v);
   }},
  {"--steps",
   "n",
   [](Parameters &p, const std::string &v) {
     p.max_steps = std::stoul(v);
   }},
  {"--stretch",
   "gamma",
   [](Parameters &p, const std::string &v) {
     p.stretch = std::stod(v);
   }},
  {"--distort",
   "eps",
   [](Parameters &p, const std::string &v) {
     p.distort = std::stod(v);
   }},
  {"--steady-tol",
   "eps",
   [](Parameters &p, const std::string &v) {
     p.steady_tol = std::stod(v);
   }},
  {"--cg-tol",
   "eps",
   [](Parameters &p, const std::string &v) {
     p.stream.mass.cg_tolerance = std::stod(v);
   }},
  {"--fused",
   "",
   [](Parameters &p, const std::string &) {
     p.stream.mass.fused = true;
   }},
  {"--checkpoint",
   "name",
   [](Parameters &p, const std::string &v) {
     p.checkpoint = v;
   }},
  {"--restart",
   "name",
   [](Parameters &p, const std::string &v) {
     p.restart = v;
   }},
  {"--no-output",
   "",
   [](Parameters &p, const std::string &) {
     p.output = false;
   }},
};

Parameters
Parameters::parse(int argc, char **argv)
{
  const auto usage = [] {
    std::string text = "options:";
    for (const auto &[name, value, set] : options)
      text += " " + std::string(name) + (value.empty() ? "" : " " + std::string(value));
    return text;
  };

  Parameters prm;
  for (int i = 1; i < argc; ++i)
    {
      const std::string_view key = argv[i];
      const auto             opt = std::ranges::find(options, key, &Option::name);
      AssertThrow(opt != options.end(), ExcMessage("unknown option " + std::string(key) + "\n" + usage()));
      std::string value;
      if (!opt->value.empty())
        {
          AssertThrow(i + 1 < argc, ExcMessage("missing value for " + std::string(key)));
          value = argv[++i];
        }
      opt->set(prm, value);
    }
  const bool cavity = prm.test_case == "cavity";
  if (prm.reynolds.empty())
    prm.reynolds = {prm.test_case == "tgv" ? 100. : cavity ? 400. : 10.};
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
  explicit CGDBE(const Parameters &prm)
    : prm(prm)
    , tc(prm.make_test_case())
    , steady(tc->steady())
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , timer(comm, pcout, TimerOutput::never, TimerOutput::wall_times)
    , disc(comm)
  {
    TimerOutput::Scope t(timer, "setup");

    // --- unit box: periodic (tgv) or wall-clustered (cavity)
    disc.reinit(prm.mesh, *tc);
    {
      using GT = internal::MatrixFreeFunctions::GeometryType;
      std::array<unsigned int, 4> n_type{};
      for (unsigned int c = 0; c < disc.matrix_free->n_cell_batches(); ++c)
        ++n_type[disc.matrix_free->get_mapping_info().get_cell_type(c)];
      pcout << "  cell batches : " << disc.matrix_free->n_cell_batches() << " (cartesian " << n_type[GT::cartesian]
            << ", affine " << n_type[GT::affine] << ", general " << n_type[GT::general] << ")\n";
    }

    if (prm.scheme == SchemeType::bardow)
      scheme = std::make_unique<Bardow>(disc, prm.stream);
    else
      scheme = std::make_unique<LeeLin>(disc, prm.stream);

    // --- time step (lattice units: |e_x| = 1, c_s^2 = 1/3, rho0 = 1). The
    // number of steps is that of the first Reynolds number; for a steady case
    // t_ref = L/U0 is the same for every Re.
    tc->set_reynolds(prm.reynolds.front());
    const double T = prm.t_end * tc->reference_time();
    dt             = disc.time_step(prm.cfl);
    n_steps        = prm.max_steps > 0 ? prm.max_steps : static_cast<unsigned int>(std::ceil(T / dt));
    if (!steady && prm.max_steps == 0) // hit t_end exactly
      dt = T / n_steps;
    scheme->set_time_step({.dt = dt, .lambda = tc->relaxation_time()});

    // --- initial condition
    scheme->set_initial_populations(*tc);
    u_previous.assign(disc.n_nodes(), {{0., 0.}});
    if (!prm.restart.empty())
      read_checkpoint(scheme->f, prm.restart, comm);

    pcout << "CGDBE (Lee & Lin 2001), deal.II " << DEAL_II_PACKAGE_VERSION << ", "
          << Utilities::MPI::n_mpi_processes(comm) << " MPI rank(s), SIMD width " << VectorizedArray<Number>::size()
          << "\n  case         : " << tc->name() << "\n  elements     : " << disc.triangulation.n_global_active_cells()
          << "  (Q" << fe_degree << ", " << disc.dof_handler.n_dofs() << " nodes x " << Q
          << " populations), h_min = " << disc.h_min << "\n  scheme       : "
          << (prm.scheme == SchemeType::leelin ? "Lee & Lin 2001 (predictor-corrector)" :
                                                 "Bardow et al. 2006 (collide, then " +
                                                   std::string(prm.stream.streaming == Streaming::tg2 ? "Lax-Wendroff" :
                                                               prm.stream.streaming == Streaming::tg3 ? "TG3" :
                                                                                                        "split TG3") +
                                                   " streaming)")
          << "\n  mass matrix  : "
          << (prm.stream.mass.type == Mass::lumped ? "lumped" :
              prm.stream.mass.type == Mass::cg     ? "consistent (CG + Jacobi, extrapolated start)" :
                                                     "lumped + " + std::to_string(prm.stream.mass.richardson) +
                                                   " Richardson pass(es)")
          << (prm.stream.mass.fused && prm.stream.mass.type != Mass::lumped ? ", vector updates fused" : "")
          << "\n  Ma = " << prm.mach << ", dt = " << dt << " (CFL " << dt / disc.h_min << ")\n";
  }

  // The Reynolds numbers in turn; false if a steady-state search ran out of time.
  bool
  run()
  {
    bool all_steady = true;

    MPI_Barrier(comm);
    Timer wall; // pure time stepping, excluding diagnostics and output
    wall.stop();

    for (const double Re : prm.reynolds)
      {
        tc->set_reynolds(Re);
        t_ref = tc->reference_time();
        scheme->set_time_step({.dt = dt, .lambda = tc->relaxation_time()});
        time = time_previous = 0.;
        step_no              = 0;
        pcout << "\n  Re = " << Re << ", nu = " << tc->nu << ", lambda = " << tc->relaxation_time()
              << ", dt/lambda = " << dt / tc->relaxation_time() << ", at most " << n_steps
              << " steps, t_ref = " << t_ref << "\n\n";

        const std::string tag = tc->tag(Re);
        history.close();
        if (prm.output && Utilities::MPI::this_mpi_process(comm) == 0)
          {
            history.open(tag + "_history.dat");
            history << (steady ? "# t  <u^2>/U0^2  -  <|du|>/U0/t_ref  mean_rho-rho0\n" :
                                 "# t  E/E0  E/E0(exact)  rel_L2_error_u  mean_rho-rho0\n");
          }

        diagnostics();
        if (prm.output && !steady)
          write_gnuplot(tag + "_initial.dat", Re);

        // n_diagnostic lines over the run, or one per reference time for a steady-state search
        const unsigned int interval = !steady ? std::max(1u, n_steps / std::max(1u, prm.n_diagnostic)) :
                                                std::max(1u, static_cast<unsigned int>(std::lround(t_ref / dt)));

        bool converged = false;
        while (step_no < n_steps && !converged)
          {
            wall.start();
            scheme->step();
            time += dt;
            ++step_no;
            ++total_steps;
            wall.stop();
            if (step_no % interval == 0 || step_no == n_steps)
              {
                converged = diagnostics() && prm.max_steps == 0;
                if (!prm.checkpoint.empty())
                  write_checkpoint(scheme->f, prm.checkpoint, comm);
                if (converged)
                  pcout << "  -> steady state\n";
              }
          }
        if (steady && !converged && prm.max_steps == 0)
          {
            pcout << "  -> no steady state within t/t_ref = " << prm.t_end << "\n";
            all_steady = false;
          }

        if (prm.output)
          write_gnuplot(steady ? tag + ".dat" : tag + "_final.dat", Re);

        if (!steady) // the Reynolds list is a continuation of steady states only
          break;
      }

    print_summary(wall.wall_time());
    timer.print_wall_time_statistics(comm);
    return all_steady;
  }

private:
  // Monitors (nodal quadrature). tgv: kinetic energy and velocity error against
  // the analytic solution; cavity: mean velocity change per reference time
  // (returns true once it drops below the steady-state tolerance).
  bool
  diagnostics()
  {
    TimerOutput::Scope t(timer, "diagnostics + output");

    const Nodal<Q>      F(scheme->f);
    std::vector<double> s(6, 0.); // |u-u_ex|^2, |u_ex|^2, |u|^2, rho, area, |u-u_prev|
    for (const unsigned int i : disc.independent)
      {
        const auto   m  = D2Q9::moments(F[i]);
        const auto   ex = tc->exact(disc.node[i], time);
        const double w  = disc.node_weight().local_element(i); // lumped-mass quadrature
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
      pcout << std::setw(8) << step_no << std::setw(12) << time / t_ref << std::fixed << std::setw(12) << v[0]
            << std::setw(12) << v[1] << std::scientific << std::setw(16) << v[2] << std::setw(14) << v[3]
            << std::defaultfloat << std::endl;
      if (history.is_open())
        history << time << ' ' << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3] << std::endl;
    };

    bool         converged = false;
    const double E0        = tc->energy_scale();
    if (!steady)
      {
        if (step_no == 0)
          pcout << "    step     t/t_ref        E/E0  E/E0 exact   rel. L2 err(u)    <rho>-rho0\n";
        line({{s[2] / s[4] / E0, s[1] / s[4] / E0, std::sqrt(s[0] / std::max(s[1], 1e-300)), mass_drift}});
      }
    else
      {
        if (step_no == 0)
          pcout << "    step     t/t_ref  <u^2>/U0^2           -  <|du|>/U0/t_ref    <rho>-rho0\n";
        const double rate = step_no == 0 ? 1. : s[5] / s[4] / tc->U0 / ((time - time_previous) / t_ref);
        line({{s[2] / s[4] / E0, 0., rate, mass_drift}});
        converged = rate < prm.steady_tol;
      }
    time_previous = time;
    return converged;
  }

  // tgv:    "x y u v |u|"         (velocities / U0)
  // cavity: "x y u v psi rho-1"   (psi / (U0 L)), plus the vortex data of Table I
  void
  write_gnuplot(const std::string &name, const double Re)
  {
    TimerOutput::Scope t(timer, "diagnostics + output");

    for (unsigned int a = 0; a < Q; ++a) // fill in the periodic slave nodes
      disc.constraints.distribute(scheme->f.block(a));

    using Row = std::array<double, 6>;
    const Nodal<Q>   F(scheme->f);
    std::vector<Row> local(disc.n_nodes());
    for (unsigned int i = 0; i < disc.n_nodes(); ++i)
      {
        const auto m = D2Q9::moments(F[i]);
        local[i]     = {{disc.node[i][0],
                         disc.node[i][1],
                         m.ux / tc->U0,
                         m.uy / tc->U0,
                         std::hypot(m.ux, m.uy) / tc->U0,
                         m.rho - tc->rho0}};
      }
    for (unsigned int k = 0; k < disc.walls.nodes.size(); ++k) // the wall velocity on the wall nodes
      {
        const auto [ux, uy] = disc.walls.velocity[k];
        Row &row            = local[disc.walls.nodes[k]];
        row[2]              = ux / tc->U0;
        row[3]              = uy / tc->U0;
        row[4]              = std::hypot(ux, uy) / tc->U0;
      }
    if (steady)
      {
        BlockVectorType velocity;
        disc.initialize(velocity, 2);
        for (unsigned int i = 0; i < disc.n_nodes(); ++i)
          {
            velocity.block(0).local_element(i) = local[i][2];
            velocity.block(1).local_element(i) = local[i][3];
          }
        VectorType psi;
        compute_stream_function(disc, velocity, psi);
        for (unsigned int i = 0; i < disc.n_nodes(); ++i)
          local[i][4] = psi.local_element(i) / tc->L;
      }

    std::ofstream out;
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      {
        out.open(name);
        out << "# t/t_ref = " << time / t_ref << " (step " << step_no << "), Re = " << Re << ", Ma = " << prm.mach
            << ", " << std::lround(std::sqrt(disc.triangulation.n_global_active_cells())) << "^2 Q" << fe_degree
            << " elements\n"
            << (steady ? "# x y u/U0 v/U0 psi/(U0 L) rho-rho0\n" : "# x y u/U0 v/U0 |u|/U0\n");
      }
    const auto all = write_grid_rows(out, local, steady ? 6u : 5u, tc->L, comm);
    if (!steady || Utilities::MPI::this_mpi_process(comm) != 0)
      return;

    // extrema of psi: primary, lower-left and lower-right vortices (Table I)
    const auto extremum = [&](const char *label, const double sign, const auto &inside) {
      const Row *best = nullptr;
      for (const auto &p : all)
        if (inside(p) && (!best || sign * p[4] > sign * (*best)[4]))
          best = &p;
      pcout << "  " << std::left << std::setw(20) << label << std::right << " psi = " << std::setw(12) << (*best)[4]
            << "  at (" << (*best)[0] << ", " << (*best)[1] << ")\n";
      out << "# " << label << ": psi = " << (*best)[4] << " at " << (*best)[0] << ' ' << (*best)[1] << '\n';
    };
    pcout << "\n";
    extremum("primary vortex", +1., [](const Row &) { return true; });
    extremum("lower left vortex", -1., [](const Row &p) { return p[0] < 0.5 && p[1] < 0.5; });
    extremum("lower right vortex", -1., [](const Row &p) { return p[0] > 0.5 && p[1] < 0.5; });
  }

  // Stepping time, throughput and, per stage, the effective bandwidth of a
  // single-pass traffic model (every vector read or written once per pass, no
  // cache reuse; a vector pass is 8 N_nodes bytes) and the rate of a flop model
  // (a multiply-add counts 2); see README. The times are the maxima over the
  // ranks; the work is per node and step.
  void
  print_summary(const double stepping_seconds) const
  {
    using S              = StageTimers;
    const auto   t       = scheme->timers.max_wall_times(stepping_seconds, comm);
    const double seconds = Utilities::MPI::max(stepping_seconds, comm);
    const double nodes   = disc.dof_handler.n_dofs();
    const double GB      = 1e-9 * nodes * sizeof(Number) * total_steps; // one vector pass, all steps

    const bool   leelin = prm.scheme == SchemeType::leelin;
    const auto  &mass   = prm.stream.mass;
    const double sweeps = prm.stream.streaming == Streaming::tg3_split ? 2. : 1.;
    const double solves = sweeps == 1. ? 8. : 12.; // mass solves per step (the split sweeps move 6 populations each)
    const double its =
      mass.type == Mass::cg ? double(scheme->streaming.mass.cg_iterations) / (total_steps * solves) : 0.;

    struct Work
    {
      double passes, flops;
    };
    // collision, leelin: reads f (9), writes feq (9), reads f, feq, incr (9+9+8), writes f (9); equilibrium 122,
    // predictor 32, equilibrium of fhat 122, corrector 27. bardow: collide in place (9+9), add each increment
    // (8+8+8 per sweep of the moving populations); moments 20, equilibrium 102, relaxation 27, increment 1 each.
    const Work collision = leelin ? Work{53., 303.} : Work{18. + 3. * solves, 149. + solves};
    // advection per sweep: reads f (8; leelin also feq), writes rhs (8); 184 flops per cell and population for Q1
    // on a cartesian cell (gradients 72, quadrature 24, integration 88), leelin 1.4 x for its second field.
    const double cells_per_node = double(disc.triangulation.n_global_active_cells()) / nodes;
    const double flops_per_cell = 184. * (fe_degree == 1 ? 1. : 3.5 * fe_degree) * n_moving * cells_per_node;
    const Work   advection      = {sweeps * (leelin ? 24. : 16.), sweeps * flops_per_cell * (leelin ? 1.4 : 1.)};
    // per mass solve: vmult 68 flops per node; CG per iteration vmult 2, Jacobi 2, updates of x, r, p 6 passes and
    // 11 flops (fused: 8 passes, 21 flops with its 7 reductions), plus 4 passes; Richardson x_0 3 passes, per pass
    // 7 (fused 5) and 3 flops.
    const Work per_solve =
      mass.type == Mass::cg     ? Work{(mass.fused ? 8. : 10.) * its + 4., (68. + (mass.fused ? 21. : 11.)) * its} :
      mass.type == Mass::lumped ? Work{2., 1.} :
                                  Work{3. + (mass.fused ? 5. : 7.) * mass.richardson, 1. + 71. * mass.richardson};
    const Work solve_work = {per_solve.passes * solves, per_solve.flops * solves};

    pcout << "\n  time stepping : " << seconds << " s for " << total_steps << " steps  (" << 1e3 * seconds / total_steps
          << " ms/step)\n"
          << "  throughput    : " << 1e-6 * nodes * total_steps / seconds
          << " million node updates/s (MNUPS = MDoF/s, one DoF = all " << Q << " populations)\n";
    if (mass.type == Mass::cg)
      pcout << "  CG iterations : " << its << " per mass solve\n";

    const auto stage = [&](const char *name, const double sec, const Work &w) {
      pcout << "  " << std::left << std::setw(12) << name << std::right << std::setw(9) << std::fixed
            << std::setprecision(3) << sec << " s " << std::setw(5) << std::setprecision(1) << 100 * sec / seconds
            << " % " << std::setw(6) << std::setprecision(2) << w.passes * GB / sec << " GB/s " << std::setw(6)
            << std::setprecision(2) << 1e-9 * w.flops * nodes * total_steps / sec << " GFlop/s  ("
            << std::setprecision(0) << w.passes << " vector passes, " << w.flops
            << " flops per node and step, intensity " << std::setprecision(2) << w.flops / (w.passes * sizeof(Number))
            << " flop/byte)" << std::defaultfloat << "\n";
    };
    pcout << "  breakdown of the time stepping (max over ranks; single-pass traffic model, flop model, see README):\n";
    stage("collision", t[S::collision], collision);
    stage("advection", t[S::advection], advection);
    stage("mass solves", t[S::mass], solve_work);
    pcout << "  other (timers, loop overhead): " << std::fixed << std::setprecision(3) << t[S::n_stages] << " s"
          << std::defaultfloat << "\n";
  }

  const Parameters          prm;
  std::unique_ptr<TestCase> tc;
  const bool                steady; // run to a steady state
  const MPI_Comm            comm = MPI_COMM_WORLD;

  ConditionalOStream pcout;
  TimerOutput        timer; // setup, diagnostics and output (outside the time loop)

  Discretization          disc;
  std::unique_ptr<Scheme> scheme;
  std::ofstream           history; // the diagnostics of the current Reynolds number, rank 0

  std::vector<std::array<Number, 2>> u_previous; // for the steady-state residual
  double                             time_previous = 0.;

  double   dt, t_ref, time = 0.;
  unsigned n_steps, step_no = 0, total_steps = 0;
};



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
