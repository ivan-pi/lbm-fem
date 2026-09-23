// Post-processing and I/O: the stream function of a velocity field, nodal
// fields written for gnuplot, and raw checkpoints of the populations.
#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <lbfem/discretization.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace lbfem
{
  // Stream function with u = -dpsi/dy, v = dpsi/dx (primary cavity vortex > 0):
  //   (grad phi, grad psi) = (dphi/dx, v) - (dphi/dy, u),   psi = 0 on the walls,
  // i.e. the L2-best fit to the (weakly compressible) velocity field; velocity
  // has two blocks in the population layout. Sets up its own MatrixFree with
  // the zero boundary values (DoF index 1), which the populations do not need.
  inline void
  compute_stream_function(const Discretization &disc, BlockVectorType &velocity, VectorType &psi)
  {
    AffineConstraints<Number> zero;
#if DEAL_II_VERSION_GTE(9, 6, 0)
    zero.reinit(disc.dof_handler.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(disc.dof_handler));
#else
    zero.reinit(DoFTools::extract_locally_relevant_dofs(disc.dof_handler));
#endif
    DoFTools::make_zero_boundary_constraints(disc.dof_handler, zero);
    zero.close();

    typename MatrixFree<dim, Number>::AdditionalData data;
    data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
    data.mapping_update_flags  = update_values | update_gradients | update_JxW_values;
    const auto mf              = std::make_shared<MatrixFree<dim, Number>>();
    mf->reinit(disc.mapping,
               std::vector<const DoFHandler<dim> *>{&disc.dof_handler, &disc.dof_handler},
               std::vector<const AffineConstraints<Number> *>{&disc.constraints, &zero},
               std::vector<Quadrature<1>>{QGauss<1>(fe_degree + 1)},
               data);

    MatrixFreeOperators::LaplaceOperator<dim, fe_degree, fe_degree + 1, 1, VectorType> laplace;
    laplace.initialize(mf, {1});
    laplace.compute_diagonal();

    VectorType b;
    mf->initialize_dof_vector(b, 1);
    mf->initialize_dof_vector(psi, 1);

    mf->cell_loop<VectorType, BlockVectorType>(
      [](const auto &mf, auto &dst, const auto &src, const auto &range) {
        FEEval<2> u(mf, 0);
        FEEval<1> phi(mf, 1);
        for (unsigned int cell = range.first; cell < range.second; ++cell)
          {
            u.reinit(cell);
            u.read_dof_values(src);
            u.evaluate(EvaluationFlags::values);
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const auto                              uq = u.get_value(q);
                Tensor<1, dim, VectorizedArray<Number>> flux;
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
    zero.distribute(psi);
  }



  // Nodal rows (x, y, values...) of all ranks, gathered on rank 0, sorted into
  // grid rows (by y, robust to rounding, then x) and written blank-line
  // separated for gnuplot's splot, n_columns per row. Returns the sorted rows
  // on rank 0, nothing elsewhere.
  template <std::size_t N>
  std::vector<std::array<double, N>>
  write_grid_rows(std::ostream                             &out,
                  const std::vector<std::array<double, N>> &local,
                  const unsigned int                        n_columns,
                  const double                              L,
                  const MPI_Comm                            comm)
  {
    using Row = std::array<double, N>;
    std::vector<Row> all;
    for (const auto &part : Utilities::MPI::gather(comm, local, 0))
      all.insert(all.end(), part.begin(), part.end());
    const auto row = [&](const Row &p) {
      return std::llround(p[1] / L * (1ll << 40));
    };
    std::sort(all.begin(), all.end(), [&](const Row &a, const Row &b) {
      return row(a) != row(b) ? row(a) < row(b) : a[0] < b[0];
    });
    for (std::size_t i = 0; i < all.size(); ++i)
      {
        if (i > 0 && row(all[i]) != row(all[i - 1]))
          out << '\n';
        for (unsigned int c = 0; c < n_columns; ++c)
          out << all[i][c] << ' ';
        out << '\n';
      }
    return all;
  }



  // Raw dump of the locally owned populations, one file per rank. Valid for the
  // same mesh and number of ranks; transformed populations (Bardow) also assume
  // the same dt.
  inline std::string
  checkpoint_file(const std::string &name, const MPI_Comm comm)
  {
    return name + "." + std::to_string(Utilities::MPI::this_mpi_process(comm));
  }

  inline void
  write_checkpoint(const BlockVectorType &f, const std::string &name, const MPI_Comm comm)
  {
    const std::string file  = checkpoint_file(name, comm);
    const std::size_t bytes = f.block(0).locally_owned_size() * sizeof(Number);
    std::ofstream     out(file + ".tmp", std::ios::binary);
    for (unsigned int b = 0; b < f.n_blocks(); ++b)
      out.write(reinterpret_cast<const char *>(f.block(b).begin()), bytes);
    out.close();
    std::rename((file + ".tmp").c_str(), file.c_str()); // never leave a torn file
  }

  inline void
  read_checkpoint(BlockVectorType &f, const std::string &name, const MPI_Comm comm)
  {
    const std::string file  = checkpoint_file(name, comm);
    const std::size_t bytes = f.block(0).locally_owned_size() * sizeof(Number);
    std::ifstream     in(file, std::ios::binary);
    for (unsigned int b = 0; b < f.n_blocks(); ++b)
      in.read(reinterpret_cast<char *>(f.block(b).begin()), bytes);
    AssertThrow(in.good(), ExcMessage("cannot read checkpoint " + file));
  }
} // namespace lbfem
