// Post-processing and I/O: the stream function of a velocity field and raw
// checkpoints of the populations.
#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <lbfem/discretization.h>

#include <cstdio>
#include <fstream>
#include <string>

namespace lbfem
{
  // Stream function with u = -dpsi/dy, v = dpsi/dx (primary cavity vortex > 0):
  //   (grad phi, grad psi) = (dphi/dx, v) - (dphi/dy, u),   psi = 0 on the walls,
  // i.e. the L2-best fit to the (weakly compressible) velocity field. Needs
  // Settings::stream_function; velocity has two blocks in the population layout.
  template <int fe_degree>
  void
  compute_stream_function(const Discretization<fe_degree> &disc, BlockVectorType &velocity, VectorType &psi)
  {
    constexpr int dim = Discretization<fe_degree>::dim, n_q_1d = Discretization<fe_degree>::n_q_1d;
    using VA          = VectorizedArray<Number>;

    MatrixFreeOperators::LaplaceOperator<dim, fe_degree, n_q_1d, 1, VectorType> laplace;
    laplace.initialize(disc.matrix_free, {1});
    laplace.compute_diagonal();

    VectorType b;
    disc.matrix_free->initialize_dof_vector(b, 1);
    disc.matrix_free->initialize_dof_vector(psi, 1);

    disc.matrix_free->template cell_loop<VectorType, BlockVectorType>(
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
    disc.constraints_wall.distribute(psi);
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
