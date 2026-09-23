// Continuous Q_p discretization of the populations on a (distributed) box
// [0, L]^2: mesh, degrees of freedom, periodicity constraints, the MatrixFree
// data shared by all operators, the mass operator, and the nodal data used by
// the nodal collision (support points, lumped-mass weights, wall nodes).
#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

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

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <lbfem/d2q9.h>
#include <lbfem/test_case.h>

#include <memory>
#include <vector>

#ifndef LBFEM_DEGREE
#  define LBFEM_DEGREE 1
#endif

namespace lbfem
{
  using namespace dealii;

  // The polynomial degree of the FE_Q elements is a compile-time constant of the
  // matrix-free kernels (cmake -DLBFEM_DEGREE=p).
  inline constexpr int fe_degree = LBFEM_DEGREE;
  inline constexpr int dim       = 2;

  using Number          = double;
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;
  using MassOperator    = MatrixFreeOperators::MassOperator<dim, fe_degree, fe_degree + 1, 1, VectorType>;

  template <int n_components>
  using FEEval = FEEvaluation<dim, fe_degree, fe_degree + 1, n_components, Number>;

  struct MeshSettings
  {
    unsigned int refinements = 6; // 2^n x 2^n elements ...
    unsigned int n_cells     = 0; // ... or, if > 0, n_cells x n_cells elements
  };

  // The locally owned wall nodes and the velocity of the wall each is on.
  struct Walls
  {
    std::vector<unsigned int> nodes;
    std::vector<Direction>    velocity;
  };

  struct Discretization
  {
    explicit Discretization(const MPI_Comm comm)
      : triangulation(comm)
      , fe(fe_degree)
      , dof_handler(triangulation)
    {}

    // Box [0, tc.L]^2 with the periodicity, mesh transformation and walls of the test case.
    void
    reinit(const MeshSettings &settings, const TestCase &tc)
    {
      const double L = tc.L;
      if (settings.n_cells > 0)
        GridGenerator::subdivided_hyper_cube(triangulation, settings.n_cells, 0., L, /*colorize*/ true);
      else
        GridGenerator::hyper_cube(triangulation, 0., L, /*colorize*/ true);
      const unsigned int n_per_dir  = settings.n_cells > 0 ? settings.n_cells : (1u << settings.refinements);
      const unsigned int n_periodic = tc.n_periodic();
      if (n_periodic > 0)
        {
          std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>>
            periodic_faces;
          for (unsigned int d = 0; d < n_periodic; ++d)
            GridTools::collect_periodic_faces(triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
          triangulation.add_periodicity(periodic_faces);
        }
      if (settings.n_cells == 0)
        triangulation.refine_global(settings.refinements);

      h_min = L / n_per_dir;
      tc.transform_mesh(triangulation, h_min);

      dof_handler.distribute_dofs(fe);

      const IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
#if DEAL_II_VERSION_GTE(9, 6, 0)
      constraints.reinit(dof_handler.locally_owned_dofs(), relevant);
#else
      constraints.reinit(relevant);
#endif
      for (unsigned int d = 0; d < n_periodic; ++d)
        DoFTools::make_periodicity_constraints(dof_handler, 2 * d, 2 * d + 1, d, constraints);
      constraints.close();

      typename MatrixFree<dim, Number>::AdditionalData data;
      data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
      data.mapping_update_flags  = update_values | update_gradients | update_JxW_values;
      if (n_periodic < dim) // walls
        data.mapping_update_flags_boundary_faces = update_gradients | update_JxW_values | update_normal_vectors;

      matrix_free = std::make_shared<MatrixFree<dim, Number>>();
      matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(fe_degree + 1), data);

      mass.initialize(matrix_free);
      mass.compute_lumped_diagonal(); // lumped mass, Richardson passes, node_weight()
      mass.compute_diagonal();        // Jacobi preconditioner for CG

      const auto  support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
      const auto &owned          = dof_handler.locally_owned_dofs();
      node.resize(owned.n_elements());
      independent.clear();
      walls = {};
      for (unsigned int i = 0; i < node.size(); ++i)
        {
          const auto global = owned.nth_index_in_set(i);
          node[i]           = support_points.at(global);
          if (!constraints.is_constrained(global))
            independent.push_back(i);
          if (const auto u = tc.wall_velocity(node[i]))
            {
              walls.nodes.push_back(i);
              walls.velocity.push_back(*u);
            }
        }
    }

    // A block vector with n_blocks copies of the population layout.
    void
    initialize(BlockVectorType &v, const unsigned int n_blocks) const
    {
      v.reinit(n_blocks);
      for (unsigned int b = 0; b < n_blocks; ++b)
        matrix_free->initialize_dof_vector(v.block(b));
      v.collect_sizes();
    }

    // The time step for a Courant number cfl = dt |e_x| / h_min on Q1; for Q_p
    // it is reduced by p^2.
    double
    time_step(const double cfl) const
    {
      return cfl * h_min / (fe_degree * fe_degree);
    }

    // int phi_i: the row sums of M (lumped mass), a nodal quadrature weight.
    const VectorType &
    node_weight() const
    {
      return mass.get_matrix_lumped_diagonal()->get_vector();
    }

    unsigned int
    n_nodes() const // locally owned
    {
      return node.size();
    }

    parallel::distributed::Triangulation<dim> triangulation;
    const FE_Q<dim>                           fe;
    const MappingQ1<dim>                      mapping;
    DoFHandler<dim>                           dof_handler;
    AffineConstraints<Number>                 constraints; // periodicity, or none
    std::shared_ptr<MatrixFree<dim, Number>>  matrix_free;
    MassOperator                              mass;
    double                                    h_min = 0.;

    std::vector<Point<dim>>   node;        // support point per locally owned dof
    std::vector<unsigned int> independent; // owned dofs that are not periodic slaves
    Walls                     walls;
  };
} // namespace lbfem
