// Continuous Q_p discretization of the populations on a (distributed) box
// [0, L]^2: mesh, degrees of freedom, periodicity constraints, the MatrixFree
// data shared by all operators, the mass operator and the nodal data (support
// points and lumped-mass weights) used by the nodal collision.
//
// MatrixFree DoF index 0 carries the populations (constrained only by
// periodicity), DoF index 1 a scalar with zero boundary values (the stream
// function, see io.h) when Settings::stream_function is set.
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

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <lbfem/test_case.h>

#include <array>
#include <memory>
#include <vector>

namespace lbfem
{
  using namespace dealii;

  using Number          = double;
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;
  using Direction       = std::array<double, 2>;

  struct MeshSettings
  {
    unsigned int refinements     = 6;     // 2^n x 2^n elements ...
    unsigned int n_cells         = 0;     // ... or, if > 0, n_cells x n_cells elements
    bool         stream_function = false; // set up DoF index 1 with psi = 0 on the boundary
  };

  template <int fe_degree>
  struct Discretization
  {
    static constexpr int dim    = 2;
    static constexpr int n_q_1d = fe_degree + 1;

    using MassOperator = MatrixFreeOperators::MassOperator<dim, fe_degree, n_q_1d, 1, VectorType>;
    using Settings     = MeshSettings;

    explicit Discretization(const MPI_Comm comm)
      : triangulation(comm)
      , fe(fe_degree)
      , dof_handler(triangulation)
    {}

    // Box [0, tc.L]^2 with the periodicity and mesh transformation of the test case.
    void
    reinit(const Settings &settings, const TestCase &tc)
    {
      const double L = tc.L;
      if (settings.n_cells > 0)
        GridGenerator::subdivided_hyper_cube(triangulation, settings.n_cells, 0., L, /*colorize*/ true);
      else
        GridGenerator::hyper_cube(triangulation, 0., L, /*colorize*/ true);
      const unsigned int n_per_dir = settings.n_cells > 0 ? settings.n_cells : (1u << settings.refinements);
      n_periodic                   = tc.n_periodic();
      if (n_periodic > 0)
        {
          std::vector<
            GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>>
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
      for (auto *c : {&constraints, &constraints_wall})
#if DEAL_II_VERSION_GTE(9, 6, 0)
        c->reinit(dof_handler.locally_owned_dofs(), relevant);
#else
        c->reinit(relevant);
#endif
      for (unsigned int d = 0; d < n_periodic; ++d)
        DoFTools::make_periodicity_constraints(dof_handler, 2 * d, 2 * d + 1, d, constraints);
      if (settings.stream_function)
        DoFTools::make_zero_boundary_constraints(dof_handler, constraints_wall);
      constraints.close();
      constraints_wall.close();

      typename MatrixFree<dim, Number>::AdditionalData data;
      data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
      data.mapping_update_flags  = update_values | update_gradients | update_JxW_values;
      if (n_periodic < dim) // walls
        data.mapping_update_flags_boundary_faces = update_gradients | update_JxW_values | update_normal_vectors;

      matrix_free = std::make_shared<MatrixFree<dim, Number>>();
      matrix_free->reinit(mapping,
                          std::vector<const DoFHandler<dim> *>{&dof_handler, &dof_handler},
                          std::vector<const AffineConstraints<Number> *>{&constraints, &constraints_wall},
                          std::vector<Quadrature<1>>{QGauss<1>(n_q_1d)},
                          data);

      mass.initialize(matrix_free, {0});
      mass.compute_lumped_diagonal(); // lumped mass, Richardson passes
      mass.compute_diagonal();        // Jacobi preconditioner for CG

      VectorType ones;
      matrix_free->initialize_dof_vector(ones);
      matrix_free->initialize_dof_vector(node_weight);
      ones = 1.;
      mass.vmult(node_weight, ones);

      const auto  support_points = DoFTools::map_dofs_to_support_points(mapping, dof_handler);
      const auto &owned          = dof_handler.locally_owned_dofs();
      node.resize(owned.n_elements());
      independent.clear();
      for (unsigned int i = 0; i < node.size(); ++i)
        {
          const auto global = owned.nth_index_in_set(i);
          node[i]           = support_points.at(global);
          if (!constraints.is_constrained(global))
            independent.push_back(i);
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

    unsigned int
    n_nodes() const // locally owned
    {
      return node.size();
    }

    // Number of cell batches with cartesian, affine and general geometry.
    std::array<unsigned int, 3>
    cell_batch_types() const
    {
      using GT = internal::MatrixFreeFunctions::GeometryType;
      std::array<unsigned int, 4> n_type{};
      for (unsigned int c = 0; c < matrix_free->n_cell_batches(); ++c)
        ++n_type[matrix_free->get_mapping_info().get_cell_type(c)];
      return {{n_type[GT::cartesian], n_type[GT::affine], n_type[GT::general]}};
    }

    parallel::distributed::Triangulation<dim> triangulation;
    const FE_Q<dim>                           fe;
    const MappingQ1<dim>                      mapping;
    DoFHandler<dim>                           dof_handler;
    AffineConstraints<Number>                 constraints;      // periodicity, or none
    AffineConstraints<Number>                 constraints_wall; // zero boundary values (DoF index 1)
    std::shared_ptr<MatrixFree<dim, Number>>  matrix_free;
    MassOperator                              mass;
    unsigned int                              n_periodic = 0;
    double                                    h_min      = 0.;

    std::vector<Point<dim>>   node;        // support point per locally owned dof
    VectorType                node_weight; // int phi_i (row sums of M)
    std::vector<unsigned int> independent; // owned dofs that are not periodic slaves
  };
} // namespace lbfem
