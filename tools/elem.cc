// Element matrices M, C_e, K_e of FE_Q(p) on the unit square with deal.II
// FEValues (Gauss p+1), printed as fractions for comparison with sympy.
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>

#include <cmath>
#include <cstdio>
using namespace dealii;
static const char *
frac(double v)
{
  static char b[32];
  for (int d = 1; d <= 3600; ++d)
    if (std::abs(v * d - std::round(v * d)) < 1e-10)
      {
        snprintf(b, 32, "%ld/%d", long(std::round(v * d)), d);
        return b;
      }
  snprintf(b, 32, "%g", v);
  return b;
}
int
main()
{
  for (int p : {1, 2})
    {
      Triangulation<2> tria;
      GridGenerator::hyper_cube(tria, 0, 1);
      FE_Q<2>       fe(p);
      DoFHandler<2> dh(tria);
      dh.distribute_dofs(fe);
      QGauss<2>   q(p + 1);
      FEValues<2> fev(fe, q, update_values | update_gradients | update_JxW_values);
      fev.reinit(dh.begin_active());
      const unsigned n = fe.dofs_per_cell;
      Tensor<1, 2>   e;
      e[0] = 1;
      e[1] = 0;
      FullMatrix<double> M(n, n), C(n, n), K(n, n);
      for (unsigned qp = 0; qp < q.size(); ++qp)
        for (unsigned a = 0; a < n; ++a)
          for (unsigned b = 0; b < n; ++b)
            {
              const double w = fev.JxW(qp), ea = e * fev.shape_grad(a, qp), eb = e * fev.shape_grad(b, qp);
              M(a, b) += fev.shape_value(a, qp) * fev.shape_value(b, qp) * w;
              C(a, b) += fev.shape_value(a, qp) * eb * w;
              K(a, b) += ea * eb * w;
            }
      printf("Q%d, deal.II dof ordering (vertices, then edges, then interior); support points:\n", p);
      for (unsigned a = 0; a < n; ++a)
        printf("  %u:(%g,%g)", a, fe.get_unit_support_points()[a][0], fe.get_unit_support_points()[a][1]);
      printf("\n");
      for (auto [name, A] : {std::pair{"M", &M}, std::pair{"C_(1,0)", &C}, std::pair{"K_(1,0)", &K}})
        {
          printf("%s =\n", name);
          for (unsigned a = 0; a < n; ++a)
            {
              for (unsigned b = 0; b < n; ++b)
                printf(" %8s", frac((*A)(a, b)));
              printf("\n");
            }
        }
    }
}
