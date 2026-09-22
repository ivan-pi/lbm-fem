// Nodal (collision) part of the schemes. The populations are block vectors
// with one block per lattice direction; the nodal operations work on the
// locally owned entries only.
//
// Walls enter through the moments: on a wall node, macroscopic() returns the
// prescribed wall velocity, which is how the boundary condition acts in the
// equilibria of both schemes.
#pragma once

#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>
#include <lbfem/test_case.h>

#include <array>
#include <vector>

namespace lbfem
{
  inline constexpr unsigned int Q        = D2Q9::Q;
  inline constexpr unsigned int n_moving = Q - 1; // the rest population needs no streaming

  // Pointers to the locally owned entries of each block.
  inline std::array<Number *, Q>
  raw(BlockVectorType &v)
  {
    std::array<Number *, Q> p{};
    for (unsigned int b = 0; b < v.n_blocks(); ++b)
      p[b] = v.block(b).begin();
    return p;
  }

  // The Q populations of node i.
  inline std::array<Number, Q>
  gather(const std::array<Number *, Q> &F, const unsigned int i)
  {
    std::array<Number, Q> fi;
    for (unsigned int a = 0; a < Q; ++a)
      fi[a] = F[a][i];
    return fi;
  }

  // Wall nodes and their velocities, from TestCase::wall_of and wall_velocity.
  struct Walls
  {
    template <int dim>
    void
    reinit(const std::vector<Point<dim>> &node, const TestCase &tc)
    {
      of_node.resize(node.size());
      for (unsigned int i = 0; i < node.size(); ++i)
        of_node[i] = tc.wall_of(node[i]);
      velocity = {tc.wall_velocity(0), tc.wall_velocity(1)};
    }

    // Moments of node i, with the wall velocity on wall nodes.
    D2Q9::Moments
    macroscopic(const unsigned int i, const std::array<Number, Q> &fi) const
    {
      auto m = D2Q9::moments(fi);
      if (const int wall = of_node[i]; wall >= 0)
        {
          m.ux = velocity[wall][0];
          m.uy = velocity[wall][1];
        }
      return m;
    }

    std::vector<int>                   of_node;  // -1: interior, else index into velocity
    std::vector<std::array<Number, 2>> velocity; // per wall
  };

  // feq <- feq(f), nodal (Lee & Lin).
  inline void
  compute_equilibrium(BlockVectorType &f, BlockVectorType &feq, const Walls &walls)
  {
    const auto F = raw(f), FEQ = raw(feq);
    for (unsigned int i = 0; i < walls.of_node.size(); ++i)
      {
        const auto e = D2Q9::equilibrium(walls.macroscopic(i, gather(F, i)));
        for (unsigned int a = 0; a < Q; ++a)
          FEQ[a][i] = e[a];
      }
  }

  // Lee & Lin, with tau = dt/lambda and the streaming increments x = M^{-1} r of
  // the moving populations: predictor (Eq. 17) (1 + tau) fhat = f + tau feq + x,
  // corrector (Eq. 18) f^{n+1} = fhat + tau (feq(fhat) - feq(f^n)).
  inline void
  predictor_corrector(BlockVectorType       &f,
                      BlockVectorType       &feq,
                      BlockVectorType       &incr,
                      const Walls           &walls,
                      const Number           tau)
  {
    const auto   F = raw(f), FEQ = raw(feq), X = raw(incr);
    const Number inv = 1. / (1. + tau);
    for (unsigned int i = 0; i < walls.of_node.size(); ++i)
      {
        std::array<Number, Q> fhat;
        fhat[0] = inv * (F[0][i] + tau * FEQ[0][i]);
        for (unsigned int a = 1; a < Q; ++a)
          fhat[a] = inv * (F[a][i] + tau * FEQ[a][i] + X[a - 1][i]);

        const auto eq_hat = D2Q9::equilibrium(walls.macroscopic(i, fhat));
        for (unsigned int a = 0; a < Q; ++a)
          F[a][i] = fhat[a] + tau * (eq_hat[a] - FEQ[a][i]);
      }
  }

  // BGK collision of the transformed populations g (Bardow et al.),
  // g <- g - omega (g - geq), omega = dt / (lambda + dt/2). Wall nodes: the
  // non-equilibrium part is relaxed as everywhere else, the equilibrium part is
  // rebuilt with the wall velocity, so that the post-collision momentum is
  // exactly rho u_wall before streaming.
  inline void
  collide_bgk(BlockVectorType &g, const Walls &walls, const Number omega)
  {
    const auto G = raw(g);
    for (unsigned int i = 0; i < walls.of_node.size(); ++i)
      {
        const auto gi  = gather(G, i);
        const auto geq = D2Q9::equilibrium(D2Q9::moments(gi));
        if (walls.of_node[i] < 0)
          for (unsigned int a = 0; a < Q; ++a)
            G[a][i] -= omega * (gi[a] - geq[a]);
        else
          {
            const auto geq_wall = D2Q9::equilibrium(walls.macroscopic(i, gi));
            for (unsigned int a = 0; a < Q; ++a)
              G[a][i] = geq_wall[a] + (1. - omega) * (gi[a] - geq[a]);
          }
      }
  }
} // namespace lbfem
