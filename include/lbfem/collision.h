// Nodal (collision) part of the schemes. The populations are block vectors
// with one block per lattice direction; nodal operations are lambdas mapped
// over the locally owned nodes by nodal_map().
//
// Walls enter through the moments: on a wall node, Walls::macroscopic()
// returns the prescribed wall velocity, which is how the boundary condition
// acts in the equilibria of both schemes.
#pragma once

#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>
#include <lbfem/test_case.h>

#include <array>
#include <type_traits>
#include <vector>

namespace lbfem
{
  inline constexpr unsigned int Q        = D2Q9::Q;
  inline constexpr unsigned int n_moving = Q - 1; // the rest population needs no streaming

  using Populations = std::array<Number, Q>; // at one node

  // A vector of the moving populations only: block b holds population b + 1
  // (the streaming increments, for example).
  struct Moving
  {
    const BlockVectorType &v;
  };

  // Read access to the populations first..Q-1 of each locally owned node,
  // stored in blocks 0..Q-1-first; populations below first read as zero.
  // The layout is part of the type, so the gathers are fully unrolled.
  template <unsigned int first>
  class NodalView
  {
  public:
    explicit NodalView(const BlockVectorType &v)
    {
      AssertDimension(v.n_blocks(), Q - first);
      for (unsigned int b = 0; b < Q - first; ++b)
        p[b] = v.block(b).begin();
    }

    Populations
    operator[](const unsigned int i) const
    {
      Populations fi;
      for (unsigned int a = 0; a < first; ++a)
        fi[a] = 0.;
      for (unsigned int a = first; a < Q; ++a)
        fi[a] = p[a - first][i];
      return fi;
    }

  private:
    std::array<const Number *, Q - first> p{};
  };

  inline NodalView<0>
  view(const BlockVectorType &v)
  {
    return NodalView<0>(v);
  }

  inline NodalView<1>
  view(const Moving &m)
  {
    return NodalView<1>(m.v);
  }

  template <typename T>
  concept NodalInput = std::same_as<T, BlockVectorType> || std::same_as<T, Moving>;

  template <typename>
  using PopulationsOf = Populations; // maps a pack of inputs to a pack of Populations

  // out_i <- op(i, in_i...) at every locally owned node i, with the
  // populations of each input at that node (a BlockVectorType of all
  // populations, or Moving{v}). out may be one of the inputs: each node is read
  // before it is written.
  template <typename Op, NodalInput... In>
    requires std::is_invocable_r_v<Populations, Op &, unsigned int, const PopulationsOf<In> &...>
  void
  nodal_map(BlockVectorType &out, Op &&op, const In &...in)
  {
    AssertDimension(out.n_blocks(), Q);
    std::array<Number *, Q> O;
    for (unsigned int a = 0; a < Q; ++a)
      O[a] = out.block(a).begin();
    const unsigned int n_nodes = out.block(0).locally_owned_size();
    [&](const auto &...views) {
      for (unsigned int i = 0; i < n_nodes; ++i)
        {
          const Populations fi = op(i, views[i]...);
          for (unsigned int a = 0; a < Q; ++a)
            O[a][i] = fi[a];
        }
    }(view(in)...);
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
      velocity.resize(tc.n_walls());
      for (unsigned int w = 0; w < velocity.size(); ++w)
        velocity[w] = tc.wall_velocity(w);
    }

    // Moments of node i, with the wall velocity on wall nodes.
    D2Q9::Moments
    macroscopic(const unsigned int i, const Populations &fi) const
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
  compute_equilibrium(const BlockVectorType &f, BlockVectorType &feq, const Walls &walls)
  {
    nodal_map(
      feq,
      [&](const unsigned int i, const Populations &fi) { return D2Q9::equilibrium(walls.macroscopic(i, fi)); },
      f);
  }

  // Lee & Lin, with tau = dt/lambda and the streaming increments x = M^{-1} r of
  // the moving populations (x_0 = 0): predictor (Eq. 17)
  // (1 + tau) fhat = f + tau feq + x, corrector (Eq. 18)
  // f^{n+1} = fhat + tau (feq(fhat) - feq(f^n)).
  inline void
  predictor_corrector(BlockVectorType       &f,
                      const BlockVectorType &feq,
                      const Moving          &incr,
                      const Walls           &walls,
                      const Number           tau)
  {
    const Number inv = 1. / (1. + tau);
    nodal_map(
      f,
      [&](const unsigned int i, const Populations &fi, const Populations &feqi, const Populations &x) {
        Populations fhat;
        fhat[0] = inv * (fi[0] + tau * feqi[0]);
        for (unsigned int a = 1; a < Q; ++a)
          fhat[a] = inv * (fi[a] + tau * feqi[a] + x[a]);

        const auto  eq_hat = D2Q9::equilibrium(walls.macroscopic(i, fhat));
        Populations out;
        for (unsigned int a = 0; a < Q; ++a)
          out[a] = fhat[a] + tau * (eq_hat[a] - feqi[a]);
        return out;
      },
      f,
      feq,
      incr);
  }

  // BGK collision of the transformed populations g (Bardow et al.),
  // g <- g - omega (g - geq), omega = dt / (lambda + dt/2). Wall nodes: the
  // non-equilibrium part is relaxed as everywhere else, the equilibrium part is
  // rebuilt with the wall velocity, so that the post-collision momentum is
  // exactly rho u_wall before streaming.
  inline void
  collide_bgk(BlockVectorType &g, const Walls &walls, const Number omega)
  {
    nodal_map(
      g,
      [&](const unsigned int i, const Populations &gi) {
        const auto  geq = D2Q9::equilibrium(D2Q9::moments(gi));
        Populations out;
        if (walls.of_node[i] < 0)
          for (unsigned int a = 0; a < Q; ++a)
            out[a] = gi[a] - omega * (gi[a] - geq[a]);
        else
          {
            const auto geq_wall = D2Q9::equilibrium(walls.macroscopic(i, gi));
            for (unsigned int a = 0; a < Q; ++a)
              out[a] = geq_wall[a] + (1. - omega) * (gi[a] - geq[a]);
          }
        return out;
      },
      g);
  }
} // namespace lbfem
