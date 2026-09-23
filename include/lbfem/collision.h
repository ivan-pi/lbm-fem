// Nodal (collision) part of the schemes. The populations are block vectors
// with one block per lattice direction (structure of arrays); nodal operations
// are lambdas mapped over the locally owned nodes by nodal_map(), which the
// compiler vectorizes across nodes. For that, an operation must be free of
// branches that depend on the node and of calls it cannot inline.
//
// Walls enter through the moments: a nodal operation receives the velocity
// the node imposes (FluidNode: its own, WallNode: the wall's), which is how
// the boundary condition acts in the equilibria of both schemes. The loop over
// all nodes is the fluid one; the few wall nodes are computed separately and
// overwritten, so that the vectorized loop reads no wall data.
#pragma once

#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>
#include <lbfem/test_case.h>

#include <array>
#include <type_traits>
#include <utility>
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
  // before it is written. The nodes are independent (ivdep), so the compiler
  // vectorizes the loop across them once op is inlined.
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
    [&](const auto... views) { // local copies: their block pointers are loop invariant
#pragma GCC ivdep
      for (unsigned int i = 0; i < n_nodes; ++i)
        {
          const Populations fi = op(i, views[i]...);
          for (unsigned int a = 0; a < Q; ++a)
            O[a][i] = fi[a];
        }
    }(view(in)...);
  }

  // The velocity a node imposes on its moments: the fluid's own, or that of
  // the wall it is on.
  struct FluidNode
  {
    D2Q9::Moments
    operator()(const D2Q9::Moments &m) const
    {
      return m;
    }
  };

  struct WallNode
  {
    D2Q9::Moments
    operator()(D2Q9::Moments m) const
    {
      m.ux = ux;
      m.uy = uy;
      return m;
    }
    Number ux, uy;
  };

  // Wall nodes and their velocities, from TestCase::wall_of and wall_velocity.
  struct Walls
  {
    void
    reinit(const std::vector<Point<2>> &node, const TestCase &tc)
    {
      std::vector<int> wall_of_node(node.size());
      for (unsigned int i = 0; i < node.size(); ++i)
        wall_of_node[i] = tc.wall_of(node[i]);
      std::vector<std::array<Number, 2>> wall_velocity(tc.n_walls());
      for (unsigned int w = 0; w < wall_velocity.size(); ++w)
        wall_velocity[w] = tc.wall_velocity(w);
      set(std::move(wall_of_node), std::move(wall_velocity));
    }

    // The wall index of every node (-1: interior) and the velocity of every wall.
    void
    set(std::vector<int> wall_of_node, std::vector<std::array<Number, 2>> wall_velocity)
    {
      of_node  = std::move(wall_of_node);
      velocity = std::move(wall_velocity);
      nodes.clear();
      for (unsigned int i = 0; i < of_node.size(); ++i)
        if (of_node[i] >= 0)
          nodes.push_back(i);
    }

    // What wall node k (of nodes) imposes.
    WallNode
    at(const unsigned int k) const
    {
      const auto [ux, uy] = velocity[of_node[nodes[k]]];
      return {ux, uy};
    }

    // Moments of node i, with the wall velocity on wall nodes (for output).
    D2Q9::Moments
    macroscopic(const unsigned int i, const Populations &fi) const
    {
      const auto m = D2Q9::moments(fi);
      return of_node[i] < 0 ? m : WallNode{velocity[of_node[i]][0], velocity[of_node[i]][1]}(m);
    }

    std::vector<int>                   of_node;  // -1: interior, else index into velocity
    std::vector<std::array<Number, 2>> velocity; // per wall
    std::vector<unsigned int>          nodes;    // the wall nodes
  };

  // out_i <- op(at_i, in_i...) at every locally owned node i, where at_i is
  // what node i imposes on its moments (FluidNode or WallNode). The loop over
  // all nodes, vectorized, is the fluid one; the wall nodes are computed
  // before it from the inputs as they are (out may be one of them) and
  // overwritten after it.
  template <typename Op, NodalInput... In>
    requires std::is_invocable_r_v<Populations, Op &, const FluidNode &, const PopulationsOf<In> &...> &&
             std::is_invocable_r_v<Populations, Op &, const WallNode &, const PopulationsOf<In> &...>
  void
  nodal_map(BlockVectorType &out, const Walls &walls, Op &&op, const In &...in)
  {
    std::vector<Populations> at_wall(walls.nodes.size());
    [&](const auto... views) {
      for (unsigned int k = 0; k < at_wall.size(); ++k)
        at_wall[k] = op(walls.at(k), views[walls.nodes[k]]...);
    }(view(in)...);

    nodal_map(out, [&](const unsigned int, const auto &...fi) { return op(FluidNode{}, fi...); }, in...);

    for (unsigned int a = 0; a < Q; ++a)
      for (unsigned int k = 0; k < at_wall.size(); ++k)
        out.block(a).local_element(walls.nodes[k]) = at_wall[k][a];
  }

  // feq <- feq(f), nodal (Lee & Lin).
  inline void
  compute_equilibrium(const BlockVectorType &f, BlockVectorType &feq, const Walls &walls)
  {
    nodal_map(
      feq, walls, [](const auto &at, const Populations &fi) { return D2Q9::equilibrium(at(D2Q9::moments(fi))); }, f);
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
      walls,
      [&](const auto &at, const Populations &fi, const Populations &feqi, const Populations &x) {
        Populations fhat;
        fhat[0] = inv * (fi[0] + tau * feqi[0]);
        for (unsigned int a = 1; a < Q; ++a)
          fhat[a] = inv * (fi[a] + tau * feqi[a] + x[a]);

        const auto  eq_hat = D2Q9::equilibrium(at(D2Q9::moments(fhat)));
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
  // g <- geq + (1 - omega) (g - geq), omega = dt / (lambda + dt/2): the
  // non-equilibrium part is relaxed, and the equilibrium part is that of the
  // imposed velocity, so that on a wall node the post-collision momentum is
  // exactly rho u_wall before streaming.
  inline void
  collide_bgk(BlockVectorType &g, const Walls &walls, const Number omega)
  {
    nodal_map(
      g,
      walls,
      [omega](const auto &at, const Populations &gi) {
        const auto  m          = D2Q9::moments(gi);
        const auto  geq        = D2Q9::equilibrium(m);
        const auto  geq_target = D2Q9::equilibrium(at(m)); // = geq on fluid nodes
        Populations out;
        for (unsigned int a = 0; a < Q; ++a)
          out[a] = geq_target[a] + (1. - omega) * (gi[a] - geq[a]);
        return out;
      },
      g);
  }
} // namespace lbfem
