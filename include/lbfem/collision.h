// Nodal (collision) part of the schemes. The populations are block vectors
// with one block per lattice direction (structure of arrays); a nodal
// operation is a lambda mapped over the locally owned nodes by nodal_map(),
// which the compiler vectorizes across nodes. For that, an operation must be
// free of branches that depend on the node and of calls it cannot inline.
//
// Walls enter through the moments: a nodal operation receives what the node
// imposes on its moments (FluidNode: nothing, WallNode: the wall velocity),
// which is how the boundary condition acts in the equilibria of both schemes.
// The loop over all nodes is the fluid one; the few wall nodes are computed
// apart and overwritten, so that the vectorized loop reads no wall data.
#pragma once

#include <lbfem/d2q9.h>
#include <lbfem/discretization.h>

#include <array>
#include <vector>

namespace lbfem
{
  inline constexpr unsigned int Q        = D2Q9::Q;
  inline constexpr unsigned int n_moving = Q - 1; // the rest population needs no streaming

  using Populations = std::array<Number, Q>; // at one node

  // The n blocks of a vector at a node (all Q populations, or the Q-1 moving
  // ones of an increment). Passed by value into the nodal loop, so that the
  // block pointers are loop invariant.
  template <unsigned int n>
  struct Nodal
  {
    explicit Nodal(const BlockVectorType &v)
    {
      AssertDimension(v.n_blocks(), n);
      for (unsigned int b = 0; b < n; ++b)
        p[b] = v.block(b).begin();
    }

    std::array<Number, n>
    operator[](const unsigned int i) const
    {
      std::array<Number, n> x;
      for (unsigned int b = 0; b < n; ++b)
        x[b] = p[b][i];
      return x;
    }

    std::array<const Number *, n> p;
  };

  // What a node imposes on its moments: nothing, or the velocity of its wall.
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
      m.ux = u[0];
      m.uy = u[1];
      return m;
    }
    Direction u;
  };

  // out_i <- op(impose_i, in_i...) at every locally owned node i, where impose_i
  // is FluidNode or WallNode. The wall nodes are computed first, from the
  // inputs as they are (out may be one of them); then all nodes as fluid
  // nodes, vectorized; then the wall nodes are written over.
  template <typename Op, unsigned int... n>
  void
  nodal_map(BlockVectorType &out, const Walls &walls, Op &&op, const Nodal<n>... in)
  {
    std::vector<Populations> at_wall(walls.nodes.size());
    for (unsigned int k = 0; k < at_wall.size(); ++k)
      at_wall[k] = op(WallNode{walls.velocity[k]}, in[walls.nodes[k]]...);

    AssertDimension(out.n_blocks(), Q);
    std::array<Number *, Q> O;
    for (unsigned int a = 0; a < Q; ++a)
      O[a] = out.block(a).begin();
    const unsigned int n_nodes = out.block(0).locally_owned_size();
#pragma GCC ivdep
    for (unsigned int i = 0; i < n_nodes; ++i)
      {
        const Populations fi = op(FluidNode{}, in[i]...);
        for (unsigned int a = 0; a < Q; ++a)
          O[a][i] = fi[a];
      }

    for (unsigned int a = 0; a < Q; ++a)
      for (unsigned int k = 0; k < at_wall.size(); ++k)
        O[a][walls.nodes[k]] = at_wall[k][a];
  }

  // feq <- feq(f), nodal (Lee & Lin).
  inline void
  compute_equilibrium(const BlockVectorType &f, BlockVectorType &feq, const Walls &walls)
  {
    nodal_map(
      feq,
      walls,
      [](const auto &impose, const Populations &fi) { return D2Q9::equilibrium(impose(D2Q9::moments(fi))); },
      Nodal<Q>(f));
  }

  // Lee & Lin, with theta = dt/lambda and the streaming increments x = M^{-1} r
  // of the moving populations (block a - 1 for population a): predictor
  // (Eq. 17) (1 + theta) fhat = f + theta feq + x, corrector (Eq. 18)
  // f^{n+1} = fhat + theta (feq(fhat) - feq(f^n)).
  inline void
  predictor_corrector(BlockVectorType       &f,
                      const BlockVectorType &feq,
                      const BlockVectorType &incr,
                      const Walls           &walls,
                      const Number           theta)
  {
    const Number inv = 1. / (1. + theta);
    nodal_map(
      f,
      walls,
      [&](const auto &impose, const Populations &fi, const Populations &feqi, const std::array<Number, n_moving> &x) {
        Populations fhat;
        fhat[0] = inv * (fi[0] + theta * feqi[0]);
        for (unsigned int a = 1; a < Q; ++a)
          fhat[a] = inv * (fi[a] + theta * feqi[a] + x[a - 1]);

        const auto  eq_hat = D2Q9::equilibrium(impose(D2Q9::moments(fhat)));
        Populations out;
        for (unsigned int a = 0; a < Q; ++a)
          out[a] = fhat[a] + theta * (eq_hat[a] - feqi[a]);
        return out;
      },
      Nodal<Q>(f),
      Nodal<Q>(feq),
      Nodal<n_moving>(incr));
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
      [omega](const auto &impose, const Populations &gi) {
        const auto  m          = D2Q9::moments(gi);
        const auto  geq        = D2Q9::equilibrium(m);
        const auto  geq_target = D2Q9::equilibrium(impose(m)); // = geq on fluid nodes
        Populations out;
        for (unsigned int a = 0; a < Q; ++a)
          out[a] = geq_target[a] + (1. - omega) * (gi[a] - geq[a]);
        return out;
      },
      Nodal<Q>(g));
  }
} // namespace lbfem
