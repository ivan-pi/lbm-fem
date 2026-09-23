# The method and the code

`cgdbe` solves the D2Q9 discrete Boltzmann equation with continuous $`Q_p`$
finite elements (bilinear $`Q_1`$ by default) on deal.II's matrix-free
infrastructure. Nothing is assembled: the streaming step is a matrix-free
Taylor–Galerkin (characteristic Galerkin) step, the collision a nodal loop.
This page describes the discretisation, the two time-stepping schemes, the
wall treatment, the streaming variants, and how the code is organised. The
numerical experiments are in [results.md](results.md), the von Neumann
analyses in [analysis.md](analysis.md), and the performance measurements in
[performance.md](performance.md).

## Test cases and schemes

| | |
|---|---|
| `--case tgv` | Taylor–Green vortex in a periodic box, wave numbers $`k_i = 2\pi n_i/L`$ (`--modes n1,n2`, default `1,4` so that the shear $`u_y + v_x \neq 0`$), analytic solution |
| `--case couette` | start-up Couette flow, periodic in $`x`$ (Lee & Lin, Sec. 3.1), analytic solution |
| `--case cavity` | lid-driven cavity on a wall-clustered mesh (Lee & Lin, Sec. 3.2) |
| `--scheme leelin` | predictor–corrector of T. Lee, C.-L. Lin, *J. Comput. Phys.* **171** (2001) 336, Eqs. (17)–(18) |
| `--scheme bardow` | collide-then-stream of A. Bardow, I. V. Karlin, A. A. Gusev, *EPL* **75** (2006) 434 |

## Formulation

Discrete velocities $`\boldsymbol{e}_\alpha`$, $`\lambda = \nu / c_s^2`$ the relaxation
time of the *continuous* Boltzmann equation in both schemes, and the element
matrices

```math
M = \int_{\Omega_e} N N^T d\Omega, \qquad
C_\alpha = \int_{\Omega_e} N (\boldsymbol{e}_\alpha \cdot \nabla N^T) d\Omega, \qquad
D_\alpha = \frac{1}{2} \int_{\Omega_e} (\boldsymbol{e}_\alpha \cdot \nabla N) (\boldsymbol{e}_\alpha \cdot \nabla N^T) d\Omega, \qquad
Q_\alpha = -\frac{1}{2\lambda} C_\alpha .
```

$`D_\alpha`$ is Eq. (21) of Lee & Lin after integration by parts (their Eq. (23)); the
strong form (21) itself is unusable for $`C^0`$ elements, since for $`Q_1`$ the
second derivative $`\partial^2 N/\partial x^2`$ vanishes inside every element and the
whole operator sits in the jumps of $`\partial N/\partial x`$ across faces. $`M`$,
$`C_\alpha`$ and $`D_\alpha`$ are identical to Bardow's Eq. (13); Bardow's
$`Q_i = -C_i/\lambda`$ is twice Lee & Lin's, which is the "factor of 2" Bardow
mentions and reflects the different collision treatment of their Eq. (10) versus
Lee & Lin's Eq. (12) — irrelevant for collide-then-stream, which uses no $`Q`$ at
all. The matrices are tabulated as stencils by `analysis/stencils.py` (sympy and deal.II
`FEValues`, `tools/elem.cc`, agree; on a uniform $`Q_1`$ grid they are the stencils of
[`periodic_lw_fem.F90`](https://github.com/ivan-pi/dugks-mwe/blob/main/src/periodic_lw_fem.F90)).
The surface term of the integration by parts is

```math
S_\alpha f = \frac{1}{2} \oint_{\Gamma_e} N (\boldsymbol{n} \cdot \boldsymbol{e}_\alpha) (\boldsymbol{e}_\alpha \cdot \nabla f) d\Gamma .
```

### Lee & Lin (`--scheme leelin`)

With $`\theta = \Delta t / \lambda`$, for every moving population $`\alpha = 1, \dots, 8`$:

```math
(1 + \theta) M \hat{f}_\alpha = M (f_\alpha^n + \theta f_\alpha^{eq,n})
  - \Delta t C_\alpha f_\alpha^n
  - \Delta t^2 \left[ (D_\alpha - S_\alpha) f_\alpha^n + Q_\alpha (f_\alpha^n - f_\alpha^{eq,n}) \right]
```

```math
f_\alpha^{n+1} = \hat{f}_\alpha + \theta \left( f_\alpha^{eq}(\hat{f}) - f_\alpha^{eq,n} \right)
```

### Bardow, Karlin & Gusev (`--scheme bardow`)

The populations are the transformed ones,
$`g_\alpha = f_\alpha + \frac{\Delta t}{2 \lambda} (f_\alpha - f_\alpha^{eq})`$, which have
the same density and momentum as $`f_\alpha`$. The BGK collision is done in place and
is exactly the lattice-Boltzmann one; the post-collision populations are then
streamed by a purely advective Lax–Wendroff (Taylor–Galerkin) step, i.e. the weak
form of Bardow's Eq. (9):

```math
g_\alpha^{*} = g_\alpha^n - \omega (g_\alpha^n - g_\alpha^{eq,n}), \qquad
\omega = \frac{\Delta t}{\lambda + \Delta t / 2}
```

```math
M (g_\alpha^{n+1} - g_\alpha^{*}) = - \Delta t C_\alpha g_\alpha^{*} - \Delta t^2 D_\alpha g_\alpha^{*}
```

The transformation of Guo & Zhao (a half collision step) is what makes the
collision exact: $`g`$ has the same density and momentum as $`f`$ (collision
invariants), and the modified rate $`\omega = \Delta t / (\lambda + \Delta t/2)`$ is
the standard LBM expression for a non-unit $`\Delta t`$. The time step is then
restricted by the CFL condition of the advection term only, not by the collision.
In the Lee–Lin scheme the same $`\lambda`$ enters the Crank–Nicolson-type
predictor–corrector; the $`-1/2`$ correction quoted in their Sec. 2.2 refers to the
$`\theta = 0`$ lattice reduction (their Eq. (9)) and is not applied here — the
Taylor–Green decay rates of the two schemes agree to $`10^{-3}`$ with the same
$`\lambda`$ (see [results.md](results.md#taylorgreen-vortex)).

No equilibrium gradients, no predictor–corrector, and on a uniform grid with a
lumped $`M`$ the streaming step reduces to the 9-point stencil
$`-\Delta t C - \frac{1}{2} \Delta t^2 K`$ (with $`K = 2 D`$ the directional
stiffness stencil) of
[`periodic_lw_fem.F90`](https://github.com/ivan-pi/dugks-mwe/blob/main/src/periodic_lw_fem.F90).

## Walls (couette, cavity)

Wall nodes are solved for like interior nodes; the wall velocity
$`\boldsymbol{u}_w`$ enters through the equilibrium only.

* `leelin`: the "no boundary condition" of Sec. 2.3 of the paper. The surface term
  $`S_\alpha f^n`$ is kept ((kept)) and
  $`f^{eq} = f^{eq}(\rho, \boldsymbol{u}_w)`$ at wall nodes.
* `bardow`: at wall nodes the collision relaxes the non-equilibrium part as
  everywhere else, but rebuilds the equilibrium part with the wall velocity, so
  that the post-collision momentum is exactly $`\rho \boldsymbol{u}_w`$ before
  streaming:

```math
g_\alpha^{*} = g_\alpha^{eq}(\rho, \boldsymbol{u}_w) + (1 - \omega) \left( g_\alpha - g_\alpha^{eq}(\rho, \boldsymbol{u}) \right)
```

* The surface term cannot be dropped (tested and rejected): that imposes
  $`(\boldsymbol{n} \cdot \boldsymbol{e}_\alpha)(\boldsymbol{e}_\alpha \cdot \nabla g_\alpha) = 0`$,
  which is inconsistent with a sheared profile (see the Couette results in [results.md](results.md#start-up-couette-flow-wall-boundary-test)).
  Kept as it is, its zeroth moment
  $`\frac{\Delta t^2}{2} \boldsymbol{n} \cdot (\nabla \cdot \Pi)`$,
  $`\Pi = \sum_\alpha \boldsymbol{e}_\alpha \boldsymbol{e}_\alpha g_\alpha`$, is a mass flux through
  the wall. The default for `bardow` ((mass flux removed)) therefore removes this
  moment, distributed over the moving populations with the lattice weights
  (no momentum is added):

```math
s_\alpha = (\boldsymbol{n} \cdot \boldsymbol{e}_\alpha)(\boldsymbol{e}_\alpha \cdot \nabla g_\alpha^{*}), \qquad
s_\alpha \leftarrow s_\alpha - \frac{w_\alpha}{1 - w_0} \sum_{\beta = 1}^{8} s_\beta .
```

* The two top corner nodes of the cavity belong to the stationary side walls.

The stream function is the $`L_2`$-best fit to the (weakly compressible) velocity
field with $`u = -\partial \psi / \partial y`$, $`v = \partial \psi / \partial x`$ (primary
vortex positive, as in Table I of Lee & Lin):

```math
\int_\Omega \nabla \varphi \cdot \nabla \psi d\Omega = \int_\Omega \left( \frac{\partial \varphi}{\partial x} v - \frac{\partial \varphi}{\partial y} u \right) d\Omega, \qquad \psi = 0 \quad \text{on} \quad \Gamma .
```

## Matrix-free realisation

Nothing is assembled. For both schemes the right-hand side is

```math
r_{\alpha,i} = \int_\Omega \varphi_i \left[ a (\boldsymbol{e}_\alpha \cdot \nabla f_\alpha) + b (\boldsymbol{e}_\alpha \cdot \nabla f_\alpha^{eq}) \right] d\Omega
 - \frac{\Delta t^2}{2} \int_\Omega (\boldsymbol{e}_\alpha \cdot \nabla \varphi_i) (\boldsymbol{e}_\alpha \cdot \nabla f_\alpha) d\Omega
 + \frac{\Delta t^2}{2} \oint_\Gamma \varphi_i (\boldsymbol{n} \cdot \boldsymbol{e}_\alpha) (\boldsymbol{e}_\alpha \cdot \nabla f_\alpha) d\Gamma
```

with $`a = -\Delta t + \frac{\Delta t^2}{2\lambda}`$, $`b = -\frac{\Delta t^2}{2\lambda}`$ for
Lee–Lin and $`a = -\Delta t`$, $`b = 0`$, $`f \rightarrow g^{*}`$ for Bardow.

| piece | how |
|---|---|
| $`r_\alpha`$ | one `MatrixFree::loop`; 8-component `FEEvaluation` / `FEFaceEvaluation` on a *scalar* `DoFHandler`, reading blocks 1…8 of a `LinearAlgebra::distributed::BlockVector` |
| $`M^{-1}`$ | one `SolverCG` per moving population on the scalar `MatrixFreeOperators::MassOperator` (the same $`M`$ for all $`\alpha`$), Jacobi preconditioner, starting vector extrapolated from the previous increments (see [results.md](results.md#starting-vector-of-the-mass-solves)); `--mass lumped` uses `compute_lumped_diagonal()` |
| collision, equilibria, moments | plain nodal loops |
| periodicity (tgv) | `add_periodicity` + `make_periodicity_constraints` |
| stream function (cavity) | `MatrixFreeOperators::LaplaceOperator` + CG on a second DoF index with $`\psi = 0`$ constraints |
| rest population | $`\boldsymbol{e}_0 = 0`$, hence $`r_0 = 0`$: no loop, no solve |
| memory layout | structure of arrays: `BlockVector` holds one contiguous `Vector` per population, so all nodal values of a direction are together; the cell loop streams eight such arrays, the nodal collision loop gathers across them |

## Streaming variants

The default streaming step is the second-order Taylor–Galerkin step above
(`--streaming tg2`) with a consistent mass matrix solved by CG (`--mass cg`).
The variants below change the left-hand side of the streaming step or the way
the mass matrix is inverted; the `tg3*` steps exist for `bardow` with
`--mass cg` only.

### Third-order Taylor–Galerkin streaming (`--streaming tg3`)

Keeping the $`\Delta t^3`$ term of the Taylor series along the characteristic and
eliminating $`(\boldsymbol{e}\cdot\nabla)^3`$ through the equation (Donea 1984) gives

```math
\Big(M + \tfrac{\Delta t^2}{6} K_\alpha\Big)(g_\alpha^{n+1} - g_\alpha^{*}) = -\Delta t\, C_\alpha g_\alpha^{*} - \tfrac{\Delta t^2}{2} K_\alpha g_\alpha^{*},

a direction-dependent SPD left-hand side, applied as a matrix-free values +
gradients kernel and solved by CG with the mass diagonal as preconditioner
(consistent mass only). $`K_\alpha = 2 D_\alpha`$ is the directional stiffness
matrix. Its symbol at the highest wavenumber is the reason it is stable up to
a Courant number of 1: see [analysis.md](analysis.md#why-tg3-is-stable-to-cfl-1-von-neumann-analysis).

### Dimensionally split TG3 streaming (`--streaming tg3-split`)

Since the shift by $`\boldsymbol{e}\Delta t`$ is the product of the shifts by
$`(e_x\Delta t, 0)`$ and $`(0, e_y\Delta t)`$, and on a tensor-product grid the
discrete 1-D operators commute, the streaming can be done as an $`x`$ sweep
followed by a $`y`$ sweep, each a 1-D TG3 step with $`K_{(1,0)}`$ or
$`K_{(0,1)}`$ (the diagonal populations get both sweeps, the axis ones one;
12 solves per step instead of 8). On a uniform lattice-aligned grid at
$`C = 1`$ the $`x`$ sweep operator is $`(M_x + K_x/6)\otimes M_y = I \otimes M_y`$
and the right-hand side carries the same $`M_y`$, so the update is the exact
one-node upwind shift: the scheme *is* the standard lattice Boltzmann method,
and the CG solves compute an identity the hard way. The formulation earns its
keep for $`C < 1`$, on stretched tensor grids (spatially varying Courant
number, still stable for $`\max C \le 1`$ and fourth-order in phase), and as
the finite-element analogue that degrades gracefully. On unstructured meshes
the two sweeps no longer commute exactly and the splitting adds an
$`O(\Delta t^2)`$ commutator error; the operators themselves
($`M + \tfrac{\Delta t^2}{6}K_{(1,0)}`$) are ordinary bilinear forms on any mesh.

### Iterated lumping instead of CG (`--mass richardson[k]`)

Donea's iterated lumping replaces the mass solve by $`k`$ lumped-preconditioned
Richardson passes, $`x_0 = M_L^{-1}r`$, $`x_{j+1} = x_j + M_L^{-1}(r - Mx_j)`$, i.e.
$`M^{-1} \approx M_L^{-1}\sum_{j\le k}(I - M M_L^{-1})^j`$: $`k`$ `vmult`s per
population and no reductions. `richardson` without a number is $`k = 2`$. Its accuracy and
stability are in [results.md](results.md#iterated-lumping-instead-of-cg---mass-richardsonk).

## General quadrilaterals (`--distort eps`)

`--distort eps` maps the periodic grid by
$`\boldsymbol{x} \to \boldsymbol{x} + \varepsilon L \sin(2\pi x/L)\sin(2\pi y/L)\,(1,1)`$:
every cell becomes a general (non-parallelogram) quadrilateral, `MatrixFree`
stores the Jacobians per quadrature point, the tensor structure is gone, and the
local cell size varies by $`\pm 2\pi\varepsilon`$ (so the local Courant number is up to
$`1/(1 - 2\pi\varepsilon)`$ times the nominal one). It exists to test the schemes
away from the lattice.

## Code structure

The numerics live in the header-only library `include/lbfem` (namespace
`lbfem`); the element degree is its compile-time constant `fe_degree`
(`-DLBFEM_DEGREE=p`). `apps/cgdbe/cgdbe.cc` is one driver built on it. Each header is a piece a driver can pull in on its own:

| header | contents |
|---|---|
| `d2q9.h` | velocities, weights, moments, equilibrium; its isotropy is checked at compile time |
| `test_case.h`, `test_cases.h` | the `TestCase` interface (periodicity, mesh transformation, wall nodes and velocities, initial populations, reference solution, time scale) and the three flows above |
| `discretization.h` | mesh, `FE_Q`, `MatrixFree`, mass operator, nodal coordinates and weights, the wall nodes |
| `collision.h` | `nodal_map` (a lambda mapped over the nodes, the wall nodes apart), BGK collision, Lee & Lin equilibria and predictor-corrector |
| `advection.h` | `AdvectionOperator`, the virtual interface a scheme streams through, and its two implementations (`TaylorGalerkinAdvection`, `LeeLinAdvection`), built from generic matrix-free loops that take the weak form at a quadrature point as a lambda |
| `mass.h` | CG, lumped and Richardson mass solves, the TG3 left-hand side |
| `streaming.h` | `TaylorGalerkin`: the increments A^{-1} r of one sweep or of the two split sweeps, the extrapolated CG start |
| `schemes.h` | `LeeLin` and `Bardow`, behind a `Scheme` interface |
| `io.h`, `timers.h` | stream function, checkpoints; per-stage timers |

A driver adds a flow by implementing `TestCase` and can exchange the advective
part by passing its own `AdvectionOperator` to a scheme. The virtual calls are
made once per sweep; the physics at a quadrature point or node is a lambda that
the compiler inlines into the loops. The nodal loop (`nodal_map`) is vectorized
across nodes; the wall nodes, $`O(\sqrt{N})`$, are computed apart, so that the
loop reads no wall data (measurements: [performance.md](performance.md)).
New drivers are added in `CMakeLists.txt` with `lbfem_add_driver(name
sources...)`; the library itself is the interface target `lbfem` (include path
and C++20), for projects that pull this repository in with `add_subdirectory`.

### Implementation notes

* **Sum factorization.** `FEEvaluation::evaluate/integrate` use the tensor-product
  structure of the $`Q_p`$ basis (1-D interpolation and differentiation matrices
  applied direction by direction), for every degree.
* **Geometry.** `MatrixFree` classifies each cell batch at `reinit`; the program
  prints the counts. All meshes here are axis-parallel (uniform for `tgv` and
  `couette`, tanh-stretched for `cavity`), so every batch is *cartesian*: one
  diagonal Jacobian per cell, $`J\!\times\!w`$ derived from it at run time — nothing
  per quadrature point is stored. General (curved) cells would store the inverse
  Jacobian and $`J\!\times\!w`$ per quadrature point.
* **Performance summary.** The end-of-run summary gives the stepping time,
  MNUPS ($`10^6`$ node updates per second, equal to MDoF/s, a DoF being one
  node with all 9 populations), and per stage — collision (all nodal work),
  advection (cell + face loop), mass solves — the time, its share of the
  stepping time, and an *effective bandwidth* from a single-pass traffic model
  (every vector read or written once per pass, no cache reuse; a vector pass
  is $`8 N_{nodes}`$ bytes): `bardow` collision 42 vector passes (collide in
  place 9+9, add the increment 8+8+8), advection 16 (read 8, write 8), lumped
  mass 16; `leelin` collision 53, advection 24 (it also reads $`f^{eq}`$); CG
  10 passes per iteration plus 4, Richardson 3 plus 7 per pass (with `--fused`
  8 and 5; the passes are deal.II's `PreconditionRelaxation`). The stage times
  are accumulated per MPI rank, without synchronizing the time loop, and the
  summary shows their maximum over the ranks.

  Next to it a **flop model** (multiply-add counted as 2; the per-node counts
  are in `print_summary` of the driver): `bardow` collision 157 flops per node
  (moments 20, equilibrium 102, relaxation 27, increment 8), `leelin` 303;
  advection 184 flops per cell and population for $`Q_1`$ on a cartesian cell
  (sum-factorised gradients 72, quadrature 24, integration 88), i.e. 1461 per
  node and step for 8 populations; mass vmult 68 per cell, CG 11 per node and
  iteration on top (21 with `--fused`: 7 reductions); lumped 1. What these
  models say about the kernels is discussed in
  [performance.md](performance.md).
* **Fused vector updates (`--fused`).** The vector updates and reductions of
  CG (deal.II's `SolverCG` detects the `vmult` of `StreamingMatrix` that takes
  operations on ranges of the vectors) and of the Richardson passes
  (`PreconditionRelaxation`) run inside the cell loop, on each range of
  entries just before the loop first touches it and just after it last does,
  with $`Ap`$ or $`Mx`$ still in cache. The Jacobi preconditioner of the fused
  CG offers `apply_to_subrange()` only (`RangeJacobi`): with `DiagonalMatrix`
  itself, `SolverCG` preconditions lane by lane, which is slower than the
  updates it fuses ([dealii-fused-cg.md](dealii-fused-cg.md)). Whether fusing
  pays off depends on the degree and the mesh size
  ([performance.md](performance.md)).
* **Higher-order elements.** `fe_degree` is a compile-time constant
  (`-DLBFEM_DEGREE=p`); nothing else changes: `FE_Q(p)` has Gauss–Lobatto
  nodes, the kernels use $`p + 1`$ Gauss points, wall nodes are found by
  coordinate, and $`\Delta t = c\, h_{min} / p^2`$ with $`c`$ = `--cfl`.

## Options and output

`cgdbe --help` lists the options with their defaults (`bardow`, `cg`, `tg2`;
`--streaming tg3*` requires `bardow` and `cg`). The wall treatment is fixed
per scheme (see [Walls](#walls-couette-cavity)); the CG solves start from the
extrapolation of the previous increments. Notes on some of the options:

* `--tend` is in units of $`t_{ref}`$: the velocity decay time $`1/(\nu(k_1^2+k_2^2))`$
  for `tgv`, the diffusion time $`L^2/\nu`$ for `couette`, the lid time $`L/U_0`$ for
  `cavity`.
* `--cfl` is $`\Delta t |e_x| / h_{min}`$.
* `--fused` runs the vector updates of CG and of the Richardson passes inside
  the cell loop of the mass operator (see [Implementation notes](#implementation-notes));
  off by default, since it only pays off for higher degrees on large meshes
  ([performance.md](performance.md)).
* A list of Reynolds numbers is run as a continuation, each starting from the
  previous steady state. The cavity stops when the mean velocity change per
  $`t_{ref}`$ drops below `--steady-tol` (default $`10^{-4} U_0`$). If that does not
  happen within `--tend`, the output is still written but `cgdbe` exits with status 2.
* `--checkpoint` writes the populations (one file per rank) at every diagnostic
  line, `--restart` reads them; same mesh, number of ranks and $`\Delta t`$.
* `--stretch` $`\gamma`$: nodes at
  $`x = \frac{1}{2} \left[ 1 + \tanh(\gamma (2 \xi - 1)) / \tanh \gamma \right]`$ for uniform $`\xi`$.

The benchmarks and the tools are built with `-DLBFEM_BUILD_EXTRAS=ON`; the
benchmarks need only a C++ compiler (`g++ -O3 -march=native benchmarks/roofline.cc`).
The cavity reference results are plotted from the repository root with

    gnuplot -e "dir='data/cavity'" scripts/plot_cavity.gp

The Python analysis needs `numpy`, `sympy` and `matplotlib`
(`pip install -r requirements.txt`) and can be run from any directory:

    python analysis/make_tikz.py                          # -> docs/figures/fem_stencils_*.tex
    python analysis/tg_spectral.py                        # -> tg_spectral_{1d,2d}.png
    python analysis/lbm_spectral.py                       # critical Courant numbers (slow)

Output: `vortex_{initial,final}.dat`, `vortex_history.dat` (tgv);
`cavity_Re<Re>.dat` with columns $`x`$, $`y`$, $`u/U_0`$, $`v/U_0`$, $`\psi/(U_0 L)`$,
$`\rho - \rho_0`$ and `cavity_Re<Re>_history.dat` (cavity).
