# cgdbe – characteristic Galerkin discrete Boltzmann, matrix-free (deal.II)

D2Q9, bilinear (Q1) elements, two test cases and two time-stepping schemes:

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
all. The matrices are tabulated as stencils in `stencils/` (sympy and deal.II
`FEValues` agree; on a uniform $`Q_1`$ grid they are the stencils of
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
$`\lambda`$ (see Results).

No equilibrium gradients, no predictor–corrector, and on a uniform grid with a
lumped $`M`$ the streaming step reduces to the 9-point stencil
$`-\Delta t C - \frac{1}{2} \Delta t^2 K`$ (with $`K = 2 D`$ the directional
stiffness stencil) of
[`periodic_lw_fem.F90`](https://github.com/ivan-pi/dugks-mwe/blob/main/src/periodic_lw_fem.F90).

### Code structure

`cgdbe.cc` is one file: the D2Q9 lattice, the parameters, the three test cases
behind a `TestCase` interface (periodicity, mesh transformation, wall nodes and
velocities, initial populations, reference solution, time scale, output stem —
the solver is case-agnostic and only asks whether the case has an exact
solution or is run to a steady state), the TG3 left-hand-side operator, and the
solver class with sections *collision* (`macroscopic`, `compute_equilibrium`,
`predictor_corrector`, `collide`), *advection operators* (`advection_leelin`,
`advection_bardow`, `wall_faces`), *streaming* (`apply_mass_inverse`, `stream`),
the two time steps, and diagnostics/I/O.

### Matrix-free realisation

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
| $`M^{-1}`$ | one `SolverCG` per moving population on the scalar `MatrixFreeOperators::MassOperator` (the same $`M`$ for all $`\alpha`$), Jacobi preconditioner, starting vector from previous steps (see below); `--mass lumped` uses `compute_lumped_diagonal()` |
| collision, equilibria, moments | plain nodal loops |
| periodicity (tgv) | `add_periodicity` + `make_periodicity_constraints` |
| stream function (cavity) | `MatrixFreeOperators::LaplaceOperator` + CG on a second DoF index with $`\psi = 0`$ constraints |
| rest population | $`\boldsymbol{e}_0 = 0`$, hence $`r_0 = 0`$: no loop, no solve |
| memory layout | structure of arrays: `BlockVector` holds one contiguous `Vector` per population, so all nodal values of a direction are together; the cell loop streams eight such arrays, the nodal collision loop gathers across them |

### Walls (couette, cavity)

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
  which is inconsistent with a sheared profile (see the Couette results below).
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

## Build and run

    cmake -S . -B build -DDEAL_II_DIR=/path/to/dealii     # deal.II >= 9.5, MPI + p4est
                                                          # -DCGDBE_DEGREE=2 for Q2 elements
    cmake --build build && cd build

    ./cgdbe                                               # tgv, 64^2, Re = 100, bardow, CG mass
    ./cgdbe --scheme leelin
    ./cgdbe --mass richardson2 --streaming tg3-split --cfl 1
    gnuplot ../plot_vortex.gp                             # -> vortex.png

    ./cgdbe --case couette                                # wall-boundary test
    ./cgdbe --case cavity --mass lumped --reynolds 400,1000,3200,5000
    gnuplot ../plot_cavity.gp                             # -> cavity.png (streamlines)

Options: `--case tgv|couette|cavity --scheme leelin|bardow --mass cg|lumped|richardson[k]
--streaming tg2|tg3|tg3-split --refine n | --cells N --mach Ma --modes n1,n2
--reynolds Re[,Re...] --cfl c --tend t --steps n --stretch gamma --distort eps
--steady-tol eps --cg-tol eps --checkpoint name --restart name --no-output`.
Defaults: `bardow`, `cg`, `tg2`. `--streaming tg3*` requires `bardow` and `cg`.
The wall treatment is fixed per scheme (see above); the CG solves start from the
extrapolation of the previous increments.

* `--tend` is in units of $`t_{ref}`$: the velocity decay time $`1/(\nu(k_1^2+k_2^2))`$
  for `tgv`, the diffusion time $`L^2/\nu`$ for `couette`, the lid time $`L/U_0`$ for
  `cavity`.
* `--cfl` is $`\Delta t |e_x| / h_{min}`$.
* A list of Reynolds numbers is run as a continuation, each starting from the
  previous steady state. The cavity stops when the mean velocity change per
  $`t_{ref}`$ drops below `--steady-tol` (default $`10^{-4} U_0`$).
* `--checkpoint` writes the populations (one file per rank) at every diagnostic
  line, `--restart` reads them; same mesh, number of ranks and $`\Delta t`$.
* `--stretch` $`\gamma`$: nodes at
  $`x = \frac{1}{2} \left[ 1 + \tanh(\gamma (2 \xi - 1)) / \tanh \gamma \right]`$ for uniform $`\xi`$.

Output: `vortex_{initial,final}.dat`, `vortex_history.dat` (tgv);
`cavity_Re<Re>.dat` with columns $`x`$, $`y`$, $`u/U_0`$, $`v/U_0`$, $`\psi/(U_0 L)`$,
$`\rho - \rho_0`$ and `cavity_Re<Re>_history.dat` (cavity).

## Results (this sandbox)

All runs: Ubuntu package of deal.II 9.5.1 (no SIMD, `VectorizedArray` width 1),
GCC 13.3 `-O3 -march=native`, one core of a 2.1 GHz Xeon VM, $`Ma = 0.1`$ unless
noted.

### Taylor–Green vortex

All numbers in this subsection are for the equal-wave-number vortex
(`--modes 1,1`); the unequal case follows in the next subsection.

Diffusive scaling as in Sec. 3.1 of Lee & Lin ($`Ma \propto h`$, $`Re = 20`$,
$`t = t_{ref}/2`$, CFL 0.25), relative $`L_2`$ velocity error:

| grid | $`Ma`$ | leelin, consistent | leelin, lumped | bardow, consistent | bardow, lumped |
|---|---|---|---|---|---|
| $`16^2`$ | 0.2  | 9.2e-3 | 3.4e-2 | 9.2e-3 | 3.4e-2 |
| $`32^2`$ | 0.1  | 3.2e-3 | 9.0e-3 | 3.2e-3 | 9.0e-3 |
| $`64^2`$ | 0.05 | 4.3e-4 | 2.2e-3 | 4.4e-4 | 2.1e-3 |

Both schemes are second order and practically indistinguishable here. Run with
`--modes 1,1` ($`64^2`$, $`Re = 100`$, $`\Delta t / \lambda = 2.26`$, `leelin`):
$`E/E_0 = 0.13513`$ against $`0.13534`$ exact, velocity error $`9.3 \cdot 10^{-4}`$.

Stability in $`\Delta t |e_x| / h`$ ($`32^2`$, $`Re = 100`$): `leelin` consistent
stable at 0.6; `bardow` consistent stable at 0.45, unstable at 0.5; `bardow`
lumped stable at 0.7, unstable at 1.0. At $`Re = 1000`$ the consistent-mass
`bardow` scheme is already unstable at 0.4 on the uniform grid. The error of the
lumped scheme grows quickly with the time step ($`1.6 \cdot 10^{-2}`$ at CFL 0.25,
$`5.0 \cdot 10^{-2}`$ at CFL 0.5 on $`32^2`$, $`Re = 100`$).

### Taylor–Green vortex with $`k_1 \neq k_2`$ (modes 1,4)

With $`n_2 = 4`$ the short mode has $`N/4`$ elements per wavelength, so $`16^2`$ and
$`32^2`$ are essentially unresolved. $`Re = 100`$, CFL 0.25, $`t = 0.25\,t_{ref}`$,
fixed $`Ma = 0.05`$ ($`\Delta t/\lambda`$ = 9.0, 4.5, 2.3, 1.1 for $`N`$ = 32 … 256),
relative $`L_2`$ velocity error and the order estimated from successive grids:

| $`N`$ | bardow, Q1 | order | leelin, Q1 | order | bardow, Q1 lumped | order | bardow, Q2 ($`N/2`$ elements) | order |
|---|---|---|---|---|---|---|---|---|
| 32  | 5.8e-2 | – | 6.4e-2 | – | 1.0e-1 | – | 1.7e-2 | – |
| 64  | 7.6e-3 | 2.9 | 7.8e-3 | 3.0 | 8.8e-3 | 3.6 | 2.5e-3 | 2.8 |
| 128 | 2.1e-3 | 1.9 | 2.0e-3 | 1.9 | 1.9e-3 | 2.2 | 1.8e-3 | 0.4 |
| 256 | 1.8e-3 | 0.2 | – | – | 1.9e-3 | 0.0 | 1.8e-3 | 0.0 |

* The two schemes give the same errors to within 3 %.
* All variants floor at $`1.8 \cdot 10^{-3}`$, which is the compressibility error of the
  weakly compressible solution at $`Ma = 0.05`$ ($`O(Ma^2)`$; the analytic vortex is
  incompressible). Q2 reaches it at $`33^2`$ nodes, Q1 at $`257^2`$. Above the floor
  the Q1 orders are ~2 (the 2.9–3.6 between $`32^2`$ and $`64^2`$ is the transition
  out of the unresolved regime, not an asymptotic order).
* To measure the spatial order without the floor one has to lower $`Ma`$, and the
  cost scales as $`1/Ma`$ in steps: an earlier attempt at $`Ma = 0.01`$ (and at
  $`Ma \propto 1/N`$ with $`\Delta t/\lambda = 90`$, i.e. $`\omega = 1.98`$) gave 8.5e-3
  at $`128^2`$ and $`t = 0.5\,t_{ref}`$, but $`256^2`$ would need 13 000 steps at
  0.16 s. The deeply over-relaxed regime is also where the excess numerical decay
  is largest ($`\nu_{num}/\nu \approx 0.5`$ at $`64^2`$, $`\Delta t/\lambda = 90`$).

A finer, non-doubling mesh sequence (`--cells N`) is cheaper and gives the order
from a least-squares fit. `bardow`, Q1, $`Re = 100`$, CFL 0.25, $`t = 0.25\,t_{ref}`$:

| $`N`$ | fixed $`Ma = 0.05`$ | local order | $`Ma = 1.6/N`$ ($`\Delta t/\lambda = 9`$) | local order |
|---|---|---|---|---|
| 24 | 1.41e-1 | – | 1.08e-1 | – |
| 32 | 5.82e-2 | 3.08 | 5.82e-2 | 2.16 |
| 40 | 2.97e-2 | 3.02 | 3.65e-2 | 2.09 |
| 48 | 1.73e-2 | 2.96 | 2.51e-2 | 2.05 |
| 64 | 7.58e-3 | 2.82 | 1.39e-2 | 2.04 |
| 80 | 4.24e-3 | 2.60 | 8.87e-3 | 2.03 |
| 96 | 2.89e-3 | 2.10 | 6.15e-3 | 2.01 |

Least-squares slopes: 2.98 over $`N = 24 \dots 64`$ at fixed $`Ma`$ (dropping to
2.1 as the floor is approached), 2.03 at constant $`\Delta t/\lambda`$. The
difference is the confounding of $`h`$ and $`\Delta t/\lambda`$ at fixed $`Ma`$: with
$`\Delta t \propto h`$ and $`\lambda`$ fixed, an error term $`\propto (\Delta t/\lambda) h^2`$
appears as $`h^3`$. Held at constant $`\Delta t/\lambda`$ the scheme is second
order in $`h`$, as expected for Q1 with the collision included (the fourth-order
result of Lee & Lin's appendix is for pure advection with the consistent mass
matrix). For Q2 at fixed $`Ma`$ the same fit over 12–20 elements per direction
gives 3.4, again above the floor only.

### Third-order Taylor–Galerkin streaming (`--streaming tg3`)

Keeping the $`\Delta t^3`$ term of the Taylor series along the characteristic and
eliminating $`(\boldsymbol{e}\cdot\nabla)^3`$ through the equation (Donea 1984) gives

```math
\Big(M + \tfrac{\Delta t^2}{6} K_\alpha\Big)(g_\alpha^{n+1} - g_\alpha^{*}) = -\Delta t\, C_\alpha g_\alpha^{*} - \tfrac{\Delta t^2}{2} K_\alpha g_\alpha^{*},
```

a direction-dependent SPD left-hand side (matrix-free values + gradients kernel,
CG with the mass diagonal as preconditioner; consistent mass only). `bardow`,
modes (1,4), $`Re = 100`$, $`t = 0.25\,t_{ref}`$, relative $`L_2`$ velocity error:

| $`N`$ | TG2, CFL 0.25 | TG3, CFL 0.25 | TG3, CFL 1.0 | steps TG2 / TG3@1 | CG its TG3@1 |
|---|---|---|---|---|---|
| 24 | 1.41e-1 | 1.49e-1 | 1.04e-1 | 124 / 31 | 5.1 |
| 32 | 5.82e-2 | 6.16e-2 | 5.00e-2 | 166 / 42 | 4.9 |
| 48 | 1.73e-2 | 1.80e-2 | 1.40e-2 | 248 / 62 | 3.8 |
| 64 | 7.58e-3 | 7.66e-3 | 6.76e-3 | 331 / 83 | 2.9 |
| 96 | 2.89e-3 | 2.83e-3 | 2.84e-3 | 496 / 124 | 2.1 |

(fixed $`Ma = 0.05`$; at constant $`\Delta t/\lambda = 9`$ the TG3/TG2 ratio is
0.94 at every $`N`$.)

* At the same CFL, TG3 changes nothing (a few per cent *worse*). The reason is
  in the Taylor series: the odd terms are dispersive, the even ones dissipative.
  TG3 adds the $`\Delta t^3`$ (dispersive) term; the dissipation error, the
  truncation of the $`\Delta t^4`$ term, is untouched. The excess decay of the
  vortex, $`\nu_{num}/\nu - 1`$ = 25 % at $`32^2`$ and 2.6 % at $`96^2`$ at
  $`\Delta t/\lambda = 9`$, is therefore *not* removed — which also settles that the
  error we measure is dissipative, and that a fourth-order (TG4) or, better, the
  exact shift is what would remove it.
* What TG3 does buy is stability: TG2 (consistent mass) blows up at CFL 0.7,
  TG3 runs at CFL 1.0 (Donea's limit), and its error is *smallest* there —
  1.40e-2 at CFL 1 against 3.0e-2 at 0.5 on $`48^2`$, the $`C^2(1 - C^2)`$
  signature of Lax–Wendroff dissipation. At CFL 1 it needs 4× fewer steps than
  TG2 at 0.25 for an equal or lower error, at about 2× the cost per step
  (2–5 CG iterations on the stiffer operator), i.e. a net factor of 2.
* Limit: deeply over-relaxed *and* CFL 1 is unstable — at $`\Delta t/\lambda = 36`$
  ($`Ma = 1.6/N`$) the $`64^2`$ and $`96^2`$ runs diverge after ~100 steps, while CFL
  0.95 is fine (1.44e-2). The instability is a slow one of the collision–streaming
  coupling near $`\omega = 2`$, not of the advection step alone (see the analysis
  below).

#### Why TG3 is stable to CFL 1: von Neumann analysis

`stencils/tg_spectral.py` (pure advection) and `stencils/lbm_spectral.py` (the
coupled scheme). For $`Q_1`$ on a uniform grid the 1-D symbols of the element
matrices are $`m(\theta) = (2 + \cos\theta)/3`$, $`c = i\sin\theta`$,
$`k = 2(1 - \cos\theta)`$, $`\theta = kh`$, and the amplification factors are

```math
G_{TG2} = 1 - \frac{C c + \tfrac12 C^2 k}{m}, \qquad
G_{TG3} = 1 - \frac{C c + \tfrac12 C^2 k}{m + \tfrac16 C^2 k}, \qquad C = \Delta t/h .
```

The consistent mass matrix is the culprit: its symbol drops to $`1/3`$ at the
odd–even mode $`\theta = \pi`$, so $`M^{-1}`$ *amplifies* the highest wavenumber by 3.
There $`G_{TG2}(\pi) = 1 - 6C^2`$, which gives Donea's limit $`C \le 1/\sqrt3`$. The
TG3 term is a wavenumber-dependent addition to the mass matrix,
$`m + C^2 k/6 = (2 + \cos\theta)/3 + C^2(1 - \cos\theta)/3`$: it restores inertia
exactly where the consistent mass lacks it, giving
$`G_{TG3}(\pi) = (1 - 4C^2)/(1 + 2C^2)`$ and the limit $`C \le 1`$. Two further
consequences: at $`C = 1`$ the TG3 left-hand side symbol is identically 1 — the
*lumped* mass matrix — so TG3 at $`C = 1`$ coincides with lumped Lax–Wendroff at
$`C = 1`$, which is the exact shift ($`|G - e^{-i\theta}| < 10^{-15}`$); and for
$`0 < C < 1`$ TG3 is a Courant-number-controlled selective lumping, consistent at
$`C \to 0`$ and lumped at $`C = 1`$. That is the spectral reason for the error
minimum at CFL 1 in the table above (`tg_spectral_1d.png`).

In 2-D with the tensor-product $`Q_1`$ symbols the diagonal directions
($`|\boldsymbol{e}| = \sqrt2`$) are the critical ones, and for pure advection they are
bad: per-axis limits TG2 0.20, TG3 0.35, lumped 0.60 (axis directions: 0.57, 1.0,
1.0). The collision rescues them. Linearising D2Q9 BGK about the rest state,
$`J_{\alpha\beta} = w_\alpha(1 + 3\boldsymbol{e}_\alpha\cdot\boldsymbol{e}_\beta)`$, the coupled
step is $`\hat g^{n+1} = S(\theta)\,[I - \omega(I - J)]\,\hat g^n`$ with
$`S = \mathrm{diag}(G_\alpha)`$; its spectral radius over all $`\theta`$ gives

| $`\Delta t/\lambda`$ | $`\omega`$ | TG2 consistent | TG3 consistent | TG2 lumped |
|---|---|---|---|---|
| 1 | 0.67 | 0.45 | 0.85 | 1.00 |
| 2.26 | 1.06 | 0.50 | 0.90 | 1.00 |
| 4.5 | 1.38 | 0.50 | 0.90 | 1.00 |
| 9 | 1.64 | 0.40 | 0.90 | 0.95 |
| 36 | 1.89 | 0.30 | 0.70 | 0.80 |
| 200 | 1.98 | 0.25 | 0.50 | 0.70 |
| pure advection | 0 | 0.20 | 0.35 | 0.60 |

(critical per-axis $`C`$ in steps of 0.05, $`\rho(G) \le 1 + 10^{-6}`$). The
non-equilibrium modes of the diagonal populations that pure streaming amplifies are
multiplied by $`1 - \omega`$ each step and redistributed through $`J`$, which is what
lifts the limits to 0.5 / 0.9 / 1.0 at moderate $`\omega`$; as $`\omega \to 2`$ the
damping $`|1 - \omega| \to 1`$ disappears and the limits fall back towards the
pure-advection ones. This reproduces every observation: TG2 consistent stable at
0.45 and unstable at 0.5; TG3 stable at 1.0 for the short fixed-Ma runs
($`\omega \le 1.9`$) and slowly divergent at $`\Delta t/\lambda = 36`$ (limit 0.7);
lumped stable at 0.7 and unstable at 1.0.

### Dimensionally split TG3 streaming (`--streaming tg3-split`)

What the analysis suggests: the consistent mass matrix, the diagonal directions
and the over-relaxed collision are the three ingredients of the instability, and
TG3 at $`C = 1`$ is exact in 1-D. Since the shift by $`\boldsymbol{e}\Delta t`$ is the
product of the shifts by $`(e_x\Delta t, 0)`$ and $`(0, e_y\Delta t)`$, and on a
tensor-product grid the discrete 1-D operators commute, the streaming can be done
as an $`x`$ sweep followed by a $`y`$ sweep, each a 1-D TG3 step with
$`K_{(1,0)}`$ or $`K_{(0,1)}`$ (the diagonal populations get both sweeps, the axis
ones one; 12 solves per step instead of 8). The linearised analysis
(`stencils/lbm_spectral2.py`) gives a critical per-axis Courant number of 1.00 for
*every* $`\omega`$, and at $`C = 1`$ the diagonal amplification equals
$`e^{-i(\theta_x+\theta_y)}`$ to round-off: the exact lattice shift. (A
regularised collision, by contrast, does not raise the limits: 0.50/0.85 for
TG2/TG3 at $`\omega = 1.06`$.)

Same test as above (modes (1,4), $`Re = 100`$, $`t = 0.25\,t_{ref}`$), split TG3 at
CFL 1 against TG2 at CFL 0.25:

| $`N`$ | split TG3 @ 1, Ma 0.05 | TG2 @ 0.25 | ratio | split TG3 @ 1, Ma = 1.6/N | TG2 @ 0.25 | ratio | steps |
|---|---|---|---|---|---|---|---|
| 24 | 3.26e-2 | 1.41e-1 | 4.3 | 5.38e-2 | 1.08e-1 | 2.0 | 31 / 124 |
| 32 | 2.15e-2 | 5.82e-2 | 2.7 | 2.15e-2 | 5.82e-2 | 2.7 | 42 / 166 |
| 48 | 5.75e-3 | 1.73e-2 | 3.0 | 5.58e-3 | 2.51e-2 | 4.5 | 62 / 248 |
| 64 | 3.56e-3 | 7.58e-3 | 2.1 | 3.43e-3 | 1.39e-2 | 4.1 | 83 / 331 |
| 96 | 2.16e-3 | 2.89e-3 | 1.3 | 1.28e-3 | 6.15e-3 | 4.8 | 124 / 496 |

Stable at $`\Delta t/\lambda = 36`$ throughout (the unsplit TG3 diverged there),
3–5× lower error above the $`Ma^2`$ floor, 4× fewer steps, at 4–7 CG iterations
on 12 systems per step (about 2.5× the cost of a TG2 step): a net factor of 1.5–2
in time and 3–5 in accuracy. The orders along the sequence are irregular
(1.4–3.3) because at $`C = 1`$ the streaming error is gone and what remains is
the lattice-LBM error of the under-resolved short mode.

Two honest remarks. On a uniform lattice-aligned grid at $`C = 1`$ the $`x`$ sweep
operator is $`(M_x + K_x/6)\otimes M_y = I \otimes M_y`$ and the right-hand side
carries the same $`M_y`$, so the update is the exact one-node upwind shift — the
scheme *is* the standard lattice Boltzmann method, and the CG solves compute an
identity the hard way; a tensor-grid implementation would skip them. The
formulation earns its keep for $`C < 1`$, on stretched tensor grids (spatially
varying Courant number, still stable for $`\max C \le 1`$ and fourth-order in
phase), and as the finite-element analogue that degrades gracefully. On
unstructured meshes the two sweeps no longer commute exactly and the splitting
adds an $`O(\Delta t^2)`$ commutator error; the operators themselves
($`M + \tfrac{\Delta t^2}{6}K_{(1,0)}`$) are ordinary bilinear forms on any mesh.

### Other continuous spaces: smooth B-splines (analysis only)

The obvious candidate beyond $`Q_p`$ is a $`C^{p-1}`$ B-spline space (isogeometric
analysis), which has one DoF per cell like $`Q_1`$ and far better dispersion. The
1-D Galerkin symbols on a uniform grid (`stencils/`, same analysis as above):

| space | mass stencil | $`m(\pi)`$ | phase error of $`M^{-1}C`$ at $`\pi/4`$, $`\pi/2`$ | CFL limit TG2 / TG3 |
|---|---|---|---|---|
| $`Q_1`$, $`C^0`$ | (1, 4, 1)/6 | 0.333 | 2.3e-3, 4.5e-2 | 0.57 / 1.00 |
| B-spline $`p = 2`$, $`C^1`$ | (1, 26, 66, 26, 1)/120 | 0.133 | 5.4e-5, 5.3e-3 | 0 / 0.05 |
| B-spline $`p = 3`$, $`C^2`$ | (1, 120, 1191, 2416, 1191, 120, 1)/5040 | 0.054 | 1.2e-6, 6.0e-4 | 0 / 0 |

Two orders of magnitude better phase accuracy — and *no* stable explicit
Taylor–Galerkin step at all: the smooth spaces under-weight the odd–even mode even
more than $`Q_1`$ ($`M^{-1}`$ amplifies it by 7.5 and 18.5), so the explicit update
is unstable for every Courant number, and the TG3 correction cannot rescue it.
Splines therefore need an implicit or filtered streaming step (or the
semi-Lagrangian one), and lumping them throws the dispersion advantage away. Not
tested in the code, and deal.II's matrix-free path has no spline space.

### Active flux (analysis only)

Roe's active flux scheme keeps a cell average and shared point values at the
interfaces (2 DoF per cell in 1-D; vertex, edge-midpoint and average in 2-D),
reconstructs a continuous piecewise quadratic, evolves the point values *exactly*
along the characteristics and the averages conservatively with a Simpson flux
— for linear constant-coefficient advection, i.e. exactly our streaming step, and
without any global mass matrix. Von Neumann analysis of the 1-D scheme
(`stencils/active_flux_spectral.py`) against Q1 Taylor–Galerkin at equal DoF
density, error of the physical mode per cell traversed:

| DoF per wavelength, $`C`$ | AF: dissipation / phase | Q1-TG2 | Q1-TG3 |
|---|---|---|---|
| 16, 0.25 | 3.1e-3 / 4.1e-4 | 2.1e-4 / 6.0e-4 | 2.4e-4 / 3.7e-5 |
| 16, 0.9 | 4.5e-4 / 8.2e-5 | 1.3e-3 / 8.1e-3 | 1.7e-4 / 2.2e-5 |
| 16, 1.0 | 0 / 0 | 2.0e-3 / 9.9e-3 | 0 / 0 |
| 32, 0.25 | 2.0e-4 / 1.4e-5 | 1.3e-5 / 7.8e-5 | 1.5e-5 / 1.1e-6 |
| 32, 0.9 | 3.0e-5 / 2.8e-6 | 8.0e-5 / 1.0e-3 | 1.1e-5 / 6.9e-7 |

Stable for $`C \le 1`$, exact at $`C = 1`$, no spurious growth at any $`C`$ (the
under-weighted odd–even mode of the Galerkin mass matrix does not exist); per DoF
its phase accuracy matches TG3 and its dissipation is larger at small Courant
number, decreasing towards $`C = 1`$. Its attraction here is structural, not
spectral: every update is local and explicit (no solve, no Richardson passes),
conservation is built in, and the 2-D version on quadrilaterals is third order
with 4 DoF per cell — a stencil-style code, not a Galerkin one. Open points for a
lattice Boltzmann use: the collision acts on point values and cell averages
alike, so the average of the post-collision populations needs a quadrature of
the nonlinear equilibrium over the reconstruction; and the 2-D stability in the
diagonal directions has to be established for the coupled scheme as was done
above for TG3.

### Iterated lumping instead of CG (`--mass richardson[k]`)

Donea's iterated lumping replaces the mass solve by $`k`$ lumped-preconditioned
Richardson passes, $`x_0 = M_L^{-1}r`$, $`x_{j+1} = x_j + M_L^{-1}(r - Mx_j)`$, i.e.
$`M^{-1} \approx M_L^{-1}\sum_{j\le k}(I - M M_L^{-1})^j`$, with 1-D symbol
$`\sum_j (1-m)^j`$: 1, 5/3, 19/9, 2.41 at $`\theta = \pi`$ for $`k = 0 \dots 3`$ against
3 for the consistent inverse. Phase error of $`M^{-1}C`$ at $`\theta = \pi/4`$:
$`10^{-1}`$ (lumped), $`1.2\cdot10^{-2}`$, $`3.2\cdot10^{-3}`$, $`2.4\cdot10^{-3}`$, consistent
$`2.3\cdot10^{-3}`$ — two passes give the consistent accuracy. Critical per-axis
Courant number of the coupled scheme (`stencils/richardson_spectral.py`):

| $`\Delta t/\lambda`$ | lumped | $`k=1`$ | $`k=2`$ | $`k=3`$ | consistent |
|---|---|---|---|---|---|
| 2.26 | 1.00 | 0.75 | 0.65 | 0.60 | 0.50 |
| 9 | 0.95 | 0.65 | 0.55 | 0.50 | 0.40 |
| 36 | 0.80 | 0.45 | 0.40 | 0.35 | 0.30 |

Measured ($`48^2`$, modes (1,4), $`Ma = 0.05`$, TG2):

| mass | CFL 0.25 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|
| lumped | 2.71e-2 | 6.26e-2 | 6.90e-2 | 7.00e-2 |
| Richardson 1 | 1.78e-2 | 2.16e-2 | 1.49e-2 | 2.59e-3 |
| Richardson 2 | 1.74e-2 | 1.95e-2 | 1.22e-2 | 3.83e-3 |
| Richardson 3 | 1.73e-2 | 1.94e-2 | 1.20e-2 | unstable |
| consistent (CG) | 1.73e-2 | 1.94e-2 | unstable | unstable |

Two passes reproduce the consistent-mass error to three digits at CFL 0.25 with
no solve (two `vmult`s per population, no reductions), and extend the stability
range. The striking $`2.6`$–$`3.8\cdot10^{-3}`$ at CFL 0.7 is real (it persists over
$`0.5\,t_{ref}`$) but is the zero crossing of the numerical viscosity, not a robust
operating point: the leading dissipation of the whole family with left-hand
side $`M + \beta K_e`$ is

```math
|G|^2 = 1 + \frac{C^2\,(3C^2 - 12\beta - 1)}{12}\,\theta^4 + O(\theta^6),
```

so the consistent scheme ($`\beta = 0`$) is dissipative below $`C = 1/\sqrt3`$, has zero
numerical viscosity exactly at its stability limit, and is anti-dissipative
beyond; the Richardson variants sit slightly to the lumped side and cross zero
near $`C \approx 0.65`$–$`0.7`$, where $`\nu_{num}/\nu`$ passes through 1 (measured
1.078 → 1.023 → 0.989 → 0.366 for $`k = 2`$ at CFL 0.5, 0.65, 0.7, 0.75) and the
scheme blows up just above. TG3 ($`\beta = C^2/6`$) gives
$`C^2(C^2-1)/12`$: dissipative for all $`C < 1`$, exact at 1; the lumped matrix
($`\beta = 1/6`$) three times as dissipative. The zero-dissipation choice
$`\beta^* = C^2/4 - 1/12`$ is stable only for $`C \le 1/\sqrt3`$ (its symbol at
$`\theta = \pi`$ still under-weights the odd–even mode), so within this family there
is no scheme that is both dissipation-free and stable beyond $`1/\sqrt3`$ except
TG3 at the single point $`C = 1`$.

### General quadrilaterals (`--distort eps`)

`--distort eps` maps the periodic grid by
$`\boldsymbol{x} \to \boldsymbol{x} + \varepsilon L \sin(2\pi x/L)\sin(2\pi y/L)\,(1,1)`$:
every cell becomes a general (non-parallelogram) quadrilateral, MatrixFree
stores the Jacobians per quadrature point, the tensor structure is gone, and the
local cell size varies by $`\pm 2\pi\varepsilon`$ (so the local Courant number is up to
$`1/(1 - 2\pi\varepsilon)`$ times the nominal one). $`48^2`$, modes (1,4), $`Re = 100`$,
$`Ma = 0.05`$, $`t = 0.25\,t_{ref}`$, relative $`L_2`$ velocity error:

| scheme, nominal CFL | $`\varepsilon = 0`$ | 0.02 (±13 %) | 0.04 (±25 %) |
|---|---|---|---|
| TG2, 0.25 | 1.73e-2 | 1.81e-2 | 2.07e-2 |
| TG2, 0.45 | 2.11e-2 | 2.28e-2 | unstable |
| TG3, 0.5 | 3.01e-2 | 3.17e-2 | 3.68e-2 |
| TG3, 1.0 | 1.40e-2 | 2.01e-2 | 3.38e-2 |
| split TG3, 0.5 | 2.90e-2 | 3.04e-2 | 3.49e-2 |
| split TG3, 1.0 | 5.75e-3 | 1.23e-2 | 2.47e-2 |

* Stability follows the *local* Courant number: TG2 at nominal 0.45 fails once
  the smallest cells see $`C \approx 0.6`$. TG3 and the split scheme stay stable at
  nominal 1.0 with local $`C`$ up to about 1.3, i.e. the collision-provided
  margin of the coupled analysis carries over.
* The exactness at $`C = 1`$ is a lattice property and goes away with the
  lattice: the split scheme's advantage shrinks from 3× (uniform) to 1.7× at
  ±13 % and to parity at ±25 %, where all variants sit at 2–4e-2. Per unit
  cost it is still ahead (62 steps at ~2.5× against 248), but only just.
* The splitting commutator error is small: split and unsplit TG3 agree to
  within 5 % at every distortion.

### Start-up Couette flow (wall boundary test)

16 elements across the channel, $`Re = 10`$, relative $`L_2`$ error of the steady
linear profile (at $`t = 2 L^2/\nu`$) and of the transient (at $`t = 0.2 L^2/\nu`$):

| scheme, mass | surface term | steady | transient |
|---|---|---|---|
| leelin, consistent | 1 | 2e-10 | 7.7e-4 |
| leelin, lumped | 1 | 2e-10 | 1.1e-3 |
| bardow, consistent | 0 (dropped) | 1.5e-2 | 2.3e-2 |
| bardow, lumped | 0 (dropped) | 1.8e-3 | 3.1e-3 |
| bardow, consistent | 1 or 2 | 4e-10 | 1.2e-3 |
| bardow, lumped | 1 or 2 | 2e-10 | 9.1e-4 |

Dropping the surface term over-drives the flow; with it, both schemes reproduce
the steady shear profile to solver tolerance.

### Lid-driven cavity

$`65 \times 65`$ nodes, $`\gamma = 1.2`$ ($`h_{min} = 0.0071`$, $`h \approx 0.0225`$ in
the core), CFL 0.4 on $`h_{min}`$, steady-state tolerance $`10^{-4}`$. Primary,
lower-left and lower-right vortex: $`\psi / (U_0 L)`$ at the nodal extremum.
"Ghia" and "Lee & Lin" are the values of Table I of the paper.

| $`Re`$ | case | primary | lower left | lower right | primary location |
|---|---|---|---|---|---|
| 400 | Ghia | 0.1139 | -1.42e-5 | -6.42e-4 | (0.5547, 0.6055) |
| 400 | Lee & Lin | 0.1158 | -1.32e-5 | -6.82e-4 | (0.5516, 0.6024) |
| 400 | bardow, Q1, consistent | 0.1148 | -1.02e-5 | -6.20e-4 | (0.5449, 0.6112) |
| 400 | bardow, Q2, lumped | 0.1138 | -1.38e-5 | -6.43e-4 | (0.5449, 0.6110) |
| 1000 | Ghia | 0.1179 | -2.31e-4 | -1.75e-3 | (0.5313, 0.5625) |
| 1000 | Lee & Lin | 0.1204 | -2.26e-4 | -1.76e-3 | (0.5259, 0.5771) |
| 1000 | leelin, Q1, consistent | 0.1232 | -2.20e-4 | -1.69e-3 | (0.5225, 0.5672) |
| 1000 | bardow, Q1, lumped | 0.1247 | -2.10e-4 | -1.68e-3 | (0.5225, 0.5672) |
| 1000 | bardow, Q2, lumped | 0.1195 | -2.31e-4 | -1.74e-3 | (0.5224, 0.5671) |
| 3200 | Ghia | 0.1204 | -9.78e-4 | -3.14e-3 | (0.5165, 0.5469) |
| 3200 | Lee & Lin | 0.1206 | -1.16e-3 | -3.02e-3 | (0.5259, 0.5516) |
| 3200 | bardow, Q2, lumped | 0.1221 | -1.11e-3 | -2.94e-3 | (0.5224, 0.5449) |
| 5000 | Ghia | 0.1190 | -1.36e-3 | -3.08e-3 | (0.5117, 0.5352) |
| 5000 | Lee & Lin | 0.1217 | -1.49e-3 | -3.47e-3 | (0.5259, 0.5516) |
| 5000 | bardow, Q2, lumped (*) | 0.1224 | -1.35e-3 | -3.21e-3 | (0.5224, 0.5449) |

(*) $`Re = 5000`$ was run at CFL 0.2 (at CFL 0.4 it did not settle) and needed
about $`90 L/U_0`$ after the continuation from $`Re = 1000`$ to reach the
steady-state tolerance.

![streamlines](cavity.png)

`cavity.png`: `bardow`, Q2 elements ($`32^2`$ elements, the same $`65^2`$ nodes),
lumped mass; the contour levels of Fig. 5 of Lee & Lin (per Reynolds number:
0.11, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001 at $`Re = 400`$; 0.117 or 0.12,
0.11, 0.09, 0.06, 0.03, 0.01, 0.001 above), counter-rotating vortices in red at
$`-10^{-6} \dots -3 \cdot 10^{-3}`$. `cavity_ghia.png` (`gnuplot -e "levels='ghia'"
plot_cavity.gp`) uses the 24 levels of Ghia, Ghia & Shin (1982) instead. The Q1
rows of the table use $`64^2`$ elements.

* With Q1 elements both schemes overshoot the primary vortex at $`Re = 1000`$ by
  4–6 % (the paper reports 2 %); they agree with each other (domain mean of
  $`|\boldsymbol{u}|^2/U_0^2`$: 0.0948 `leelin`, 0.0954 `bardow`), so this is
  spatial resolution, not time stepping. With Q2 elements on the same nodes
  the error drops to 1.3 % for the primary and below 1 % for the secondary
  vortices, and the kinetic energy $`\frac{1}{2} \int |\boldsymbol{u}|^2 = 0.0447`$
  matches the literature value 0.0445.
* Cost per $`L/U_0`$ at $`65^2`$ nodes, lumped `bardow`: Q1 6100 steps at 2.4 ms;
  Q2 11900 steps at 0.95 ms.
* Mass: `bardow` with (mass flux removed) conserves the mean density to
  $`10^{-8}`$ or better (round-off with the lumped mass, CG tolerance otherwise);
  `leelin` drifts by about $`-2 \cdot 10^{-7}`$ per $`L/U_0`$. With the full surface
  term ((kept)) the `bardow` scheme gained 3.2 % mass at $`Re = 400`$;
  with moving top corners `leelin` lost $`3 \cdot 10^{-4}`$ per $`L/U_0`$.
* $`Re = 3200`$ with Q1 elements: the lumped `bardow` scheme ($`\omega = 1.93`$)
  does not settle on this mesh (the kinetic energy keeps growing past physical
  values), and `leelin` with consistent mass became very slow (about 5 minutes
  per $`L/U_0`$ here). With Q2 elements the lumped `bardow` scheme reaches the
  steady state at $`Re = 3200`$.
* Cost near steady state at $`65^2`$ nodes: `bardow` lumped 2.4 ms/step, `bardow`
  consistent 6.5 ms/step (about one CG iteration per solve), `leelin` consistent
  9.2 ms/step (1.6 iterations). One $`L/U_0`$ is about 6100 steps.

### Starting vector of the mass solves

The unknown of each solve is the increment $`x^n = M^{-1} r^n`$, which varies
slowly in time. Four starting vectors were compared (the code now uses the
extrapolation; the others were removed after this study): zero; the previous
increment $`x^{n-1}`$; the extrapolation $`2 x^{n-1} - x^{n-2}`$; the explicit
lumped predictor $`M_L^{-1} r^n`$.
CG iterations per solve at tolerance $`10^{-8} |r|`$, cavity, $`65^2`$ nodes, `bardow`:

| guess | start-up transient (600 steps from rest) | near steady state ($`Re = 1000`$) |
|---|---|---|
| zero | 25.3 | 24.9 |
| lumped | 24.3 | – |
| previous | 13.2 | 3.5 |
| extrapolate | 10.1 | 1.7 |

`previous` costs nothing (the vector is simply left in place), `extrapolate` one
`sadd` (DAXPY-type) per population, `lumped` one pointwise product with the
precomputed inverse row sums of $`M`$ (no solve involved: it is the explicit
lumped-mass update, used as a predictor for the consistent one). The lumped
predictor is worth about one iteration only: $`M_L^{-1} M`$ has eigenvalues down
to $`1/9`$ for Q1, so its error is $`O(1)`$ in the high wavenumbers, whereas the
time history is accurate to $`O(\Delta t)`$ or $`O(\Delta t^2)`$ in all of them.
`zero` and `lumped` are kept for testing only. Near steady state the step time
drops from 11.9 to 8.4 ms (`previous` to `extrapolate`). Note that only the CG
starting vector is extrapolated; unlike the $`f^{eq}`$ extrapolation of Mei &
Shyy (1998) this does not touch the scheme. On the Taylor–Green vortex the
choice hardly matters (2.4–3.5 iterations including `zero`), because
the right-hand side consists of very few eigenvectors of $`M`$.

### Higher-order elements

`fe_degree` is a compile-time constant (`-DCGDBE_DEGREE=p`); nothing else
changes: `FE_Q(p)` has Gauss–Lobatto nodes, the kernels use $`p + 1`$ Gauss points,
wall nodes are found by coordinate, and $`\Delta t = c h_{min} / p^2`$ with
$`c`$ = `--cfl`. Taylor–Green, `bardow`, $`Ma = 0.02`$, $`Re = 100`$,
$`t = 0.05 t_{ref}`$, $`c = 0.25`$, relative $`L_2`$ velocity error:

| elements | Q1 | Q2 | Q3 |
|---|---|---|---|
| $`8^2`$  | 2.1e-1 | 6.8e-3 | 4.4e-4 |
| $`16^2`$ | 2.7e-2 | 1.1e-3 | 3.9e-5 |
| $`32^2`$ | 3.3e-3 | 1.2e-4 | – |

Work–precision (wall time of the same runs, consistent mass, one core, no SIMD):

| error | Q1 | Q2 | Q3 |
|---|---|---|---|
| $`10^{-2}`$ | 0.29 s ($`16^2`$, 2.7e-2) | 0.28 s ($`8^2`$, 6.8e-3) | – |
| $`10^{-3}`$ | 2.0 s ($`32^2`$, 3.3e-3) | 1.2 s ($`16^2`$, 1.1e-3) | 0.62 s ($`8^2`$, 4.4e-4) |
| $`10^{-4}`$ | (~30 s extrapolated) | 8.1 s ($`32^2`$, 1.2e-4) | 3.1 s ($`16^2`$, 3.9e-5) |

The time step scales as $`h/p^2`$, so a higher degree buys accuracy per DoF at the
price of more steps; at a given error the higher degree still wins here.
* The row-sum lumped mass on the Gauss–Lobatto nodes costs almost no accuracy
  for $`p \geq 2`$ (Q2 $`16^2`$: 1.09e-3 lumped vs 1.06e-3 consistent; Q3:
  3.8e-5 vs 3.9e-5), in contrast to Q1. Higher order with `--mass lumped` therefore
  removes the mass solves altogether.
* Stability, consistent mass, in units of $`c`$: Q2 stable at 0.6, unstable at
  0.8; Q3 stable at 0.8.
* Walls: the steady Couette profile is reproduced to $`10^{-11}`$ with Q2 and Q3.

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
* **Profiling.** The end-of-run summary gives the stepping time, MNUPS (equal to
  MDoF/s, a DoF being one node with all 9 populations), and per stage — collision
  (all nodal work), advection (cell + face loop), mass solves — the time, its share
  of the stepping time, and an *effective bandwidth* from a single-pass traffic
  model (every vector read or written once per pass, no cache reuse): `bardow`
  collision 42 vector passes (collide in place 9+9, add the increment 8+8+8),
  advection 16 (read 8, write 8), lumped mass 16; `leelin` collision 53, advection
  24 (it also reads $`f^{eq}`$); CG 10 passes per iteration plus 4. A vector pass is
  $`8 N_{nodes}`$ bytes. On this VM a numpy copy runs at 14 GB/s, a triad at 8 GB/s.

  Next to it a **flop model** (multiply-add counted as 2; the per-node counts are
  in the source next to `flops_collision_bardow` etc.): `bardow` collision 148 flops
  per node (moments 20, equilibrium 102, relaxation 18, increment 8), `leelin`
  303; advection 184 flops per cell and population for $`Q_1`$ on a cartesian cell
  (sum-factorised gradients 72, quadrature 24, integration 88), i.e. 1461 per
  node and step for 8 populations; mass vmult 68 per cell, CG 11 per node and
  iteration on top; lumped 1.

  The roofline itself is measured, not assumed (`stencils/roofline.cc`, one core):

  | | working set of $`256^2`$ (0.5 MB/array, in L3) | DRAM-sized (32 MB/array) |
  |---|---|---|
  | scalar FMA peak, 8 independent chains | 10.1 GFlop/s | – |
  | STREAM-like, 9 × (in → out) | 27 GB/s | 11 GB/s |
  | collision pattern, 9 arrays in, 9 out (gather per node) | 20 GB/s | 12 GB/s |
  | collision pattern, in place (as `bardow`) | 31 GB/s | 19 GB/s |

  (The L3 of this VM is reported as 260 MB, so all cgdbe runs shown here are
  cache-resident; the DRAM column is what a $`2000^2`$ mesh would see. A numpy copy
  gives 14 GB/s, i.e. the DRAM figure.)

  $`256^2`$ elements, 50 steps, one core, SIMD width 1:

  | scheme, mass | stage | time | share | GB/s | GFlop/s | flop/byte |
  |---|---|---|---|---|---|---|
  | bardow, lumped | collision | 0.09 s | 5.0 % | 12.2 | 5.4 | 0.44 |
  | | advection | 1.72 s | 93.5 % | 0.25 | 2.8 | 11.4 |
  | | mass (lumped) | 0.03 s | 1.5 % | 15.2 | 1.0 | 0.06 |
  | bardow, consistent | collision | 0.10 s | 1.8 % | 10.8 | 4.8 | 0.44 |
  | | advection | 1.74 s | 30.5 % | 0.24 | 2.8 | 11.4 |
  | | mass (CG, 1.3 it.) | 3.85 s | 67.6 % | 0.90 | 0.67 | 0.75 |
  | leelin, consistent | collision | 0.15 s | 2.2 % | 9.3 | 6.7 | 0.71 |
  | | advection | 2.74 s | 40.1 % | 0.23 | 2.5 | 10.7 |
  | | mass (CG, 1.3 it.) | 3.93 s | 57.6 % | 0.89 | 0.67 | 0.75 |

  Reading the two columns against the measured roofline: the in-cache ridge point
  is 10.1 GFlop/s / 31 GB/s = 0.33 flop/byte, so the `bardow` collision (0.44
  flop/byte in the single-pass model, 1.0 counted in place) is on the compute
  side; it reaches 5.4 GFlop/s, 54 % of scalar peak, with the two divisions of
  the moments and the dependency chains of the equilibrium accounting for the
  rest. Its 12 GB/s is 40 % of the in-place pattern roofline. Vectorising the
  nodal loop across nodes (AVX-512, 8 lanes) would raise the ceiling to about
  80 GFlop/s and make the collision bandwidth-bound at ≈ 31 GB/s / 144 B =
  0.2 GNUPS. The advection loop has an intensity of 11 flop/byte — far on the
  compute side — and still reaches only 2.8 GFlop/s, 28 % of scalar peak, which
  confirms that its cost is instruction overhead around the arithmetic (see
  below), not flops and not bytes.

  The 11 flop/byte itself is an artefact of the element-by-element formulation,
  not a property of the operator. Per population and cell the model counts 184
  flops: sum-factorised gradient at the 4 Gauss points (2 directions × 2 stages
  × 8 multiply-adds) 64, inverse-Jacobian scaling 8, quadrature 24, integration
  of the values 16 and of the gradients 64 + 8. The explicit $`4 \times 4`$ element
  matrix would do the same action in 32 flops, and the assembled 9-point stencil
  in 17 per node — the operator applied once per node instead of once per cell
  with each node touched by four cells. So the numerator carries a factor of
  about 10 of redundant arithmetic, while the denominator counts the minimal
  traffic of one read and one write of the eight arrays; within the cache the
  loop actually moves four reads and four read-modify-writes per cell and
  population, i.e. its in-cache intensity is 184/96 ≈ 2 flop/byte. The
  sum-factorised evaluation costs $`O(p)`$ per DoF and direction against
  $`O(p^d)`$ for the element matrix, which is why the matrix-free action pays off
  for $`p \gtrsim 3`$ and loses for $`Q_1`$. The CG mass solves are as slow in both
  measures because each `vmult` is a cell loop with the same overhead per cell as
  the advection loop but only 68 useful flops per cell. The stencil reference
  (`stencil_bench.cc`) does 17 flops per node and direction, 3.8 GFlop/s at 3.5 GB/s:
  the same discretisation at one tenth of the model flops and 15× the speed.

  The nodal loops run at the memory bandwidth of the machine. The advection loop
  does not: 0.54 µs per cell for 8 populations, about 70 ns per population per
  cell. It is not a memory problem — the whole working set ($`f`$ and $`r`$, 8.5 MB
  at $`256^2`$) sits in cache, and the `distribute_local_to_global` scatter is a
  read-modify-write of four entries per cell whichever way the loop is organised.
  Two experiments pin it down:

  * `--per-direction` runs eight scalar cell loops (one destination array at a
    time, contiguous scatter) instead of the single 8-component loop: 1.5× *slower*
    (2.69 vs 1.75 s for 50 steps), because the per-cell work of `reinit`, index
    lookup and constraint handling is then paid eight times. The 8-component loop
    already amortises it.
  * `stencils/stencil_bench.cc` applies the *same* discretisation ($`Q_1`$, lumped
    mass, uniform periodic grid) as the assembled 9-point stencil, direction by
    direction, structure of arrays, no FEM machinery: 5 ns per cell and direction,
    2.4 ms per step at $`256^2`$, 28 MNUPS — 15× the deal.II loop, on the same core
    and compiler.

  A `gprof` build shows where the instructions go: `read_dof_values` and
  `distribute_local_to_global` walk the DoFs one by one through the index map
  (42 million `local_element` calls for 20 steps), with a vector-compatibility
  check per component and cell; the sum-factorised arithmetic of a $`Q_1`$ cell is
  a small part. This is the known trade-off of `FEEvaluation`: its gather/scatter
  cost per DoF is meant to be amortised over vectorised cell batches and over the
  arithmetic of higher degrees, and this build has neither (width-1
  `VectorizedArray`, $`p = 1`$). Accordingly $`Q_2`$ already does 4.4 MNUPS against
  1.8 for $`Q_1`$ at the same node count. With AVX-512 the loop processes 8 cells per
  instruction stream and the index work is shared by the batch; that, and
  $`p \geq 2`$, is where the factor to gain is. On a structured grid the stencil
  form remains unbeatable, which is the point of the `dugks-mwe` code.

### Throughput (Taylor–Green, `run_perf.sh`, 200 steps, `leelin`, modes 1,1)

MNUPS = $`10^6`$ node updates (all 9 populations) per second. These numbers were
taken before the wall-face loop was added.

| mass | grid | nodes | ms/step | MNUPS | CG its/solve |
|---|---|---|---|---|---|
| consistent | $`32^2`$  | 1 089  | 3.47   | 0.31 | 3.6 |
| consistent | $`64^2`$  | 4 225  | 10.96  | 0.39 | 2.5 |
| consistent | $`128^2`$ | 16 641 | 39.05  | 0.43 | 2.0 |
| consistent | $`256^2`$ | 66 049 | 158.21 | 0.42 | 2.0 |
| lumped     | $`32^2`$  | 1 089  | 0.90   | 1.21 | – |
| lumped     | $`64^2`$  | 4 225  | 3.63   | 1.17 | – |
| lumped     | $`128^2`$ | 16 641 | 14.86  | 1.12 | – |
| lumped     | $`256^2`$ | 66 049 | 59.29  | 1.11 | – |

At $`256^2`$ (consistent): advection loop 35 %, mass solves 63 %, nodal loops 2 %.
The `bardow` scheme needs one instead of two gradient evaluations per population
in the advection loop (about 0.6 of the `leelin` loop time). Obvious next steps:
a SIMD-enabled deal.II build; a single block CG on an 8-component mass kernel.
