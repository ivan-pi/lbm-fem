# Von Neumann analyses

Linear stability and dispersion analyses of the streaming step and of the
coupled collision–streaming scheme, computed by the scripts in `analysis/`
(`pip install -r requirements.txt`; each script can be run from any
directory). They depend on the algorithm only, not on the machine. The
measured errors and stability limits they explain are in
[results.md](results.md).

| script | what it computes |
|---|---|
| `stencils.py` | element matrices and stencils of $`M`$, $`C_\alpha`$, $`D_\alpha`$ (sympy), checked against `tools/elem.cc` |
| `tg_spectral.py` | amplification factors of the TG2 and TG3 advection step (`tg_spectral_{1d,2d}.png`) |
| `lbm_spectral.py` | critical Courant numbers of the coupled D2Q9 scheme (slow) |
| `lbm_spectral2.py` | the same for the dimensionally split TG3 step |
| `richardson_spectral.py` | the same for the iterated-lumping mass solves |
| `active_flux_spectral.py` | 1-D active flux against $`Q_1`$ Taylor–Galerkin |
| `make_tikz.py` | the stencil figures in `figures/` |

## Why TG3 is stable to CFL 1: von Neumann analysis

`analysis/tg_spectral.py` (pure advection) and `analysis/lbm_spectral.py` (the
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
minimum at CFL 1 in the TG3 table of [results.md](results.md#third-order-taylorgalerkin-streaming---streaming-tg3) (`tg_spectral_1d.png`).

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
pure-advection ones. This reproduces every observation of [results.md](results.md): TG2 consistent stable at
0.45 and unstable at 0.5; TG3 stable at 1.0 for the short fixed-Ma runs
($`\omega \le 1.9`$) and slowly divergent at $`\Delta t/\lambda = 36`$ (limit 0.7);
lumped stable at 0.7 and unstable at 1.0.

## Dimensionally split TG3

`lbm_spectral2.py`: with the streaming done as an $`x`$ sweep followed by a
$`y`$ sweep ([method.md](method.md#dimensionally-split-tg3-streaming---streaming-tg3-split)),
the critical per-axis Courant number of the coupled scheme is 1.00 for *every*
$`\omega`$, and at $`C = 1`$ the diagonal amplification equals
$`e^{-i(\theta_x+\theta_y)}`$ to round-off: the exact lattice shift. A
regularised collision, by contrast, does not raise the limits of the unsplit
schemes: 0.50/0.85 for TG2/TG3 at $`\omega = 1.06`$.

## Iterated lumping

`richardson_spectral.py`: the 1-D symbol of $`k`$ Richardson passes is
$`\sum_{j \le k} (1-m)^j`$, i.e. 1, 5/3, 19/9, 2.41 at $`\theta = \pi`$ for
$`k = 0 \dots 3`$ against 3 for the consistent inverse; the phase error of
$`M^{-1}C`$ at $`\theta = \pi/4`$ is $`10^{-1}`$ (lumped), $`1.2\cdot10^{-2}`$,
$`3.2\cdot10^{-3}`$, $`2.4\cdot10^{-3}`$ against $`2.3\cdot10^{-3}`$ consistent,
so two passes give the consistent accuracy. The critical Courant numbers of
the coupled scheme and the leading dissipation of the whole family with
left-hand side $`M + \beta K_e`$ are discussed with the measurements in
[results.md](results.md#iterated-lumping-instead-of-cg---mass-richardsonk).

## Other continuous spaces: smooth B-splines (analysis only)

The obvious candidate beyond $`Q_p`$ is a $`C^{p-1}`$ B-spline space (isogeometric
analysis), which has one DoF per cell like $`Q_1`$ and far better dispersion. The
1-D Galerkin symbols on a uniform grid (same analysis as above):

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

## Active flux (analysis only)

Roe's active flux scheme keeps a cell average and shared point values at the
interfaces (2 DoF per cell in 1-D; vertex, edge-midpoint and average in 2-D),
reconstructs a continuous piecewise quadratic, evolves the point values *exactly*
along the characteristics and the averages conservatively with a Simpson flux
— for linear constant-coefficient advection, i.e. exactly our streaming step, and
without any global mass matrix. Von Neumann analysis of the 1-D scheme
(`analysis/active_flux_spectral.py`) against Q1 Taylor–Galerkin at equal DoF
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

