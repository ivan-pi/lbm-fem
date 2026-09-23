# Results of the numerical experiments

The tables on this page are the numerical experiments behind the design
choices described in [method.md](method.md). They were collected on a
virtual machine of a coding-assistant sandbox with the settings below, so the
timings, iteration counts and step times quoted next to the accuracy figures
are those of that machine at the time of the run; they are not representative
of the current code or of other computers. The accuracy figures depend only on
the algorithm and should reproduce anywhere. Every table names the `cgdbe`
options that produced it, so that it can be rerun.

Environment of all runs: Ubuntu package of deal.II 9.5.1 (no SIMD,
`VectorizedArray` width 1), GCC 13.3 `-O3 -march=native`, one core of a
2.1 GHz Xeon VM, $`Ma = 0.1`$ unless noted. The runs predate the split of
`cgdbe` into the `lbfem` library and the driver; the populations of the
current code agree with the ones of the runs to the CG tolerance
([performance.md](performance.md)).

The von Neumann analyses referred to below are in [analysis.md](analysis.md),
the performance measurements in [performance.md](performance.md).

## Taylor–Green vortex

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

## Taylor–Green vortex with $`k_1 \neq k_2`$ (modes 1,4)

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

## Third-order Taylor–Galerkin streaming (`--streaming tg3`)

TG3 (defined in [method.md](method.md#third-order-taylorgalerkin-streaming---streaming-tg3))
against the default TG2 step. `bardow`,
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
  coupling near $`\omega = 2`$, not of the advection step alone (see
  [analysis.md](analysis.md#why-tg3-is-stable-to-cfl-1-von-neumann-analysis)).

## Dimensionally split TG3 streaming (`--streaming tg3-split`)

What the analysis suggests: the consistent mass matrix, the diagonal directions
and the over-relaxed collision are the three ingredients of the instability, and
TG3 at $`C = 1`$ is exact in 1-D, which is what the split scheme
([method.md](method.md#dimensionally-split-tg3-streaming---streaming-tg3-split))
exploits. The linearised analysis (`analysis/lbm_spectral2.py`) gives a
critical per-axis Courant number of 1.00 for *every* $`\omega`$, and at
$`C = 1`$ the diagonal amplification equals $`e^{-i(\theta_x+\theta_y)}`$ to
round-off: the exact lattice shift. (A regularised collision, by contrast,
does not raise the limits: 0.50/0.85 for TG2/TG3 at $`\omega = 1.06`$.)

Same test as in the previous section (modes (1,4), $`Re = 100`$, $`t = 0.25\,t_{ref}`$), split TG3 at
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

On a uniform lattice-aligned grid at $`C = 1`$ the split scheme is the standard
lattice Boltzmann method computed the hard way ([method.md](method.md#dimensionally-split-tg3-streaming---streaming-tg3-split));
the general-quadrilateral runs below show what remains of its advantage away
from the lattice.

## Iterated lumping instead of CG (`--mass richardson[k]`)

Donea's iterated lumping ([method.md](method.md#iterated-lumping-instead-of-cg---mass-richardsonk))
approximates $`M^{-1}`$ by $`M_L^{-1}\sum_{j\le k}(I - M M_L^{-1})^j`$, with 1-D symbol
$`\sum_j (1-m)^j`$: 1, 5/3, 19/9, 2.41 at $`\theta = \pi`$ for $`k = 0 \dots 3`$ against
3 for the consistent inverse. Phase error of $`M^{-1}C`$ at $`\theta = \pi/4`$:
$`10^{-1}`$ (lumped), $`1.2\cdot10^{-2}`$, $`3.2\cdot10^{-3}`$, $`2.4\cdot10^{-3}`$, consistent
$`2.3\cdot10^{-3}`$ — two passes give the consistent accuracy. Critical per-axis
Courant number of the coupled scheme (`analysis/richardson_spectral.py`):

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

## General quadrilaterals (`--distort eps`)

The mapping of `--distort eps` is in [method.md](method.md#general-quadrilaterals---distort-eps);
the local cell size varies by $`\pm 2\pi\varepsilon`$. $`48^2`$, modes (1,4), $`Re = 100`$,
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

## Start-up Couette flow (wall boundary test)

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

## Lid-driven cavity

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

The streamline plot (`gnuplot -e "dir='data/cavity'" scripts/plot_cavity.gp`
from the repository root, which writes `cavity.png` from the reference results
in `data/cavity/`): `bardow`, Q2 elements ($`32^2`$ elements, the same $`65^2`$ nodes),
lumped mass; the contour levels of Fig. 5 of Lee & Lin (per Reynolds number:
0.11, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001 at $`Re = 400`$; 0.117 or 0.12,
0.11, 0.09, 0.06, 0.03, 0.01, 0.001 above), counter-rotating vortices in red at
$`-10^{-6} \dots -3 \cdot 10^{-3}`$. `cavity_ghia.png` (`gnuplot -e "levels='ghia'"
scripts/plot_cavity.gp`) uses the 24 levels of Ghia, Ghia & Shin (1982) instead. The Q1
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

## Starting vector of the mass solves

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

## Higher-order elements

Built with `-DLBFEM_DEGREE=p` ([method.md](method.md#implementation-notes));
the time step is $`\Delta t = c\, h_{min} / p^2`$ with $`c`$ = `--cfl`. Taylor–Green, `bardow`, $`Ma = 0.02`$, $`Re = 100`$,
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

