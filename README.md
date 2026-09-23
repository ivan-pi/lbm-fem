# lbm-fem – characteristic Galerkin discrete Boltzmann, matrix-free (deal.II)

[![CI](https://github.com/ivan-pi/lbm-fem/actions/workflows/ci.yml/badge.svg)](https://github.com/ivan-pi/lbm-fem/actions/workflows/ci.yml)

A finite-element lattice Boltzmann solver: the D2Q9 discrete Boltzmann
equation on continuous $`Q_p`$ elements (bilinear by default), streamed by a
matrix-free Taylor–Galerkin (characteristic Galerkin) step on deal.II's
`MatrixFree` infrastructure and collided in a nodal loop. Nothing is
assembled. Two time-stepping schemes from the literature are implemented,
the predictor–corrector of Lee & Lin and the collide-then-stream scheme of
Bardow, Karlin & Gusev, and three flows exercise them: a periodic
Taylor–Green vortex, start-up Couette flow and the lid-driven cavity.

The method, the code and the driver options are described in
[docs/method.md](docs/method.md); the numerical experiments, the stability
analyses and the performance measurements are in the other pages of
[docs/](docs/).

## References

- T. Lee, C.-L. Lin, *A characteristic Galerkin method for discrete Boltzmann
  equation*, J. Comput. Phys. **171**(1) (2001) 336–356.
  [doi:10.1006/jcph.2001.6791](https://doi.org/10.1006/jcph.2001.6791)
- T. Lee, C.-L. Lin, *An Eulerian description of the streaming process in
  the lattice Boltzmann equation*, J. Comput. Phys. **185**(2) (2003) 445–471.
  [doi:10.1016/S0021-9991(02)00065-7](https://doi.org/10.1016/S0021-9991(02)00065-7)
- A. Bardow, I. V. Karlin, A. A. Gusev, *General characteristic-based
  algorithm for off-lattice Boltzmann simulations*, EPL (Europhysics Letters)
  **75**(3) (2006) 434–440.
  [doi:10.1209/epl/i2006-10138-1](https://doi.org/10.1209/epl/i2006-10138-1)
- J. Donea, *A Taylor–Galerkin method for convective transport problems*,
  Int. J. Numer. Methods Eng. **20**(1) (1984) 101–119.
  [doi:10.1002/nme.1620200108](https://doi.org/10.1002/nme.1620200108)
- Z. Guo, T. S. Zhao, *Explicit finite-difference lattice Boltzmann method
  for curvilinear coordinates*, Phys. Rev. E **67**(6) (2003) 066709.
  [doi:10.1103/PhysRevE.67.066709](https://doi.org/10.1103/PhysRevE.67.066709)
- R. Mei, W. Shyy, *On the finite difference-based lattice Boltzmann method
  in curvilinear coordinates*, J. Comput. Phys. **143**(2) (1998) 426–448.
  [doi:10.1006/jcph.1998.5984](https://doi.org/10.1006/jcph.1998.5984)
- U. Ghia, K. N. Ghia, C. T. Shin, *High-Re solutions for incompressible flow
  using the Navier–Stokes equations and a multigrid method*, J. Comput. Phys.
  **48**(3) (1982) 387–411.
  [doi:10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4)
- D. Arndt, W. Bangerth, D. Davydov, T. Heister, L. Heltai, M. Kronbichler,
  M. Maier, J.-P. Pelteret, B. Turcksin, D. Wells, *The deal.II finite element
  library: Design, features, and insights*, Comput. Math. Appl. **81** (2021)
  407–422.
  [doi:10.1016/j.camwa.2020.02.022](https://doi.org/10.1016/j.camwa.2020.02.022)

## Requirements

- [deal.II](https://www.dealii.org) 9.5 or newer, built with MPI and p4est
  (the CI uses the `dealii/dealii:v9.7.1-noble` Docker image)
- CMake 3.13.4 or newer and a C++20 compiler
- optional: gnuplot for the plot scripts in `scripts/`; Python 3 with
  `numpy`, `sympy` and `matplotlib` (`pip install -r requirements.txt`) for
  the analyses in `analysis/`

## Build and run

    cmake -S . -B build -DDEAL_II_DIR=/path/to/dealii
    cmake --build build
    cd build
    ./cgdbe --help                                        # all options and their defaults
    ./cgdbe                                               # Taylor-Green vortex, 64^2 elements, Re = 100
    ./cgdbe --case couette --scheme leelin
    ./cgdbe --case cavity --mass lumped --reynolds 400,1000
    gnuplot ../scripts/plot_cavity.gp                     # -> cavity.png (streamlines)

CMake options: `-DLBFEM_DEGREE=p` selects the element degree (a compile-time
constant, default 1), `-DLBFEM_BUILD_EXTRAS=ON` also builds the benchmarks and
tools, `-DLBFEM_NATIVE=OFF` drops `-O3 -march=native`. The CI
(`.github/workflows/ci.yml`) builds the driver, runs a coarse lid-driven cavity
to steady state, checks the primary vortex against Ghia, Ghia & Shin (1982) and
uploads the streamline plot as a build artifact.

## Repository layout

| | |
|---|---|
| `include/lbfem/` | the header-only library: lattice, discretization, collision, advection, mass solves, streaming, schemes, test cases |
| `apps/cgdbe/` | the `cgdbe` driver: options, Reynolds continuation, diagnostics, output, performance summary |
| `scripts/` | run and plot drivers: `run_perf.sh` (throughput scan), `plot_vortex.gp`, `plot_cavity.gp` (gnuplot) |
| `analysis/` | sympy/numpy analysis of the schemes: element matrices and stencils, von Neumann analyses, TikZ figure generator |
| `benchmarks/` | stand-alone micro-benchmarks: empirical roofline, stencil "speed of light", vectorized collision |
| `tools/` | element matrices from deal.II `FEValues`, spectra of the mass preconditioners, reproducer of a fused-CG slowdown in deal.II |
| `data/cavity/` | lid-driven cavity reference results ($`Re`$ = 400, 1000, 3200, 5000; Q2, $`32^2`$ elements) |
| `docs/` | documentation: [method.md](docs/method.md) (formulation, schemes, code structure, options), [results.md](docs/results.md) (numerical experiments), [analysis.md](docs/analysis.md) (von Neumann analyses), [performance.md](docs/performance.md) (measurements), [dealii-fused-cg.md](docs/dealii-fused-cg.md) (a deal.II performance note), `figures/` (stencil figures) |
