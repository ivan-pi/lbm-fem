# Project rules

- One C++ statement per line: no short blocks, ifs, loops or functions on a
  single line; a lambda stays on one line only as a call argument.
  `.clang-format` (deal.II's style at 120 columns) enforces this; run
  `clang-format -i` on changed C++ files.
- The README is minimal: what the repository is, the references, the
  requirements, how to build and run, the layout. The method, the code and
  the driver options are described in `docs/method.md`; the options
  themselves are documented by `cgdbe --help`.
- No results in the README or in `docs/method.md`: no timings, speed-ups,
  iteration counts or accuracy numbers there. Such results go to
  `docs/results.md`, `docs/analysis.md` or `docs/performance.md` (with what
  produced them, so they can be rerun) or into the PR.
