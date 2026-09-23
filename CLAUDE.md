# Project rules

- One C++ statement per line: no short blocks, ifs, loops or functions on a
  single line; a lambda stays on one line only as a call argument.
  `.clang-format` (deal.II's style at 120 columns) enforces this; run
  `clang-format -i` on changed C++ files.
- The README describes the method and the code, not results: no timings,
  speed-ups, iteration counts or accuracy numbers there. Such results go to
  `docs/` (with what produced them, so they can be rerun) or into the PR.
