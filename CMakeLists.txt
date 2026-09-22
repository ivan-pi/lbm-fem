cmake_minimum_required(VERSION 3.13.4)

# Point to your installation with  -DDEAL_II_DIR=/path/to/dealii  (or the
# DEAL_II_DIR environment variable). Needs deal.II >= 9.5 with MPI and p4est.
find_package(deal.II 9.5 REQUIRED HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR "cgdbe requires a deal.II built with MPI and p4est.")
endif()

# A deal.II that was itself configured for C++17 (e.g. distribution packages)
# exports "-std=c++17" in its flags, which would override CXX_STANDARD below.
string(REGEX REPLACE "-std=[^ ]+" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug or Release")
endif()

deal_ii_initialize_cached_variables()
project(cgdbe LANGUAGES CXX)

add_executable(cgdbe src/cgdbe.cc)
deal_ii_setup_target(cgdbe)

set_target_properties(cgdbe PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

# Polynomial degree of the FE_Q elements (compile-time constant of the kernels).
set(CGDBE_DEGREE 1 CACHE STRING "Polynomial degree of the elements")
target_compile_definitions(cgdbe PRIVATE CGDBE_DEGREE=${CGDBE_DEGREE})

# The matrix-free kernels are header templates that are compiled here, so they
# benefit from native tuning. The SIMD width of VectorizedArray, however, is
# fixed by how deal.II itself was configured.
option(CGDBE_NATIVE "Compile with -O3 -march=native" ON)
if(CGDBE_NATIVE)
  target_compile_options(cgdbe PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
endif()

# Optional helpers: stand-alone micro-benchmarks (plain C++) and the deal.II
# element-matrix check that is compared against analysis/stencils.py.
option(CGDBE_BUILD_EXTRAS "Build benchmarks/ and tools/" OFF)
if(CGDBE_BUILD_EXTRAS)
  foreach(bench roofline stencil_bench)
    add_executable(${bench} benchmarks/${bench}.cc)
    set_target_properties(${bench} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_compile_options(${bench} PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
  endforeach()
  add_executable(elem tools/elem.cc)
  deal_ii_setup_target(elem)
  set_target_properties(elem PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
endif()
