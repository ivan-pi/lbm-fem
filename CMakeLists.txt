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
project(lbfem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Polynomial degree of the FE_Q elements (compile-time constant of the kernels).
set(CGDBE_DEGREE 1 CACHE STRING "Polynomial degree of the elements")

# The matrix-free kernels are header templates that are compiled with each
# driver, so they benefit from native tuning. The SIMD width of VectorizedArray,
# however, is fixed by how deal.II itself was configured.
option(CGDBE_NATIVE "Compile with -O3 -march=native" ON)

# A driver: an executable built on deal.II and the header-only lbfem library
# (include/lbfem). Its sources see CGDBE_DEGREE as a compile definition.
# (Include directories rather than target_link_libraries, whose plain and
# keyword forms deal_ii_setup_target does not mix with.)
function(lbfem_add_driver name)
  add_executable(${name} ${ARGN})
  deal_ii_setup_target(${name})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(${name} PRIVATE CGDBE_DEGREE=${CGDBE_DEGREE})
  if(CGDBE_NATIVE)
    target_compile_options(${name} PRIVATE $<$<CONFIG:Release>:-O3 -march=native>)
  endif()
endfunction()

lbfem_add_driver(cgdbe apps/cgdbe/cgdbe.cc)

# Optional helpers: stand-alone micro-benchmarks (plain C++, always tuned for
# the build machine, whose peak they measure) and the deal.II element-matrix
# check that is compared against analysis/stencils.py.
option(CGDBE_BUILD_EXTRAS "Build benchmarks/ and tools/" OFF)
if(CGDBE_BUILD_EXTRAS)
  foreach(bench roofline stencil_bench collision_simd)
    add_executable(${bench} benchmarks/${bench}.cc)
    target_compile_options(${bench} PRIVATE -O3 -march=native)
  endforeach()
  target_include_directories(collision_simd PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(collision_simd PRIVATE -fopenmp-simd)
  add_executable(elem tools/elem.cc)
  deal_ii_setup_target(elem)
  lbfem_add_driver(mass_preconditioners tools/mass_preconditioners.cc)
endif()
