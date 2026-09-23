cmake_minimum_required(VERSION 3.18)
project(pramaan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # needed for VSCode IntelliSense

# --- core library: everything except the CLI entry points ---
# Add new .cpp files here as you write them.
add_library(pramaan_core
    src/io/mps_parser.cpp
    src/ir/model_ir.cpp
    src/linalg/sparse_matrix.cpp
    src/presolve/presolve.cpp
    src/presolve/scaling.cpp
    src/presolve/transformation_ledger.cpp
    src/simplex/revised_simplex.cpp
    src/simplex/dual_simplex.cpp
    src/certificate/certificate.cpp
    src/mip/node.cpp
    src/mip/branch_and_bound.cpp
    src/structure/fingerprint.cpp
    src/ipm/interior_point.cpp
)
target_include_directories(pramaan_core PUBLIC include)
target_compile_options(pramaan_core PRIVATE -Wall -Wextra -Wpedantic)

# --- pramaan-solve: the main solver CLI ---
add_executable(pramaan-solve src/cli/solve_main.cpp)
target_link_libraries(pramaan-solve PRIVATE pramaan_core)

# --- pramaan-verify: the INDEPENDENT verifier ---
# Links pramaan_core for MPS parsing, certificate I/O, and CSRMatrix::multiply.
# Does NOT call RevisedSimplex or any solver routine -- independence is
# enforced by code review, not by link-time isolation.
add_executable(pramaan-verify src/certificate/verifier_main.cpp)
target_link_libraries(pramaan-verify PRIVATE pramaan_core)

# --- Tests ---
enable_testing()

# --- GPU target ---
option(PRAMAAN_ENABLE_CUDA "Enable CUDA GPU support" OFF)
# CUDA architecture: override with -DPRAMAAN_CUDA_ARCHITECTURES=XX at configure time.
# Default covers Turing (sm_75) and Ampere (sm_86); adjust for your GPU.
set(PRAMAAN_CUDA_ARCHITECTURES "75;86" CACHE STRING "CUDA GPU architectures to compile for")
if(PRAMAAN_ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${PRAMAAN_CUDA_ARCHITECTURES})
    enable_language(CUDA)
    add_library(pramaan_gpu src/gpu/pdhg_solver.cu)
    target_include_directories(pramaan_gpu PUBLIC include)
    target_link_libraries(pramaan_gpu PRIVATE pramaan_core)

    target_link_libraries(pramaan-solve PRIVATE pramaan_gpu)
    target_compile_definitions(pramaan-solve PRIVATE PRAMAAN_ENABLE_CUDA)
    target_compile_definitions(pramaan_core PUBLIC PRAMAAN_ENABLE_CUDA)

    add_executable(test_gpu_pdhg tests/test_gpu_pdhg.cpp)
    target_link_libraries(test_gpu_pdhg PRIVATE pramaan_core pramaan_gpu)
    add_test(NAME test_gpu_pdhg COMMAND test_gpu_pdhg
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

add_executable(test_sparse_matrix tests/test_sparse_matrix.cpp)
target_link_libraries(test_sparse_matrix PRIVATE pramaan_core)
add_test(NAME test_sparse_matrix COMMAND test_sparse_matrix
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_simplex_tiny_lp tests/test_simplex_tiny_lp.cpp)
target_link_libraries(test_simplex_tiny_lp PRIVATE pramaan_core)
add_test(NAME test_simplex_tiny_lp COMMAND test_simplex_tiny_lp
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_mps_parser tests/test_mps_parser.cpp)
target_link_libraries(test_mps_parser PRIVATE pramaan_core)
add_test(NAME test_mps_parser COMMAND test_mps_parser
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_sparse_revised_simplex tests/test_sparse_revised_simplex.cpp)
target_link_libraries(test_sparse_revised_simplex PRIVATE pramaan_core)
add_test(NAME test_sparse_revised_simplex COMMAND test_sparse_revised_simplex
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_presolve_scaling tests/test_presolve_scaling.cpp)
target_link_libraries(test_presolve_scaling PRIVATE pramaan_core)
add_test(NAME test_presolve_scaling COMMAND test_presolve_scaling
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_certificate tests/test_certificate.cpp)
target_link_libraries(test_certificate PRIVATE pramaan_core)
target_compile_definitions(test_certificate PRIVATE
    VERIFIER_EXECUTABLE="$<TARGET_FILE:pramaan-verify>"
    SOLVER_EXECUTABLE="$<TARGET_FILE:pramaan-solve>")
add_test(NAME test_certificate COMMAND test_certificate
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_dual_simplex tests/test_dual_simplex.cpp)
target_link_libraries(test_dual_simplex PRIVATE pramaan_core)
add_test(NAME test_dual_simplex COMMAND test_dual_simplex
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_mip_node tests/test_mip_node.cpp)
target_link_libraries(test_mip_node PRIVATE pramaan_core)
add_test(NAME test_mip_node COMMAND test_mip_node
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_branch_and_bound tests/test_branch_and_bound.cpp)
target_link_libraries(test_branch_and_bound PRIVATE pramaan_core)
add_test(NAME test_branch_and_bound COMMAND test_branch_and_bound
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_structural_fingerprint tests/test_structural_fingerprint.cpp)
target_link_libraries(test_structural_fingerprint PRIVATE pramaan_core)
add_test(NAME test_structural_fingerprint COMMAND test_structural_fingerprint
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_interior_point tests/test_interior_point.cpp)
target_link_libraries(test_interior_point PRIVATE pramaan_core)
add_test(NAME test_interior_point COMMAND test_interior_point
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(rolling_horizon_bench benchmark/rolling_horizon_bench.cpp)
target_link_libraries(rolling_horizon_bench PRIVATE pramaan_core)

add_executable(mrpl_demo benchmark/mrpl_demo.cpp)
target_link_libraries(mrpl_demo PRIVATE pramaan_core)

add_executable(test_rolling_horizon tests/test_rolling_horizon.cpp)
target_link_libraries(test_rolling_horizon PRIVATE pramaan_core)
target_compile_definitions(test_rolling_horizon PRIVATE BENCHMARK_EXECUTABLE="$<TARGET_FILE:rolling_horizon_bench>")
add_test(NAME test_rolling_horizon COMMAND test_rolling_horizon
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_numerical_stress tests/test_numerical_stress.cpp)
target_link_libraries(test_numerical_stress PRIVATE pramaan_core)
add_test(NAME test_numerical_stress COMMAND test_numerical_stress
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_executable(test_cli_milp_routing tests/test_cli_milp_routing.cpp)
add_test(NAME test_cli_milp_routing COMMAND test_cli_milp_routing
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(test_cli_precision_ladder tests/test_cli_precision_ladder.cpp)
add_test(NAME test_cli_precision_ladder COMMAND test_cli_precision_ladder
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
