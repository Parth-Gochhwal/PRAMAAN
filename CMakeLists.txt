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
)
target_include_directories(pramaan_core PUBLIC include)

# --- pramaan-solve: the main solver CLI ---
add_executable(pramaan-solve src/cli/solve_main.cpp)
target_link_libraries(pramaan-solve PRIVATE pramaan_core)

# --- pramaan-verify: the INDEPENDENT verifier ---
# Links pramaan_core for MPS parsing, certificate I/O, and CSRMatrix::multiply.
# Does NOT call RevisedSimplex or any solver routine -- independence is
# enforced by code review, not by link-time isolation.
add_executable(pramaan-verify src/certificate/verifier_main.cpp)
target_link_libraries(pramaan-verify PRIVATE pramaan_core)

# --- GPU target: uncommented once you start Phase 4 (CUDA) ---
# enable_language(CUDA)
# add_library(pramaan_gpu src/gpu/pdhg.cu)
# target_link_libraries(pramaan-solve PRIVATE pramaan_gpu)

# --- Tests ---
enable_testing()

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
