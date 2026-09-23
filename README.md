# PRAMAAN

Compute Fast. Polish Precisely. Prove Independently.

Sovereign, structure-aware, precision-adaptive, proof-carrying LP/MILP solver.
SIH Problem Statement 26119 (MRPL).

See docs/spec.md for the full architecture spec.

## Core Features
*   **CPU Revised/Dual Simplex**: Sparse, robust, warm-start-capable core solvers.
*   **CPU MILP Branch-and-Cut**: Multi-core B&B with knapsack cover cuts and lazy bounds.
*   **GPU PDHG Fast Pass**: FP32 precision LP candidate generation via PyTorch-free CUDA.
*   **Precision Ladder**: Automated FP64 CPU polish when GPU primal residuals exceed 1e-5.
*   **Independent Certificate**: Verifiable solution certificates decoupled from the solver core.
*   **Rolling Horizon**: Predictable benchmarks for sequential resolve performance.

## Build

### CPU Only (Default)
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPRAMAAN_ENABLE_CUDA=OFF ..
cmake --build . -j$(nproc)
```

### With CUDA GPU PDHG Support
```bash
mkdir -p build-gpu && cd build-gpu
cmake -DCMAKE_BUILD_TYPE=Release -DPRAMAAN_ENABLE_CUDA=ON ..
cmake --build . -j$(nproc)
```

## Usage

### Solve
```bash
./pramaan-solve model.mps output_certificate.json
```
The solver will automatically route MILP problems to the CPU Branch-and-Bound solver.
For LP, it attempts to use the GPU PDHG solver (if built with CUDA and size > 1000) followed by the Precision Ladder polish, or falls back to CPU if disabled.

You can explicitly force GPU or CPU LP solvers (for testing) with `--gpu` or `--cpu`.

### Verify
```bash
./pramaan-verify model.mps output_certificate.json
```
The verifier recomputes objective functions, residuals, and variable bounds in the original space to establish mathematical correctness. It detects tampered models (via fingerprinting) and tampered certificates.

### Test
Run CTest from the build directory:
```bash
ctest --output-on-failure
```

### Benchmark
```bash
./rolling_horizon_bench --days 100 --products 50 --resources 30 --groups 5
```

## Roadmap / P2 Scope
- Distributed / MPI scale-out solver.
- Comprehensive KKT optimality proof export.
- Support for MIQP/NLP models.
- General purpose Presolve algorithms (only subset implemented for current bounds).

```
pramaan
├─ .clang-format
├─ CMakeLists.txt
├─ README.md
├─ benchmark
│  ├─ data
│  │  ├─ adlittle.mps
│  │  ├─ afiro.mps
│  │  ├─ beaconfd.mps
│  │  ├─ israel.mps
│  │  ├─ kb2.mps
│  │  ├─ lotfi.mps
│  │  ├─ sc205.mps
│  │  ├─ sc50a.mps
│  │  ├─ scorpion.mps
│  │  └─ share2b.mps
│  ├─ fetch_benchmarks.sh
│  ├─ mrpl_demo.cpp
│  ├─ output
│  │  ├─ adlittle.cert
│  │  ├─ afiro.cert
│  │  ├─ beaconfd.cert
│  │  ├─ israel.cert
│  │  ├─ kb2.cert
│  │  ├─ lotfi.cert
│  │  ├─ sc105.cert
│  │  ├─ sc205.cert
│  │  ├─ sc50a.cert
│  │  ├─ sc50b.cert
│  │  ├─ scorpion.cert
│  │  └─ share2b.cert
│  ├─ rolling_horizon_bench.cpp
│  ├─ rolling_horizon_benchmark.py
│  └─ run_benchmark.py
├─ build-cpu
│  ├─ err3.txt
│  ├─ err4.txt
│  ├─ err5.txt
│  ├─ mrpl_demo
│  ├─ pramaan-solve
│  ├─ pramaan-verify
│  ├─ rolling_horizon_bench
│  ├─ test_branch_and_bound
│  ├─ test_certificate
│  ├─ test_cli_milp_routing
│  ├─ test_cli_precision_ladder
│  ├─ test_dual_simplex
│  ├─ test_interior_point
│  ├─ test_mip_node
│  ├─ test_mps_parser
│  ├─ test_numerical_stress
│  ├─ test_presolve_scaling
│  ├─ test_rolling_horizon
│  ├─ test_simplex_tiny_lp
│  ├─ test_sparse_matrix
│  ├─ test_sparse_revised_simplex
│  └─ test_structural_fingerprint
├─ docs
│  └─ spec.md
├─ finish_report.sh
├─ include
│  └─ pramaan
│     ├─ branch_and_bound.hpp
│     ├─ certificate.hpp
│     ├─ dual_simplex.hpp
│     ├─ gpu
│     │  └─ pdhg_solver.hpp
│     ├─ ipm
│     │  └─ interior_point.hpp
│     ├─ ir.hpp
│     ├─ mps_parser.hpp
│     ├─ node.hpp
│     ├─ presolve.hpp
│     ├─ simplex.hpp
│     ├─ sparse_matrix.hpp
│     ├─ structure
│     │  └─ fingerprint.hpp
│     └─ transformation_ledger.hpp
├─ python
│  └─ README.md
├─ src
│  ├─ certificate
│  │  ├─ certificate.cpp
│  │  └─ verifier_main.cpp
│  ├─ cli
│  │  └─ solve_main.cpp
│  ├─ gpu
│  │  ├─ cuda_utils.cuh
│  │  └─ pdhg_solver.cu
│  ├─ io
│  │  └─ mps_parser.cpp
│  ├─ ipm
│  │  └─ interior_point.cpp
│  ├─ ir
│  │  └─ model_ir.cpp
│  ├─ linalg
│  │  └─ sparse_matrix.cpp
│  ├─ mip
│  │  ├─ branch_and_bound.cpp
│  │  └─ node.cpp
│  ├─ presolve
│  │  ├─ presolve.cpp
│  │  ├─ scaling.cpp
│  │  └─ transformation_ledger.cpp
│  ├─ simplex
│  │  ├─ dual_simplex.cpp
│  │  └─ revised_simplex.cpp
│  └─ structure
│     └─ fingerprint.cpp
└─ tests
   ├─ data
   │  ├─ adlittle.mps
   │  └─ afiro.mps
   ├─ test_branch_and_bound.cpp
   ├─ test_certificate.cpp
   ├─ test_cli_milp_routing.cpp
   ├─ test_cli_precision_ladder.cpp
   ├─ test_dual_simplex.cpp
   ├─ test_gpu_pdhg.cpp
   ├─ test_interior_point.cpp
   ├─ test_mip_node.cpp
   ├─ test_mps_parser.cpp
   ├─ test_numerical_stress.cpp
   ├─ test_presolve_scaling.cpp
   ├─ test_rolling_horizon.cpp
   ├─ test_simplex_tiny_lp.cpp
   ├─ test_sparse_matrix.cpp
   ├─ test_sparse_revised_simplex.cpp
   └─ test_structural_fingerprint.cpp

```
