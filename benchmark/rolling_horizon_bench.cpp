// rolling_horizon_bench.cpp
// P1 Step 8 — Rolling-Horizon Warm-Start Benchmark
//
// Purpose: Measure the ACTUAL cumulative cold-vs-warm-start performance of
// PRAMAAN's DualSimplex::warmSolve implementation over a simulated sequence
// of similar LP instances (rolling-horizon re-solves).
//
// This benchmark does NOT fabricate, assume, or hard-code any speedup.
// It measures wall-clock time and reports whatever the real result is.
//
// Usage:
//   ./rolling_horizon_bench [--days N] [--seed S] [--tolerance T] [--json]
//
// The benchmark:
//   1. Builds a synthetic multi-resource production-planning LP in memory.
//   2. Generates a deterministic sequence of small RHS and bound perturbations.
//   3. Runs two strategies on the IDENTICAL sequence of LP instances:
//      - COLD: RevisedSimplex::solve() on each day's LP (from scratch).
//      - WARM: Day 0 via DualSimplex::captureBasis(), subsequent days via
//              DualSimplex::warmSolve() with basis chaining through out_basis.
//   4. Verifies correctness: status, objective agreement, feasibility.
//   5. Reports measured cumulative times and actual speedup.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/dual_simplex.hpp"

using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::VarType;
using pramaan::RevisedSimplex;
using pramaan::DualSimplex;
using pramaan::BasisState;
using pramaan::SolveResult;
using pramaan::SolveStatus;

// ============================================================================
// Deterministic PRNG (LCG)
// ============================================================================
// Simple linear congruential generator for reproducible perturbations.
// Not cryptographic — just deterministic and portable.
struct LCG {
    uint64_t state;
    explicit LCG(uint64_t seed) : state(seed) {}

    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    }

    // Returns a double in [lo, hi]
    double uniform(double lo, double hi) {
        uint64_t v = next();
        double t = static_cast<double>(v >> 11) / static_cast<double>(1ULL << 53);
        return lo + t * (hi - lo);
    }
};

// ============================================================================
// Helper: build CSRMatrix from dense data
// ============================================================================
static CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense,
                            CSRMatrix::Index num_cols) {
    std::vector<CSRMatrix::Index> row_ptr;
    std::vector<CSRMatrix::Index> col_idx;
    std::vector<double> values;
    row_ptr.push_back(0);
    for (const auto& row : dense) {
        for (CSRMatrix::Index c = 0; c < static_cast<CSRMatrix::Index>(row.size()); ++c) {
            if (row[static_cast<std::size_t>(c)] != 0.0) {
                col_idx.push_back(c);
                values.push_back(row[static_cast<std::size_t>(c)]);
            }
        }
        row_ptr.push_back(static_cast<CSRMatrix::Index>(col_idx.size()));
    }
    return CSRMatrix(std::move(row_ptr), std::move(col_idx), std::move(values), num_cols);
}

// ============================================================================
// Build synthetic production-planning LP
// ============================================================================
// A multi-resource, multi-product production planning model:
//
//   maximize  sum_j profit_j * x_j
//   subject to:
//     Resource constraints: sum_j a_{rj} * x_j <= capacity_r   (R resources)
//     Demand lower bounds:  x_j >= demand_j                    (per product)
//     Production capacity:  x_j <= max_production_j            (per product)
//
// Parameters chosen so all variables have finite bounds (required for
// warm-start compatibility) and the LP has a non-trivial optimal solution.
//
// Size: P products x R resource constraints, so the model has P variables
// and R+0 rows (demand/capacity are variable bounds, not row constraints).
// With P=50, R=30, this gives a 30x50 constraint matrix.
//
// Actually, to make it more interesting for warm-start (more rows in
// standard form), we'll include some of the demand constraints as
// explicit row constraints: sum_j >= min_total_demand per product group.
//
// Final size: ~50 variables, ~40 constraints.
struct BenchmarkConfig {
    int num_days = 30;
    uint64_t seed = 12345;
    double tolerance = 1e-8;
    bool json_output = false;
    int num_products = 50;
    int num_resources = 30;
    int num_demand_groups = 5;
    double perturb_rhs = 0.10;
    double perturb_bounds = 0.08;
};


static ModelIR buildProductionPlanningLP(const BenchmarkConfig& cfg) {
    LCG rng(cfg.seed);

    const int P = cfg.num_products;
    const int R = cfg.num_resources;
    const int G = cfg.num_demand_groups;
    const int total_rows = R + G;

    // 1. Generate a known feasible reference solution
    std::vector<double> x_ref(static_cast<std::size_t>(P));
    for (int j = 0; j < P; ++j) {
        x_ref[static_cast<std::size_t>(j)] = rng.uniform(2.0, 10.0);
    }

    // Objective: profit per product
    std::vector<double> obj_coeffs(static_cast<std::size_t>(P));
    for (int j = 0; j < P; ++j) {
        obj_coeffs[static_cast<std::size_t>(j)] = rng.uniform(1.0, 10.0);
    }

    // Constraint matrix: resource usage coefficients
    std::vector<std::vector<double>> A_dense(static_cast<std::size_t>(total_rows),
                                              std::vector<double>(static_cast<std::size_t>(P), 0.0));

    // Resource constraints: each resource uses a subset of products
    for (int r = 0; r < R; ++r) {
        for (int j = 0; j < P; ++j) {
            if (rng.uniform(0.0, 1.0) < 0.6) {
                A_dense[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] = rng.uniform(0.5, 5.0);
            }
        }
        int ensure_col = static_cast<int>(rng.next() % static_cast<uint64_t>(P));
        if (A_dense[static_cast<std::size_t>(r)][static_cast<std::size_t>(ensure_col)] == 0.0) {
            A_dense[static_cast<std::size_t>(r)][static_cast<std::size_t>(ensure_col)] = rng.uniform(1.0, 3.0);
        }
    }

    // Demand group constraints
    if (G > 0) {
        int products_per_group = P / G;
        if (products_per_group == 0) products_per_group = 1;
        for (int g = 0; g < G; ++g) {
            int start = g * products_per_group;
            int end = (g == G - 1) ? P : (g + 1) * products_per_group;
            if (start >= P) break; // Guard against P < G
            for (int j = start; j < end; ++j) {
                A_dense[static_cast<std::size_t>(R + g)][static_cast<std::size_t>(j)] = 1.0;
            }
        }
    }

    CSRMatrix A = denseToCSR(A_dense, P);

    // Row bounds
    std::vector<double> row_lower(static_cast<std::size_t>(total_rows));
    std::vector<double> row_upper(static_cast<std::size_t>(total_rows));
    std::vector<std::string> row_names(static_cast<std::size_t>(total_rows));

    // Calculate safe bounds so that x_ref remains strictly feasible even after perturbation
    double min_rhs_upper = 1.0 / std::max(0.01, 1.0 - cfg.perturb_rhs) + 0.1;
    double max_rhs_lower = std::max(0.01, 1.0 / (1.0 + cfg.perturb_rhs) - 0.1);
    double min_var_upper = 1.0 / std::max(0.01, 1.0 - cfg.perturb_bounds) + 0.1;

    // 2. Derive resource capacities so x_ref is feasible with slack
    for (int r = 0; r < R; ++r) {
        double usage = 0.0;
        for (int j = 0; j < P; ++j) {
            usage += A_dense[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] * x_ref[static_cast<std::size_t>(j)];
        }
        row_lower[static_cast<std::size_t>(r)] = -kInfinity;
        row_upper[static_cast<std::size_t>(r)] = usage * rng.uniform(min_rhs_upper, min_rhs_upper + 0.6);
        row_names[static_cast<std::size_t>(r)] = "resource_" + std::to_string(r);
    }

    // 3. Derive demand group minimums so x_ref is feasible with slack
    for (int g = 0; g < G; ++g) {
        double group_sum = 0.0;
        for (int j = 0; j < P; ++j) {
            group_sum += A_dense[static_cast<std::size_t>(R + g)][static_cast<std::size_t>(j)] * x_ref[static_cast<std::size_t>(j)];
        }
        row_lower[static_cast<std::size_t>(R + g)] = group_sum * rng.uniform(std::max(0.01, max_rhs_lower - 0.4), max_rhs_lower);
        row_upper[static_cast<std::size_t>(R + g)] = kInfinity;
        row_names[static_cast<std::size_t>(R + g)] = "demand_group_" + std::to_string(g);
    }

    // 4. Derive variable upper bounds so x_ref is feasible
    std::vector<double> var_lower(static_cast<std::size_t>(P));
    std::vector<double> var_upper(static_cast<std::size_t>(P));
    std::vector<VarType> var_types(static_cast<std::size_t>(P), VarType::kContinuous);
    std::vector<std::string> var_names(static_cast<std::size_t>(P));

    for (int j = 0; j < P; ++j) {
        var_lower[static_cast<std::size_t>(j)] = 0.0;
        var_upper[static_cast<std::size_t>(j)] = x_ref[static_cast<std::size_t>(j)] * rng.uniform(min_var_upper, min_var_upper + 1.5);
        var_names[static_cast<std::size_t>(j)] = "prod_" + std::to_string(j);
    }

    return ModelIR(
        ObjSense::kMaximize, 0.0, std::move(obj_coeffs),
        std::move(A),
        std::move(row_lower), std::move(row_upper), std::move(row_names),
        std::move(var_lower), std::move(var_upper), std::move(var_types), std::move(var_names));
}

// ============================================================================
// Perturbation: modify RHS and bounds for a given day
// ============================================================================
// Applies small deterministic perturbations to the base model.
// - Resource capacities (row_upper for resource rows): configurable perturbation magnitude
// - Demand minimums (row_lower for demand rows): configurable perturbation magnitude
// - Variable upper bounds: configurable perturbation magnitude
//
// Preserves:
// - Matrix A (unchanged)
// - Objective coefficients (unchanged)

// - Bound types (finite stays finite, infinite stays infinite)
// - LP structure

static ModelIR perturbModel(const ModelIR& base, int day, const BenchmarkConfig& cfg) {
    ModelIR model = base;

    // Deterministic seed per day
    LCG rng(cfg.seed * 1000003ULL + static_cast<uint64_t>(day) * 7919ULL);

    const int R = cfg.num_resources;

    // Perturb resource capacities (row_upper for resource rows)
    for (int r = 0; r < R; ++r) {
        double orig = base.row_upper[static_cast<std::size_t>(r)];
        if (orig < kInfinity) {
            double factor = rng.uniform(1.0 - cfg.perturb_rhs, 1.0 + cfg.perturb_rhs);
            model.row_upper[static_cast<std::size_t>(r)] = orig * factor;
        }
    }

    // Perturb demand group minimums (row_lower for demand rows)
    for (int r = R; r < base.numRows(); ++r) {
        double orig = base.row_lower[static_cast<std::size_t>(r)];
        if (orig > -kInfinity) {
            double factor = rng.uniform(1.0 - cfg.perturb_rhs, 1.0 + cfg.perturb_rhs);
            model.row_lower[static_cast<std::size_t>(r)] = orig * factor;
        }
    }

    // Perturb variable upper bounds (small shift, keep positive)
    for (int j = 0; j < base.numVars(); ++j) {
        double orig = base.var_upper[static_cast<std::size_t>(j)];
        if (orig < kInfinity) {
            double factor = rng.uniform(1.0 - cfg.perturb_bounds, 1.0 + cfg.perturb_bounds);
            double new_val = orig * factor;
            // Ensure upper bound stays above lower bound
            if (new_val > base.var_lower[static_cast<std::size_t>(j)] + 1e-6) {
                model.var_upper[static_cast<std::size_t>(j)] = new_val;
            }
        }
    }

    return model;
}

// ============================================================================
// Correctness verification
// ============================================================================

struct CorrectnessResult {
    bool passed = true;
    std::string error_message;
};

static CorrectnessResult verifyAgreement(const ModelIR& model,
                                          const SolveResult& cold,
                                          const SolveResult& warm,
                                          int day, double tol) {
    CorrectnessResult cr;

    // Status check
    if (cold.status != warm.status) {
        cr.passed = false;
        std::ostringstream oss;
        oss << "Day " << day << ": cold status = " << static_cast<int>(cold.status)
            << ", warm status = " << static_cast<int>(warm.status);
        cr.error_message = oss.str();
        return cr;
    }

    // Only verify details for optimal solutions
    if (cold.status != SolveStatus::kOptimal || warm.status != SolveStatus::kOptimal) {
        return cr;
    }

    // Solution dimensions
    if (cold.x.size() != warm.x.size()) {
        cr.passed = false;
        cr.error_message = "Day " + std::to_string(day) + ": solution dimension mismatch";
        return cr;
    }

    // Objective agreement (relative tolerance)
    double ref = std::max(std::abs(cold.objective_value), 1.0);
    double obj_diff = std::abs(cold.objective_value - warm.objective_value);
    if (obj_diff / ref > tol) {
        cr.passed = false;
        std::ostringstream oss;
        oss << "Day " << day << ": objective mismatch — cold=" << cold.objective_value
            << ", warm=" << warm.objective_value << ", rel_diff=" << (obj_diff / ref);
        cr.error_message = oss.str();
        return cr;
    }

    // Primal feasibility of warm solution
    auto activity = model.A.multiply(warm.x);
    for (int r = 0; r < model.numRows(); ++r) {
        double lo = model.row_lower[static_cast<std::size_t>(r)];
        double up = model.row_upper[static_cast<std::size_t>(r)];
        double ar = activity[static_cast<std::size_t>(r)];
        if ((lo > -kInfinity && ar < lo - tol * 100) ||
            (up < kInfinity && ar > up + tol * 100)) {
            cr.passed = false;
            std::ostringstream oss;
            oss << "Day " << day << ": warm solution infeasible at row " << r
                << " (activity=" << ar << ", bounds=[" << lo << ", " << up << "])";
            cr.error_message = oss.str();
            return cr;
        }
    }

    for (int j = 0; j < model.numVars(); ++j) {
        double lo = model.var_lower[static_cast<std::size_t>(j)];
        double up = model.var_upper[static_cast<std::size_t>(j)];
        double xj = warm.x[static_cast<std::size_t>(j)];
        if ((lo > -kInfinity && xj < lo - tol * 100) ||
            (up < kInfinity && xj > up + tol * 100)) {
            cr.passed = false;
            std::ostringstream oss;
            oss << "Day " << day << ": warm solution infeasible at var " << j
                << " (x=" << xj << ", bounds=[" << lo << ", " << up << "])";
            cr.error_message = oss.str();
            return cr;
        }
    }

    return cr;
}

// ============================================================================
// Timing helper
// ============================================================================

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

// ============================================================================
// JSON output helper
// ============================================================================
static std::string jsonArray(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << std::setprecision(9) << v[i];
    }
    oss << "]";
    return oss.str();
}

// ============================================================================
// Main benchmark
// ============================================================================

struct DayResult {
    double cold_time = 0.0;
    double warm_time = 0.0;
    bool warm_attempted = false;
    bool warm_succeeded = false;
    bool warm_fell_back = false;
    double cold_obj = 0.0;
    double warm_obj = 0.0;
    int cold_iters = 0;
    int warm_iters = 0;
    int warm_attempt_iters = 0;
    int warm_fallback_iters = 0;
    bool correctness_ok = true;
    std::string correctness_error;
    SolveResult cold_result;
    SolveResult warm_result;
};

static int runBenchmark(const BenchmarkConfig& cfg) {
    // --- Build the base model ---
    std::cerr << "Building base LP model...\n";
    ModelIR base = buildProductionPlanningLP(cfg);
    base.validate();
    std::cerr << "  Variables:   " << base.numVars() << "\n";
    std::cerr << "  Constraints: " << base.numRows() << "\n\n";

    // --- Solver instances ---
    RevisedSimplex::Options rs_opts;
    rs_opts.tolerance = cfg.tolerance;
    rs_opts.max_iterations = 50000;
    RevisedSimplex cold_solver(rs_opts);

    DualSimplex::Options ds_opts;
    ds_opts.tolerance = cfg.tolerance;
    ds_opts.max_iterations = 50000;
    DualSimplex dual_solver(ds_opts);

    // --- Pre-generate ALL perturbed models ---
    // This ensures cold and warm strategies get EXACTLY the same LPs.
    std::vector<ModelIR> day_models(static_cast<std::size_t>(cfg.num_days));
    for (int d = 0; d < cfg.num_days; ++d) {
        if (d == 0) {
            day_models[0] = base;
        } else {
            day_models[static_cast<std::size_t>(d)] =
                perturbModel(base, d, cfg);
        }
    }

    // --- Pre-validate workload feasibility ---
    for (int d = 0; d < cfg.num_days; ++d) {
        const ModelIR& model = day_models[static_cast<std::size_t>(d)];
        SolveResult res = cold_solver.solve(model);
        if (res.status != SolveStatus::kOptimal) {
            std::cerr << "BENCHMARK INVALID: generated workload is infeasible\n";
            std::cerr << "  Day " << d << " cold solve returned status: " << static_cast<int>(res.status) << "\n";
            return 1;
        }
        if (d == 0) {
            BasisState b = dual_solver.captureBasis(model);
            if (b.empty()) {
                std::cerr << "BENCHMARK INVALID: generated workload is infeasible\n";
                std::cerr << "  Day 0 basis capture failed.\n";
                return 1;
            }
        }
    }

    // --- Results storage ---
    std::vector<DayResult> results(static_cast<std::size_t>(cfg.num_days));

    // =======================================================================
    // STRATEGY 1: COLD RESTART
    // =======================================================================
    std::cerr << "Running COLD strategy (" << cfg.num_days << " days)...\n";
    for (int d = 0; d < cfg.num_days; ++d) {
        const ModelIR& model = day_models[static_cast<std::size_t>(d)];

        auto t0 = Clock::now();
        SolveResult cold_result = cold_solver.solve(model);
        auto t1 = Clock::now();

        results[static_cast<std::size_t>(d)].cold_time =
            Duration(t1 - t0).count();
        results[static_cast<std::size_t>(d)].cold_obj = cold_result.objective_value;
        results[static_cast<std::size_t>(d)].cold_iters = cold_result.iterations;
        results[static_cast<std::size_t>(d)].cold_result = cold_result;

        if (cold_result.status != SolveStatus::kOptimal) {
            std::cerr << "  WARNING: Cold solve day " << d << " status = "
                      << static_cast<int>(cold_result.status) << "\n";
        }
    }

    // =======================================================================
    // STRATEGY 2: SEQUENTIAL WARM START
    // =======================================================================
    std::cerr << "Running WARM strategy (" << cfg.num_days << " days)...\n";

    BasisState current_basis;
    int warm_start_attempts = 0;
    int warm_start_successes = 0;
    int warm_start_fallbacks = 0;

    for (int d = 0; d < cfg.num_days; ++d) {
        const ModelIR& model = day_models[static_cast<std::size_t>(d)];

        if (d == 0) {
            // Day 0: cold solve + capture basis
            auto t0 = Clock::now();
            SolveResult cold_result;
            current_basis = dual_solver.captureBasis(model, &cold_result);
            auto t1 = Clock::now();

            results[0].warm_time = Duration(t1 - t0).count();
            results[0].warm_obj = cold_result.objective_value;
            results[0].warm_iters = cold_result.iterations;
            results[0].warm_attempted = false;
            results[0].warm_succeeded = false;
            results[0].warm_result = cold_result;

            if (current_basis.empty()) {
                std::cerr << "  WARNING: Day 0 basis capture failed. "
                          << "Warm-start chain is broken.\n";
            }
        } else {
            // Day 1+: warm-solve using previous day's basis
            results[static_cast<std::size_t>(d)].warm_attempted = true;
            warm_start_attempts++;

            if (current_basis.empty()) {
                // Basis chain broken — fall back to cold solve
                warm_start_fallbacks++;

                auto t0 = Clock::now();
                SolveResult cold_fallback;
                current_basis = dual_solver.captureBasis(model, &cold_fallback);
                auto t1 = Clock::now();

                results[static_cast<std::size_t>(d)].warm_time = Duration(t1 - t0).count();
                results[static_cast<std::size_t>(d)].warm_obj = cold_fallback.objective_value;
                results[static_cast<std::size_t>(d)].warm_fallback_iters = cold_fallback.iterations;
                results[static_cast<std::size_t>(d)].warm_iters = cold_fallback.iterations;
                results[static_cast<std::size_t>(d)].warm_result = cold_fallback;
            } else {
                BasisState next_basis;

                auto t0 = Clock::now();
                try {
                    SolveResult warm_result = dual_solver.warmSolve(model, current_basis,
                                                                     &next_basis);
                    auto t1 = Clock::now();

                    if (warm_result.status == SolveStatus::kOptimal) {
                        results[static_cast<std::size_t>(d)].warm_time = Duration(t1 - t0).count();
                        results[static_cast<std::size_t>(d)].warm_obj = warm_result.objective_value;
                        results[static_cast<std::size_t>(d)].warm_iters = warm_result.iterations;
                        results[static_cast<std::size_t>(d)].warm_succeeded = true;
                        results[static_cast<std::size_t>(d)].warm_result = warm_result;
                        warm_start_successes++;

                        // Chain the basis for the next day
                        if (!next_basis.empty()) {
                            current_basis = next_basis;
                        } else {
                            // Warm solve succeeded but no reusable basis — re-capture
                            current_basis = dual_solver.captureBasis(model);
                        }
                    } else {
                        // Warm-start failed to find optimal solution — fall back
                        warm_start_fallbacks++;

                        std::cerr << "  Day " << d << ": warm-start sub-optimal (status "
                                  << static_cast<int>(warm_result.status)
                                  << "), falling back to cold solve.\n";

                        auto t0b = Clock::now();
                        SolveResult cold_fallback;
                        current_basis = dual_solver.captureBasis(model, &cold_fallback);
                        auto t1b = Clock::now();

                        // Include the failed warm-start attempt time + cold fallback time
                        results[static_cast<std::size_t>(d)].warm_time =
                            Duration(t1 - t0).count() + Duration(t1b - t0b).count();
                        results[static_cast<std::size_t>(d)].warm_obj = cold_fallback.objective_value;
                        results[static_cast<std::size_t>(d)].warm_attempt_iters = warm_result.iterations;
                        results[static_cast<std::size_t>(d)].warm_fallback_iters = cold_fallback.iterations;
                        results[static_cast<std::size_t>(d)].warm_iters = warm_result.iterations + cold_fallback.iterations;
                        results[static_cast<std::size_t>(d)].warm_result = cold_fallback;
                    }
                } catch (const std::invalid_argument& e) {
                    auto t1 = Clock::now();
                    // Warm-start rejected — fall back to cold solve
                    warm_start_fallbacks++;

                    std::cerr << "  Day " << d << ": warm-start rejected ("
                              << e.what() << "), falling back to cold solve.\n";

                    auto t0b = Clock::now();
                    SolveResult cold_fallback;
                    current_basis = dual_solver.captureBasis(model, &cold_fallback);
                    auto t1b = Clock::now();

                    results[static_cast<std::size_t>(d)].warm_time =
                        Duration(t1 - t0).count() + Duration(t1b - t0b).count();
                    results[static_cast<std::size_t>(d)].warm_obj = cold_fallback.objective_value;
                    results[static_cast<std::size_t>(d)].warm_fallback_iters = cold_fallback.iterations;
                    results[static_cast<std::size_t>(d)].warm_iters = cold_fallback.iterations;
                    results[static_cast<std::size_t>(d)].warm_result = cold_fallback;
                }
            }
        }
    }

    // =======================================================================
    // CORRECTNESS VERIFICATION
    // =======================================================================
    std::cerr << "Verifying correctness...\n";
    bool all_correct = true;
    for (int d = 0; d < cfg.num_days; ++d) {
        // Re-solve cold for verification (we already have cold_obj stored)
        // Compare objectives
        const auto& r = results[static_cast<std::size_t>(d)];
        double ref = std::max(std::abs(r.cold_obj), 1.0);
        double obj_diff = std::abs(r.cold_obj - r.warm_obj);
        if (obj_diff / ref > cfg.tolerance * 1e4) {
            std::cerr << "  CORRECTNESS FAILURE Day " << d
                      << ": cold_obj=" << r.cold_obj << ", warm_obj=" << r.warm_obj
                      << ", rel_diff=" << (obj_diff / ref) << "\n";
            results[static_cast<std::size_t>(d)].correctness_ok = false;
            all_correct = false;
        }
    }

    // Also do full feasibility verification on all days using recorded results
    for (int d = 0; d < cfg.num_days; ++d) {
        const ModelIR& model = day_models[static_cast<std::size_t>(d)];
        const auto& r = results[static_cast<std::size_t>(d)];

        CorrectnessResult cr = verifyAgreement(model, r.cold_result, r.warm_result, d, cfg.tolerance);

        if (!cr.passed) {
            std::cerr << "  FEASIBILITY FAILURE Day " << d << ": " << cr.error_message << "\n";
            results[static_cast<std::size_t>(d)].correctness_ok = false;
            all_correct = false;
        }
    }

    if (!all_correct) {
        std::cerr << "\n*** BENCHMARK FAILED: Correctness check failed ***\n\n";
    }

    // =======================================================================
    // COMPUTE AGGREGATES
    // =======================================================================
    double cold_cumulative = 0.0;
    double warm_cumulative = 0.0;
    int total_cold_iters = 0;
    int total_warm_iters = 0;
    int total_warm_attempt_iters = 0;
    int total_warm_fallback_iters = 0;

    double cold_steady_state = 0.0;
    double warm_steady_state = 0.0;

    int cold_optimal_days = 0;
    int warm_optimal_days = 0;
    std::vector<double> cold_per_day, warm_per_day;

    for (int d = 0; d < cfg.num_days; ++d) {
        const auto& r = results[static_cast<std::size_t>(d)];
        cold_cumulative += r.cold_time;
        warm_cumulative += r.warm_time;
        total_cold_iters += r.cold_iters;
        total_warm_iters += r.warm_iters;
        total_warm_attempt_iters += r.warm_attempt_iters;
        total_warm_fallback_iters += r.warm_fallback_iters;
        cold_per_day.push_back(r.cold_time);
        warm_per_day.push_back(r.warm_time);

        if (d > 0) {
            cold_steady_state += r.cold_time;
            warm_steady_state += r.warm_time;
        }

        if (r.cold_result.status == SolveStatus::kOptimal) {
            cold_optimal_days++;
        }
        if (r.warm_result.status == SolveStatus::kOptimal) {
            warm_optimal_days++;
        }
    }

    double speedup = (warm_cumulative > 0.0) ? (cold_cumulative / warm_cumulative) : 0.0;
    double steady_speedup = (warm_steady_state > 0.0) ? (cold_steady_state / warm_steady_state) : 0.0;
    double iter_ratio = (total_warm_iters > 0) ?
        static_cast<double>(total_cold_iters) / static_cast<double>(total_warm_iters) : 0.0;

    bool benchmark_valid = all_correct && (cold_optimal_days == cfg.num_days) && (warm_optimal_days == cfg.num_days);

    // =======================================================================
    // OUTPUT
    // =======================================================================
    if (cfg.json_output) {
        std::cout << "{\n";
        std::cout << "  \"instance\": \"synthetic_production_planning\",\n";
        std::cout << "  \"num_vars\": " << base.numVars() << ",\n";
        std::cout << "  \"num_rows\": " << base.numRows() << ",\n";
        std::cout << "  \"days\": " << cfg.num_days << ",\n";
        std::cout << "  \"seed\": " << cfg.seed << ",\n";
        std::cout << "  \"tolerance\": " << std::setprecision(12) << cfg.tolerance << ",\n";
        std::cout << "  \"perturb_rhs\": " << std::setprecision(4) << cfg.perturb_rhs << ",\n";
        std::cout << "  \"perturb_bounds\": " << std::setprecision(4) << cfg.perturb_bounds << ",\n";
        std::cout << "  \"cold_optimal_days\": " << cold_optimal_days << ",\n";
        std::cout << "  \"warm_optimal_days\": " << warm_optimal_days << ",\n";
        std::cout << "  \"benchmark_valid\": " << (benchmark_valid ? "true" : "false") << ",\n";
        std::cout << "  \"cold\": {\n";
        std::cout << "    \"cumulative_seconds\": " << std::setprecision(9) << cold_cumulative << ",\n";
        std::cout << "    \"total_iterations\": " << total_cold_iters << ",\n";
        std::cout << "    \"per_day_seconds\": " << jsonArray(cold_per_day) << "\n";
        std::cout << "  },\n";
        std::cout << "  \"warm\": {\n";
        std::cout << "    \"cumulative_seconds\": " << std::setprecision(9) << warm_cumulative << ",\n";
        std::cout << "    \"total_iterations\": " << total_warm_iters << ",\n";
        std::cout << "    \"per_day_seconds\": " << jsonArray(warm_per_day) << "\n";
        std::cout << "  },\n";
        std::cout << "  \"cold_steady_state_seconds\": " << std::setprecision(9) << cold_steady_state << ",\n";
        std::cout << "  \"warm_steady_state_seconds\": " << std::setprecision(9) << warm_steady_state << ",\n";
        std::cout << "  \"steady_state_speedup\": " << std::setprecision(4) << steady_speedup << ",\n";
        std::cout << "  \"warm_attempt_iterations\": " << total_warm_attempt_iters << ",\n";
        std::cout << "  \"warm_fallback_iterations\": " << total_warm_fallback_iters << ",\n";
        std::cout << "  \"warm_start_attempts\": " << warm_start_attempts << ",\n";
        std::cout << "  \"warm_start_successes\": " << warm_start_successes << ",\n";
        std::cout << "  \"warm_start_fallbacks\": " << warm_start_fallbacks << ",\n";
        std::cout << "  \"speedup\": " << std::setprecision(4) << speedup << ",\n";
        std::cout << "  \"iteration_ratio\": " << std::setprecision(4) << iter_ratio << ",\n";
        std::cout << "  \"correctness\": " << (all_correct ? "true" : "false") << "\n";
        std::cout << "}\n";
    } else {
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "  PRAMAAN Rolling-Horizon Warm-Start Benchmark\n";
        std::cout << "================================================================\n";
        std::cout << "Instance:       synthetic_production_planning\n";
        std::cout << "Variables:      " << base.numVars() << "\n";
        std::cout << "Constraints:    " << base.numRows() << "\n";
        std::cout << "Days:           " << cfg.num_days << "\n";
        std::cout << "Seed:           " << cfg.seed << "\n";
        std::cout << "Tolerance:      " << cfg.tolerance << "\n";
        std::cout << "Perturbation:   ±" << std::lround(cfg.perturb_rhs * 100.0)
                  << "% RHS / ±" << std::lround(cfg.perturb_bounds * 100.0)
                  << "% bounds (deterministic LCG)\n";
        std::cout << "\n";
        std::cout << "--- Timing Results ---\n";
        std::cout << "Cold cumulative:  " << std::fixed << std::setprecision(6)
                  << cold_cumulative << " s\n";
        std::cout << "Warm cumulative:  " << std::fixed << std::setprecision(6)
                  << warm_cumulative << " s\n";
        std::cout << "\n";
        std::cout << "Steady-state cold cumulative (excluding Day 0): " << std::fixed << std::setprecision(6)
                  << cold_steady_state << " s\n";
        std::cout << "Steady-state warm cumulative (excluding Day 0): " << std::fixed << std::setprecision(6)
                  << warm_steady_state << " s\n";
        std::cout << "Steady-state speedup: " << std::fixed << std::setprecision(2)
                  << steady_speedup << "x\n";
        std::cout << "\n";
        std::cout << "Cold total iterations: " << total_cold_iters << "\n";
        std::cout << "Warm total iterations: " << total_warm_iters << "\n";
        std::cout << "Iteration ratio (cold/warm): " << std::fixed << std::setprecision(2)
                  << iter_ratio << "x\n";
        std::cout << "\n";
        std::cout << "--- Warm-Start Statistics ---\n";
        std::cout << "Warm-start attempts:   " << warm_start_attempts << "\n";
        std::cout << "Warm-start successes:  " << warm_start_successes << "\n";
        std::cout << "Warm-start fallbacks:  " << warm_start_fallbacks << "\n";
        std::cout << "\n";
        std::cout << "--- Correctness ---\n";
        std::cout << "All days correct:  " << (all_correct ? "YES" : "NO") << "\n";
        std::cout << "Benchmark valid:   " << (benchmark_valid ? "YES" : "NO") << "\n";
        std::cout << "\n";
        std::cout << "=== Cumulative speedup: " << std::fixed << std::setprecision(2)
                  << speedup << "x ===\n";
        std::cout << "\n";
        std::cout << "(speedup = cold_cumulative_time / warm_cumulative_time,\n";
        std::cout << " measured from actual wall-clock data)\n";
        std::cout << "================================================================\n";

        // Per-day detail table
        std::cout << "\n--- Per-Day Detail ---\n";
        std::cout << std::setw(5) << "Day"
                  << std::setw(14) << "Cold(s)"
                  << std::setw(14) << "Warm(s)"
                  << std::setw(12) << "Cold Iter"
                  << std::setw(12) << "Warm Iter"
                  << std::setw(10) << "WS"
                  << std::setw(14) << "Cold Obj"
                  << std::setw(14) << "Warm Obj"
                  << "\n";
        std::cout << std::string(95, '-') << "\n";
        for (int d = 0; d < cfg.num_days; ++d) {
            const auto& r = results[static_cast<std::size_t>(d)];
            std::string ws_status;
            if (d == 0) ws_status = "cold";
            else if (r.warm_fell_back) ws_status = "fallback";
            else if (r.warm_succeeded) ws_status = "warm";
            else ws_status = "???";

            std::cout << std::setw(5) << d
                      << std::setw(14) << std::fixed << std::setprecision(6) << r.cold_time
                      << std::setw(14) << std::fixed << std::setprecision(6) << r.warm_time
                      << std::setw(12) << r.cold_iters
                      << std::setw(12) << r.warm_iters
                      << std::setw(10) << ws_status
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.cold_obj
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.warm_obj
                      << "\n";
        }
    }

    return all_correct ? 0 : 1;
}

// ============================================================================
// CLI
// ============================================================================

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --days N         Number of simulated days (default: 30)\n"
              << "  --seed S         Random seed (default: 12345)\n"
              << "  --tolerance T    Solver tolerance (default: 1e-8)\n"
              << "  --products P     Number of products/variables (default: 50)\n"
              << "  --resources R    Number of resource constraints (default: 30)\n"
              << "  --groups G       Number of demand groups (default: 5)\n"
              << "  --perturb-rhs P  RHS perturbation magnitude (default: 0.10)\n"
              << "  --perturb-bounds P Bounds perturbation magnitude (default: 0.08)\n"
              << "  --json           Output results as JSON\n"
              << "  --help           Show this help\n";
}

int main(int argc, char* argv[]) {
    BenchmarkConfig cfg;

    auto parse_int = [](const std::string& s) -> int {
        try {
            std::size_t pos = 0;
            int val = std::stoi(s, &pos);
            if (pos != s.size()) throw std::invalid_argument("trailing garbage");
            return val;
        } catch (...) {
            std::cerr << "Error: Invalid integer argument: " << s << "\n";
            std::exit(1);
        }
    };
    auto parse_double = [](const std::string& s) -> double {
        try {
            std::size_t pos = 0;
            double val = std::stod(s, &pos);
            if (pos != s.size()) throw std::invalid_argument("trailing garbage");
            return val;
        } catch (...) {
            std::cerr << "Error: Invalid floating-point argument: " << s << "\n";
            std::exit(1);
        }
    };
    auto parse_ll = [](const std::string& s) -> long long {
        try {
            std::size_t pos = 0;
            long long val = std::stoll(s, &pos);
            if (pos != s.size()) throw std::invalid_argument("trailing garbage");
            return val;
        } catch (...) {
            std::cerr << "Error: Invalid integer argument: " << s << "\n";
            std::exit(1);
        }
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--days" && i + 1 < argc) {
            cfg.num_days = parse_int(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            long long s = parse_ll(argv[++i]);
            if (s < 0) {
                std::cerr << "Error: --seed must be >= 0\n";
                return 1;
            }
            cfg.seed = static_cast<uint64_t>(s);
        } else if (arg == "--tolerance" && i + 1 < argc) {
            cfg.tolerance = parse_double(argv[++i]);
        } else if (arg == "--products" && i + 1 < argc) {
            cfg.num_products = parse_int(argv[++i]);
        } else if (arg == "--resources" && i + 1 < argc) {
            cfg.num_resources = parse_int(argv[++i]);
        } else if (arg == "--groups" && i + 1 < argc) {
            cfg.num_demand_groups = parse_int(argv[++i]);
        } else if (arg == "--perturb-rhs" && i + 1 < argc) {
            cfg.perturb_rhs = parse_double(argv[++i]);
        } else if (arg == "--perturb-bounds" && i + 1 < argc) {
            cfg.perturb_bounds = parse_double(argv[++i]);
        } else if (arg == "--json") {
            cfg.json_output = true;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (cfg.num_days < 1) {
        std::cerr << "Error: --days must be >= 1\n";
        return 1;
    }
    if (cfg.num_products < 1) {
        std::cerr << "Error: --products must be >= 1\n";
        return 1;
    }
    if (cfg.num_resources < 0) {
        std::cerr << "Error: --resources must be >= 0\n";
        return 1;
    }
    if (cfg.num_demand_groups < 0) {
        std::cerr << "Error: --groups must be >= 0\n";
        return 1;
    }
    if (cfg.num_demand_groups > cfg.num_products) {
        std::cerr << "Error: --groups must be <= --products\n";
        return 1;
    }
    if (!(cfg.tolerance > 0.0) || std::isinf(cfg.tolerance) || std::isnan(cfg.tolerance)) {
        std::cerr << "Error: --tolerance must be finite and strictly > 0\n";
        return 1;
    }
    if (cfg.perturb_rhs < 0.0 || cfg.perturb_rhs >= 1.0 || std::isinf(cfg.perturb_rhs) || std::isnan(cfg.perturb_rhs)) {
        std::cerr << "Error: --perturb-rhs must be >= 0 and < 1\n";
        return 1;
    }
    if (cfg.perturb_bounds < 0.0 || cfg.perturb_bounds >= 1.0 || std::isinf(cfg.perturb_bounds) || std::isnan(cfg.perturb_bounds)) {
        std::cerr << "Error: --perturb-bounds must be >= 0 and < 1\n";
        return 1;
    }

    return runBenchmark(cfg);
}
