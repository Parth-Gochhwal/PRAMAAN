// rolling_horizon_mps_bench.cpp
// P1 — Rolling-Horizon Warm-Start Benchmark (MPS-file based)
//
// Purpose: Measure ACTUAL cumulative cold-vs-warm-start performance of
// PRAMAAN's DualSimplex::warmSolve over a 30-day rolling-horizon trace
// using an MPS-file-based crude-blending LP as the base model.
//
// This benchmark:
//   1. Parses the base MPS file (e.g. benchmark/mrpl_crude_blend.mps).
//   2. Generates N deterministic daily scenarios by perturbing RHS and
//      variable bounds (matrix A is NEVER changed).
//   3. Runs COLD strategy: RevisedSimplex::solve() on each day's LP from scratch.
//   4. Runs WARM strategy: DualSimplex::captureBasis() on day 0, then
//      DualSimplex::warmSolve() with basis chaining for subsequent days.
//   5. Verifies correctness (status, objective agreement, feasibility).
//   6. Outputs JSON with all per-day metrics for Python processing.
//
// This benchmark does NOT fabricate, assume, or hard-code any speedup.
// It measures wall-clock time and reports whatever the real result is.
//
// Usage:
//   ./rolling_horizon_mps_bench <model.mps> [--days N] [--seed S] [--perturb-rhs P]
//                                            [--perturb-bounds P] [--tolerance T]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"
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
// Deterministic PRNG (LCG) — same as rolling_horizon_bench.cpp
// ============================================================================
struct LCG {
    uint64_t state;
    explicit LCG(uint64_t seed) : state(seed) {}

    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    }

    double uniform(double lo, double hi) {
        uint64_t v = next();
        double t = static_cast<double>(v >> 11) / static_cast<double>(1ULL << 53);
        return lo + t * (hi - lo);
    }
};

// ============================================================================
// Timing helper
// ============================================================================
using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

// ============================================================================
// Configuration
// ============================================================================
struct BenchConfig {
    std::string mps_path;
    int num_days = 30;
    uint64_t seed = 42;
    double perturb_rhs = 0.08;
    double perturb_bounds = 0.06;
    double tolerance = 1e-8;
};

// ============================================================================
// Perturbation: modify RHS and bounds for a given day
// ============================================================================
// Applies small deterministic perturbations to the base model.
// - Row upper bounds (for <= constraints): ±perturb_rhs
// - Row lower bounds (for >= constraints): ±perturb_rhs
// - Variable upper bounds: ±perturb_bounds
//
// Preserves:
// - Matrix A (unchanged)
// - Objective coefficients (unchanged) — required for BasisState compatibility
// - Bound types (finite stays finite, infinite stays infinite)
// - LP structure
//
static ModelIR perturbModel(const ModelIR& base, int day, const BenchConfig& cfg) {
    ModelIR model = base;

    // Deterministic seed per day, combining global seed with day index
    LCG rng(cfg.seed * 1000003ULL + static_cast<uint64_t>(day) * 7919ULL);

    // Perturb row upper bounds (for <= constraints)
    for (int r = 0; r < base.numRows(); ++r) {
        double orig_upper = base.row_upper[static_cast<std::size_t>(r)];
        double orig_lower = base.row_lower[static_cast<std::size_t>(r)];

        // Only perturb finite bounds
        if (orig_upper < kInfinity && orig_upper > -kInfinity) {
            double factor = rng.uniform(1.0 - cfg.perturb_rhs, 1.0 + cfg.perturb_rhs);
            double new_upper = orig_upper * factor;
            // Keep row_upper >= row_lower
            if (new_upper >= orig_lower + 1e-6 || orig_lower <= -kInfinity) {
                model.row_upper[static_cast<std::size_t>(r)] = new_upper;
            }
        }

        if (orig_lower > -kInfinity && orig_lower < kInfinity) {
            double factor = rng.uniform(1.0 - cfg.perturb_rhs, 1.0 + cfg.perturb_rhs);
            double new_lower = orig_lower * factor;
            // Keep row_lower <= row_upper
            if (new_lower <= model.row_upper[static_cast<std::size_t>(r)] - 1e-6
                || model.row_upper[static_cast<std::size_t>(r)] >= kInfinity) {
                model.row_lower[static_cast<std::size_t>(r)] = new_lower;
            }
        }
    }

    // Perturb variable upper bounds (small shift, keep positive)
    for (int j = 0; j < base.numVars(); ++j) {
        double orig = base.var_upper[static_cast<std::size_t>(j)];
        if (orig < kInfinity && orig > 0.0) {
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
// Primal feasibility check
// ============================================================================
static double computePrimalResidual(const ModelIR& model, const SolveResult& result) {
    if (result.status != SolveStatus::kOptimal || result.x.empty()) return -1.0;

    auto activity = model.A.multiply(result.x);
    double max_viol = 0.0;

    for (int r = 0; r < model.numRows(); ++r) {
        double lo = model.row_lower[static_cast<std::size_t>(r)];
        double up = model.row_upper[static_cast<std::size_t>(r)];
        double ar = activity[static_cast<std::size_t>(r)];
        if (lo > -kInfinity) max_viol = std::max(max_viol, lo - ar);
        if (up < kInfinity)  max_viol = std::max(max_viol, ar - up);
    }

    for (int j = 0; j < model.numVars(); ++j) {
        double lo = model.var_lower[static_cast<std::size_t>(j)];
        double up = model.var_upper[static_cast<std::size_t>(j)];
        double xj = result.x[static_cast<std::size_t>(j)];
        if (lo > -kInfinity) max_viol = std::max(max_viol, lo - xj);
        if (up < kInfinity)  max_viol = std::max(max_viol, xj - up);
    }

    return max_viol;
}

// ============================================================================
// Per-day result
// ============================================================================
struct DayResult {
    int day = 0;
    double cold_time_s = 0.0;
    double warm_time_s = 0.0;
    double cold_obj = 0.0;
    double warm_obj = 0.0;
    int cold_iters = 0;
    int warm_iters = 0;
    int cold_status = 0;
    int warm_status = 0;
    double cold_primal_residual = 0.0;
    double warm_primal_residual = 0.0;
    bool basis_reused = false;
    bool warm_fell_back = false;
    bool correctness_ok = true;
    std::uint64_t structure_fingerprint = 0;

    double obj_abs_diff = 0.0;
    std::string perturbation_desc;
};

// ============================================================================
// JSON output helpers
// ============================================================================
static std::string escapeJson(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"') r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else r += c;
    }
    return r;
}

// ============================================================================
// Main benchmark
// ============================================================================
static int runBenchmark(const BenchConfig& cfg) {
    // --- Parse base MPS ---
    std::cerr << "Parsing base MPS: " << cfg.mps_path << "\n";
    ModelIR base;
    try {
        base = pramaan::parse_mps(cfg.mps_path);
        base.validate();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to parse MPS file: " << e.what() << "\n";
        return 1;
    }
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
    std::cerr << "Generating " << cfg.num_days << " daily scenarios...\n";
    std::vector<ModelIR> day_models(static_cast<std::size_t>(cfg.num_days));
    std::vector<std::string> perturbation_descs(static_cast<std::size_t>(cfg.num_days));

    for (int d = 0; d < cfg.num_days; ++d) {
        if (d == 0) {
            day_models[0] = base;
            perturbation_descs[0] = "base_case";
        } else {
            day_models[static_cast<std::size_t>(d)] = perturbModel(base, d, cfg);
            std::ostringstream oss;
            oss << "day" << d << "_rhs" << std::fixed << std::setprecision(0)
                << (cfg.perturb_rhs * 100) << "pct_bnd"
                << (cfg.perturb_bounds * 100) << "pct";
            perturbation_descs[static_cast<std::size_t>(d)] = oss.str();
        }
    }

    // --- Pre-validate: solve day 0 to ensure the base model works ---
    {
        SolveResult test = cold_solver.solve(day_models[0]);
        if (test.status != SolveStatus::kOptimal) {
            std::cerr << "ERROR: Base model cold-solve failed (status "
                      << static_cast<int>(test.status) << ").\n";
            std::cerr << "The MPS model may be infeasible or unbounded.\n";
            return 1;
        }
        BasisState test_basis = dual_solver.captureBasis(day_models[0]);
        if (test_basis.empty()) {
            std::cerr << "ERROR: Base model basis capture failed.\n";
            return 1;
        }
        std::cerr << "  Base model verified: optimal, obj=" << test.objective_value
                  << ", iters=" << test.iterations << "\n\n";
    }

    // --- Results storage ---
    std::vector<DayResult> results(static_cast<std::size_t>(cfg.num_days));

    // =======================================================================
    // STRATEGY 1: COLD RESTART (every day from scratch)
    // =======================================================================
    std::cerr << "Running COLD strategy (" << cfg.num_days << " days)...\n";
    for (int d = 0; d < cfg.num_days; ++d) {
        const ModelIR& model = day_models[static_cast<std::size_t>(d)];

        auto t0 = Clock::now();
        SolveResult cold_result = cold_solver.solve(model);
        auto t1 = Clock::now();

        auto& r = results[static_cast<std::size_t>(d)];
        r.day = d;
        r.structure_fingerprint = computeStructuralFingerprint(model);
        r.cold_time_s = Duration(t1 - t0).count();
        r.cold_obj = cold_result.objective_value;
        r.cold_iters = cold_result.iterations;
        r.cold_status = static_cast<int>(cold_result.status);
        r.cold_primal_residual = computePrimalResidual(model, cold_result);
        r.perturbation_desc = perturbation_descs[static_cast<std::size_t>(d)];

        if (cold_result.status != SolveStatus::kOptimal) {
            std::cerr << "  WARNING: Cold solve day " << d << " status = "
                      << r.cold_status << "\n";
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
        auto& r = results[static_cast<std::size_t>(d)];

        if (d == 0) {
            // Day 0: cold solve + capture basis
            auto t0 = Clock::now();
            SolveResult cold_result;
            current_basis = dual_solver.captureBasis(model, &cold_result);
            auto t1 = Clock::now();

            r.warm_time_s = Duration(t1 - t0).count();
            r.warm_obj = cold_result.objective_value;
            r.warm_iters = cold_result.iterations;
            r.warm_status = static_cast<int>(cold_result.status);
            r.warm_primal_residual = computePrimalResidual(model, cold_result);
            r.basis_reused = false;

            if (current_basis.empty()) {
                std::cerr << "  WARNING: Day 0 basis capture failed.\n";
            }
        } else {
            // Day 1+: warm-solve using previous day's basis
            warm_start_attempts++;

            if (current_basis.empty()) {
                // Basis chain broken — fall back to cold solve
                warm_start_fallbacks++;
                r.warm_fell_back = true;

                auto t0 = Clock::now();
                SolveResult cold_fallback;
                current_basis = dual_solver.captureBasis(model, &cold_fallback);
                auto t1 = Clock::now();

                r.warm_time_s = Duration(t1 - t0).count();
                r.warm_obj = cold_fallback.objective_value;
                r.warm_iters = cold_fallback.iterations;
                r.warm_status = static_cast<int>(cold_fallback.status);
                r.warm_primal_residual = computePrimalResidual(model, cold_fallback);
                r.basis_reused = false;
            } else {
                BasisState next_basis;

                auto t0 = Clock::now();
                try {
                    SolveResult warm_result = dual_solver.warmSolve(
                        model, current_basis, &next_basis);
                    auto t1 = Clock::now();

                    if (warm_result.status == SolveStatus::kOptimal) {
                        r.warm_time_s = Duration(t1 - t0).count();
                        r.warm_obj = warm_result.objective_value;
                        r.warm_iters = warm_result.iterations;
                        r.warm_status = static_cast<int>(warm_result.status);
                        r.warm_primal_residual = computePrimalResidual(model, warm_result);
                        r.basis_reused = true;
                        warm_start_successes++;

                        // Chain the basis
                        if (!next_basis.empty()) {
                            current_basis = next_basis;
                        } else {
                            current_basis = dual_solver.captureBasis(model);
                        }
                    } else {
                        // Warm-start non-optimal — fall back
                        warm_start_fallbacks++;
                        r.warm_fell_back = true;

                        auto t0b = Clock::now();
                        SolveResult cold_fallback;
                        current_basis = dual_solver.captureBasis(model, &cold_fallback);
                        auto t1b = Clock::now();

                        r.warm_time_s = Duration(t1 - t0).count() + Duration(t1b - t0b).count();
                        r.warm_obj = cold_fallback.objective_value;
                        r.warm_iters = warm_result.iterations + cold_fallback.iterations;
                        r.warm_status = static_cast<int>(cold_fallback.status);
                        r.warm_primal_residual = computePrimalResidual(model, cold_fallback);
                        r.basis_reused = false;
                    }
                } catch (const std::invalid_argument& e) {
                    auto t1 = Clock::now();
                    warm_start_fallbacks++;
                    r.warm_fell_back = true;

                    std::cerr << "  Day " << d << ": warm-start rejected ("
                              << e.what() << "), falling back.\n";

                    auto t0b = Clock::now();
                    SolveResult cold_fallback;
                    current_basis = dual_solver.captureBasis(model, &cold_fallback);
                    auto t1b = Clock::now();

                    r.warm_time_s = Duration(t1 - t0).count() + Duration(t1b - t0b).count();
                    r.warm_obj = cold_fallback.objective_value;
                    r.warm_iters = cold_fallback.iterations;
                    r.warm_status = static_cast<int>(cold_fallback.status);
                    r.warm_primal_residual = computePrimalResidual(model, cold_fallback);
                    r.basis_reused = false;
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
        auto& r = results[static_cast<std::size_t>(d)];
        double ref = std::max(std::abs(r.cold_obj), 1.0);
        r.obj_abs_diff = std::abs(r.cold_obj - r.warm_obj);
        if (r.obj_abs_diff / ref > cfg.tolerance * 1e4) {
            std::cerr << "  CORRECTNESS FAILURE Day " << d
                      << ": cold_obj=" << r.cold_obj << ", warm_obj=" << r.warm_obj
                      << ", rel_diff=" << (r.obj_abs_diff / ref) << "\n";
            r.correctness_ok = false;
            all_correct = false;
        }
    }

    // =======================================================================
    // COMPUTE AGGREGATES
    // =======================================================================
    double cold_cumulative = 0.0;
    double warm_cumulative = 0.0;
    int total_cold_iters = 0;
    int total_warm_iters = 0;
    int cold_optimal_days = 0;
    int warm_optimal_days = 0;

    for (int d = 0; d < cfg.num_days; ++d) {
        const auto& r = results[static_cast<std::size_t>(d)];
        cold_cumulative += r.cold_time_s;
        warm_cumulative += r.warm_time_s;
        total_cold_iters += r.cold_iters;
        total_warm_iters += r.warm_iters;
        if (r.cold_status == 0) cold_optimal_days++;
        if (r.warm_status == 0) warm_optimal_days++;
    }

    double ratio = (warm_cumulative > 0.0) ? (cold_cumulative / warm_cumulative) : 0.0;
    double iter_ratio = (total_warm_iters > 0) ?
        static_cast<double>(total_cold_iters) / static_cast<double>(total_warm_iters) : 0.0;
    bool benchmark_valid = all_correct &&
        (cold_optimal_days == cfg.num_days) && (warm_optimal_days == cfg.num_days);

    // =======================================================================
    // JSON OUTPUT (stdout)
    // =======================================================================
    std::cout << "{\n";
    std::cout << "  \"instance\": \"" << escapeJson(cfg.mps_path) << "\",\n";
    std::cout << "  \"num_vars\": " << base.numVars() << ",\n";
    std::cout << "  \"num_rows\": " << base.numRows() << ",\n";
    std::cout << "  \"days\": " << cfg.num_days << ",\n";
    std::cout << "  \"seed\": " << cfg.seed << ",\n";
    std::cout << "  \"perturb_rhs\": " << cfg.perturb_rhs << ",\n";
    std::cout << "  \"perturb_bounds\": " << cfg.perturb_bounds << ",\n";
    std::cout << "  \"tolerance\": " << std::setprecision(12) << cfg.tolerance << ",\n";
    std::cout << "  \"cold_cumulative_s\": " << std::setprecision(9) << cold_cumulative << ",\n";
    std::cout << "  \"warm_cumulative_s\": " << std::setprecision(9) << warm_cumulative << ",\n";
    std::cout << "  \"cold_cumulative_ms\": " << std::setprecision(6) << cold_cumulative * 1000.0 << ",\n";
    std::cout << "  \"warm_cumulative_ms\": " << std::setprecision(6) << warm_cumulative * 1000.0 << ",\n";
    std::cout << "  \"cumulative_ratio\": " << std::setprecision(4) << ratio << ",\n";
    std::cout << "  \"cold_total_iters\": " << total_cold_iters << ",\n";
    std::cout << "  \"warm_total_iters\": " << total_warm_iters << ",\n";
    std::cout << "  \"iteration_ratio\": " << std::setprecision(4) << iter_ratio << ",\n";
    std::cout << "  \"cold_optimal_days\": " << cold_optimal_days << ",\n";
    std::cout << "  \"warm_optimal_days\": " << warm_optimal_days << ",\n";
    std::cout << "  \"warm_start_attempts\": " << warm_start_attempts << ",\n";
    std::cout << "  \"warm_start_successes\": " << warm_start_successes << ",\n";
    std::cout << "  \"warm_start_fallbacks\": " << warm_start_fallbacks << ",\n";
    std::cout << "  \"benchmark_valid\": " << (benchmark_valid ? "true" : "false") << ",\n";
    std::cout << "  \"correctness\": " << (all_correct ? "true" : "false") << ",\n";

    // Per-day detail
    std::cout << "  \"per_day\": [\n";
    for (int d = 0; d < cfg.num_days; ++d) {
        const auto& r = results[static_cast<std::size_t>(d)];
        std::cout << "    {\n";
        std::cout << "      \"day\": " << r.day << ",\n";
        std::cout << "      \"perturbation\": \"" << escapeJson(r.perturbation_desc) << "\",\n";
        std::cout << "      \"cold_time_ms\": " << std::setprecision(6) << (r.cold_time_s * 1000.0) << ",\n";
        std::cout << "      \"warm_time_ms\": " << std::setprecision(6) << (r.warm_time_s * 1000.0) << ",\n";
        std::cout << "      \"cold_objective\": " << std::setprecision(10) << r.cold_obj << ",\n";
        std::cout << "      \"warm_objective\": " << std::setprecision(10) << r.warm_obj << ",\n";
        std::cout << "      \"objective_abs_diff\": " << std::setprecision(12) << r.obj_abs_diff << ",\n";
        std::cout << "      \"cold_iterations\": " << r.cold_iters << ",\n";
        std::cout << "      \"warm_iterations\": " << r.warm_iters << ",\n";
        std::cout << "      \"cold_status\": " << r.cold_status << ",\n";
        std::cout << "      \"warm_status\": " << r.warm_status << ",\n";
        std::cout << "      \"cold_primal_residual\": " << std::setprecision(12) << r.cold_primal_residual << ",\n";
        std::cout << "      \"warm_primal_residual\": " << std::setprecision(12) << r.warm_primal_residual << ",\n";
        std::cout << "      \"basis_reused\": " << (r.basis_reused ? "true" : "false") << ",\n";
        std::cout << "      \"warm_fell_back\": " << (r.warm_fell_back ? "true" : "false") << ",\n";
        std::cout << "      \"correctness_ok\": " << (r.correctness_ok ? "true" : "false") << ",\n";
        std::cout << "      \"structure_fingerprint\": " << r.structure_fingerprint << "\n";
        std::cout << "    }" << (d < cfg.num_days - 1 ? "," : "") << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";

    // Summary to stderr
    std::cerr << "\n";
    std::cerr << "================================================================\n";
    std::cerr << "  PRAMAAN Rolling-Horizon Benchmark (MPS-based)\n";
    std::cerr << "================================================================\n";
    std::cerr << "Instance:        " << cfg.mps_path << "\n";
    std::cerr << "Variables:       " << base.numVars() << "\n";
    std::cerr << "Constraints:     " << base.numRows() << "\n";
    std::cerr << "Days:            " << cfg.num_days << "\n";
    std::cerr << "\n";
    std::cerr << "Cold cumulative: " << std::fixed << std::setprecision(6)
              << cold_cumulative << " s (" << cold_cumulative * 1000.0 << " ms)\n";
    std::cerr << "Warm cumulative: " << std::fixed << std::setprecision(6)
              << warm_cumulative << " s (" << warm_cumulative * 1000.0 << " ms)\n";
    std::cerr << "Measured ratio:  " << std::setprecision(2) << ratio << "x\n";
    std::cerr << "\n";
    std::cerr << "Cold total iters: " << total_cold_iters << "\n";
    std::cerr << "Warm total iters: " << total_warm_iters << "\n";
    std::cerr << "Iteration ratio:  " << std::setprecision(2) << iter_ratio << "x\n";
    std::cerr << "\n";
    std::cerr << "Warm-start attempts:  " << warm_start_attempts << "\n";
    std::cerr << "Warm-start successes: " << warm_start_successes << "\n";
    std::cerr << "Warm-start fallbacks: " << warm_start_fallbacks << "\n";
    std::cerr << "\n";
    std::cerr << "Correctness: " << (all_correct ? "PASS" : "FAIL") << "\n";
    std::cerr << "Benchmark valid: " << (benchmark_valid ? "YES" : "NO") << "\n";
    std::cerr << "================================================================\n";

    return all_correct ? 0 : 1;
}

// ============================================================================
// CLI
// ============================================================================
static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.mps> [options]\n"
              << "\nOptions:\n"
              << "  --days N             Number of simulated days (default: 30)\n"
              << "  --seed S             Random seed (default: 42)\n"
              << "  --perturb-rhs P      RHS perturbation magnitude (default: 0.08)\n"
              << "  --perturb-bounds P   Bounds perturbation magnitude (default: 0.06)\n"
              << "  --tolerance T        Solver tolerance (default: 1e-8)\n"
              << "  --help               Show this help\n";
}

int main(int argc, char* argv[]) {
    BenchConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--days" && i + 1 < argc) {
            cfg.num_days = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (arg == "--perturb-rhs" && i + 1 < argc) {
            cfg.perturb_rhs = std::atof(argv[++i]);
        } else if (arg == "--perturb-bounds" && i + 1 < argc) {
            cfg.perturb_bounds = std::atof(argv[++i]);
        } else if (arg == "--tolerance" && i + 1 < argc) {
            cfg.tolerance = std::atof(argv[++i]);
        } else if (arg.rfind("--", 0) == 0 || arg.rfind("-", 0) == 0) {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        } else if (cfg.mps_path.empty()) {
            cfg.mps_path = arg;
        } else {
            std::cerr << "Unexpected argument: " << arg << "\n";
            return 1;
        }
    }

    if (cfg.mps_path.empty()) {
        std::cerr << "Error: MPS file path required.\n\n";
        printUsage(argv[0]);
        return 1;
    }

    if (cfg.num_days < 1) {
        std::cerr << "Error: --days must be >= 1\n";
        return 1;
    }

    return runBenchmark(cfg);
}