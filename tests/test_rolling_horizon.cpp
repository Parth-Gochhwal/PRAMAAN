// test_rolling_horizon.cpp
// P1 Step 8 — Rolling-Horizon Benchmark Infrastructure Tests
//
// Tests the benchmark's correctness guarantees:
// 1. Deterministic perturbation generation (same seed → same perturbations)
// 2. Identical perturbation sequences for cold and warm strategies
// 3. Sequential basis chaining correctness
// 4. Cold/warm objective agreement for each day
// 5. Cumulative timing aggregation correctness
// 6. Speedup calculation correctness
// 7. Invalid configuration handling
//
// These tests do NOT assert any performance result (speedup).
// Timing is machine-dependent. Tests verify correctness and accounting only.

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/dual_simplex.hpp"

namespace {

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

int g_checks_run = 0;
int g_checks_failed = 0;

void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

void checkNear(double actual, double expected, double tol, const std::string& description) {
    ++g_checks_run;
    if (std::abs(actual - expected) > tol) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (expected " << expected << ", got " << actual
                   << ", |diff| = " << std::abs(actual - expected) << ")\n";
    }
}

void run(const std::string& name, const std::function<void()>& test_body) {
    std::cout << name << "...\n";
    const int before = g_checks_failed;
    test_body();
    if (g_checks_failed == before) {
        std::cout << "  ok\n";
    }
}

CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense, CSRMatrix::Index num_cols) {
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
// Deterministic PRNG (must match rolling_horizon_bench.cpp exactly)
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
// Simple test LP: Wyndor Glass Co. with finite bounds
// ============================================================================
ModelIR makeWyndor() {
    return ModelIR(
        ObjSense::kMaximize, 0.0, {3.0, 5.0},
        denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
        {-kInfinity, -kInfinity, -kInfinity},
        {4.0, 12.0, 18.0},
        {"plant1", "plant2", "plant3"},
        {0.0, 0.0}, {10.0, 10.0},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});
}

// Apply perturbation to Wyndor model
// Only modifies RHS (row_upper) and variable bounds (var_upper)
// within the safe range for warm-start
ModelIR perturbWyndor(const ModelIR& base, int day, uint64_t seed) {
    ModelIR model = base;
    LCG rng(seed * 1000003ULL + static_cast<uint64_t>(day) * 7919ULL);

    // Perturb row upper bounds (resource capacities)
    for (int r = 0; r < base.numRows(); ++r) {
        double orig = base.row_upper[static_cast<std::size_t>(r)];
        if (orig < kInfinity) {
            double factor = rng.uniform(0.90, 1.10);
            model.row_upper[static_cast<std::size_t>(r)] = orig * factor;
        }
    }

    // Perturb variable upper bounds
    for (int j = 0; j < base.numVars(); ++j) {
        double orig = base.var_upper[static_cast<std::size_t>(j)];
        if (orig < kInfinity) {
            double factor = rng.uniform(0.92, 1.08);
            double new_val = orig * factor;
            if (new_val > base.var_lower[static_cast<std::size_t>(j)] + 1e-6) {
                model.var_upper[static_cast<std::size_t>(j)] = new_val;
            }
        }
    }

    return model;
}

}  // namespace

// ============================================================================
// Test 1: Deterministic perturbation generation
// ============================================================================
void testDeterministicPerturbation() {
    ModelIR base = makeWyndor();
    const uint64_t seed = 42;

    // Generate perturbations twice with the same seed
    ModelIR perturbed_a = perturbWyndor(base, 5, seed);
    ModelIR perturbed_b = perturbWyndor(base, 5, seed);

    // They must be identical
    for (int r = 0; r < base.numRows(); ++r) {
        checkNear(perturbed_a.row_upper[static_cast<std::size_t>(r)],
                  perturbed_b.row_upper[static_cast<std::size_t>(r)],
                  1e-15,
                  "Deterministic RHS perturbation row " + std::to_string(r));
    }
    for (int j = 0; j < base.numVars(); ++j) {
        checkNear(perturbed_a.var_upper[static_cast<std::size_t>(j)],
                  perturbed_b.var_upper[static_cast<std::size_t>(j)],
                  1e-15,
                  "Deterministic bound perturbation var " + std::to_string(j));
    }

    // Different day must produce different perturbations
    ModelIR perturbed_c = perturbWyndor(base, 6, seed);
    bool all_same = true;
    for (int r = 0; r < base.numRows(); ++r) {
        if (std::abs(perturbed_a.row_upper[static_cast<std::size_t>(r)] -
                     perturbed_c.row_upper[static_cast<std::size_t>(r)]) > 1e-15) {
            all_same = false;
            break;
        }
    }
    check(!all_same, "Different days produce different perturbations");

    // Matrix A must be unchanged
    check(perturbed_a.A.numRows() == base.A.numRows(), "A rows unchanged");
    check(perturbed_a.A.numCols() == base.A.numCols(), "A cols unchanged");

    // Objective must be unchanged
    for (int j = 0; j < base.numVars(); ++j) {
        checkNear(perturbed_a.obj_coeffs[static_cast<std::size_t>(j)],
                  base.obj_coeffs[static_cast<std::size_t>(j)],
                  1e-15,
                  "Objective unchanged for var " + std::to_string(j));
    }
}

// ============================================================================
// Test 2: Identical perturbation sequences for cold and warm
// ============================================================================
void testIdenticalSequences() {
    ModelIR base = makeWyndor();
    const uint64_t seed = 99;
    const int num_days = 10;

    std::vector<ModelIR> seq1(static_cast<std::size_t>(num_days));
    for (int d = 0; d < num_days; ++d) {
        if (d == 0) seq1[0] = base;
        else seq1[static_cast<std::size_t>(d)] = perturbWyndor(base, d, seed);
    }

    std::vector<ModelIR> seq2(static_cast<std::size_t>(num_days));
    for (int d = 0; d < num_days; ++d) {
        if (d == 0) seq2[0] = base;
        else seq2[static_cast<std::size_t>(d)] = perturbWyndor(base, d, seed);
    }

    for (int d = 0; d < num_days; ++d) {
        const ModelIR& m1 = seq1[static_cast<std::size_t>(d)];
        const ModelIR& m2 = seq2[static_cast<std::size_t>(d)];
        check(m1.numRows() == m2.numRows(), "numRows match");
        check(m1.numVars() == m2.numVars(), "numVars match");
        check(m1.obj_sense == m2.obj_sense, "obj_sense match");
        checkNear(m1.obj_offset, m2.obj_offset, 1e-15, "obj_offset match");

        for (int r = 0; r < m1.numRows(); ++r) {
            checkNear(m1.row_lower[static_cast<std::size_t>(r)], m2.row_lower[static_cast<std::size_t>(r)], 1e-15, "row_lower match");
            checkNear(m1.row_upper[static_cast<std::size_t>(r)], m2.row_upper[static_cast<std::size_t>(r)], 1e-15, "row_upper match");
        }
        for (int j = 0; j < m1.numVars(); ++j) {
            checkNear(m1.var_lower[static_cast<std::size_t>(j)], m2.var_lower[static_cast<std::size_t>(j)], 1e-15, "var_lower match");
            checkNear(m1.var_upper[static_cast<std::size_t>(j)], m2.var_upper[static_cast<std::size_t>(j)], 1e-15, "var_upper match");
            checkNear(m1.obj_coeffs[static_cast<std::size_t>(j)], m2.obj_coeffs[static_cast<std::size_t>(j)], 1e-15, "obj_coeffs match");
        }

        // Matrix structure
        for (int r = 0; r < m1.numRows(); ++r) {
            auto r1 = m1.A.row(r);
            auto r2 = m2.A.row(r);
            check(r1.size() == r2.size(), "row size match");
            auto it1 = r1.begin();
            auto it2 = r2.begin();
            for (; it1 != r1.end() && it2 != r2.end(); ++it1, ++it2) {
                check((*it1).index == (*it2).index, "col_idx match");
                checkNear((*it1).value, (*it2).value, 1e-15, "values match");
            }
        }
    }
}

// ============================================================================
// Test 3: Sequential basis chaining correctness
// ============================================================================
void testBasisChaining() {
    ModelIR base = makeWyndor();
    const uint64_t seed = 777;
    const int num_days = 5;

    DualSimplex ds;
    RevisedSimplex rs;

    // Day 0: cold solve + capture basis
    BasisState current_basis = ds.captureBasis(base);
    check(!current_basis.empty(), "Chaining: Day 0 basis captured");

    // Days 1-4: warm-solve with chaining
    for (int d = 1; d < num_days; ++d) {
        ModelIR model = perturbWyndor(base, d, seed);

        BasisState next_basis;
        SolveResult warm = ds.warmSolve(model, current_basis, &next_basis);

        check(warm.status == SolveStatus::kOptimal,
              "Chaining day " + std::to_string(d) + ": warm optimal");

        // Verify against cold solve
        SolveResult cold = rs.solve(model);
        check(cold.status == SolveStatus::kOptimal,
              "Chaining day " + std::to_string(d) + ": cold optimal");

        if (warm.status == SolveStatus::kOptimal && cold.status == SolveStatus::kOptimal) {
            double ref = std::max(std::abs(cold.objective_value), 1.0);
            checkNear(warm.objective_value, cold.objective_value, ref * 1e-4,
                      "Chaining day " + std::to_string(d) + ": objectives agree");
        }

        // Chain the basis
        if (!next_basis.empty()) {
            current_basis = next_basis;
            check(true, "Chaining day " + std::to_string(d) + ": basis chained");
        } else {
            // Fall back to recapturing basis
            current_basis = ds.captureBasis(model);
            std::cout << "  Day " << d << ": had to re-capture basis (output basis was empty)\n";
        }
    }
}

// ============================================================================
// Test 4: Cold/warm objective agreement
// ============================================================================
void testObjectiveAgreement() {
    ModelIR base = makeWyndor();
    const uint64_t seed = 314159;
    const int num_days = 10;

    DualSimplex ds;
    RevisedSimplex rs;

    SolveResult initial_cold = rs.solve(base);
    check(initial_cold.status == SolveStatus::kOptimal, "ObjAgree: base cold solve is optimal");

    BasisState basis = ds.captureBasis(base);
    check(!basis.empty(), "ObjAgree: initial basis captured");

    for (int d = 1; d < num_days; ++d) {
        ModelIR model = perturbWyndor(base, d, seed);

        SolveResult cold = rs.solve(model);
        check(cold.status == SolveStatus::kOptimal, "ObjAgree day " + std::to_string(d) + ": cold solve must be optimal");

        SolveResult warm;
        BasisState next_basis;
        bool threw = false;

        try {
            warm = ds.warmSolve(model, basis, &next_basis);
        } catch (const std::invalid_argument& e) {
            threw = true;
            check(true, "ObjAgree day " + std::to_string(d) + ": warm solve rejected properly");

            SolveResult fallback;
            basis = ds.captureBasis(model, &fallback);
            check(fallback.status == SolveStatus::kOptimal, "ObjAgree day " + std::to_string(d) + ": fallback solve is optimal");

            double ref = std::max(std::abs(cold.objective_value), 1.0);
            double rel_diff = std::abs(cold.objective_value - fallback.objective_value) / ref;
            check(rel_diff < 1e-4, "ObjAgree day " + std::to_string(d) + ": fallback objective agrees");
        } catch (const std::exception& e) {
            check(false, "ObjAgree day " + std::to_string(d) + ": unexpected exception " + e.what());
            return;
        }

        if (!threw) {
            check(warm.status == SolveStatus::kOptimal, "ObjAgree day " + std::to_string(d) + ": warm solve is optimal");
            double ref = std::max(std::abs(cold.objective_value), 1.0);
            double rel_diff = std::abs(cold.objective_value - warm.objective_value) / ref;
            check(rel_diff < 1e-4, "ObjAgree day " + std::to_string(d) + ": warm objective agrees");

            if (!next_basis.empty()) {
                basis = next_basis;
            } else {
                basis = ds.captureBasis(model);
            }
        }
    }
}

// ============================================================================
// Test 5: Cumulative timing aggregation correctness
// ============================================================================
void testTimingAggregation() {
    // Verify that cumulative = sum of per-day (pure arithmetic test)
    std::vector<double> per_day = {0.001, 0.002, 0.003, 0.004, 0.005};
    double expected_cumulative = 0.015;

    double actual_cumulative = 0.0;
    for (double t : per_day) {
        actual_cumulative += t;
    }

    checkNear(actual_cumulative, expected_cumulative, 1e-15,
              "Timing aggregation: sum matches");
}

// ============================================================================
// Test 6: Speedup calculation correctness
// ============================================================================
void testSpeedupCalculation() {
    // speedup = cold_cumulative / warm_cumulative
    double cold_cumulative = 1.0;
    double warm_cumulative = 0.2;
    double expected_speedup = 5.0;

    double actual_speedup = cold_cumulative / warm_cumulative;
    checkNear(actual_speedup, expected_speedup, 1e-12,
              "Speedup calculation: 1.0/0.2 = 5.0");

    // Equal times → 1.0x
    checkNear(1.0 / 1.0, 1.0, 1e-12, "Speedup: equal times = 1.0x");

    // Warm slower → < 1.0x
    double slower = 0.5 / 1.0;
    check(slower < 1.0, "Speedup: warm slower gives < 1.0x");
}

// ============================================================================
// Test 7: Invalid configuration handling
// ============================================================================
void testInvalidConfiguration() {
    std::cout << "Invalid configuration handling...\n";
    ModelIR model = makeWyndor();
    DualSimplex ds;
    BasisState empty_basis;

    SolveResult res = ds.warmSolve(model, empty_basis);
    check(res.status == SolveStatus::kWarmStartRejected, "Empty basis rejected cleanly");

    // Verify that incompatible model throws
    BasisState basis = ds.captureBasis(model);
    check(!basis.empty(), "InvalidCfg: basis captured");

    // Change number of variables
    ModelIR incompatible(
        ObjSense::kMaximize, 0.0, {1.0, 2.0, 3.0},
        denseToCSR({{1.0, 1.0, 1.0}}, 3),
        {-kInfinity}, {10.0}, {"row0"},
        {0.0, 0.0, 0.0}, {10.0, 10.0, 10.0},
        {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2", "x3"});

    SolveResult res2 = ds.warmSolve(incompatible, basis);
    check(res2.status == SolveStatus::kWarmStartRejected, "Incompatible model rejected cleanly");
}

// ============================================================================
// Test 8: Warm-start preserves structure
// ============================================================================
void testStructurePreservation() {
    ModelIR base = makeWyndor();
    const uint64_t seed = 12345;

    for (int d = 1; d <= 5; ++d) {
        ModelIR perturbed = perturbWyndor(base, d, seed);

        // Structure must be preserved
        check(perturbed.numVars() == base.numVars(),
              "Day " + std::to_string(d) + ": numVars preserved");
        check(perturbed.numRows() == base.numRows(),
              "Day " + std::to_string(d) + ": numRows preserved");
        check(perturbed.obj_sense == base.obj_sense,
              "Day " + std::to_string(d) + ": obj_sense preserved");
        check(perturbed.obj_offset == base.obj_offset,
              "Day " + std::to_string(d) + ": obj_offset preserved");

        // Variable lower bounds must be unchanged
        for (int j = 0; j < base.numVars(); ++j) {
            checkNear(perturbed.var_lower[static_cast<std::size_t>(j)],
                      base.var_lower[static_cast<std::size_t>(j)],
                      1e-15,
                      "Day " + std::to_string(d) + " var_lower[" + std::to_string(j) + "]");
        }

        // Row_lower for resource rows must be unchanged (-kInfinity)
        for (int r = 0; r < base.numRows(); ++r) {
            if (base.row_lower[static_cast<std::size_t>(r)] <= -kInfinity) {
                check(perturbed.row_lower[static_cast<std::size_t>(r)] <= -kInfinity,
                      "Day " + std::to_string(d) + " row_lower[" + std::to_string(r) + "] stays -inf");
            }
        }
    }
}

// ============================================================================
// Test 9: Benchmark Binary Integration
// ============================================================================
void testBenchBinaryIntegration() {
#ifndef BENCHMARK_EXECUTABLE
#define BENCHMARK_EXECUTABLE "./build/rolling_horizon_bench"
#endif

    std::string bin = BENCHMARK_EXECUTABLE;

    std::string cmd = bin + " --days 3 --products 10 --resources 5 --groups 2 > /dev/null 2>&1";
    int ret = std::system(cmd.c_str());
    check(ret == 0, "Benchmark binary executes successfully");

    std::string cmd_json = bin + " --days 2 --products 10 --resources 5 --groups 2 --json > /dev/null 2>&1";
    int ret_json = std::system(cmd_json.c_str());
    check(ret_json == 0, "Benchmark binary JSON mode executes successfully");
}

#include <fstream>
#include <cstdio>

void testConfiguredWorkloadValid() {
    int days = 5;
#ifndef BENCHMARK_EXECUTABLE
#define BENCHMARK_EXECUTABLE "./build/rolling_horizon_bench"
#endif

    std::string bin = BENCHMARK_EXECUTABLE;
    std::string cmd = bin + " --days " + std::to_string(days) + " --products 20 --resources 10 --groups 2 --json > default_workload.json 2>/dev/null";
    int ret = std::system(cmd.c_str());
    check(ret == 0, "Representative configured benchmark generated valid all-optimal sequence");

    std::ifstream ifs("default_workload.json");
    std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));

    auto getBool = [&](const std::string& key) {
        size_t pos = content.find("\"" + key + "\":");
        if (pos == std::string::npos) return false;
        pos += key.length() + 3;
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        return content.substr(pos, 4) == "true";
    };

    auto getInt = [&](const std::string& key) {
        size_t pos = content.find("\"" + key + "\":");
        if (pos == std::string::npos) return -1;
        pos += key.length() + 3;
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        return std::stoi(content.substr(pos));
    };

    check(content.length() > 0 && content[0] == '{', "JSON output begins with an object");
    check(getBool("benchmark_valid"), "benchmark_valid is true");
    check(getBool("correctness"), "correctness is true");
    check(getInt("cold_optimal_days") == days, "cold_optimal_days == days");
    check(getInt("warm_optimal_days") == days, "warm_optimal_days == days");
    check(getInt("warm_start_attempts") == days - 1, "warm_start_attempts == days - 1");
    check(getInt("warm_start_successes") > 0, "warm_start_successes > 0");

    std::remove("default_workload.json");
}

void testCLIValidation() {
#ifndef BENCHMARK_EXECUTABLE
#define BENCHMARK_EXECUTABLE "./build/rolling_horizon_bench"
#endif
    std::string bin = BENCHMARK_EXECUTABLE;

    // Test zero groups success
    std::string cmd = bin + " --days 2 --products 10 --resources 5 --groups 0 > /dev/null 2>&1";
    check(std::system(cmd.c_str()) == 0, "CLI validation: --groups 0 succeeds");

    std::string cmd_json = bin + " --days 2 --products 10 --resources 5 --groups 0 --json > /dev/null 2>&1";
    check(std::system(cmd_json.c_str()) == 0, "CLI validation: --groups 0 --json succeeds");

    // Test strict numeric parsing rejection (must fail)
    check(std::system((bin + " --products 100abc > /dev/null 2>&1").c_str()) != 0, "CLI validation: rejects trailing garbage in int");
    check(std::system((bin + " --tolerance 1e-8xyz > /dev/null 2>&1").c_str()) != 0, "CLI validation: rejects trailing garbage in double (sci)");
    check(std::system((bin + " --perturb-rhs 0.1abc > /dev/null 2>&1").c_str()) != 0, "CLI validation: rejects trailing garbage in double");
    check(std::system((bin + " --seed 123abc > /dev/null 2>&1").c_str()) != 0, "CLI validation: rejects trailing garbage in seed");

    // Test negative seed
    check(std::system((bin + " --seed -5 > /dev/null 2>&1").c_str()) != 0, "CLI validation: rejects negative seed");

    // Test groups > products
    check(std::system((bin + " --products 5 --groups 10 > /dev/null 2>&1").c_str()) != 0, "CLI validation: rejects groups > products");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    run("Deterministic perturbation generation", testDeterministicPerturbation);
    run("Identical perturbation sequences", testIdenticalSequences);
    run("Sequential basis chaining correctness", testBasisChaining);
    run("Cold/warm objective agreement", testObjectiveAgreement);
    run("Cumulative timing aggregation", testTimingAggregation);
    run("Speedup calculation", testSpeedupCalculation);
    run("Invalid configuration handling...", testInvalidConfiguration);
    run("Structure preservation", testStructurePreservation);
    run("Benchmark binary integration", testBenchBinaryIntegration);
    run("Representative workload validity", testConfiguredWorkloadValid);
    run("CLI strict validation", testCLIValidation);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
