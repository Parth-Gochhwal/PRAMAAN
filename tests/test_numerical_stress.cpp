#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <random>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

using namespace pramaan;

int g_checks_run = 0;
int g_checks_failed = 0;

void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

void checkNear(double a, double b, double tol, const std::string& description) {
    ++g_checks_run;
    if (std::abs(a - b) > tol) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (" << a << " vs " << b << ")\n";
    }
}

CSRMatrix denseToCSR(const std::vector<std::vector<double>>& dense, CSRMatrix::Index num_cols) {
    std::vector<CSRMatrix::Index> row_ptr;
    std::vector<CSRMatrix::Index> col_idx;
    std::vector<double> values;
    row_ptr.push_back(0);
    for (const auto& row : dense) {
        for (CSRMatrix::Index c = 0; c < num_cols; ++c) {
            if (row[c] != 0.0) {
                col_idx.push_back(c);
                values.push_back(row[c]);
            }
        }
        row_ptr.push_back(col_idx.size());
    }
    return CSRMatrix(std::move(row_ptr), std::move(col_idx), std::move(values), num_cols);
}

void run(const std::string& name, const std::function<void()>& test_body) {
    std::cout << name << "...\n";
    int before = g_checks_failed;
    test_body();
    if (g_checks_failed == before) {
        std::cout << "  ok\n";
    }
}

void testIllConditioned() {
    // 2x2 with bad condition number
    // x1 + x2 <= 1
    // x1 + (1 + 1e-7) x2 <= 1 + 1e-7
    // max x2
    // Optimal: x1 = 0, x2 = 1. Obj = 1.
    ModelIR model(ObjSense::kMaximize, 0.0, {0.0, 1.0}, denseToCSR({{1.0, 1.0}, {1.0, 1.0 + 1e-7}}, 2),
                  {-kInfinity, -kInfinity}, {1.0, 1.0 + 1e-7}, {"r1", "r2"},
                  {0.0, 0.0}, {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "IllConditioned: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 1.0, 1e-5, "IllConditioned: obj");
}

void testDegenerate() {
    // 3x3 highly degenerate
    ModelIR model(ObjSense::kMaximize, 0.0, {1.0, 1.0, 1.0}, denseToCSR({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}}, 3),
                  {-kInfinity, -kInfinity, -kInfinity, -kInfinity}, {0.0, 0.0, 0.0, 0.0}, {"r1", "r2", "r3", "r4"},
                  {0.0, 0.0, 0.0}, {kInfinity, kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous}, {"x1", "x2", "x3"});
    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Degenerate: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 0.0, 1e-8, "Degenerate: obj");
}

void testScalingStress() {
    // Variable bounds very large
    ModelIR model(ObjSense::kMaximize, 0.0, {1e6, 1e-6}, denseToCSR({{1e6, 1e-6}}, 2),
                  {-kInfinity}, {1e12}, {"r1"},
                  {0.0, 0.0}, {1e6, 1e6}, {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    RevisedSimplex solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "ScalingStress: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 1e12, 1e4, "ScalingStress: obj");
}

int main() {
    run("Ill-conditioned basis", testIllConditioned);
    run("Highly degenerate LP", testDegenerate);
    run("Scaling stress", testScalingStress);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
