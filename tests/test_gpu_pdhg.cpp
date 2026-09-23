#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>

#include "pramaan/ir.hpp"
#include "pramaan/gpu/pdhg_solver.hpp"

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

void testWyndorGlass() {
    ModelIR model(ObjSense::kMaximize, 0.0, {3.0, 5.0}, denseToCSR({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, 2),
                  {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0}, {"r1", "r2", "r3"},
                  {0.0, 0.0}, {kInfinity, kInfinity}, {VarType::kContinuous, VarType::kContinuous}, {"x1", "x2"});
    gpu::PdhgSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Wyndor: optimal");
    if (result.status != SolveStatus::kOptimal) return;
    checkNear(result.objective_value, 36.0, 1e-4, "Wyndor: obj");
}

void testZeroRowLP() {
    ModelIR model(ObjSense::kMinimize, 10.0, {2.0, -1.0}, denseToCSR({}, 2), {}, {}, {}, {0.0, -5.0}, {10.0, 5.0}, {VarType::kContinuous, VarType::kContinuous}, {"x", "y"});
    gpu::PdhgSolver solver;
    const SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "ZeroRowLP: optimal");
    checkNear(result.objective_value, 10.0 + 2.0 * 0.0 + (-1.0) * 5.0, 1e-4, "ZeroRowLP: objective");
}

int main() {
    run("Wyndor Glass Co. (maximize)", testWyndorGlass);
    run("Zero row LP", testZeroRowLP);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
