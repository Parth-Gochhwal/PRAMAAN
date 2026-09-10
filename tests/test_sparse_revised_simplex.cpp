// test_sparse_revised_simplex.cpp
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/mps_parser.hpp"

namespace {

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

void checkFeasible(const pramaan::ModelIR& model, const pramaan::SolveResult& result, double tol,
                    const std::string& case_name) {
    check(result.x.size() == static_cast<std::size_t>(model.numVars()), case_name + ": x has num_vars entries");
    for (pramaan::CSRMatrix::Index j = 0; j < model.numVars(); ++j) {
        const double lo = model.var_lower[static_cast<std::size_t>(j)];
        const double up = model.var_upper[static_cast<std::size_t>(j)];
        const double xj = result.x[static_cast<std::size_t>(j)];
        check(std::isfinite(xj), case_name + ": x[" + std::to_string(j) + "] is finite");
        check((lo <= -pramaan::kInfinity || xj >= lo - tol) && (up >= pramaan::kInfinity || xj <= up + tol), 
               case_name + ": x[" + std::to_string(j) + "] bounds");
    }
    const std::vector<double> activity = model.A.multiply(result.x);
    for (pramaan::CSRMatrix::Index r = 0; r < model.numRows(); ++r) {
        const double lo = model.row_lower[static_cast<std::size_t>(r)];
        const double up = model.row_upper[static_cast<std::size_t>(r)];
        const double ar = activity[static_cast<std::size_t>(r)];
        check(std::isfinite(ar), case_name + ": row[" + std::to_string(r) + "] activity is finite");
        check((lo <= -pramaan::kInfinity || ar >= lo - tol) && (up >= pramaan::kInfinity || ar <= up + tol), 
               case_name + ": row[" + std::to_string(r) + "] bounds");
    }
}

void testAfiro() {
    std::cout << "testAfiro...\n";
    std::string path;
    for (const char* candidate : {
             "tests/data/afiro.mps",
             "../tests/data/afiro.mps",
             "../../tests/data/afiro.mps"}) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            path = candidate;
            break;
        }
    }
    if (path.empty()) {
        std::cerr << "  [SKIP] afiro.mps not found in expected locations\n";
        return;
    }

    pramaan::ModelIR model = pramaan::parse_mps(path);
    pramaan::RevisedSimplex solver;
    const pramaan::SolveResult result = solver.solve(model);

    check(result.status == pramaan::SolveStatus::kOptimal, "AFIRO: status optimal");
    if (result.status == pramaan::SolveStatus::kOptimal) {
        checkNear(result.objective_value, -464.75314286, 1e-4, "AFIRO: obj near -464.75314286");
        checkFeasible(model, result, 1e-5, "AFIRO");
        std::cout << "  AFIRO objective = " << result.objective_value
                  << " (iterations: " << result.iterations << ")\n";
    }
    std::cout << "  ok\n";
}

void testAdlittle() {
    std::cout << "testAdlittle...\n";
    std::string path;
    for (const char* candidate : {
             "tests/data/adlittle.mps",
             "../tests/data/adlittle.mps",
             "../../tests/data/adlittle.mps"}) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            path = candidate;
            break;
        }
    }
    if (path.empty()) {
        std::cerr << "  [SKIP] adlittle.mps not found in expected locations\n";
        return;
    }

    pramaan::ModelIR model = pramaan::parse_mps(path);
    pramaan::RevisedSimplex solver;
    const pramaan::SolveResult result = solver.solve(model);

    check(result.status == pramaan::SolveStatus::kOptimal, "ADLITTLE: status optimal");
    if (result.status == pramaan::SolveStatus::kOptimal) {
        checkNear(result.objective_value, 225494.96316, 1e-4, "ADLITTLE: obj near 225494.96316");
        checkFeasible(model, result, 1e-5, "ADLITTLE");
        std::cout << "  ADLITTLE objective = " << result.objective_value
                  << " (iterations: " << result.iterations << ")\n";
    }
    std::cout << "  ok\n";
}

}  // namespace

int main() {
    testAfiro();
    testAdlittle();

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed
               << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
