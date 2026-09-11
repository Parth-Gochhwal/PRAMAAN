// test_certificate.cpp
//
// Stage 7 tests for:
//   - Certificate creation from a known tiny LP
//   - Certificate serialization and parsing
//   - Objective recomputation
//   - Primal residual computation
//   - Bound violation detection
//   - Integrality check behavior
//   - Malformed certificate rejection
//   - Wrong dimension rejection
//   - Corrupted x rejection
//   - Corrupted objective rejection
//   - Corrupted residual rejection
//   - AFIRO end-to-end: solve -> certificate -> verify
//   - AFIRO tampering detection
#include <cmath>
#include <cstdlib>
#include <sys/wait.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

#include "pramaan/certificate.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"
#include "pramaan/presolve.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/transformation_ledger.hpp"

namespace {

using pramaan::Certificate;
using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::RevisedSimplex;
using pramaan::ScalingFactors;
using pramaan::SolveResult;
using pramaan::SolveStatus;
using pramaan::TransformationLedger;
using pramaan::VarType;

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

#ifndef VERIFIER_EXECUTABLE
#define VERIFIER_EXECUTABLE "./pramaan-verify"
#endif

#ifndef SOLVER_EXECUTABLE
#define SOLVER_EXECUTABLE "./pramaan-solve"
#endif

int runCommand(const std::string& cmd, std::string& out_stdout) {
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) return -1;
    char buffer[128];
    out_stdout.clear();
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        out_stdout += buffer;
    }
    int status = pclose(pipe);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
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

// Temp file path helper.
const std::string CERT_TMP = "test_cert_tmp.cert";

// =========================================================================
// 1. Certificate creation from a known tiny LP
// =========================================================================
void testCertificateCreation() {
    std::cout << "testCertificateCreation...\n";

    // min x1 + 2*x2
    // s.t. x1 + x2 <= 10
    //      x1 >= 0, x2 >= 0
    // Optimal: x1=0, x2=0, obj=0
    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0, 2.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {-kInfinity}, {10.0},
        {"row1"},
        {0.0, 0.0}, {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    std::vector<double> x = {0.0, 0.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);

    check(cert.status == "OPTIMAL", "Creation: status OPTIMAL");
    check(cert.num_rows == 1, "Creation: num_rows == 1");
    check(cert.num_cols == 2, "Creation: num_cols == 2");
    checkNear(cert.objective_value, 0.0, 1e-15, "Creation: objective == 0");
    checkNear(cert.primal_residual, 0.0, 1e-15, "Creation: primal_residual == 0");
    checkNear(cert.bound_violation, 0.0, 1e-15, "Creation: bound_violation == 0");
    checkNear(cert.integrality_violation, 0.0, 1e-15, "Creation: integrality_violation == 0");
    check(cert.x.size() == 2, "Creation: x.size() == 2");
    check(!cert.dual_residual.has_value(), "Creation: dual_residual is empty");
    check(!cert.complementarity_residual.has_value(), "Creation: complementarity_residual is empty");
    check(!cert.objective_bound_gap.has_value(), "Creation: objective_bound_gap is empty");

    std::cout << "  ok\n";
}

// =========================================================================
// 2-3. Certificate serialization and parsing
// =========================================================================
void testCertificateSerialization() {
    std::cout << "testCertificateSerialization...\n";

    Certificate cert;
    cert.status = "OPTIMAL";
    cert.num_rows = 5;
    cert.num_cols = 3;
    cert.objective_value = -123.456;
    cert.tolerance = 1e-6;
    cert.primal_residual = 1e-10;
    cert.dual_residual = std::nullopt;
    cert.complementarity_residual = std::nullopt;
    cert.bound_violation = 2e-11;
    cert.integrality_violation = 0.0;
    cert.objective_bound_gap = std::nullopt;
    cert.ledger_entries = 2;
    cert.x = {1.5, -2.3, 0.0};

    pramaan::write_certificate(cert, CERT_TMP);
    Certificate parsed = pramaan::read_certificate(CERT_TMP);

    check(parsed.format_id == "PRAMAAN_CERTIFICATE_V1", "Serialize: format_id");
    check(parsed.status == "OPTIMAL", "Serialize: status");
    check(parsed.num_rows == 5, "Serialize: num_rows");
    check(parsed.num_cols == 3, "Serialize: num_cols");
    checkNear(parsed.objective_value, -123.456, 1e-12, "Serialize: objective_value");
    checkNear(parsed.tolerance, 1e-6, 1e-15, "Serialize: tolerance");
    checkNear(parsed.primal_residual, 1e-10, 1e-20, "Serialize: primal_residual");
    check(!parsed.dual_residual.has_value(), "Serialize: dual_residual");
    check(!parsed.complementarity_residual.has_value(), "Serialize: complementarity_residual");
    checkNear(parsed.bound_violation, 2e-11, 1e-20, "Serialize: bound_violation");
    checkNear(parsed.integrality_violation, 0.0, 1e-15, "Serialize: integrality_violation");
    check(!parsed.objective_bound_gap.has_value(), "Serialize: objective_bound_gap");
    check(parsed.ledger_entries == 2, "Serialize: ledger_entries");
    check(parsed.x.size() == 3, "Serialize: x.size()");
    if (parsed.x.size() == 3) {
        checkNear(parsed.x[0], 1.5, 1e-15, "Serialize: x[0]");
        checkNear(parsed.x[1], -2.3, 1e-15, "Serialize: x[1]");
        checkNear(parsed.x[2], 0.0, 1e-15, "Serialize: x[2]");
    }

    std::remove(CERT_TMP.c_str());
    std::cout << "  ok\n";
}

// =========================================================================
// 4. Objective recomputation
// =========================================================================
void testObjectiveRecomputation() {
    std::cout << "testObjectiveRecomputation...\n";

    // obj = 3 + 2*x1 + 5*x2, with x1=4, x2=1 => obj = 3 + 8 + 5 = 16
    ModelIR model(
        ObjSense::kMinimize, 3.0,
        {2.0, 5.0},
        denseToCSR({{1.0, 0.0}}, 2),
        {-kInfinity}, {100.0},
        {"row1"},
        {0.0, 0.0}, {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    std::vector<double> x = {4.0, 1.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.objective_value, 16.0, 1e-15, "ObjRecomp: 3 + 2*4 + 5*1 == 16");

    std::cout << "  ok\n";
}

// =========================================================================
// 5. Primal residual computation
// =========================================================================
void testPrimalResidual() {
    std::cout << "testPrimalResidual...\n";

    // row: x1 + x2 <= 10, x1 + x2 >= 5 (i.e. 5 <= x1+x2 <= 10)
    // With x1=3, x2=1 => activity=4, violation from lower bound: 5-4=1.
    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0, 1.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {5.0}, {10.0},
        {"row1"},
        {0.0, 0.0}, {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    std::vector<double> x = {3.0, 1.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.primal_residual, 1.0, 1e-15, "PrimalRes: violation 5-(3+1) == 1");

    // Feasible point: x1=5, x2=3 => activity=8, in [5,10].
    x = {5.0, 3.0};
    cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.primal_residual, 0.0, 1e-15, "PrimalRes: feasible => 0");

    // Equality row: 1*x1 = 5. x1=5.1 => violation = 0.1.
    ModelIR model_eq(
        ObjSense::kMinimize, 0.0,
        {1.0},
        denseToCSR({{1.0}}, 1),
        {5.0}, {5.0},
        {"row1"},
        {0.0}, {kInfinity},
        {VarType::kContinuous},
        {"x1"});

    x = {5.1};
    cert = pramaan::generate_certificate(model_eq, x, "OPTIMAL", 0);
    checkNear(cert.primal_residual, 0.1, 1e-15, "PrimalRes: equality violation 0.1");

    std::cout << "  ok\n";
}

// =========================================================================
// 6. Bound violation detection
// =========================================================================
void testBoundViolation() {
    std::cout << "testBoundViolation...\n";

    // x1 in [2, 8], x2 in [0, 5].
    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0, 1.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {-kInfinity}, {100.0},
        {"row1"},
        {2.0, 0.0}, {8.0, 5.0},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    // x1=1 violates lower bound by 1.
    std::vector<double> x = {1.0, 3.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.bound_violation, 1.0, 1e-15, "BoundViol: lb violation == 1");

    // x2=6 violates upper bound by 1.
    x = {3.0, 6.0};
    cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.bound_violation, 1.0, 1e-15, "BoundViol: ub violation == 1");

    // Feasible.
    x = {3.0, 3.0};
    cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.bound_violation, 0.0, 1e-15, "BoundViol: feasible => 0");

    std::cout << "  ok\n";
}

// =========================================================================
// 7. Integrality check behavior
// =========================================================================
void testIntegralityCheck() {
    std::cout << "testIntegralityCheck...\n";

    // Pure LP model (all continuous) -- integrality violation should be 0.
    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0},
        denseToCSR({{1.0}}, 1),
        {-kInfinity}, {10.0},
        {"row1"},
        {0.0}, {kInfinity},
        {VarType::kContinuous},
        {"x1"});

    std::vector<double> x = {3.7};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.integrality_violation, 0.0, 1e-15, "Integrality: continuous => 0");

    std::cout << "  ok\n";
}

// =========================================================================
// 8. Malformed certificate rejection
// =========================================================================
void testMalformedCertificate() {
    std::cout << "testMalformedCertificate...\n";

    // Empty file.
    {
        std::ofstream out(CERT_TMP);
        out.close();
        bool threw = false;
        try { pramaan::read_certificate(CERT_TMP); }
        catch (const std::exception&) { threw = true; }
        check(threw, "Malformed: empty file throws");
    }

    // Wrong format identifier.
    {
        std::ofstream out(CERT_TMP);
        out << "WRONG_FORMAT_V1\n";
        out.close();
        bool threw = false;
        try { pramaan::read_certificate(CERT_TMP); }
        catch (const std::exception&) { threw = true; }
        check(threw, "Malformed: wrong format throws");
    }

    // Truncated file (no END_CERTIFICATE).
    {
        std::ofstream out(CERT_TMP);
        out << "PRAMAAN_CERTIFICATE_V1\n";
        out << "STATUS OPTIMAL\n";
        out.close();
        bool threw = false;
        try { pramaan::read_certificate(CERT_TMP); }
        catch (const std::exception&) { threw = true; }
        check(threw, "Malformed: truncated file throws");
    }

    std::remove(CERT_TMP.c_str());
    std::cout << "  ok\n";
}

// =========================================================================
// 9. Wrong dimension rejection (via verifier logic)
// =========================================================================
void testWrongDimension() {
    std::cout << "testWrongDimension...\n";

    // Create a valid certificate, then change COLS.
    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0, 2.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {-kInfinity}, {10.0},
        {"row1"},
        {0.0, 0.0}, {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    std::vector<double> x = {0.0, 0.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);

    // Corrupt the dimension.
    cert.num_cols = 999;
    pramaan::write_certificate(cert, CERT_TMP);

    // Read back and verify the dimension mismatch is rejected by the parser.
    bool threw = false;
    try {
        pramaan::read_certificate(CERT_TMP);
    } catch (const std::exception& e) {
        threw = true;
    }
    check(threw, "WrongDim: COLS mismatch throws exception");

    std::remove(CERT_TMP.c_str());
    std::cout << "  ok\n";
}

// =========================================================================
// 10. Corrupted x rejection
// =========================================================================
void testCorruptedX() {
    std::cout << "testCorruptedX...\n";

    // Solve a tiny LP, generate certificate, corrupt x[0], verify recomputed
    // objective differs.
    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0, 2.0},
        denseToCSR({{1.0, 1.0}}, 2),
        {-kInfinity}, {10.0},
        {"row1"},
        {0.0, 0.0}, {kInfinity, kInfinity},
        {VarType::kContinuous, VarType::kContinuous},
        {"x1", "x2"});

    std::vector<double> x = {0.0, 0.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);

    // Corrupt x[0].
    cert.x[0] = 100.0;
    // Now recomputed objective would be 100.0, but cert.objective_value is 0.0.
    double recomputed_obj = model.obj_offset;
    for (int j = 0; j < model.numVars(); ++j) {
        recomputed_obj += model.obj_coeffs[static_cast<std::size_t>(j)]
                        * cert.x[static_cast<std::size_t>(j)];
    }
    double diff = std::abs(recomputed_obj - cert.objective_value);
    check(diff > cert.tolerance, "CorruptX: objective mismatch detects x corruption");

    std::cout << "  ok\n";
}

// =========================================================================
// 11. Corrupted objective rejection
// =========================================================================
void testCorruptedObjective() {
    std::cout << "testCorruptedObjective...\n";

    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0},
        denseToCSR({{1.0}}, 1),
        {-kInfinity}, {10.0},
        {"row1"},
        {0.0}, {kInfinity},
        {VarType::kContinuous},
        {"x1"});

    std::vector<double> x = {5.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.objective_value, 5.0, 1e-15, "CorruptObj: original obj == 5");

    // Corrupt objective.
    cert.objective_value = 999.0;
    pramaan::write_certificate(cert, CERT_TMP);
    Certificate parsed = pramaan::read_certificate(CERT_TMP);

    // Recompute objective from model and x.
    double recomputed = model.obj_offset + model.obj_coeffs[0] * parsed.x[0];
    check(std::abs(recomputed - parsed.objective_value) > parsed.tolerance,
          "CorruptObj: tampered objective detectable");

    std::remove(CERT_TMP.c_str());
    std::cout << "  ok\n";
}

// =========================================================================
// 12. Corrupted residual rejection
// =========================================================================
void testCorruptedResidual() {
    std::cout << "testCorruptedResidual...\n";

    ModelIR model(
        ObjSense::kMinimize, 0.0,
        {1.0},
        denseToCSR({{1.0}}, 1),
        {-kInfinity}, {10.0},
        {"row1"},
        {0.0}, {kInfinity},
        {VarType::kContinuous},
        {"x1"});

    std::vector<double> x = {5.0};
    Certificate cert = pramaan::generate_certificate(model, x, "OPTIMAL", 0);
    checkNear(cert.primal_residual, 0.0, 1e-15, "CorruptRes: original residual == 0");

    // Corrupt primal_residual to claim a large violation.
    cert.primal_residual = 100.0;
    pramaan::write_certificate(cert, CERT_TMP);
    Certificate parsed = pramaan::read_certificate(CERT_TMP);

    // Recompute residual from model and x.
    std::vector<double> activity = model.A.multiply(parsed.x);
    double recomputed = 0.0;
    for (int r = 0; r < model.numRows(); ++r) {
        double ar = activity[static_cast<std::size_t>(r)];
        double hi = model.row_upper[static_cast<std::size_t>(r)];
        if (hi < kInfinity) {
            double viol = ar - hi;
            if (viol > recomputed) recomputed = viol;
        }
    }
    check(std::abs(recomputed - parsed.primal_residual) > parsed.tolerance,
          "CorruptRes: tampered residual detectable");

    std::remove(CERT_TMP.c_str());
    std::cout << "  ok\n";
}

// =========================================================================
// 13. AFIRO end-to-end: solve -> certificate -> verify
// =========================================================================
void testAfiroEndToEnd() {
    std::cout << "testAfiroEndToEnd...\n";

    std::string afiro_path;
    for (const char* candidate : {
             "tests/data/afiro.mps",
             "../tests/data/afiro.mps"}) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            afiro_path = candidate;
            break;
        }
    }
    if (afiro_path.empty()) {
        std::cerr << "  [SKIP] afiro.mps not found\n";
        return;
    }

    // Parse original model.
    ModelIR original = pramaan::parse_mps(afiro_path);

    // Pipeline: presolve -> scale -> solve -> unscale -> postsolve.
    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(original, ledger);
    ModelIR pre_scale = reduced;
    ScalingFactors factors = pramaan::ruiz_scale(reduced, ledger, 10);

    RevisedSimplex solver;
    SolveResult result = solver.solve(reduced);
    check(result.status == SolveStatus::kOptimal, "AFIRO E2E: status OPTIMAL");

    if (result.status == SolveStatus::kOptimal) {
        pramaan::unscale_solution(result.x, result.objective_value, pre_scale, factors);
        std::vector<double> full_x = pramaan::postsolve(result.x, ledger);

        // Generate certificate.
        int ledger_count = static_cast<int>(ledger.size());
        Certificate cert = pramaan::generate_certificate(
            original, full_x, "OPTIMAL", ledger_count);

        checkNear(cert.objective_value, -464.7531, 0.01, "AFIRO E2E: objective ≈ -464.75");
        check(cert.primal_residual < cert.tolerance, "AFIRO E2E: primal feasible");
        check(cert.bound_violation < cert.tolerance, "AFIRO E2E: bounds feasible");
        checkNear(cert.integrality_violation, 0.0, 1e-15, "AFIRO E2E: no integrality violation");

        // Serialize and parse back.
        pramaan::write_certificate(cert, CERT_TMP);
        Certificate parsed = pramaan::read_certificate(CERT_TMP);

        check(parsed.status == "OPTIMAL", "AFIRO E2E: parsed status OPTIMAL");
        check(parsed.num_rows == original.numRows(), "AFIRO E2E: parsed rows match");
        check(parsed.num_cols == original.numVars(), "AFIRO E2E: parsed cols match");
        checkNear(parsed.objective_value, cert.objective_value, 1e-12,
                  "AFIRO E2E: parsed objective matches");
        check(parsed.x.size() == full_x.size(), "AFIRO E2E: parsed x size matches");

        // Independently verify: recompute objective from model and parsed x.
        double recomputed_obj = original.obj_offset;
        for (int j = 0; j < original.numVars(); ++j) {
            recomputed_obj += original.obj_coeffs[static_cast<std::size_t>(j)]
                            * parsed.x[static_cast<std::size_t>(j)];
        }
        checkNear(recomputed_obj, parsed.objective_value, parsed.tolerance,
                  "AFIRO E2E: recomputed objective matches certificate");

        // Independently verify: recompute primal residual.
        std::vector<double> activity = original.A.multiply(parsed.x);
        double max_viol = 0.0;
        for (int r = 0; r < original.numRows(); ++r) {
            double ar = activity[static_cast<std::size_t>(r)];
            double lo = original.row_lower[static_cast<std::size_t>(r)];
            double hi = original.row_upper[static_cast<std::size_t>(r)];
            if (lo > -kInfinity) {
                double v = lo - ar;
                if (v > max_viol) max_viol = v;
            }
            if (hi < kInfinity) {
                double v = ar - hi;
                if (v > max_viol) max_viol = v;
            }
        }
        check(max_viol < parsed.tolerance, "AFIRO E2E: independently verified primal feasible");

        std::remove(CERT_TMP.c_str());
    }

    std::cout << "  ok\n";
}

// =========================================================================
// 14. AFIRO tampering detection
// =========================================================================
void testAfiroTampering() {
    std::cout << "testAfiroTampering...\n";

    std::string afiro_path;
    for (const char* candidate : {
             "tests/data/afiro.mps",
             "../tests/data/afiro.mps"}) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            afiro_path = candidate;
            break;
        }
    }
    if (afiro_path.empty()) {
        std::cerr << "  [SKIP] afiro.mps not found\n";
        return;
    }

    ModelIR original = pramaan::parse_mps(afiro_path);

    // Solve.
    TransformationLedger ledger;
    ModelIR reduced = pramaan::presolve_fixed_variables(original, ledger);
    ModelIR pre_scale = reduced;
    ScalingFactors factors = pramaan::ruiz_scale(reduced, ledger, 10);
    RevisedSimplex solver;
    SolveResult result = solver.solve(reduced);
    check(result.status == SolveStatus::kOptimal, "Tamper: status OPTIMAL");
    if (result.status != SolveStatus::kOptimal) return;

    pramaan::unscale_solution(result.x, result.objective_value, pre_scale, factors);
    std::vector<double> full_x = pramaan::postsolve(result.x, ledger);

    Certificate cert = pramaan::generate_certificate(
        original, full_x, "OPTIMAL", static_cast<int>(ledger.size()));

    // --- Tamper A: change one x value ---
    {
        Certificate tampered = cert;
        tampered.x[0] += 50.0;  // significant change
        pramaan::write_certificate(tampered, CERT_TMP);
        Certificate parsed = pramaan::read_certificate(CERT_TMP);

        // Recompute objective.
        double recomputed_obj = original.obj_offset;
        for (int j = 0; j < original.numVars(); ++j) {
            recomputed_obj += original.obj_coeffs[static_cast<std::size_t>(j)]
                            * parsed.x[static_cast<std::size_t>(j)];
        }
        bool obj_mismatch = std::abs(recomputed_obj - parsed.objective_value) > parsed.tolerance;

        // Recompute primal residual.
        std::vector<double> act = original.A.multiply(parsed.x);
        double max_viol = 0.0;
        for (int r = 0; r < original.numRows(); ++r) {
            double ar = act[static_cast<std::size_t>(r)];
            double lo = original.row_lower[static_cast<std::size_t>(r)];
            double hi = original.row_upper[static_cast<std::size_t>(r)];
            if (lo > -kInfinity) { double v = lo - ar; if (v > max_viol) max_viol = v; }
            if (hi < kInfinity) { double v = ar - hi; if (v > max_viol) max_viol = v; }
        }
        bool primal_infeasible = max_viol > parsed.tolerance;

        // Corruption must be detected by at least one check.
        check(obj_mismatch || primal_infeasible,
              "Tamper A: corrupted x detected via objective or primal residual");
    }

    // --- Tamper B: change reported objective ---
    {
        Certificate tampered = cert;
        tampered.objective_value += 100.0;
        pramaan::write_certificate(tampered, CERT_TMP);
        Certificate parsed = pramaan::read_certificate(CERT_TMP);

        double recomputed_obj = original.obj_offset;
        for (int j = 0; j < original.numVars(); ++j) {
            recomputed_obj += original.obj_coeffs[static_cast<std::size_t>(j)]
                            * parsed.x[static_cast<std::size_t>(j)];
        }
        bool obj_mismatch = std::abs(recomputed_obj - parsed.objective_value) > parsed.tolerance;
        check(obj_mismatch, "Tamper B: corrupted objective detected");
    }

    // --- Tamper C: change reported primal residual ---
    {
        Certificate tampered = cert;
        tampered.primal_residual = 50.0;  // claim large violation
        pramaan::write_certificate(tampered, CERT_TMP);
        Certificate parsed = pramaan::read_certificate(CERT_TMP);

        // Recompute residual.
        std::vector<double> activity = original.A.multiply(parsed.x);
        double recomputed_res = 0.0;
        for (int r = 0; r < original.numRows(); ++r) {
            double ar = activity[static_cast<std::size_t>(r)];
            double lo = original.row_lower[static_cast<std::size_t>(r)];
            double hi = original.row_upper[static_cast<std::size_t>(r)];
            if (lo > -kInfinity) {
                double v = lo - ar;
                if (v > recomputed_res) recomputed_res = v;
            }
            if (hi < kInfinity) {
                double v = ar - hi;
                if (v > recomputed_res) recomputed_res = v;
            }
        }
        bool res_mismatch = std::abs(recomputed_res - parsed.primal_residual) > parsed.tolerance;
        check(res_mismatch, "Tamper C: corrupted residual detected");
    }

    // --- Tamper D: change dimension ---
    {
        Certificate tampered = cert;
        tampered.num_cols = 999;
        pramaan::write_certificate(tampered, CERT_TMP);
        bool threw = false;
        try {
            pramaan::read_certificate(CERT_TMP);
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "Tamper D: corrupted dimension throws exception");
    }

    std::remove(CERT_TMP.c_str());
    std::cout << "  ok\n";
}

// =========================================================================
// 15. Actual Verifier Binary End-to-End
// =========================================================================
void testActualVerifierBinary() {
    std::cout << "testActualVerifierBinary...\n";
    
    std::string afiro_path;
    for (const char* candidate : {
             "tests/data/afiro.mps",
             "../tests/data/afiro.mps"}) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            afiro_path = candidate;
            break;
        }
    }
    if (afiro_path.empty()) {
        std::cerr << "  [SKIP] afiro.mps not found\n";
        return;
    }

    std::string cert_path = "test_actual_afiro.cert";
    std::string out;
    int rc;

    // A. Valid AFIRO certificate
    rc = runCommand(std::string(SOLVER_EXECUTABLE) + " " + afiro_path + " " + cert_path, out);
    check(rc == 0, "Solver executed successfully");
    
    rc = runCommand(std::string(VERIFIER_EXECUTABLE) + " " + afiro_path + " " + cert_path, out);
    check(rc == 0, "Verifier returns 0 for valid certificate");
    check(out.find("CERTIFICATE VALID") != std::string::npos, "Verifier prints CERTIFICATE VALID");

    auto corruptFile = [&](const std::string& target, const std::string& replacement) {
        std::ifstream in(cert_path);
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        size_t pos = content.find(target);
        if (pos != std::string::npos) {
            content.replace(pos, target.length(), replacement);
            std::ofstream out_f(cert_path);
            out_f << content;
        }
    };

    auto runAndCheckInvalid = [&](const std::string& desc) {
        rc = runCommand(std::string(VERIFIER_EXECUTABLE) + " " + afiro_path + " " + cert_path, out);
        check(rc != 0, desc + ": Verifier returns nonzero exit code");
        check(out.find("CERTIFICATE INVALID") != std::string::npos, desc + ": Verifier prints CERTIFICATE INVALID");
    };

    auto resetAndLoad = [&]() {
        std::remove(cert_path.c_str());
        rc = runCommand(std::string(SOLVER_EXECUTABLE) + " " + afiro_path + " " + cert_path, out);
        check(rc == 0, "Solver reset successful");
        return pramaan::read_certificate(cert_path);
    };

    // B. Corrupted x
    Certificate cert = resetAndLoad();
    cert.x[0] += 100.0;
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Corrupted x");

    // C. Corrupted objective
    cert = resetAndLoad();
    cert.objective_value += 50.0;
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Corrupted objective");

    // D. Corrupted primal residual
    cert = resetAndLoad();
    cert.primal_residual += 50.0;
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Corrupted primal residual");
    
    // D2. Corrupted integrality residual
    cert = resetAndLoad();
    cert.integrality_violation += 50.0;
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Corrupted integrality residual");

    // E. Corrupted dimensions
    cert = resetAndLoad();
    cert.num_rows = 999;
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Corrupted dimensions");

    // F. Malformed/truncated certificate
    std::ofstream trunc(cert_path);
    trunc << "PRAMAAN_CERTIFICATE_V1\nSTATUS OPTIMAL\n";
    trunc.close();
    runAndCheckInvalid("Malformed/truncated certificate");
    
    // G. Invalid tolerance security check
    cert = resetAndLoad();
    cert.tolerance = 1.0; // Above max trusted 1e-4
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Tolerance too large");
    
    // H. Unsupported fields check
    cert = resetAndLoad();
    cert.dual_residual = 0.0; // NOT UNAVAILABLE
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Unsupported dual_residual");
    
    cert = resetAndLoad();
    cert.complementarity_residual = 0.0;
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Unsupported complementarity_residual");
    
    cert = resetAndLoad();
    cert.objective_bound_gap = 0.0;
    pramaan::write_certificate(cert, cert_path);
    runAndCheckInvalid("Unsupported objective_bound_gap");

    std::remove(cert_path.c_str());
    std::cout << "  ok\n";
}

// =========================================================================
// 16. Parser Hardening
// =========================================================================
void testParserHardening() {
    std::cout << "testParserHardening...\n";

    auto expectThrow = [](const Certificate& c, const std::string& desc) {
        pramaan::write_certificate(c, CERT_TMP);
        bool threw = false;
        try { pramaan::read_certificate(CERT_TMP); }
        catch (const std::exception&) { threw = true; }
        check(threw, desc);
    };

    Certificate base;
    base.status = "OPTIMAL";
    base.num_rows = 1;
    base.num_cols = 1;
    base.ledger_entries = 0;
    base.x = {0.0};

    // Negative ROWS
    Certificate c1 = base; c1.num_rows = -1;
    expectThrow(c1, "Parser: rejects negative ROWS");

    // Negative COLS
    Certificate c2 = base; c2.num_cols = -1;
    expectThrow(c2, "Parser: rejects negative COLS");

    // Negative LEDGER_ENTRIES
    Certificate c3 = base; c3.ledger_entries = -1;
    expectThrow(c3, "Parser: rejects negative LEDGER_ENTRIES");

    // Zero tolerance
    Certificate c4 = base; c4.tolerance = 0.0;
    expectThrow(c4, "Parser: rejects zero tolerance");

    // Negative tolerance
    Certificate c5 = base; c5.tolerance = -1e-6;
    expectThrow(c5, "Parser: rejects negative tolerance");

    // NaN / Inf
    Certificate c6 = base; c6.primal_residual = std::numeric_limits<double>::quiet_NaN();
    expectThrow(c6, "Parser: rejects NaN");
    Certificate c7 = base; c7.primal_residual = std::numeric_limits<double>::infinity();
    expectThrow(c7, "Parser: rejects Inf");

    auto testMalformedLine = [&](const std::string& key, const std::string& bad_line, const std::string& desc) {
        pramaan::write_certificate(base, CERT_TMP);
        std::ifstream in(CERT_TMP);
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        size_t pos = content.find(key);
        if (pos != std::string::npos) {
            size_t end_pos = content.find('\n', pos);
            if (end_pos != std::string::npos) {
                content.replace(pos, end_pos - pos, bad_line);
                std::ofstream out_f(CERT_TMP); out_f << content;
                bool threw = false;
                try { pramaan::read_certificate(CERT_TMP); } catch (const std::exception&) { threw = true; }
                check(threw, desc);
            }
        }
    };

    // Trailing numeric garbage
    testMalformedLine("OBJECTIVE ", "OBJECTIVE 0.0abc", "Parser: rejects trailing numeric garbage");
    
    // Non-integer fields
    testMalformedLine("ROWS ", "ROWS 1.5", "Parser: rejects non-integer ROWS");
    testMalformedLine("COLS ", "COLS 1.5", "Parser: rejects non-integer COLS");
    testMalformedLine("LEDGER_ENTRIES ", "LEDGER_ENTRIES 1.5", "Parser: rejects non-integer LEDGER_ENTRIES");
    
    // Non-numeric fields
    testMalformedLine("OBJECTIVE ", "OBJECTIVE abc", "Parser: rejects non-numeric OBJECTIVE");

    // X vector too short
    Certificate short_x = base;
    short_x.num_cols = 2; // expects 2, but x has 1
    expectThrow(short_x, "Parser: rejects X vector too short for COLS");

    // X vector too long
    Certificate long_x = base;
    long_x.num_cols = 1;
    long_x.x = {0.0, 1.0}; // expects 1, but x has 2
    expectThrow(long_x, "Parser: rejects X vector too long for COLS");

    // Unexpected data after END_CERTIFICATE
    pramaan::write_certificate(base, CERT_TMP);
    std::ofstream out_app(CERT_TMP, std::ios_base::app);
    out_app << "SOME_EXTRA_DATA\n";
    out_app.close();
    bool threw_extra = false;
    try { pramaan::read_certificate(CERT_TMP); } catch (const std::exception&) { threw_extra = true; }
    check(threw_extra, "Parser: rejects unexpected data after END_CERTIFICATE");

    std::remove(CERT_TMP.c_str());
    std::cout << "  ok\n";
}

}  // namespace

int main() {
    testCertificateCreation();
    testCertificateSerialization();
    testObjectiveRecomputation();
    testPrimalResidual();
    testBoundViolation();
    testIntegralityCheck();
    testMalformedCertificate();
    testWrongDimension();
    testCorruptedX();
    testCorruptedObjective();
    testCorruptedResidual();
    testAfiroEndToEnd();
    testAfiroTampering();
    testActualVerifierBinary();
    testParserHardening();

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed
               << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
