// test_mps_parser.cpp
//
// Rigorous test suite for the MPS parser (src/io/mps_parser.cpp).
// Tests cover:
//   - tiny handwritten MPS parsing (variable/row counts, obj coeffs,
//     matrix coeffs, RHS values, L/G/E constraint bounds, variable bounds)
//   - ModelIR validity after parsing
//   - malformed-input / error handling
//   - MPS -> ModelIR -> RevisedSimplex end-to-end solve
//   - real Netlib AFIRO instance (parse, validate, solve, objective check)
//
// No external test framework; plain assertions with context on failure.
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pramaan/ir.hpp"
#include "pramaan/mps_parser.hpp"
#include "pramaan/simplex.hpp"

namespace {

using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::RevisedSimplex;
using pramaan::SolveResult;
using pramaan::SolveStatus;
using pramaan::VarType;
using pramaan::parse_mps;

int g_checks_run = 0;
int g_checks_failed = 0;

void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

void checkNear(double actual, double expected, double tol,
               const std::string& description) {
    ++g_checks_run;
    if (std::abs(actual - expected) > tol) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (expected " << expected
                   << ", got " << actual << ", |diff| = "
                   << std::abs(actual - expected) << ")\n";
    }
}

// Write a string to a temporary file and return the path.
// The file is placed next to the test binary in the build dir.
std::string writeTmpMps(const std::string& content, const std::string& name) {
    std::string path = "/tmp/pramaan_test_" + name + ".mps";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "  [FATAL] Cannot create temp file: " << path << "\n";
        std::exit(1);
    }
    out << content;
    out.close();
    return path;
}

// Checks that calling parse_mps throws a std::runtime_error.
void checkThrows(const std::string& content, const std::string& name,
                 const std::string& description) {
    std::string path = writeTmpMps(content, name);
    ++g_checks_run;
    try {
        parse_mps(path);
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << " (no exception thrown)\n";
    } catch (const std::runtime_error&) {
        // expected
    } catch (...) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description
                   << " (unexpected exception type)\n";
    }
}

// Returns a dense value from a CSRMatrix at (r, c).  Returns 0 if the
// entry is not stored (sparse zero).
double getA(const CSRMatrix& A, int r, int c) {
    for (auto e : A.row(static_cast<CSRMatrix::Index>(r))) {
        if (e.index == static_cast<CSRMatrix::Index>(c)) return e.value;
    }
    return 0.0;
}

// -----------------------------------------------------------------------
// Test 1: Tiny hand-written MPS -- the Wyndor Glass LP from the simplex
// tests, expressed in MPS format.
//
//   maximize   3 x1 + 5 x2
//   subject to      x1      <= 4
//                   2 x2     <= 12
//               3 x1 + 2 x2 <= 18
//               x1, x2 >= 0
//
// MPS convention is to minimize the N row, so we negate the objective
// coefficients: c = [-3, -5] with min sense.  Optimal objective (min)
// is -36, optimal point x1=2, x2=6.
// -----------------------------------------------------------------------
void testTinyHandwritten() {
    std::cout << "testTinyHandwritten...\n";

    const std::string mps = R"(NAME          WYNDOR
ROWS
 N  OBJ
 L  PLANT1
 L  PLANT2
 L  PLANT3
COLUMNS
    X1        OBJ           -3.0   PLANT1         1.0
    X1        PLANT3         3.0
    X2        OBJ           -5.0   PLANT2         2.0
    X2        PLANT3         2.0
RHS
    RHS1      PLANT1         4.0   PLANT2        12.0
    RHS1      PLANT3        18.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "wyndor");
    ModelIR model = parse_mps(path);

    // Shape
    check(model.numVars() == 2, "Wyndor: 2 variables");
    check(model.numRows() == 3, "Wyndor: 3 constraint rows");

    // Variable names
    check(model.var_names[0] == "X1", "Wyndor: var_names[0] == X1");
    check(model.var_names[1] == "X2", "Wyndor: var_names[1] == X2");

    // Row names
    check(model.row_names[0] == "PLANT1", "Wyndor: row_names[0] == PLANT1");
    check(model.row_names[1] == "PLANT2", "Wyndor: row_names[1] == PLANT2");
    check(model.row_names[2] == "PLANT3", "Wyndor: row_names[2] == PLANT3");

    // Objective coefficients (min sense, so c = [-3, -5])
    check(model.obj_sense == ObjSense::kMinimize, "Wyndor: minimize");
    checkNear(model.obj_coeffs[0], -3.0, 1e-12, "Wyndor: obj_coeffs[0]");
    checkNear(model.obj_coeffs[1], -5.0, 1e-12, "Wyndor: obj_coeffs[1]");

    // Matrix coefficients
    checkNear(getA(model.A, 0, 0), 1.0, 1e-12, "Wyndor: A[0,0]");
    checkNear(getA(model.A, 0, 1), 0.0, 1e-12, "Wyndor: A[0,1]");
    checkNear(getA(model.A, 1, 0), 0.0, 1e-12, "Wyndor: A[1,0]");
    checkNear(getA(model.A, 1, 1), 2.0, 1e-12, "Wyndor: A[1,1]");
    checkNear(getA(model.A, 2, 0), 3.0, 1e-12, "Wyndor: A[2,0]");
    checkNear(getA(model.A, 2, 1), 2.0, 1e-12, "Wyndor: A[2,1]");

    // RHS and row bounds (L rows: -inf <= Ax <= rhs)
    checkNear(model.row_upper[0], 4.0, 1e-12, "Wyndor: row_upper[0]");
    checkNear(model.row_upper[1], 12.0, 1e-12, "Wyndor: row_upper[1]");
    checkNear(model.row_upper[2], 18.0, 1e-12, "Wyndor: row_upper[2]");
    check(model.row_lower[0] <= -kInfinity, "Wyndor: row_lower[0] == -inf");
    check(model.row_lower[1] <= -kInfinity, "Wyndor: row_lower[1] == -inf");
    check(model.row_lower[2] <= -kInfinity, "Wyndor: row_lower[2] == -inf");

    // Variable bounds (default: 0 <= x <= +inf)
    checkNear(model.var_lower[0], 0.0, 1e-12, "Wyndor: var_lower[0]");
    checkNear(model.var_lower[1], 0.0, 1e-12, "Wyndor: var_lower[1]");
    check(model.var_upper[0] >= kInfinity, "Wyndor: var_upper[0] == +inf");
    check(model.var_upper[1] >= kInfinity, "Wyndor: var_upper[1] == +inf");

    // Variable types
    check(model.var_types[0] == VarType::kContinuous, "Wyndor: var_types[0]");
    check(model.var_types[1] == VarType::kContinuous, "Wyndor: var_types[1]");

    // ModelIR validity (should not throw)
    try {
        model.validate();
        check(true, "Wyndor: validate() passes");
    } catch (const std::exception& e) {
        check(false, std::string("Wyndor: validate() threw: ") + e.what());
    }

    // Solve and verify
    RevisedSimplex solver;
    SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Wyndor: solver optimal");
    if (result.status == SolveStatus::kOptimal) {
        checkNear(result.x[0], 2.0, 1e-6, "Wyndor: x1 = 2");
        checkNear(result.x[1], 6.0, 1e-6, "Wyndor: x2 = 6");
        checkNear(result.objective_value, -36.0, 1e-6, "Wyndor: obj = -36");
    }

    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 2: Mixed constraint senses (L, G, E) in one model.
//
//   minimize  4 x1 + 3 x2
//   subject to  x1 +  x2 >= 10     (G)
//              2 x1 +  x2 <= 24     (L)
//                x1 -  x2  = 2      (E)
//               x1, x2 >= 0
//
// Optimal: x1=6, x2=4, obj=36.
// -----------------------------------------------------------------------
void testMixedSenses() {
    std::cout << "testMixedSenses...\n";

    const std::string mps = R"(NAME          MIXED
ROWS
 N  OBJ
 G  DEMAND
 L  CAPACITY
 E  RATIO
COLUMNS
    X1        OBJ           4.0   DEMAND        1.0
    X1        CAPACITY      2.0   RATIO         1.0
    X2        OBJ           3.0   DEMAND        1.0
    X2        CAPACITY      1.0   RATIO        -1.0
RHS
    RHS1      DEMAND       10.0   CAPACITY     24.0
    RHS1      RATIO         2.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "mixed");
    ModelIR model = parse_mps(path);

    check(model.numVars() == 2, "Mixed: 2 variables");
    check(model.numRows() == 3, "Mixed: 3 rows");

    // G row: row_lower = rhs, row_upper = +inf
    checkNear(model.row_lower[0], 10.0, 1e-12, "Mixed: DEMAND row_lower");
    check(model.row_upper[0] >= kInfinity, "Mixed: DEMAND row_upper = +inf");

    // L row: row_lower = -inf, row_upper = rhs
    check(model.row_lower[1] <= -kInfinity, "Mixed: CAPACITY row_lower = -inf");
    checkNear(model.row_upper[1], 24.0, 1e-12, "Mixed: CAPACITY row_upper");

    // E row: row_lower = row_upper = rhs
    checkNear(model.row_lower[2], 2.0, 1e-12, "Mixed: RATIO row_lower");
    checkNear(model.row_upper[2], 2.0, 1e-12, "Mixed: RATIO row_upper");

    model.validate();

    RevisedSimplex solver;
    SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "Mixed: solver optimal");
    if (result.status == SolveStatus::kOptimal) {
        checkNear(result.x[0], 6.0, 1e-6, "Mixed: x1 = 6");
        checkNear(result.x[1], 4.0, 1e-6, "Mixed: x2 = 4");
        checkNear(result.objective_value, 36.0, 1e-6, "Mixed: obj = 36");
    }

    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 3: BOUNDS section parsing (LO, UP, FX, FR).
// -----------------------------------------------------------------------
void testBoundsSection() {
    std::cout << "testBoundsSection...\n";

    // minimize x1 + x2 + x3 + x4
    // subject to x1 + x2 + x3 + x4 >= 0  (trivial, just to have a row)
    // x1: default [0, +inf)
    // x2: LO -5, UP 10  =>  [-5, 10]
    // x3: FX 7           =>  [7, 7]
    // x4: FR             =>  [-inf, +inf]
    const std::string mps = R"(NAME          BOUNDS_TEST
ROWS
 N  OBJ
 G  ROW1
COLUMNS
    X1        OBJ         1.0   ROW1        1.0
    X2        OBJ         1.0   ROW1        1.0
    X3        OBJ         1.0   ROW1        1.0
    X4        OBJ         1.0   ROW1        1.0
RHS
    RHS1      ROW1        0.0
BOUNDS
 LO BND1      X2          -5.0
 UP BND1      X2          10.0
 FX BND1      X3           7.0
 FR BND1      X4
ENDATA
)";
    std::string path = writeTmpMps(mps, "bounds_test");
    ModelIR model = parse_mps(path);

    check(model.numVars() == 4, "Bounds: 4 variables");

    // X1: default [0, +inf)
    checkNear(model.var_lower[0], 0.0, 1e-12, "Bounds: X1 lower = 0");
    check(model.var_upper[0] >= kInfinity, "Bounds: X1 upper = +inf");

    // X2: [-5, 10]
    checkNear(model.var_lower[1], -5.0, 1e-12, "Bounds: X2 lower = -5");
    checkNear(model.var_upper[1], 10.0, 1e-12, "Bounds: X2 upper = 10");

    // X3: [7, 7]
    checkNear(model.var_lower[2], 7.0, 1e-12, "Bounds: X3 lower = 7");
    checkNear(model.var_upper[2], 7.0, 1e-12, "Bounds: X3 upper = 7");

    // X4: [-inf, +inf]
    check(model.var_lower[3] <= -kInfinity, "Bounds: X4 lower = -inf");
    check(model.var_upper[3] >= kInfinity, "Bounds: X4 upper = +inf");

    model.validate();
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 4: Zero-RHS rows (RHS not mentioned for a constraint defaults to 0).
// -----------------------------------------------------------------------
void testDefaultRhsZero() {
    std::cout << "testDefaultRhsZero...\n";

    const std::string mps = R"(NAME          ZERO_RHS
ROWS
 N  OBJ
 E  ZERO_ROW
COLUMNS
    X1        OBJ         1.0   ZERO_ROW    1.0
RHS
ENDATA
)";
    std::string path = writeTmpMps(mps, "zero_rhs");
    ModelIR model = parse_mps(path);

    check(model.numRows() == 1, "ZeroRHS: 1 row");
    checkNear(model.row_lower[0], 0.0, 1e-12, "ZeroRHS: equality row lower = 0");
    checkNear(model.row_upper[0], 0.0, 1e-12, "ZeroRHS: equality row upper = 0");

    model.validate();
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 5: Coefficient accumulation -- when the same (var, row) appears
// on multiple lines, the values should be added.
// -----------------------------------------------------------------------
void testCoefficientAccumulation() {
    std::cout << "testCoefficientAccumulation...\n";

    const std::string mps = R"(NAME          ACCUM
ROWS
 N  OBJ
 L  ROW1
COLUMNS
    X1        ROW1        2.0
    X1        ROW1        3.0
    X1        OBJ         1.0
    X1        OBJ         0.5
RHS
    RHS1      ROW1       10.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "accum");
    ModelIR model = parse_mps(path);

    // Objective: 1.0 + 0.5 = 1.5
    checkNear(model.obj_coeffs[0], 1.5, 1e-12, "Accum: obj_coeff = 1.5");

    // Matrix: 2.0 + 3.0 = 5.0
    checkNear(getA(model.A, 0, 0), 5.0, 1e-12, "Accum: A[0,0] = 5.0");

    model.validate();
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 6: Comments and blank lines are skipped.
// -----------------------------------------------------------------------
void testCommentsAndBlanks() {
    std::cout << "testCommentsAndBlanks...\n";

    const std::string mps =
        "* This is a comment\n"
        "NAME          COMMENT_TEST\n"
        "ROWS\n"
        "* Another comment\n"
        " N  OBJ\n"
        "\n"
        " L  ROW1\n"
        "COLUMNS\n"
        "    X1        OBJ         1.0   ROW1        2.0\n"
        "RHS\n"
        "    RHS1      ROW1        5.0\n"
        "ENDATA\n";
    std::string path = writeTmpMps(mps, "comment_test");
    ModelIR model = parse_mps(path);

    check(model.numVars() == 1, "Comments: 1 variable");
    check(model.numRows() == 1, "Comments: 1 row");
    checkNear(model.obj_coeffs[0], 1.0, 1e-12, "Comments: obj_coeff");
    checkNear(getA(model.A, 0, 0), 2.0, 1e-12, "Comments: A[0,0]");
    checkNear(model.row_upper[0], 5.0, 1e-12, "Comments: row_upper");

    model.validate();
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 7: Error handling -- malformed inputs.
// -----------------------------------------------------------------------
void testErrorHandling() {
    std::cout << "testErrorHandling...\n";

    // No ENDATA
    checkThrows(
        "NAME T\nROWS\n N OBJ\nCOLUMNS\n    X1 OBJ 1.0\nRHS\n",
        "no_endata", "No ENDATA");

    // Unknown row sense
    checkThrows(
        "NAME T\nROWS\n Z BAD\nCOLUMNS\nRHS\nENDATA\n",
        "bad_sense", "Unknown row sense Z");

    // COLUMNS references unknown row
    checkThrows(
        "NAME T\nROWS\n N OBJ\nCOLUMNS\n    X1 GHOST 1.0\nRHS\nENDATA\n",
        "unknown_row", "Unknown row in COLUMNS");

    // RHS references unknown row
    checkThrows(
        "NAME T\nROWS\n N OBJ\n L R1\nCOLUMNS\n    X1 OBJ 1.0 R1 1.0\n"
        "RHS\n    RHS1 GHOST 5.0\nENDATA\n",
        "unknown_rhs_row", "Unknown row in RHS");

    // Invalid number in COLUMNS
    checkThrows(
        "NAME T\nROWS\n N OBJ\nCOLUMNS\n    X1 OBJ abc\nRHS\nENDATA\n",
        "bad_number", "Invalid number in COLUMNS");

    // Duplicate row name
    checkThrows(
        "NAME T\nROWS\n N OBJ\n L DUP\n L DUP\nCOLUMNS\nRHS\nENDATA\n",
        "dup_row", "Duplicate row name");

    // File does not exist
    ++g_checks_run;
    try {
        parse_mps("/tmp/pramaan_nonexistent_file_12345.mps");
        ++g_checks_failed;
        std::cerr << "  [FAIL] Nonexistent file (no exception)\n";
    } catch (const std::runtime_error&) {
        // expected
    }

    // No variables at all
    checkThrows(
        "NAME T\nROWS\n N OBJ\nCOLUMNS\nRHS\nENDATA\n",
        "no_vars", "No variables");

    // BV bounds not supported
    checkThrows(
        "NAME T\nROWS\n N OBJ\nCOLUMNS\n    X1 OBJ 1.0\nRHS\nBOUNDS\n BV BND X1\nENDATA\n",
        "bv_reject", "BV bounds not supported");

    // RANGES not supported
    checkThrows(
        "NAME T\nROWS\n N OBJ\nCOLUMNS\n    X1 OBJ 1.0\nRHS\nRANGES\n    R1 R1 1.0\nENDATA\n",
        "ranges_reject", "RANGES not supported");

    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test: Objective RHS handling.
// -----------------------------------------------------------------------
void testObjectiveRhs() {
    std::cout << "testObjectiveRhs...\n";
    const std::string mps = R"(NAME          OBJRHS
ROWS
 N  OBJ
 L  R1
COLUMNS
    X1        OBJ         1.0   R1          1.0
RHS
    RHS1      OBJ         5.0   R1         10.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "obj_rhs");
    ModelIR model = parse_mps(path);

    checkNear(model.obj_offset, -5.0, 1e-12, "ObjRHS: obj_offset = -5.0");
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 8: CRLF line endings are handled correctly.
// -----------------------------------------------------------------------
void testCrlfLineEndings() {
    std::cout << "testCrlfLineEndings...\n";

    const std::string mps =
        "NAME          CRLF\r\n"
        "ROWS\r\n"
        " N  OBJ\r\n"
        " L  ROW1\r\n"
        "COLUMNS\r\n"
        "    X1        OBJ         1.0   ROW1        2.0\r\n"
        "RHS\r\n"
        "    RHS1      ROW1        5.0\r\n"
        "ENDATA\r\n";
    std::string path = writeTmpMps(mps, "crlf");
    ModelIR model = parse_mps(path);

    check(model.numVars() == 1, "CRLF: 1 variable");
    check(model.numRows() == 1, "CRLF: 1 row");
    checkNear(model.obj_coeffs[0], 1.0, 1e-12, "CRLF: obj_coeff");

    model.validate();
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 9: MPS -> ModelIR -> RevisedSimplex end-to-end for a small LP.
//
//   min    x1 + 2*x2
//   s.t.   x1 + x2 = 5   (E)
//          x1      >= 1   (G)
//          x1, x2 >= 0
//
// On the equality x1+x2=5, obj = x1 + 2(5-x1) = 10 - x1, which
// decreases with x1.  Maximizing x1 subject to x1<=5 (from x2>=0)
// and x1>=1 gives x1=5, x2=0, obj=5.
// -----------------------------------------------------------------------
void testEndToEndSmallLP() {
    std::cout << "testEndToEndSmallLP...\n";

    const std::string mps = R"(NAME          SMALL_E2E
ROWS
 N  OBJ
 E  SUM
 G  FLOOR
COLUMNS
    X1        OBJ         1.0   SUM         1.0
    X1        FLOOR       1.0
    X2        OBJ         2.0   SUM         1.0
RHS
    RHS1      SUM         5.0   FLOOR       1.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "small_e2e");
    ModelIR model = parse_mps(path);
    model.validate();

    RevisedSimplex solver;
    SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kOptimal, "SmallE2E: optimal");
    if (result.status == SolveStatus::kOptimal) {
        checkNear(result.x[0], 5.0, 1e-6, "SmallE2E: x1 = 5");
        checkNear(result.x[1], 0.0, 1e-6, "SmallE2E: x2 = 0");
        checkNear(result.objective_value, 5.0, 1e-6, "SmallE2E: obj = 5");
    }

    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 10: Infeasible LP expressed in MPS, parsed and solved.
// -----------------------------------------------------------------------
void testMpsInfeasible() {
    std::cout << "testMpsInfeasible...\n";

    // x >= 5 and x <= 2 via separate rows
    const std::string mps = R"(NAME          INFEASIBLE
ROWS
 N  OBJ
 G  LO
 L  HI
COLUMNS
    X1        OBJ         1.0   LO          1.0
    X1        HI          1.0
RHS
    RHS1      LO          5.0   HI          2.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "infeasible");
    ModelIR model = parse_mps(path);
    model.validate();

    RevisedSimplex solver;
    SolveResult result = solver.solve(model);
    check(result.status == SolveStatus::kInfeasible, "MpsInfeasible: infeasible");

    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 11: MI bound type (minus-infinity lower bound).
// -----------------------------------------------------------------------
void testMiBound() {
    std::cout << "testMiBound...\n";

    const std::string mps = R"(NAME          MI_TEST
ROWS
 N  OBJ
 E  ROW1
COLUMNS
    X1        OBJ         1.0   ROW1        1.0
RHS
    RHS1      ROW1        0.0
BOUNDS
 MI BND1      X1
 UP BND1      X1          5.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "mi_test");
    ModelIR model = parse_mps(path);

    check(model.var_lower[0] <= -kInfinity, "MI: lower = -inf");
    checkNear(model.var_upper[0], 5.0, 1e-12, "MI: upper = 5");

    model.validate();
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 12: Variable ordering is by first appearance in COLUMNS.
// -----------------------------------------------------------------------
void testVariableOrdering() {
    std::cout << "testVariableOrdering...\n";

    const std::string mps = R"(NAME          ORDERING
ROWS
 N  OBJ
 L  ROW1
COLUMNS
    BETA      OBJ         2.0   ROW1        1.0
    ALPHA     OBJ         1.0   ROW1        1.0
    GAMMA     OBJ         3.0   ROW1        1.0
RHS
    RHS1      ROW1       10.0
ENDATA
)";
    std::string path = writeTmpMps(mps, "ordering");
    ModelIR model = parse_mps(path);

    check(model.var_names[0] == "BETA", "Ordering: var_names[0] == BETA");
    check(model.var_names[1] == "ALPHA", "Ordering: var_names[1] == ALPHA");
    check(model.var_names[2] == "GAMMA", "Ordering: var_names[2] == GAMMA");
    checkNear(model.obj_coeffs[0], 2.0, 1e-12, "Ordering: BETA obj = 2");
    checkNear(model.obj_coeffs[1], 1.0, 1e-12, "Ordering: ALPHA obj = 1");
    checkNear(model.obj_coeffs[2], 3.0, 1e-12, "Ordering: GAMMA obj = 3");

    model.validate();
    std::cout << "  ok\n";
}

// -----------------------------------------------------------------------
// Test 13: Netlib AFIRO instance.
//
// Expected:
//   - 27 constraint rows (28 total with the N objective row)
//   - 32 structural variables
//   - ModelIR is valid
//   - Solver reaches OPTIMAL
//   - Optimal objective ≈ -464.75314286 (to 2 decimal places: -464.75)
// -----------------------------------------------------------------------
void testAfiro() {
    std::cout << "testAfiro...\n";

    // The test data file is at tests/data/afiro.mps relative to the repo
    // root. The test binary may be run from the build dir or the repo root,
    // so we try several common relative paths.
    std::string afiro_path;
    for (const char* candidate : {
             "tests/data/afiro.mps",
             "../tests/data/afiro.mps",
             "../../tests/data/afiro.mps"}) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            afiro_path = candidate;
            break;
        }
    }
    if (afiro_path.empty()) {
        std::cerr << "  [SKIP] afiro.mps not found in expected locations\n";
        return;
    }

    ModelIR model = parse_mps(afiro_path);

    // Shape checks
    check(model.numRows() == 27, "AFIRO: 27 constraint rows");
    check(model.numVars() == 32, "AFIRO: 32 variables");

    // ModelIR validity
    try {
        model.validate();
        check(true, "AFIRO: validate() passes");
    } catch (const std::exception& e) {
        check(false, std::string("AFIRO: validate() threw: ") + e.what());
    }

    // isPureLP (AFIRO has no integer variables)
    check(model.isPureLP(), "AFIRO: isPureLP() == true");

    // All variables should have default bounds [0, +inf) since AFIRO
    // has no BOUNDS section.
    for (int j = 0; j < model.numVars(); ++j) {
        check(model.var_lower[static_cast<std::size_t>(j)] == 0.0,
              "AFIRO: var_lower[" + std::to_string(j) + "] == 0");
        check(model.var_upper[static_cast<std::size_t>(j)] >= kInfinity,
              "AFIRO: var_upper[" + std::to_string(j) + "] == +inf");
    }

    // Spot-check a few known objective coefficients from the MPS file:
    // X02 has COST = -0.4, X14 has COST = -0.32, X23 has COST = -0.6,
    // X36 has COST = -0.48, X39 has COST = 10.0.
    // Variable order is by first appearance in COLUMNS, so we look up by name.
    auto findVar = [&](const std::string& name) -> int {
        for (int j = 0; j < model.numVars(); ++j) {
            if (model.var_names[static_cast<std::size_t>(j)] == name) return j;
        }
        return -1;
    };

    int idx_X02 = findVar("X02");
    int idx_X14 = findVar("X14");
    int idx_X23 = findVar("X23");
    int idx_X36 = findVar("X36");
    int idx_X39 = findVar("X39");

    check(idx_X02 >= 0, "AFIRO: found X02");
    check(idx_X14 >= 0, "AFIRO: found X14");
    check(idx_X23 >= 0, "AFIRO: found X23");
    check(idx_X36 >= 0, "AFIRO: found X36");
    check(idx_X39 >= 0, "AFIRO: found X39");

    if (idx_X02 >= 0) checkNear(model.obj_coeffs[static_cast<std::size_t>(idx_X02)], -0.4, 1e-12, "AFIRO: X02 obj = -0.4");
    if (idx_X14 >= 0) checkNear(model.obj_coeffs[static_cast<std::size_t>(idx_X14)], -0.32, 1e-12, "AFIRO: X14 obj = -0.32");
    if (idx_X23 >= 0) checkNear(model.obj_coeffs[static_cast<std::size_t>(idx_X23)], -0.6, 1e-12, "AFIRO: X23 obj = -0.6");
    if (idx_X36 >= 0) checkNear(model.obj_coeffs[static_cast<std::size_t>(idx_X36)], -0.48, 1e-12, "AFIRO: X36 obj = -0.48");
    if (idx_X39 >= 0) checkNear(model.obj_coeffs[static_cast<std::size_t>(idx_X39)], 10.0, 1e-12, "AFIRO: X39 obj = 10.0");

    // Spot-check a few known RHS values from the MPS file:
    // X50 rhs=310, X51 rhs=300, X05 rhs=80, X17 rhs=80, X27 rhs=500,
    // R23 rhs=44, X40 rhs=500.
    auto findRow = [&](const std::string& name) -> int {
        for (int r = 0; r < model.numRows(); ++r) {
            if (model.row_names[static_cast<std::size_t>(r)] == name) return r;
        }
        return -1;
    };

    int idx_X50 = findRow("X50");
    int idx_R23 = findRow("R23");
    check(idx_X50 >= 0, "AFIRO: found row X50");
    check(idx_R23 >= 0, "AFIRO: found row R23");
    if (idx_X50 >= 0) checkNear(model.row_upper[static_cast<std::size_t>(idx_X50)], 310.0, 1e-12, "AFIRO: X50 rhs = 310");
    if (idx_R23 >= 0) {
        // R23 is an E row, so row_lower = row_upper = rhs = 44
        checkNear(model.row_lower[static_cast<std::size_t>(idx_R23)], 44.0, 1e-12, "AFIRO: R23 rhs (lower) = 44");
        checkNear(model.row_upper[static_cast<std::size_t>(idx_R23)], 44.0, 1e-12, "AFIRO: R23 rhs (upper) = 44");
    }

    // Solve AFIRO
    RevisedSimplex solver;
    SolveResult result = solver.solve(model);

    check(result.status == SolveStatus::kOptimal,
          "AFIRO: solver reaches OPTIMAL");

    if (result.status == SolveStatus::kOptimal) {
        // Known optimal: -4.6475314286e+02
        checkNear(result.objective_value, -464.75314286, 0.01,
                  "AFIRO: objective ≈ -464.75");

        // Verify primal feasibility independently
        auto activity = model.A.multiply(result.x);
        bool feasible = true;
        for (int r = 0; r < model.numRows(); ++r) {
            double lo = model.row_lower[static_cast<std::size_t>(r)];
            double up = model.row_upper[static_cast<std::size_t>(r)];
            double a = activity[static_cast<std::size_t>(r)];
            if ((lo > -kInfinity && a < lo - 1e-6) ||
                (up < kInfinity && a > up + 1e-6)) {
                feasible = false;
                std::cerr << "  [INFO] row " << r << " (" << model.row_names[static_cast<std::size_t>(r)]
                           << "): activity=" << a << " bounds=[" << lo << ", " << up << "]\n";
            }
        }
        check(feasible, "AFIRO: primal feasible");

        bool var_feasible = true;
        for (int j = 0; j < model.numVars(); ++j) {
            double xj = result.x[static_cast<std::size_t>(j)];
            if (xj < model.var_lower[static_cast<std::size_t>(j)] - 1e-6 ||
                xj > model.var_upper[static_cast<std::size_t>(j)] + 1e-6) {
                var_feasible = false;
            }
        }
        check(var_feasible, "AFIRO: variable bounds satisfied");

        std::cout << "  AFIRO objective = " << result.objective_value
                   << " (iterations: " << result.iterations << ")\n";
    }

    std::cout << "  ok\n";
}

}  // namespace

int main() {
    testTinyHandwritten();
    testMixedSenses();
    testBoundsSection();
    testDefaultRhsZero();
    testCoefficientAccumulation();
    testCommentsAndBlanks();
    testErrorHandling();
    testObjectiveRhs();
    testCrlfLineEndings();
    testEndToEndSmallLP();
    testMpsInfeasible();
    testMiBound();
    testVariableOrdering();
    testAfiro();

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed
               << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
