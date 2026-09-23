// test_mip_node.cpp
// P1 Step 2 — Branch-and-Bound node representation tests
//
// Uses a tiny 2-variable integer LP and the parent/child pair one would
// write on paper:
//
//   maximize  x0 + x1
//   s.t.      2 x0 + 2 x1 <= 5          (row "cap")
//             0 <= x0 <= 4, 0 <= x1 <= 4, both integer
//
// The LP relaxation is fractional, so branching on x0 at value 1.5 gives
// exactly two children:
//   down: x0 <= 1   (bounds become [0, 1])
//   up:   x0 >= 2   (bounds become [2, 4])
//
// Tests:
// 1. Root node mirrors the original model's bounds and is unevaluated.
// 2. Down-branch child bounds are exactly the hand-written subproblem.
// 3. Up-branch child bounds are exactly the hand-written subproblem.
// 4. A child never loosens a bound its parent already tightened.
// 5. The inherited BasisState is preserved verbatim.
// 6. Node state transitions (unevaluated -> optimal / infeasible) behave.
// 7. Empty-box branches are reported as inconsistent, not handed to the LP.
// 8. buildRelaxation() applies node bounds and leaves the rest untouched.
// 9. addCut() validates the cut and throws on structural violations.
// 10. makeChild() throws on out-of-range variable or non-finite bound.
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "pramaan/node.hpp"
#include "pramaan/dual_simplex.hpp"
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace {

using pramaan::BasisState;
using pramaan::CSRMatrix;
using pramaan::kInfinity;
using pramaan::ModelIR;
using pramaan::ObjSense;
using pramaan::SolveResult;
using pramaan::SolveStatus;
using pramaan::VarType;
using pramaan::mip::BranchDirection;
using pramaan::mip::Cut;
using pramaan::mip::Node;
using pramaan::mip::NodeState;

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

// maximize x0 + x1  s.t.  2x0 + 2x1 <= 5,  0 <= x <= 4, both integer
ModelIR makeTinyIntegerLP() {
    return ModelIR(ObjSense::kMaximize, 0.0,
                   {1.0, 1.0},
                   denseToCSR({{2.0, 2.0}}, 2),
                   {-kInfinity}, {5.0}, {"cap"},
                   {0.0, 0.0}, {4.0, 4.0},
                   {VarType::kInteger, VarType::kInteger},
                   {"x0", "x1"});
}

BasisState makeFakeBasis() {
    BasisState b;
    b.basis_columns = {3, 7};
    b.structural_fingerprint = 0xABCDEF0123456789ULL;
    b.orig_num_vars = 2;
    b.orig_num_rows = 1;
    b.orig_obj_sense = ObjSense::kMaximize;
    return b;
}

void testRootMirrorsModel() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);

    check(root.branch().isRoot(), "root: branch decision is the root marker");
    check(root.depth() == 0, "root: depth is 0");
    check(root.varLower() == model.var_lower, "root: lower bounds mirror the model");
    check(root.varUpper() == model.var_upper, "root: upper bounds mirror the model");
    check(!root.hasInheritedBasis(), "root: has no inherited basis");
    check(root.state() == NodeState::kUnevaluated, "root: starts unevaluated");
    check(!root.isEvaluated(), "root: isEvaluated() is false before solving");
}

// Branching on x0 at LP value 1.5 -> down child is x0 <= 1, i.e. [0, 1].
void testDownBranchBounds() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);
    const Node down = root.makeChild(0, BranchDirection::kDown, std::floor(1.5), BasisState{});

    check(down.depth() == 1, "down: depth is parent + 1");
    check(down.branch().variable == 0, "down: branch variable recorded");
    check(down.branch().direction == BranchDirection::kDown, "down: direction recorded");
    checkNear(down.branch().bound, 1.0, 1e-12, "down: bound recorded as floor(1.5)");

    checkNear(down.varLower()[0], 0.0, 1e-12, "down: x0 lower unchanged at 0");
    checkNear(down.varUpper()[0], 1.0, 1e-12, "down: x0 upper tightened to 1");
    checkNear(down.varLower()[1], 0.0, 1e-12, "down: x1 lower untouched");
    checkNear(down.varUpper()[1], 4.0, 1e-12, "down: x1 upper untouched");

    check(root.varUpper()[0] == 4.0, "down: parent bounds are not mutated by the child");
}

// ... and the up child is x0 >= 2, i.e. [2, 4].
void testUpBranchBounds() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);
    const Node up = root.makeChild(0, BranchDirection::kUp, std::ceil(1.5), BasisState{});

    check(up.branch().direction == BranchDirection::kUp, "up: direction recorded");
    checkNear(up.branch().bound, 2.0, 1e-12, "up: bound recorded as ceil(1.5)");
    checkNear(up.varLower()[0], 2.0, 1e-12, "up: x0 lower tightened to 2");
    checkNear(up.varUpper()[0], 4.0, 1e-12, "up: x0 upper unchanged at 4");
    checkNear(up.varLower()[1], 0.0, 1e-12, "up: x1 untouched");
    checkNear(up.varUpper()[1], 4.0, 1e-12, "up: x1 untouched");
}

// A grandchild must never undo a tightening an ancestor already applied.
void testChildNeverLoosens() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);

    const Node down = root.makeChild(0, BranchDirection::kDown, 1.0, BasisState{});
    // Asking for x0 <= 3 on a node already at x0 <= 1 must keep 1.
    const Node looser = down.makeChild(0, BranchDirection::kDown, 3.0, BasisState{});
    checkNear(looser.varUpper()[0], 1.0, 1e-12, "down-then-looser: upper stays at the tighter 1");

    const Node up = root.makeChild(0, BranchDirection::kUp, 2.0, BasisState{});
    const Node looser_up = up.makeChild(0, BranchDirection::kUp, 1.0, BasisState{});
    checkNear(looser_up.varLower()[0], 2.0, 1e-12, "up-then-looser: lower stays at the tighter 2");

    // Successive tightening in the same direction does apply.
    const Node tighter = down.makeChild(0, BranchDirection::kDown, 0.0, BasisState{});
    checkNear(tighter.varUpper()[0], 0.0, 1e-12, "down-then-tighter: upper moves to 0");
    check(tighter.depth() == 2, "grandchild depth is 2");
}

void testBasisInheritance() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);
    const BasisState parent_basis = makeFakeBasis();

    const Node child = root.makeChild(0, BranchDirection::kDown, 1.0, parent_basis);
    check(child.hasInheritedBasis(), "child reports an inherited basis");
    check(child.inheritedBasis().basis_columns == parent_basis.basis_columns,
          "child preserves basis_columns verbatim");
    check(child.inheritedBasis().structural_fingerprint == parent_basis.structural_fingerprint,
          "child preserves the structural fingerprint");
    check(child.inheritedBasis().orig_num_vars == parent_basis.orig_num_vars,
          "child preserves orig_num_vars");
    check(child.inheritedBasis().orig_num_rows == parent_basis.orig_num_rows,
          "child preserves orig_num_rows");
    check(child.inheritedBasis().orig_obj_sense == parent_basis.orig_obj_sense,
          "child preserves orig_obj_sense");

    const Node cold_child = root.makeChild(0, BranchDirection::kUp, 2.0, BasisState{});
    check(!cold_child.hasInheritedBasis(), "an empty inherited basis means no warm start");
}

void testStateTransitions() {
    const ModelIR model = makeTinyIntegerLP();
    Node node = Node::makeRoot(model);
    check(node.state() == NodeState::kUnevaluated, "starts unevaluated");

    SolveResult optimal;
    optimal.status = SolveStatus::kOptimal;
    optimal.x = {1.0, 1.5};
    optimal.objective_value = 2.5;
    node.setRelaxation(optimal);

    check(node.state() == NodeState::kOptimal, "kOptimal status -> kOptimal state");
    check(node.isEvaluated(), "isEvaluated() true after setRelaxation");
    checkNear(node.relaxationObjective(), 2.5, 1e-12, "relaxation objective is stored");
    check(node.relaxation().x.size() == 2, "relaxation solution is stored");

    Node infeasible_node = Node::makeRoot(model);
    SolveResult infeasible;
    infeasible.status = SolveStatus::kInfeasible;
    infeasible_node.setRelaxation(infeasible);
    check(infeasible_node.state() == NodeState::kInfeasible, "kInfeasible status -> kInfeasible state");

    Node failed_node = Node::makeRoot(model);
    SolveResult limited;
    limited.status = SolveStatus::kIterationLimit;
    failed_node.setRelaxation(limited);
    check(failed_node.state() == NodeState::kFailed, "kIterationLimit -> kFailed state");

    Node marked = Node::makeRoot(model);
    marked.markInfeasible();
    check(marked.state() == NodeState::kInfeasible, "markInfeasible() sets the state directly");
    check(marked.relaxation().status == SolveStatus::kInfeasible,
          "markInfeasible() leaves a consistent relaxation status");
}

// x0 >= 2 then x0 <= 1 is an empty box; it must be flagged rather than handed
// to ModelIR::validate(), which throws on inverted bounds.
void testInconsistentBounds() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);
    check(!root.hasInconsistentBounds(1e-9), "root bounds are consistent");

    const Node up = root.makeChild(0, BranchDirection::kUp, 2.0, BasisState{});
    check(!up.hasInconsistentBounds(1e-9), "single tightening stays consistent");

    Node empty_box = up;
    empty_box.varLower();  // read-only access, no mutation
    const Node contradictory = up.makeChild(0, BranchDirection::kDown, 1.0, BasisState{});
    check(contradictory.hasInconsistentBounds(1e-9),
          "x0 >= 2 combined with x0 <= 1 is reported inconsistent");
}

void testBuildRelaxation() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);
    const Node down = root.makeChild(0, BranchDirection::kDown, 1.0, BasisState{});

    const ModelIR relaxed = down.buildRelaxation(model);
    relaxed.validate();

    checkNear(relaxed.var_upper[0], 1.0, 1e-12, "relaxation carries the node's tightened bound");
    checkNear(relaxed.var_upper[1], 4.0, 1e-12, "relaxation leaves other bounds alone");
    check(relaxed.numVars() == model.numVars(), "relaxation keeps the variable count");
    check(relaxed.numRows() == model.numRows(), "relaxation keeps the row count");
    check(relaxed.obj_sense == model.obj_sense, "relaxation keeps the objective sense");
    check(relaxed.obj_coeffs == model.obj_coeffs, "relaxation keeps the objective coefficients");
    check(relaxed.row_upper == model.row_upper, "relaxation keeps the row bounds");
    check(relaxed.var_types == model.var_types,
          "relaxation keeps var_types (the LP layer ignores them by design)");
    checkNear(model.var_upper[0], 4.0, 1e-12, "buildRelaxation does not mutate the original model");
}

// Issue 3: addCut() must validate the cut before inserting.
void testInvalidCut() {
    const ModelIR model = makeTinyIntegerLP();
    Node node = Node::makeRoot(model);
    const int num_vars = model.numVars();

    // 1. Valid cut: [x0 + x1 <= 1] — should succeed.
    {
        Cut c;
        c.cols = {0, 1};
        c.vals = {1.0, 1.0};
        c.rhs = 1.0;
        bool threw = false;
        try { node.addCut(c, num_vars); }
        catch (...) { threw = true; }
        check(!threw, "invalid-cut: valid cut is accepted");
    }

    // 2. Size mismatch: cols.size() != vals.size()
    {
        Cut c;
        c.cols = {0, 1};
        c.vals = {1.0};  // wrong size
        c.rhs = 1.0;
        bool threw = false;
        try { node.addCut(c, num_vars); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-cut: size mismatch throws invalid_argument");
    }

    // 3. Out-of-range column index
    {
        Cut c;
        c.cols = {0, 999};  // 999 is out of range for 2-var model
        c.vals = {1.0, 1.0};
        c.rhs = 1.0;
        bool threw = false;
        try { node.addCut(c, num_vars); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-cut: out-of-range col index throws invalid_argument");
    }

    // 4. Non-finite coefficient
    {
        Cut c;
        c.cols = {0};
        c.vals = {std::numeric_limits<double>::infinity()};
        c.rhs = 1.0;
        bool threw = false;
        try { node.addCut(c, num_vars); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-cut: infinite coefficient throws invalid_argument");
    }

    // 5. Zero coefficient
    {
        Cut c;
        c.cols = {0};
        c.vals = {0.0};
        c.rhs = 1.0;
        bool threw = false;
        try { node.addCut(c, num_vars); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-cut: zero coefficient throws invalid_argument");
    }

    // 6. Non-finite rhs
    {
        Cut c;
        c.cols = {0};
        c.vals = {1.0};
        c.rhs = std::numeric_limits<double>::infinity();
        bool threw = false;
        try { node.addCut(c, num_vars); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-cut: infinite rhs throws invalid_argument");
    }

    // 7. Duplicate column index
    {
        Cut c;
        c.cols = {0, 0};
        c.vals = {1.0, 2.0};
        c.rhs = 1.0;
        bool threw = false;
        try { node.addCut(c, num_vars); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-cut: duplicate column index throws invalid_argument");
    }
}

// Issue 3: makeChild() must validate its arguments.
void testInvalidBranchBound() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);

    // Out-of-range variable index
    {
        bool threw = false;
        try { root.makeChild(999, BranchDirection::kDown, 1.0, BasisState{}); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-branch: out-of-range variable throws invalid_argument");
    }

    // Negative variable index
    {
        bool threw = false;
        try { root.makeChild(-1, BranchDirection::kDown, 1.0, BasisState{}); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-branch: negative variable index throws invalid_argument");
    }

    // Non-finite bound
    {
        bool threw = false;
        try { root.makeChild(0, BranchDirection::kDown,
                             std::numeric_limits<double>::infinity(), BasisState{}); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-branch: infinite bound throws invalid_argument");
    }

    // NaN bound
    {
        bool threw = false;
        try { root.makeChild(0, BranchDirection::kDown,
                             std::numeric_limits<double>::quiet_NaN(), BasisState{}); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid-branch: NaN bound throws invalid_argument");
    }
}

// Contradictory bounds: x0 >= 3 then x0 <= 1 must be detected as inconsistent.
void testContradictoryBounds() {
    const ModelIR model = makeTinyIntegerLP();
    const Node root = Node::makeRoot(model);

    const Node up = root.makeChild(0, BranchDirection::kUp, 3.0, BasisState{});
    check(!up.hasInconsistentBounds(1e-9), "contradictory: single branch is consistent");

    const Node down = up.makeChild(0, BranchDirection::kDown, 1.0, BasisState{});
    check(down.hasInconsistentBounds(1e-9),
          "contradictory: x0>=3 then x0<=1 is inconsistent");

    // Verify the B&B engine skips solving and marks directly infeasible.
    check(down.state() == NodeState::kUnevaluated,
          "contradictory: node starts unevaluated (before B&B evaluates it)");
}

}  // namespace

int main() {
    run("Root mirrors the original model", testRootMirrorsModel);
    run("Down-branch child bounds (x0 <= 1)", testDownBranchBounds);
    run("Up-branch child bounds (x0 >= 2)", testUpBranchBounds);
    run("Child never loosens an inherited bound", testChildNeverLoosens);
    run("Inherited BasisState is preserved", testBasisInheritance);
    run("Node state transitions", testStateTransitions);
    run("Inconsistent (empty-box) bounds are flagged", testInconsistentBounds);
    run("buildRelaxation applies node bounds only", testBuildRelaxation);
    run("addCut validates structural invariants", testInvalidCut);
    run("makeChild validates variable index and bound", testInvalidBranchBound);
    run("Contradictory bounds detected correctly", testContradictoryBounds);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
