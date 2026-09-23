#include <iostream>
#include <vector>
#include <string>
#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"
#include "pramaan/branch_and_bound.hpp"

using namespace pramaan;

// Generates a tiny synthetic MRPL scheduling model.
ModelIR generateMrplModel() {
    // 2 crude types (A, B) over 3 days.
    // We want to decide how much to buy of each.
    // Integer variables: buy A? buy B? (binary)
    // Continuous variables: amount of A, amount of B.

    // Variables:
    // 0: buy_A (int, 0-1)
    // 1: buy_B (int, 0-1)
    // 2: amt_A (cont, 0-100)
    // 3: amt_B (cont, 0-100)

    // Objective: Minimize cost: -10 * amt_A - 15 * amt_B + 50 * buy_A + 60 * buy_B
    std::vector<double> obj = {50, 60, -10, -15};

    // Constraints:
    // 1) amt_A <= 100 * buy_A => amt_A - 100 * buy_A <= 0
    // 2) amt_B <= 100 * buy_B => amt_B - 100 * buy_B <= 0
    // 3) amt_A + amt_B <= 150

    auto denseToCSR = [](const std::vector<std::vector<double>>& dense, CSRMatrix::Index num_cols) {
        std::vector<CSRMatrix::Index> rp;
        std::vector<CSRMatrix::Index> ci;
        std::vector<double> vals;
        rp.push_back(0);
        for (const auto& r : dense) {
            for (CSRMatrix::Index c = 0; c < num_cols; ++c) {
                if (r[c] != 0.0) {
                    ci.push_back(c);
                    vals.push_back(r[c]);
                }
            }
            rp.push_back(ci.size());
        }
        return CSRMatrix(rp, ci, vals, num_cols);
    };

    std::vector<std::vector<double>> dense_A = {
        {-100.0,    0.0, 1.0, 0.0},
        {     0.0, -100.0, 0.0, 1.0},
        {     0.0,    0.0, 1.0, 1.0}
    };

    CSRMatrix A = denseToCSR(dense_A, 4);

    std::vector<double> row_lower = {-kInfinity, -kInfinity, -kInfinity};
    std::vector<double> row_upper = {0.0, 0.0, 150.0};
    std::vector<std::string> row_names = {"link_A", "link_B", "cap"};

    std::vector<double> var_lower = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> var_upper = {1.0, 1.0, 100.0, 100.0};
    std::vector<VarType> var_types = {VarType::kInteger, VarType::kInteger, VarType::kContinuous, VarType::kContinuous};
    std::vector<std::string> var_names = {"buy_A", "buy_B", "amt_A", "amt_B"};

    return ModelIR(ObjSense::kMinimize, 0.0, obj, A, row_lower, row_upper, row_names, var_lower, var_upper, var_types, var_names);
}

int main() {
    std::cout << "PRAMAAN Flagship MRPL Demo\n";
    std::cout << "Generating synthetic scheduling model...\n";
    ModelIR model = generateMrplModel();

    std::cout << "Solving root LP relaxation directly for debug...\n";
    RevisedSimplex primal;
    SolveResult lp_res = primal.solve(model);
    std::cout << "LP Status: " << static_cast<int>(lp_res.status) << "\n";
    if (lp_res.status == SolveStatus::kOptimal) {
        std::cout << "LP Obj: " << lp_res.objective_value << "\n";
    }

    std::cout << "Solving MILP...\n";
    mip::BranchAndBound::Options opts;
    opts.num_threads = 4;
    opts.node_selection = mip::BranchAndBound::Options::NodeSelection::kBestBound;
    mip::BranchAndBound solver(opts);

    mip::MipResult result = solver.solve(model);

    std::cout << "Status: " << static_cast<int>(result.status) << "\n";
    if (result.hasSolution()) {
        std::cout << "Objective: " << result.objective_value << "\n";
        std::cout << "buy_A: " << result.x[0] << "\n";
        std::cout << "buy_B: " << result.x[1] << "\n";
        std::cout << "amt_A: " << result.x[2] << "\n";
        std::cout << "amt_B: " << result.x[3] << "\n";
    }

    std::cout << "Nodes explored: " << result.statistics.nodes_explored << "\n";

    return 0;
}
