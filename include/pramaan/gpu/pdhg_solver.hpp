#pragma once

#include "pramaan/ir.hpp"
#include "pramaan/simplex.hpp"

namespace pramaan {
namespace gpu {

struct PdhgOptions {
    double tolerance = 1e-6;
    int max_iterations = 100000;
    int check_frequency = 100;
};

class PdhgSolver {
public:
    PdhgSolver(const PdhgOptions& options = PdhgOptions());
    ~PdhgSolver();

    SolveResult solve(const ModelIR& model);

private:
    PdhgOptions options_;
};

} // namespace gpu
} // namespace pramaan
