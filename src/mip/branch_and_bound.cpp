#include <iostream>
#include <algorithm>
// branch_and_bound.cpp -- see include/pramaan/branch_and_bound.hpp
#include "pramaan/branch_and_bound.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>

#include "pramaan/dual_simplex.hpp"

namespace pramaan {
namespace mip {

namespace {

using Index = CSRMatrix::Index;

struct Relaxation {
    SolveResult result;
    BasisState basis;
    bool warm_attempted = false;
    bool warm_succeeded = false;
};

// Solves the LP relaxation of `node_model`.
// When use_warm_start is true and the node has an inherited basis, attempts
// DualSimplex::warmSolve() first:
//   - kOptimal      -> warm-start succeeded; return immediately.
//   - kWarmStartRejected -> basis structurally incompatible; fall through to cold solve.
//   - any other status   -> inconclusive failure; return immediately (NOT a warm-start success).
// Cold solve is always used when warm-start is disabled or has no basis.
Relaxation solveRelaxation(const ModelIR& node_model, const Node& node,
                           const RevisedSimplex::Options& lp_options, bool use_warm_start) {
    Relaxation out;
    if (use_warm_start && node.hasInheritedBasis()) {
        const DualSimplex dual(lp_options);
        out.warm_attempted = true;
        try {
            out.result = dual.warmSolve(node_model, node.inheritedBasis(), &out.basis);
            if (out.result.status == SolveStatus::kOptimal) {
                out.warm_succeeded = true;
                return out;
            } else if (out.result.status == SolveStatus::kWarmStartRejected) {
                // Basis structurally incompatible; fall through to cold solve.
                out.basis = BasisState{};
            } else {
                // kIterationLimit or kNumericalFailure: not a warm-start success.
                // Do NOT silently turn this into a successful solve.
                return out;
            }
        } catch (const std::invalid_argument&) {
            out.basis = BasisState{};
        }
    }
    const DualSimplex dual(lp_options);
    out.basis = dual.captureBasis(node_model, &out.result);
    return out;
}

}  // namespace

bool isIntegerFeasible(const ModelIR& model, const std::vector<double>& x, double tolerance) {
    return firstFractionalInteger(model, x, tolerance) < 0;
}

Index firstFractionalInteger(const ModelIR& model, const std::vector<double>& x, double tolerance) {
    if (x.size() != static_cast<std::size_t>(model.numVars())) return -1;
    for (Index j = 0; j < model.numVars(); ++j) {
        if (!model.isInteger(j)) continue;
        const double v = x[static_cast<std::size_t>(j)];
        if (std::abs(v - std::round(v)) > tolerance) return j;
    }
    return -1;
}


// --- Cover Cut Generator (Knapsack Cover) ---
//
// Validity criteria (strictly enforced to prevent invalid cuts):
//   For each row, we ONLY attempt to generate a cover cut when:
//     1. row_lower[i] <= -kInfinity  (pure <= row, not ranged or >=)
//     2. row_upper[i] is finite
//     3. ALL nonzero coefficients in the row are finite and positive
//     4. ALL variables with nonzero coefficients are VarType::kInteger with
//        exact bounds [0, 1]  (binary variables)
//   If ANY variable in the row fails criteria 4, the row is SKIPPED entirely.
//   This is the only valid case for the current simple cover cut family.
//
// Statistics semantics (out_ parameters):
//   out_generated: incremented for each non-duplicate cut whose violation was tested
//   out_duplicates: incremented for each cut already present in existing_cuts
//   Applied/rejected accounting happens at the call site.
void generateCoverCuts(const ModelIR& model, const std::vector<double>& x,
                       const std::vector<Cut>& existing_cuts, std::vector<Cut>& cuts,
                       int& out_generated, int& out_duplicates) {
    const double tol = 1e-4;
    for (CSRMatrix::Index i = 0; i < model.numRows(); ++i) {
        // Only pure <= rows (row_lower = -inf, row_upper finite).
        if (model.row_lower[i] > -kInfinity) continue;
        if (model.row_upper[i] >= kInfinity) continue;

        // Scan ALL nonzero entries: if any fails the binary-integer criteria,
        // skip this row entirely. This prevents generating invalid cuts for
        // mixed rows (binary + continuous, binary + general integer, etc.).
        auto rv = model.A.row(i);
        bool row_valid = true;
        std::vector<std::pair<double, CSRMatrix::Index>> items;

        for (auto entry : rv) {
            // Coefficient must be finite and strictly positive.
            if (!std::isfinite(entry.value) || entry.value <= 0.0) {
                row_valid = false;
                break;
            }
            // Variable must be integer with exact [0, 1] bounds (binary).
            if (!model.isInteger(entry.index)) {
                row_valid = false;
                break;
            }
            if (model.var_lower[entry.index] != 0.0 || model.var_upper[entry.index] != 1.0) {
                row_valid = false;
                break;
            }
            items.push_back({entry.value, entry.index});
        }

        if (!row_valid || items.empty()) continue;

        // Sort by decreasing weight to build the minimal cover greedily.
        std::sort(items.rbegin(), items.rend());

        double current_weight = 0.0;
        double current_x_sum = 0.0;
        std::vector<CSRMatrix::Index> cover;

        for (const auto& item : items) {
            current_weight += item.first;
            current_x_sum += x[static_cast<std::size_t>(item.second)];
            cover.push_back(item.second);

            if (current_weight > model.row_upper[i] + tol) {
                // We have a cover (sum of weights exceeds capacity).
                // Check if it is violated: sum(x_j for j in cover) > |cover| - 1.
                if (current_x_sum > static_cast<double>(cover.size() - 1) + tol) {
                    Cut c;
                    c.rhs = static_cast<double>(cover.size()) - 1.0;
                    for (auto idx : cover) {
                        c.cols.push_back(idx);
                        c.vals.push_back(1.0);
                    }

                    // Check for duplicate against existing cuts AND cuts already
                    // produced in this same pass (so two rows that generate the
                    // same cover are both counted correctly).
                    auto isCutEqual = [](const Cut& a, const Cut& b) -> bool {
                        if (a.cols.size() != b.cols.size()) return false;
                        if (std::abs(a.rhs - b.rhs) > 1e-6) return false;
                        for (std::size_t k = 0; k < a.cols.size(); ++k) {
                            if (a.cols[k] != b.cols[k] || std::abs(a.vals[k] - b.vals[k]) > 1e-6)
                                return false;
                        }
                        return true;
                    };

                    bool is_duplicate = false;
                    for (const auto& ec : existing_cuts) {
                        if (isCutEqual(ec, c)) { is_duplicate = true; break; }
                    }
                    if (!is_duplicate) {
                        for (const auto& nc : cuts) {
                            if (isCutEqual(nc, c)) { is_duplicate = true; break; }
                        }
                    }

                    if (is_duplicate) {
                        // Duplicate: do NOT count as generated, applied, or rejected.
                        out_duplicates++;
                    } else {
                        // Non-duplicate: count as generated; applied/rejected at call site.
                        out_generated++;
                        cuts.push_back(std::move(c));
                    }
                }
                break;
            }
        }
    }
}

MipResult BranchAndBound::solve(const ModelIR& model) const {
    model.validate();

    MipResult out;
    if (options_.max_nodes <= 0) {
        out.status = MipStatus::kNodeLimit;
        return out;
    }
    const double tol = options_.lp_options.tolerance;
    const double int_tol = options_.integrality_tolerance;
    const bool minimize = (model.obj_sense == ObjSense::kMinimize);
    const int num_vars = model.numVars();

    const auto improves = [&](double candidate, double incumbent) {
        return minimize ? (candidate < incumbent - tol) : (candidate > incumbent + tol);
    };
    const auto cannotImprove = [&](double bound, double incumbent) {
        return minimize ? (bound >= incumbent - tol) : (bound <= incumbent + tol);
    };

    std::mutex mtx;
    std::condition_variable cv;

    bool have_incumbent = false;
    double incumbent_obj = minimize ? kInfinity : -kInfinity;

    int active_threads = 0;
    std::atomic<int> reserved_nodes{0};

    // Termination state: once set to true, no more nodes are started.
    bool terminate = false;

    // Separate flag for node-limit termination. `terminate` is a general
    // shutdown signal (set for any reason: numerical failure, infeasible root,
    // node limit). `node_limit_hit` is only set when the node limit was
    // actually exhausted, so the final status can distinguish kNodeLimit from
    // kInfeasible even though both set `terminate = true`.
    bool node_limit_hit = false;

    // Precedence: kNumericalFailure > kNodeLimit > kOptimal/kInfeasible.
    // Track whether any worker set a "sticky" status (numerical failure).
    bool numerical_failure = false;

    // Create root node
    auto root = std::make_unique<Node>(Node::makeRoot(model));
    ++out.statistics.nodes_created;

    struct TaskNode {
        std::unique_ptr<Node> node;
        double parent_bound;
        bool minimize;
        bool operator<(const TaskNode& other) const {
            if (minimize) return parent_bound > other.parent_bound;
            return parent_bound < other.parent_bound;
        }
    };

    std::vector<TaskNode> dfs_stack;
    std::priority_queue<TaskNode> best_bound_pq;

    auto push_node = [&](std::unique_ptr<Node> n, double parent_bound) {
        if (options_.node_selection == Options::NodeSelection::kDepthFirst) {
            dfs_stack.push_back({std::move(n), parent_bound, minimize});
        } else {
            best_bound_pq.push({std::move(n), parent_bound, minimize});
        }
    };

    push_node(std::move(root), minimize ? -kInfinity : kInfinity);

    auto pop_node = [&]() -> std::unique_ptr<Node> {
        if (options_.node_selection == Options::NodeSelection::kDepthFirst) {
            if (dfs_stack.empty()) return nullptr;
            auto n = std::move(dfs_stack.back().node);
            dfs_stack.pop_back();
            return n;
        } else {
            if (best_bound_pq.empty()) return nullptr;
            auto n = std::move(const_cast<TaskNode&>(best_bound_pq.top()).node);
            best_bound_pq.pop();
            return n;
        }
    };

    auto is_empty = [&]() {
        if (options_.node_selection == Options::NodeSelection::kDepthFirst) {
            return dfs_stack.empty();
        } else {
            return best_bound_pq.empty();
        }
    };

    int num_workers = std::max(1, options_.num_threads);
    std::vector<std::thread> workers;

    auto worker_loop = [&]() {
        while (true) {
            std::unique_ptr<Node> current_node;

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]() {
                    return !is_empty() || terminate || (active_threads == 0);
                });

                if (terminate || (is_empty() && active_threads == 0)) {
                    return;
                }

                if (!is_empty()) {
                    // Reserve a node slot atomically. If max_nodes is already reached,
                    // set terminate (and node_limit_hit) and return without evaluating.
                    int current = reserved_nodes.fetch_add(1, std::memory_order_relaxed);
                    if (current >= options_.max_nodes) {
                        node_limit_hit = true;
                        terminate = true;
                        cv.notify_all();
                        return;
                    }
                    current_node = pop_node();
                    ++active_threads;
                } else {
                    continue; // spurious wakeup or waiting for others to push
                }
            }

            // --- EVALUATE NODE ---
            bool node_pruned = false;

            if (current_node->hasInconsistentBounds(tol)) {
                current_node->markInfeasible();
                node_pruned = true;

                std::lock_guard<std::mutex> lock(mtx);
                ++out.statistics.nodes_explored;
                ++out.statistics.nodes_pruned_infeasible;
            } else {
                ModelIR node_model = current_node->buildRelaxation(model);

                // Cut-and-resolve loop.
                // Correctness: after adding a cut, the LP structure changes.
                // We MUST NOT warm-start the augmented LP with a pre-cut basis.
                // Strategy: invalidate the inherited basis on the node before each
                // cut-augmented re-solve; after the final solve capture the new basis.
                bool cut_added = false;
                Relaxation rel;
                do {
                    cut_added = false;

                    // After a cut was added in the previous iteration, the node_model
                    // has been rebuilt but the node's inherited_basis is stale (it came
                    // from the parent, not the pre-cut solve). Ensure we cold-solve.
                    rel = solveRelaxation(node_model, *current_node,
                                         options_.lp_options, options_.use_warm_start);

                    if (rel.result.status == SolveStatus::kOptimal) {
                        std::vector<Cut> new_cuts;
                        int generated = 0, duplicates = 0;
                        generateCoverCuts(node_model, rel.result.x, current_node->cuts(),
                                          new_cuts, generated, duplicates);

                        int applied = 0;
                        int rejected = 0;
                        for (auto& c : new_cuts) {
                            if (cutViolation(c, rel.result.x) > 1e-4) {
                                current_node->addCut(std::move(c), num_vars);
                                cut_added = true;
                                applied++;
                            } else {
                                rejected++;
                            }
                        }

                        if (generated > 0 || duplicates > 0 || applied > 0 || rejected > 0) {
                            std::unique_lock<std::mutex> lock(mtx);
                            out.statistics.cuts_generated += generated;
                            out.statistics.duplicate_cuts += duplicates;
                            out.statistics.cuts_applied += applied;
                            out.statistics.cuts_rejected += rejected;
                        }

                        if (cut_added) {
                            // Rebuild the LP with the new cut and cold-solve next iteration.
                            node_model = current_node->buildRelaxation(model);
                            // Invalidate the inherited basis: the pre-cut basis is not
                            // valid for the augmented LP.
                            current_node->clearInheritedBasis();
                        }
                    }
                } while (cut_added);

                // Record the final solve result. `rel.basis` is the post-cut optimal
                // basis (or empty if the solve was not optimal). This is what children
                // will warm-start from.
                current_node->setRelaxation(std::move(rel.result), std::move(rel.basis));

                std::unique_lock<std::mutex> lock(mtx);
                ++out.statistics.nodes_explored;
                if (rel.warm_attempted) ++out.statistics.warm_start_attempts;
                if (rel.warm_succeeded) ++out.statistics.warm_start_successes;

                const bool is_root = current_node->branch().isRoot();

                switch (current_node->state()) {
                    case NodeState::kInfeasible:
                        if (is_root) {
                            out.status = MipStatus::kInfeasible;
                            terminate = true;
                        }
                        ++out.statistics.nodes_pruned_infeasible;
                        node_pruned = true;
                        break;

                    case NodeState::kUnbounded:
                        // An unbounded LP relaxation does NOT by itself prove that
                        // the MILP is unbounded. Return kNumericalFailure (inconclusive).
                        // A proper MILP unboundedness proof is not implemented.
                        numerical_failure = true;
                        terminate = true;
                        node_pruned = true;
                        break;

                    case NodeState::kFailed:
                    case NodeState::kUnevaluated:
                        numerical_failure = true;
                        terminate = true;
                        node_pruned = true;
                        break;

                    case NodeState::kOptimal:
                        break;
                }

                if (!node_pruned && terminate) {
                    node_pruned = true;
                }

                if (!node_pruned) {
                    const double node_bound = current_node->relaxationObjective();
                    if (have_incumbent && cannotImprove(node_bound, incumbent_obj)) {
                        ++out.statistics.nodes_pruned_by_bound;
                        node_pruned = true;
                    } else {
                        // Check integrality
                        const std::vector<double>& x = current_node->relaxation().x;
                        const Index branch_var = firstFractionalInteger(model, x, int_tol);

                        if (branch_var < 0) {
                            // Integer feasible
                            if (!have_incumbent || improves(node_bound, incumbent_obj)) {
                                have_incumbent = true;
                                incumbent_obj = node_bound;
                                out.x = x;
                                out.objective_value = node_bound;
                                ++out.statistics.incumbent_updates;
                            }
                            node_pruned = true;
                        } else {
                            // Basic Rounding Heuristic
                            std::vector<double> rounded_x = x;
                            for (Index j = 0; j < model.numVars(); ++j) {
                                if (model.isInteger(j)) {
                                    rounded_x[static_cast<std::size_t>(j)] = std::round(x[static_cast<std::size_t>(j)]);
                                }
                            }

                            // Check feasibility of rounded_x
                            bool feasible = true;
                            auto activity = model.A.multiply(rounded_x);
                            for (Index r = 0; r < model.numRows(); ++r) {
                                if (activity[static_cast<std::size_t>(r)] < model.row_lower[static_cast<std::size_t>(r)] - tol ||
                                    activity[static_cast<std::size_t>(r)] > model.row_upper[static_cast<std::size_t>(r)] + tol) {
                                    feasible = false;
                                    break;
                                }
                            }
                            for (Index j = 0; j < model.numVars(); ++j) {
                                if (rounded_x[static_cast<std::size_t>(j)] < model.var_lower[static_cast<std::size_t>(j)] - tol ||
                                    rounded_x[static_cast<std::size_t>(j)] > model.var_upper[static_cast<std::size_t>(j)] + tol) {
                                    feasible = false;
                                    break;
                                }
                            }

                            if (feasible) {
                                double rounded_obj = model.obj_offset;
                                for (Index j = 0; j < model.numVars(); ++j) {
                                    rounded_obj += model.obj_coeffs[static_cast<std::size_t>(j)] * rounded_x[static_cast<std::size_t>(j)];
                                }
                                if (!have_incumbent || improves(rounded_obj, incumbent_obj)) {
                                    have_incumbent = true;
                                    incumbent_obj = rounded_obj;
                                    out.x = rounded_x;
                                    out.objective_value = rounded_obj;
                                    ++out.statistics.incumbent_updates;
                                }
                            }

                            // Branch. Pass the post-cut optimal basis to children for warm-starting.
                            const double v = x[static_cast<std::size_t>(branch_var)];
                            const double floor_v = std::floor(v);
                            const double ceil_v = std::ceil(v);

                            auto up_child = std::make_unique<Node>(
                                current_node->makeChild(branch_var, BranchDirection::kUp, ceil_v,
                                                        current_node->optimalBasis()));
                            auto down_child = std::make_unique<Node>(
                                current_node->makeChild(branch_var, BranchDirection::kDown, floor_v,
                                                        current_node->optimalBasis()));

                            out.statistics.nodes_created += 2;
                            ++out.statistics.branchings;

                            // Queued children inherit the strongest currently available
                            // valid bound from their parent (node_bound). This is valid
                            // because the child's feasible region is a subset of the parent's.
                            push_node(std::move(up_child), node_bound);
                            push_node(std::move(down_child), node_bound);

                            cv.notify_one();
                            cv.notify_one();
                        }
                    }
                }
                lock.unlock();
                if (terminate) cv.notify_all();
            }

            {
                std::lock_guard<std::mutex> lock(mtx);
                --active_threads;
                if (active_threads == 0 && is_empty()) {
                    cv.notify_all();
                }
            }
        }
    };

    // Special case for single thread to avoid threading overhead in tests
    if (num_workers == 1) {
        worker_loop();
    } else {
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back(worker_loop);
        }
        for (auto& w : workers) {
            w.join();
        }
    }

    // Determine final status with correct precedence:
    //   kNumericalFailure > kNodeLimit > kOptimal / kInfeasible
    //
    // kOptimal:          tree exhausted, incumbent exists, no node limit, no failure.
    // kInfeasible:       tree exhausted (or root infeasible), no incumbent, no node limit, no failure.
    // kNodeLimit:        search stopped because node_limit_hit was set; proof incomplete.
    // kNumericalFailure: an LP relaxation was inconclusive (highest precedence).
    //
    // Note: root-LP-infeasible sets `terminate` but NOT `node_limit_hit`, so it
    // correctly falls through to the kOptimal/kInfeasible branch.
    if (numerical_failure) {
        out.status = MipStatus::kNumericalFailure;
    } else if (node_limit_hit) {
        // Node limit was actually exhausted; proof is incomplete.
        out.status = MipStatus::kNodeLimit;
    } else {
        // Tree exhausted (including early-terminated-infeasible root):
        // return whichever terminal status is correct.
        out.status = have_incumbent ? MipStatus::kOptimal : MipStatus::kInfeasible;
    }

    return out;
}

}  // namespace mip
}  // namespace pramaan
