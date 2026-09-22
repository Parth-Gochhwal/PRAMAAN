// fingerprint.cpp
// PILLAR 1 (Structure) -- Structural Fingerprint Engine
//
#include "pramaan/structure/fingerprint.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

namespace pramaan {
namespace structure {

namespace {

class UnionFind {
public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }
private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

}  // namespace

StructuralFingerprint analyze(const CSRMatrix& A) {
    const Index m = A.numRows();
    const Index n = A.numCols();
    StructuralFingerprint result;
    if (n == 0) return result;

    // 1. Identify coupling constraints using Articulation Points (O(M+N+NNZ)).
    // A constraint is coupling if its removal increases the number of variable
    // components that have at least one remaining constraint.
    // This is mathematically equivalent to finding constraint-node articulation points among
    // nodes in the bipartite graph restricted to variables of degree >= 2.
    std::vector<int> var_degree(n, 0);
    for (Index i = 0; i < m; ++i) {
        for (auto entry : A.row(i)) {
            var_degree[entry.index]++;
        }
    }

    // Build reduced graph: nodes 0..m-1 are constraints, m..m+n-1 are variables.
    int total_nodes = m + n;
    std::vector<std::vector<int>> adj(total_nodes);
    for (Index i = 0; i < m; ++i) {
        for (auto entry : A.row(i)) {
            if (var_degree[entry.index] >= 2) {
                adj[i].push_back(m + entry.index);
                adj[m + entry.index].push_back(i);
            }
        }
    }

    std::vector<int> disc(total_nodes, -1);
    std::vector<int> low(total_nodes, -1);
    std::vector<int> parent(total_nodes, -1);
    std::vector<bool> is_coupling(m, false);
    int time = 0;

    // Iterative Tarjan articulation-point DFS avoids recursion-depth
    // limitations on large LP matrices.
    struct DFSNode {
        int u;
        int i; // edge index
        int children;
    };
    std::vector<DFSNode> stack;

    for (int start_node = 0; start_node < total_nodes; ++start_node) {
        if (disc[start_node] != -1) continue;

        stack.push_back({start_node, 0, 0});
        disc[start_node] = low[start_node] = ++time;

        while (!stack.empty()) {
            auto& frame = stack.back();
            int u = frame.u;

            if (frame.i < (int)adj[u].size()) {
                int v = adj[u][frame.i++];
                if (disc[v] == -1) {
                    parent[v] = u;
                    frame.children++;
                    disc[v] = low[v] = ++time;
                    stack.push_back({v, 0, 0});
                } else if (v != parent[u]) {
                    low[u] = std::min(low[u], disc[v]);
                }
            } else {
                // Post-visit
                stack.pop_back();
                if (!stack.empty()) {
                    int p = stack.back().u;
                    low[p] = std::min(low[p], low[u]);
                    if (parent[p] == -1 && stack.back().children > 1) {
                        if (p < m) is_coupling[p] = true;
                    }
                    if (parent[p] != -1 && low[u] >= disc[p]) {
                        if (p < m) is_coupling[p] = true;
                    }
                }
            }
        }
    }

    for (Index i = 0; i < m; ++i) {
        if (is_coupling[i]) {
            result.coupling_constraints.push_back(i);
        }
    }

    // 2. Build blocks using ONLY non-coupling constraints.
    UnionFind uf(n);
    for (Index i = 0; i < m; ++i) {
        if (is_coupling[i]) continue;
        auto rv = A.row(i);
        if (rv.size() == 0) continue;
        Index first_col = -1;
        for (auto entry : rv) {
            if (first_col < 0) {
                first_col = entry.index;
            } else {
                uf.unite(first_col, entry.index);
            }
        }
    }

    // Map each variable to a sequential block ID.
    std::map<int, int> rep_to_block;
    std::vector<int> var_block(n);
    int num_blocks = 0;
    for (Index j = 0; j < n; ++j) {
        int rep = uf.find(j);
        auto it = rep_to_block.find(rep);
        if (it == rep_to_block.end()) {
            rep_to_block[rep] = num_blocks;
            var_block[j] = num_blocks;
            ++num_blocks;
        } else {
            var_block[j] = it->second;
        }
    }

    std::vector<Block> blocks(num_blocks);
    for (Index j = 0; j < n; ++j) {
        blocks[var_block[j]].variables.push_back(j);
    }

    // Assign non-coupling, non-empty rows to their blocks.
    for (Index i = 0; i < m; ++i) {
        if (is_coupling[i]) continue;
        auto rv = A.row(i);
        if (rv.size() == 0) continue;
        int b = var_block[(*rv.begin()).index];
        blocks[b].constraints.push_back(i);
    }

    // Sort for determinism.
    for (auto& block : blocks) {
        std::sort(block.constraints.begin(), block.constraints.end());
    }
    std::sort(blocks.begin(), blocks.end(),
              [](const Block& a, const Block& b) {
                  if (a.variables.empty() && b.variables.empty()) return false;
                  if (a.variables.empty()) return false;
                  if (b.variables.empty()) return true;
                  return a.variables.front() < b.variables.front();
              });
    std::sort(result.coupling_constraints.begin(), result.coupling_constraints.end());

    result.blocks = std::move(blocks);
    return result;
}

}  // namespace structure
}  // namespace pramaan
