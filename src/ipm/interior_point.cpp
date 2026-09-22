#include "pramaan/ipm/interior_point.hpp"
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace pramaan {

namespace {

// -----------------------------------------------------------------------
// Dense Matrix and LU Factorization Helper
// -----------------------------------------------------------------------
struct DenseMatrix {
    int rows;
    int cols;
    std::vector<double> data;

    DenseMatrix() : rows(0), cols(0) {}
    DenseMatrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& at(int r, int c) { return data[r * cols + c]; }
    const double& at(int r, int c) const { return data[r * cols + c]; }
};

class DenseLU {
public:
    DenseLU(const DenseMatrix& A) : n(A.rows), lu(A), p(n) {
        for (int i = 0; i < n; ++i) p[i] = i;
    }

    bool factorize(double tol = 1e-12) {
        if (n == 0) return true;

        // Find max element magnitude for scale-aware tolerance
        double max_mag = 0.0;
        for (int i = 0; i < n * n; ++i) {
            max_mag = std::max(max_mag, std::abs(lu.data[i]));
        }
        double pivot_tol = tol * std::max(1.0, max_mag);

        for (int k = 0; k < n; ++k) {
            int best = k;
            double best_val = std::abs(lu.at(k, k));
            for (int i = k + 1; i < n; ++i) {
                double v = std::abs(lu.at(i, k));
                if (v > best_val) { best = i; best_val = v; }
            }
            if (best_val < pivot_tol) return false; // Numerically singular

            if (best != k) {
                std::swap(p[k], p[best]);
                for (int j = 0; j < n; ++j) {
                    std::swap(lu.at(k, j), lu.at(best, j));
                }
            }

            double pivot = lu.at(k, k);
            for (int i = k + 1; i < n; ++i) {
                lu.at(i, k) /= pivot;
                for (int j = k + 1; j < n; ++j) {
                    lu.at(i, j) -= lu.at(i, k) * lu.at(k, j);
                }
            }
        }
        return true;
    }

    std::vector<double> solve(const std::vector<double>& b) const {
        if (n == 0) return {};
        std::vector<double> x(n);
        for (int i = 0; i < n; ++i) x[i] = b[p[i]];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                x[i] -= lu.at(i, j) * x[j];
            }
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                x[i] -= lu.at(i, j) * x[j];
            }
            x[i] /= lu.at(i, i);
        }
        return x;
    }

private:
    int n;
    DenseMatrix lu;
    std::vector<int> p;
};

// -----------------------------------------------------------------------
// Standard-form LP data:  min c^T x,  A x = b,  x >= 0.
// -----------------------------------------------------------------------
struct StandardForm {
    int m = 0;
    int n = 0;
    DenseMatrix A;
    std::vector<double> b;
    std::vector<double> c;
    double obj_offset = 0.0;

    int num_orig_vars = 0;
    std::vector<double> shift;
    std::vector<double> sign_mul;
    std::vector<int> pos_idx;
    std::vector<int> neg_idx;
};

StandardForm convertToStandardForm(const ModelIR& model) {
    const int orig_m = model.numRows();
    const int orig_n = model.numVars();
    const bool negate_obj = (model.obj_sense == ObjSense::kMaximize);

    StandardForm sf;
    sf.num_orig_vars = orig_n;
    sf.shift.resize(orig_n, 0.0);
    sf.sign_mul.resize(orig_n, 1.0);
    sf.pos_idx.resize(orig_n, -1);
    sf.neg_idx.resize(orig_n, -1);

    int std_n = 0;
    for (int j = 0; j < orig_n; ++j) {
        const double lb = model.var_lower[j];
        const double ub = model.var_upper[j];
        const bool free_lb = (lb <= -kInfinity);
        const bool free_ub = (ub >= kInfinity);

        if (!free_lb) {
            sf.pos_idx[j] = std_n++;
            sf.shift[j] = lb;
            sf.sign_mul[j] = 1.0;
        } else if (!free_ub) {
            sf.pos_idx[j] = std_n++;
            sf.shift[j] = ub;
            sf.sign_mul[j] = -1.0;
        } else {
            sf.pos_idx[j] = std_n++;
            sf.neg_idx[j] = std_n++;
            sf.shift[j] = 0.0;
            sf.sign_mul[j] = 1.0;
        }
    }

    std::vector<int> ub_slack_idx(orig_n, -1);
    for (int j = 0; j < orig_n; ++j) {
        const double lb = model.var_lower[j];
        const double ub = model.var_upper[j];
        if (lb > -kInfinity && ub < kInfinity) {
            ub_slack_idx[j] = std_n++;
        }
    }

    std::vector<int> row_le_slack(orig_m, -1);
    std::vector<int> row_ge_slack(orig_m, -1);
    for (int i = 0; i < orig_m; ++i) {
        const double rl = model.row_lower[i];
        const double ru = model.row_upper[i];
        const bool has_lower = (rl > -kInfinity);
        const bool has_upper = (ru < kInfinity);
        if (has_lower && has_upper && std::abs(rl - ru) < 1e-15) {
            // Equality
        } else {
            if (has_upper) row_le_slack[i] = std_n++;
            if (has_lower) row_ge_slack[i] = std_n++;
        }
    }

    sf.n = std_n;

    int std_m = 0;
    for (int j = 0; j < orig_n; ++j) {
        if (ub_slack_idx[j] >= 0) ++std_m;
    }
    for (int i = 0; i < orig_m; ++i) {
        const double rl = model.row_lower[i];
        const double ru = model.row_upper[i];
        const bool has_lower = (rl > -kInfinity);
        const bool has_upper = (ru < kInfinity);
        if (has_lower && has_upper && std::abs(rl - ru) < 1e-15) {
            ++std_m;
        } else {
            if (has_upper) ++std_m;
            if (has_lower) ++std_m;
        }
    }
    sf.m = std_m;

    sf.A = DenseMatrix(std_m, std_n);
    sf.b.resize(std_m, 0.0);
    sf.c.resize(std_n, 0.0);

    sf.obj_offset = model.obj_offset;
    for (int j = 0; j < orig_n; ++j) {
        const double c_orig = model.obj_coeffs[j];
        sf.obj_offset += c_orig * sf.shift[j];

        double c_pos = c_orig * sf.sign_mul[j];
        if (negate_obj) c_pos = -c_pos;
        sf.c[sf.pos_idx[j]] = c_pos;

        if (sf.neg_idx[j] >= 0) {
            double c_neg = -c_orig;
            if (negate_obj) c_neg = -c_neg;
            sf.c[sf.neg_idx[j]] = c_neg;
        }
    }

    int row_eq = 0;
    for (int j = 0; j < orig_n; ++j) {
        if (ub_slack_idx[j] >= 0) {
            sf.A.at(row_eq, sf.pos_idx[j]) = 1.0;
            sf.A.at(row_eq, ub_slack_idx[j]) = 1.0;
            sf.b[row_eq] = model.var_upper[j] - model.var_lower[j];
            ++row_eq;
        }
    }

    for (int i = 0; i < orig_m; ++i) {
        const double rl = model.row_lower[i];
        const double ru = model.row_upper[i];
        const bool has_lower = (rl > -kInfinity);
        const bool has_upper = (ru < kInfinity);
        const bool is_eq = (has_lower && has_upper && std::abs(rl - ru) < 1e-15);

        auto rv = model.A.row(i);
        auto fillRow = [&](int eq_row) {
            double rhs_shift = 0.0;
            for (auto entry : rv) {
                const int ej = entry.index;
                const double aij = entry.value;
                sf.A.at(eq_row, sf.pos_idx[ej]) += aij * sf.sign_mul[ej];
                if (sf.neg_idx[ej] >= 0) {
                    sf.A.at(eq_row, sf.neg_idx[ej]) += -aij;
                }
                rhs_shift += aij * sf.shift[ej];
            }
            return rhs_shift;
        };

        if (is_eq) {
            const double rhs_shift = fillRow(row_eq);
            sf.b[row_eq] = rl - rhs_shift;
            ++row_eq;
        } else {
            if (has_upper) {
                const double rhs_shift = fillRow(row_eq);
                sf.A.at(row_eq, row_le_slack[i]) = 1.0;
                sf.b[row_eq] = ru - rhs_shift;
                ++row_eq;
            }
            if (has_lower) {
                const double rhs_shift = fillRow(row_eq);
                sf.A.at(row_eq, row_ge_slack[i]) = -1.0;
                sf.b[row_eq] = rl - rhs_shift;
                ++row_eq;
            }
        }
    }

    return sf;
}

double step_length(const std::vector<double>& v, const std::vector<double>& dv, double eta) {
    double alpha = 1.0;
    for (size_t j = 0; j < v.size(); ++j) {
        if (dv[j] < 0.0) {
            double ratio = -v[j] / dv[j];
            if (ratio < alpha) alpha = ratio;
        }
    }
    return alpha * eta; // safe scaling
}

}  // anonymous namespace

SolveResult InteriorPointSolver::solve(const ModelIR& model) const {
    model.validate();
    SolveResult result;

    // Reject non-continuous models for LP IPM
    for (auto vt : model.var_types) {
        if (vt != VarType::kContinuous) {
            result.status = SolveStatus::kNumericalFailure; // Cannot solve MILP with IPM alone
            return result;
        }
    }

    if (options_.max_iterations == 0) {
        result.status = SolveStatus::kIterationLimit;
        return result;
    }

    if (options_.max_iterations < 0 || options_.tolerance <= 0.0 ||
        options_.barrier_reduction < 0.0 || options_.barrier_reduction > 1.0 ||
        options_.initial_bound_slack <= 0.0) {
        result.status = SolveStatus::kNumericalFailure;
        return result;
    }

    const int orig_n = model.numVars();
    if (orig_n == 0) {
        result.status = SolveStatus::kOptimal;
        result.objective_value = model.obj_offset;
        result.iterations = 0;
        return result;
    }

    StandardForm sf = convertToStandardForm(model);
    const int m = sf.m;
    const int n = sf.n;
    const bool negate_obj = (model.obj_sense == ObjSense::kMaximize);

    if (m == 0) {
        bool has_neg = false;
        for (int j = 0; j < n; ++j) {
            if (sf.c[j] < -1e-15) { has_neg = true; break; }
        }
        if (!has_neg) {
            result.status = SolveStatus::kOptimal;
            result.x.resize(orig_n);
            for (int j = 0; j < orig_n; ++j) {
                result.x[j] = sf.shift[j];
            }
            result.objective_value = model.obj_offset;
            for (int j = 0; j < orig_n; ++j) {
                result.objective_value += model.obj_coeffs[j] * result.x[j];
            }
            result.row_activity = model.A.multiply(result.x);
            result.is_basic.assign(orig_n, false);
            result.iterations = 0;
            return result;
        } else {
            result.status = SolveStatus::kUnbounded;
            return result;
        }
    }

    double norm_b = 0.0;
    for (int i = 0; i < m; ++i) norm_b = std::max(norm_b, std::abs(sf.b[i]));
    double norm_c = 0.0;
    for (int j = 0; j < n; ++j) norm_c = std::max(norm_c, std::abs(sf.c[j]));

    // Initialization
    std::vector<double> x(n, options_.initial_bound_slack);
    std::vector<double> s(n, options_.initial_bound_slack);
    std::vector<double> y(m, 0.0);

    const double tol = options_.tolerance;
    int iter = 0;
    for (; iter < options_.max_iterations; ++iter) {
        std::vector<double> rp(m);
        double rp_norm = 0.0;
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) sum += sf.A.at(i, j) * x[j];
            rp[i] = sf.b[i] - sum;
            rp_norm = std::max(rp_norm, std::abs(rp[i]));
        }

        std::vector<double> rd(n);
        double rd_norm = 0.0;
        for (int j = 0; j < n; ++j) {
            double ATy_j = 0.0;
            for (int i = 0; i < m; ++i) ATy_j += sf.A.at(i, j) * y[i];
            rd[j] = sf.c[j] - ATy_j - s[j];
            rd_norm = std::max(rd_norm, std::abs(rd[j]));
        }

        double mu = 0.0;
        for (int j = 0; j < n; ++j) mu += x[j] * s[j];
        if (n > 0) mu /= n;

        // Scale-aware convergence check
        if (rp_norm / (1.0 + norm_b) < tol &&
            rd_norm / (1.0 + norm_c) < tol &&
            mu < tol) {
            result.status = SolveStatus::kOptimal;
            break;
        }

        // Setup Normal Equations: M = A (X S^{-1}) A^T
        DenseMatrix M(m, m);
        std::vector<double> x_over_s(n);
        for (int j = 0; j < n; ++j) {
            x_over_s[j] = x[j] / s[j];
        }

        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < m; ++k) {
                double val = 0.0;
                for (int j = 0; j < n; ++j) {
                    val += sf.A.at(i, j) * x_over_s[j] * sf.A.at(k, j);
                }
                M.at(i, k) = val;
            }
        }

        DenseLU lu(M);
        if (!lu.factorize()) {
            result.status = SolveStatus::kNumericalFailure;
            break;
        }

        // Predictor step: rc_aff = -X S e
        std::vector<double> rhs_system_aff(m, 0.0);
        std::vector<double> dx_aff_inner(n, 0.0);
        for (int j = 0; j < n; ++j) {
            dx_aff_inner[j] = -rd[j] - s[j];
        }
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += sf.A.at(i, j) * x_over_s[j] * dx_aff_inner[j];
            }
            rhs_system_aff[i] = rp[i] - sum;
        }

        std::vector<double> dy_aff = lu.solve(rhs_system_aff);
        std::vector<double> dx_aff(n);
        std::vector<double> ds_aff(n);
        for (int j = 0; j < n; ++j) {
            double ATdy = 0.0;
            for (int i = 0; i < m; ++i) ATdy += sf.A.at(i, j) * dy_aff[i];
            dx_aff[j] = x_over_s[j] * (ATdy + dx_aff_inner[j]);
            ds_aff[j] = -s[j] - (s[j] / x[j]) * dx_aff[j];
        }

        // Mehrotra Centering Parameter
        double alpha_aff_p = step_length(x, dx_aff, 1.0);
        double alpha_aff_d = step_length(s, ds_aff, 1.0);

        double mu_aff = 0.0;
        for (int j = 0; j < n; ++j) {
            mu_aff += (x[j] + alpha_aff_p * dx_aff[j]) * (s[j] + alpha_aff_d * ds_aff[j]);
        }
        if (n > 0) mu_aff /= n;

        double sigma = std::pow(mu_aff / mu, 3.0);
        sigma = std::max(0.0, std::min(1.0, sigma));
        double mu_target = sigma * mu;

        // Corrector step: rc_corr = mu_target e - X S e - dx_aff .* ds_aff
        std::vector<double> rhs_system_corr(m, 0.0);
        std::vector<double> dx_corr_inner(n, 0.0);
        for (int j = 0; j < n; ++j) {
            double rc_j = mu_target - x[j] * s[j] - dx_aff[j] * ds_aff[j];
            dx_corr_inner[j] = -rd[j] + rc_j / x[j];
        }
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += sf.A.at(i, j) * x_over_s[j] * dx_corr_inner[j];
            }
            rhs_system_corr[i] = rp[i] - sum;
        }

        std::vector<double> dy = lu.solve(rhs_system_corr);
        std::vector<double> dx(n);
        std::vector<double> ds(n);
        for (int j = 0; j < n; ++j) {
            double ATdy = 0.0;
            for (int i = 0; i < m; ++i) ATdy += sf.A.at(i, j) * dy[i];
            dx[j] = x_over_s[j] * (ATdy + dx_corr_inner[j]);
            double rc_j = mu_target - x[j] * s[j] - dx_aff[j] * ds_aff[j];
            ds[j] = rc_j / x[j] - (s[j] / x[j]) * dx[j];
        }

        // Apply fraction to boundary
        double alpha_p = step_length(x, dx, 0.995);
        double alpha_d = step_length(s, ds, 0.995);

        for (int j = 0; j < n; ++j) {
            x[j] += alpha_p * dx[j];
            s[j] += alpha_d * ds[j];
            // Safety guard strictly for fp drift
            if (x[j] <= 0.0 || std::isnan(x[j])) { result.status = SolveStatus::kNumericalFailure; return result; }
            if (s[j] <= 0.0 || std::isnan(s[j])) { result.status = SolveStatus::kNumericalFailure; return result; }
        }
        for (int i = 0; i < m; ++i) {
            y[i] += alpha_d * dy[i];
            if (std::isnan(y[i])) { result.status = SolveStatus::kNumericalFailure; return result; }
        }
    }

    if (result.status != SolveStatus::kOptimal && result.status != SolveStatus::kNumericalFailure) {
        result.status = SolveStatus::kIterationLimit;
    }

    // Recover original solution
    if (result.status == SolveStatus::kOptimal) {
        result.x.assign(orig_n, 0.0);
        for (int j = 0; j < orig_n; ++j) {
            double val = sf.shift[j];
            if (sf.pos_idx[j] >= 0) {
                val += sf.sign_mul[j] * x[sf.pos_idx[j]];
            }
            if (sf.neg_idx[j] >= 0) {
                val -= x[sf.neg_idx[j]];
            }
            result.x[j] = val;
        }

        double final_obj = 0.0;
        for (int j = 0; j < n; ++j) {
            final_obj += sf.c[j] * x[j];
        }
        result.objective_value = negate_obj ? -(final_obj + sf.obj_offset) : (final_obj + sf.obj_offset);
        result.row_activity = model.A.multiply(result.x);
        result.is_basic.assign(orig_n, false);
    }
    result.iterations = iter;
    return result;
}

}  // namespace pramaan
