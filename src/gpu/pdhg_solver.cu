#include "pramaan/gpu/pdhg_solver.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace pramaan {
namespace gpu {

__global__ void compute_Ax_bar_kernel(int m, const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_x_bar, float* d_Ax_bar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        int start = d_row_ptr[i];
        int end = d_row_ptr[i+1];
        float sum = 0.0f;
        for (int k = start; k < end; ++k) {
            sum += d_values[k] * d_x_bar[d_col_idx[k]];
        }
        d_Ax_bar[i] = sum;
    }
}

__global__ void update_y_kernel(int m, const float* d_Ax_bar, const float* d_sigma, const float* d_l_r, const float* d_u_r, float* d_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float ax = d_Ax_bar[i];
        float yi = d_y[i];
        float sig = d_sigma[i];
        float w = yi / sig + ax;
        float w_proj = fminf(fmaxf(w, d_l_r[i]), d_u_r[i]);
        d_y[i] = yi + sig * (ax - w_proj);
    }
}

__global__ void compute_ATy_kernel(int m, const int* d_row_ptr, const int* d_col_idx, const float* d_values, const float* d_y, float* d_ATy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float yi = d_y[i];
        int start = d_row_ptr[i];
        int end = d_row_ptr[i+1];
        for (int k = start; k < end; ++k) {
            atomicAdd(&d_ATy[d_col_idx[k]], d_values[k] * yi);
        }
    }
}

__global__ void update_x_kernel(int n, const float* d_ATy, const float* d_tau, const float* d_c, const float* d_l_x, const float* d_u_x, float* d_x, float* d_x_bar) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float xj = d_x[j];
        float tau = d_tau[j];
        float unproj = xj - tau * (d_c[j] + d_ATy[j]);
        float next_x = fminf(fmaxf(unproj, d_l_x[j]), d_u_x[j]);
        d_x_bar[j] = 2.0f * next_x - xj;
        d_x[j] = next_x;
    }
}

PdhgSolver::PdhgSolver(const PdhgOptions& options) : options_(options) {}
PdhgSolver::~PdhgSolver() {}


namespace {
template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};
template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
CudaPtr<T> make_cuda_array(size_t n) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    return CudaPtr<T>(ptr);
}
} // namespace

SolveResult PdhgSolver::solve(const ModelIR& model) {
    const int m = model.numRows();
    const int n = model.numVars();

    SolveResult result;
    result.status = SolveStatus::kIterationLimit;
    result.iterations = 0;

    if (n == 0) {
        result.status = SolveStatus::kOptimal;
        result.objective_value = model.obj_offset;
        return result;
    }

    // Diagonal preconditioning
    std::vector<float> h_sigma(m, 0.0f);
    std::vector<float> h_tau(n, 0.0f);

    std::vector<double> row_norms(m, 0.0);
    std::vector<double> col_norms(n, 0.0);

    for (int i = 0; i < m; ++i) {
        auto rv = model.A.row(i);
        for (auto entry : rv) {
            double val = std::abs(entry.value);
            row_norms[i] += val;
            col_norms[entry.index] += val;
        }
    }

    for (int i = 0; i < m; ++i) {
        h_sigma[i] = 0.99f / std::max(static_cast<float>(row_norms[i]), 1e-8f);
    }
    for (int j = 0; j < n; ++j) {
        h_tau[j] = 0.99f / std::max(static_cast<float>(col_norms[j]), 1e-8f);
    }

    std::vector<float> h_c(n);
    for (int j = 0; j < n; ++j) {
        double sign = (model.obj_sense == ObjSense::kMaximize) ? -1.0 : 1.0;
        h_c[j] = static_cast<float>(sign * model.obj_coeffs[j]);
    }

    std::vector<float> h_l_r(m), h_u_r(m);
    for (int i = 0; i < m; ++i) {
        h_l_r[i] = static_cast<float>(model.row_lower[i]);
        h_u_r[i] = static_cast<float>(model.row_upper[i]);
    }

    std::vector<float> h_l_x(n), h_u_x(n);
    for (int j = 0; j < n; ++j) {
        h_l_x[j] = static_cast<float>(model.var_lower[j]);
        h_u_x[j] = static_cast<float>(model.var_upper[j]);
    }

    std::vector<float> h_values(model.A.values().size());
    for (size_t i = 0; i < h_values.size(); ++i) {
        h_values[i] = static_cast<float>(model.A.values()[i]);
    }

    // Allocate device memory


    auto d_row_ptr_raii = make_cuda_array<int>(model.A.rowPtr().size()); int* d_row_ptr = d_row_ptr_raii.get();
    auto d_col_idx_raii = make_cuda_array<int>(model.A.colIdx().size()); int* d_col_idx = d_col_idx_raii.get();
    auto d_values_raii = make_cuda_array<float>(h_values.size()); float* d_values = d_values_raii.get();

    CUDA_CHECK(cudaMemcpy(d_row_ptr, model.A.rowPtr().data(), model.A.rowPtr().size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, model.A.colIdx().data(), model.A.colIdx().size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), h_values.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto d_c_raii = make_cuda_array<float>(n); float* d_c = d_c_raii.get();
    auto d_l_r_raii = make_cuda_array<float>(m); float* d_l_r = d_l_r_raii.get();
    auto d_u_r_raii = make_cuda_array<float>(m); float* d_u_r = d_u_r_raii.get();
    auto d_l_x_raii = make_cuda_array<float>(n); float* d_l_x = d_l_x_raii.get();
    auto d_u_x_raii = make_cuda_array<float>(n); float* d_u_x = d_u_x_raii.get();
    auto d_sigma_raii = make_cuda_array<float>(m); float* d_sigma = d_sigma_raii.get();
    auto d_tau_raii = make_cuda_array<float>(n); float* d_tau = d_tau_raii.get();

    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    if (m > 0) {
        CUDA_CHECK(cudaMemcpy(d_l_r, h_l_r.data(), m * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_r, h_u_r.data(), m * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_l_x, h_l_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_x, h_u_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, h_tau.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto d_x_raii = make_cuda_array<float>(n); float* d_x = d_x_raii.get();
    auto d_y_raii = make_cuda_array<float>(m); float* d_y = d_y_raii.get();
    auto d_x_bar_raii = make_cuda_array<float>(n); float* d_x_bar = d_x_bar_raii.get();
    auto d_Ax_bar_raii = make_cuda_array<float>(m); float* d_Ax_bar = d_Ax_bar_raii.get();
    auto d_ATy_raii = make_cuda_array<float>(n); float* d_ATy = d_ATy_raii.get();

    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));
    if (m > 0) CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x_bar, 0, n * sizeof(float)));

    int block_size = 256;
    int grid_m = (m + block_size - 1) / block_size;
    int grid_n = (n + block_size - 1) / block_size;
    if (grid_m == 0) grid_m = 1;
    if (grid_n == 0) grid_n = 1;

    std::vector<float> h_x(n);

    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        if (m > 0) {
            compute_Ax_bar_kernel<<<grid_m, block_size>>>(m, d_row_ptr, d_col_idx, d_values, d_x_bar, d_Ax_bar);
            update_y_kernel<<<grid_m, block_size>>>(m, d_Ax_bar, d_sigma, d_l_r, d_u_r, d_y);
            CUDA_CHECK(cudaMemset(d_ATy, 0, n * sizeof(float)));
            compute_ATy_kernel<<<grid_m, block_size>>>(m, d_row_ptr, d_col_idx, d_values, d_y, d_ATy);
        } else {
            CUDA_CHECK(cudaMemset(d_ATy, 0, n * sizeof(float)));
        }
        update_x_kernel<<<grid_n, block_size>>>(n, d_ATy, d_tau, d_c, d_l_x, d_u_x, d_x, d_x_bar);

        if ((iter + 1) % options_.check_frequency == 0) {
            CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

            // Check residuals
            std::vector<double> x_d(n);
            for (int j = 0; j < n; ++j) x_d[j] = h_x[j];

            auto activity = model.A.multiply(x_d);
            double primal_viol = 0.0;
            for (int i = 0; i < m; ++i) {
                double viol = 0.0;
                if (activity[i] < model.row_lower[i]) viol = model.row_lower[i] - activity[i];
                if (activity[i] > model.row_upper[i]) viol = activity[i] - model.row_upper[i];
                primal_viol = std::max(primal_viol, viol);
            }
            for (int j = 0; j < n; ++j) {
                double viol = 0.0;
                if (x_d[j] < model.var_lower[j]) viol = model.var_lower[j] - x_d[j];
                if (x_d[j] > model.var_upper[j]) viol = x_d[j] - model.var_upper[j];
                primal_viol = std::max(primal_viol, viol);
            }

            if (primal_viol <= options_.tolerance) {
                result.status = SolveStatus::kOptimal;
                result.iterations = iter + 1;
                break;
            }
        }
    }

    if (result.status != SolveStatus::kOptimal) {
        result.iterations = options_.max_iterations;
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    result.x.resize(n);
    double dot_prod = 0.0;
    for (int j = 0; j < n; ++j) {
        result.x[j] = static_cast<double>(h_x[j]);
        dot_prod += model.obj_coeffs[j] * result.x[j];
    }
    result.objective_value = model.obj_offset + dot_prod;
    result.row_activity = model.A.multiply(result.x);

















    return result;
}

} // namespace gpu
} // namespace pramaan
