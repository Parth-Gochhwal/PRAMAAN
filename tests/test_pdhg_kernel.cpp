// test_pdhg_kernel.cpp
// P2 Step 1 — CUDA PDHG kernel smoke test
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

namespace pramaan {
namespace gpu {
void launch_pdhg_iteration(float* h_x, float* h_y, const float* h_gradient,
                           const float* h_Ax_bar, const float* h_b, int n, int m, float tau,
                           float sigma);
}  // namespace gpu
}  // namespace pramaan

static int g_checks_run = 0;
static int g_checks_failed = 0;

static void check(bool condition, const std::string& description) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << description << "\n";
    }
}

static void checkVectorsNear(const std::vector<float>& a, const std::vector<float>& b, float atol, float rtol, const std::string& desc) {
    check(a.size() == b.size(), desc + " (sizes match)");
    if (a.size() != b.size()) return;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) {
            continue;
        }
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            ++g_checks_failed;
            std::cerr << "  [FAIL] " << desc << " at index " << i << "\n"
                      << "    expected: " << a[i] << "\n"
                      << "    actual:   " << b[i] << "\n"
                      << "    Reason: Non-finite value without exact equality.\n";
            return;
        }
        float diff = std::abs(a[i] - b[i]);
        float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
        if (diff > atol + rtol * max_val) {
            ++g_checks_failed;
            std::cerr << "  [FAIL] " << desc << " at index " << i << "\n"
                      << "    expected: " << a[i] << "\n"
                      << "    actual:   " << b[i] << "\n"
                      << "    diff:     " << diff << "\n"
                      << "    tol:      " << (atol + rtol * max_val) << "\n";
            return;
        }
    }
    ++g_checks_run; // Count as 1 pass for the whole vector if all match
}

static void run(const std::string& name, const std::function<void()>& test_body) {
    std::cout << name << "...\n";
    int before = g_checks_failed;
    test_body();
    if (g_checks_failed == before) {
        std::cout << "  ok\n";
    }
}

// ---- A. Kernel mathematical correctness (CPU reference) ----
struct PdhgState {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> x_bar;
};

static PdhgState cpu_pdhg_step(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& gradient,
    const std::vector<float>& Ax_bar,
    const std::vector<float>& b,
    float tau, float sigma) 
{
    PdhgState state;
    state.x = x;
    state.y = y;
    state.x_bar.resize(x.size());

    // Primal update
    for (size_t j = 0; j < x.size(); ++j) {
        float update = x[j] - tau * gradient[j];
        float x_next = std::fmax(0.0f, update);
        state.x_bar[j] = 2.0f * x_next - x[j];
        state.x[j] = x_next;
    }

    // Dual update
    for (size_t i = 0; i < y.size(); ++i) {
        float update = Ax_bar[i] - b[i];
        state.y[i] = y[i] + sigma * update;
    }
    return state;
}

static void runAndCompare(
    std::vector<float> x,
    std::vector<float> y,
    const std::vector<float>& gradient,
    const std::vector<float>& Ax_bar,
    const std::vector<float>& b,
    float tau, float sigma, const std::string& desc)
{
    PdhgState expected = cpu_pdhg_step(x, y, gradient, Ax_bar, b, tau, sigma);
    
    // Note regarding D. x_bar correctness:
    // The current launcher dynamically allocates d_x_bar and writes to it in the primal kernel, 
    // but NEVER reads it back to the host or uses it in the dual kernel (which relies on h_Ax_bar instead).
    // Because it is completely unobservable from the host without modifying production code,
    // we cannot assert on it here. We only verify the observable x and y.
    
    pramaan::gpu::launch_pdhg_iteration(
        x.data(), y.data(), gradient.data(), Ax_bar.data(), b.data(),
        static_cast<int>(x.size()), static_cast<int>(y.size()), tau, sigma);
    
    checkVectorsNear(expected.x, x, 1e-5f, 1e-4f, desc + " (primal)");
    checkVectorsNear(expected.y, y, 1e-5f, 1e-4f, desc + " (dual)");
}

// ---- B. Primal projection edge cases ----
void testPrimalEdgeCases() {
    std::vector<float> x =        { 1.0f,  1.0f,  2.0f,  1.0f, 0.0f, 5.0f,  1.0f,   1.0f };
    std::vector<float> gradient = { 0.5f, -0.5f,  2.0f,  3.0f, 1.0f, 0.0f, 1e5f, -1e5f };
    std::vector<float> y;
    std::vector<float> Ax_bar;
    std::vector<float> b;
    runAndCompare(x, y, gradient, Ax_bar, b, 1.0f, 1.0f, "Primal edge cases");
}

// ---- C. Dual update edge cases ----
void testDualEdgeCases() {
    std::vector<float> x;
    std::vector<float> gradient;
    std::vector<float> y =      { 0.0f,  0.0f, 0.0f, 5.0f, 0.0f, 2.0f,  1e5f,  1e-5f, -2.0f };
    std::vector<float> Ax_bar = { 1.0f, -1.0f, 1.0f, 2.0f, 1.0f, 0.0f, -1e5f,  1e-5f,  0.0f };
    std::vector<float> b =      { 0.0f,  0.0f, 1.0f, 2.0f, 0.0f, 0.0f,  0.0f,  0.0f,   0.0f };
    
    runAndCompare(x, y, gradient, Ax_bar, b, 1.0f, 1.0f, "Dual edge cases (sigma=1.0)");
    runAndCompare(x, y, gradient, Ax_bar, b, 1.0f, 0.0f, "Dual edge cases (sigma=0.0)");
}

// ---- E. Dimension coverage ----
void testDimensionCoverage() {
    std::vector<int> dims = {0, 1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1024, 1025};
    for (int d : dims) {
        std::vector<float> x(d, 1.0f);
        std::vector<float> y(d, 1.0f);
        std::vector<float> gradient(d, 0.5f);
        std::vector<float> Ax_bar(d, 0.5f);
        std::vector<float> b(d, 0.0f);
        runAndCompare(x, y, gradient, Ax_bar, b, 1.0f, 1.0f, "Dimension " + std::to_string(d));
    }
}

// ---- F. Matrix-free launcher behavior ----
void testMatrixFreeBehavior() {
    runAndCompare({1.0f}, {1.0f}, {1.0f}, {1.0f}, {1.0f}, 1.0f, 1.0f, "n>0, m>0");
    runAndCompare({}, {1.0f}, {}, {1.0f}, {1.0f}, 1.0f, 1.0f, "n=0, m>0");
    runAndCompare({1.0f}, {}, {1.0f}, {}, {}, 1.0f, 1.0f, "n>0, m=0");
    runAndCompare({}, {}, {}, {}, {}, 1.0f, 1.0f, "n=0, m=0");
}

// ---- G. Randomized/property-style testing ----
void testRandomized() {
    std::mt19937 gen(42); // Fixed seed for determinism
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    int n = 150;
    int m = 120;
    
    for (int iter = 0; iter < 50; ++iter) {
        std::vector<float> x(n);
        std::vector<float> y(m);
        std::vector<float> gradient(n);
        std::vector<float> Ax_bar(m);
        std::vector<float> b(m);
        
        for (int i = 0; i < n; ++i) {
            x[i] = dist(gen);
            if (i % 7 == 0) x[i] = 0.0f;
            gradient[i] = dist(gen);
        }
        for (int i = 0; i < m; ++i) {
            y[i] = dist(gen);
            Ax_bar[i] = dist(gen);
            b[i] = dist(gen);
            if (i % 7 == 0) b[i] = 0.0f;
        }
        
        float tau = std::abs(dist(gen));
        float sigma = std::abs(dist(gen));
        
        runAndCompare(x, y, gradient, Ax_bar, b, tau, sigma, "Random iter " + std::to_string(iter));
    }
}

// ---- H. Numerical stress ----
void testNumericalStress() {
    std::vector<float> x = { 1e-10f, 1e10f, 1e-38f, 1.0f, 0.0f, -0.0f };
    std::vector<float> gradient = { 1e-10f, -1e10f, -1e-38f, 1.0f, 1e-38f, -1e-38f };
    
    std::vector<float> y = { 1e-10f, 1e10f, 1e-38f, -1e-38f };
    std::vector<float> Ax_bar = { 1e-10f, 1e10f, 1e-38f, -1e-38f };
    std::vector<float> b = { 1e-10f, 1e10f, 1e-38f, -1e-38f };
    
    runAndCompare(x, y, gradient, Ax_bar, b, 1.0f, 1.0f, "Numerical stress");
}

// ---- J. Repeated execution / resource behavior ----
void testRepeatedExecution() {
    std::vector<float> x(1000, 1.0f);
    std::vector<float> y(1000, 1.0f);
    std::vector<float> gradient(1000, 0.1f);
    std::vector<float> Ax_bar(1000, 0.1f);
    std::vector<float> b(1000, 0.0f);
    
    std::vector<float> cpu_x = x;
    std::vector<float> cpu_y = y;
    
    for (int i = 0; i < 50; ++i) {
        pramaan::gpu::launch_pdhg_iteration(
            x.data(), y.data(), gradient.data(), Ax_bar.data(), b.data(),
            1000, 1000, 1.0f, 1.0f);
            
        PdhgState state = cpu_pdhg_step(cpu_x, cpu_y, gradient, Ax_bar, b, 1.0f, 1.0f);
        cpu_x = state.x;
        cpu_y = state.y;
    }
    
    checkVectorsNear(cpu_x, x, 1e-3f, 1e-3f, "Repeated execution (primal)");
    checkVectorsNear(cpu_y, y, 1e-3f, 1e-3f, "Repeated execution (dual)");
}

int main() {
    run("B. Primal edge cases", testPrimalEdgeCases);
    run("C. Dual edge cases", testDualEdgeCases);
    run("E. Dimension coverage", testDimensionCoverage);
    run("F. Matrix-free behavior", testMatrixFreeBehavior);
    run("G. Randomized", testRandomized);
    run("H. Numerical stress", testNumericalStress);
    run("J. Repeated execution", testRepeatedExecution);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}