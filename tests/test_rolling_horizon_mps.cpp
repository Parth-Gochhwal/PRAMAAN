#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <cctype>

static int g_checks_run = 0;
static int g_checks_failed = 0;

template <typename Func>
void run(const std::string& name, Func&& f) {
    std::cout << name << "...\n";
    f();
}

void check(bool condition, const std::string& message) {
    ++g_checks_run;
    if (!condition) {
        ++g_checks_failed;
        std::cerr << "  [FAIL] " << message << "\n";
    } else {
        std::cout << "  ok\n";
    }
}

void checkNear(double a, double b, double tol, const std::string& message) {
    check(std::abs(a - b) <= tol, message);
}

// Basic JSON extraction utilities
std::string getJsonStringValue(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos < json.length() && json[pos] == '"') {
        size_t end = json.find("\"", pos + 1);
        if (end == std::string::npos) throw std::runtime_error("Malformed string value for " + key);
        return json.substr(pos + 1, end - pos - 1);
    }
    throw std::runtime_error("Value is not a string for " + key);
}

double getJsonDoubleValue(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    size_t end = pos;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == '+' || json[end] == 'e' || json[end] == 'E')) {
        end++;
    }
    if (end == pos) throw std::runtime_error("Value is not a number for " + key);
    return std::stod(json.substr(pos, end - pos));
}

int getJsonIntValue(const std::string& json, const std::string& key) {
    return static_cast<int>(getJsonDoubleValue(json, key));
}

bool getJsonBoolValue(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (json.substr(pos, 4) == "true") return true;
    if (json.substr(pos, 5) == "false") return false;
    throw std::runtime_error("Value is not a boolean for " + key);
}

uint64_t getJsonUint64Value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    size_t end = pos;
    while (end < json.length() && isdigit(json[end])) {
        end++;
    }
    if (end == pos) throw std::runtime_error("Value is not a number for " + key);
    return std::stoull(json.substr(pos, end - pos));
}

std::vector<std::string> getJsonArrayObjects(const std::string& json, const std::string& array_key) {
    std::vector<std::string> result;
    std::string search = "\"" + array_key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) throw std::runtime_error("Array key not found: " + array_key);
    pos = json.find("[", pos);
    if (pos == std::string::npos) throw std::runtime_error("Array start not found for " + array_key);
    
    size_t brace_level = 0;
    size_t start = std::string::npos;
    
    for (size_t i = pos; i < json.length(); i++) {
        if (json[i] == '{') {
            if (brace_level == 0) start = i;
            brace_level++;
        } else if (json[i] == '}') {
            brace_level--;
            if (brace_level == 0 && start != std::string::npos) {
                result.push_back(json.substr(start, i - start + 1));
                start = std::string::npos;
            }
        } else if (json[i] == ']' && brace_level == 0) {
            break;
        }
    }
    return result;
}

std::string runBenchmarkAndGetJson(const std::string& extra_args, const std::string& outfile) {
#ifndef BENCHMARK_EXECUTABLE
#define BENCHMARK_EXECUTABLE "./build/rolling_horizon_mps_bench"
#endif
    std::string bin = BENCHMARK_EXECUTABLE;
    std::string cmd = bin + " benchmark/mrpl_crude_blend.mps " + extra_args + " > " + outfile + " 2>/dev/null";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("Benchmark execution failed");
    }
    
    std::ifstream ifs(outfile);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open JSON output file");
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    if (content.empty() || content[0] != '{') {
        throw std::runtime_error("Invalid JSON format");
    }
    return content;
}

void testRegressionSuite() {
    std::string outfile = "test_regression_suite.json";
    std::string content;
    
    try {
        content = runBenchmarkAndGetJson("--days 3 --seed 42", outfile);
    } catch (const std::exception& e) {
        check(false, std::string("Benchmark failed to run: ") + e.what());
        std::remove(outfile.c_str());
        return;
    }

    try {
        // A. Top-level fields
        check(getJsonBoolValue(content, "benchmark_valid") == true, "benchmark_valid is true");
        check(getJsonBoolValue(content, "correctness") == true, "correctness is true");
        check(getJsonIntValue(content, "days") == 3, "days == 3");
        check(getJsonIntValue(content, "cold_optimal_days") == 3, "cold_optimal_days == 3");
        check(getJsonIntValue(content, "warm_optimal_days") == 3, "warm_optimal_days == 3");
        check(getJsonIntValue(content, "warm_start_attempts") == 2, "warm_start_attempts == 2");
        
        int successes = getJsonIntValue(content, "warm_start_successes");
        int fallbacks = getJsonIntValue(content, "warm_start_fallbacks");
        check(successes + fallbacks == 2, "successes + fallbacks == 2");
        
        double cold_cumulative_ms = getJsonDoubleValue(content, "cold_cumulative_ms");
        double warm_cumulative_ms = getJsonDoubleValue(content, "warm_cumulative_ms");
        check(cold_cumulative_ms > 0.0, "cold_cumulative_ms > 0");
        check(warm_cumulative_ms > 0.0, "warm_cumulative_ms > 0");
        
        double ratio = getJsonDoubleValue(content, "cumulative_ratio");
        check(!std::isnan(ratio) && !std::isinf(ratio) && ratio > 0, "cold/warm ratio is finite and positive");

        // B. Per-day records
        std::vector<std::string> days = getJsonArrayObjects(content, "per_day");
        check(days.size() == 3, "Exactly 3 day records exist");
        
        double sum_cold_ms = 0;
        double sum_warm_ms = 0;
        int sum_cold_iters = 0;
        int sum_warm_iters = 0;
        
        uint64_t base_structure = 0;

        for (int i = 0; i < 3; ++i) {
            std::string day_json = days[i];
            
            // Check day index
            check(getJsonIntValue(day_json, "day") == i, "Day index matches " + std::to_string(i));
            
            // Statuses optimal
            check(getJsonIntValue(day_json, "cold_status") == 0, "Day " + std::to_string(i) + " cold status is optimal");
            check(getJsonIntValue(day_json, "warm_status") == 0, "Day " + std::to_string(i) + " warm status is optimal");
            
            double cold_obj = getJsonDoubleValue(day_json, "cold_objective");
            double warm_obj = getJsonDoubleValue(day_json, "warm_objective");
            check(!std::isnan(cold_obj) && !std::isinf(cold_obj), "Day " + std::to_string(i) + " cold_objective is finite");
            check(!std::isnan(warm_obj) && !std::isinf(warm_obj), "Day " + std::to_string(i) + " warm_objective is finite");
            
            double obj_diff = getJsonDoubleValue(day_json, "objective_abs_diff");
            check(!std::isnan(obj_diff) && !std::isinf(obj_diff), "Day " + std::to_string(i) + " objective_abs_diff is finite");
            check(obj_diff <= 1e-8, "Day " + std::to_string(i) + " objective_abs_diff <= tolerance");
            
            // D. Objective consistency
            checkNear(std::abs(cold_obj - warm_obj), obj_diff, 1e-12, "Day " + std::to_string(i) + " abs(cold-warm) agrees with reported diff");
            
            double cold_res = getJsonDoubleValue(day_json, "cold_primal_residual");
            double warm_res = getJsonDoubleValue(day_json, "warm_primal_residual");
            check(!std::isnan(cold_res) && !std::isinf(cold_res) && cold_res >= 0.0, "Day " + std::to_string(i) + " cold primal residual is finite and >= 0");
            check(!std::isnan(warm_res) && !std::isinf(warm_res) && warm_res >= 0.0, "Day " + std::to_string(i) + " warm primal residual is finite and >= 0");
            check(cold_res <= 1e-8 && warm_res <= 1e-8, "Day " + std::to_string(i) + " residuals satisfy tolerance");
            
            int c_iters = getJsonIntValue(day_json, "cold_iterations");
            int w_iters = getJsonIntValue(day_json, "warm_iterations");
            check(c_iters >= 0, "Day " + std::to_string(i) + " cold iterations >= 0");
            check(w_iters >= 0, "Day " + std::to_string(i) + " warm iterations >= 0");
            
            sum_cold_iters += c_iters;
            sum_warm_iters += w_iters;
            
            sum_cold_ms += getJsonDoubleValue(day_json, "cold_time_ms");
            sum_warm_ms += getJsonDoubleValue(day_json, "warm_time_ms");
            
            // Structure invariants
            uint64_t current_struct = getJsonUint64Value(day_json, "structure_fingerprint");
            if (i == 0) {
                base_structure = current_struct;
            } else {
                check(current_struct == base_structure, "Day " + std::to_string(i) + " structure fingerprint unchanged");
                
                // C. Warm-start accounting
                bool reused = getJsonBoolValue(day_json, "basis_reused");
                bool fell_back = getJsonBoolValue(day_json, "warm_fell_back");
                check(reused || fell_back, "Day " + std::to_string(i) + " either reused basis or fell back");
            }
        }
        
        // E. Cumulative timing consistency
        checkNear(sum_cold_ms, cold_cumulative_ms, 1e-3, "Sum of per-day cold_time_ms agrees with cold_cumulative_ms");
        checkNear(sum_warm_ms, warm_cumulative_ms, 1e-3, "Sum of per-day warm_time_ms agrees with warm_cumulative_ms");
        checkNear(ratio, cold_cumulative_ms / warm_cumulative_ms, 1e-3, "Reported ratio agrees with cumulative times");
        
        // F. Iteration consistency
        check(sum_cold_iters == getJsonIntValue(content, "cold_total_iters"), "Cold iterations sum matches reported total");
        check(sum_warm_iters == getJsonIntValue(content, "warm_total_iters"), "Warm iterations sum matches reported total");
        
    } catch (const std::exception& e) {
        check(false, std::string("Validation failed: ") + e.what());
    }
    
    std::remove(outfile.c_str());
}

void testDeterminism() {
    std::string outfile1 = "test_determinism_1.json";
    std::string outfile2 = "test_determinism_2.json";
    std::string outfile3 = "test_determinism_3.json";
    
    try {
        std::string json1 = runBenchmarkAndGetJson("--days 3 --seed 42", outfile1);
        std::string json2 = runBenchmarkAndGetJson("--days 3 --seed 42", outfile2);
        std::string json3 = runBenchmarkAndGetJson("--days 3 --seed 99", outfile3);
        
        // Check identical sequence for same seed
        std::vector<std::string> days1 = getJsonArrayObjects(json1, "per_day");
        std::vector<std::string> days2 = getJsonArrayObjects(json2, "per_day");
        check(days1.size() == 3 && days2.size() == 3, "Same seed runs generated 3 days");
        
        bool identical = true;
        for (int i = 0; i < 3; ++i) {
            std::string d1 = getJsonStringValue(days1[i], "perturbation");
            std::string d2 = getJsonStringValue(days2[i], "perturbation");
            if (d1 != d2) identical = false;
            
            double obj1 = getJsonDoubleValue(days1[i], "cold_objective");
            double obj2 = getJsonDoubleValue(days2[i], "cold_objective");
            if (std::abs(obj1 - obj2) > 1e-9) identical = false;
        }
        check(identical, "Determinism: Same seed yields same scenario sequence and objectives");
        
        // Check different seed yields different sequence
        std::vector<std::string> days3 = getJsonArrayObjects(json3, "per_day");
        bool different = false;
        for (int i = 0; i < 3; ++i) {
            double obj1 = getJsonDoubleValue(days1[i], "cold_objective");
            double obj3 = getJsonDoubleValue(days3[i], "cold_objective");
            if (std::abs(obj1 - obj3) > 1e-6) different = true;
        }
        check(different, "Determinism: Different seed yields different objective sequence");
        
    } catch (const std::exception& e) {
        check(false, std::string("Determinism validation failed: ") + e.what());
    }
    
    std::remove(outfile1.c_str());
    std::remove(outfile2.c_str());
    std::remove(outfile3.c_str());
}

void testCLIValidation() {
#ifndef BENCHMARK_EXECUTABLE
#define BENCHMARK_EXECUTABLE "./build/rolling_horizon_mps_bench"
#endif
    std::string bin = BENCHMARK_EXECUTABLE;

    std::string cmd_help = bin + " --help > /dev/null 2>&1";
    check(std::system(cmd_help.c_str()) == 0, "CLI validation: --help succeeds");

    std::string cmd_days = bin + " benchmark/mrpl_crude_blend.mps --days 0 > /dev/null 2>&1";
    check(std::system(cmd_days.c_str()) != 0, "CLI validation: --days 0 fails");

    std::string cmd_missing = bin + " benchmark/nonexistent_model.mps --days 3 > /dev/null 2>&1";
    check(std::system(cmd_missing.c_str()) != 0, "CLI validation: nonexistent MPS fails");

    std::string cmd_unknown = bin + " benchmark/mrpl_crude_blend.mps --unknown-opt > /dev/null 2>&1";
    check(std::system(cmd_unknown.c_str()) != 0, "CLI validation: unknown option fails");
}

int main() {
    run("Regression Suite", testRegressionSuite);
    run("Determinism Validation", testDeterminism);
    run("CLI strict validation", testCLIValidation);

    std::cout << "\n" << g_checks_run << " checks run, " << g_checks_failed << " failed.\n";
    if (g_checks_failed > 0) {
        std::cout << "TEST SUITE FAILED\n";
        return 1;
    }
    std::cout << "TEST SUITE PASSED\n";
    return 0;
}
