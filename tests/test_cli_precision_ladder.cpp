#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>

void check(bool condition, const std::string& description) {
    if (!condition) {
        std::cerr << "  [FAIL] " << description << "\n";
        std::exit(1);
    } else {
        std::cout << "  ok: " << description << "\n";
    }
}

int main() {
    std::string bin = "./pramaan-solve";

    // Create a tiny LP MPS file
    std::ofstream mps("test_lp.mps");
    mps << "NAME TESTLP\n"
        << "ROWS\n"
        << " N  OBJ\n"
        << " L  R1\n"
        << "COLUMNS\n"
        << "    X1        OBJ       1.0   R1        1.0\n"
        << "RHS\n"
        << "    RHS1      R1        5.0\n"
        << "BOUNDS\n"
        << " UP BND1      X1        5.0\n"
        << "ENDATA\n";
    mps.close();

    // Test --gpu on LP
    std::string cmd1 = bin + " test_lp.mps \"\" --gpu > out_lp.txt 2> err_lp.txt";
    int ret1 = std::system(cmd1.c_str());
    check(ret1 == 0, "CLI accepts --gpu on LP");

    std::ifstream out("out_lp.txt");
    std::string out_str((std::istreambuf_iterator<char>(out)), std::istreambuf_iterator<char>());

    // In our CI environment, PRAMAAN_ENABLE_CUDA might be OFF.
    // If it's OFF, it will fallback to CPU.
    // If it's ON, it will use GPU and print "Precision Ladder".
    std::ifstream err("err_lp.txt"); std::string err_str((std::istreambuf_iterator<char>(err)), std::istreambuf_iterator<char>());
    bool fallback = (err_str.find("Falling back to CPU") != std::string::npos);
    bool ladder = (out_str.find("Precision Ladder") != std::string::npos);
    check(fallback || ladder, "CLI routed to GPU (or fell back if no CUDA)");

    std::remove("test_lp.mps");
    std::remove("out_lp.txt");
    std::remove("err_lp.txt");

    return 0;
}
