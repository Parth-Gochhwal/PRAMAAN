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

    // Create a tiny MILP MPS file
    std::ofstream mps("test_milp.mps");
    mps << "NAME TESTMILP\n"
        << "ROWS\n"
        << " N  OBJ\n"
        << " L  R1\n"
        << "COLUMNS\n"
        << "    MARKER    'MARKER'                 'INTORG'\n"
        << "    X1        OBJ       1.0   R1        1.0\n"
        << "    MARKER    'MARKER'                 'INTEND'\n"
        << "RHS\n"
        << "    RHS1      R1        5.0\n"
        << "BOUNDS\n"
        << " UP BND1      X1        5.0\n"
        << "ENDATA\n";
    mps.close();

    // Test --gpu rejects MILP
    std::string cmd1 = bin + " test_milp.mps \"\" --gpu > out1.txt 2> err1.txt";
    int ret1 = std::system(cmd1.c_str());
    check(ret1 != 0, "CLI rejects --gpu on MILP");

    std::ifstream err1("err1.txt");
    std::string err_str((std::istreambuf_iterator<char>(err1)), std::istreambuf_iterator<char>());
    check(err_str.find("does not support MILP") != std::string::npos, "Explicit rejection message");

    // Test --auto uses CPU MILP
    std::string cmd2 = bin + " test_milp.mps \"\" --auto > out2.txt 2> err2.txt";
    int ret2 = std::system(cmd2.c_str());
    check(ret2 == 0, "CLI accepts --auto on MILP");

    std::ifstream out2("out2.txt");
    std::string out2_str((std::istreambuf_iterator<char>(out2)), std::istreambuf_iterator<char>());
    check(out2_str.find("Backend: CPU (MILP Branch-and-Bound)") != std::string::npos, "--auto uses CPU MILP backend");

    std::remove("test_milp.mps");
    std::remove("out1.txt");
    std::remove("err1.txt");
    std::remove("out2.txt");
    std::remove("err2.txt");


    // Test malformed arguments
    std::string cmd3 = bin + " test_milp.mps mps2.mps mps3.mps > /dev/null 2> err3.txt";
    int ret3 = std::system(cmd3.c_str());
    check(ret3 != 0, "CLI rejects too many positional arguments");

    std::string cmd4 = bin + " test_milp.mps --gpu --cpu > /dev/null 2> err4.txt";
    int ret4 = std::system(cmd4.c_str());
    check(ret4 != 0, "CLI rejects multiple backend flags");

    std::string cmd5 = bin + " test_milp.mps --unknown-flag > /dev/null 2> err5.txt";
    int ret5 = std::system(cmd5.c_str());
    check(ret5 != 0, "CLI rejects unknown flag");

    return 0;
}
