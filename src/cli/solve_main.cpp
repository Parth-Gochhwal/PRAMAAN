// solve_main.cpp
// The pramaan-solve entry point. Wires together: mps_parser -> ModelIR ->
// Presolver -> RevisedSimplex -> Certificate -> print result.
//
// FIRST TASK for right now: just get this printing something and building
// successfully so `cmake --build .` gives you a working executable on
// day one. Fill in the real pipeline incrementally as each piece above
// gets implemented -- this file changes constantly throughout development,
// that's expected.
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "PRAMAAN solver CLI - not yet implemented\n";
    if (argc > 1) {
        std::cout << "  (would solve: " << argv[1] << ")\n";
    }
    return 0;
}
