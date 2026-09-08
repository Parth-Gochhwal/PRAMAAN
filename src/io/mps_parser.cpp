// mps_parser.cpp
// PILLAR 1 (Structure) -- Zone 1 "Model Ingestion"
//
// Purpose: parse the industry-standard .mps file format into a ModelIR.
// This is your FIRST real milestone after sparse_matrix + ir compile --
// get this reading a tiny hand-written .mps file (3-4 lines, 2 variables)
// before pointing it at a real Netlib instance.
//
// .mps format reference: search "MPS file format specification" -- the
// classic reference is the IBM MPSX manual, but any modern solver's docs
// (HiGHS's own docs page) explain the section structure (ROWS, COLUMNS,
// RHS, BOUNDS) clearly enough to implement from.
//
// FIRST TASK: parse just the ROWS and COLUMNS sections for a pure LP
// (no RANGES, no integer markers) -- add MILP/QP support later.
#include "pramaan/ir.hpp"

namespace pramaan {

// TODO: ModelIR parse_mps(const std::string& filepath);

}  // namespace pramaan
