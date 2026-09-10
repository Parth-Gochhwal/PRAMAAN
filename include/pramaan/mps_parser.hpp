// mps_parser.hpp
// PILLAR 1 (Structure) -- Zone 1 "Model Ingestion"
//
// Declares the MPS file parser that converts industry-standard .mps files
// into PRAMAAN's canonical ModelIR representation.
#pragma once
#include <string>

#include "pramaan/ir.hpp"

namespace pramaan {

// Parses a free-format (whitespace-delimited) MPS file at `filepath` and
// returns a ModelIR. Supports ROWS (N/L/G/E), COLUMNS, RHS, BOUNDS, and
// ENDATA sections for pure continuous LPs.  Throws std::runtime_error on
// I/O or format errors. Spaces within names are not supported.
ModelIR parse_mps(const std::string& filepath);

}  // namespace pramaan

