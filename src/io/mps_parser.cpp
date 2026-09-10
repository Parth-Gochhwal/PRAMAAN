// mps_parser.cpp
// PILLAR 1 (Structure) -- Zone 1 "Model Ingestion"
//
// Parses free-format MPS files into a ModelIR.
// Supports ROWS (N/L/G/E), COLUMNS, RHS, BOUNDS (LO/UP/FX/FR/MI/PL),
// and ENDATA sections for pure continuous LPs.
//
// Note: This parser tokenizes strictly on whitespace. It does not claim
// to support strict fixed-format MPS features (e.g., spaces within names
// or fields packed together without delimiter spaces).
#include "pramaan/mps_parser.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pramaan {

namespace {

// Strip trailing \r so the parser works on both LF and CRLF line endings.
std::string stripCR(const std::string& line) {
    std::string s = line;
    if (!s.empty() && s.back() == '\r') s.pop_back();
    return s;
}

// Split a line into whitespace-delimited tokens.
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}

// True if the line starts a new MPS section header (NAME, ROWS, COLUMNS,
// RHS, RANGES, BOUNDS, ENDATA, or any future section).  Section headers
// begin in column 1 (i.e. no leading whitespace).
bool isSectionHeader(const std::string& line) {
    if (line.empty()) return false;
    // A data line always starts with at least one space or tab.
    return line[0] != ' ' && line[0] != '\t';
}

double parseDouble(const std::string& s, int lineno) {
    try {
        std::size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) {
            throw std::runtime_error(
                "MPS parse error (line " + std::to_string(lineno) +
                "): invalid number '" + s + "'");
        }
        return v;
    } catch (const std::invalid_argument&) {
        throw std::runtime_error(
            "MPS parse error (line " + std::to_string(lineno) +
            "): invalid number '" + s + "'");
    } catch (const std::out_of_range&) {
        throw std::runtime_error(
            "MPS parse error (line " + std::to_string(lineno) +
            "): number out of range '" + s + "'");
    }
}

}  // namespace

ModelIR parse_mps(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("MPS parse error: cannot open file '" + filepath + "'");
    }

    // --- Row bookkeeping: name -> (sense, index-into-constraint-arrays).
    // The objective row (sense 'N') gets index -1 and is tracked separately.
    enum class RowSense { kN, kL, kG, kE };
    struct RowInfo {
        RowSense sense;
        int index;  // -1 for objective
    };
    std::unordered_map<std::string, RowInfo> row_map;
    std::vector<std::string> constraint_names;  // ordered constraint rows
    std::vector<RowSense> constraint_senses;
    std::string obj_row_name;
    bool have_obj = false;

    // --- Variable bookkeeping: name -> column index (insertion order).
    std::unordered_map<std::string, int> var_map;
    std::vector<std::string> var_names_ordered;

    auto getOrCreateVar = [&](const std::string& name) -> int {
        auto it = var_map.find(name);
        if (it != var_map.end()) return it->second;
        int idx = static_cast<int>(var_names_ordered.size());
        var_map[name] = idx;
        var_names_ordered.push_back(name);
        return idx;
    };

    // Accumulate matrix entries as (row_constraint_idx, col_var_idx, value).
    struct Triplet { int row; int col; double val; };
    std::vector<Triplet> triplets;
    std::vector<double> obj_coeffs_accum;  // sized lazily
    double obj_offset = 0.0;

    // RHS values per constraint (index into constraint_names).
    std::vector<double> rhs_values;  // sized lazily

    // Bounds: default is 0 <= x < +inf for each variable.
    struct BoundInfo { double lo; double hi; bool lo_set; bool hi_set; };
    std::unordered_map<int, BoundInfo> bounds_override;

    // --- Parse the file line by line, dispatching on the current section.
    enum class Section { kNone, kName, kRows, kColumns, kRhs, kRanges, kBounds };
    Section section = Section::kNone;

    std::string line;
    int lineno = 0;
    bool saw_endata = false;

    while (std::getline(in, line)) {
        ++lineno;
        line = stripCR(line);

        // Skip blank lines and comment lines (starting with '*' or '$').
        if (line.empty()) continue;
        if (line[0] == '*' || line[0] == '$') continue;

        if (isSectionHeader(line)) {
            auto tokens = tokenize(line);
            if (tokens.empty()) continue;
            const std::string& hdr = tokens[0];
            if (hdr == "NAME")         { section = Section::kName; continue; }
            if (hdr == "ROWS")         { section = Section::kRows; continue; }
            if (hdr == "COLUMNS")      { section = Section::kColumns; continue; }
            if (hdr == "RHS")          { section = Section::kRhs; continue; }
            if (hdr == "RANGES")       { section = Section::kRanges; continue; }
            if (hdr == "BOUNDS")       { section = Section::kBounds; continue; }
            if (hdr == "ENDATA")       { saw_endata = true; break; }
            // Unknown section: skip until next known header.
            section = Section::kNone;
            continue;
        }

        auto tokens = tokenize(line);
        if (tokens.empty()) continue;

        switch (section) {
        case Section::kRows: {
            if (tokens.size() < 2) {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): ROWS line needs at least sense + name");
            }
            const std::string& sense_str = tokens[0];
            const std::string& rname = tokens[1];
            RowSense sense;
            if      (sense_str == "N") sense = RowSense::kN;
            else if (sense_str == "L") sense = RowSense::kL;
            else if (sense_str == "G") sense = RowSense::kG;
            else if (sense_str == "E") sense = RowSense::kE;
            else {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): unknown row sense '" + sense_str + "'");
            }
            if (row_map.count(rname)) {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): duplicate row name '" + rname + "'");
            }
            if (sense == RowSense::kN) {
                // Use the first N row as the objective; additional N rows are
                // treated as free rows (standard MPS convention).
                if (!have_obj) {
                    obj_row_name = rname;
                    have_obj = true;
                    row_map[rname] = RowInfo{sense, -1};
                } else {
                    // Extra N row: store as a free constraint row.
                    int cidx = static_cast<int>(constraint_names.size());
                    constraint_names.push_back(rname);
                    constraint_senses.push_back(RowSense::kN);
                    row_map[rname] = RowInfo{sense, cidx};
                }
            } else {
                int cidx = static_cast<int>(constraint_names.size());
                constraint_names.push_back(rname);
                constraint_senses.push_back(sense);
                row_map[rname] = RowInfo{sense, cidx};
            }
            break;
        }

        case Section::kColumns: {
            // Format: var_name  row_name  value  [row_name  value]
            if (tokens.size() < 3 || tokens.size() == 4) {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): COLUMNS line has unexpected token count (" +
                    std::to_string(tokens.size()) + ")");
            }
            const std::string& vname = tokens[0];
            int vidx = getOrCreateVar(vname);

            // Process one or two (row_name, value) pairs on this line.
            for (std::size_t p = 1; p + 1 < tokens.size(); p += 2) {
                const std::string& rname = tokens[p];
                double val = parseDouble(tokens[p + 1], lineno);

                auto it = row_map.find(rname);
                if (it == row_map.end()) {
                    throw std::runtime_error(
                        "MPS parse error (line " + std::to_string(lineno) +
                        "): COLUMNS references unknown row '" + rname + "'");
                }
                if (it->second.index == -1) {
                    // Objective row.
                    if (static_cast<int>(obj_coeffs_accum.size()) <= vidx) {
                        obj_coeffs_accum.resize(static_cast<std::size_t>(vidx) + 1, 0.0);
                    }
                    obj_coeffs_accum[static_cast<std::size_t>(vidx)] += val;
                } else {
                    triplets.push_back(Triplet{it->second.index, vidx, val});
                }
            }
            break;
        }

        case Section::kRhs: {
            // Format: rhs_name  row_name  value  [row_name  value]
            if (tokens.size() < 3) {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): RHS line needs at least 3 tokens");
            }
            // tokens[0] is the RHS vector name (ignored -- we only support
            // one RHS vector, the standard case for LP).
            for (std::size_t p = 1; p + 1 < tokens.size(); p += 2) {
                const std::string& rname = tokens[p];
                double val = parseDouble(tokens[p + 1], lineno);
                auto it = row_map.find(rname);
                if (it == row_map.end()) {
                    throw std::runtime_error(
                        "MPS parse error (line " + std::to_string(lineno) +
                        "): RHS references unknown row '" + rname + "'");
                }
                int cidx = it->second.index;
                if (cidx < 0) {
                    // Standard MPS convention: an RHS value on the objective row
                    // is subtracted from the objective function.
                    obj_offset -= val;
                    continue;
                }
                if (rhs_values.empty()) {
                    rhs_values.assign(constraint_names.size(), 0.0);
                }
                rhs_values[static_cast<std::size_t>(cidx)] = val;
            }
            break;
        }

        case Section::kBounds: {
            // Format: bound_type  bound_name  var_name  [value]
            if (tokens.size() < 3) {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): BOUNDS line needs at least 3 tokens");
            }
            const std::string& btype = tokens[0];
            // tokens[1] is the bound vector name (ignored).
            const std::string& vname = tokens[2];
            int vidx = getOrCreateVar(vname);

            auto& bi = bounds_override[vidx];

            if (btype == "LO") {
                if (tokens.size() < 4) {
                    throw std::runtime_error(
                        "MPS parse error (line " + std::to_string(lineno) +
                        "): LO bound needs a value");
                }
                bi.lo = parseDouble(tokens[3], lineno);
                bi.lo_set = true;
            } else if (btype == "UP") {
                if (tokens.size() < 4) {
                    throw std::runtime_error(
                        "MPS parse error (line " + std::to_string(lineno) +
                        "): UP bound needs a value");
                }
                bi.hi = parseDouble(tokens[3], lineno);
                bi.hi_set = true;
            } else if (btype == "FX") {
                if (tokens.size() < 4) {
                    throw std::runtime_error(
                        "MPS parse error (line " + std::to_string(lineno) +
                        "): FX bound needs a value");
                }
                double v = parseDouble(tokens[3], lineno);
                bi.lo = v;
                bi.hi = v;
                bi.lo_set = true;
                bi.hi_set = true;
            } else if (btype == "FR") {
                bi.lo = -kInfinity;
                bi.hi = kInfinity;
                bi.lo_set = true;
                bi.hi_set = true;
            } else if (btype == "MI") {
                bi.lo = -kInfinity;
                bi.lo_set = true;
            } else if (btype == "PL") {
                bi.hi = kInfinity;
                bi.hi_set = true;
            } else if (btype == "BV") {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): BV bounds not supported in pure LP milestone");
            } else {
                throw std::runtime_error(
                    "MPS parse error (line " + std::to_string(lineno) +
                    "): unknown bound type '" + btype + "'");
            }
            break;
        }

        case Section::kRanges:
            throw std::runtime_error(
                "MPS parse error (line " + std::to_string(lineno) +
                "): RANGES not supported");

        default:
            // Inside NAME or an unrecognized section: skip data lines.
            break;
        }
    }

    if (!saw_endata) {
        throw std::runtime_error("MPS parse error: file ended without ENDATA");
    }

    // --- Assemble the ModelIR ---

    const int num_vars = static_cast<int>(var_names_ordered.size());
    const int num_rows = static_cast<int>(constraint_names.size());

    if (num_vars == 0) {
        throw std::runtime_error("MPS parse error: no variables found");
    }

    // Objective coefficients: extend to num_vars (may be shorter if the last
    // variables had no objective coefficient).
    obj_coeffs_accum.resize(static_cast<std::size_t>(num_vars), 0.0);

    // RHS: extend to num_rows if needed.
    rhs_values.resize(static_cast<std::size_t>(num_rows), 0.0);

    // Build the constraint matrix in CSR format.  triplets are already in
    // (row, col, val) form; sort by row then column for CSR construction.
    std::sort(triplets.begin(), triplets.end(),
              [](const Triplet& a, const Triplet& b) {
                  return a.row < b.row || (a.row == b.row && a.col < b.col);
              });

    // Merge duplicate (row, col) entries: when the same variable appears in
    // the same row on multiple COLUMNS lines, the values must be summed.
    // After sorting, duplicates are adjacent, so a single pass suffices.
    {
        std::vector<Triplet> merged;
        merged.reserve(triplets.size());
        for (const auto& t : triplets) {
            if (!merged.empty() && merged.back().row == t.row &&
                merged.back().col == t.col) {
                merged.back().val += t.val;
            } else {
                merged.push_back(t);
            }
        }
        triplets = std::move(merged);
    }

    std::vector<CSRMatrix::Index> row_ptr(static_cast<std::size_t>(num_rows) + 1, 0);
    std::vector<CSRMatrix::Index> col_idx;
    std::vector<double> values;
    col_idx.reserve(triplets.size());
    values.reserve(triplets.size());

    for (const auto& t : triplets) {
        row_ptr[static_cast<std::size_t>(t.row) + 1]++;
    }
    for (int r = 0; r < num_rows; ++r) {
        row_ptr[static_cast<std::size_t>(r) + 1] += row_ptr[static_cast<std::size_t>(r)];
    }

    // Triplets are already sorted by (row, col), so we can fill directly.
    col_idx.resize(triplets.size());
    values.resize(triplets.size());
    std::vector<CSRMatrix::Index> cursor(row_ptr.begin(), row_ptr.end());
    for (const auto& t : triplets) {
        auto pos = static_cast<std::size_t>(cursor[static_cast<std::size_t>(t.row)]++);
        col_idx[pos] = static_cast<CSRMatrix::Index>(t.col);
        values[pos] = t.val;
    }

    CSRMatrix A(std::move(row_ptr), std::move(col_idx), std::move(values),
                static_cast<CSRMatrix::Index>(num_vars));

    // Row bounds: derived from row sense and RHS.
    std::vector<double> row_lower(static_cast<std::size_t>(num_rows));
    std::vector<double> row_upper(static_cast<std::size_t>(num_rows));

    for (int r = 0; r < num_rows; ++r) {
        double rhs = rhs_values[static_cast<std::size_t>(r)];
        switch (constraint_senses[static_cast<std::size_t>(r)]) {
            case RowSense::kL:
                row_lower[static_cast<std::size_t>(r)] = -kInfinity;
                row_upper[static_cast<std::size_t>(r)] = rhs;
                break;
            case RowSense::kG:
                row_lower[static_cast<std::size_t>(r)] = rhs;
                row_upper[static_cast<std::size_t>(r)] = kInfinity;
                break;
            case RowSense::kE:
                row_lower[static_cast<std::size_t>(r)] = rhs;
                row_upper[static_cast<std::size_t>(r)] = rhs;
                break;
            case RowSense::kN:
                // Free row (extra N rows beyond the first objective).
                row_lower[static_cast<std::size_t>(r)] = -kInfinity;
                row_upper[static_cast<std::size_t>(r)] = kInfinity;
                break;
        }
    }

    // Variable bounds: default is [0, +inf) unless overridden.
    std::vector<double> var_lower(static_cast<std::size_t>(num_vars), 0.0);
    std::vector<double> var_upper(static_cast<std::size_t>(num_vars), kInfinity);

    for (const auto& [vidx, bi] : bounds_override) {
        if (bi.lo_set) var_lower[static_cast<std::size_t>(vidx)] = bi.lo;
        if (bi.hi_set) var_upper[static_cast<std::size_t>(vidx)] = bi.hi;
    }

    // Variable types: all continuous (integer support deferred).
    std::vector<VarType> var_types(static_cast<std::size_t>(num_vars),
                                   VarType::kContinuous);

    return ModelIR(
        ObjSense::kMinimize,
        obj_offset,
        std::move(obj_coeffs_accum),
        std::move(A),
        std::move(row_lower),
        std::move(row_upper),
        std::vector<std::string>(constraint_names.begin(), constraint_names.end()),
        std::move(var_lower),
        std::move(var_upper),
        std::move(var_types),
        std::vector<std::string>(var_names_ordered.begin(), var_names_ordered.end()));
}

}  // namespace pramaan
