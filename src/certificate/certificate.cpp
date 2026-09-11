// certificate.cpp -- see include/pramaan/certificate.hpp
//
// Stage 7: Certificate generation + serialization + parsing.
// This file is part of pramaan_core and is used by both pramaan-solve
// (generation) and pramaan-verify (parsing). It contains NO solver logic.
#include "pramaan/certificate.hpp"
#include "pramaan/ir.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace pramaan {

// ---------------------------------------------------------------------------
// generate_certificate
// ---------------------------------------------------------------------------
// Independently recomputes all residuals from the original model and x.

Certificate generate_certificate(const ModelIR& model,
                                 const std::vector<double>& x,
                                 const std::string& status,
                                 int ledger_entries) {
    Certificate cert;
    cert.status = status;
    cert.num_rows = model.numRows();
    cert.num_cols = model.numVars();
    cert.ledger_entries = ledger_entries;
    cert.x = x;

    const int m = model.numRows();
    const int n = model.numVars();

    // -- Recompute objective from original model --
    double obj = model.obj_offset;
    for (int j = 0; j < n; ++j) {
        obj += model.obj_coeffs[static_cast<std::size_t>(j)]
             * x[static_cast<std::size_t>(j)];
    }
    cert.objective_value = obj;

    // -- Compute row activities and primal residual --
    std::vector<double> activity = model.A.multiply(x);
    double max_row_viol = 0.0;
    for (int r = 0; r < m; ++r) {
        double ar = activity[static_cast<std::size_t>(r)];
        double lo = model.row_lower[static_cast<std::size_t>(r)];
        double hi = model.row_upper[static_cast<std::size_t>(r)];

        // Violation below lower bound (for >= and = rows).
        if (lo > -kInfinity) {
            double viol = lo - ar;
            if (viol > max_row_viol) max_row_viol = viol;
        }
        // Violation above upper bound (for <= and = rows).
        if (hi < kInfinity) {
            double viol = ar - hi;
            if (viol > max_row_viol) max_row_viol = viol;
        }
    }
    cert.primal_residual = max_row_viol;

    // -- Compute variable bound violations --
    double max_bound_viol = 0.0;
    for (int j = 0; j < n; ++j) {
        double xj = x[static_cast<std::size_t>(j)];
        double lo = model.var_lower[static_cast<std::size_t>(j)];
        double hi = model.var_upper[static_cast<std::size_t>(j)];

        if (lo > -kInfinity) {
            double viol = lo - xj;
            if (viol > max_bound_viol) max_bound_viol = viol;
        }
        if (hi < kInfinity) {
            double viol = xj - hi;
            if (viol > max_bound_viol) max_bound_viol = viol;
        }
    }
    cert.bound_violation = max_bound_viol;

    // -- Integrality violation (0.0 for pure LP) --
    double max_int_viol = 0.0;
    for (int j = 0; j < n; ++j) {
        if (model.var_types[static_cast<std::size_t>(j)] == VarType::kInteger) {
            double xj = x[static_cast<std::size_t>(j)];
            double rounded = std::round(xj);
            double viol = std::abs(xj - rounded);
            if (viol > max_int_viol) max_int_viol = viol;
        }
    }
    cert.integrality_violation = max_int_viol;

    // -- Dual and Complementarity residuals --
    // The current primal simplex does not compute independent dual variables.
    cert.dual_residual = std::nullopt;
    cert.complementarity_residual = std::nullopt;

    // -- Objective bound gap --
    // We cannot independently verify optimality gap without duals.
    cert.objective_bound_gap = std::nullopt;

    return cert;
}

// ---------------------------------------------------------------------------
// write_certificate
// ---------------------------------------------------------------------------

void write_certificate(const Certificate& cert, const std::string& filepath) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open certificate file for writing: " + filepath);
    }

    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    out << cert.format_id << "\n";
    out << "STATUS " << cert.status << "\n";
    out << "ROWS " << cert.num_rows << "\n";
    out << "COLS " << cert.num_cols << "\n";
    out << "OBJECTIVE " << cert.objective_value << "\n";
    out << "TOLERANCE " << cert.tolerance << "\n";
    out << "PRIMAL_RESIDUAL " << cert.primal_residual << "\n";

    auto write_opt = [&out](const std::string& key, const std::optional<double>& val) {
        out << key << " ";
        if (val.has_value()) out << val.value() << "\n";
        else out << "UNAVAILABLE\n";
    };

    write_opt("DUAL_RESIDUAL", cert.dual_residual);
    write_opt("COMPLEMENTARITY_RESIDUAL", cert.complementarity_residual);

    out << "BOUND_VIOLATION " << cert.bound_violation << "\n";
    out << "INTEGRALITY_VIOLATION " << cert.integrality_violation << "\n";
    write_opt("OBJECTIVE_BOUND_GAP", cert.objective_bound_gap);
    out << "LEDGER_ENTRIES " << cert.ledger_entries << "\n";
    out << "X\n";
    for (const double v : cert.x) {
        out << v << "\n";
    }
    out << "END_X\n";
    out << "END_CERTIFICATE\n";

    if (!out.good()) {
        throw std::runtime_error("Error writing certificate file: " + filepath);
    }
}

// ---------------------------------------------------------------------------
// read_certificate
// ---------------------------------------------------------------------------

Certificate read_certificate(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open certificate file: " + filepath);
    }

    Certificate cert;
    std::string line;

    // Line 1: format identifier.
    if (!std::getline(in, line)) {
        throw std::runtime_error("Certificate file is empty");
    }
    // Strip trailing \r if present (CRLF).
    if (!line.empty() && line.back() == '\r') line.pop_back();
    cert.format_id = line;
    if (cert.format_id != "PRAMAAN_CERTIFICATE_V1") {
        throw std::runtime_error("Unknown certificate format: " + cert.format_id);
    }

    auto read_line = [&]() -> std::string {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected end of certificate file");
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return line;
    };

    auto parse_key_str = [](const std::string& l, const std::string& key) -> std::string {
        if (l.substr(0, key.size()) != key) {
            throw std::runtime_error("Expected '" + key + "', got: " + l);
        }
        return l.substr(key.size());
    };

    auto parse_int = [](const std::string& val, const std::string& key) -> int {
        if (val.empty()) throw std::runtime_error("Empty value for " + key);
        std::size_t pos = 0;
        int i = std::stoi(val, &pos);
        while (pos < val.size() && std::isspace(val[pos])) pos++;
        if (pos != val.size()) throw std::runtime_error("Trailing garbage in " + key + ": " + val);
        return i;
    };

    auto parse_double = [](const std::string& val, const std::string& key) -> double {
        if (val.empty()) throw std::runtime_error("Empty value for " + key);
        std::size_t pos = 0;
        double d = std::stod(val, &pos);
        while (pos < val.size() && std::isspace(val[pos])) pos++;
        if (pos != val.size()) throw std::runtime_error("Trailing garbage in " + key + ": " + val);
        if (std::isnan(d) || std::isinf(d)) {
            throw std::runtime_error("Certificate contains NaN/Inf for " + key + ": " + val);
        }
        return d;
    };

    auto parse_key_int = [&](const std::string& l, const std::string& key) -> int {
        return parse_int(parse_key_str(l, key), key);
    };

    auto parse_key_double = [&](const std::string& l, const std::string& key) -> double {
        return parse_double(parse_key_str(l, key), key);
    };

    auto parse_key_opt_double = [&](const std::string& l, const std::string& key) -> std::optional<double> {
        std::string val = parse_key_str(l, key);
        if (val == "UNAVAILABLE" || val == "NOT_COMPUTED") return std::nullopt;
        return parse_double(val, key);
    };

    cert.status = parse_key_str(read_line(), "STATUS ");
    cert.num_rows = parse_key_int(read_line(), "ROWS ");
    if (cert.num_rows < 0) throw std::runtime_error("Negative ROWS");

    cert.num_cols = parse_key_int(read_line(), "COLS ");
    if (cert.num_cols < 0) throw std::runtime_error("Negative COLS");

    cert.objective_value = parse_key_double(read_line(), "OBJECTIVE ");

    cert.tolerance = parse_key_double(read_line(), "TOLERANCE ");
    if (cert.tolerance <= 0.0) throw std::runtime_error("Zero or negative TOLERANCE");

    cert.primal_residual = parse_key_double(read_line(), "PRIMAL_RESIDUAL ");
    cert.dual_residual = parse_key_opt_double(read_line(), "DUAL_RESIDUAL ");
    cert.complementarity_residual = parse_key_opt_double(read_line(), "COMPLEMENTARITY_RESIDUAL ");
    cert.bound_violation = parse_key_double(read_line(), "BOUND_VIOLATION ");
    cert.integrality_violation = parse_key_double(read_line(), "INTEGRALITY_VIOLATION ");
    cert.objective_bound_gap = parse_key_opt_double(read_line(), "OBJECTIVE_BOUND_GAP ");

    cert.ledger_entries = parse_key_int(read_line(), "LEDGER_ENTRIES ");
    if (cert.ledger_entries < 0) throw std::runtime_error("Negative LEDGER_ENTRIES");

    // X section.
    std::string x_header = read_line();
    if (x_header != "X") {
        throw std::runtime_error("Expected 'X', got: " + x_header);
    }

    cert.x.clear();
    cert.x.reserve(static_cast<std::size_t>(cert.num_cols));
    while (true) {
        std::string xl = read_line();
        if (xl == "END_X") break;
        double val = parse_double(xl, "X element");
        cert.x.push_back(val);
    }

    if (static_cast<int>(cert.x.size()) != cert.num_cols) {
        throw std::runtime_error("X vector size mismatch with COLS");
    }

    // END_CERTIFICATE.
    std::string end_line = read_line();
    if (end_line != "END_CERTIFICATE") {
        throw std::runtime_error("Expected 'END_CERTIFICATE', got: " + end_line);
    }

    // Reject trailing garbage lines
    std::string extra_line;
    if (std::getline(in, extra_line)) {
        if (!extra_line.empty() && extra_line.find_first_not_of(" \r\n\t") != std::string::npos) {
             throw std::runtime_error("Unexpected extra data after END_CERTIFICATE");
        }
    }

    return cert;
}

}  // namespace pramaan
