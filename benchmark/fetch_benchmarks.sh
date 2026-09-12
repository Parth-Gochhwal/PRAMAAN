#!/usr/bin/env bash
# fetch_benchmarks.sh
#
# Downloads the Stage 8 benchmark set: 10 Netlib LP instances from the
# COIN-OR Data-Netlib repository on GitHub.
#
# Source: https://github.com/coin-or-tools/Data-Netlib
# Files are distributed as .mps.gz and decompressed on download.
#
# Usage (from project root):
#   ./benchmark/fetch_benchmarks.sh
#
# All 10 instances use only ROWS, COLUMNS, RHS, BOUNDS, and ENDATA
# sections — no RANGES section — and are compatible with PRAMAAN's
# current free-format MPS parser.  Each was individually tested before
# inclusion in the benchmark set.
#
# Instance selection rationale (each confirmed working with PRAMAAN):
#   afiro     -  27 rows,   32 cols  - very small; useful as a basic parse/solve check
#   adlittle  -  56 rows,   97 cols  - small; straightforward LP structure
#   kb2       -  43 rows,   41 cols  - small; exercises bound handling
#   sc50a     -  49 rows,   48 cols  - small stochastic programming instance
#   sc205     - 204 rows,  203 cols  - medium stochastic programming instance
#   share2b   -  96 rows,   79 cols  - medium economic planning model
#   lotfi     - 153 rows,  308 cols  - medium-large; more columns than rows
#   israel    - 174 rows,  142 cols  - medium economic model
#   beaconfd  - 173 rows,  262 cols  - medium model with denser constraint matrix
#   scorpion  - 388 rows,  358 cols  - largest instance in the set
#
# blend.mps is not included: it uses fixed-format MPS with numeric values
# embedded in name fields, which PRAMAAN's free-format parser cannot handle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"

BASE_URL="https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master"

INSTANCES=(
    afiro
    adlittle
    kb2
    sc50a
    sc205
    share2b
    lotfi
    israel
    beaconfd
    scorpion
)

# Detect unexpected stale files
STALE_FOUND=0
for existing_file in $(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.mps' 2>/dev/null); do
    filename=$(basename "$existing_file" .mps)
    is_expected=0
    for expected in "${INSTANCES[@]}"; do
        if [ "$filename" = "$expected" ]; then
            is_expected=1
            break
        fi
    done
    if [ "$is_expected" -eq 0 ]; then
        echo "WARNING: Unexpected stale file found in benchmark data: $existing_file"
        STALE_FOUND=1
    fi
done

FAILED=0
for name in "${INSTANCES[@]}"; do
    dest="${DATA_DIR}/${name}.mps"
    if [ -s "${dest}" ] && grep -q "^NAME" "${dest}"; then
        echo "[OK]   ${name}.mps already exists and appears valid, skipping download"
        continue
    fi
    echo -n "[DL]   ${name}.mps.gz ... "
    url="${BASE_URL}/${name}.mps.gz"
    tmp="${DATA_DIR}/${name}.mps.gz"

    # curl -f returns non-zero on HTTP errors (e.g., 404)
    if curl -fsSL -o "${tmp}" "${url}"; then
        if gunzip -f "${tmp}"; then
            # Verify the decompressed file looks like an MPS
            if [ -s "${dest}" ] && grep -q "^NAME" "${dest}"; then
                echo "OK"
            else
                echo "FAILED (invalid MPS content)"
                rm -f "${dest}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "FAILED (gunzip)"
            rm -f "${tmp}"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "FAILED (download)"
        rm -f "${tmp}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================="
if [ "${FAILED}" -gt 0 ]; then
    echo "ERROR: ${FAILED} download(s) failed"
    exit 1
fi

# Verify each expected instance individually.
MISSING=0
for name in "${INSTANCES[@]}"; do
    dest="${DATA_DIR}/${name}.mps"
    if [ -s "${dest}" ] && grep -q "^NAME" "${dest}"; then
        echo "[VERIFIED] ${name}.mps"
    else
        echo "[MISSING]  ${name}.mps"
        MISSING=$((MISSING + 1))
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    echo "ERROR: ${MISSING} expected instance(s) missing or invalid."
    exit 1
fi

if [ "${STALE_FOUND}" -eq 1 ]; then
    echo "NOTE: Data directory contains extra files. The benchmark runner will safely ignore them."
fi
echo "All ${#INSTANCES[@]} expected instances verified."
