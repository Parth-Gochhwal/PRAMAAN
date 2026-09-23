#!/bin/bash
set -e

echo "=== BUILD AND TEST REPORT ==="
rm -rf build-final
cmake -S . -B build-final -DCMAKE_BUILD_TYPE=Release -DPRAMAAN_ENABLE_CUDA=ON > /dev/null
cmake --build build-final -j"$(nproc)" > /dev/null
cd build-final
ctest --output-on-failure
cd ..

echo "=== GIT CHECK ==="
git diff --check
git diff --cached --check
echo "(git diff check passed silently with no errors)"
