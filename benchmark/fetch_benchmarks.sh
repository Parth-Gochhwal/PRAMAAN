#!/usr/bin/env bash
# fetch_benchmarks.sh
# Downloads a small Netlib LP subset into benchmark/data/ for local testing.
# Netlib LP set: see https://plato.asu.edu/sub/testcases.html for current
# mirror links (the old netlib.org ones move around -- check that page).
#
# FIRST TASK: start with just ONE instance -- "afiro" is the smallest,
# most famous Netlib LP (32 constraints, 3 nonzeros/column) and is the
# standard first-instance-to-solve for any new LP solver. Get PRAMAAN
# solving afiro correctly before downloading the rest of the set.
set -e
mkdir -p data
echo "TODO: curl/wget the afiro.mps instance from a current Netlib mirror"
echo "See https://plato.asu.edu/sub/testcases.html for current links"
