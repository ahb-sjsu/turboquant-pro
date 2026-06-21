#!/usr/bin/env bash
# Build the tq-pro M1 CPU SIMD ADC kernel as a Python extension.
#   PY=/path/to/python ./build.sh [out_dir]
set -euo pipefail
PY="${PY:-python3}"
OUT="${1:-$(dirname "$0")}"
EXT="$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
INC="$("$PY" -m pybind11 --includes)"
mkdir -p "$OUT"
# -march=native enables AVX2 (pshufb fast path); falls back to scalar otherwise.
# shellcheck disable=SC2086
g++ -O3 -march=native -fopenmp -shared -fPIC -std=c++17 $INC \
    "$(dirname "$0")/adc_scan.cpp" -o "$OUT/adc_scan$EXT"
echo "built $OUT/adc_scan$EXT"
