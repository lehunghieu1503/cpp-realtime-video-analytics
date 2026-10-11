#!/usr/bin/env bash
# Build with a sanitizer and run every test under it, then the whole
# pipeline on a video if one is given.
#
# Usage: scripts/sanitize.sh [asan] [tsan]    (default: both)
# Environment:
#   ONNXRUNTIME_ROOT  where ONNX Runtime is installed, if not /usr/local
#   VIDEO             video file for the pipeline run (needs
#                     models/yolov8n.onnx); skipped when unset
#
# Notes on the setup:
#   - OpenVINO is left out: its TBB loads with RTLD_DEEPBIND, which
#     AddressSanitizer refuses. So is TensorRT: the sanitizers cannot see
#     into the GPU libraries.
#   - ThreadSanitizer is built with g++-12 when the default is older. GCC 11's
#     runtime does not know pthread_cond_clockwait, which
#     condition_variable::wait_for uses, and reports a double lock and data
#     races in correct code.
#   - Address-space randomisation is turned off for the runs: ThreadSanitizer
#     aborts with "unexpected memory mapping" on kernels with 32 bits of it.
set -euo pipefail

cd "$(dirname "$0")/.."
[ $# -eq 0 ] && set -- asan tsan
export LSAN_OPTIONS="suppressions=$PWD/tests/lsan.supp:print_suppressions=0"
export TSAN_OPTIONS="suppressions=$PWD/tests/tsan.supp:print_suppressions=0"
[ -n "${ONNXRUNTIME_ROOT:-}" ] &&
  export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run() { setarch "$(uname -m)" -R "$@"; }
failed=0

for sanitizer in "$@"; do
  dir=build-$sanitizer
  flags=(-DCMAKE_BUILD_TYPE=Debug -DWITH_OPENVINO=OFF -DWITH_TENSORRT=OFF
    "-DENABLE_$(echo "$sanitizer" | tr a-z A-Z)=ON")
  [ -n "${ONNXRUNTIME_ROOT:-}" ] && flags+=("-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT")
  if [ "$sanitizer" = tsan ] && [ "$(g++ -dumpversion | cut -d. -f1)" -lt 12 ] &&
    command -v g++-12 >/dev/null; then
    flags+=(-DCMAKE_CXX_COMPILER=g++-12)
  fi
  echo "== $sanitizer: building in $dir"
  cmake -B "$dir" "${flags[@]}" >/dev/null 2>&1
  cmake --build "$dir" --parallel "$(nproc)" >/dev/null

  for test in "$dir"/tests/test_*; do
    if run "$test" >"$dir/$(basename "$test").log" 2>&1; then
      echo "   pass  $(basename "$test")"
    else
      echo "   FAIL  $(basename "$test")  (see $dir/$(basename "$test").log)"
      failed=1
    fi
  done

  if [ -n "${VIDEO:-}" ]; then
    # Two workers and two sources exercise the queues, the tracker, the
    # rules and the HTTP server together
    cat >"$dir/pipeline.yaml" <<YAML
backend: cpu
workers: 2
headless: true
port: 9099
sources:
  - id: a
    file: $VIDEO
    lines:
      - {name: mid, a: [0, 240], b: [4000, 240]}
  - id: b
    file: $VIDEO
    zones:
      - {name: all, points: [[0, 0], [4000, 0], [4000, 4000], [0, 4000]]}
YAML
    if run "$dir/src/video_analytics" --config "$dir/pipeline.yaml" \
      >"$dir/pipeline.log" 2>&1; then
      echo "   pass  pipeline on $VIDEO"
    else
      echo "   FAIL  pipeline  (see $dir/pipeline.log)"
      failed=1
    fi
  fi
done

exit $failed
