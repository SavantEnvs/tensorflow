#!/usr/bin/env bash
#
# mayhem/build.sh — build the TFLite fuzz harnesses (+ standalone reproducers), upstream's kernel gtests
# (mayhem/kernel_tests.txt) and the KAT test probe.
#
# TFLite is built through its CMake entry point (tensorflow/lite/CMakeLists.txt), not Bazel. That file
# does not configure standalone at this commit: it names a nonexistent tensorflow/lite/converter/ (the
# sources live in tensorflow/compiler/mlir/lite/) and assumes it is the project root. Upstream files
# are not edited; a patched COPY is generated under build/mayhem/ (gitignored) and built instead.
#
# Network is only used on the very first run, to pre-fetch the pinned FetchContent dependencies into the
# image-level cache $TFLITE_DEPS (outside the git tree, so it survives `git clean -ffdX`); every later
# run, including the air-gapped re-run, configures against that cache via FETCHCONTENT_SOURCE_DIR_*.
set -euo pipefail

[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH

: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
: "${COVERAGE_FLAGS=}"
: "${TFLITE_DEPS:=/opt/tflite-deps}"
export SANITIZER_FLAGS DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

SRC="${SRC:-/mayhem}"
cd "$SRC"
B="$SRC/build/mayhem"
mkdir -p "$B"

DEP_NAMES="abseil-cpp cpuinfo eigen farmhash fft2d flatbuffers gemmlowp ml_dtypes neon2sse ruy googletest google_benchmark re2"
CMAKE_COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=None -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
  -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_RUY=OFF
  -DTENSORFLOW_HEADERS_INCLUDE_DIR="$SRC/third_party/xla" -Wno-dev)

# 0) Patched copy of TFLite's CMakeLists (see header), minus the label_image/pywrap tail that needs
#    an in-tree source dir, plus our harness targets. The kernel-test subdirectory (TFLITE_KERNEL_TEST)
#    has the same two problems, so it gets a patched copy too (0b).
mkdir -p "$B/cm/kernels"
sed -e "s|set(TFLITE_SOURCE_DIR \"\${CMAKE_CURRENT_LIST_DIR}\")|set(TFLITE_SOURCE_DIR \"$SRC/tensorflow/lite\")\nset(TF_ROOT \"$SRC\")|" \
    -e 's|${TFLITE_SOURCE_DIR}/converter/|${TF_ROOT}/tensorflow/compiler/mlir/lite/|g' \
    -e 's|${TFLITE_SOURCE_DIR}/\.\.|${TF_ROOT}|g' \
    -e 's|${CMAKE_CURRENT_SOURCE_DIR}/\.\.|${TF_ROOT}|g' \
    -e 's|add_subdirectory(${TFLITE_SOURCE_DIR}/kernels)|add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/kernels kernels)|' \
    "$SRC/tensorflow/lite/CMakeLists.txt" > "$B/cm/CMakeLists.txt"
sed -i '/^# The label_image example/,$d' "$B/cm/CMakeLists.txt"
echo 'include(${TF_ROOT}/mayhem/tflite_targets.cmake)' >> "$B/cm/CMakeLists.txt"

# 0b) Patched copy of kernels/CMakeLists.txt: converter/ sources moved to tensorflow/compiler/mlir/lite/;
#     the generated mutable schema header goes where the sources include it from; its relative sources
#     (the TEST_FRAMEWORK_SRC list, test_main.cc, the test source given to add_executable) are made
#     absolute because the copy lives elsewhere; the XNNPACK delegate provider is dropped (XNNPACK is off); ctest runs from the repo root so tests find
#     "tensorflow/lite/testdata/...".
sed -e '1i set(KDIR "${TFLITE_SOURCE_DIR}/kernels")' \
    -e 's|${TFLITE_SOURCE_DIR}/converter/|${TF_ROOT}/tensorflow/compiler/mlir/lite/|g' \
    -e 's|${PROJECT_BINARY_DIR}/tflite/converter/schema/mutable|${PROJECT_BINARY_DIR}/tensorflow/compiler/mlir/lite/schema/mutable|' \
    -e '/^  ${TFLITE_SOURCE_DIR}\/tools\/delegates\/xnnpack_delegate_provider.cc$/d' \
    -e '/^set(TEST_FRAMEWORK_SRC/,/^)/ s|^\(  \)\([A-Za-z0-9_/]*\.cc\)$|\1${KDIR}/\2|' \
    -e 's|^set(TEST_FRAMEWORK_MAIN_SRC test_main.cc)|set(TEST_FRAMEWORK_MAIN_SRC ${KDIR}/test_main.cc)|' \
    -e 's|add_executable(${TEST_NAME} ${TEST_SRC})|add_executable(${TEST_NAME} ${KDIR}/${TEST_SRC})|' \
    -e 's|WORKING_DIRECTORY ${TFLITE_SOURCE_DIR}/\.\.|WORKING_DIRECTORY ${TF_ROOT}|' \
    "$SRC/tensorflow/lite/kernels/CMakeLists.txt" > "$B/cm/kernels/CMakeLists.txt"

# 1) Dependency cache: fetch once (network), reuse offline forever.
if [ ! -e "$TFLITE_DEPS/.complete-kernel-tests" ]; then
  mkdir -p "$TFLITE_DEPS"
  cmake -S "$B/cm" -B "$B/fetch" "${CMAKE_COMMON[@]}" -DTFLITE_KERNEL_TEST=ON -DMAYHEM_BUILD=none >"$B/fetch.log" 2>&1 \
    || { tail -30 "$B/fetch.log" >&2; exit 1; }
  for n in $DEP_NAMES; do
    [ -d "$B/fetch/$n" ] || { echo "dependency $n was not fetched" >&2; exit 1; }
    rm -rf "${TFLITE_DEPS:?}/$n"; cp -a "$B/fetch/$n" "$TFLITE_DEPS/$n"
  done
  touch "$TFLITE_DEPS/.complete-kernel-tests"
fi
DEP_ARGS=()
for n in $DEP_NAMES; do
  DEP_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_$(echo "$n" | tr a-z A-Z)=$TFLITE_DEPS/$n")
done

# 2) Fuzz build: the whole TFLite runtime instrumented with $SANITIZER_FLAGS (+ libFuzzer coverage
#    instrumentation when sanitizers are on), linked into each harness; plus *-standalone reproducers.
COV=""
[ -z "$SANITIZER_FLAGS" ] || COV="-fsanitize=fuzzer-no-link"
FF="$SANITIZER_FLAGS $COV -O1 $DEBUG_FLAGS"
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o "$B/standalone_main.o"
# A cache from an earlier run may hold a different compiler path (e.g. built through ccache's masquerade
# dir); CMake would then drop its whole cache, including our -D options. Re-seeding it is deterministic,
# so unchanged commands still no-op in ninja.
rm -f "$B/fuzz/CMakeCache.txt" "$B/kat/CMakeCache.txt"
cmake -S "$B/cm" -B "$B/fuzz" "${CMAKE_COMMON[@]}" "${DEP_ARGS[@]}" \
  -DCMAKE_C_FLAGS="$FF" -DCMAKE_CXX_FLAGS="$FF" \
  -DCMAKE_EXE_LINKER_FLAGS="$SANITIZER_FLAGS" \
  -DMAYHEM_BUILD=fuzz -DMAYHEM_OUT="$SRC" -DMAYHEM_LSAN_OFF_SRC="$SRC/mayhem/lsan_off.cc" \
  -DMAYHEM_FUZZ_ENGINE="$LIB_FUZZING_ENGINE" -DMAYHEM_STANDALONE_MAIN_OBJ="$B/standalone_main.o"
cmake --build "$B/fuzz" -j"$MAYHEM_JOBS" \
  --target tflite_interp tflite_interp-standalone tflite_resize tflite_resize-standalone

# 3) Test build: an independent, unsanitized TFLite with the project's normal flags: the known-answer
#    probe tflite_kat plus the upstream kernel gtests listed in mayhem/kernel_tests.txt (run by test.sh
#    through ctest from $B/kat).
KF="-O1 $COVERAGE_FLAGS"
cmake -S "$B/cm" -B "$B/kat" "${CMAKE_COMMON[@]}" "${DEP_ARGS[@]}" \
  -DCMAKE_C_FLAGS="$KF" -DCMAKE_CXX_FLAGS="$KF" -DCMAKE_EXE_LINKER_FLAGS="$COVERAGE_FLAGS" \
  -DTFLITE_KERNEL_TEST=ON -DMAYHEM_BUILD=kat -DMAYHEM_OUT="$SRC"
KERNEL_TESTS="$(grep -v '^#' "$SRC/mayhem/kernel_tests.txt" | tr '\n' ' ')"
# shellcheck disable=SC2086
cmake --build "$B/kat" -j"$MAYHEM_JOBS" --target tflite_kat $KERNEL_TESTS
