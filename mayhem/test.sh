#!/usr/bin/env bash
#
# mayhem/test.sh — RUN the TFLite test suites built by mayhem/build.sh (normal flags, no sanitizers).
# Nothing is compiled here.
#  1. Upstream's own kernel/interpreter gtests (tensorflow/lite/kernels/*_test.cc, built through TFLite's
#     CMake TFLITE_KERNEL_TEST option), listed in mayhem/kernel_tests.txt and run through ctest. Each test
#     drives the interpreter (SingleOpModel) and asserts op results. A test binary that exits 0 without
#     reporting a passed gtest case (crash-free no-op, neutered binary) counts as failed.
#  2. tflite_kat, an additional known-answer probe: loads committed testdata models, runs them through the
#     interpreter and asserts exact output values (3*x for add.bin, a+b+c / d+b+c for multi_add.bin,
#     resized shapes, quantized monotonicity, rejection of malformed models).
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
cd "$SRC"

emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}

# --- 1) upstream kernel gtests through ctest ---
CT_DIR="$SRC/build/mayhem/kat"
names="$(grep -v '^#' "$SRC/mayhem/kernel_tests.txt" | grep -v '^$')"
expected_ct=$(printf '%s\n' "$names" | wc -l)
ct_pass=0; ct_fail=0; ct_skip=0
if [ ! -f "$CT_DIR/CTestTestfile.cmake" ] || ! command -v ctest >/dev/null; then
  echo "kernel tests missing: mayhem/build.sh did not produce $CT_DIR (or ctest is absent)" >&2
  ct_fail=$expected_ct
else
  log="$(mktemp "${TMPDIR:-/tmp}/ctest.XXXXXX")"
  regex="^($(printf '%s\n' "$names" | paste -sd'|'))\$"
  (cd "$CT_DIR" && ctest -R "$regex" -V -j "$(nproc)") >"$log" 2>&1
  ct_rc=$?
  tail -n 12 "$log"
  # Per ctest test: its result line, plus the gtest summary lines ("N: [  PASSED  ] k tests." etc.).
  read -r ct_pass ct_fail ct_skip < <(awk -v expected="$expected_ct" '
    /^ *[0-9]+\/[0-9]+ Test +#[0-9]+: / {
      n = $0; sub(/^.*Test +#/, "", n); sub(/:.*$/, "", n); seen[n] = 1
      if ($0 ~ / Passed /) ok[n] = 1
    }
    /^[0-9]+: \[  PASSED  \] [0-9]+ test/ { n = $1; sub(/:/, "", n); P[n] += $5 }
    /^[0-9]+: \[  SKIPPED \] [0-9]+ test.*listed below/ { n = $1; sub(/:/, "", n); S[n] += $5 }
    /^[0-9]+: \[  FAILED  \] [0-9]+ test.*listed below/ { n = $1; sub(/:/, "", n); F[n] += $5 }
    END {
      for (n in seen) {
        if (!ok[n]) { fail += (F[n] > 0 ? F[n] : 1); pass += P[n]; skip += S[n] }
        else if (P[n] == 0) { fail += 1 }              # exit 0 but no gtest case passed
        else { pass += P[n]; skip += S[n] }
      }
      nseen = 0; for (n in seen) nseen++
      if (nseen < expected) fail += expected - nseen    # a listed test did not run at all
      printf "%d %d %d\n", pass, fail, skip
    }' "$log")
  [ "$ct_rc" -eq 0 ] || [ "$ct_fail" -gt 0 ] || ct_fail=1
  rm -f "$log"
fi

# --- 2) known-answer probe ---
EXPECTED=26   # number of checks tflite_kat prints; fewer means it was cut short
KAT="$SRC/tflite_kat"
if [ ! -x "$KAT" ]; then
  echo "tflite_kat missing: mayhem/build.sh did not produce it" >&2
  emit_ctrf cmake-ctest "$ct_pass" $(( ct_fail + 1 )) "$ct_skip"
  exit 1
fi

out="$("$KAT" "$SRC/tensorflow/lite/testdata" 2>&1)"; rc=$?
printf '%s\n' "$out"
passed=$(printf '%s\n' "$out" | grep -c '^PASS ')
failed=$(printf '%s\n' "$out" | grep -c '^FAIL ')
summary=$(printf '%s\n' "$out" | grep -E '^KAT passed=[0-9]+ failed=[0-9]+$' | tail -1)
# A probe that exits early (crash, exit(0) no-op) prints no summary or too few checks: count that as failure.
[ "$rc" -eq 0 ] || failed=$(( failed + 1 ))
[ -n "$summary" ] || failed=$(( failed + 1 ))
[ $(( passed + failed )) -ge "$EXPECTED" ] || failed=$(( failed + EXPECTED - passed - failed ))
emit_ctrf cmake-ctest "$(( ct_pass + passed ))" "$(( ct_fail + failed ))" "$ct_skip"
