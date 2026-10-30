// libFuzzer harness: TFLite model verify, interpreter build, tensor allocation and invoke.
// Port of tensorflow/security/fuzzing/cc/tflite_interp_overflow_fuzz.cc (FuzzTest) to libFuzzer.
// Models enter via FlatBufferModel::VerifyAndBuildFromBuffer, so structurally invalid
// flatbuffers are rejected at ingress; the uint64 external-offset fields the verifier cannot
// check are exercised in InterpreterBuilder::ParseNodes/ParseTensors.
#include <cstddef>
#include <cstdint>
#include <string>

#include "tflite_common.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Copy so the buffer is heap-aligned, as a loaded model file would be.
  const std::string bytes(reinterpret_cast<const char*>(data), size);
  mayhem::NullErrorReporter reporter;
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      bytes.data(), bytes.size(), /*extra_verifier=*/nullptr, &reporter);
  if (model == nullptr) return -1;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk ||
      interpreter == nullptr) {
    return 0;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) return 0;
  mayhem::ZeroInputs(*interpreter);
  interpreter->Invoke();
  return 0;
}
