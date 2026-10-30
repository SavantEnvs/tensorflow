// libFuzzer harness: like tflite_interp, but first resizes model inputs to fuzzer-chosen shapes.
// Input layout: [sel][ndims][d0][d1][d2][d3] followed by the model flatbuffer. The 6-byte header
// picks the input tensor (or all of them when the high bit of `ndims` is set) and its new
// dimensions; shape inference in each kernel's Prepare() runs against attacker-chosen shapes.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tflite_common.h"

namespace {
constexpr size_t kHeader = 6;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size <= kHeader) return -1;
  const uint8_t sel = data[0];
  const bool all_inputs = (data[1] & 0x80) != 0;
  const int ndims = data[1] & 0x7f;
  if (ndims > 4) return -1;
  std::vector<int> dims;
  for (int i = 0; i < ndims; ++i) dims.push_back(data[2 + i] % 33);

  const std::string bytes(reinterpret_cast<const char*>(data + kHeader), size - kHeader);
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
  const std::vector<int> inputs = interpreter->inputs();
  if (inputs.empty()) return 0;
  if (all_inputs) {
    for (int idx : inputs) interpreter->ResizeInputTensor(idx, dims);
  } else {
    interpreter->ResizeInputTensor(inputs[sel % inputs.size()], dims);
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) return 0;
  mayhem::ZeroInputs(*interpreter);
  interpreter->Invoke();
  return 0;
}
