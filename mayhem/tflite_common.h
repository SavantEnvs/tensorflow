// Shared helpers for the TFLite fuzz harnesses (model verify -> build -> allocate -> invoke).
#ifndef MAYHEM_TFLITE_COMMON_H_
#define MAYHEM_TFLITE_COMMON_H_

#include <cstdarg>
#include <cstring>
#include <memory>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace mayhem {

// Nearly every fuzz input is malformed; swallow the per-rejection stderr noise.
class NullErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char*, va_list) override { return 0; }
};

// A runtime input that is safe and useful to zero (not a const/mmap'd tensor, not a string-like).
inline bool IsZeroableInput(const TfLiteTensor* tensor) {
  if (tensor == nullptr || tensor->data.raw == nullptr || tensor->bytes == 0) return false;
  if (tensor->allocation_type == kTfLiteMmapRo ||
      tensor->allocation_type == kTfLitePersistentRo) {
    return false;
  }
  if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
      tensor->type == kTfLiteVariant) {
    return false;
  }
  return true;
}

inline void ZeroInputs(tflite::Interpreter& interpreter) {
  for (int idx : interpreter.inputs()) {
    TfLiteTensor* t = interpreter.tensor(idx);
    if (IsZeroableInput(t)) std::memset(t->data.raw, 0, t->bytes);
  }
}

}  // namespace mayhem

#endif  // MAYHEM_TFLITE_COMMON_H_
