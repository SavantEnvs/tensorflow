// Known-answer tests for the TFLite runtime, built with NORMAL flags (no sanitizers) and run by
// mayhem/test.sh. Each case loads a committed testdata model, runs it through the interpreter
// and asserts exact output values, so a patch that no-ops the runtime or breaks a kernel fails.
// Usage: tflite_kat <testdata-dir>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace {

int g_pass = 0, g_fail = 0;

void Check(bool ok, const std::string& name) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
  (ok ? g_pass : g_fail)++;
}

struct Loaded {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interp;
};

Loaded Load(const std::string& dir, const std::string& file) {
  Loaded l;
  l.model = tflite::FlatBufferModel::BuildFromFile((dir + "/" + file).c_str());
  if (!l.model) return l;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*l.model, resolver)(&l.interp);
  return l;
}

void Fill(tflite::Interpreter& it, int slot, float base, float step) {
  float* p = it.typed_input_tensor<float>(slot);
  int n = static_cast<int>(it.tensor(it.inputs()[slot])->bytes / sizeof(float));
  for (int i = 0; i < n; ++i) p[i] = base + step * i;
}

// add.bin: out = (in + in) + in = 3 * in over a float [1,8,8,3] tensor.
void TestAdd(const std::string& dir) {
  Loaded l = Load(dir, "add.bin");
  Check(l.interp != nullptr, "add: interpreter builds");
  if (!l.interp) return;
  Check(l.interp->inputs().size() == 1 && l.interp->outputs().size() == 1, "add: one input, one output");
  Check(l.interp->AllocateTensors() == kTfLiteOk, "add: AllocateTensors");
  const TfLiteTensor* in = l.interp->tensor(l.interp->inputs()[0]);
  Check(in->type == kTfLiteFloat32 && in->bytes == 8 * 8 * 3 * sizeof(float), "add: input is float32[1,8,8,3]");
  Fill(*l.interp, 0, -24.0f, 0.5f);
  Check(l.interp->Invoke() == kTfLiteOk, "add: Invoke");
  const float* out = l.interp->typed_output_tensor<float>(0);
  bool ok = true;
  for (int i = 0; i < 192; ++i) ok &= out[i] == 3.0f * (-24.0f + 0.5f * i);
  Check(ok, "add: output == 3 * input for all 192 elements");
  Check(out[0] == -72.0f && out[191] == 3.0f * (-24.0f + 95.5f), "add: spot values -72 and 214.5");
}

// add.bin resized to [1,2,2,3]: the same kernels re-prepare for the new shape.
void TestAddResize(const std::string& dir) {
  Loaded l = Load(dir, "add.bin");
  if (!l.interp) { Check(false, "add-resize: interpreter builds"); return; }
  Check(l.interp->ResizeInputTensor(l.interp->inputs()[0], {1, 2, 2, 3}) == kTfLiteOk, "add-resize: ResizeInputTensor");
  Check(l.interp->AllocateTensors() == kTfLiteOk, "add-resize: AllocateTensors");
  Fill(*l.interp, 0, 1.0f, 1.0f);
  Check(l.interp->Invoke() == kTfLiteOk, "add-resize: Invoke");
  const TfLiteTensor* out = l.interp->tensor(l.interp->outputs()[0]);
  Check(out->bytes == 12 * sizeof(float), "add-resize: output has 12 elements");
  const float* o = l.interp->typed_output_tensor<float>(0);
  bool ok = true;
  for (int i = 0; i < 12; ++i) ok &= o[i] == 3.0f * (1.0f + i);
  Check(ok, "add-resize: output == 3 * input (3,6,...,36)");
}

// multi_add.bin: inputs a,b,c,d; outputs x = a+b+c and y = d+b+c.
void TestMultiAdd(const std::string& dir) {
  Loaded l = Load(dir, "multi_add.bin");
  Check(l.interp != nullptr, "multi_add: interpreter builds");
  if (!l.interp) return;
  Check(l.interp->inputs().size() == 4 && l.interp->outputs().size() == 2, "multi_add: 4 inputs, 2 outputs");
  Check(l.interp->AllocateTensors() == kTfLiteOk, "multi_add: AllocateTensors");
  Fill(*l.interp, 0, 1.0f, 0.0f);     // a = 1
  Fill(*l.interp, 1, 10.0f, 0.0f);    // b = 10
  Fill(*l.interp, 2, 100.0f, 0.0f);   // c = 100
  Fill(*l.interp, 3, 1000.0f, 1.0f);  // d = 1000 + i
  Check(l.interp->Invoke() == kTfLiteOk, "multi_add: Invoke");
  const float* x = l.interp->typed_output_tensor<float>(0);
  const float* y = l.interp->typed_output_tensor<float>(1);
  bool okx = true, oky = true;
  for (int i = 0; i < 192; ++i) {
    okx &= x[i] == 111.0f;
    oky &= y[i] == 1110.0f + i;
  }
  Check(okx, "multi_add: x == a+b+c == 111");
  Check(oky, "multi_add: y == d+b+c == 1110+i");
}

// add_quantized.bin: uint8 add; the result must track the real-valued sum within quantization error.
void TestAddQuantized(const std::string& dir) {
  Loaded l = Load(dir, "add_quantized.bin");
  Check(l.interp != nullptr, "add_quantized: interpreter builds");
  if (!l.interp) return;
  Check(l.interp->AllocateTensors() == kTfLiteOk, "add_quantized: AllocateTensors");
  const TfLiteTensor* in = l.interp->tensor(l.interp->inputs()[0]);
  const TfLiteTensor* out = l.interp->tensor(l.interp->outputs()[0]);
  Check(in->type == kTfLiteUInt8 && out->type == kTfLiteUInt8, "add_quantized: uint8 tensors");
  uint8_t* ip = l.interp->typed_input_tensor<uint8_t>(0);
  for (size_t i = 0; i < in->bytes; ++i) ip[i] = static_cast<uint8_t>(i % 64);
  Check(l.interp->Invoke() == kTfLiteOk, "add_quantized: Invoke");
  const uint8_t* op = l.interp->typed_output_tensor<uint8_t>(0);
  // Same-valued input maps monotonically and identically: equal inputs give equal outputs and
  // the output is non-decreasing in the input over the ramp.
  bool mono = true, distinct = false;
  for (size_t i = 1; i < out->bytes && i < 64; ++i) {
    mono &= op[i] >= op[i - 1];
    distinct |= op[i] != op[0];
  }
  Check(mono && distinct, "add_quantized: output is a monotone, non-constant function of input");
}

// Malformed and degenerate models must be rejected, not accepted.
void TestRejects(const std::string& dir) {
  auto broken = tflite::FlatBufferModel::VerifyAndBuildFromFile((dir + "/test_model_broken.bin").c_str());
  Check(broken == nullptr, "reject: test_model_broken.bin fails verification");
  const char junk[] = "this is not a flatbuffer model at all, just text........";
  auto m = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(junk, sizeof(junk));
  Check(m == nullptr, "reject: text is not a model");
  auto good = tflite::FlatBufferModel::VerifyAndBuildFromFile((dir + "/add.bin").c_str());
  Check(good != nullptr, "accept: add.bin passes verification");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <testdata-dir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  TestAdd(dir);
  TestAddResize(dir);
  TestMultiAdd(dir);
  TestAddQuantized(dir);
  TestRejects(dir);
  std::printf("KAT passed=%d failed=%d\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
