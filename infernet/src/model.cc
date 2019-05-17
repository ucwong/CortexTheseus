#include <cvm/c_api.h>
#include <cvm/model.h>
#include <cvm/dlpack.h>
#include <cvm/runtime/module.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>
#include <stdio.h>
#include <string.h>

#include <time.h>

using std::string;

namespace cvm {
namespace runtime {

const PackedFunc* module_creator = Registry::Get("cvm.runtime.create");

CVMModel::CVMModel(const string& graph, DLContext _ctx)
{
  ctx = _ctx;
  if (module_creator != nullptr) {
    module = (*module_creator)(
        graph, 
        static_cast<int>(ctx.device_type),
        static_cast<int>(ctx.device_id)
      );
    auto setup = module.GetFunction("setup");
    setup();
    loaded = true;
  } else {
    return;
  }
  set_input = module.GetFunction("set_input");
  get_output = module.GetFunction("get_output");
  load_params = module.GetFunction("load_params");
  run = module.GetFunction("run");
  auto get_input_shape = module.GetFunction("get_input_shape");
  DLTensor* t = new DLTensor();
  get_input_shape("data", t);
  in_ndim = t->ndim;
  in_shape = new int64_t[in_ndim];
  memcpy(in_shape, t->shape, in_ndim * sizeof(int64_t));
  in_size = 1;
  for (int i = 0; i < in_ndim; ++i) in_size *= in_shape[i];

  auto get_output_shape = module.GetFunction("get_output_shape");
  get_output_shape(0, t);
  out_ndim = t->ndim;
  out_shape = new int64_t[out_ndim];
  memcpy(out_shape, t->shape, out_ndim * sizeof(int64_t));
  out_size = 1;
  for (int i = 0; i < out_ndim; ++i) out_size *= out_shape[i];

  delete t->shape;
  delete t;
}

CVMModel::~CVMModel() {
  delete in_shape;
  delete out_shape;
}

DLTensor* CVMModel::LoadRandomInput() {
  DLTensor* ret;
  CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int*>(ret->data);
  for (int i = 0; i < in_size; ++i) {
    data[i] = i % 255 - 127;
  }
  return ret;
}

DLTensor* CVMModel::PlanOutput() {
  DLTensor* ret;
  CVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  return ret;
}

int CVMModel::LoadParams(CVMByteArray& params) {
  return params.size == 0 ? -1 : load_params(params);
}

int CVMModel::LoadParams(string params_str) {
  params_str_ = params_str;
  CVMByteArray params;
  params.data = params_str_.c_str();
  params.size = params_str_.length();
  return LoadParams(params);
}

int CVMModel::SetInput_(string index, DLTensor* input) {
  return input == nullptr ? -1 : set_input(index, input);
}

int CVMModel::GetOutput_(int index, DLTensor* output) {
  return output == nullptr ? -1 : get_output(index, output);
}

int CVMModel::Run_() {
  return run();
}

int CVMModel::Run(DLTensor*& input, DLTensor*& output) {
  int ret = 
    SetInput_("data", input) ||
    Run_() ||
    GetOutput_(0, output);
  return ret;
}

int CVMModel::GetInputLength() {
  return static_cast<int>(in_size);
}

int CVMModel::GetOutputLength() {
  return static_cast<int>(out_size);
}

DLTensor* CVMModel::GetInputTensor(char* data_) {
  DLTensor* ret;
  CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int*>(ret->data);
  for (int i = 0; i < in_size; ++i) {
    data[i] = data_[i];
  }
  return ret;
}

DLTensor* CVMModel::Run() {
  auto input = LoadRandomInput();
  DLTensor *output = PlanOutput();
  Run(input, output);
  CVMArrayFree(input);
  return output;
}
/*
 * int estimate_ops() {
  std::ifstream json_in("/tmp/mnist.nnvm.compile.json", std::ios::in);
  string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();
  auto f = Registry::Get("cvm.runtime.estimate_ops");
  if (f == nullptr) return;
  int ret;
  ret = (*f)(json_data);
}
*/

}
}

using cvm::runtime::CVMModel;

string LoadFromFile(string filepath) {
  std::ifstream input_stream(filepath, std::ios::in);
  string str = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
  input_stream.close();
  return str;
}

string LoadFromBinary(string filepath) {
  std::ifstream input_stream(filepath, std::ios::binary);
  string str = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
  input_stream.close();
  return str;
}

void* _cvm_load_model(char *graph_fname, char *model_fname) {
  string graph = LoadFromFile(string(graph_fname));
  CVMModel* model = new CVMModel(graph, DLContext{kDLGPU, 0});
  string params = LoadFromBinary(string(model_fname));
  model->LoadParams(params);
  return (void*)model;
}

void _cvm_free_model(void *model_) {
  CVMModel* model = (CVMModel*)model_;
  delete model;
}

int _cvm_get_output_length(void *model_) {
  CVMModel* model = (CVMModel*)model_;
  return model->GetOutputLength();
  int LoadParams(CVMByteArray& params);
 }

int _cvm_predict(void *model_, char *input_data, char *output_data) {
  CVMModel* model = (CVMModel*)model_;
  DLTensor* input = model->GetInputTensor(input_data);
  DLTensor* output = model->PlanOutput();
  int t = model->Run(input, output);
  if (t) return t;
  int32_t *data = static_cast<int32_t*>(output->data);
  for (int i = 0; i < model->GetOutputLength(); ++i) {
    output_data[i] = static_cast<int8_t>(data[i]);
  }
  CVMArrayFree(input);
  CVMArrayFree(output);
  return 0;
}

