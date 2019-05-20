#include <cvm/c_api.h>
#include <cvm/model.h>
#include <cvm/dlpack.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <mutex>

#include <time.h>

using std::string;


namespace cvm {
namespace runtime {


CVMModel::CVMModel(const string& graph, DLContext _ctx)
{
  loaded = false;
  ctx = _ctx;
  std::vector<CVMContext> contexts;
  CVMContext ctx_tmp;
  ctx_tmp.device_type = static_cast<DLDeviceType>(_ctx.device_type);
  ctx_tmp.device_id = ctx.device_id;
  contexts.push_back(ctx_tmp);
  exec_ = std::make_shared<CvmRuntime>();
  exec_->Init(graph, contexts);

  exec_->SetupGraph();

  loaded = true;
  // load_params = module.GetFunction("load_params");
  DLTensor* t = new DLTensor();

  int in_idx = exec_->GetInputIndex("data");
  if (in_idx >= 0) {
      exec_->GetShape(in_idx, t);
  }
  else {
      loaded = false;
      delete t->shape;
      delete t;
      return ;
  }
  in_ndim = t->ndim;
  in_shape = new int64_t[in_ndim];
  memcpy(in_shape, t->shape, in_ndim * sizeof(int64_t));
  in_size = 1;
  for (int i = 0; i < in_ndim; ++i) in_size *= in_shape[i];

  exec_->GetOutputShape(0, t);
  out_ndim = t->ndim;
  out_shape = new int64_t[out_ndim];
  memcpy(out_shape, t->shape, out_ndim * sizeof(int64_t));
  out_size = 1;
  for (int i = 0; i < out_ndim; ++i) out_size *= out_shape[i];

  delete t->shape;
  delete t;
}

CVMModel::~CVMModel() {
  if (in_shape) delete in_shape;
  if (out_shape) delete out_shape;
}

DLTensor* CVMModel::PlanInput() {
  DLTensor* ret;
  CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  return ret;
}

DLTensor* CVMModel::PlanInput(char *input) {
  DLTensor* ret = nullptr;
  CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int*>(ret->data);
  for (int i = 0; i < in_size; ++i) {
    data[i] = input[i];
  }
  return ret;
}

DLTensor* CVMModel::PlanOutput() {
  DLTensor* ret;
  CVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  return ret;
}

void CVMModel::SaveTensor(DLTensor* input, char* mem) {
  auto data = static_cast<int*>(input->data);
  // std::cerr << "save tensor" << input->ndim << " " << input->shape[0] << " " <<  input->shape[1] <<  "insize:" << in_size << "\n";
  for (int i = 0; i < out_size; ++i) {
    mem[i] = static_cast<int8_t>(data[i]);
  }
}

int CVMModel::LoadParams(const string &params) {
  if (params.size() == 0) return -1;
  // CVMByteArray arr;
  arr.data = params.c_str();
  arr.size = params.length();
  exec_->LoadParams(params);
  return 0;
}

int CVMModel::SetInput_(string index, DLTensor* input) {

  int in_idx = exec_->GetInputIndex(index);
  if (in_idx >= 0) exec_->SetInput(in_idx, input);
  // return input == nullptr ? -1 : set_input(index, input);
  return 0;
}

int CVMModel::GetOutput_(int index, DLTensor* output) {
  exec_->CopyOutputTo(index, output);
  // return output == nullptr ? -1 : get_output(index, output);
  return 0;
}

int CVMModel::Run_() {
  exec_->Run();
  return 0;
}

int CVMModel::Run(DLTensor*& input, DLTensor*& output) {
  int ret;
  ret = SetInput_("data", input);
  // std::cerr << "SetInput_ " << ret << "\n";
  if (ret != 0)
      return ret;
  ret = Run_();
  // std::cerr << "Run_ " << ret << "\n";
  if (ret != 0)
      return ret;
  ret = GetOutput_(0, output);
  // std::cerr << "GetOutput_ " << ret << "\n";
  if (ret != 0)
      return ret;
  return ret;
}

int CVMModel::GetInputLength() {
  return static_cast<int>(in_size);
}

int CVMModel::GetOutputLength() {
  return static_cast<int>(out_size);
}

int CVMModel::LoadParamsFromFile(string filepath) {
  std::ifstream input_stream(filepath, std::ios::binary);
  std::string params = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
  input_stream.close();
  return LoadParams(params);
}

/*
int estimate_ops() {
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

using cvm::runtime::CVMModel;

void* CVMAPILoadModel(const char *graph_fname, const char *model_fname) {
  // std::lock_guard<std::mutex> _(cvm_global_mtx);
  string graph = LoadFromFile(string(graph_fname));
  CVMModel* model = new CVMModel(graph, DLContext{kDLCPU, 0});
  string params = LoadFromBinary(string(model_fname));
  if (!model->loaded || model->LoadParams(params)) {
    if (model)
        delete model;
    return NULL;
  }
  return model;
}

void CVMAPIFreeModel(void* model_) {
  // std::lock_guard<std::mutex> _(cvm_global_mtx);
  CVMModel* model = static_cast<CVMModel*>(model_);
  if (model_)
      delete model;
}

int CVMAPIGetInputLength(void* model_) {
  // std::lock_guard<std: :mutex> _(cvm_global_mtx);
  CVMModel* model = (CVMModel*)model_;
  return model->GetInputLength();
}

int CVMAPIGetOutputLength(void* model_) {
  // std::lock_guard<std::mutex> _(cvm_global_mtx);
  CVMModel* model = static_cast<CVMModel*>(model_);
  if (model == nullptr)
      return 0;
  return model->GetOutputLength();
}

int CVMAPIInfer(void* model_, char *input_data, char *output_data) {
  // std::lock_guard<std::mutex> _(cvm_global_mtx);
  if (input_data == nullptr) {
    // std::cerr << "input_data error" << std::endl;
    return -1;
  }
  if (output_data == nullptr) {
    // std::cerr << "output error" << std::endl;
    return -1;
  }
  CVMModel* model = (CVMModel*)model_;
  if (model == nullptr) {
    // std::cerr << "model error" << std::endl;
    return -1;
  }
  DLTensor* input = model->PlanInput(input_data);
  DLTensor* output = model->PlanOutput();
  if (input == nullptr || output == nullptr) {
      // std::cerr << "input == nullptr || output == nullptr" << std::endl;
      return -1;
  }
  // std::cerr << "run start\n";
  int ret = model->Run(input, output);
  // std::cerr << "run end\n";
  // std::cerr << "save start\n";
  if (ret == 0) {
      model->SaveTensor(output, output_data);
  }
  // std::cerr << "save end\n";
  if (input)
      CVMArrayFree(input);
  if (output)
      CVMArrayFree(output);
  return ret;
}

