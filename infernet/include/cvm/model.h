/*!
 *  Copyright (c) 2017 by Contributors
 * \file module_util.h
 * \brief Helper utilities for module building
 */
#ifndef CVM_RUNTIME_CVMMODEL_H_
#define CVM_RUNTIME_CVMMODEL_H_

#include <cvm/dlpack.h>
#include <cvm/runtime/packed_func.h>

#include <string>

using std::string;

namespace cvm {
namespace runtime {

struct CVMModel {
public:
  bool loaded{false};
  DLContext ctx;
  CVMModel(const string& graph, DLContext _ctx);
  ~CVMModel();
  int LoadParams(string params_str);
  int LoadParamsFromFile(string filepath);
  int GetInputLength();
  int GetOutputLength();
  DLTensor* GetInputTensor(char* data);
  DLTensor* GetOutputTensor(char* data);

  int Run(DLTensor*& input, DLTensor*& output);
  DLTensor* Run();
  DLTensor* LoadRandomInput();
  DLTensor* PlanOutput();
private:
  int SetInput_(string index, DLTensor* input);
  int Run_();
  int GetOutput_(int index, DLTensor* output);
  int LoadParams(CVMByteArray& params);
  PackedFunc set_input;
  PackedFunc get_output;
  PackedFunc load_params;
  PackedFunc run;
  string params_str_;
  Module module;
  int64_t *in_shape, *out_shape;
  int in_ndim, out_ndim;
  int64_t in_size, out_size;
  int dtype_code{kDLInt};
  int dtype_bits{32};
  int dtype_lanes{1};
};

}
}

#endif // CVM_RUNTIME_CVMMODEL_H_
