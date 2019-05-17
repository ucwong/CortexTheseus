#include <cvm/c_api.h>


void* cvm_load_model(char *graph_fname, char *model_fname) {
  return _cvm_load_model(graph_fname, model_fname);
}

void cvm_free_model(void *model_) {
  _cvm_free_model(model_);
}

int cvm_get_output_length(void *model_) {
  return _cvm_get_output_length(model_);
}

int cvm_predict(void *model_, char *input_data, char *output_data) {
  return _cvm_predict(model_, input_data, output_data);
}

