#include "interface.h"
#include <cvm/c_api.h>


void *load_model(char *graph_fname, char *model_fname) {
  return cvm_load_model(graph_fname, model_fname);
}

void free_model(void *model_) {
  cvm_free_model(model_);
}

int get_output_length(void *model_) {
  return cvm_get_output_length(model_);
}

int predict(void *model_, char *input_data, char *output_data) {
  return cvm_predict(model_, input_data, output_data);
}

