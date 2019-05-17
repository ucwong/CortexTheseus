/*!
 *  Copyright (c) 2017 by Contributors
 * \file module_util.h
 * \brief Helper utilities for module building
 */
#ifndef CVM_RUNTIME_INTERFACE_H_
#define CVM_RUNTIME_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

void *load_model(char *graph_fname, char *model_fname);
void free_model(void *model);
int get_output_length(void *model);
int predict(void *model, char *input_data, char *output_data);

#ifdef __cplusplus
}
#endif

#endif // CVM_RUNTIME_INTERFACE_H_
