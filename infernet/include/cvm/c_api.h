/*!
 *  Copyright (c) 2016 by Contributors
 * \file cvm/c_api.h
 * \brief C API of CVM symbolic construction and pass.
 *  Enables construction and transformation of Graph
 *  in any other host languages.
 */
#ifndef CVM_C_API_H_
#define CVM_C_API_H_

/*! \brief CVM_DLL prefix for windows */
#ifdef _WIN32
#ifdef CVM_EXPORTS
#define CVM_DLL __declspec(dllexport)
#else
#define CVM_DLL __declspec(dllimport)
#endif
#else
#define CVM_DLL __attribute__((visibility("default")))
#endif

/*! \brief manually define unsigned int */
typedef unsigned int nn_uint;

/*! \brief handle to a function that takes param and creates symbol */

#ifdef __cplusplus
extern "C" {
#endif

void* CVMAPILoadModel(const char *graph_fname, int  json_len, const char *model_fname, int mdoel_len, int device_type, int device_id);

void CVMAPIFreeModel(void* model);

int CVMAPIGetInputLength(void* model);

int CVMAPIGetOutputLength(void* model);

void CVMAPIGetVersion(void* model, char* version);

void CVMAPIGetPreprocessMethod(void* model, char* method);

int CVMAPIInfer(void* model, char *input_data, char *output_data);

int CVMAPIInferInt32(void* model, char *input_data, char *output_data);

int CVMAPISizeOfOutputType(void *model);

int CVMAPISizeOfInputType(void *model);

long long CVMAPIGetGasFromModel(void *model);

long long CVMAPIGetStorageSize(void *model);

long long CVMAPIGetGasFromGraphFile(char *graph_fname);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // CVM_C_API_H_
