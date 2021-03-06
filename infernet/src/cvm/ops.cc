#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>

#include <cvm/op.h>
#include <cvm/top/tensor.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "omp.h"
#include <immintrin.h>

#include "graph_runtime.h"
#include "nms.h"

#define CVM_PROFILING

namespace cvm {
namespace runtime {

double transpose_int8_avx256_transpose_cnt = 0;
double transpose_int8_avx256_gemm_cnt = 0;
double im2col_cnt = 0;
double cvm_op_cvm_shift_cnt = 0;
double cvm_op_clip_cnt = 0;
double cvm_op_dense_cnt = 0;
double cvm_op_maxpool_cnt = 0;
double cvm_op_broadcast_cnt = 0;
double cvm_op_concat_cnt = 0;
double cvm_op_upsampling_cnt = 0;
double cvm_op_inline_matmul_cnt = 0;
double cvm_op_elemwise_cnt = 0;
double cvm_op_chnwise_conv_cnt = 0;
double cvm_op_chnwise_conv1x1_cnt = 0;
double cvm_op_depthwise_conv_cnt = 0;
double cvm_op_depthwise_conv1x1_cnt = 0;

// #define CVM_PROFILING
// #define CVM_PRINT_OP_RESULT

inline uint64_t getSize(DLTensor *dlTensor){
  uint64_t size = 1;
  for(int i = 0; i < dlTensor->ndim; i++){
      size *= dlTensor->shape[i];
  }
  return size;
}

const std::string DIR = "/tmp/zkh/random_3_1/";
void print_to_file(DLTensor *y, std::string filename){
#if defined(CVM_PRINT_OP_RESULT)
  FILE *fp = fopen((DIR + filename).c_str(), "a+");
  int32_t *y_data = static_cast<int32_t*>(y->data);

  int32_t min = y_data[0], max= y_data[0];
  for(uint64_t i = 0; i < getSize(y); i++){
      min = min > y_data[i] ? y_data[i] : min;
      max = max < y_data[i] ? y_data[i] : max;
  }
  fprintf(fp, "%d %d\n", min, max);
  for(uint64_t i = 0; i < 20 && i < getSize(y); i++){
      fprintf(fp, "%d ", y_data[i]);
  }
  fprintf(fp, "\n");
  fclose(fp);
#endif
}

/**
* x
* y
* a_min -127
* a_max 127
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.clip").set_body([](CVMArgs args, CVMRetValue* rv)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
   VERIFY(args.num_args == 3);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   void *_attr = args[2];
   auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
   auto& param = cvm::get<cvm::top::ClipParam>(attr->parsed);
   int max = param.a_max;
   int min = param.a_min;
   for (uint64_t i = 0; i < getSize(x); i++) {
    static_cast<int32_t*>(y->data)[i] = std::max(std::min(max, static_cast<int32_t*>(x->data)[i]), min);
   }
#ifdef CVM_PROFILING
    cvm_op_elemwise_cnt += omp_get_wtime() - start;
#endif
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.relu").set_body([](CVMArgs args, CVMRetValue* rv)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
   VERIFY(args.num_args == 3);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
#pragma omp parallel for
   for (uint64_t i = 0; i < getSize(x); i++) {
        auto tmp = static_cast<int32_t*>(x->data)[i];
        if (tmp < 0)
            tmp = 0;
    static_cast<int32_t*>(y->data)[i] = tmp;
   }
#ifdef CVM_PROFILING
    cvm_op_elemwise_cnt += omp_get_wtime() - start;
#endif
  print_to_file(y, "relu.txt");
});

/*
* x
* w
* b
* y
* units 1000
* use_bias True
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.dense").set_body([](CVMArgs args, CVMRetValue* rv) {
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  int ndim = args.num_args;
  VERIFY(ndim == 5 || ndim == 4);
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *b = nullptr;
  DLTensor *y = nullptr;
  int32_t* db = nullptr;
  if(ndim == 5){
    b = args[2];
    VERIFY(b->ndim == 1) << "dense requires 1-D bias";
    y = args[3];
    db = static_cast<int32_t*>(b->data);
  } else{
    y = args[2];
  }
  VERIFY(x->ndim == 2) << "dense requires 2-D data";
  VERIFY(w->ndim == 2) << "dense reuqires 2-D weight";

  auto dx = static_cast<int32_t*>(x->data);
  auto dy = static_cast<int32_t*>(y->data);
  auto dw = static_cast<int32_t*>(w->data);
  if (true) {

  for (uint32_t di = 0; di < y->shape[0]; di++) {
      for (uint32_t oi = 0; oi < y->shape[1]; oi++) {
          int32_t sum = 0;
          for (uint32_t xi = 0; xi < x->shape[1]; xi++) {
              sum += dx[di * x->shape[1] + xi] * dw[oi * w->shape[1] + xi];
          }
          if(db != nullptr){
              sum += db[oi];
          }
          dy[di * y->shape[1] + oi] = sum;
      }
  }
  print_to_file(y, "dense.txt");

  }  else {
  auto N = y->shape[1], K = x->shape[1];
  int blocks = K / 32 * 32;
  // std::cerr << y->shape[0] << " " << y->shape[1] << "\n";
  // std::cerr << x->shape[0] << " " << x->shape[1] << "\n";
  // std::cerr << w->shape[0] << " " << w->shape[1] << "\n";
  int32_t weight_size = w->shape[0] * w->shape[1];
  std::unique_ptr<int8_t> int8_filter(new int8_t[sizeof(int8_t) * weight_size]);
  if(!int8_filter) {
      CHECK(false) << "create buffer int8_filter failed";
  }

  for(int32_t i = 0; i < weight_size; i++){
    int8_filter.get()[i] = static_cast<int8_t>(dw[i]);
  }

  int32_t x_size = x->shape[0] * x->shape[1];
  std::unique_ptr<int8_t> int8_x(new int8_t[sizeof(int8_t) * x_size]);
  if(!int8_x) {
      CHECK(false) << "create buffer int8_x failed";
  }
  bool all_positive = true;
  for(int32_t i = 0; i < x_size; i++){
    int8_x.get()[i] = static_cast<int8_t>(dx[i]);
    if ((int8_x.get()[i]) < 0)
      all_positive = false;
  }
  // std::cerr << "all_positive = " << all_positive << "\n";

  int16_t int16[16];
  for(int i = 0; i < 16; i++)
      int16[i] = 1;
  __m256i vint16 = _mm256_loadu_si256((__m256i*)&int16);
  // std::cerr << " batch = " << y->shape[0] << "\n";
  for (uint32_t di = 0; di < y->shape[0]; di++) {
      auto cdy = dy + di * N;
      auto data_outer = int8_x.get() + di * K;
      #pragma omp parallel for
      for (uint32_t oi = 0; oi < N; oi++) {
          auto bp_inner = int8_filter.get() + oi * K;
          auto data_inner = data_outer;
          int sum = 0;

          int k = 0;
          if (all_positive) {
              __m256i vc = _mm256_setzero_si256();
              for(k = 0; k < blocks; k+=32, data_inner+=32, bp_inner+=32){
                  __m256i v_weight = _mm256_loadu_si256((__m256i*)bp_inner);
                  __m256i v_data = _mm256_loadu_si256((__m256i*)data_inner);
                  __m256i vresult1 = _mm256_maddubs_epi16(v_data, v_weight);
                  __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
                  vc = _mm256_add_epi32(vresult2, vc);
              }
              for(uint32_t ti = 0; ti < 8; ti++){
                  sum += ((int32_t*)&vc)[ti];
              }
          }

          // remained part
          for(; k < K; k++){
              sum += data_inner[k] * bp_inner[k];
          }
          if(db != nullptr){
              sum += db[oi];
          }
          cdy[oi] = sum;
      }
  }
  }

#ifdef CVM_PROFILING
  cvm_op_dense_cnt += omp_get_wtime() - start;
#endif
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.flatten")
    .set_body([](CVMArgs args, CVMRetValue* rv)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
     VERIFY(args.num_args == 3);
     DLTensor *x = args[0];
     DLTensor *y = args[1];
     for (uint64_t i = 0; i < getSize(x); i++) {
         static_cast<int32_t*>(y->data)[i] = static_cast<int32_t*>(x->data)[i];
     }
#ifdef CVM_PROFILING
    cvm_op_elemwise_cnt += omp_get_wtime() - start;
#endif

  print_to_file(y, "flatten.txt");
});

bool transpose_int8_avx256(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
    int8_t *tr_b = (int8_t*)malloc(sizeof(int8_t) * K*N);
    if (tr_b == NULL) {
      return false;
    }

    int i = 0, j = 0;
    const int32_t tK = K / 32 * 32;
    const int32_t tN = N / 32 * 32;
    for(i = 0; i < tK; i+=32){
        for(j = 0; j < tN; j+=32){
            int8_t tile[32][32];
            for(int ti = 0; ti < 32; ti++){
                for(int tj = 0; tj < 32; tj++){
                    tile[tj][ti] = b[(i+ti)*N + j+tj];
                }
            }
            for(int ti = 0; ti < 32; ti++){
                for(int tj = 0; tj < 32; tj++){
                    tr_b[(j+ti) * K + i + tj] = tile[ti][tj];
                }
            }
        }
        for(int ti = 0; ti < 32; ti++){
            for(int tj = j; tj < N; tj++){
                tr_b[tj * K + i+ti] = b[(i+ti) * N + tj];
            }
        }
    }
    for(; i < K; i++){
        for(j = 0; j < N; j++){
            tr_b[j * K + i] = b[i * N + j];
        }
    }
#ifdef CVM_PROFILING
    transpose_int8_avx256_transpose_cnt += omp_get_wtime() - start;
    start = omp_get_wtime();
#endif
    int16_t int16[16];
    for(int i = 0; i < 16; i++) int16[i] = 1;
    __m256i vint16 = _mm256_loadu_si256((__m256i*)&int16);
    int8_t ap [32], bp[32];
    memset(ap, 0, sizeof(ap));
    memset(bp, 0, sizeof(bp));

    int blocks = K / 32 * 32;
    if (K % 32 == 0) {
      #pragma omp parallel for
      for(int i = 0; i < M; i++){
        int32_t bV = bias != NULL ? bias[i] : 0;
        for(int j = 0; j < N; j++){
          __m256i vc = _mm256_setzero_si256();
          int k = 0;
          auto ap_inner = a + i * K;
          auto bp_inner = tr_b + j * K;
          for(k = 0; k < blocks; k+=32, ap_inner+=32, bp_inner+=32){
            __m256i va = _mm256_loadu_si256((__m256i*)(ap_inner));
            __m256i vb = _mm256_loadu_si256((__m256i*)bp_inner);
            __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
            __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
            vc = _mm256_add_epi32(vresult2, vc);
          }
          int sum = 0;
          for(int ti = 0; ti < 8; ti++){
            sum += ((int32_t*)&vc)[ti];
          }
          c[i*N+j] = sum + bV;
        }
      }
    } else {
      for(int i = 0; i < M; i++){
        int32_t bV = bias != NULL ? bias[i] : 0;
        for(int j = 0; j < N; j++){
          __m256i vc = _mm256_setzero_si256();
          int k = 0;
          auto ap_inner = a + i * K;
          auto bp_inner = tr_b + j * K;
          for(k = 0; k < blocks; k+=32, ap_inner+=32, bp_inner+=32){
            __m256i va = _mm256_loadu_si256((__m256i*)(ap_inner));
            __m256i vb = _mm256_loadu_si256((__m256i*)bp_inner);
            __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
            __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
            vc = _mm256_add_epi32(vresult2, vc);

          }
          if (K % 32 != 0) {
            memcpy(ap, ap_inner, sizeof(int8_t) * (K - k));
            memcpy(bp, bp_inner, sizeof(int8_t) * (K - k));
            {
              __m256i va = _mm256_loadu_si256((__m256i*)ap);
              __m256i vb = _mm256_loadu_si256((__m256i*)bp);
              __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
              __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
              vc = _mm256_add_epi32(vresult2, vc);
            }
            k = K;
          }
          int sum = 0;
          for(int ti = 0; ti < 8; ti++){
            sum += ((int32_t*)&vc)[ti];
          }
          c[i*N+j] = sum + bV;
        }
      }

    }

    free(tr_b);
#ifdef CVM_PROFILING
    double et = omp_get_wtime() - start;
    transpose_int8_avx256_gemm_cnt += et;
#endif
    return true;
}

void transpose(const int8_t *A, int8_t *B, int K, int N) {
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < K; k++) {
            B[i * K + k] = A[k * N + i];
        }
    }
}

void matrix_mul(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N, int algo = 0)
{
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    if(std::memset(c, 0, sizeof(int32_t) * M * N) == NULL){
        CHECK(false);
    }

    if (N > M ) {
#pragma omp parallel for
        for(int i = 0; i < M; i++){
            for(int k = 0; k < K; k++){
                int32_t aV = static_cast<int32_t>(a[i * K + k]);
                for(int j = 0; j < N; j++){
                    c[i*N + j] += aV * static_cast<int32_t>(b[k*N + j]);
                }
            }
        }
    } else {
      std::vector<int8_t> tr_b;
      try{
        tr_b.resize(N*K);
      }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
      }

      transpose(b, tr_b.data(), K, N);
      #pragma omp parallel
      {
          int i, j, k;
          #pragma omp for
          for (i = 0; i < M; i++) {
              auto ap = a + i * K;
              for (j = 0; j < N; j++) {
                  int32_t dot = 0;
                  auto tr_bp = tr_b.data() + j * K;
                  for (k = 0; k < K; k++) {
                      dot += ap[k] * static_cast<int32_t>(tr_bp[k]);
                  }
                  c[i*N + j] = dot;
              }
          }
      }
    }

    if(bias != NULL){
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                c[i*N+j] += bias[i];
            }
        }
    }
#ifdef CVM_PROFILING
    double cost_time = omp_get_wtime() - start;
    // std::cerr << "matrix_mul = " << M << " " << K << " " << N << " " << M * K * N << "  " << cost_time << "\n";
    cvm_op_inline_matmul_cnt += cost_time;
#endif
}
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
void im2col_cpu(const int32_t* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    int8_t* data_col, bool &has_negetive)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  // auto data_col_init = data_col;
  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                int32_t tv = data_im[input_row * width + input_col];
                if(tv < 0) {
                  has_negetive = true;
                }
                *(data_col++) = static_cast<int8_t>(tv);
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
#ifdef CVM_PROFILING
  im2col_cnt +=  omp_get_wtime() - start;
#endif
}

void depthwise_conv2d(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups)
{
  // TODO(kaihuo) optimize cpu's depthwise conv efficiency, e.g. using modified im2col
  for(int n = 0; n < n_batch; ++n){
    for(int c = 0; c < in_channels; ++c){
      for(int h = 0; h < o_h; ++h){
        for(int w = 0; w < o_w; ++w){
          int32_t sum = 0;
          for(int fh = 0; fh < filter_h; ++fh){
            for(int fw = 0; fw < filter_w; ++fw){
                int th = h * stride_h + fh*dilation_h - padding[0];
                int tw = w * stride_w + fw*dilation_w - padding[1];
                if(th < 0 || tw < 0 || th >= x_h || tw >= x_w)
                    continue;
                sum += x_data[n * in_channels * x_h * x_w + c * x_h * x_w + th * x_w + tw]
                    * w_data[c * filter_h * filter_w + fh * filter_w + fw];
            }
          }
          y_data[n * in_channels * o_h * o_w + c * o_h * o_w + h * o_w + w] = sum + (b_data != nullptr ? b_data[c] : 0);
        }
      }
    }
  }
}

void depthwise_conv2d_single(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups)
{
  // std::cerr << "depth wise imcol\n";
  int32_t fn = out_channels * filter_h * filter_w;
  int8_t *int8_filter = (int8_t*)malloc(sizeof(int8_t) * fn);
  if(int8_filter == NULL){
    CHECK(false);
  }
  for(int32_t i = 0; i < fn; i++){
    int8_filter[i] = static_cast<int8_t>(w_data[i]);
  }
  const int M = 1;
  const int K = filter_h * filter_w;
  const int N = o_h * o_w;

  int8_t *data_col = (int8_t*)malloc(sizeof(int8_t) * in_channels * filter_h * filter_w * o_h * o_w);
  if(data_col == NULL){
    delete int8_filter;
    CHECK(false) << "malloc failed when alloc " << data_col;
  }
  bool has_negetive = false;
  im2col_cpu(
    x_data + 0* in_channels * x_h * x_w, //+ channel * x_h * x_w,
    in_channels, x_h, x_w,
    filter_h, filter_w,
    padding[0], padding[1],
    stride_h, stride_w,
    dilation_h, dilation_w,
    data_col, has_negetive
  );
  if(std::memset(y_data, 0, sizeof(int32_t) * in_channels * M * N) == NULL){
    CHECK(false);
  }
  for(int batch = 0; batch < n_batch; batch++) {
    auto y_data_batch = y_data + batch * in_channels * N;
    #pragma omp parallel for
    for (int channel = 0; channel < out_channels; channel++) {
      auto c = y_data_batch + channel * N;
      auto a = int8_filter + channel * K;
      auto b = data_col + channel * K * N;
      for(int k = 0; k < K; k++){
        int32_t aV = static_cast<int32_t>(a[k]);
        for(int j = 0; j < N; j++){
          c[j] += aV * static_cast<int32_t>(b[k*N + j]);
        }
      }
      if (b_data) {
        for(int j = 0; j < N; j++){
          c[j] += b_data[channel];
        }
      }
    }
  }
  free(data_col);
  free(int8_filter);
}
/*
input
weight
bias
output
groups 1
dilation (1, 1)
channels 512
layout NCHW
kernel_layout OIHW
kernel_size [1, 1]
padding (0, 0)
use_bias True
strides (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.conv2d")
    .set_body([](CVMArgs args, CVMRetValue* rv)
{
  VERIFY(args.num_args == 4 || args.num_args == 5);
  DLTensor *x = args[0];
  VERIFY(x->ndim == 4);
  DLTensor *w = args[1];
  VERIFY(w->ndim == 4);
  DLTensor *b = nullptr; //args[2];
  DLTensor *y = nullptr;
  void *_attr;

  if(args.num_args == 5){
    b = args[2];
    y = args[3];
    _attr = args[4];
  } else {
    y = args[2];
    _attr = args[3];
  }
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::Conv2DParam>(attr->parsed);
  int groups = param.groups;
  int dilation[2] = {(int)param.dilation[0], (int)param.dilation[1]};
    //TODO(@kaihuo) check kernel_size == w->shape
  // int kernel_size[2] = {(int)param.kernel_size[0], (int)param.kernel_size[1]};
  int padding[2] = {(int)param.padding[0], (int)param.padding[1]};
  int strides[2] = {(int)param.strides[0], (int)param.strides[1]};

  int stride_h = strides[0];
  int stride_w = strides[1];
  int dilation_h = dilation[0];
  int dilation_w = dilation[1];

  int32_t* x_data = (int32_t*)x->data;
  int32_t* w_data = (int32_t*)w->data;
  int32_t* y_data = (int32_t*)y->data;
  int32_t* b_data = b != nullptr ? (int32_t*)b->data : nullptr;

  int out_channels = static_cast<int>(w->shape[0]);
  int filter_c = static_cast<int>(w->shape[1]);
  int filter_h = static_cast<int>(w->shape[2]);
  int filter_w = static_cast<int>(w->shape[3]);
  int t_filter_h = (filter_h - 1) * dilation[0] + 1;
  int t_filter_w = (filter_w - 1) * dilation[1] + 1;

  int n_batch = static_cast<int>(x->shape[0]);
  int in_channels = static_cast<int>(x->shape[1]);
  int x_h = static_cast<int>(x->shape[2]);
  int x_w = static_cast<int>(x->shape[3]);
  int o_h = (x_h + 2 * padding[0] - t_filter_h) / strides[0] + 1;
  int o_w = (x_w + 2 * padding[1] - t_filter_w) / strides[1] + 1;
  if(n_batch < 1 || in_channels < 1 || x_h < 1 || x_w < 1 || filter_c < 1 || filter_h < 1 || filter_w < 1 ||
          padding[0] < 0 || padding[1] < 0 || stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1 ||
           out_channels < 1 || o_h < 1 || o_w < 1)
  {
      VERIFY(false) << "error args of conv2d";
  }

  if(groups > 1){
    VERIFY(groups == in_channels && groups == out_channels)
      << "only support depthwise conv with groups = channels"
      << "Got: " << groups << " " << in_channels << " " << out_channels << "\n";
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
    depthwise_conv2d(
        x_data, n_batch, in_channels, x_h, x_w,
        w_data, filter_c, filter_h, filter_w,
        y_data, out_channels, o_h, o_w,
        b_data,
        padding, stride_h, stride_w, dilation[0], dilation[1],
        groups);
#ifdef CVM_PROFILING
    cvm_op_depthwise_conv_cnt += omp_get_wtime() - start;
#endif
    } else {
#ifdef CVM_PROFILING
      double start = omp_get_wtime();
      //double start_1x1 = omp_get_wtime();
#endif
      int8_t *data_col = (int8_t*)malloc(sizeof(int8_t) * in_channels * filter_h * filter_w * o_h * o_w);
      if(data_col == NULL){
          CHECK(false) << "malloc failed.";
      }
      int32_t fn = out_channels * in_channels * filter_h * filter_w;
      int8_t *int8_filter = (int8_t*)malloc(sizeof(int8_t) * fn);
      if(int8_filter == NULL){
          free(data_col);
          CHECK(false);
      }

      for(int32_t i = 0; i < fn; i++){
          int8_filter[i] = static_cast<int8_t>(w_data[i]);
      }
      for(int i = 0; i < n_batch; i++){
          bool has_negetive = false;
          im2col_cpu(x_data + i * in_channels * x_h * x_w, in_channels, x_h, x_w, filter_h, filter_w, padding[0], padding[1],
                  stride_h, stride_w, dilation_h, dilation_w, data_col, has_negetive);
          const int M = out_channels;
          const int K = in_channels * filter_h * filter_w;
          const int N = o_h * o_w;
          if(has_negetive) {
              matrix_mul(int8_filter, data_col, b_data, y_data + i * out_channels * o_h * o_w,
                  M, K, N);
          }else{
              bool ret = transpose_int8_avx256(int8_filter, data_col, b_data, y_data + i * out_channels * o_h * o_w,
                  M, K, N);
              if(ret == false){
                free(data_col);
                free(int8_filter);
                CHECK(false);
              }
          }
      }
      free(data_col);
      free(int8_filter);
#ifdef CVM_PROFILING
        cvm_op_chnwise_conv_cnt += omp_get_wtime() - start;
        if (filter_h == 1 && filter_w == 1) {
          cvm_op_chnwise_conv1x1_cnt += omp_get_wtime() - start;
        }
#endif
    }
  print_to_file(y, "conv2d.txt");
});

inline int32_t broadcast_i_index(int64_t* oshape, uint64_t o_index, int64_t* ishape, int idim, int odim){
    if(idim == 1 && ishape[0] == 1) return 0;
    uint64_t index = 0;
    uint64_t allIndex = 1;
    for(int i = 0; i < idim; i++){
        int idx = idim - 1 - i;
        int ovar = o_index % oshape[idx+odim-idim];
        if(ovar < ishape[idx]){
            index += allIndex * ovar;
        }
        allIndex =  allIndex * ishape[idx];
        o_index /= oshape[idx + odim-idim];
    }
    return index;
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_add")
    .set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        if(getSize(args1) == 1){
            for(uint64_t i = 0; i < getSize(args2); ++i){
                c[i] = a[i] + b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args2); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                int64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim, args2->ndim);
                int64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim, args2->ndim);
                c[i] = a[a_index] + b[b_index];
            }
        }
#ifdef CVM_PROFILING
        cvm_op_broadcast_cnt += omp_get_wtime() - start;
#endif
        print_to_file(args2, "broadcast_add.txt");
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_sub")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        if(getSize(args1) == 1){
            for(uint64_t i = 0; i < getSize(args2); ++i){
                c[i] = a[i] - b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args2); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim, args2->ndim);
                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim, args2->ndim);
                c[i] = a[a_index] - b[b_index];
            }
        }
#ifdef CVM_PROFILING
        cvm_op_broadcast_cnt += omp_get_wtime() - start;
#endif
        print_to_file(args2, "broadcast_sub.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_mul")
    .set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        if(getSize(args1) == 1){
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args2); ++i){
                c[i] = a[i] * b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args2); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim, args2->ndim);
                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim, args2->ndim);
                c[i] = a[a_index] * b[b_index];
            }
        }

#ifdef CVM_PROFILING
        cvm_op_broadcast_cnt += omp_get_wtime() - start;
#endif
    print_to_file(args2, "broadcast_mul.txt");
});

//CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_div")
//    .set_body([](CVMArgs args, CVMRetValue *ret){
//#ifdef CVM_PROFILING
//        double start = omp_get_wtime();
//#endif
//        VERIFY(args.num_args == 4);
//        DLTensor *args0 = args[0];
//        DLTensor *args1 = args[1];
//        DLTensor *args2 = args[2];
//        int32_t *a = static_cast<int32_t*>(args0->data);
//        int32_t *b = static_cast<int32_t*>(args1->data);
//        int32_t *c = static_cast<int32_t*>(args2->data);
//        if(args1->ndim == 1){
//            VERIFY(b[0] != 0);
//            for(uint64_t i = 0; i < getSize(args0); ++i){
//                c[i] = a[i] / b[0];
//            }
//        }else{
//#pragma omp parallel for
//            for(uint64_t i = 0; i < getSize(args0); ++i){
//                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
//                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
//                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
//                VERIFY(b[b_index] != 0);
//                c[i] = a[a_index] / b[b_index];
//            }
//        }
//#ifdef CVM_PROFILING
//        cvm_op_broadcast_cnt += omp_get_wtime() - start;
//#endif
//});
//
//CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_right_shift")
//    .set_body([](CVMArgs args, CVMRetValue *ret)
//{
//#ifdef CVM_PROFILING
//        double start = omp_get_wtime();
//#endif
//        VERIFY(args.num_args == 4);
//        DLTensor *args0 = args[0];
//        DLTensor *args1 = args[1];
//        DLTensor *args2 = args[2];
//        int32_t *a = static_cast<int32_t*>(args0->data);
//        int32_t *b = static_cast<int32_t*>(args1->data);
//        int32_t *c = static_cast<int32_t*>(args2->data);
//
//        if(args1->ndim == 1){
//            for(uint64_t i = 0; i < getSize(args0); ++i){
//                c[i] = a[i] >> b[0];
//            }
//        }else{
//#pragma omp parallel for
//            for(uint64_t i = 0; i < getSize(args0); ++i){
//                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
//                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
//                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
//                c[i] = a[a_index] >> b[b_index];
//            }
//        }
//#ifdef CVM_PROFILING
//        cvm_op_broadcast_cnt += omp_get_wtime() - start;
//#endif
//});
//
//CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_left_shift")
//    .set_body([](CVMArgs args, CVMRetValue *ret){
//        VERIFY(args.num_args == 4);
//        DLTensor *args0 = args[0];
//        DLTensor *args1 = args[1];
//        DLTensor *args2 = args[2];
//        int32_t *a = static_cast<int32_t*>(args0->data);
//        int32_t *b = static_cast<int32_t*>(args1->data);
//        int32_t *c = static_cast<int32_t*>(args2->data);
//        if(args1->ndim == 1){
//            for(uint64_t i = 0; i < getSize(args0); ++i){
//                c[i] = a[i] << b[0];
//            }
//        }else{
//#pragma omp parallel for
//            for(uint64_t i = 0; i < getSize(args0); ++i){
//                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
//                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
//                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
//                c[i] = a[a_index] << b[b_index];
//            }
//        }
//    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.max_pool2d")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
  // TODO(kaihuo) optimize cpu's maxpool efficiency, e.g. using modified im2col
  VERIFY(args.num_args == 3);
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
  TShape tpadding = param.padding;
  VERIFY(tpadding.ndim() == 1 || tpadding.ndim() == 2);
  int strides[2] = {(int)param.strides[0], (int)param.strides[1]};
  int pool_size[2] = {(int)param.pool_size[0], (int)param.pool_size[1]};
  int padding[2] = {(int)param.padding[0], (int)param.padding[0]};
  if(tpadding.ndim() == 2){
    padding[1] = (int)param.padding[1];
  }
//  bool ceil_mode = param.ceil_mode;

  int stride_h = strides[0];
  int stride_w = strides[1];

  int32_t* x_data = (int32_t*)x->data;
  int32_t* y_data = (int32_t*)y->data;

  int filter_h = pool_size[0];
  int filter_w = pool_size[1];

  int n_batch = static_cast<int>(x->shape[0]);
  int in_channels = static_cast<int>(x->shape[1]);
  int out_channels = in_channels;
  int x_h = static_cast<int>(x->shape[2]);
  int x_w = static_cast<int>(x->shape[3]);
  int o_h = static_cast<int>(y->shape[2]);
  int o_w = static_cast<int>(y->shape[3]);
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
  auto calc_func = [&](int n, int k, int p, int q) {
    int y_sum = int(1)<<31;
    for (int r = 0; r < filter_h; ++r) {
      for (int s = 0; s < filter_w; ++s) {
        auto tp = p * stride_h + r - padding[0];
        auto tq = q * stride_w + s - padding[1];
        int32_t x_tmp = 0;
        if (!(tp < 0 || tq < 0 || tp >= x_h || tq >= x_w))
          x_tmp = GETX(n, k, tp, tq);
        y_sum = std::max(x_tmp, y_sum);
      }
    }
    return y_sum;

  };
    for (int n = 0; n < n_batch; ++n) {
        for (int k = 0; k < out_channels; ++k) {
            for (int p = 0; p < o_h; ++p) {
                for (int q = 0; q < o_w; ++q) {
                    GETY(n, k, p, q) = calc_func(n, k, p, q);
                }
            }
        }
    }
#ifdef CVM_PROFILING
        cvm_op_maxpool_cnt += omp_get_wtime() - start;
#endif

});

/*
* axis (2, 3)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.sum")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  VERIFY(args.num_args == 3);
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
  TShape axis = param.axis;
  int64_t *axis_data = axis.begin();
  //bool keepdims = param.keepdims; //the reduce axis is always 1
  bool exclude = param.exclude;
  int32_t *x_data = static_cast<int32_t*>(x->data);
  int32_t *y_data = static_cast<int32_t*>(y->data);

  for(size_t i = 0; i < axis.ndim(); i++){
    if(axis_data[i] < 0) axis_data[i] += x->ndim;
      VERIFY(axis_data[i] >= 0 && axis_data[i] < x->ndim);
  }
  std::vector<int64_t> raxis;
  try{
  if(!exclude){
    for(size_t i = 0; i < axis.ndim(); i++){
      raxis.push_back(axis[i]);
    }
  }else{
    raxis.resize(x->ndim - axis.ndim());
    for(int i = 0, k = 0; i < x->ndim; i++){
      bool flag = false;
      for(size_t j = 0; j < axis.ndim(); j++){
        if(axis_data[j] == i) {
          flag = true;
          break;
        }
      }
      if(!flag){
        raxis[k++] = i;
      }
    }
  }
  }catch(const std::bad_alloc& e){
    CHECK(false) << e.what();
  }

  if(exclude && raxis.size() == 0){
    if(NULL == memcpy(y_data, x_data, getSize(x) * sizeof(int32_t))){
        CHECK(false);
    }
  }
  else if(raxis.size() == 0){
    int32_t sum = 0;
    for(uint64_t i = 0; i < getSize(x); i++){
      sum += x_data[i];
    }
    y_data[0] = sum;
  }else{
    //std::vector<int32_t> realAxis(axis.ndim(), 0);
    //for(uint32_t i = 0; i < axis.ndim(); i++){
    //  int32_t val = axis_data[i];
    //  if(val < 0) val += x->ndim;
    //  VERIFY(val < x->ndim && val >= 0);
    //  realAxis[val] = 1;
    //}
    //memset(y_data, 0, getSize(y)*sizeof(int32_t));
    //for(uint64_t i = 0; i < getSize(x); i++){
    //  uint64_t in_i = i, o_i = 0, shapeSize = 1;
    //  for(int j = x->ndim-1, yj = y->ndim-1; j>=0; j--){
    //    uint64_t col = in_i % x->shape[j];
    //    in_i /= x->shape[j];
    //    if(realAxis[j] == 0){
    //      o_i += col * shapeSize;
    //      shapeSize *= y->shape[yj];
    //      yj -= 1;
    //    }
    //  }
    //  y_data[o_i] += x_data[i];
    //}
    try{
      std::vector<int32_t> realAxis(raxis.size());
      std::vector<bool> flag(x->ndim, false);
      for(uint32_t i = 0; i < raxis.size(); i++){
        int32_t val = raxis[i];
        realAxis[i] = val;
        flag[val] = true;
      }
      std::sort(realAxis.begin(), realAxis.end());
      realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

      uint64_t axis_size = 1;
      for(uint32_t i = 0; i < realAxis.size(); i++){
        axis_size *= x->shape[realAxis[i]];
      }
      std::vector<uint64_t> every_xdim_size(x->ndim, 1);
      for(int i = x->ndim-2; i >= 0; i--){
        every_xdim_size[i] = x->shape[i+1] * every_xdim_size[i+1];
      }

      int32_t yndim = y->ndim;
      std::vector<int64_t> yshape(y->ndim);
      for(int i = 0; i < y->ndim; i++){
        yshape[i] = y->shape[i];
      }
      for(int i = 0, j = 0; i < y->ndim; i++){
        if(y->shape[i] == 1) {
          yndim -= 1;
        }else{
          yshape[j++] = y->shape[i];
        }
      }
      for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t in_i = 0, o_i = i;
        for(int j = yndim-1, xj = x->ndim-1; j>=0; j--){
          uint64_t col = o_i % yshape[j];
          o_i /= yshape[j];
          while(xj >= 0 && flag[xj--]);
          in_i += col * every_xdim_size[xj+1];
        }
        int32_t sum = 0;
        for(uint64_t xi = 0; xi < axis_size; xi++){
          uint64_t o_i = xi, tmp_in_i = 0;
          for(int j = realAxis.size() - 1; j>=0; j--){
            uint64_t col = o_i % x->shape[realAxis[j]];
            o_i /= x->shape[realAxis[j]];
            tmp_in_i += col * every_xdim_size[realAxis[j]];
          }
          sum += x_data[in_i + tmp_in_i];
        }
        y_data[i] = sum;
      }
    }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
    }
  }
  print_to_file(y, "sum.txt");
});


CVM_REGISTER_GLOBAL("cvm.runtime.cvm.elemwise_add")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  VERIFY(args.num_args == 4);
  DLTensor *args0 = args[0];
  DLTensor *args1 = args[1];
  DLTensor *args2 = args[2];
  int32_t *a = static_cast<int32_t*>(args0->data);
  int32_t *b = static_cast<int32_t*>(args1->data);
  int32_t *c = static_cast<int32_t*>(args2->data);

#pragma omp parallel for
  for(uint64_t i = 0; i < getSize(args0); i++){
    c[i] = a[i] + b[i];
  }

#ifdef CVM_PROFILING
  cvm_op_elemwise_cnt += omp_get_wtime() - start;
#endif
  print_to_file(args2, "elemwise_add.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.elemwise_sub")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  VERIFY(args.num_args == 4);
  DLTensor *args0 = args[0];
  DLTensor *args1 = args[1];
  DLTensor *args2 = args[2];
  int32_t *a = static_cast<int32_t*>(args0->data);
  int32_t *b = static_cast<int32_t*>(args1->data);
  int32_t *c = static_cast<int32_t*>(args2->data);

#pragma omp parallel for
  for(uint64_t i = 0; i < getSize(args0); i++){
      c[i] = a[i] - b[i];
  }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.reshape")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  VERIFY(args.num_args == 3);
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  // void *_attr = args[2];
  // auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  // auto &param = cvm::get<cvm::top::ReshapeParam>(attr->parsed);
  if(x->data == y->data) return;
  std::memcpy(y->data, x->data, getSize(x) * sizeof(int32_t));
  print_to_file(y, "reshape.txt");
});

/*\brief:
 * x, input data
 * y, output data
 * precision, clip precision
 */
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_clip")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  VERIFY(args.num_args == 3);
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  int32_t *x_data = static_cast<int32_t*>(x->data);
  int32_t *y_data = static_cast<int32_t*>(y->data);

  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::CVMClipParam>(attr->parsed);
  int32_t precision = param.precision;
  VERIFY(precision > 0) << "precision must greater zero";
  int32_t min = -((1 << (precision-1))-1);
  int32_t max = -min;

#pragma omp parallel for
  for(uint64_t i = 0; i < getSize(x); i++){
  int& tmp = x_data[i];
  if (tmp > max)
    tmp = max;
  if (tmp < min)
    tmp = min;
  y_data[i] = tmp;
  }
#ifdef CVM_PROFILING
  cvm_op_clip_cnt += omp_get_wtime() - start;
#endif
  print_to_file(y, "clip.txt");
}
);

/*
 * a, input data
 * c, output data
 * precision, clip precision
 * b, shift b
 * */
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_right_shift")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *a = args[0];
    DLTensor *c = args[1];

#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::CVMRightShiftParam>(attr->parsed);
    int32_t precision = param.precision;
    int32_t b = param.shift_bit;
    int32_t* a_data = static_cast<int32_t*>(a->data);
    int32_t* c_data = static_cast<int32_t*>(c->data);
    VERIFY_GT(precision, 0) << "precision must greater zero";
    int32_t min = -((1 << (precision-1)) - 1);
    int32_t max = -min;
    auto size = getSize(a);

    if (b == 0) {
      memcpy(c_data, a_data, size * sizeof(int32_t));
    } else if (b == 1) {
#pragma omp parallel for
      for(uint64_t i = 0; i < size; i++){
        int32_t shift_a = (a_data[i] + 1) >> 1;
        if (shift_a > max)
          shift_a = max;
        if (shift_a < min)
          shift_a = min;
        c_data[i] = shift_a;
      }
    } else {
      b -= 1;
#pragma omp parallel
      {
#pragma omp for
        for(uint64_t i = 0; i < size; i++){
          c_data[i] = a_data[i] >> b;
          ++c_data[i];
          c_data[i] >>= 1;
        }
#pragma omp for
        for(uint64_t i = 0; i < size; i++){
          auto& shift_a = c_data[i];
          if (shift_a > max)
            shift_a = max;
          if (shift_a < min)
            shift_a = min;
          c_data[i] = shift_a;
        }
      }
    }

#ifdef CVM_PROFILING
    cvm_op_cvm_shift_cnt += omp_get_wtime() - start;
#endif
  print_to_file(c, "cvm_right_shift.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_left_shift")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
#ifdef CVM_PROFILING
    //double start = omp_get_wtime();
#endif
    DLTensor *a = args[0];
    DLTensor *c = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::CVMLeftShiftParam>(attr->parsed);
    int32_t precision = param.precision;
    int32_t b = param.shift_bit;std::string str_precision = args[2];
    int32_t* a_data = static_cast<int32_t*>(a->data);
    int32_t* c_data = static_cast<int32_t*>(c->data);
    VERIFY_GT(precision, 0) << "precision must greater zero";
    int32_t min = -((1 << (precision-1)) - 1);
    int32_t max = -min;

    for(uint64_t i = 0; i < getSize(a); i++){
    int32_t shift_a = a_data[i];
    if(b == 0) c_data[i] = shift_a;
    else {
      shift_a = a_data[i] << b;
      c_data[i] = std::max(std::min(shift_a, max), min);
    }
    }
#ifdef CVM_PROFILING
    // cvm_op_requant_cnt += omp_get_wtime() - start;
#endif
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.log2")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    //        std::string x_str = args[0];
    DLTensor *dlx = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t *x = static_cast<int32_t*>(dlx->data);
    VERIFY(x[0] != 0);
    for(int i = 0; i < 64; i++){
      int64_t tmp = (int64_t)1 << i;
      if(x[0] < tmp){
        y_data[0] = i;
        return;
      }
    }
    y_data[0] = 64;
});
//CVM_REGISTER_GLOBAL("cvm.runtime.cvm.__div_scalar__")
//    .set_body([](CVMArgs args, CVMRetValue *ret){
//        VERIFY(args.num_args == 3);
//        DLTensor *dlx = args[0];
//        DLTensor *y = args[1];
//        void *_attr = args[2];
//        auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
//        auto &param = cvm::get<cvm::top::ScalarParam>(attr->parsed);
//        int32_t *y_data = static_cast<int32_t*>(y->data);
//        int32_t scalar = param.scalar;
//        VERIFY(scalar != 0);
//        int32_t* x = static_cast<int32_t*>(dlx->data);
//        for(uint64_t i = 0; i < getSize(dlx); i++){
//            y_data[i] = x[i] / scalar;
//        }
//    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.abs")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *dlx = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t* x = static_cast<int32_t*>(dlx->data);
    for(uint64_t i = 0; i < getSize(dlx); i++){
      y_data[i] = std::abs(x[i]);
    }
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.max")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void* _attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
    TShape axis = param.axis;
    int64_t *axis_data = axis.begin();
    //bool keepdims = param.keepdims;
    bool exclude = param.exclude;

    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t* x_data = static_cast<int32_t*>(x->data);

    for(size_t i = 0; i < axis.ndim(); i++){
    if(axis_data[i] < 0) axis_data[i] += x->ndim;
    VERIFY(axis_data[i] >= 0 && axis_data[i] < x->ndim);
    }
    std::vector<int64_t> raxis;
    try{
      if(!exclude){
        for(size_t i = 0; i < axis.ndim(); i++){
          raxis.push_back(axis[i]);
        }
      }else{
        raxis.resize(x->ndim - axis.ndim());
        for(int i = 0, k = 0; i < x->ndim; i++){
          bool flag = false;
          for(size_t j = 0; j < axis.ndim(); j++){
            if(axis_data[j] == i) {
              flag = true;
              break;
            }
          }
          if(!flag){
            raxis[k++] = i;
          }
        }
      }
    }catch(const std::bad_alloc& e){
      CHECK(false) << e.what();
    }
    if(exclude && raxis.size() == 0){
      memcpy(y_data, x_data, getSize(x) * sizeof(int32_t));
    }
    else if(raxis.size() == 0){
      int32_t max = x_data[0];
      for(uint64_t i = 1; i < getSize(x); i++){
        if(max < x_data[i]) max = x_data[i];
      }
      y_data[0] = max;
    }else{
      try{
        std::vector<int32_t> realAxis(raxis.size());
        std::vector<bool> flag(x->ndim, false);
        for(uint32_t i = 0; i < raxis.size(); i++){
          int32_t val = raxis[i];
          realAxis[i] = val;
          flag[val] = true;
        }
        std::sort(realAxis.begin(), realAxis.end());
        realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

        uint64_t axis_size = 1;
        for(uint32_t i = 0; i < realAxis.size(); i++){
          axis_size *= x->shape[realAxis[i]];
        }
        std::vector<uint64_t> every_xdim_size(x->ndim, 1);
        for(int i = x->ndim-2; i >= 0; i--){
          every_xdim_size[i] = x->shape[i+1] * every_xdim_size[i+1];
        }
        int32_t yndim = y->ndim;
        std::vector<int64_t> yshape(y->ndim);
        for(int i = 0; i < y->ndim; i++){
          yshape[i] = y->shape[i];
        }
        for(int i = 0, j = 0; i < y->ndim; i++){
          if(y->shape[i] == 1) {
            yndim -= 1;
          }else{
            yshape[j++] = y->shape[i];
          }
        }
        for(uint64_t i = 0; i < getSize(y); i++){
          uint64_t in_i = 0, o_i = i;
          for(int j = yndim-1, xj = x->ndim-1; j>=0; j--){
            uint64_t col = o_i % yshape[j];
            o_i /= yshape[j];
            while(xj >= 0 && flag[xj--]);
            in_i += col * every_xdim_size[xj+1];
          }
          int32_t max = x_data[in_i];//(int32_t)1<<31;
          for(uint64_t xi = 0; xi < axis_size; xi++){
            uint64_t o_i = xi, tmp_in_i = 0;
            for(int j = realAxis.size() - 1; j>=0; j--){
              uint64_t col = o_i % x->shape[realAxis[j]];
              o_i /= x->shape[realAxis[j]];
              tmp_in_i += col * every_xdim_size[realAxis[j]];
            }
            if(max < x_data[in_i+tmp_in_i]) max = x_data[in_i+tmp_in_i];
          }
          y_data[i] = max;
        }
      }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
      }
    }
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_max")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 4);
    DLTensor *a = args[0];
    DLTensor *b = args[1];
    DLTensor *c = args[2];
    int32_t *a_data = static_cast<int32_t*>(a->data);
    int32_t* b_data = static_cast<int32_t*>(b->data);
    int32_t* c_data = static_cast<int32_t*>(c->data);
    if(getSize(b) == 1){
#pragma omp parallel for
      for(uint64_t i = 0; i < getSize(c); i++){
        c_data[i] = a_data[i] > b_data[0] ? a_data[i] : b_data[0];
      }
    }else{
#pragma omp parallel for
      for(uint64_t i = 0; i < getSize(c); i++){
        uint64_t o_index = i;//broadcast_o_index(c->shape, c->ndim, o_index);
        uint64_t a_index = broadcast_i_index(c->shape, o_index, a->shape, a->ndim, c->ndim);
        uint64_t b_index = broadcast_i_index(c->shape, o_index, b->shape, b->ndim, c->ndim);
        //c_data[i] = (a_data[i] > b_data[i] ? a_data[i] : b_data[i]);
        c_data[i] = a_data[a_index] > b_data[b_index] ? a_data[a_index] : b_data[b_index];
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.concatenate")
.set_body([](CVMArgs args, CVMRetValue *ret){

#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    int len = args.num_args;
    VERIFY(len >= 3);
    DLTensor *input0 = args[0];
    void *_attr = args[--len];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::ConcatenateParam>(attr->parsed);
    DLTensor *out = args[--len];
    int32_t axis = param.axis;
    int32_t ndim = static_cast<int32_t>(input0->ndim);
    VERIFY(-ndim <= axis && axis < ndim);
    if(axis < 0) axis += ndim;
    VERIFY(axis < input0->ndim && axis >= 0);
    int n_batch = input0->shape[0];
    if (axis == 1 && n_batch == 1) {
      int32_t *out_data = static_cast<int32_t*>(out->data);
      uint64_t offset = 0;
      for(int k = 0; k < len; k++){
        DLTensor* input = args[k];
        int input_size_current = 1;
        //std::cerr << "\n";
        for (int i = 0; i < input->ndim; ++i) {
          input_size_current *= input->shape[i];
          //    std::cerr << input->shape[i] << " " ;
        }
        //std::cerr << "\n";
        //std::cerr << "k = " << k << " " << input_size_current << "\n";
        memcpy(out_data + offset, input->data, sizeof(int32_t) * input_size_current);
        offset += input_size_current;
      }
    } else {
      int32_t *out_data = static_cast<int32_t*>(out->data);
      for(uint64_t i = 0; i < getSize(out); i++){
        uint64_t o_i = i, in_i = 0, in_i2 = 0, shapeSize = 0;
        for(int j = out->ndim-1; j >= 0; j--){
          uint64_t col = o_i % out->shape[j];
          o_i /= out->shape[j];
          uint64_t tmpcol = col;
          if(j == axis){
            uint64_t allShapeSize = 0;
            for(int k = 0; k < len; k++){
              tmpcol = col - allShapeSize;
              DLTensor *input = args[k];
              allShapeSize += input->shape[axis];
              if(col < allShapeSize){
                in_i = k;
                break;
              }
            }
          }
          in_i2 += (j == out->ndim-1 ? tmpcol : tmpcol * shapeSize);
          DLTensor* input = args[in_i];
          shapeSize = (j == out->ndim-1 ? input->shape[j] : shapeSize * input->shape[j]);
        }
        DLTensor *input = args[in_i];
        int32_t *input_data = static_cast<int32_t*>(input->data);
        out_data[i] = input_data[in_i2];
      }
    }
#ifdef CVM_PROFILING
    cvm_op_concat_cnt += omp_get_wtime() - start;
#endif
  print_to_file(out, "concatenate.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.repeat")
.set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::RepeatParam>(attr->parsed);
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t axis = param.axis;
    int32_t repeat = param.repeats;
    int32_t ndim = x->ndim;
    //printf("axis = %d, repeat = %d\n", axis, repeat);
    {
      if(axis < 0) axis = axis + ndim;
      VERIFY(axis >= 0 && axis < ndim);

      #pragma omp parallel for
      for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
          uint64_t col = o_i % y->shape[j];
          o_i /= y->shape[j];
          if(j == axis) col = col / repeat;
          in_i += (j == ndim-1 ? col : col * shapeSize);
          shapeSize = (j == ndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
        }
        y_data[i] = x_data[in_i];
      }
    }
#ifdef CVM_PROFILING
    double end = omp_get_wtime();
    static double use_time = 0.0;
    use_time += end-start;
#endif
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.negative")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    for(uint64_t i = 0; i < getSize(x); i++){
        y_data[i] = -x_data[i];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.tile")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void* _attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TileParam>(attr->parsed);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    int32_t xndim = x->ndim;
    TShape ts_reps = param.reps;
    int64_t *reps = ts_reps.begin();
    for(uint32_t i = 0; i < ts_reps.ndim(); i++){
        VERIFY(reps[i] > 0);
    }

    int i = 0, j = 0, k = 0;
    for(i = yndim-1, j = xndim-1, k = ts_reps.ndim()-1; i >= 0 && j >= 0 && k >= 0; i--, j--, k--){
        VERIFY(x->shape[j] * reps[k] == y->shape[i]);
    }
    for(; i >= 0 && k >= 0; i--, k--){
        VERIFY(reps[k] == y->shape[i]);
    }

    uint64_t tmp_y_size = 1;
    for(int i = 0; i < xndim; i++){
        tmp_y_size *= y->shape[i + yndim - xndim];
    }

    for(uint64_t i = 0; i < tmp_y_size; i++){
       uint64_t o_i = i, in_i = 0, shapeSize = 0;
       for(int j = xndim-1; j >= 0; j--){
            int yj = j + yndim - xndim;
            int col = o_i % y->shape[yj];
            o_i /= y->shape[yj];
            col = col % x->shape[j];
            in_i += (j == xndim-1 ? col : col * shapeSize);
            shapeSize = (j == xndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
       }
       y_data[i] = x_data[in_i];
    }

    uint64_t othery = 1;
    for(int i = 0; i < yndim-xndim; i++){
        othery *= y->shape[i];
    }
    for(size_t i = 1; i < othery; i++){
        memcpy(y_data + i*tmp_y_size, y_data, tmp_y_size * sizeof(int32_t));
    }
    print_to_file(y, "tile.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.expand_dims")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
    VERIFY(args.num_args == 3);
    DLTensor *ishape = args[0];
    DLTensor *oshape = args[1];
    //void *_attr = args[2];
    //auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    //auto &param = cvm::get<cvm::top::ExpandDimsParam>(attr->parsed);

   // int32_t axis = param.axis;
   // axis = axis < 0 ? axis + ishape->ndim : axis;
   // VERIFY(axis >= 0 && axis <= ishape->ndim) << axis << " ishape->dim: " << ishape->ndim;
    int32_t *ishape_data = static_cast<int32_t*>(ishape->data);
    int32_t *oshape_data = static_cast<int32_t*>(oshape->data);
    if(ishape_data == oshape_data){
        return;
    }
    memcpy(oshape_data, ishape_data, getSize(ishape)* sizeof(int32_t));
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.squeeze")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
    VERIFY(args.num_args == 3);
    DLTensor *ishape = args[0];
    DLTensor *oshape = args[1];
    // void *_attr = args[2];
    // auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    // auto &param = cvm::get<cvm::top::SqueezeParam>(attr->parsed);
    int32_t *ishape_data = static_cast<int32_t*>(ishape->data);
    int32_t *oshape_data = static_cast<int32_t*>(oshape->data);
    // std::cerr << ishape_data << " " << oshape_data << "\n";
    if(ishape_data == oshape_data){
        return;
    }
    memcpy(oshape_data, ishape_data, getSize(ishape)* sizeof(int32_t));
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.transpose")
.set_body([](CVMArgs args, CVMRetValue *ret){
    int num_args = args.num_args;
    VERIFY(num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TransposeParam>(attr->parsed);

    TShape axes = param.axes;
    int64_t *axes_data = axes.begin();
    VERIFY(axes.ndim() == 0 || axes.ndim() == (uint32_t)x->ndim);
    for(uint32_t i = 0; i < axes.ndim(); i++){
        if(axes_data[i] < 0) axes_data[i] += x->ndim;
        VERIFY(axes_data[i] >= 0 && axes_data[i] < x->ndim);
    }
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int ndim = y->ndim;
   // int mul_xj[8];
   // mul_xj[ndim] = 1;
   // for(int i = ndim - 1; i > 0; i--){
   //     mul_xj[i] = mul_xj[i + 1] * x->shape[i];
   // }
    if (axes.ndim() == 3 && axes[0] == 1 && axes[1] == 2 && axes[2] == 0) {
      int step = x->shape[1] * x->shape[2];
      for (int i = 0; i < step; i++) {
        for (int j = 0; j < x->shape[0]; j++) {
          y_data[i * x->shape[0]+ j ] = x_data[j * step + i];
        }
      }
    }
    else {
      for(uint64_t i = 0; i < getSize(y); i++) {
        uint64_t o_i = i, in_i = 0;
        for(int j = ndim - 1; j >= 0; j--){
          uint64_t col = o_i % y->shape[j];
          o_i /= y->shape[j];
          int xj = j;//axes != nullptr ? axes[j] : j;
          if(axes.ndim() > 0) {
            xj = axes_data[j];
          } else {
            xj = ndim - 1 - j;
          }
          int xi = 1;
          for(int tx = ndim-1; tx > xj; tx--){
            xi *= x->shape[tx];
          }
          in_i += col * xi;
          //in_i += col * mul_xj[xj + 1];
        }
        y_data[i] = x_data[in_i];
      }
    }
    print_to_file(x, "transpose_x.txt");
    print_to_file(y, "transpose.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.strided_slice")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::StridedSliceParam>(attr->parsed);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    TShape begin = param.begin;
    TShape end = param.end;
    TShape stride = param.stride;
    int ndim = y->ndim;
    int32_t num_axis = x->ndim;
    std::vector<int64_t> begin_vec;
    std::vector<int64_t> end_vec;
    std::vector<int64_t> stride_vec;
    try{
      std::copy(begin.begin(), begin.end(), std::back_inserter(begin_vec));
      for (dim_t i = begin_vec.size(); i < num_axis; ++i) {
        begin_vec.push_back(0);
      }

      std::copy(end.begin(), end.end(), std::back_inserter(end_vec));
      for (dim_t i = end_vec.size(); i < num_axis; ++i) {
        end_vec.push_back(x->shape[i]);
      }

      std::copy(stride.begin(), stride.end(), std::back_inserter(stride_vec));
      for (dim_t i = stride_vec.size(); i < num_axis; ++i) {
        stride_vec.push_back(1);
      }
    }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
    }

    int64_t *begin_data = begin_vec.data();//begin.begin();
    //int64_t *end_data = end_vec.data();//end.begin();
    int64_t *step_data = stride_vec.data();//stride.begin();
    for(uint32_t i = 0; i < begin.ndim();i++){
        if(begin_data[i] < 0) {
          begin_data[i] += x->shape[i];
          begin_data[i] = std::min(std::max(begin_data[i], (int64_t)0), (int64_t)x->shape[i]-1);
        }
      //  if(end_data[i] < 0) {
      //    end_data[i] += x->shape[i];
      //    end_data[i] += std::min(std::max(end_data[i], (int64_t)0), (int64_t)x->shape[i]-1);
      //  }
      //  if(stride.ndim() > 0){
      //    if(step_data[i] > 0) {
      //      VERIFY(begin_data[i] < end_data[i]);
      //    }else{
      //      VERIFY(begin_data[i] > end_data[i]);
      //    }
      //  }
    }

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            int64_t tbegin = begin.ndim() > (uint32_t)j ? begin_data[j] : 0;
            int64_t tstep = stride.ndim() > (uint32_t)j ? step_data[j] : 1;
            col = tbegin + col * tstep;
            in_i += (j == ndim-1 ? col : col * shapeSize);
            shapeSize = (j == ndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
        }
        y_data[i] = x_data[in_i];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.slice_like")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    //DLTensor *shape = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::SliceLikeParam>(attr->parsed);
    Tuple<int> axis = param.axis;
    //int *axis_data = axis.begin();

    int32_t *x_data = static_cast<int32_t*>(x->data);
    //  int32_t *shape_like = static_cast<int32_t*>(shape->data);
    //VERIFY(axis.ndim() < (uint32_t)x->ndim && axis.ndim() <= (uint32_t)shape->ndim);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int ndim = x->ndim;

    for(uint64_t i = 0; i < getSize(y); i++){
      uint64_t o_i = i, in_i = 0, shapeSize = 0;
      for(int j = ndim-1; j >= 0; j--){
        int col = o_i % y->shape[j];
        o_i /= y->shape[j];
        in_i += (j == ndim-1 ? col : col * shapeSize);
        shapeSize = (j == ndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
      }
      y_data[i] = x_data[in_i];
    }
});

/**
 * box_nms:
 */


CVM_REGISTER_GLOBAL("cvm.runtime.cvm.get_valid_counts")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::GetValidCountsParam>(attr->parsed);

    int32_t score_threshold = param.score_threshold; //TODO get from attr

    VERIFY(x->ndim == 3);
    int32_t batches = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];
    VERIFY(k >= 2);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    get_valid_count(x_data, y_data, valid_count_data, batches, n, k, score_threshold);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.non_max_suppression")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::NonMaximumSuppressionParam>(attr->parsed);

    int32_t max_output_size = param.max_output_size;
    int32_t iou_threshold = param.iou_threshold;
    int32_t topk = param.top_k;
    int32_t coord_start = param.coord_start;
    int32_t score_index = param.score_index;
    int32_t id_index = param.id_index;
    bool force_suppress = param.force_suppress;
    bool return_indices = param.return_indices;
    //bool invalid_to_bottom = param.invalid_to_bottom;
    CHECK(return_indices == false) << "no support return_indices and invalid_to_bottom";

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t batchs = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];

    int ret = non_max_suppression(
            x_data, valid_count_data, y_data, batchs, n, k,
            max_output_size, iou_threshold, topk, coord_start, score_index, id_index, force_suppress);
    VERIFY(ret >= 0);
});

//CVM_REGISTER_GLOBAL("cvm.runtime.cvm.bias_add")
//.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
//    DLTensor *x = args[0];
//    DLTensor *bias = args[1];
//    DLTensor *y = args[2];
//    int32_t axis=1; //TODO get from attr
//    int32_t ndim = x->ndim;
//    VERIFY(axis > 0 && axis < ndim);
//
//    const int32_t *x_data = static_cast<int32_t*>(x->data);
//    const int32_t *bias_data = static_cast<int32_t*>(bias->data);
//    int32_t *y_data = static_cast<int32_t*>(y->data);
//
//    for(uint64_t i = 0; i < getSize(y); i++){
//        int32_t bV = 0;
//        for(int32_t j = ndim - 1; j >= 0; j--){
//            if(j == axis){
//                bV = bias_data[axis];
//                break;
//            }
//        }
//        y_data[i] = x_data[i] + bV;
//    }
//});

void take(DLTensor *x, DLTensor *indices, DLTensor *y){
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    uint64_t xs = getSize(x);

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t in_i = std::min((uint64_t)std::max(indices_data[i], 0), xs-1);
        y_data[i] = x_data[in_i];
    }
}

void take(DLTensor *x, DLTensor *indices, DLTensor *y, const int32_t axis){
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    int32_t xndim = x->ndim;
    int32_t indices_ndim = indices->ndim;
    if (axis == 0 && xndim == 2 && yndim == 3) {
      const int K = x->shape[1];
      // std::cerr << "axis == 0 && xndim == 2 && yndim == 3" << "\n";
      uint64_t wn = getSize(indices);
      auto indices_data = static_cast<int32_t*>(indices->data);
      for (uint64_t row = 0; row < wn; row++) {
        uint64_t x_indices_i = std::min((int64_t)std::max(indices_data[row], 0), x->shape[0] - 1);
        memcpy(y_data +  row * K, x_data + x_indices_i * K, K * sizeof(int32_t));
      }
    }
    else {
      for(uint64_t i = 0; i < getSize(y); i++){
        //y_data[i] = x_data[indices_data[i]];
        uint64_t o_i = i, x_i = 0, indices_i = 0, x_shape_size = 0, indices_shape_size = 0;
        for(int32_t j = yndim - 1, k = indices_ndim-1; j>=axis; j--){
          uint64_t col = o_i % y->shape[j];
          o_i /= y->shape[j];
          if(j < axis + indices_ndim){
            indices_i += (indices_shape_size == 0 ? col : col * indices_shape_size);
            indices_shape_size = (indices_shape_size == 0 ? indices->shape[k]
                : indices_shape_size * indices->shape[k]);
            --k;
          }
        }

        o_i = i;
        int32_t k = xndim - 1;
        for(int32_t j = yndim - 1; j >= axis + indices_ndim; j--, k--){
          uint64_t col = o_i % y->shape[j];
          o_i /= y->shape[j];
          x_i += (j == yndim-1 ? col : col * x_shape_size);
          x_shape_size = (j == yndim-1 ? x->shape[k] : x_shape_size * x->shape[k]);
        }

        uint64_t x_indices_i = std::min(std::max(indices_data[indices_i], 0), (int32_t)x->shape[k]-1);
        x_i += (x_shape_size == 0 ? x_indices_i : x_indices_i * x_shape_size);
        x_shape_size = (x_shape_size == 0 ? x->shape[k] : x_shape_size * x->shape[k]);
        --k;

        o_i = i;
        for(int32_t j = yndim - 1; j>=0 && k >= 0; j--){
          uint64_t col = o_i % y->shape[j];
          o_i /= y->shape[j];
          if(j < axis){
            x_i += x_shape_size == 0 ? col : col * x_shape_size;
            x_shape_size = x_shape_size == 0 ? x->shape[k] : x_shape_size * x->shape[k];
            --k;
          }
        }
        y_data[i] = x_data[x_i];
      }
    }
}
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.take")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *indices = args[1];
    DLTensor *y = args[2];
    void *_attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TakeParam>(attr->parsed);

    // std::cerr << "axis = " << axis << " " << x->ndim << " " << y->ndim << "\n";
    if(!param.axis.has_value()){
      take(x, indices, y);
    }else{
      int32_t axis = param.axis.value();
      if(axis < 0){
          axis += x->ndim;
      }
      take(x, indices, y, axis);
    }
    print_to_file(x, "take_x.txt");
    print_to_file(y, "take.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_lut")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *indices = args[1];
    DLTensor *y = args[2];
  //  void *_attr = args[3];
  //  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  //  auto &param = cvm::get<cvm::top::CVMLUTParam>(attr->parsed);

    take(indices, x, y);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.upsampling")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  VERIFY(args.num_args == 3);
  DLTensor *x = args[0];
  DLTensor *y = args[1];

  VERIFY_EQ(x->ndim,     4) << "dimension should be 4D, Got: " << x->ndim;
  VERIFY_EQ(x->ndim,     y->ndim) << "dimension should match " << x->ndim << "!=" << y->ndim;
  VERIFY_EQ(x->shape[0], y->shape[0]) << "batch size should match";
  VERIFY_EQ(x->shape[1], y->shape[1]) << "batch size should match";

  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::UpSamplingParam>(attr->parsed);
  VERIFY_EQ(param.method, "NEAREST_NEIGHBOR") << "only accept method = NEAREST_NEIGHBOR ";
  VERIFY_EQ(param.layout, "NCHW") << "only accept NHWC, Got:" << param.layout;

  uint32_t scale = {(uint32_t)param.scale};
  uint32_t h = x->shape[2], w = x->shape[3];
  uint32_t oh = y->shape[2], ow = y->shape[3];
  uint32_t n_batch = x->shape[0], n_channels = x->shape[1];

  auto x_data = static_cast<int32_t*>(x->data);
  auto y_data = static_cast<int32_t*>(y->data);

  // std::cerr << "scale = " << scale << "\n";
  // std::cerr << "input = " << x->shape[0] << " " << x->shape[1]
  //           << " " << x->shape[2] << " " << x->shape[3]
  //           << "\n";

  // std::cerr << "output = " << y->shape[0] << " " << y->shape[1]
  //           << " " << y->shape[2] << " " << y->shape[3]
  //           << "\n";

  //TODO(zkh) optimize nested for-loop for scale
  #pragma omp parallel for collapse(2)
  for (uint32_t batch = 0; batch < n_batch; batch++) {
    for (uint32_t c = 0; c< n_channels; c++) {
      auto bc_y_data = y_data + batch * n_channels * oh * ow + c * oh * ow;
      auto bc_x_data = x_data + batch * n_channels *  h *  w + c *  h *  w;
      for(uint32_t y = 0; y < oh; y++){
        for(uint32_t x = 0; x < ow; x++){
            bc_y_data[y * ow + x] = bc_x_data[y/scale * w + x/scale];
        }
      }
    }
  }
#ifdef CVM_PROFILING
    cvm_op_upsampling_cnt += omp_get_wtime() - start;
    start = omp_get_wtime();
#endif
});

}
}

