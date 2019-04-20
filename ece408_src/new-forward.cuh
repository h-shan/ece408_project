
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace mxnet
{
namespace op
{


#define TILE_SIZE 8
#define KERNEL_SIZE 5
#define MAX_NUM_THREADS 1024
__constant__ float cst_ptr[KERNEL_SIZE * KERNEL_SIZE * 500]; // 50 max number of channels

__global__ void unroll_Kernel(int C, int H, int W, int K, float *X, float *X_unroll)
{
  int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q, m;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int total_elements = H_out * W_out;

  #define x3d(i2, i1, i0) X[(i2) * (H * W) + (i1) * (W) + (i0)]
  #define x_unroll2d(i1, i0) X_unroll[total_elements * (i1) + (i0)]

  c = threadIdx.y;
  if (t < total_elements) {
    h_out = t / W_out;
    w_out = t % W_out;
    h_unroll = h_out * W_out + w_out;
    w_base = c * K * K;
    for(p = 0; p < K; p++) {
      for(q = 0; q < K; q++) {
        w_unroll = w_base + p * K + q;
        x_unroll2d(w_unroll, h_unroll) = x3d(c, h_out + p, w_out + q);
      }
    }
  }

  #undef x3d
  #undef x_unroll2d
}

void unroll_gpu(int C, int H, int W, int K, float *X, float *X_unroll)
{
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int num_threads = H_out * W_out;
  int num_blocks = ceil((num_threads * C + 0.0) / (MAX_NUM_THREADS / C * C));
  int num_blocks2 = ceil((H_out * W_out * C + 0.0) / MAX_NUM_THREADS);
  dim3 gridDim(num_blocks, C, 1);
  dim3 blockDim(MAX_NUM_THREADS / C, C, 1);

  unroll_Kernel<<<num_blocks, blockDim>>>(C, H, W, K, X, X_unroll);
  // unroll_Kernel2<<<gridDim, MAX_NUM_THREADS>>>(C, H, W, K, X, X_unroll);
}

template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
  const int N = x.shape_[0];  // Batch Size
  const int M = y.shape_[1];  // Output Channels (Feature Maps)
  const int C = x.shape_[1];  // Input Channels (Feature Maps) 
  const int H = x.shape_[2];  // Output height (y)
  const int W = x.shape_[3];  // Output width (x)
  const int K = w.shape_[3];  // Filter Width/Height

  printf("w: %d %d %d %d\n", w.shape_[0], w.shape_[1], w.shape_[2], w.shape_[3]);
  printf("x: %d %d %d %d\n", x.shape_[0], x.shape_[1], x.shape_[2], x.shape_[3]);
  printf("y: %d %d %d %d\n", y.shape_[0], y.shape_[1], y.shape_[2], y.shape_[3]);
  int W_out = W - K + 1;
  int H_out = H - K + 1;
  int ckk = C * K * K;
  int total_elements = H_out * W_out;
  
  cublasHandle_t handle;
  cublasCreate(&handle);

  float *X_unrolled;
  cudaMalloc((void **) &X_unrolled, ckk * total_elements * sizeof(float));

  int m = M, k = ckk;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  for (int n=0; n < N; n++) {
    unroll_gpu(C, H, W, K, x.dptr_ + (H * W * C * n), X_unrolled);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      total_elements, M, ckk, alpha,
      X_unrolled, total_elements,
      w.dptr_, ckk,
      beta,
      y.dptr_ + (n * M * total_elements), total_elements);
  }
  cublasDestroy(handle);
  MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
