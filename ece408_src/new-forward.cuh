
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


#define TILE_SIZE 32
#define KERNEL_SIZE 5
#define MAX_NUM_THREADS 1024

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
  int numARows, int numAColumns,
  int numBRows, int numBColumns,
  int numCRows, int numCColumns,
  int ckk, int total_elements, int M
) {
  B += blockIdx.z * total_elements * ckk;
  C += blockIdx.z * M * total_elements;
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM[TILE_SIZE][TILE_SIZE];
  __shared__ float subTileN[TILE_SIZE][TILE_SIZE];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  int width = numAColumns;
  float p = 0;

  int numTiles = width/TILE_SIZE;
  if (row < numCRows || col < numCColumns) {
    for (int i = 0; i < numTiles; i++) {
      subTileM[ty][tx] = A[row * numAColumns + i * TILE_SIZE + tx];
      subTileN[ty][tx] = B[(i * TILE_SIZE + ty) * numBColumns + col];
      __syncthreads();
      for (int k = 0; k < TILE_SIZE; k++) {
        p += subTileM[ty][k] * subTileN[k][tx]; 
      }
      __syncthreads();
    }
    for (int i = numTiles * TILE_SIZE; i < width; i++) {
      p += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
    if (row < numCRows && col < numCColumns) {
      C[row * numCColumns + col] = p;
    }
  }
}

__global__ void unroll_Kernel(int C, int H, int W, int K, float *X, float *X_unroll)
{
  X += blockIdx.z * H * W * C;
  int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q, m;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int total_elements = H_out * W_out;
  X_unroll += blockIdx.z * C * K * K * total_elements;

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

void unroll_gpu(int C, int H, int W, int K, int N, float *X, float *X_unroll)
{
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int num_threads = H_out * W_out;
  int num_blocks = ceil((num_threads * C + 0.0) / (MAX_NUM_THREADS / C * C));
  int num_blocks2 = ceil((H_out * W_out * C + 0.0) / MAX_NUM_THREADS);
  dim3 gridDim(num_blocks, 1, N);
  dim3 blockDim(MAX_NUM_THREADS / C, C, 1);

  unroll_Kernel<<<gridDim, blockDim>>>(C, H, W, K, X, X_unroll);
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
  cudaMalloc((void **) &X_unrolled, N * ckk * total_elements * sizeof(float));

  int m = M, k = ckk;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  int numARows = M;
  int numAColumns = ckk;
  int numBRows = ckk;
  int numBColumns = total_elements;
  int numCRows = M;
  int numCColumns = total_elements;

  unroll_gpu(C, H, W, K, N, x.dptr_, X_unrolled);
  dim3 dimGrid(ceil(numCColumns/(TILE_SIZE + 0.0)), ceil(numCRows/(TILE_SIZE + 0.0)), N);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(w.dptr_, X_unrolled, y.dptr_, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, ckk, total_elements, M);

  // for (int n=0; n < N; n++) {
  //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
  //     total_elements, M, ckk, alpha,
  //     X_unrolled + n * ckk * total_elements, total_elements,
  //     w.dptr_, ckk,
  //     beta,
  //     y.dptr_ + (n * M * total_elements), total_elements);
  // }
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
