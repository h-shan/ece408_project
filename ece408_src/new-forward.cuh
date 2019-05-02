
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>

namespace mxnet
{
namespace op
{


#define KERNEL_SIZE 5
#define MAX_NUM_THREADS 1024
__constant__ float cst_ptr [KERNEL_SIZE * KERNEL_SIZE * 500]; // 50 max number of channels

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
  int numARows, int numAColumns,
  int numBRows, int numBColumns,
  int numCRows, int numCColumns,
  int ckk, int total_elements, int M
) {
  #define TILE_SIZE 16
  B += blockIdx.z * total_elements * ckk;
  C += blockIdx.z * M * total_elements;
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM1[TILE_SIZE * TILE_SIZE];
  __shared__ float subTileN1[TILE_SIZE * TILE_SIZE];
  __shared__ float subTileM2[TILE_SIZE * TILE_SIZE];
  __shared__ float subTileN2[TILE_SIZE * TILE_SIZE];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  int width = numAColumns;
  float p = 0;

  int numTiles = width/TILE_SIZE;
  float *subTileM, *subTileN;

  if (row < numCRows || col < numCColumns) {
    for (int i = 0; i < numTiles; i++) {
      if (i%2) {
        subTileM = subTileM1;
        subTileN = subTileN1;
      } else {
        subTileM = subTileM2;
        subTileN = subTileN2;
      }
      subTileM[ty * TILE_SIZE + tx] = A[row * numAColumns + i * TILE_SIZE + tx];
      subTileN[ty * TILE_SIZE + tx] = B[(i * TILE_SIZE + ty) * numBColumns + col];
      __syncthreads();
      for (int k = 0; k < TILE_SIZE; k++) {
        p += subTileM[ty * TILE_SIZE + k] * subTileN[k * TILE_SIZE + tx]; 
      }
    }
    for (int i = numTiles * TILE_SIZE; i < width; i++) {
      p += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
    if (row < numCRows && col < numCColumns) {
      C[row * numCColumns + col] = p;
    }
  }
  #undef TILE_SIZE
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    #define TILE_SIZE 8
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) cst_ptr[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_SIZE + K-1;
    extern __shared__ float shmem[];
    float* X_shared1 = &shmem[0];
    float* X_shared2 = &shmem[X_tile_width * X_tile_width];
    const int W_grid = ceil((W_out + 0.0) / TILE_SIZE); // number of horizontal tiles per output map
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_SIZE; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid) * TILE_SIZE; // horizontal base out data index for the block
    h = h_base + h0;
    w = w_base + w0;
    
    float acc = 0.;
    int c, p, q;
    for (c = 0; c < C; c++) { // sum over all input channels
      float *X_shared = c % 2 ? X_shared1 : X_shared2;
      for (int i = h; i < h_base + X_tile_width; i += TILE_SIZE) {
        for (int j = w; j < w_base + X_tile_width; j += TILE_SIZE) {
          if (i < H && j < W) {
            X_shared[(i - h_base) * X_tile_width + j - w_base] = x4d(n, c, i, j);
          }
        }
      }
      __syncthreads();
      for (p = 0; p < K; p++) {
        for (q = 0; q < K; q++) {
          acc += X_shared[(h0 + p) * X_tile_width + w0 + q] * k4d(m, c, p, q);
        }
      }
    }
    if (h < H_out && w < W_out) {
      y4d(n, m, h, w) = acc; 
    }

    #undef y4d
    #undef x4d
    #undef k4d
    #undef TILE_SIZE
}

__global__ void unroll_Kernel(int C, int H, int W, int K, float *X, float *X_unroll)
{
  X += blockIdx.z * H * W * C;
  int c, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
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

  int W_out = W - K + 1;
  int H_out = H - K + 1;
  if (C == 1) {
    const intTILE_SIZE = 8;
    const int Z = ceil((W_out + 0.0) / TILE_SIZE) * ceil((H_out + 0.0)/ TILE_SIZE);
    
    // Set the kernel dimensions
    dim3 gridDim(N, M, Z);
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);

    cudaMemcpyToSymbol(cst_ptr, w.dptr_, KERNEL_SIZE * KERNEL_SIZE * M * C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t shmem_size = sizeof(float) * ( (TILE_SIZE + K-1) * (TILE_SIZE + K-1) * 2);
    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, N,M,C,H,W,K);
  } else {
    const int TILE_SIZE = 16;
    int ckk = C * K * K;
    int total_elements = H_out * W_out;
    float *X_unrolled;
    cudaMalloc((void **) &X_unrolled, N * ckk * total_elements * sizeof(float));
  
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
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    cudaFree(X_unrolled);
  }
}

template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
