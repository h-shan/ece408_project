
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>
namespace mxnet
{
namespace op
{


#define TILE_SIZE 16
#define KERNEL_SIZE 5
__constant__ float cst_ptr [KERNEL_SIZE * KERNEL_SIZE * 500]; // 50 max number of channels


__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

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
}

__global__ void forward_kernel_original(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    const int W_grid = ceil((W_out + 0.0) / TILE_SIZE); // number of horizontal tiles per output map

    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
    
    if (h >= H_out || w >= W_out) {
       return;
    }
    
    float acc = 0.;
    for (c = 0; c < C; c++) { // sum over all input channels
      for (p = 0; p < K; p++) // loop over KxK filter
        for (q = 0; q < K; q++)
          acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
    }
    y4d(n, m, h, w) = acc;
    #undef y4d
    #undef x4d
    #undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];  // Batch Size
    const int M = y.shape_[1];  // Output Channels (Feature Maps)
    const int C = x.shape_[1];  // Input Channels (Feature Maps) 
    const int H = x.shape_[2];  // Output height (y)
    const int W = x.shape_[3];  // Output width (x)
    const int K = w.shape_[3];  // Filter Width/Height

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int Z = ceil((W_out + 0.0) / TILE_SIZE) * ceil((H_out + 0.0)/ TILE_SIZE);
    
    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);

    cudaMemcpyToSymbol(cst_ptr, w.dptr_, KERNEL_SIZE * KERNEL_SIZE * M * C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    size_t shmem_size = sizeof(float) * ( (TILE_SIZE + K-1) * (TILE_SIZE + K-1) * 2);
    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
