#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE_WIDTH 32
#define KERNEL_SIZE 8*BLOCK_SIZE

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


__constant__ float Kernel[KERNEL_SIZE];

__global__ void matrixMultiplyShared(float *B, float *C, int numAColumns, int numCRows, int numCColumns) {
  __shared__ float shared_mem[TILE_WIDTH][TILE_WIDTH];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0;
  for (int i = 0; i < ceil(1.0*numAColumns/TILE_WIDTH); i++) {
    if (i*TILE_WIDTH+threadIdx.y < numAColumns)
      shared_mem[threadIdx.y][threadIdx.x] = B[(i*TILE_WIDTH+threadIdx.y)*numCColumns + col];
    else 
      shared_mem[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
      if (i*TILE_WIDTH+k < numAColumns)
        acc += Kernel[row*numAColumns+i*TILE_WIDTH+k]*shared_mem[k][threadIdx.x];
    __syncthreads();   
  }
  if ((row < numCRows) && (col < numCColumns)) {
    C[row*numCColumns+col] = acc;
  }
  
}
void gemm(float* X_unrolled,  float* Y, int C, int M, int H) {
    // matrixMultiplyShared(float *A, float *B, float *C,
    //                                  int numAColumns, int numCRows, int numCColumns)
    // W_unroll = K
    int blockDimX = TILE_WIDTH, blockDimY = TILE_WIDTH;
    int gridDimY = ceil(1.0*M/blockDimY), gridDimX = ceil(1.0*H/blockDimX);
    dim3 gridDim (gridDimX, gridDimY), blockDim (blockDimX, blockDimY);
    matrixMultiplyShared<<<gridDim, blockDim>>>(X_unrolled, Y, C, M, H);
}
__global__ void unrollKernel(float* X_unrolled, int size, float* X, int C, int K, int H, int W) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int H_out, W_out, row, col, q, p, c, w, h;
    if (i<size){
      H_out = H-K+1;
      W_out = W-K+1;
      row = i/(H_out*W_out);
      col = i%(H_out*W_out);
      q = row % K;
      row /= K;
      p = row % K;
      c = row / K;
      w = col % W_out;
      h = col / W_out;
      X_unrolled[i] = X[c*H*W+(h+p)*W+w+q];
    }
}
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    float* Y = y.dptr_;
    float* X = x.dptr_;
    cudaMemcpyToSymbol(Kernel, k.dptr_, sizeof(float)*M*C*K*K);
    float* X_unrolled;
    int size = C*K*K*H_out*W_out;
    cudaMalloc(&X_unrolled, sizeof(float)*size);
    int b =B;
    while (b>0) {
        unrollKernel<<<ceil(1.0*size/BLOCK_SIZE), BLOCK_SIZE>>>(X_unrolled, size, X+b*C*H*W, C, K, H, W);
        gemm(X_unrolled,  Y+b*M*H_out*W_out,  C*K*K,  M,  H_out*W_out);
        b--;
        
    }
    cudaFree(X_unrolled);

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