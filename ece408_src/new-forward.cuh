#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define WIDTH 32

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void matrixMultiplyShared(float *Kernel, float *X, float *Y, int M, int C, int H, int W, int K, int H_out, int W_out, int numCColumns) 
{
  __shared__ float BLOCK_M[WIDTH][WIDTH];
  __shared__ float BLOCK_N[WIDTH][WIDTH];
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.z;
  Y += z * M * H_out * W_out;
  X += z * C * H * W;

  float acc = 0;
  int numCRows = M; 
  int index, row, col, q, p, c, w, h;
  for (int i = 0; i < ceil(1.0*C*K*K/WIDTH); i++) {
    if ((i*WIDTH+threadIdx.x)< C*K*K)
      BLOCK_M[threadIdx.y][threadIdx.x] = Kernel[row_index*C*K*K+(i*WIDTH+threadIdx.x)];
    if ((i*WIDTH+threadIdx.y)< C*K*K) {
      index = (i*WIDTH+threadIdx.y)*numCColumns + col_index;
      row = index/(H_out*W_out);
      col = index%(H_out*W_out);
      q = row % K;
      row /= K;
      p = row % K;
      c = row / K;
      w = col % W_out;
      h = col / W_out;
      BLOCK_N[threadIdx.y][threadIdx.x] = X[(c) * (H * W) + (h+p) * (W) + w+q];
    }
    __syncthreads();
    for (int k = 0; k < WIDTH; k++)
       acc += BLOCK_M[threadIdx.y][k]*BLOCK_N[k][threadIdx.x];
      
    __syncthreads();   
  }
  if ((row_index < numCRows) && (col_index < numCColumns)) {
    Y[row_index*numCColumns+col_index] = acc;
  }
  
}
int unroll(float* X, int numCColumns, int col_index) {

  return 0;
}
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

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
    float* Kernel = k.dptr_;
    
    // int b=B; 
    // while (b>0) {
    //   dim3 gridDim (ceil(1.0*H*W/WIDTH), ceil(1.0*M/WIDTH));
    //   dim3 blockDim (WIDTH, WIDTH);
    //   matrixMultiplyShared<<<gridDim, blockDim>>>(Kernel, X+b*C*H*W, Y+b*M*H_out*W_out, M, C, H, W, K, H_out, W_out, H_out*W_out);
    //   b--;
    // }
    dim3 gridDim (ceil(1.0*H*W/WIDTH), ceil(1.0*M/WIDTH), B);
    dim3 blockDim (WIDTH, WIDTH);
    matrixMultiplyShared<<<gridDim, blockDim>>>(Kernel, X, Y, M, C, H, W, K, H_out, W_out, H_out*W_out);
      
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