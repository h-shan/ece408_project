#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define BLOCK_SIZE 1024
#define TILE 32
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

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

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	
	const int TILE_WIDTH = 16;
	const int W_grid = ceil(1.*W_out/TILE_WIDTH);
	const int H_grid = ceil(1.*H_out/TILE_WIDTH);

	int b, m, h, w, c, p, q;
	b = blockIdx.x;
	m = blockIdx.y;
	h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
	w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
	if (h >= H_out || w >= W_out)
		return;
	float acc = 0;
	for (c = 0; c < C; c++) 
		for (p = 0; p < K; p++) 
			for (q = 0; q < K; q++)
				acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
	y4d(b, m, h, w) = acc;


    

#undef y4d
#undef x4d
#undef k4d
}
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numAColumns, int numCRows, int numCColumns) 
{
  // numARows = numCRows
  // numBRows = numAColumns
  // numBColumns = numCColumns

  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float SHAREDA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float SHAREDB[TILE_WIDTH][TILE_WIDTH];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0;
  int width=ceil(1.0*numAColumns/TILE_WIDTH);
  for (int i= 0; i < width; i++) {

    int j = i*TILE_WIDTH+threadIdx.x;
    if (j < numAColumns)
      SHAREDA[threadIdx.y][threadIdx.x] = A[row*numAColumns+j];
    int l = i*TILE_WIDTH+threadIdx.y;
    if (l < numAColumns)
      SHAREDB[threadIdx.y][threadIdx.x] = B[row*numCColumns+l];

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
       result += tileA[threadIdx.y][k]*tileB[k][threadIdx.x];
    __syncthreads();   
  }
  
  if ((row < numCRows) && (col < numCColumns)) {
    C[row*numCColumns+col] = acc;
  }
}
__global__ void unrollKernel(float* X_unrolled, int size, float* X, int C, int K, int H, int W) {
    int c,s,, h_out, w_out, h_unroll, w_base, p, q;
     int t=blockIdx.x*CUDA MAX_NUM_THREADS+threadIdx.x;
     int H_out=H-K+1;
     int W_out=H-K+1;
     int W_unroll=H_out*W_out; 

     if(t<C*W_unroll){
         c=t/W_unroll;
         s=t%W_unroll; 
         h_out=s/W_out;
         w_out=s%W_out; 
         h_unroll=h_out*W_out+w_out; 
         w_base=c*K*K;
         for(p=0; p<K; p++)
            for(q=0; q<K; q++){
                w_unroll=w_base+p*K+q;
                X_unroll[h_unroll+w_unroll]=X(c, h_out+p, w_out+q);
            }
     }
    
}

void unroll(float* X_unrolled, int size, float* X, int C, int K, int H, int W) {
    int gridDim = ceil(1.0*size/BLOCK_SIZE);
    unrollKernel<<<gridDim, BLOCK_SIZE>>>(X_unrolled, size, X, C, K, H, W);
}
void launch(float* Kernel, float* X_unrolled,  float* Y, int CKK, int M, int HW) {
    // matrixMultiplyShared(float *A, float *B, float *C,
    //                                  int numAColumns, int numCRows, int numCColumns)
    // W_unroll = K
    int blockDimX = TILE_WIDTH, blockDimY = TILE_WIDTH;
    int gridDimY = ceil(1.0*M/blockDimY), gridDimX = ceil(1.0*HW/blockDimX);
    dim3 gridDim (gridDimX, gridDimY), blockDim (blockDimX, blockDimY);
    matrixMultiplyShared<<<gridDim, blockDim>>>(Kernel, X_unrolled, Y, CKK, M, HW);
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
    const int TILE_WIDTH = 16;

	  float* Ker = k.dptr_;
    
    float* X_unrolled;
    int elements = C*K*K*H_out*W_out;
    cudaMalloc(&X_unrolled, sizeof(float)*size);
    for (int b = B; b--; ) {
        unroll(X_unrolled, elements, X+b*C*H*W, C, K, H, W);
        launch(Ker,  X_unrolled,  Y+b*M*H_out*W_out,  C*K*K,  M,  H_out*W_out);
    }
    cudaFree(X_unrolled);
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