#include <cuda_runtime.h>
#include "activations.h"
#include "tensor.h"
__global__ void relu_forward_kernel(const float* __restrict__ X,
                                    float* __restrict__ Y,
                                    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = X[idx];
        Y[idx] = (x > 0.0f) ? x : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* __restrict__ X,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dX,
                                     int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dX[idx] = (X[idx] > 0.0f) ? dY[idx] : 0.0f;
    }
}

void relu_forward(Tensor& X, Tensor& Y, int blocksize)
{
    int N = X.N * X.C * X.H * X.W;

    int threads = blocksize;
    if (threads <= 0 || threads > 1024) threads = 256;

    int blocks = (N + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(X.data, Y.data, N);
}

void relu_backward(Tensor& X,Tensor& dY, Tensor& dX, int blocksize)
{
    int N = X.N * X.C * X.H * X.W;

    int threads = blocksize;
    if (threads <= 0 || threads > 1024) threads = 256;

    int blocks = (N + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(X.data, dY.data, dX.data, N);
}
