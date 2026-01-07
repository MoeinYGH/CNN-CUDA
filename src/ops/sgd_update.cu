#include <cuda_runtime.h>
#include "optimizer.h"

__global__ void sgd_update_kernel(float* __restrict__ param,
                                  float* __restrict__ grad,
                                  int n,
                                  float lr,
                                  int zero_grad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        param[idx] -= lr * g;
        if (zero_grad) grad[idx] = 0.0f;
    }
}

void sgd_update_tensor(Tensor& param,
                       Tensor& grad,
                       float lr,
                       bool zero_grad,
                       int blocksize)
{
    int n = param.N * param.C * param.H * param.W;

    int threads = blocksize;
    if (threads <= 0 || threads > 1024) threads = 256;

    int blocks = (n + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(param.data, grad.data, n, lr, zero_grad ? 1 : 0);
}
