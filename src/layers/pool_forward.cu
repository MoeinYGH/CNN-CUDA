#include <cuda_runtime.h>
#include <float.h>
#include "../../include/pool_layer.h"
#include "../../include/tensor.h"
#include "../../include/utils.h"

// Flatten NCHW index helper (Must match backward pass)
__device__ __forceinline__ int idx4_nchw(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

__global__ void maxpool_forward_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int* __restrict__ argmax,
    int B, int C, int Hin, int Win,
    int Hout, int Wout,
    int poolH, int poolW,
    int strideH, int strideW
) {
    // 1. Calculate Global Output Index
    // Strategy: One thread per output pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * Hout * Wout;

    if (idx < total_elements) {
        // 2. Decompose Index -> (b, c, ho, wo)
        int temp_idx = idx;
        int wo = temp_idx % Wout; temp_idx /= Wout;
        int ho = temp_idx % Hout; temp_idx /= Hout;
        int c  = temp_idx % C;    temp_idx /= C;
        int b  = temp_idx;

        // 3. Determine Input Window Coordinates
        int h_start = ho * strideH;
        int w_start = wo * strideW;
        int h_end = h_start + poolH;
        int w_end = w_start + poolW;

        // 4. Find Max in Window
        float max_val = -FLT_MAX; // Initialize with lowest possible float
        int max_idx_offset = 0;   // To store local offset (0..poolH*poolW-1)

        // Iterate over the pooling window
        for (int i = h_start; i < h_end; ++i) {
            for (int j = w_start; j < w_end; ++j) {
                // Check Bounds (in case of padding/odd sizes)
                if (i < Hin && j < Win) {
                    int input_idx = idx4_nchw(b, c, i, j, C, Hin, Win);
                    float val = X[input_idx];

                    if (val > max_val) {
                        max_val = val;
                        // Calculate offset relative to window start for argmax
                        // Offset = (dy * poolW) + dx
                        int dy = i - h_start;
                        int dx = j - w_start;
                        max_idx_offset = dy * poolW + dx;
                    }
                }
            }
        }

        // 5. Store Result and Index
        // idx corresponds to the output flattened index (b, c, ho, wo)
        Y[idx] = max_val;
        argmax[idx] = max_idx_offset; 
    }
}

/**
 * @brief Host wrapper for MaxPool Forward.
 */
void maxpool_forward(
    const Tensor* d_input,
    Tensor* d_output,
    int* d_argmax,
    int pool_size,
    int stride,
    int blocksize
) {
    // 1. Extract Dimensions
    int B = d_input->N;
    int C = d_input->C;
    int Hin = d_input->H;
    int Win = d_input->W;
    
    // Output dimensions usually pre-calculated in the Tensor, 
    // but here we use the actual formula to be safe/consistent.
    int Hout = d_output->H; 
    int Wout = d_output->W;

    int total_output = B * C * Hout * Wout;

    // 2. Configure Grid
    int threadsPerBlock = blocksize;
    int blocksPerGrid = (total_output + threadsPerBlock - 1) / threadsPerBlock;

    // 3. Launch Kernel
    // We assume square pooling window (pool_size x pool_size) as per project requirements
    maxpool_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input->data,
        d_output->data,
        d_argmax,
        B, C, Hin, Win,
        Hout, Wout,
        pool_size, pool_size, // poolH, poolW
        stride, stride        // strideH, strideW
    );

    check_kernel_launch("MaxPool Forward");
}