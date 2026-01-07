#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/conv_layer.h"
#include "../../include/tensor.h" // Ensure this includes your struct definition
#include "../../include/utils.h"

/**
 * @brief Baseline Naive Convolution Forward Kernel using Tensor Struct.
 * * One thread per Output Pixel.
 * * Corresponding to Convolution v1 (baseline)[cite: 39].
 */
__global__ void conv_forward_naive_kernel(
    const Tensor input,
    const Tensor weights,
    const float* __restrict__ bias,
    Tensor output,
    int pad,
    int stride
) {
    // 1. Calculate Global Thread Index
    int total_elements = numel(output);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        // 2. Decompose Index into (b, oc, oh, ow)
        int temp_idx = idx;
        int ow = temp_idx % output.W;
        temp_idx /= output.W;
        int oh = temp_idx % output.H;
        temp_idx /= output.H;
        int oc = temp_idx % output.C;
        int b  = temp_idx / output.C;

        float sum = 0.0f;

        // 3. Convolution Loops
        // Loop over input channels (input.C)
        for (int ic = 0; ic < input.C; ++ic) {
            // Loop over kernel spatial dimensions (weights.H x weights.W)
            for (int kh = 0; kh < weights.H; ++kh) {
                for (int kw = 0; kw < weights.W; ++kw) {
                    
                    int ih_in = oh * stride - pad + kh;
                    int iw_in = ow * stride - pad + kw;

                    // Boundary Check for Padding [cite: 17]
                    if (ih_in >= 0 && ih_in < input.H && iw_in >= 0 && iw_in < input.W) {
                        
                        // Indexing using NCHW layout
                        int input_idx = ((b * input.C + ic) * input.H + ih_in) * input.W + iw_in;
                        int weight_idx = ((oc * input.C + ic) * weights.H + kh) * weights.W + kw;

                        sum += input.data[input_idx] * weights.data[weight_idx];
                    }
                }
            }
        }

        // 4. Add Bias (Bias is per output channel) [cite: 17]
        if (bias != nullptr) {
            sum += bias[oc];
        }

        // 5. Write to Output
        output.data[idx] = sum;
    }
}

/**
 * @brief Host wrapper using Tensor structs.
 */
void conv_forward(
    const Tensor* d_input,
    const Tensor* d_weights,
    const float* d_bias,
    Tensor* d_output,
    int pad,
    int stride,
    int blocksize
) {
    int total_output_elements = numel(*d_output);

    int threadsPerBlock = blocksize;
    int blocksPerGrid = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Passing the structs by value to the kernel (CUDA handles this by copying the struct to constant memory)
    conv_forward_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        *d_input, *d_weights, d_bias, *d_output, pad, stride
    );

    check_kernel_launch("Conv Forward Naive");
}