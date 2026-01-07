#include <cuda_runtime.h>
#include <stdio.h>
#include "../../include/fc_layer.h"
#include "../../include/tensor.h"
#include "../../include/utils.h"

/**
 * @brief Fully Connected Forward Kernel (Naive).
 * * Computes: Y = X * W^T + b
 * * Strategy: One thread per output element (Batch_i, Output_j).
 * * Input X: (B, Cin)
 * * Weights W: (Cout, Cin) - Row-major, where each row is a neuron's weights.
 * * Output Y: (B, Cout)
 */
__global__ void fc_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    // 1. Calculate Global Thread Index (Linear)
    // We map the 1D index to the 2D output matrix (Batch, Out_Features)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;

    if (idx < total_elements) {
        // 2. Decompose Index -> (b, out_idx)
        // b: Which image in the batch?
        // out_idx: Which output neuron?
        int out_idx = idx % out_features;
        int b = idx / out_features;

        // 3. Initialize Accumulator
        float sum = 0.0f;

        // 4. Dot Product
        // We iterate over the Input Feature dimension.
        // Input row 'b' starts at: input + b * in_features
        // Weight row 'out_idx' starts at: weights + out_idx * in_features
        int input_row_start = b * in_features;
        int weight_row_start = out_idx * in_features;

        for (int i = 0; i < in_features; ++i) {
            sum += input[input_row_start + i] * weights[weight_row_start + i];
        }

        // 5. Add Bias
        if (bias != nullptr) {
            sum += bias[out_idx];
        }

        // 6. Store Result
        output[idx] = sum;
    }
}

/**
 * @brief Host wrapper for Fully Connected Forward.
 * * Automatically handles flattening of 4D input tensors (e.g., from Conv/Pool layers)
 * * into 2D matrices (Batch, Features).
 */
void fc_forward(
    const Tensor* d_input,
    const Tensor* d_weights,
    const float* d_bias,
    Tensor* d_output,
    int blocksize
) {
    // 1. Determine Dimensions
    int batch_size = d_input->N;
    
    // Calculate total input features (flatten C, H, W)
    // Example: For shape (64, 16, 7, 7), in_features = 16*7*7 = 784
    int in_features = d_input->C * d_input->H * d_input->W;
    
    // Output features comes from the weight tensor dimensions
    // We assume Weights shape is (Cout, Cin, 1, 1) or just (Cout, Cin)
    // So d_weights->N is Output Features (Cout)
    // and d_weights->C * H * W is Input Features (Cin)
    int out_features = d_weights->N;

    // Validate dimensions to prevent mismatches
    int weight_in_features = d_weights->C * d_weights->H * d_weights->W;
    if (in_features != weight_in_features) {
        fprintf(stderr, "FC Layer Error: Input features (%d) != Weight input cols (%d)\n", 
            in_features, weight_in_features);
        exit(EXIT_FAILURE);
    }

    // 2. Configure Grid
    int total_elements = batch_size * out_features;
    int threadsPerBlock = blocksize;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    // 3. Launch Kernel
    fc_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input->data,
        d_weights->data,
        d_bias,
        d_output->data,
        batch_size,
        in_features,
        out_features
    );

    check_kernel_launch("FC Forward");
}