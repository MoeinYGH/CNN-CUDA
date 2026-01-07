#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "../../include/loss.h" 
#include "../../include/utils.h"

#define EPSILON 1e-7f

__global__ void softmax_cross_entropy_fused_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ labels,
    float* __restrict__ d_logits,
    float* __restrict__ loss,
    int batch_size,
    int num_classes
) {
    // One thread per sample
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        // Pointer to the start of this sample's row
        const float* row_logits = logits + idx * num_classes;
        float* row_grads = d_logits + idx * num_classes;
        int label = labels[idx];

        // --- Step 1: Find Max for Numerical Stability  ---
        float max_val = -1e30f; // Very small number
        for (int c = 0; c < num_classes; ++c) {
            if (row_logits[c] > max_val) {
                max_val = row_logits[c];
            }
        }

        // --- Step 2: Compute Sum of Exponentials (Denominator) ---
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            // Subtract max_val to prevent overflow
            sum_exp += expf(row_logits[c] - max_val);
        }

        // --- Step 3: Compute Gradients and Loss ---
        // Gradient of Softmax+CE is simply: (Probability - Target)
        // Target is 1.0 for the correct class, 0.0 otherwise.
        
        float current_loss = 0.0f;

        for (int c = 0; c < num_classes; ++c) {
            float softmax_prob = expf(row_logits[c] - max_val) / sum_exp;
            
            // Calculate gradient term (dZ)
            // If this is the correct class, subtract 1.0 (P - 1)
            // Otherwise, it remains (P - 0)
            float grad = softmax_prob;
            if (c == label) {
                grad -= 1.0f;
                // Calculate Loss only for the correct label
                // Add EPSILON to prevent log(0)
                current_loss = -logf(softmax_prob + EPSILON); 
            }

            // Write gradient to global memory
            // Normalize by batch size for mean reduction if required by optimizer,
            // otherwise standard implementation usually passes sum or mean.
            // Here we store raw gradients; optimizer usually handles learning rate scaling.
            row_grads[c] = grad / (float)batch_size; 
        }

        // Accumulate total loss atomically 
        // Note: For very large batches, block-level reduction before atomicAdd is faster,
        // but for this project scope, direct atomicAdd is acceptable correctness.
        atomicAdd(loss, current_loss);
    }
}

void compute_softmax_loss_gradient(
    const float* d_logits,
    const int* d_labels,
    float* d_dlogits,
    float* d_loss,
    int batch_size,
    int num_classes,
    int blocksize
) {
    // Reset loss to 0 on GPU before accumulation
    cudaMemset(d_loss, 0, sizeof(float));

    // Configure Grid
    int threadsPerBlock = blocksize;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Fused Kernel 
    softmax_cross_entropy_fused_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_logits,
        d_labels,
        d_dlogits,
        d_loss,
        batch_size,
        num_classes
    );

    // Check for launch errors (as per Best Practices)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Softmax Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}