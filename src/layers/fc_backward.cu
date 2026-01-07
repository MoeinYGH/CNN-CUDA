#include <cuda_runtime.h>
#include "../include/fc_layer.h"
#include "../include/tensor.h"
#include "../include/utils.h"

// ------------------------------------------------------------
// Kernel 1: Calculate Input Gradient (dX)
// Math: dX = dY * W
// dY: (B, Cout)
// W:  (Cout, Cin)
// dX: (B, Cin)
// ------------------------------------------------------------
__global__ void fc_backward_input_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ W,
    float* __restrict__ dX,
    int B, int Cin, int Cout
) {
    // One thread per input element (b, ci)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cin;

    if (idx < total) {
        int ci = idx % Cin;
        int b  = idx / Cin;

        float acc = 0.0f;
        
        // Dot product: row 'b' of dY dot column 'ci' of W
        // W is stored row-major (Cout, Cin), so column access is strided.
        // W[co, ci] is at index: co * Cin + ci
        for (int co = 0; co < Cout; ++co) {
            acc += dY[b * Cout + co] * W[co * Cin + ci];
        }

        dX[idx] = acc;
    }
}

// ------------------------------------------------------------
// Kernel 2: Calculate Weight Gradient (dW)
// Math: dW = dY^T * X
// dY: (B, Cout)
// X:  (B, Cin)
// dW: (Cout, Cin)
// ------------------------------------------------------------
__global__ void fc_backward_weight_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    float* __restrict__ dW,
    int B, int Cin, int Cout
) {
    // One thread per weight element (co, ci)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin;

    if (idx < total) {
        int ci = idx % Cin;
        int co = idx / Cin;

        float acc = 0.0f;

        // Sum over batch dimension
        // dW[co, ci] = sum_b (dY[b, co] * X[b, ci])
        for (int b = 0; b < B; ++b) {
            acc += dY[b * Cout + co] * X[b * Cin + ci];
        }

        dW[idx] = acc;
    }
}

// ------------------------------------------------------------
// Kernel 3: Calculate Bias Gradient (db)
// Math: db = sum(dY, axis=0)
// dY: (B, Cout)
// db: (Cout)
// ------------------------------------------------------------
__global__ void fc_backward_bias_kernel(
    const float* __restrict__ dY,
    float* __restrict__ db,
    int B, int Cout
) {
    // One thread per output neuron (co)
    int co = blockIdx.x * blockDim.x + threadIdx.x;

    if (co < Cout) {
        float acc = 0.0f;
        
        // Sum the gradients for this neuron across all batch items
        for (int b = 0; b < B; ++b) {
            acc += dY[b * Cout + co];
        }

        db[co] = acc;
    }
}

// ------------------------------------------------------------
// Host Wrapper
// ------------------------------------------------------------
void fc_backward(
    const Tensor* d_input,   // X
    const Tensor* d_weights, // W
    const Tensor* d_dY,      // Incoming Gradient
    Tensor* d_dX,            // Output Gradient (w.r.t Input)
    Tensor* d_dW,            // Output Gradient (w.r.t Weights)
    float* d_db              // Output Gradient (w.r.t Bias)
) {
    int B = d_input->N;
    // implicit flatten: Cin = C*H*W
    int Cin = d_input->C * d_input->H * d_input->W;
    int Cout = d_dY->C * d_dY->H * d_dY->W; // Typically just C if 1x1, but safe to multiply

    int threads = 256;

    // 1. Calculate dX (if needed)
    // Only compute if d_dX is not null (e.g., typically null for the very first layer input)
    if (d_dX != nullptr) {
        int total_dX = B * Cin;
        int blocks_dX = (total_dX + threads - 1) / threads;
        fc_backward_input_kernel<<<blocks_dX, threads>>>(
            d_dY->data, d_weights->data, d_dX->data, B, Cin, Cout
        );
        check_kernel_launch("FC Backward dX");
    }

    // 2. Calculate dW
    if (d_dW != nullptr) {
        int total_dW = Cout * Cin;
        int blocks_dW = (total_dW + threads - 1) / threads;
        fc_backward_weight_kernel<<<blocks_dW, threads>>>(
            d_dY->data, d_input->data, d_dW->data, B, Cin, Cout
        );
        check_kernel_launch("FC Backward dW");
    }

    // 3. Calculate db
    if (d_db != nullptr) {
        int total_db = Cout;
        int blocks_db = (total_db + threads - 1) / threads;
        fc_backward_bias_kernel<<<blocks_db, threads>>>(
            d_dY->data, d_db, B, Cout
        );
        check_kernel_launch("FC Backward db");
    }
}