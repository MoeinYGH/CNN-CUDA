#include <stdio.h>
#include "../../include/timer.h"
#include "../../include/utils.h"

/**
 * @brief Create CUDA events for timing.
 */
void timer_init(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventCreate(&timer->start));
        CHECK_CUDA(cudaEventCreate(&timer->stop));
    }
}

/**
 * @brief Record the start event.
 * * This places a marker in the CUDA stream.
 */
void timer_start(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventRecord(timer->start, 0));
    }
}

/**
 * @brief Record the stop event.
 * * This places a marker in the CUDA stream.
 */
void timer_stop(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventRecord(timer->stop, 0));
    }
}

/**
 * @brief Calculate elapsed time in milliseconds.
 * * Automatically synchronizes the CPU until the GPU reaches the 'stop' marker.
 * * Returns: Time in milliseconds (ms).
 */
float timer_elapsed_ms(GpuTimer* timer) {
    if (!timer) return 0.0f;

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(timer->stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, timer->start, timer->stop));
    
    return milliseconds;
}

/**
 * @brief Cleanup resources.
 */
void timer_destroy(GpuTimer* timer) {
    if (timer) {
        CHECK_CUDA(cudaEventDestroy(timer->start));
        CHECK_CUDA(cudaEventDestroy(timer->stop));
    }
}