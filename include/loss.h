#ifndef LOSS_H
#define LOSS_H

void compute_softmax_loss_gradient(
    const float* d_logits,
    const int* d_labels,
    float* d_dlogits,
    float* d_loss,
    int batch_size,
    int num_classes
);

#endif