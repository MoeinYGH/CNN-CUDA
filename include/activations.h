#pragma once
#include "tensor.h"

// ReLU forward: Y = max(0, X)
void relu_forward(const Tensor& X, Tensor& Y, int blocksize);

// ReLU backward: dX = dY * (X > 0)
void relu_backward(const Tensor& X,
const Tensor& dY,
Tensor& dX,
int blocksize);
