# CNN Training from Scratch in CUDA

Performance analysis and optimization of a LeNet-style CNN trained on MNIST, with every forward and backward kernel written by hand in CUDA (no PyTorch, no TensorFlow, no cuDNN).

**Author:** Moein Yeganeh
**Course project:** GPU Programming (February 2026)

---

## Overview

The goal of this project is not model complexity but understanding how CNN training maps onto GPU hardware. The network trains correctly on MNIST, and the main work is profiling it with NVIDIA tools, finding bottlenecks, and measuring four optimizations one after another.

**What is implemented as CUDA kernels**

- Forward: convolution, bias, ReLU, max pooling (with argmax), fully connected, softmax
- Loss: numerically stable softmax + cross-entropy
- Backward: softmax/cross-entropy gradient, fully connected (dX, dW, db), max pool, ReLU, convolution (dX, dW, db)
- Optimizer: SGD update on the GPU
- CPU side is limited to dataset loading, normalization, shuffling, batch preparation, and driving the training loop

## Network architecture

| Layer | Output shape (batch = B) | Parameters |
|---|---|---|
| Input | B x 1 x 28 x 28 | - |
| Conv1 (1 to 8, 3x3, pad 1) + Bias + ReLU | B x 8 x 28 x 28 | 80 |
| MaxPool 2x2, stride 2 | B x 8 x 14 x 14 | - |
| Conv2 (8 to 16, 3x3, pad 1) + Bias + ReLU | B x 16 x 14 x 14 | 1,168 |
| MaxPool 2x2, stride 2 | B x 16 x 7 x 7 | - |
| Flatten | B x 784 | - |
| FC1 (784 to 128) + ReLU | B x 128 | 100,480 |
| FC2 (128 to 10) | B x 10 | 1,290 |
| Softmax + Cross-Entropy | loss + gradient | - |

Total: **103,018 parameters**. Weights use He normal initialization, training uses plain SGD (no momentum), learning rate 0.01, and tensors use the NCHW layout.

## Optimizations

Each optimization is applied cumulatively on top of the previous one.

| # | Optimization | Idea |
|---|---|---|
| 0 | Baseline | Naive convolution (one thread per output pixel), separate conv / bias / ReLU kernels, atomic-heavy backward pass, synchronous H2D copies |
| 1 | Shared memory tiling | Threads cooperatively load an input tile plus halo into shared memory (16x16 output tile uses an 18x18 input tile) to cut redundant global reads |
| 2 | Kernel fusion | Conv + Bias + ReLU in a single kernel, so intermediates stay in registers instead of going through global memory |
| 3 | Optimized backward pass | Block-level tree reduction in shared memory for weight gradients, so only one `atomicAdd` per block instead of one per thread |
| 4 | Stream overlap | Two CUDA streams, ping-pong device buffers, and pinned host memory so the next batch is copied while the current one computes |

## Results

Measured on an NVIDIA GeForce RTX 4070 Laptop GPU (compute capability 8.9) with Nsight Systems, 1 epoch, learning rate 0.01.

"Kernel time" is the sum of all CUDA kernel durations reported by Nsight Systems. It excludes host-device transfers and CPU-side preprocessing.

**Batch 64, block 256, tile 16x16**

| Version | Kernel time (us) |
|---|---|
| Baseline (naive) | 4,199,899 |
| + Opt 1 (tiled conv) | 4,181,863 |
| + Opt 2 (fused conv+bias+ReLU) | 4,296,950 |
| + Opt 3 and 4 (atomic reduction + streams) | 428,901 |

**Optimization 3+4 across batch sizes (block 256)**

| Batch | Opt 2 (us) | Opt 3+4 (us) |
|---|---|---|
| 64 | 4,296,950 | 428,901 |
| 32 | 4,475,917 | 590,936 |
| 8 | 4,978,762 | 1,270,796 |

**Key takeaways**

- Tiling and fusion made almost no difference on MNIST. With 28x28 images and 3x3 kernels, L1/L2 caches already capture most of the reuse, so shared memory tiling pays off mainly for larger images or kernels.
- The big win came from the backward pass: replacing per-thread atomics with a block-level reduction cut kernel time by roughly 10x at batch 64. Opt 3 and Opt 4 were measured together, so their individual contributions are not separated.
- Batch 64 performed best. Block size 256 was best overall, and block size 1024 was clearly worse.
- Tile size 8x8 vs 16x16 made a negligible difference (about 1%).
- After optimization, GPU utilization looks lower because kernels finish so quickly. The next bottleneck is CPU-side batch preparation, not GPU compute.

## Project structure

```
project/
  include/                  # Headers
    activations.h           # ReLU
    conv_layer.h            # Convolution interface
    conv_tile.h             # Tiled convolution
    conv_fused.h            # Fused Conv + Bias + ReLU
    conv_backward.h         # Optimized backward
    fc_layer.h
    pool_layer.h
    loss.h                  # Softmax + cross-entropy
    optimizer.h             # SGD
    tensor.h                # NCHW tensor struct
    timer.h                 # CUDA event timing
    utils.h                 # Error checking
    mnist_loader.h
  src/
    main.cu                 # Entry point
    train.cu                # Training loop (baseline, tiling)
    train_fues.cu           # Training loop (fusion)
    train_op.cu             # Training loop (final: reduction + streams)
    mnist_loader.cpp        # MNIST IDX parser
    layers/                 # conv_forward/backward (baseline), fc_*, pool_*
    ops/                    # relu_activate, sgd_update, softmax_loss
    optimize/               # conv_tile, conv_fuse, conv_back_tile
    utils/                  # cuda_utils, timer
  Makefile
  build.py
```

## Requirements

- NVIDIA GPU with compute capability 6.0 or newer (tested on 8.9)
- CUDA Toolkit 12.4 or newer (tested with 13.1)
- `make` or Python 3
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`) for profiling

Check your setup:

```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version
```

> The build commands below use `-arch=sm_89` (RTX 4070, compute capability 8.9). Change it to match your GPU. On Windows, `nvcc` also needs the Visual Studio C++ build tools on your path.

## Getting started

### 1. Download MNIST

```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
```

The run command below expects the file names with a dot before `idx3` / `idx1`, so rename them:

```bash
mv train-images-idx3-ubyte train-images.idx3-ubyte
mv train-labels-idx1-ubyte train-labels.idx1-ubyte
mv t10k-images-idx3-ubyte  t10k-images.idx3-ubyte
mv t10k-labels-idx1-ubyte  t10k-labels.idx1-ubyte
```

If the original host is unavailable, use any MNIST mirror that provides the same IDX files.

### 2. Build

```bash
make
# or
python3 build.py
```

Or compile directly with `nvcc`. This example (Windows `cmd`) builds the final version with all four optimizations:

```bat
nvcc -O3 -std=c++14 -arch=sm_89 -Iinclude ^
src/main.cu ^
src/train_op.cu ^
src/optimize/conv_fuse.cu ^
src/layers/pool_forward.cu ^
src/layers/pool_backward.cu ^
src/layers/fc_forward.cu ^
src/layers/fc_backward.cu ^
src/ops/softmax_loss.cu ^
src/ops/sgd_update.cu ^
src/optimize/conv_back_tile.cu ^
src/ops/relu_activate.cu ^
src/utils/cuda_utils.cu ^
src/utils/timer.cu ^
src/mnist_loader.cpp ^
-o cnn
```

On Linux or macOS, replace each `^` with `\`.

#### Building each optimization level

All versions share the same source files except for three. To build an earlier stage, swap these three entries in the command above (or in the `CU_SRCS` list in `build.py`):

| Version | Training loop | Conv forward | Conv backward |
|---|---|---|---|
| Baseline | `train.cu` | `layers/conv_forward.cu` | `layers/conv_backward.cu` |
| Opt 1 (tiling) | `train.cu` | `optimize/conv_tile.cu` | `layers/conv_backward.cu` |
| Opt 2 (fusion) | `train_fues.cu` | `optimize/conv_fuse.cu` | `layers/conv_backward.cu` |
| Opt 3+4 (reduction + streams) | `train_op.cu` | `optimize/conv_fuse.cu` | `optimize/conv_back_tile.cu` |

Every other file in the command (pooling, fully connected, softmax/loss, SGD, ReLU, utils, `main.cu`, `mnist_loader.cpp`) stays the same.

### 3. Train

Windows `cmd`:

```bat
cnn.exe --train-images train-images.idx3-ubyte --train-labels train-labels.idx1-ubyte ^
--test-images t10k-images.idx3-ubyte --test-labels t10k-labels.idx1-ubyte ^
--epochs 14 --batch 8 --lr 0.01 --block 256
```

Linux or macOS:

```bash
./cnn --train-images train-images.idx3-ubyte --train-labels train-labels.idx1-ubyte \
      --test-images t10k-images.idx3-ubyte --test-labels t10k-labels.idx1-ubyte \
      --epochs 14 --batch 8 --lr 0.01 --block 256
```

| Flag | Meaning |
|---|---|
| `--epochs` | Number of training epochs |
| `--batch` | Mini-batch size (8, 32, 64 were tested) |
| `--lr` | SGD learning rate |
| `--block` | Threads per block (256, 512, 1024 were tested) |

### 4. Clean

```bash
make clean
```

## Profiling

```bash
nsys profile --stats=true ./cnn \
  --train-images train-images.idx3-ubyte \
  --train-labels train-labels.idx1-ubyte \
  --test-images  t10k-images.idx3-ubyte \
  --test-labels  t10k-labels.idx1-ubyte \
  --epochs 1 --batch 64 --lr 0.01 --block 256
```

## Experiment parameters

| Parameter | Values tested |
|---|---|
| Batch size | 8, 32, 64 |
| Block size (threads) | 256, 512, 1024 |
| Tile size | 8x8, 16x16 |
| Epochs | 1 (profiling runs), 14 (full training run) |

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/): 60,000 training and 10,000 test images (28x28 grayscale, 10 classes), stored in IDX format. Pixels are normalized from 0 to 255 into 0.0 to 1.0.

## Future work

- Move batch preparation off the critical path (the CPU is now the bottleneck)
- Separate the individual contributions of the atomic reduction and stream overlap
- Use Nsight Compute to study occupancy, memory throughput, and stall reasons per kernel
- Evaluate tiling on larger inputs and kernels where it should matter more

## License

Released under the [MIT License](LICENSE).
