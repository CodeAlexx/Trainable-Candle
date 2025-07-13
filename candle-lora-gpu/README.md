# Candle LoRA GPU

GPU-accelerated LoRA training for diffusion models using custom CUDA kernels.

## Overview

This project provides optimized CUDA backward propagation kernels specifically designed for LoRA (Low-Rank Adaptation) training. It enables efficient fine-tuning of large diffusion models on consumer GPUs.

## Features

- Custom CUDA kernels for LoRA backward propagation
- Optimized tiled matrix multiplication with shared memory
- Support for FP32 and FP16 mixed precision training
- Gradient accumulation for larger effective batch sizes
- Memory-efficient implementation for 24GB GPUs
- Produces ComfyUI-compatible LoRA checkpoints

## Requirements

- CUDA 11.0 or higher
- GPU with compute capability 7.5+ (RTX 2070 or newer)
- Rust 1.70+
- 24GB+ VRAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/candle-lora-gpu.git
cd candle-lora-gpu

# Build with CUDA support
cargo build --release --features cuda

# Run tests
cargo test --features cuda
```

## Usage

### Training a LoRA

```bash
# Train SDXL LoRA
cargo run --release --features cuda --bin train-lora -- \
  --model sdxl \
  --dataset /path/to/images \
  --output ./lora_output \
  --rank 32 \
  --alpha 32 \
  --steps 1000 \
  --lr 1e-4 \
  --fp16
```

### Using as a Library

```rust
use candle_lora_gpu::{LoRABackwardOps, GradientAccumulator};
use candle_core::{Device, Tensor};

// Initialize CUDA
candle_lora_gpu::initialize_cuda()?;

// Create tensors
let device = Device::cuda_if_available(0)?;
let grad_output = Tensor::randn(0.0, 1.0, (batch, seq, out_features), &device)?;
let input = Tensor::randn(0.0, 1.0, (batch, seq, in_features), &device)?;

// Compute LoRA gradients
let (grad_down, grad_up) = LoRABackwardOps::backward(
    &grad_output,
    &input,
    &lora_down,
    &lora_up,
    scale,
)?;
```

## Performance

Benchmarks on RTX 4090:

| Model | Batch | Sequence | Hidden | Rank | Time (μs) |
|-------|-------|----------|--------|------|-----------|
| SDXL  | 1     | 77       | 768    | 16   | 125       |
| SD3.5 | 1     | 256      | 1536   | 32   | 487       |
| Flux  | 1     | 256      | 3072   | 64   | 1,823     |

## Architecture

The project implements:
- **LoRA Backward**: Optimized gradient computation for LoRA adapters
- **Norm Backward**: GroupNorm and RMSNorm gradients for different architectures
- **Attention Backward**: Efficient attention gradient computation
- **Mixed Precision**: FP16 compute with FP32 accumulation

## License

This project is dual-licensed under MIT and Apache 2.0 licenses.