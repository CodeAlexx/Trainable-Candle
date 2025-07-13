# Trainable-Candle: Pure Rust Training for Neural Networks

A fork of [Candle](https://github.com/huggingface/candle) that enables true gradient-based training for deep learning models in pure Rust, without any Python or PyTorch dependencies.

## Why This Fork Exists

Candle is an excellent pure-Rust deep learning framework, but it has a critical limitation: **VarBuilder is designed for inference only**. When you load model weights using VarBuilder, you get immutable `Tensor` objects instead of trainable `Var` objects. This makes training impossible without workarounds.

This fork solves that problem, enabling:
- ✅ Training neural networks in pure Rust
- ✅ Fine-tuning large models (SDXL, SD3.5, Flux)
- ✅ LoRA/DoRA adapter training
- ✅ Gradient checkpointing for memory efficiency
- ✅ 8-bit optimizers for reduced memory usage

## Key Innovation: Bypassing Python and PyTorch

This fork represents a major contribution to the Rust ML ecosystem by providing a **100% Python-free training solution**. While other frameworks require Python bindings or PyTorch backends, Trainable-Candle enables:

- **Zero Python dependencies**: Train models without any Python runtime
- **Native Rust performance**: No FFI overhead or GIL limitations
- **True systems programming**: Full control over memory, threading, and optimization
- **Deployment ready**: Ship a single binary without Python environment

## Core Features

### 1. Generic Linear Layer
The key innovation that enables training with frozen base weights:

```rust
use candle_nn::{Linear, Module};
use candle_core::{Tensor, Var, Device, DType};

// Generic over tensor type - can be Tensor (frozen) or Var (trainable)
pub struct Linear<T: AsRef<Tensor>> {
    weight: T,
    bias: Option<T>,
}

impl Linear<Var> {
    // Create trainable linear layer
    pub fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let weight = Var::randn(0.0, 0.02, (out_dim, in_dim), device)?;
        let bias = Var::zeros((out_dim,), DType::F32, device)?;
        Ok(Self { weight, bias: Some(bias) })
    }
}

impl<T: AsRef<Tensor>> Module for Linear<T> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.as_ref().t()?;
        let y = x.matmul(&w)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b.as_ref()),
            None => Ok(y),
        }
    }
}
```

### 2. LoRA Training Example
Train LoRA adapters while keeping base model frozen:

```rust
use trainable_candle::lora::LoRALinear;

// Load base model weights as immutable Tensors
let base_weights = candle_core::safetensors::load("model.safetensors", &device)?;
let base_weight = base_weights.get("layer.weight").unwrap();

// Create trainable LoRA adapter
let lora = LoRALinear::new(
    base_weight.clone(),  // Frozen base weight
    rank: 16,             // LoRA rank
    alpha: 16.0,          // LoRA alpha
    &device
)?;

// Forward pass combines frozen + trainable
let output = lora.forward(&input)?;

// Only LoRA weights get gradients
let loss = mse_loss(&output, &target)?;
let grads = loss.backward()?;

// Update only LoRA parameters
optimizer.step(&lora.get_vars())?;
```

### 3. Gradient Checkpointing
Enable training of large models on consumer GPUs:

```rust
use trainable_candle::gradient_checkpoint::checkpoint;

// Wrap memory-intensive operations
let output = checkpoint(|| {
    // This computation will be recomputed during backward
    expensive_transformer_block(&input)
})?;

// Saves ~40% memory at the cost of ~20% speed
```

### 4. 8-bit Adam Optimizer
Reduce optimizer memory by 75%:

```rust
use trainable_candle::adam8bit::Adam8bit;

let mut optimizer = Adam8bit::new(
    learning_rate: 1e-4,
    weight_decay: 0.01,
);

// Uses 8-bit quantization for momentum terms
optimizer.step(&parameters)?;
```

## Real-World Usage: EriDiffusion

This fork was created specifically to enable [EriDiffusion](https://github.com/EricLBuehler/candle), a pure-Rust implementation of Stable Diffusion training. It powers:

- SDXL LoRA fine-tuning at 1024x1024 resolution
- SD 3.5 training with flow matching
- Flux model adaptation
- All without a single line of Python

## Installation

```toml
[dependencies]
candle-core = { git = "https://github.com/CodeAlexx/Trainable-Candle" }
candle-nn = { git = "https://github.com/CodeAlexx/Trainable-Candle" }
```

## Complete Training Example

```rust
use candle_core::{Device, Tensor, Var, DType};
use trainable_candle::{Linear, Adam, Module};

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Create a simple model with trainable parameters
    let linear1 = Linear::<Var>::new(784, 128, &device)?;
    let linear2 = Linear::<Var>::new(128, 10, &device)?;
    
    // Optimizer
    let mut optimizer = Adam::new(
        vec![&linear1.weight, &linear1.bias, &linear2.weight, &linear2.bias],
        1e-3,
    )?;
    
    // Training loop
    for epoch in 0..10 {
        let input = Tensor::randn(0.0, 1.0, (32, 784), &device)?;
        let target = Tensor::randn(0.0, 1.0, (32, 10), &device)?;
        
        // Forward pass
        let hidden = linear1.forward(&input)?.relu()?;
        let output = linear2.forward(&hidden)?;
        
        // Compute loss
        let loss = mse_loss(&output, &target)?;
        
        // Backward pass
        let grads = loss.backward()?;
        
        // Update weights
        optimizer.step(&grads)?;
        
        println!("Epoch {}: loss = {}", epoch, loss.to_scalar::<f32>()?);
    }
    
    Ok(())
}
```

## Technical Details

### The VarBuilder Problem

Standard Candle code:
```rust
// This returns immutable Tensor - cannot train!
let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
let weight = vb.get((out_dim, in_dim), "weight")?; // Returns Tensor, not Var
```

Our solution:
```rust
// Load weights directly, create Vars manually
let weights = candle_core::safetensors::load("model.safetensors", &device)?;
let frozen_weight = weights.get("weight").unwrap(); // Tensor for frozen layers
let trainable = Var::from_tensor(&fresh_tensor)?;   // Var for trainable parts
```

### Memory Efficient Training

This fork enables training on consumer GPUs through:
1. **Gradient checkpointing**: Recompute activations instead of storing
2. **8-bit optimizers**: Quantize Adam momentum terms
3. **Mixed precision**: Use f16/bf16 where appropriate
4. **Selective training**: Freeze most weights, train only adapters

## Contributing

We welcome contributions! Key areas:
- Additional optimizer implementations
- More backward operations
- CUDA kernel optimizations
- Distributed training support

## License

This project maintains the same license as the original Candle project.

## Acknowledgments

Built on top of the excellent [Candle](https://github.com/huggingface/candle) framework by Hugging Face.