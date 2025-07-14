//! Example of GPU-accelerated LoRA training
//! 
//! This demonstrates how to use the GPU-optimized LoRA backward operations
//! for faster training of low-rank adapters.

use anyhow::Result;
use candle_core::{Device, DType, Module, Tensor, Var, D};
use candle_nn::VarBuilder;
use std::time::Instant;

// Import GPU operations when available
#[cfg(feature = "cuda-backward")]
use candle_core::lora_backward_ops::LoRABackwardOps;

/// Simple LoRA adapter
struct LoRAAdapter {
    down: Var,
    up: Var,
    scale: f32,
}

impl LoRAAdapter {
    fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, device: &Device) -> Result<Self> {
        // Initialize LoRA matrices
        let down = Var::randn(0.0, 0.02, (rank, in_features), device)?;
        let up = Var::zeros((out_features, rank), DType::F32, device)?;
        
        Ok(Self {
            down,
            up,
            scale: alpha / rank as f32,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Standard LoRA forward: x @ down.T @ up.T * scale
        let h = x.matmul(&self.down.as_tensor().t()?)?;
        let out = h.matmul(&self.up.as_tensor().t()?)?;
        out.affine(self.scale as f64, 0.0)
    }
    
    /// GPU-accelerated backward pass (GPU required)
    fn backward_gpu(&self, grad_output: &Tensor, input: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda-backward")]
        {
            // Use optimized CUDA kernel
            LoRABackwardOps::backward(grad_output, input, self.down.as_tensor(), self.up.as_tensor(), self.scale)
        }
        #[cfg(not(feature = "cuda-backward"))]
        {
            anyhow::bail!("GPU required for LoRA backward. Build with --features cuda-backward")
        }
    }
    
    /// CPU backward pass (fallback)
    fn backward_cpu(&self, grad_output: &Tensor, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Compute gradients manually
        // grad_down = (grad_output @ up) @ input.T * scale
        // grad_up = grad_output.T @ (input @ down.T) * scale
        
        let grad_h = grad_output.matmul(&self.up.as_tensor())?;
        let grad_down = grad_h.t()?.matmul(&input)?.t()?.affine(self.scale as f64, 0.0)?;
        
        let h = input.matmul(&self.down.as_tensor().t()?)?;
        let grad_up = grad_output.t()?.matmul(&h)?.affine(self.scale as f64, 0.0)?;
        
        Ok((grad_down, grad_up))
    }
}

/// Benchmark GPU vs CPU LoRA backward pass
fn benchmark_lora_backward(device: &Device) -> Result<()> {
    println!("\n=== LoRA Backward Pass Benchmark ===");
    
    // Test configurations
    let configs = vec![
        (2, 512, 768, 768, 16),    // (batch, seq_len, in_feat, out_feat, rank)
        (4, 256, 1024, 1024, 32),
        (8, 128, 2048, 2048, 16),
    ];
    
    for (batch, seq_len, in_features, out_features, rank) in configs {
        println!("\nConfig: batch={}, seq_len={}, in={}, out={}, rank={}",
                 batch, seq_len, in_features, out_features, rank);
        
        // Create LoRA adapter
        let lora = LoRAAdapter::new(in_features, out_features, rank, 16.0, device)?;
        
        // Create test tensors
        let input = Tensor::randn(0.0, 1.0, (batch, seq_len, in_features), device)?;
        let grad_output = Tensor::randn(0.0, 1.0, (batch, seq_len, out_features), device)?;
        
        // Warmup
        for _ in 0..5 {
            let _ = lora.backward_cpu(&grad_output, &input)?;
        }
        
        // Benchmark CPU
        let cpu_start = Instant::now();
        let iterations = 20;
        for _ in 0..iterations {
            let (grad_down, grad_up) = lora.backward_cpu(&grad_output, &input)?;
            // Force synchronization
            let _ = grad_down.sum_all()?.to_scalar::<f32>()?;
            let _ = grad_up.sum_all()?.to_scalar::<f32>()?;
        }
        let cpu_time = cpu_start.elapsed();
        
        // Benchmark GPU (if available)
        #[cfg(feature = "cuda-backward")]
        {
            // Warmup
            for _ in 0..5 {
                let _ = lora.backward_gpu(&grad_output, &input)?;
            }
            
            let gpu_start = Instant::now();
            for _ in 0..iterations {
                let (grad_down, grad_up) = lora.backward_gpu(&grad_output, &input)?;
                // Force synchronization
                let _ = grad_down.sum_all()?.to_scalar::<f32>()?;
                let _ = grad_up.sum_all()?.to_scalar::<f32>()?;
            }
            let gpu_time = gpu_start.elapsed();
            
            println!("  CPU: {:.2} ms/iter", cpu_time.as_secs_f32() * 1000.0 / iterations as f32);
            println!("  GPU: {:.2} ms/iter", gpu_time.as_secs_f32() * 1000.0 / iterations as f32);
            println!("  Speedup: {:.2}x", cpu_time.as_secs_f32() / gpu_time.as_secs_f32());
        }
        
        #[cfg(not(feature = "cuda-backward"))]
        {
            println!("  CPU: {:.2} ms/iter", cpu_time.as_secs_f32() * 1000.0 / iterations as f32);
            println!("  GPU: Not available (build with --features cuda-backward)");
        }
    }
    
    Ok(())
}

/// Example training loop with GPU LoRA
fn train_example(device: &Device) -> Result<()> {
    println!("\n=== LoRA Training Example ===");
    
    // Model dimensions
    let batch_size = 4;
    let seq_len = 128;
    let hidden_size = 768;
    let rank = 16;
    let num_steps = 10;
    
    // Create LoRA adapter
    let lora = LoRAAdapter::new(hidden_size, hidden_size, rank, 16.0, device)?;
    
    // Simple SGD optimizer
    let learning_rate = 1e-4;
    
    println!("Training {} steps with GPU acceleration...", num_steps);
    
    for step in 0..num_steps {
        // Generate random data
        let input = Tensor::randn(0.0, 1.0, (batch_size, seq_len, hidden_size), device)?;
        let target = Tensor::randn(0.0, 1.0, (batch_size, seq_len, hidden_size), device)?;
        
        // Forward pass
        let output = lora.forward(&input)?;
        
        // Compute loss (MSE)
        let loss = (output.sub(&target))?.sqr()?.mean_all()?;
        
        // Backward pass to get loss gradient
        let grads = loss.backward()?;
        
        // Get gradient w.r.t output
        let grad_output = grads.get(&output).expect("No gradient for output");
        
        // Compute LoRA gradients using GPU kernel
        let (grad_down, grad_up) = lora.backward_gpu(grad_output, &input)?;
        
        // Update parameters (simple SGD)
        let new_down = lora.down.as_tensor().sub(&(grad_down.affine(learning_rate, 0.0)?))?;
        let new_up = lora.up.as_tensor().sub(&(grad_up.affine(learning_rate, 0.0)?))?;
        
        lora.down.set(&new_down)?;
        lora.up.set(&new_up)?;
        
        if step % 2 == 0 {
            println!("Step {}: loss = {:.6}", step, loss.to_scalar::<f32>()?);
        }
    }
    
    println!("Training complete!");
    Ok(())
}

fn main() -> Result<()> {
    // Check for GPU
    let device = Device::cuda_if_available(0)?;
    
    match &device {
        Device::Cuda(_) => {
            println!("Running on GPU");
            #[cfg(not(feature = "cuda-backward"))]
            println!("ERROR: cuda-backward feature not enabled. GPU is required for this example.");
            #[cfg(not(feature = "cuda-backward"))]
            return Err(anyhow::anyhow!("Build with --features cuda-backward to run this example"));
        }
        Device::Cpu => {
            return Err(anyhow::anyhow!("GPU required. This example does not support CPU execution."));
        }
    }
    
    // Run benchmark
    benchmark_lora_backward(&device)?;
    
    // Run training example
    train_example(&device)?;
    
    Ok(())
}