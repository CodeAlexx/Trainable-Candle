//! GPU-accelerated LoRA training for diffusion models
//! 
//! This crate provides optimized CUDA kernels for backward propagation
//! specifically designed for LoRA (Low-Rank Adaptation) training.

#![allow(dead_code)]

// Re-export key types from candle
pub use candle_core::{DType, Device, Error, Result, Tensor, Var};

// Include Rust modules from src/rust/
#[path = "rust/cuda_lora_backward.rs"]
#[cfg(feature = "cuda")]
pub mod cuda_lora_backward;

#[path = "rust/lora_backward_ops.rs"]
pub mod lora_backward_ops;

// Re-export main functionality
#[cfg(feature = "cuda")]
pub use cuda_lora_backward::lora_backward_gpu;

pub use lora_backward_ops::{
    LoRABackwardOps, 
    NormBackwardOps, 
    AttentionBackwardOps,
    GradientAccumulator,
    MixedPrecisionConfig,
    MixedPrecisionManager,
};

/// Initialize CUDA backend
#[cfg(feature = "cuda")]
pub fn initialize_cuda() -> Result<()> {
    // Ensure CUDA is available
    if !candle_core::utils::cuda_is_available() {
        return Err(Error::Msg("CUDA not available".into()));
    }
    
    // Set default CUDA device
    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_available() {
        assert!(candle_core::utils::cuda_is_available());
    }
}