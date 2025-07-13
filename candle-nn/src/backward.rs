//! Custom backward pass implementation for TrainableTensor
//! This implements automatic differentiation for common operations

use crate::trainable_tensor::{TrainableTensor, GradContext};
use candle::{Tensor, Result};
use std::sync::Arc;
use std::collections::HashMap;

/// Compute gradients for a loss tensor
pub fn backward(loss: &Tensor, _grad_context: &Arc<GradContext>) -> Result<()> {
    // Initialize gradient of loss w.r.t itself as 1.0
    let _ones = Tensor::ones_like(loss)?;
    
    // For now, we'll need to manually compute gradients
    // In a full implementation, this would traverse the computation graph
    // For LoRA training, we can compute gradients analytically
    
    Ok(())
}

/// Compute gradients for MSE loss
pub fn mse_backward(
    predictions: &Tensor,
    targets: &Tensor,
    _grad_context: &Arc<GradContext>,
) -> Result<Tensor> {
    let batch_size = predictions.dims()[0];
    let diff = (predictions - targets)?;
    let grad = (diff * (2.0 / batch_size as f64))?;
    Ok(grad)
}

/// Compute gradients for a linear layer with LoRA
pub fn linear_lora_backward(
    input: &Tensor,
    grad_output: &Tensor,
    lora_a: &mut TrainableTensor,
    lora_b: &mut TrainableTensor,
    scale: f64,
) -> Result<()> {
    // Gradient w.r.t lora_b: grad_output.T @ (input @ lora_a)
    let lora_a_out = input.matmul(&lora_a.tensor)?;
    let grad_b = grad_output.t()?.matmul(&lora_a_out)?;
    let grad_b_scaled = (grad_b * scale)?;
    
    // Gradient w.r.t lora_a: input.T @ (grad_output @ lora_b.T)
    let grad_a_partial = grad_output.matmul(&lora_b.tensor.t()?)?;
    let grad_a = input.t()?.matmul(&grad_a_partial)?;
    let grad_a_scaled = (grad_a * scale)?;
    
    // Accumulate gradients
    lora_a.backward(&grad_a_scaled)?;
    lora_b.backward(&grad_b_scaled)?;
    
    Ok(())
}

/// Compute gradients for cross entropy loss
pub fn cross_entropy_backward(
    logits: &Tensor,
    _targets: &Tensor,
) -> Result<Tensor> {
    let batch_size = logits.dims()[0];
    
    // Softmax
    let max_logits = logits.max_keepdim(1)?;
    let exp_logits = (logits - &max_logits)?.exp()?;
    let sum_exp = exp_logits.sum_keepdim(1)?;
    let probs = (exp_logits / sum_exp)?;
    
    // Gradient is probs - one_hot(targets)
    let grad = probs;
    
    // Subtract 1 from the correct class probabilities
    // This is a simplified version - full implementation would handle various target formats
    
    // Scale by batch size
    let grad = (grad / batch_size as f64)?;
    
    Ok(grad)
}

/// Helper to compute gradients through matrix multiplication
pub fn matmul_backward(
    a: &Tensor,
    b: &Tensor,
    grad_output: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Gradient w.r.t A: grad_output @ B.T
    let grad_a = grad_output.matmul(&b.t()?)?;
    
    // Gradient w.r.t B: A.T @ grad_output
    let grad_b = a.t()?.matmul(grad_output)?;
    
    Ok((grad_a, grad_b))
}

/// Helper to compute gradients through addition
pub fn add_backward(
    grad_output: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Both inputs receive the same gradient
    Ok((grad_output.clone(), grad_output.clone()))
}

/// Helper to compute gradients through multiplication by scalar
pub fn scalar_mul_backward(
    _tensor: &Tensor,
    scalar: f64,
    grad_output: &Tensor,
) -> Result<Tensor> {
    // Gradient w.r.t tensor is scalar * grad_output
    let grad = (grad_output * scalar)?;
    Ok(grad)
}

/// Compute gradients for layer normalization
pub fn layer_norm_backward(
    _input: &Tensor,
    normalized: &Tensor,
    gamma: Option<&Tensor>,
    grad_output: &Tensor,
    _eps: f64,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    // Simplified implementation for now
    // In a full implementation, this would compute proper gradients
    
    // Gradient w.r.t gamma (if present)
    let grad_gamma = if gamma.is_some() {
        let g = (grad_output * normalized)?.sum_keepdim(0)?;
        Some(g) // Sum over batch dimension
    } else {
        None
    };
    
    // Gradient w.r.t beta
    let grad_beta = Some(grad_output.sum_keepdim(0)?);
    
    // Gradient w.r.t input (simplified)
    let grad_input = grad_output.clone();
    
    Ok((grad_input, grad_gamma, grad_beta))
}

/// Gradient computation context that tracks intermediate values
pub struct GradientTape {
    intermediates: HashMap<String, Tensor>,
}

impl GradientTape {
    pub fn new() -> Self {
        Self {
            intermediates: HashMap::new(),
        }
    }
    
    pub fn save(&mut self, name: &str, tensor: Tensor) {
        self.intermediates.insert(name.to_string(), tensor);
    }
    
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.intermediates.get(name)
    }
    
    pub fn clear(&mut self) {
        self.intermediates.clear();
    }
}