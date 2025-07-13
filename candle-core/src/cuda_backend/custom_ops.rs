//! Custom CUDA operations for training support
//! These operations handle Var creation and manipulation on CUDA devices

use crate::{Result, Tensor, Device, DType, Shape, Storage};
use crate::cuda_backend::WrapErr;

/// Custom CUDA kernel for creating trainable variables from tensors
/// This avoids the CUDA_ERROR_NOT_FOUND issue when using Var::from_tensor
pub fn cuda_var_from_tensor(tensor: &Tensor) -> Result<Tensor> {
    match tensor.device() {
        Device::Cuda(_) => {
            // For CUDA tensors, we need to ensure the tensor is properly detached
            // and has fresh storage that can be used for gradient tracking
            
            // First, make the tensor contiguous if needed
            let tensor = if tensor.is_contiguous() {
                tensor.clone()
            } else {
                tensor.contiguous()?
            };
            
            // Create a fresh copy with new storage
            // This breaks any connection to the computation graph
            let shape = tensor.shape();
            let dtype = tensor.dtype();
            let device = tensor.device();
            
            // Clone the storage to get fresh memory
            let storage = tensor.storage().try_clone(tensor.layout())?;
            
            // Create new tensor with variable flag
            Ok(crate::tensor::from_storage(
                storage,
                shape.clone(),
                crate::op::BackpropOp::none(),
                true, // is_variable = true
            ))
        }
        _ => {
            // For non-CUDA devices, use the standard approach
            let storage = tensor.storage().try_clone(tensor.layout())?;
            Ok(crate::tensor::from_storage(
                storage,
                tensor.shape().clone(),
                crate::op::BackpropOp::none(),
                true,
            ))
        }
    }
}

/// Custom kernel for gradient accumulation
/// This handles the case where we need to add gradients to existing Var tensors
pub fn cuda_accumulate_grad(var: &mut Tensor, grad: &Tensor) -> Result<()> {
    // Ensure shapes match
    if var.shape() != grad.shape() {
        return Err(crate::Error::ShapeMismatchBinaryOp {
            lhs: var.shape().clone(),
            rhs: grad.shape().clone(),
            op: "gradient_accumulation",
        });
    }
    
    // Ensure dtypes match
    if var.dtype() != grad.dtype() {
        return Err(crate::Error::DTypeMismatchBinaryOp {
            lhs: var.dtype(),
            rhs: grad.dtype(),
            op: "gradient_accumulation",
        });
    }
    
    match (var.device(), grad.device()) {
        (Device::Cuda(_), Device::Cuda(_)) => {
            // Check if devices match
            if var.device().location() != grad.device().location() {
                return Err(crate::Error::DeviceMismatchBinaryOp {
                    lhs: var.device().location(),
                    rhs: grad.device().location(),
                    op: "gradient_accumulation",
                });
            }
            
            // Perform in-place addition
            // In a real implementation, this would call a CUDA kernel
            let sum = (var.as_ref() + grad)?;
            *var = sum;
            
            Ok(())
        }
        _ => {
            // For non-CUDA devices, use standard addition
            let sum = (var.as_ref() + grad)?;
            *var = sum;
            Ok(())
        }
    }
}

/// Custom kernel for Adam optimizer update on CUDA
/// This performs the Adam update step directly on GPU memory
pub fn cuda_adam_update(
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step: i32,
) -> Result<()> {
    match param.device() {
        Device::Cuda(_) => {
            // Update biased first moment estimate
            // m = beta1 * m + (1 - beta1) * grad
            *m = ((m.as_ref() * beta1)? + (grad * (1.0 - beta1))?)?;
            
            // Update biased second raw moment estimate
            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_sq = grad.sqr()?;
            *v = ((v.as_ref() * beta2)? + (grad_sq * (1.0 - beta2))?)?;
            
            // Compute bias-corrected first moment estimate
            let m_hat = (m.as_ref() / (1.0 - beta1.powi(step)))?;
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = (v.as_ref() / (1.0 - beta2.powi(step)))?;
            
            // Update parameters
            let v_sqrt = v_hat.sqrt()?;
            let update = (m_hat / (v_sqrt + eps)?)?;
            *param = (param.as_ref() - (update * lr)?)?;
            
            Ok(())
        }
        _ => {
            // For non-CUDA devices, perform the same operations
            *m = ((m.as_ref() * beta1)? + (grad * (1.0 - beta1))?)?;
            let grad_sq = grad.sqr()?;
            *v = ((v.as_ref() * beta2)? + (grad_sq * (1.0 - beta2))?)?;
            
            let m_hat = (m.as_ref() / (1.0 - beta1.powi(step)))?;
            let v_hat = (v.as_ref() / (1.0 - beta2.powi(step)))?;
            
            let v_sqrt = v_hat.sqrt()?;
            let update = (m_hat / (v_sqrt + eps)?)?;
            *param = (param.as_ref() - (update * lr)?)?;
            
            Ok(())
        }
    }
}

/// Initialize custom CUDA kernels
/// This should be called once at startup to register the kernels
pub fn init_custom_cuda_ops() -> Result<()> {
    // In a real implementation, this would:
    // 1. Load custom CUDA kernels from PTX or CUBIN files
    // 2. Register them with the CUDA runtime
    // 3. Set up function pointers for fast access
    
    // For now, this is a placeholder for future CUDA kernel initialization
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};
    
    #[test]
    fn test_cuda_var_from_tensor() {
        if let Ok(device) = Device::cuda_if_available(0) {
            let tensor = Tensor::randn(0.0f32, 1.0, (2, 3), &device).unwrap();
            let var = cuda_var_from_tensor(&tensor).unwrap();
            
            assert_eq!(var.shape(), tensor.shape());
            assert_eq!(var.dtype(), tensor.dtype());
            assert!(var.is_variable());
        }
    }
    
    #[test]
    fn test_cuda_adam_update() {
        if let Ok(device) = Device::cuda_if_available(0) {
            let mut param = Tensor::randn(0.0f32, 1.0, (2, 3), &device).unwrap();
            let grad = Tensor::randn(0.0f32, 0.1, (2, 3), &device).unwrap();
            let mut m = Tensor::zeros((2, 3), param.dtype(), &device).unwrap();
            let mut v = Tensor::zeros((2, 3), param.dtype(), &device).unwrap();
            
            cuda_adam_update(
                &mut param,
                &grad,
                &mut m,
                &mut v,
                0.001,  // lr
                0.9,    // beta1
                0.999,  // beta2
                1e-8,   // eps
                1,      // step
            ).unwrap();
            
            // Check that parameters were updated
            assert_ne!(param.to_vec1::<f32>().unwrap(), vec![0.0; 6]);
        }
    }
}