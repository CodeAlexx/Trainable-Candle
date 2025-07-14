use crate::{DType, Device, Error, Result, Tensor};
use crate::backend::BackendDevice;

#[cfg(feature = "cuda-backward")]
extern "C" {
    fn launch_lora_backward_f32_production(
        grad_out: *const f32,
        input: *const f32,
        lora_down: *const f32,
        lora_up: *const f32,
        grad_down: *mut f32,
        grad_up: *mut f32,
        batch_seq: i32,
        in_features: i32,
        out_features: i32,
        rank: i32,
        scale: f32,
        stream: *mut std::ffi::c_void,
    );
    
    fn init_cublas_handle();
    fn cleanup_lora_backward();
}

/// Compute LoRA backward pass on GPU
/// Returns (grad_down, grad_up)
#[cfg(feature = "cuda-backward")]
pub fn lora_backward_gpu(
    grad_output: &Tensor,
    input: &Tensor,
    lora_down: &Tensor,
    lora_up: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    // Validate inputs
    if !grad_output.device().is_cuda() {
        return Err(Error::Msg("All tensors must be on CUDA device".into()));
    }
    
    if grad_output.dtype() != DType::F32 {
        return Err(Error::Msg("Only F32 is supported for now".into()));
    }
    
    // Get dimensions
    let grad_shape = grad_output.dims();
    let batch_seq = if grad_shape.len() == 3 {
        grad_shape[0] * grad_shape[1]
    } else if grad_shape.len() == 2 {
        grad_shape[0]
    } else {
        return Err(Error::Msg("grad_output must be 2D or 3D".into()));
    };
    
    let out_features = grad_shape[grad_shape.len() - 1];
    let in_features = input.dim(input.dims().len() - 1)?;
    let rank = lora_down.dim(0)?;
    
    // Validate shapes
    if lora_down.dims() != [rank, in_features] {
        return Err(Error::Msg(format!(
            "lora_down shape mismatch: expected [{}, {}], got {:?}",
            rank, in_features, lora_down.dims()
        )));
    }
    
    if lora_up.dims() != [out_features, rank] {
        return Err(Error::Msg(format!(
            "lora_up shape mismatch: expected [{}, {}], got {:?}",
            out_features, rank, lora_up.dims()
        )));
    }
    
    // Flatten inputs to 2D if needed
    let grad_output_2d = if grad_shape.len() == 3 {
        grad_output.reshape((batch_seq, out_features))?
    } else {
        grad_output.clone()
    };
    
    let input_2d = if input.dims().len() == 3 {
        let input_batch_seq = input.dim(0)? * input.dim(1)?;
        input.reshape((input_batch_seq, in_features))?
    } else {
        input.clone()
    };
    
    // Allocate output tensors
    let device = grad_output.device();
    
    // Get CUDA storage and stream
    let cuda_device = match device {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };
    
    let stream = cuda_device.cuda_stream();
    
    // Get pointers from CUDA storage
    let grad_out_storage = grad_output_2d.storage();
    let grad_out_cuda = match &*grad_out_storage {
        crate::Storage::Cuda(s) => s,
        _ => return Err(Error::Msg("grad_output must be on CUDA".into())),
    };
    
    let input_storage = input_2d.storage();
    let input_cuda = match &*input_storage {
        crate::Storage::Cuda(s) => s,
        _ => return Err(Error::Msg("input must be on CUDA".into())),
    };
    
    let lora_down_storage = lora_down.storage();
    let lora_down_cuda = match &*lora_down_storage {
        crate::Storage::Cuda(s) => s,
        _ => return Err(Error::Msg("lora_down must be on CUDA".into())),
    };
    
    let lora_up_storage = lora_up.storage();
    let lora_up_cuda = match &*lora_up_storage {
        crate::Storage::Cuda(s) => s,
        _ => return Err(Error::Msg("lora_up must be on CUDA".into())),
    };
    
    // Allocate output tensors
    let grad_down = Tensor::zeros((rank, in_features), DType::F32, device)?;
    let grad_up = Tensor::zeros((out_features, rank), DType::F32, device)?;
    
    // Get device pointers - we need to get the slice first, then the raw device pointer
    let grad_out_slice = grad_out_cuda.as_cuda_slice::<f32>()?;
    let input_slice = input_cuda.as_cuda_slice::<f32>()?;
    let lora_down_slice = lora_down_cuda.as_cuda_slice::<f32>()?;
    let lora_up_slice = lora_up_cuda.as_cuda_slice::<f32>()?;
    
    // Get raw device pointers from the slices
    // The slices themselves are device pointers, we cast them directly
    let grad_out_ptr = grad_out_slice as *const _ as *const f32;
    let input_ptr = input_slice as *const _ as *const f32;
    let lora_down_ptr = lora_down_slice as *const _ as *const f32;
    let lora_up_ptr = lora_up_slice as *const _ as *const f32;
    
    // Get mutable storage for outputs and extract pointers in a scope
    let (grad_down_ptr, grad_up_ptr) = {
        let grad_down_storage = grad_down.storage();
        let grad_down_cuda = match &*grad_down_storage {
            crate::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        
        let grad_up_storage = grad_up.storage();
        let grad_up_cuda = match &*grad_up_storage {
            crate::Storage::Cuda(s) => s,
            _ => unreachable!(),
        };
        
        let grad_down_slice = grad_down_cuda.as_cuda_slice::<f32>()?;
        let grad_up_slice = grad_up_cuda.as_cuda_slice::<f32>()?;
        
        let grad_down_ptr = grad_down_slice as *const _ as *mut f32;
        let grad_up_ptr = grad_up_slice as *const _ as *mut f32;
        
        (grad_down_ptr, grad_up_ptr)
    };
    
    // Initialize cuBLAS handle once
    unsafe {
        init_cublas_handle();
    }
    
    // Launch production kernel
    unsafe {
        launch_lora_backward_f32_production(
            grad_out_ptr as *const f32,
            input_ptr as *const f32,
            lora_down_ptr as *const f32,
            lora_up_ptr as *const f32,
            grad_down_ptr as *mut f32,
            grad_up_ptr as *mut f32,
            batch_seq as i32,
            in_features as i32,
            out_features as i32,
            rank as i32,
            scale,
            stream.0 as *mut std::ffi::c_void,
        );
    }
    
    // Synchronize to ensure computation is complete
    cuda_device.synchronize()?;
    
    Ok((grad_down, grad_up))
}

/// Production version of LoRA backward pass using cuBLAS
#[cfg(feature = "cuda-backward")]
pub fn lora_backward_gpu_production(
    grad_output: &Tensor,
    input: &Tensor, 
    lora_down: &Tensor,
    lora_up: &Tensor,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    lora_backward_gpu(grad_output, input, lora_down, lora_up, scale)
}

#[cfg(not(feature = "cuda-backward"))]
pub fn lora_backward_gpu(
    _grad_output: &Tensor,
    _input: &Tensor,
    _lora_down: &Tensor,
    _lora_up: &Tensor,
    _scale: f32,
) -> Result<(Tensor, Tensor)> {
    Err(Error::Msg(
        "CUDA backward kernels not enabled. Rebuild with --features cuda-backward".into()
    ))
}

#[cfg(not(feature = "cuda-backward"))]
pub fn lora_backward_gpu_production(
    _grad_output: &Tensor,
    _input: &Tensor,
    _lora_down: &Tensor,
    _lora_up: &Tensor,
    _scale: f32,
) -> Result<(Tensor, Tensor)> {
    Err(Error::Msg(
        "CUDA backward kernels not enabled. Rebuild with --features cuda-backward".into()
    ))
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(feature = "cuda-backward")]
    fn test_lora_backward_shapes() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        
        // Test dimensions
        let batch_size = 2;
        let seq_len = 10;
        let in_features = 64;
        let out_features = 64;
        let rank = 16;
        let alpha = 32.0;
        let scale = alpha / rank as f32;
        
        // Create test tensors
        let grad_output = Tensor::randn(0f32, 1.0, (batch_size, seq_len, out_features), &device)?;
        let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, in_features), &device)?;
        let lora_down = Tensor::randn(0f32, 0.02, (rank, in_features), &device)?;
        let lora_up = Tensor::randn(0f32, 0.02, (out_features, rank), &device)?;
        
        // Test backward pass
        let (grad_down, grad_up) = lora_backward_gpu(&grad_output, &input, &lora_down, &lora_up, scale)?;
        
        // Check output shapes
        assert_eq!(grad_down.dims(), &[rank, in_features]);
        assert_eq!(grad_up.dims(), &[out_features, rank]);
        
        println!("LoRA backward shape test passed!");
        Ok(())
    }
}