//! Adam optimizer for TrainableTensor
//! Based on the Adam8bit implementation but works with TrainableTensor

use crate::trainable_tensor::GradContext;
use candle::{DType, Tensor, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Adam optimizer for TrainableTensor
pub struct AdamTrainable {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    
    // First and second moment estimates
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
    
    // Step counter
    step: usize,
}

impl AdamTrainable {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
        }
    }
    
    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
        }
    }
    
    /// Update learning rate
    pub fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
    
    /// Perform optimization step
    pub fn step(&mut self, grad_context: &Arc<GradContext>) -> Result<()> {
        self.step += 1;
        
        // Get all trainable parameters
        let mut parameters = grad_context.parameters();
        
        for param in parameters.iter_mut() {
            if let Some(grad) = &param.grad {
                // Apply weight decay if configured
                let grad = if self.weight_decay > 0.0 {
                    let param_decay = (&param.tensor * self.weight_decay)?;
                    (grad + &param_decay)?
                } else {
                    grad.clone()
                };
                
                // Initialize states if needed
                if !self.m.contains_key(&param.name) {
                    let zeros = Tensor::zeros_like(&grad)?;
                    self.m.insert(param.name.clone(), zeros.clone());
                    self.v.insert(param.name.clone(), zeros);
                }
                
                // Get current states
                let m = self.m.get_mut(&param.name).unwrap();
                let v = self.v.get_mut(&param.name).unwrap();
                
                // Update biased first moment estimate
                *m = ((m.clone() * self.beta1)? + (grad.clone() * (1.0 - self.beta1))?)?;
                
                // Update biased second raw moment estimate
                let grad_sq = grad.sqr()?;
                *v = ((v.clone() * self.beta2)? + (grad_sq * (1.0 - self.beta2))?)?;
                
                // Compute bias-corrected first moment estimate
                let m_hat = (m.clone() / (1.0 - self.beta1.powi(self.step as i32)))?;
                
                // Compute bias-corrected second raw moment estimate
                let v_hat = (v.clone() / (1.0 - self.beta2.powi(self.step as i32)))?;
                
                // Update parameters
                let v_sqrt = v_hat.sqrt()?;
                let update = (m_hat / (v_sqrt + self.eps)?)?;
                param.tensor = (&param.tensor - (update * self.learning_rate)?)?;
            }
        }
        
        Ok(())
    }
    
    /// Zero all gradients
    pub fn zero_grad(&self, grad_context: &Arc<GradContext>) {
        let mut parameters = grad_context.parameters();
        for param in parameters.iter_mut() {
            param.zero_grad();
        }
    }
    
    /// Get current step count
    pub fn get_step(&self) -> usize {
        self.step
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        let num_params = self.m.len();
        let mut total_elements = 0;
        
        for (_, m) in &self.m {
            total_elements += m.elem_count();
        }
        
        // Each element is stored as f32 (4 bytes) in both m and v
        let memory_bytes = total_elements * 8; // 4 bytes * 2 (m and v)
        
        (num_params, memory_bytes)
    }
}

/// 8-bit quantized version for memory efficiency
pub struct Adam8bitTrainable {
    optimizer: AdamTrainable,
    // 8-bit quantized states
    m_quantized: HashMap<String, QuantizedTensor>,
    v_quantized: HashMap<String, QuantizedTensor>,
}

/// Quantized tensor with scale factor
#[derive(Clone)]
pub struct QuantizedTensor {
    pub data: Tensor, // i8 tensor
    pub scale: f32,   // Scale factor for dequantization
}

impl Adam8bitTrainable {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            optimizer: AdamTrainable::new(learning_rate),
            m_quantized: HashMap::new(),
            v_quantized: HashMap::new(),
        }
    }
    
    /// Quantize tensor to 8-bit
    fn quantize(tensor: &Tensor) -> Result<QuantizedTensor> {
        let tensor_f32 = if tensor.dtype() != DType::F32 {
            tensor.to_dtype(DType::F32)?
        } else {
            tensor.clone()
        };
        
        let abs_max = tensor_f32.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = if abs_max > 0.0 {
            abs_max / 127.5
        } else {
            1.0
        };
        
        let scaled = (tensor_f32 / scale as f64)?;
        let rounded = scaled.round()?;
        let clamped = rounded.clamp(-128.0, 127.0)?;
        let quantized = clamped.to_dtype(DType::U8)?;
        
        Ok(QuantizedTensor {
            data: quantized,
            scale,
        })
    }
    
    /// Dequantize 8-bit tensor back to float
    fn dequantize(quant: &QuantizedTensor) -> Result<Tensor> {
        let float_data = quant.data.to_dtype(DType::F32)?;
        let result = (float_data * quant.scale as f64)?;
        Ok(result)
    }
    
    pub fn step(&mut self, grad_context: &Arc<GradContext>) -> Result<()> {
        // Use the base optimizer but with quantized storage
        self.optimizer.step(grad_context)?;
        
        // Quantize the moment estimates after update
        for (name, m) in &self.optimizer.m {
            self.m_quantized.insert(name.clone(), Self::quantize(m)?);
        }
        for (name, v) in &self.optimizer.v {
            self.v_quantized.insert(name.clone(), Self::quantize(v)?);
        }
        
        // Clear the full precision versions to save memory
        self.optimizer.m.clear();
        self.optimizer.v.clear();
        
        Ok(())
    }
    
    pub fn zero_grad(&self, grad_context: &Arc<GradContext>) {
        self.optimizer.zero_grad(grad_context);
    }
}