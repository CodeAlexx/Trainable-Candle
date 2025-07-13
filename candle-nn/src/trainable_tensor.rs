// Core trainable tensor that wraps Candle's Tensor with gradient tracking
use candle::{Tensor, Device, DType, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Simple gradient tracking - just what we need for LoRA
#[derive(Debug, Clone)]
pub struct TrainableTensor {
    pub tensor: Tensor,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub name: String,
}

impl TrainableTensor {
    pub fn new(tensor: Tensor, name: String) -> Self {
        Self {
            tensor,
            requires_grad: true,
            grad: None,
            name,
        }
    }

    pub fn detach(&self) -> Tensor {
        self.tensor.clone()
    }

    pub fn backward(&mut self, grad: &Tensor) -> Result<()> {
        if self.requires_grad {
            match &mut self.grad {
                Some(existing_grad) => {
                    // Use custom CUDA kernel for gradient accumulation if on CUDA
                    match existing_grad.device() {
                        candle::Device::Cuda(_) => {
                            candle::cuda_backend::cuda_accumulate_grad(existing_grad, grad)?;
                        }
                        _ => {
                            *existing_grad = (existing_grad.as_ref() + grad)?;
                        }
                    }
                }
                None => {
                    self.grad = Some(grad.clone());
                }
            }
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }
}

// Global gradient context for automatic differentiation
pub struct GradContext {
    trainable_tensors: Arc<Mutex<HashMap<String, TrainableTensor>>>,
    forward_hooks: Arc<Mutex<Vec<Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>>>>,
}

impl std::fmt::Debug for GradContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradContext")
            .field("num_parameters", &self.trainable_tensors.lock().unwrap().len())
            .field("num_hooks", &self.forward_hooks.lock().unwrap().len())
            .finish()
    }
}

impl GradContext {
    pub fn new() -> Self {
        Self {
            trainable_tensors: Arc::new(Mutex::new(HashMap::new())),
            forward_hooks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn register_parameter(&self, name: String, tensor: Tensor) -> TrainableTensor {
        let trainable = TrainableTensor::new(tensor, name.clone());
        self.trainable_tensors.lock().unwrap().insert(name, trainable.clone());
        trainable
    }

    pub fn add_forward_hook<F>(&self, hook: F) 
    where 
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static 
    {
        self.forward_hooks.lock().unwrap().push(Box::new(hook));
    }

    pub fn apply_hooks(&self, tensor: &Tensor) -> Result<Tensor> {
        let hooks = self.forward_hooks.lock().unwrap();
        let mut result = tensor.clone();
        for hook in hooks.iter() {
            result = hook(&result)?;
        }
        Ok(result)
    }

    pub fn backward(&self, _loss: &Tensor) -> Result<()> {
        // Simple backward pass - in real implementation would need proper autograd
        // For now, assume gradients are computed externally and passed in
        Ok(())
    }

    pub fn parameters(&self) -> Vec<TrainableTensor> {
        self.trainable_tensors.lock().unwrap().values().cloned().collect()
    }
}

// LoRA layer that can be injected into existing models
#[derive(Debug)]
pub struct LoRALinear {
    pub original_weight: Tensor,
    pub lora_a: TrainableTensor,
    pub lora_b: TrainableTensor,
    pub scale: f64,
    pub rank: usize,
    pub grad_context: Arc<GradContext>,
}

impl LoRALinear {
    pub fn new(
        original_weight: Tensor,
        rank: usize,
        scale: f64,
        device: &Device,
        grad_context: Arc<GradContext>,
    ) -> Result<Self> {
        let (in_features, out_features) = original_weight.dims2()?;
        
        // Initialize LoRA matrices with same dtype as original weight
        let dtype = original_weight.dtype();
        let lora_a_tensor = Tensor::randn(0.0, 1.0, (in_features, rank), device)?
            .to_dtype(dtype)?;
        let lora_b_tensor = Tensor::zeros((rank, out_features), dtype, device)?;
        
        let lora_a = grad_context.register_parameter(
            format!("lora_a_{}x{}", in_features, rank),
            lora_a_tensor,
        );
        let lora_b = grad_context.register_parameter(
            format!("lora_b_{}x{}", rank, out_features),
            lora_b_tensor,
        );

        Ok(Self {
            original_weight,
            lora_a,
            lora_b,
            scale,
            rank,
            grad_context,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Handle batched input [batch, seq_len, in_features]
        let input_shape = input.dims();
        let needs_reshape = input_shape.len() == 3;
        
        let input_2d = if needs_reshape {
            let (b, s, d) = input.dims3()?;
            input.reshape((b * s, d))?
        } else {
            input.clone()
        };
        
        // Original forward pass (need to transpose weight for linear: out = input @ weight.T)
        let original_output = input_2d.matmul(&self.original_weight.t()?)?;
        
        // LoRA forward pass: input @ A @ B
        let lora_output = input_2d
            .matmul(&self.lora_a.tensor)?
            .matmul(&self.lora_b.tensor)?;
        
        // Combine with scaling
        let scaled_lora = (lora_output * self.scale)?;
        let output_2d = (original_output + scaled_lora)?;
        
        // Reshape back if needed
        let final_output = if needs_reshape {
            let (b, s, _) = input.dims3()?;
            let out_features = self.original_weight.dims()[0];
            output_2d.reshape((b, s, out_features))?
        } else {
            output_2d
        };
        
        // Apply any registered hooks
        self.grad_context.apply_hooks(&final_output)
    }
}

// Model hook system to replace Linear layers with LoRA versions
pub trait LoRAInjection {
    fn inject_lora(&mut self, layer_name: &str, rank: usize, scale: f64, grad_context: Arc<GradContext>) -> Result<()>;
}

// Example implementation for a simple Linear layer replacement
pub struct ModelWrapper<T> {
    pub inner: T,
    pub grad_context: Arc<GradContext>,
    pub lora_layers: HashMap<String, LoRALinear>,
}

impl<T> ModelWrapper<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            grad_context: Arc::new(GradContext::new()),
            lora_layers: HashMap::new(),
        }
    }

    pub fn add_lora_to_layer(&mut self, layer_name: &str, original_weight: Tensor, rank: usize, scale: f64) -> Result<()> {
        let device = original_weight.device().clone();
        let lora_layer = LoRALinear::new(
            original_weight,
            rank,
            scale,
            &device,
            self.grad_context.clone(),
        )?;
        
        self.lora_layers.insert(layer_name.to_string(), lora_layer);
        Ok(())
    }

    pub fn get_lora_parameters(&self) -> Vec<TrainableTensor> {
        self.grad_context.parameters()
    }
}

// Simple optimizer for LoRA parameters
pub struct AdamOptimizer {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step: usize,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl AdamOptimizer {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn step(&mut self, parameters: &mut [TrainableTensor]) -> Result<()> {
        self.step += 1;
        
        for param in parameters.iter_mut() {
            if let Some(grad) = &param.grad {
                // Get or initialize momentum terms
                let m = self.m.entry(param.name.clone()).or_insert_with(|| {
                    Tensor::zeros(grad.shape(), grad.dtype(), grad.device()).unwrap()
                });
                let v = self.v.entry(param.name.clone()).or_insert_with(|| {
                    Tensor::zeros(grad.shape(), grad.dtype(), grad.device()).unwrap()
                });

                // Update momentum terms
                *m = ((m.clone() * self.beta1)? + (grad * (1.0 - self.beta1))?)?;
                *v = ((v.clone() * self.beta2)? + (grad.powf(2.0)? * (1.0 - self.beta2))?)?;

                // Bias correction
                let m_corrected = (m.clone() / (1.0 - self.beta1.powi(self.step as i32)))?;
                let v_corrected = (v.clone() / (1.0 - self.beta2.powi(self.step as i32)))?;

                // Update parameter
                let update = (m_corrected / (v_corrected.sqrt()? + self.eps)?)?;
                param.tensor = (&param.tensor - (update * self.lr)?)?;
            }
        }
        
        Ok(())
    }
}

// Usage example
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_lora_training() -> Result<()> {
        let device = Device::Cpu;
        let grad_context = Arc::new(GradContext::new());
        
        // Create a simple linear layer weight
        let original_weight = Tensor::randn(0.0, 1.0, (512, 256), &device)?;
        
        // Create LoRA layer
        let lora_layer = LoRALinear::new(
            original_weight,
            16, // rank
            0.1, // scale
            &device,
            grad_context.clone(),
        )?;
        
        // Forward pass
        let input = Tensor::randn(0.0, 1.0, (32, 512), &device)?;
        let output = lora_layer.forward(&input)?;
        
        // Get parameters for training
        let mut parameters = grad_context.parameters();
        let mut optimizer = AdamOptimizer::new(0.001);
        
        // Training loop would go here
        // optimizer.step(&mut parameters)?;
        
        Ok(())
    }
}

// Key design principles implemented:
// 1. Simple TrainableTensor with gradient tracking
// 2. Global context for parameter management
// 3. Hook system for model modification
// 4. LoRA layer that preserves original weights
// 5. Simple optimizer that works with our gradient system
// 6. No VarBuilder dependency - pure Tensor operations