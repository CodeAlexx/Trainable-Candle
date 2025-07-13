use candle::{DType, Device, Module, Result, Shape, Tensor, Var};

// Generic Linear layer that can work with both Tensor and Var
pub struct Linear<T> {
    pub weight: T,
    pub bias: Option<T>,
}

// Implementation for inference (T = Tensor)
impl Module for Linear<Tensor> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.t()?;
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

// Implementation for training (T = Var)
impl Module for Linear<Var> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.as_tensor().t()?;
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias.as_tensor()),
        }
    }
}

// Constructor for training version - returns Linear<Var>
pub fn linear_training(
    in_dim: usize, 
    out_dim: usize, 
    vb: &VarBuilderTrainingPrefixed
) -> Result<Linear<Var>> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

// Also provide a version that takes VarBuilderTraining directly
pub fn linear_training_direct(
    in_dim: usize, 
    out_dim: usize, 
    vb: &VarBuilderTraining
) -> Result<Linear<Var>> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

// Constructor for inference version - returns Linear<Tensor>
pub fn linear(
    in_dim: usize, 
    out_dim: usize, 
    vb: &super::VarBuilder
) -> Result<Linear<Tensor>> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

// VarBuilder for training that returns Var instead of Tensor
pub struct VarBuilderTraining {
    device: Device,
    dtype: DType,
    varmap: super::VarMap,
}

impl VarBuilderTraining {
    pub fn new(dtype: DType, device: Device) -> Self {
        Self {
            device,
            dtype,
            varmap: super::VarMap::new(),
        }
    }
    
    pub fn from_varmap(varmap: &super::VarMap, dtype: DType, device: Device) -> Self {
        Self {
            device,
            dtype,
            varmap: varmap.clone(),
        }
    }
    
    pub fn get<S: Into<Shape>>(&self, shape: S, _name: &str) -> Result<Var> {
        // Create new trainable Var with proper initialization
        let shape = shape.into();
        let bound = 1.0 / (shape.elem_count() as f64).sqrt();
        
        // Create tensor first then wrap in Var
        let tensor = Tensor::randn(0.0f32, bound as f32, shape, &self.device)?
            .to_dtype(self.dtype)?;
        let var = Var::from_tensor(&tensor)?;
        
        // Note: We don't add to varmap here since it expects Tensor not Var
        // The varmap is more for tracking existing variables than creating new ones
        
        Ok(var)
    }
    
    pub fn pp(&self, prefix: &str) -> VarBuilderTrainingPrefixed {
        VarBuilderTrainingPrefixed {
            inner: self,
            prefix: prefix.to_string(),
        }
    }
    
    pub fn varmap(&self) -> &super::VarMap {
        &self.varmap
    }
}

// Prefixed version for hierarchical naming
pub struct VarBuilderTrainingPrefixed<'a> {
    inner: &'a VarBuilderTraining,
    prefix: String,
}

impl<'a> VarBuilderTrainingPrefixed<'a> {
    pub fn get<S: Into<Shape>>(&self, shape: S, name: &str) -> Result<Var> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.inner.get(shape, &full_name)
    }
    
    pub fn pp(&self, prefix: &str) -> VarBuilderTrainingPrefixed {
        VarBuilderTrainingPrefixed {
            inner: self.inner,
            prefix: if self.prefix.is_empty() {
                prefix.to_string()
            } else {
                format!("{}.{}", self.prefix, prefix)
            },
        }
    }
}

// LoRA implementation with mixed Tensor/Var
pub struct LinearLoRA {
    pub base: Linear<Tensor>,      // Frozen base weights
    pub lora_a: Option<Linear<Var>>, // Trainable LoRA weights
    pub lora_b: Option<Linear<Var>>, // Trainable LoRA weights
    pub scale: f32,
}

impl LinearLoRA {
    pub fn from_pretrained(
        base_vb: &super::VarBuilder,      // For loading frozen weights
        lora_vb: &VarBuilderTraining, // For creating trainable LoRA
        name: &str,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
    ) -> Result<Self> {
        // Load frozen base weights
        let base = linear(in_dim, out_dim, &base_vb.pp(name))?;
        
        // Create trainable LoRA weights
        let lora_a = linear_training(in_dim, rank, &lora_vb.pp(&format!("{}_lora_a", name)))?;
        let lora_b = linear_training(rank, out_dim, &lora_vb.pp(&format!("{}_lora_b", name)))?;
        
        Ok(Self {
            base,
            lora_a: Some(lora_a),
            lora_b: Some(lora_b),
            scale: 1.0,
        })
    }
    
    pub fn trainable_params(&self) -> Vec<&Var> {
        let mut params = vec![];
        if let Some(ref a) = self.lora_a {
            params.push(&a.weight);
            if let Some(ref b) = &a.bias {
                params.push(b);
            }
        }
        if let Some(ref b) = self.lora_b {
            params.push(&b.weight);
            if let Some(ref b) = &b.bias {
                params.push(b);
            }
        }
        params
    }
}

impl Module for LinearLoRA {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        
        if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
            let lora_out = a.forward(x)?;
            let lora_out = b.forward(&lora_out)?;
            base_out + (lora_out * self.scale as f64)?
        } else {
            Ok(base_out)
        }
    }
}