# Trainable-Candle Examples

## Basic Neural Network Training

### Simple MLP for MNIST

```rust
use candle_core::{Device, DType, Tensor, Var, Result};
use candle_nn::{Module, Optimizer};

struct MLP {
    fc1: Linear<Var>,
    fc2: Linear<Var>,
    fc3: Linear<Var>,
}

impl MLP {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(784, 128, device)?,
            fc2: Linear::new(128, 64, device)?,
            fc3: Linear::new(64, 10, device)?,
        })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(x)?.relu()?;
        self.fc3.forward(&x)
    }
}

fn train_mnist() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let mut model = MLP::new(&device)?;
    let mut optimizer = AdamW::new(model.parameters(), 1e-3)?;
    
    for epoch in 0..10 {
        for (images, labels) in mnist_dataloader()? {
            // Forward
            let logits = model.forward(&images)?;
            let loss = cross_entropy(&logits, &labels)?;
            
            // Backward
            let grads = loss.backward()?;
            
            // Update
            optimizer.step(&grads)?;
            optimizer.zero_grad()?;
        }
    }
    Ok(())
}
```

## LoRA Fine-tuning Examples

### SDXL LoRA Training

```rust
use trainable_candle::lora::{LoRAConfig, LoRAUNet};
use candle_core::{safetensors, Device};

fn train_sdxl_lora() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Load base SDXL model
    let base_weights = safetensors::load("sdxl-base-1.0.safetensors", &device)?;
    
    // Configure LoRA
    let lora_config = LoRAConfig {
        rank: 16,
        alpha: 16.0,
        target_modules: vec![
            "attn1.to_q",
            "attn1.to_k", 
            "attn1.to_v",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
        ],
        dropout: 0.0,
    };
    
    // Create LoRA UNet
    let mut unet = LoRAUNet::from_weights(base_weights, lora_config, &device)?;
    let mut optimizer = Adam8bit::new(unet.lora_parameters(), 1e-4)?;
    
    // Training loop
    for step in 0..1000 {
        let batch = get_training_batch()?;
        
        // Add noise to latents
        let noise = Tensor::randn_like(&batch.latents)?;
        let timesteps = sample_timesteps(batch.size)?;
        let noisy_latents = add_noise(&batch.latents, &noise, &timesteps)?;
        
        // Predict noise
        let noise_pred = unet.forward(
            &noisy_latents,
            &timesteps,
            &batch.prompt_embeds,
        )?;
        
        // MSE loss
        let loss = mse_loss(&noise_pred, &noise)?;
        
        // Backward and update
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
        
        if step % 100 == 0 {
            println!("Step {}: loss = {}", step, loss.to_scalar::<f32>()?);
        }
    }
    
    // Save LoRA weights
    unet.save_lora("sdxl_lora.safetensors")?;
    Ok(())
}
```

### SD 3.5 LoRA with Flow Matching

```rust
use trainable_candle::sd3::{SD3Config, MMDiTLoRA};

fn train_sd35_lora() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Load SD 3.5 base model
    let base_weights = safetensors::load("sd3.5-large.safetensors", &device)?;
    
    // Create LoRA MMDiT
    let mut mmdit = MMDiTLoRA::new(
        &base_weights,
        rank: 32,
        alpha: 32.0,
        &device,
    )?;
    
    let mut optimizer = AdamW::new(mmdit.lora_parameters(), 5e-5)?;
    
    for step in 0..2000 {
        let batch = get_sd3_batch()?;
        
        // Flow matching training
        let t = Tensor::rand(0.0, 1.0, (batch.size,), &device)?;
        let flow = batch.target - batch.source;
        let x_t = batch.source + t.unsqueeze(1)? * flow;
        
        // Predict velocity
        let v_pred = mmdit.forward(
            &x_t,
            &t,
            &batch.prompt_embeds,
            &batch.pooled_embeds,
        )?;
        
        // Flow matching loss
        let loss = mse_loss(&v_pred, &flow)?;
        
        // SNR weighting
        let snr_weight = compute_snr_weight(&t, 5.0)?;
        let weighted_loss = (loss * snr_weight)?.mean_all()?;
        
        // Update
        let grads = weighted_loss.backward()?;
        optimizer.step(&grads)?;
    }
    
    Ok(())
}
```

### Flux LoRA Training

```rust
use trainable_candle::flux::{FluxLoRA, FluxConfig};

fn train_flux_lora() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Flux requires aggressive memory optimization
    let config = FluxConfig {
        gradient_checkpointing: true,
        use_8bit_adam: true,
        cpu_offload_optimizer: true,
    };
    
    let base_weights = safetensors::load("flux-dev.safetensors", &device)?;
    let mut flux = FluxLoRA::new(&base_weights, 16, 16.0, config, &device)?;
    
    // CPU-offloaded optimizer for memory efficiency
    let mut optimizer = CPUOffloadedAdam::new(flux.lora_parameters(), 1e-4)?;
    
    for step in 0..1000 {
        let batch = get_flux_batch()?;
        
        // Flux uses shifted timestep schedule
        let t = sample_flux_timesteps(batch.size)?;
        let sigma = flux_schedule(&t)?;
        
        // Forward with gradient checkpointing
        let noise_pred = checkpoint(|| {
            flux.forward(
                &batch.latents,
                &t,
                &batch.prompt_embeds,
                &batch.guidance,
            )
        })?;
        
        let loss = flux_loss(&noise_pred, &batch.noise, &sigma)?;
        
        // Backward and update
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
        
        // Clear gradients and cache
        optimizer.zero_grad()?;
        clear_memory_cache()?;
    }
    
    Ok(())
}
```

## Advanced Training Patterns

### Mixed Precision Training

```rust
fn mixed_precision_training() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    
    // Model in float32
    let mut model = create_model(&device, DType::F32)?;
    
    // Training loop
    for batch in dataloader {
        // Convert inputs to bf16
        let inputs_bf16 = batch.inputs.to_dtype(DType::BF16)?;
        
        // Forward in bf16
        let output_bf16 = model.forward(&inputs_bf16)?;
        
        // Loss in float32 for stability
        let output_f32 = output_bf16.to_dtype(DType::F32)?;
        let loss = compute_loss(&output_f32, &batch.targets)?;
        
        // Backward and update in float32
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
    }
    
    Ok(())
}
```

### Gradient Accumulation

```rust
fn gradient_accumulation_training() -> Result<()> {
    let accumulation_steps = 4;
    let mut accumulated_loss = None;
    
    for (i, batch) in dataloader.enumerate() {
        // Forward
        let output = model.forward(&batch.input)?;
        let loss = criterion(&output, &batch.target)?;
        
        // Scale loss by accumulation steps
        let scaled_loss = (loss / accumulation_steps as f32)?;
        
        // Accumulate
        accumulated_loss = match accumulated_loss {
            None => Some(scaled_loss),
            Some(acc) => Some((acc + scaled_loss)?),
        };
        
        // Update every N steps
        if (i + 1) % accumulation_steps == 0 {
            if let Some(loss) = accumulated_loss.take() {
                let grads = loss.backward()?;
                optimizer.step(&grads)?;
                optimizer.zero_grad()?;
            }
        }
    }
    
    Ok(())
}
```

### Custom Training Loop with Validation

```rust
fn train_with_validation() -> Result<()> {
    let mut best_val_loss = f32::MAX;
    
    for epoch in 0..num_epochs {
        // Training
        model.train();
        for batch in train_dataloader {
            let loss = train_step(&mut model, &batch)?;
        }
        
        // Validation
        model.eval();
        let mut val_losses = Vec::new();
        
        for batch in val_dataloader {
            // No gradients needed for validation
            let output = model.forward(&batch.input)?;
            let loss = criterion(&output, &batch.target)?;
            val_losses.push(loss.to_scalar::<f32>()?);
        }
        
        let val_loss = val_losses.iter().sum::<f32>() / val_losses.len() as f32;
        
        // Save best model
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            save_checkpoint(&model, "best_model.safetensors")?;
        }
        
        println!("Epoch {}: val_loss = {}", epoch, val_loss);
    }
    
    Ok(())
}
```

## Memory Optimization Examples

### Training Large Models on Limited VRAM

```rust
fn train_on_24gb_vram() -> Result<()> {
    // Enable all memory optimizations
    std::env::set_var("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512");
    
    let device = Device::cuda_if_available(0)?;
    
    // Load model with CPU offloading
    let model = load_model_with_offloading(&device)?;
    
    // 8-bit optimizer
    let mut optimizer = Adam8bit::new(model.parameters(), 1e-4)?;
    
    // Small batch size with gradient accumulation
    let batch_size = 1;
    let gradient_accumulation = 4;
    
    for step in 0..total_steps {
        for _ in 0..gradient_accumulation {
            // Forward with checkpointing
            let output = checkpoint(|| model.forward(&input))?;
            let loss = compute_loss(&output, &target)?;
            let scaled_loss = (loss / gradient_accumulation as f32)?;
            
            // Accumulate gradients
            let grads = scaled_loss.backward()?;
        }
        
        // Update after accumulation
        optimizer.step()?;
        optimizer.zero_grad()?;
        
        // Clear cache periodically
        if step % 10 == 0 {
            clear_cuda_cache()?;
        }
    }
    
    Ok(())
}
```