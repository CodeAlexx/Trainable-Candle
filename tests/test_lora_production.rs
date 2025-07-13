#[cfg(test)]
mod tests {
    use candle_core::{Device, DType, Tensor, Var};
    use std::time::Instant;
    
    #[cfg(feature = "cuda")]
    use candle_lora_gpu::{LoRABackwardOps, NormBackwardOps, AttentionBackwardOps};
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_lora_backward_correctness() -> candle_core::Result<()> {
        let device = Device::cuda_if_available(0)?;
        
        // Test parameters
        let batch_size = 2;
        let seq_len = 77;
        let in_features = 768;
        let out_features = 768;
        let rank = 16;
        let alpha = 32.0;
        let scale = alpha / rank as f32;
        let eps = 1e-4;
        
        // Create test tensors
        let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, in_features), &device)?;
        let lora_down = Tensor::randn(0f32, 0.02, (rank, in_features), &device)?;
        let lora_up = Tensor::randn(0f32, 0.02, (out_features, rank), &device)?;
        let grad_output = Tensor::randn(0f32, 1.0, (batch_size, seq_len, out_features), &device)?;
        
        // Compute analytical gradients using GPU kernel
        let (grad_down_gpu, grad_up_gpu) = LoRABackwardOps::backward(
            &grad_output,
            &input,
            &lora_down,
            &lora_up,
            scale,
        )?;
        
        // Numerical gradient check for one element
        let check_r = 0;
        let check_i = 0;
        
        // Forward with perturbation
        let mut down_vec = lora_down.to_vec2::<f32>()?;
        down_vec[check_r][check_i] += eps;
        let lora_down_plus = Tensor::new(down_vec.clone(), &device)?;
        
        down_vec[check_r][check_i] -= 2.0 * eps;
        let lora_down_minus = Tensor::new(down_vec, &device)?;
        
        // Compute outputs
        let input_flat = input.reshape((batch_size * seq_len, in_features))?;
        let out_plus = input_flat.matmul(&lora_down_plus.t()?)?.matmul(&lora_up.t()?)?.affine(scale as f64, 0.0)?;
        let out_minus = input_flat.matmul(&lora_down_minus.t()?)?.matmul(&lora_up.t()?)?.affine(scale as f64, 0.0)?;
        
        // Compute loss
        let grad_flat = grad_output.reshape((batch_size * seq_len, out_features))?;
        let loss_plus = (out_plus * &grad_flat).sum_all()?;
        let loss_minus = (out_minus * &grad_flat).sum_all()?;
        
        let numerical_grad = (loss_plus.to_scalar::<f32>()? - loss_minus.to_scalar::<f32>()?) / (2.0 * eps);
        let analytical_grad = grad_down_gpu.i((check_r, check_i))?.to_scalar::<f32>()?;
        
        let error = (numerical_grad - analytical_grad).abs() / (numerical_grad.abs() + analytical_grad.abs() + 1e-8);
        
        println!("Gradient check:");
        println!("  Numerical: {:.6}", numerical_grad);
        println!("  Analytical: {:.6}", analytical_grad);
        println!("  Error: {:.6}", error);
        
        assert!(error < 0.01, "Gradient error too large: {}", error);
        
        Ok(())
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_group_norm_backward() -> candle_core::Result<()> {
        let device = Device::cuda_if_available(0)?;
        
        let batch = 2;
        let channels = 320;
        let height = 64;
        let width = 64;
        let groups = 32;
        
        // Create test tensors
        let input = Tensor::randn(0f32, 1.0, (batch, channels, height, width), &device)?;
        let grad_output = Tensor::randn(0f32, 1.0, (batch, channels, height, width), &device)?;
        let weight = Tensor::ones(channels, DType::F32, &device)?;
        
        // Compute mean and rstd
        let input_reshaped = input.reshape((batch, groups, channels / groups, height * width))?;
        let mean = input_reshaped.mean_keepdim(2)?.mean_keepdim(3)?;
        let var = input_reshaped.var_keepdim(2)?.var_keepdim(3)?;
        let rstd = var.add_scalar(1e-5)?.powf(-0.5)?;
        
        let mean = mean.reshape((batch, groups))?;
        let rstd = rstd.reshape((batch, groups))?;
        
        // Compute backward
        let (grad_input, grad_weight, grad_bias) = NormBackwardOps::group_norm_backward(
            &grad_output,
            &input,
            &mean,
            &rstd,
            Some(&weight),
            groups,
        )?;
        
        // Basic checks
        assert_eq!(grad_input.shape(), input.shape());
        assert_eq!(grad_weight.as_ref().unwrap().dims(), &[channels]);
        assert_eq!(grad_bias.as_ref().unwrap().dims(), &[channels]);
        
        // Check gradients are finite
        let grad_sum = grad_input.sum_all()?.to_scalar::<f32>()?;
        assert!(grad_sum.is_finite(), "GroupNorm backward produced NaN/Inf");
        
        Ok(())
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn benchmark_lora_backward() -> candle_core::Result<()> {
        let device = Device::cuda_if_available(0)?;
        
        println!("\n=== LoRA Backward Benchmark ===");
        
        let configs = vec![
            ("SDXL", 1, 77, 768, 768, 16),
            ("SDXL Large", 1, 77, 1280, 1280, 32),
            ("SD3.5", 1, 256, 1536, 1536, 32),
            ("Flux", 1, 256, 3072, 3072, 64),
        ];
        
        for (name, batch, seq, in_feat, out_feat, rank) in configs {
            let input = Tensor::randn(0f32, 1.0, (batch, seq, in_feat), &device)?;
            let grad_output = Tensor::randn(0f32, 1.0, (batch, seq, out_feat), &device)?;
            let lora_down = Tensor::randn(0f32, 0.02, (rank, in_feat), &device)?;
            let lora_up = Tensor::randn(0f32, 0.02, (out_feat, rank), &device)?;
            
            // Warmup
            for _ in 0..10 {
                let _ = LoRABackwardOps::backward(&grad_output, &input, &lora_down, &lora_up, 1.0)?;
            }
            device.synchronize()?;
            
            // Benchmark
            let iterations = 100;
            let start = Instant::now();
            
            for _ in 0..iterations {
                let _ = LoRABackwardOps::backward(&grad_output, &input, &lora_down, &lora_up, 1.0)?;
            }
            device.synchronize()?;
            
            let elapsed = start.elapsed();
            let avg_time = elapsed.as_micros() as f64 / iterations as f64;
            
            println!("{:15} - batch={}, seq={:3}, in={:4}, out={:4}, rank={:2} -> {:.2} μs/iter",
                name, batch, seq, in_feat, out_feat, rank, avg_time);
        }
        
        Ok(())
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_memory_efficiency() -> candle_core::Result<()> {
        let device = Device::cuda_if_available(0)?;
        
        println!("\n=== Memory Efficiency Test ===");
        
        // Large batch test
        let batch_size = 4;
        let seq_len = 1024;
        let hidden = 1280;
        let rank = 32;
        
        let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden), &device)?;
        let grad_output = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden), &device)?;
        let lora_down = Tensor::randn(0f32, 0.02, (rank, hidden), &device)?;
        let lora_up = Tensor::randn(0f32, 0.02, (hidden, rank), &device)?;
        
        // This should not OOM on 24GB GPU
        let start = Instant::now();
        let (grad_down, grad_up) = LoRABackwardOps::backward(
            &grad_output,
            &input,
            &lora_down,
            &lora_up,
            1.0,
        )?;
        let elapsed = start.elapsed();
        
        println!("Large batch backward:");
        println!("  Input: {:?}", input.shape());
        println!("  Grad down: {:?}", grad_down.shape());
        println!("  Grad up: {:?}", grad_up.shape());
        println!("  Time: {:?}", elapsed);
        
        Ok(())
    }
    
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gradient_accumulation() -> candle_core::Result<()> {
        use candle_lora_gpu::GradientAccumulator;
        
        let device = Device::cuda_if_available(0)?;
        let accumulator = GradientAccumulator::new(device.clone());
        
        // Accumulate gradients over multiple steps
        for i in 0..4 {
            let grad = Tensor::ones((16, 768), DType::F32, &device)?.affine((i + 1) as f64, 0.0)?;
            accumulator.accumulate("test_param", &grad)?;
        }
        
        // Should have accumulated 1 + 2 + 3 + 4 = 10
        let accumulated = accumulator.get_and_reset("test_param").unwrap();
        let sum = accumulated.mean_all()?.to_scalar::<f32>()?;
        
        assert!((sum - 10.0).abs() < 1e-6, "Expected 10.0, got {}", sum);
        
        // Should be cleared now
        assert!(accumulator.get_and_reset("test_param").is_none());
        
        Ok(())
    }
}