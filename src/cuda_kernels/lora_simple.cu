// Minimal working LoRA backward kernel
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

__global__ void lora_backward_f32(
    const float* __restrict__ grad_out,    // [batch * seq, out_features]
    const float* __restrict__ input,       // [batch * seq, in_features]  
    const float* __restrict__ lora_down,   // [rank, in_features]
    const float* __restrict__ lora_up,     // [out_features, rank]
    float* __restrict__ grad_down,         // [rank, in_features]
    float* __restrict__ grad_up,           // [out_features, rank]
    const int batch_seq,    // batch_size * seq_len
    const int in_features,
    const int out_features,
    const int rank,
    const float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_down = rank * in_features;
    const int total_up = out_features * rank;
    
    // Each thread computes one element of either grad_down or grad_up
    if (idx < total_down) {
        // Computing grad_down[r, i]
        const int r = idx / in_features;
        const int i = idx % in_features;
        
        float sum = 0.0f;
        
        // sum over batch and output features
        for (int bs = 0; bs < batch_seq; bs++) {
            const float input_val = input[bs * in_features + i];
            
            for (int o = 0; o < out_features; o++) {
                const float grad_out_val = grad_out[bs * out_features + o];
                const float up_val = lora_up[o * rank + r];
                sum += grad_out_val * up_val * input_val;
            }
        }
        
        grad_down[idx] = sum * scale;
    }
    else if (idx < total_down + total_up) {
        // Computing grad_up[o, r]
        const int up_idx = idx - total_down;
        const int o = up_idx / rank;
        const int r = up_idx % rank;
        
        float sum = 0.0f;
        
        // sum over batch and input features
        for (int bs = 0; bs < batch_seq; bs++) {
            const float grad_out_val = grad_out[bs * out_features + o];
            
            for (int i = 0; i < in_features; i++) {
                const float input_val = input[bs * in_features + i];
                const float down_val = lora_down[r * in_features + i];
                sum += grad_out_val * input_val * down_val;
            }
        }
        
        grad_up[up_idx] = sum * scale;
    }
}

// Kernel launcher
void launch_lora_backward_f32(
    const float* grad_out,
    const float* input,
    const float* lora_down,
    const float* lora_up,
    float* grad_down,
    float* grad_up,
    int batch_seq,
    int in_features,
    int out_features,
    int rank,
    float scale,
    cudaStream_t stream
) {
    // Zero out gradient buffers
    cudaMemsetAsync(grad_down, 0, rank * in_features * sizeof(float), stream);
    cudaMemsetAsync(grad_up, 0, out_features * rank * sizeof(float), stream);
    
    // Launch kernel
    const int total_threads = rank * in_features + out_features * rank;
    const int threads_per_block = 256;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    lora_backward_f32<<<num_blocks, threads_per_block, 0, stream>>>(
        grad_out, input, lora_down, lora_up, grad_down, grad_up,
        batch_seq, in_features, out_features, rank, scale
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in lora_backward: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"