#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "kernels.h"

// GroupNorm backward kernel
__global__ void group_norm_backward_kernel(
    const float* grad_out,
    const float* input,
    const float* mean,
    const float* rstd,
    const float* weight,
    float* grad_input,
    float* grad_weight,
    float* grad_bias,
    int n, int c, int h, int w, int g
) {
    // Implementation for GroupNorm backward
    // This is a simplified version - production would need proper reduction
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * c * h * w) {
        int batch = idx / (c * h * w);
        int channel = (idx / (h * w)) % c;
        int group = channel / (c / g);
        
        float norm_val = (input[idx] - mean[batch * g + group]) * rstd[batch * g + group];
        grad_input[idx] = grad_out[idx] * (weight ? weight[channel] : 1.0f) * rstd[batch * g + group];
    }
}

extern "C" {

void launch_group_norm_backward(
    const void* grad_out,
    const void* input,
    const void* mean,
    const void* rstd,
    const void* weight,
    void* grad_input,
    void* grad_weight,
    void* grad_bias,
    int n, int c, int h, int w, int g,
    int dtype,
    void* stream
) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    int total_elements = n * c * h * w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    if (dtype == 0) { // f32
        group_norm_backward_kernel<<<blocks, threads, 0, cuda_stream>>>(
            (const float*)grad_out, (const float*)input,
            (const float*)mean, (const float*)rstd,
            (const float*)weight,
            (float*)grad_input, (float*)grad_weight, (float*)grad_bias,
            n, c, h, w, g
        );
    }
}

void launch_rms_norm_backward(
    const void* grad_out,
    const void* input,
    const void* weight,
    const void* rstd,
    void* grad_input,
    void* grad_weight,
    int batch_size,
    int seq_len,
    int hidden_size,
    int dtype,
    void* stream
) {
    // RMSNorm backward implementation
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    // Simplified stub - production would implement full RMSNorm backward
}

} // extern "C"