#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <algorithm>
#include "kernels.h"

// Simple LoRA backward kernel - functional but not optimized
__global__ void lora_backward_kernel_f32(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    const float* __restrict__ lora_down,
    const float* __restrict__ lora_up,
    float* __restrict__ grad_down,
    float* __restrict__ grad_up,
    const int batch_seq,
    const int in_features,
    const int out_features,
    const int rank,
    const float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute grad_down: grad_out @ lora_up.T @ input.T
    if (idx < rank * in_features) {
        const int r = idx / in_features;
        const int i = idx % in_features;
        
        float sum = 0.0f;
        for (int b = 0; b < batch_seq; b++) {
            for (int o = 0; o < out_features; o++) {
                const float g = grad_out[b * out_features + o];
                const float u = lora_up[o * rank + r];
                const float x = input[b * in_features + i];
                sum += g * u * x * scale;
            }
        }
        grad_down[r * in_features + i] = sum;
    }
}

// Optimized tiled LoRA backward with shared memory
template<typename T, int TILE_SIZE = 16>
__global__ void lora_backward_optimized(
    const T* __restrict__ grad_out,
    const T* __restrict__ input,
    const T* __restrict__ lora_down,
    const T* __restrict__ lora_up,
    T* __restrict__ grad_down,
    T* __restrict__ grad_up,
    const int batch_seq,
    const int in_features,
    const int out_features,
    const int rank,
    const float scale
) {
    __shared__ T tile_grad[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ T tile_input[TILE_SIZE][TILE_SIZE + 1];
    __shared__ T tile_up[TILE_SIZE][TILE_SIZE + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Compute grad_down using tiled matrix multiplication
    if (bx * TILE_SIZE < rank && by * TILE_SIZE < in_features) {
        const int r = bx * TILE_SIZE + tx;
        const int i = by * TILE_SIZE + ty;
        
        T sum = 0;
        
        // Loop over tiles
        for (int tile_b = 0; tile_b < (batch_seq + TILE_SIZE - 1) / TILE_SIZE; tile_b++) {
            for (int tile_o = 0; tile_o < (out_features + TILE_SIZE - 1) / TILE_SIZE; tile_o++) {
                // Load tiles into shared memory
                const int b = tile_b * TILE_SIZE + ty;
                const int o = tile_o * TILE_SIZE + tx;
                
                if (b < batch_seq && o < out_features && ty < TILE_SIZE && tx < TILE_SIZE) {
                    tile_grad[ty][tx] = grad_out[b * out_features + o];
                } else {
                    tile_grad[ty][tx] = 0;
                }
                
                if (b < batch_seq && i < in_features && ty < TILE_SIZE) {
                    tile_input[ty][tx] = (tx == ty) ? input[b * in_features + i] : 0;
                } else {
                    tile_input[ty][tx] = 0;
                }
                
                if (o < out_features && r < rank && tx < TILE_SIZE) {
                    tile_up[ty][tx] = (ty == tx) ? lora_up[o * rank + r] : 0;
                } else {
                    tile_up[ty][tx] = 0;
                }
                
                __syncthreads();
                
                // Compute partial products
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; k++) {
                    sum += tile_grad[k][tx] * tile_up[tx][k] * tile_input[k][ty] * scale;
                }
                
                __syncthreads();
            }
        }
        
        // Write result
        if (r < rank && i < in_features) {
            grad_down[r * in_features + i] = sum;
        }
    }
}

// Kernel launchers
extern "C" {

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
    void* stream
) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // Launch grad_down kernel
    const int threads = 256;
    const int blocks = (rank * in_features + threads - 1) / threads;
    
    lora_backward_kernel_f32<<<blocks, threads, 0, cuda_stream>>>(
        grad_out, input, lora_down, lora_up, grad_down, grad_up,
        batch_seq, in_features, out_features, rank, scale
    );
    
    // For grad_up, we need: input @ lora_down.T @ grad_out.T
    // This is computed similarly but transposed
    const int blocks_up = (out_features * rank + threads - 1) / threads;
    
    // Simple kernel for grad_up (can be optimized similarly)
    auto compute_grad_up = [=] __device__ (int idx) {
        if (idx < out_features * rank) {
            const int o = idx / rank;
            const int r = idx % rank;
            
            float sum = 0.0f;
            for (int b = 0; b < batch_seq; b++) {
                float inner_sum = 0.0f;
                for (int i = 0; i < in_features; i++) {
                    inner_sum += input[b * in_features + i] * lora_down[r * in_features + i];
                }
                sum += grad_out[b * out_features + o] * inner_sum * scale;
            }
            grad_up[o * rank + r] = sum;
        }
    };
    
    // Lambda kernel for grad_up
    auto grad_up_kernel = [=] __global__ (int n) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) compute_grad_up(idx);
    };
    
    void (*kernel_ptr)(int) = grad_up_kernel;
    kernel_ptr<<<blocks_up, threads, 0, cuda_stream>>>(out_features * rank);
}

void launch_lora_backward_optimized(
    const void* grad_out,
    const void* input,
    const void* lora_down,
    const void* lora_up,
    void* grad_down,
    void* grad_up,
    int batch_seq,
    int in_features,
    int out_features,
    int rank,
    float scale,
    int dtype,
    void* stream
) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    // Use 2D thread blocks for better tiling
    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks_down((rank + TILE_SIZE - 1) / TILE_SIZE, 
                     (in_features + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch optimized kernel based on dtype
    if (dtype == 0) { // f32
        lora_backward_optimized<float, TILE_SIZE><<<blocks_down, threads, 0, cuda_stream>>>(
            (const float*)grad_out, (const float*)input, 
            (const float*)lora_down, (const float*)lora_up,
            (float*)grad_down, (float*)grad_up,
            batch_seq, in_features, out_features, rank, scale
        );
    } else if (dtype == 1) { // f16
        lora_backward_optimized<__half, TILE_SIZE><<<blocks_down, threads, 0, cuda_stream>>>(
            (const __half*)grad_out, (const __half*)input,
            (const __half*)lora_down, (const __half*)lora_up,
            (__half*)grad_down, (__half*)grad_up,
            batch_seq, in_features, out_features, rank, scale
        );
    }
    
    // Similar optimization for grad_up computation...
}

} // extern "C"