#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cub/cub.cuh>

// GroupNorm backward for UNet models (SD1.5, SD2, SDXL)
template<typename T, int THREADS_PER_BLOCK = 256>
__global__ void group_norm_backward_kernel(
    const T* __restrict__ grad_out,     // [N, C, H, W]
    const T* __restrict__ input,        // [N, C, H, W]
    const T* __restrict__ mean,         // [N, G]
    const T* __restrict__ rstd,         // [N, G] reciprocal std
    const T* __restrict__ weight,       // [C] optional scale
    T* __restrict__ grad_input,         // [N, C, H, W]
    T* __restrict__ grad_weight,        // [C]
    T* __restrict__ grad_bias,          // [C]
    const int N,
    const int C,
    const int H,
    const int W,
    const int G
) {
    const int HW = H * W;
    const int C_per_G = C / G;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C * H * W;
    
    // Shared memory for reductions
    __shared__ T shared_sum1[THREADS_PER_BLOCK];
    __shared__ T shared_sum2[THREADS_PER_BLOCK];
    
    // Each block handles one channel across all batches
    const int c = blockIdx.x;
    if (c >= C) return;
    
    const int g = c / C_per_G;
    const T weight_val = (weight != nullptr) ? weight[c] : T(1);
    
    // First pass: compute grad_weight and grad_bias
    T sum_grad_weight = 0;
    T sum_grad_bias = 0;
    
    for (int n = 0; n < N; n++) {
        const T mean_ng = mean[n * G + g];
        const T rstd_ng = rstd[n * G + g];
        
        for (int hw = threadIdx.x; hw < HW; hw += blockDim.x) {
            const int idx = ((n * C + c) * H * W) + hw;
            const T normalized = (input[idx] - mean_ng) * rstd_ng;
            const T grad_out_val = grad_out[idx];
            
            sum_grad_weight += grad_out_val * normalized;
            sum_grad_bias += grad_out_val;
        }
    }
    
    // Reduce within block
    shared_sum1[threadIdx.x] = sum_grad_weight;
    shared_sum2[threadIdx.x] = sum_grad_bias;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum1[threadIdx.x] += shared_sum1[threadIdx.x + s];
            shared_sum2[threadIdx.x] += shared_sum2[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        if (grad_weight) atomicAdd(&grad_weight[c], shared_sum1[0]);
        if (grad_bias) atomicAdd(&grad_bias[c], shared_sum2[0]);
    }
    
    // Second pass: compute grad_input
    for (int n = 0; n < N; n++) {
        const T mean_ng = mean[n * G + g];
        const T rstd_ng = rstd[n * G + g];
        
        // Compute group statistics for this batch
        T group_sum1 = 0, group_sum2 = 0;
        
        // Sum over all channels in this group
        for (int c_g = 0; c_g < C_per_G; c_g++) {
            const int c_idx = g * C_per_G + c_g;
            const T w = (weight != nullptr) ? weight[c_idx] : T(1);
            
            for (int hw = 0; hw < HW; hw++) {
                const int idx = ((n * C + c_idx) * H * W) + hw;
                const T grad_out_val = grad_out[idx] * w;
                const T normalized = (input[idx] - mean_ng) * rstd_ng;
                
                group_sum1 += grad_out_val;
                group_sum2 += grad_out_val * normalized;
            }
        }
        
        const T scale = T(1) / (C_per_G * HW);
        
        // Compute grad_input for this channel
        for (int hw = threadIdx.x; hw < HW; hw += blockDim.x) {
            const int idx = ((n * C + c) * H * W) + hw;
            const T x = input[idx];
            const T grad = grad_out[idx] * weight_val;
            
            grad_input[idx] = rstd_ng * (grad - scale * (group_sum1 + (x - mean_ng) * rstd_ng * group_sum2));
        }
    }
}

// RMSNorm backward for DiT models (SD3.5, Flux)
template<typename T, int THREADS_PER_BLOCK = 256>
__global__ void rms_norm_backward_kernel(
    const T* __restrict__ grad_out,     // [batch, seq_len, hidden]
    const T* __restrict__ input,        // [batch, seq_len, hidden]
    const T* __restrict__ weight,       // [hidden]
    const T* __restrict__ rstd,         // [batch, seq_len] - reciprocal RMS
    T* __restrict__ grad_input,         // [batch, seq_len, hidden]
    T* __restrict__ grad_weight,        // [hidden]
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps = 1e-6f
) {
    extern __shared__ T shared_mem[];
    T* shared_sum = shared_mem;
    
    const int batch_seq = batch_size * seq_len;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Each block handles one sequence position
    if (bid < batch_seq) {
        const int b = bid / seq_len;
        const int s = bid % seq_len;
        const int offset = bid * hidden_size;
        
        const T rstd_val = rstd[bid];
        
        // Compute dot product: sum(grad_out * weight * input)
        T dot_prod = 0;
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            const int idx = offset + h;
            dot_prod += grad_out[idx] * weight[h] * input[idx];
        }
        
        // Reduce dot product across block
        shared_sum[tid] = dot_prod;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sum[tid] += shared_sum[tid + s];
            }
            __syncthreads();
        }
        
        const T c2 = shared_sum[0] * rstd_val * rstd_val * rstd_val / hidden_size;
        
        // Compute grad_input
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            const int idx = offset + h;
            const T grad = grad_out[idx] * weight[h];
            grad_input[idx] = rstd_val * (grad - input[idx] * c2);
        }
    }
    
    // Compute grad_weight using separate kernel launch
    __syncthreads();
    
    // Each block handles one hidden dimension for grad_weight
    const int h = blockIdx.y;
    if (h < hidden_size && bid == 0) {
        T sum = 0;
        for (int bs = tid; bs < batch_seq; bs += blockDim.x) {
            const int idx = bs * hidden_size + h;
            const T normalized = input[idx] * rstd[bs];
            sum += grad_out[idx] * normalized;
        }
        
        // Reduce across block
        shared_sum[tid] = sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sum[tid] += shared_sum[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&grad_weight[h], shared_sum[0]);
        }
    }
}

// LayerNorm backward (used in some transformer blocks)
template<typename T>
__global__ void layer_norm_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ input,
    const T* __restrict__ mean,
    const T* __restrict__ rstd,
    const T* __restrict__ weight,
    T* __restrict__ grad_input,
    T* __restrict__ grad_weight,
    T* __restrict__ grad_bias,
    const int batch_size,
    const int hidden_size
) {
    extern __shared__ T shared_mem[];
    T* s_sum1 = shared_mem;
    T* s_sum2 = &shared_mem[blockDim.x];
    
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    
    if (b < batch_size) {
        const int offset = b * hidden_size;
        const T mean_val = mean[b];
        const T rstd_val = rstd[b];
        
        // Compute reductions
        T sum1 = 0, sum2 = 0;
        
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            const int idx = offset + h;
            const T centered = input[idx] - mean_val;
            const T normalized = centered * rstd_val;
            const T grad = grad_out[idx];
            const T w = weight ? weight[h] : T(1);
            const T grad_w = grad * w;
            
            sum1 += grad_w;
            sum2 += grad_w * normalized;
            
            // Accumulate gradients for weight and bias
            if (b == 0) {
                if (grad_weight) atomicAdd(&grad_weight[h], grad * normalized);
                if (grad_bias) atomicAdd(&grad_bias[h], grad);
            }
        }
        
        // Block reduction
        s_sum1[tid] = sum1;
        s_sum2[tid] = sum2;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum1[tid] += s_sum1[tid + s];
                s_sum2[tid] += s_sum2[tid + s];
            }
            __syncthreads();
        }
        
        const T scale = T(1) / hidden_size;
        const T c1 = s_sum1[0] * scale;
        const T c2 = s_sum2[0] * scale;
        
        // Compute grad_input
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            const int idx = offset + h;
            const T centered = input[idx] - mean_val;
            const T w = weight ? weight[h] : T(1);
            grad_input[idx] = rstd_val * w * (grad_out[idx] - c1 - centered * rstd_val * c2);
        }
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
    int N, int C, int H, int W, int G,
    int dtype,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = C;  // One block per channel
    
    // Clear gradient accumulators
    if (grad_weight) {
        cudaMemsetAsync(grad_weight, 0, C * (dtype == 0 ? 4 : 2), stream);
    }
    if (grad_bias) {
        cudaMemsetAsync(grad_bias, 0, C * (dtype == 0 ? 4 : 2), stream);
    }
    
    if (dtype == 0) { // FP32
        group_norm_backward_kernel<float><<<blocks, threads, 0, stream>>>(
            (const float*)grad_out, (const float*)input,
            (const float*)mean, (const float*)rstd,
            (const float*)weight,
            (float*)grad_input, (float*)grad_weight, (float*)grad_bias,
            N, C, H, W, G
        );
    } else if (dtype == 1) { // FP16
        group_norm_backward_kernel<half><<<blocks, threads, 0, stream>>>(
            (const half*)grad_out, (const half*)input,
            (const half*)mean, (const half*)rstd,
            (const half*)weight,
            (half*)grad_input, (half*)grad_weight, (half*)grad_bias,
            N, C, H, W, G
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
    cudaStream_t stream
) {
    const int threads = 256;
    const int batch_seq = batch_size * seq_len;
    
    // Clear grad_weight
    if (grad_weight) {
        cudaMemsetAsync(grad_weight, 0, hidden_size * (dtype == 0 ? 4 : 2), stream);
    }
    
    // Launch configuration
    dim3 blocks(batch_seq, hidden_size);
    size_t shared_size = threads * (dtype == 0 ? 4 : 2);
    
    if (dtype == 0) { // FP32
        rms_norm_backward_kernel<float><<<blocks, threads, shared_size, stream>>>(
            (const float*)grad_out, (const float*)input,
            (const float*)weight, (const float*)rstd,
            (float*)grad_input, (float*)grad_weight,
            batch_size, seq_len, hidden_size
        );
    } else if (dtype == 1) { // FP16
        rms_norm_backward_kernel<half><<<blocks, threads, shared_size, stream>>>(
            (const half*)grad_out, (const half*)input,
            (const half*)weight, (const half*)rstd,
            (half*)grad_input, (half*)grad_weight,
            batch_size, seq_len, hidden_size
        );
    }
}

void launch_layer_norm_backward(
    const void* grad_out,
    const void* input,
    const void* mean,
    const void* rstd,
    const void* weight,
    void* grad_input,
    void* grad_weight,
    void* grad_bias,
    int batch_size,
    int hidden_size,
    int dtype,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = batch_size;
    size_t shared_size = threads * 2 * (dtype == 0 ? 4 : 2);
    
    // Clear gradients
    if (grad_weight) {
        cudaMemsetAsync(grad_weight, 0, hidden_size * (dtype == 0 ? 4 : 2), stream);
    }
    if (grad_bias) {
        cudaMemsetAsync(grad_bias, 0, hidden_size * (dtype == 0 ? 4 : 2), stream);
    }
    
    if (dtype == 0) { // FP32
        layer_norm_backward_kernel<float><<<blocks, threads, shared_size, stream>>>(
            (const float*)grad_out, (const float*)input,
            (const float*)mean, (const float*)rstd,
            (const float*)weight,
            (float*)grad_input, (float*)grad_weight, (float*)grad_bias,
            batch_size, hidden_size
        );
    }
}

} // extern "C"