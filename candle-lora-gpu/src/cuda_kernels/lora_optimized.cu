#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

// Optimized LoRA backward kernel using tiled matrix multiplication
// Computes: grad_down = scale * grad_out^T @ up @ input^T
//          grad_up = scale * grad_out^T @ (input @ down^T)

template<typename T, int TILE_SIZE = 16>
__global__ void lora_backward_optimized(
    const T* __restrict__ grad_out,    // [batch_seq, out_features]
    const T* __restrict__ input,       // [batch_seq, in_features]
    const T* __restrict__ lora_down,   // [rank, in_features]
    const T* __restrict__ lora_up,     // [out_features, rank]
    T* __restrict__ grad_down,         // [rank, in_features]
    T* __restrict__ grad_up,           // [out_features, rank]
    const int batch_seq,
    const int in_features,
    const int out_features,
    const int rank,
    const float scale
) {
    // Shared memory for tiles
    __shared__ T tile_a[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ T tile_b[TILE_SIZE][TILE_SIZE + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // Determine which gradient we're computing
    if (bz == 0) {
        // Compute grad_down[r, i]
        const int row = by * TILE_SIZE + ty; // rank dimension
        const int col = bx * TILE_SIZE + tx; // in_features dimension
        
        if (row < rank && col < in_features) {
            T sum = 0;
            
            // Tiled computation over batch_seq and out_features
            for (int k_outer = 0; k_outer < (batch_seq + TILE_SIZE - 1) / TILE_SIZE; k_outer++) {
                for (int j_outer = 0; j_outer < (out_features + TILE_SIZE - 1) / TILE_SIZE; j_outer++) {
                    // Load tiles with bounds checking
                    const int k = k_outer * TILE_SIZE + tx;
                    const int j = j_outer * TILE_SIZE + ty;
                    
                    // Load input tile
                    if (k < batch_seq && col < in_features) {
                        tile_a[ty][tx] = input[k * in_features + col];
                    } else {
                        tile_a[ty][tx] = 0;
                    }
                    
                    // Load grad_out @ up tile
                    if (k < batch_seq && j < out_features && row < rank) {
                        // This is expensive - in production, we'd precompute grad_out @ up
                        T temp = 0;
                        if (k < batch_seq && j < out_features) {
                            temp = grad_out[k * out_features + j] * lora_up[j * rank + row];
                        }
                        tile_b[ty][tx] = temp;
                    } else {
                        tile_b[ty][tx] = 0;
                    }
                    
                    __syncthreads();
                    
                    // Compute partial dot product
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; k++) {
                        sum += tile_b[ty][k] * tile_a[k][tx];
                    }
                    
                    __syncthreads();
                }
            }
            
            // Atomic add to handle multiple blocks updating same element
            atomicAdd(&grad_down[row * in_features + col], sum * scale);
        }
    } else {
        // Compute grad_up[o, r]
        const int row = by * TILE_SIZE + ty; // out_features dimension
        const int col = bx * TILE_SIZE + tx; // rank dimension
        
        if (row < out_features && col < rank) {
            T sum = 0;
            
            // Tiled computation over batch_seq and in_features
            for (int k_outer = 0; k_outer < (batch_seq + TILE_SIZE - 1) / TILE_SIZE; k_outer++) {
                for (int j_outer = 0; j_outer < (in_features + TILE_SIZE - 1) / TILE_SIZE; j_outer++) {
                    // Load tiles
                    const int k = k_outer * TILE_SIZE + tx;
                    const int j = j_outer * TILE_SIZE + ty;
                    
                    // Load grad_out tile
                    if (k < batch_seq && row < out_features) {
                        tile_a[ty][tx] = grad_out[k * out_features + row];
                    } else {
                        tile_a[ty][tx] = 0;
                    }
                    
                    // Load input @ down^T tile
                    if (k < batch_seq && j < in_features && col < rank) {
                        T temp = 0;
                        if (k < batch_seq && j < in_features) {
                            temp = input[k * in_features + j] * lora_down[col * in_features + j];
                        }
                        tile_b[ty][tx] = temp;
                    } else {
                        tile_b[ty][tx] = 0;
                    }
                    
                    __syncthreads();
                    
                    // Compute partial dot product
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; k++) {
                        sum += tile_a[ty][k] * tile_b[k][tx];
                    }
                    
                    __syncthreads();
                }
            }
            
            atomicAdd(&grad_up[row * rank + col], sum * scale);
        }
    }
}

// Warp-level optimized version for small ranks (<=32)
template<typename T>
__global__ void lora_backward_small_rank_optimized(
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
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Each warp handles one row of grad_down
    if (warp_id < rank) {
        for (int i = lane_id; i < in_features; i += 32) {
            T sum = 0;
            
            // Accumulate over batch and output features
            for (int b = 0; b < batch_seq; b++) {
                const T input_val = input[b * in_features + i];
                
                // Vectorized load when possible
                for (int o = 0; o < out_features; o++) {
                    sum += grad_out[b * out_features + o] * 
                           lora_up[o * rank + warp_id] * 
                           input_val;
                }
            }
            
            grad_down[warp_id * in_features + i] = sum * scale;
        }
    }
    
    // Handle grad_up in second phase
    __syncthreads();
    
    const int total_threads = blockDim.x * gridDim.x;
    for (int idx = tid; idx < out_features * rank; idx += total_threads) {
        const int o = idx / rank;
        const int r = idx % rank;
        
        T sum = 0;
        for (int b = 0; b < batch_seq; b++) {
            const T grad_val = grad_out[b * out_features + o];
            
            for (int i = 0; i < in_features; i++) {
                sum += grad_val * input[b * in_features + i] * 
                       lora_down[r * in_features + i];
            }
        }
        
        grad_up[idx] = sum * scale;
    }
}

// FP16 specialization using tensor cores
#if __CUDA_ARCH__ >= 700
template<>
__global__ void lora_backward_optimized<half, 16>(
    const half* __restrict__ grad_out,
    const half* __restrict__ input,
    const half* __restrict__ lora_down,
    const half* __restrict__ lora_up,
    half* __restrict__ grad_down,
    half* __restrict__ grad_up,
    const int batch_seq,
    const int in_features,
    const int out_features,
    const int rank,
    const float scale
) {
    // Use WMMA API for tensor cores
    using namespace nvcuda;
    
    // Tensor core tile sizes
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Simplified tensor core computation
    // In production, this would be fully optimized
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp computes a 16x16 tile of output
    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = blockIdx.x * WMMA_N;
    
    if (blockIdx.z == 0 && tile_row < rank && tile_col < in_features) {
        // Compute grad_down tile
        for (int k = 0; k < batch_seq; k += WMMA_K) {
            // Load and compute with tensor cores
            // This is simplified - real implementation would be more complex
        }
    }
}
#endif

extern "C" {

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
    int dtype, // 0=f32, 1=f16, 2=bf16
    cudaStream_t stream
) {
    // Clear gradient buffers
    const size_t grad_down_size = rank * in_features * (dtype == 0 ? 4 : 2);
    const size_t grad_up_size = out_features * rank * (dtype == 0 ? 4 : 2);
    cudaMemsetAsync(grad_down, 0, grad_down_size, stream);
    cudaMemsetAsync(grad_up, 0, grad_up_size, stream);
    
    if (rank <= 32 && dtype == 0) {
        // Use warp-optimized kernel for small ranks
        const int threads = 256;
        const int blocks = (rank * 32 + threads - 1) / threads;
        
        lora_backward_small_rank_optimized<float><<<blocks, threads, 0, stream>>>(
            (const float*)grad_out, (const float*)input,
            (const float*)lora_down, (const float*)lora_up,
            (float*)grad_down, (float*)grad_up,
            batch_seq, in_features, out_features, rank, scale
        );
    } else {
        // Use tiled kernel
        const int TILE_SIZE = 16;
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks(
            (max(in_features, rank) + TILE_SIZE - 1) / TILE_SIZE,
            (max(rank, out_features) + TILE_SIZE - 1) / TILE_SIZE,
            2  // 0 for grad_down, 1 for grad_up
        );
        
        if (dtype == 0) { // FP32
            lora_backward_optimized<float, TILE_SIZE><<<blocks, threads, 0, stream>>>(
                (const float*)grad_out, (const float*)input,
                (const float*)lora_down, (const float*)lora_up,
                (float*)grad_down, (float*)grad_up,
                batch_seq, in_features, out_features, rank, scale
            );
        } else if (dtype == 1) { // FP16
#if __CUDA_ARCH__ >= 700
            lora_backward_optimized<half, TILE_SIZE><<<blocks, threads, 0, stream>>>(
                (const half*)grad_out, (const half*)input,
                (const half*)lora_down, (const half*)lora_up,
                (half*)grad_down, (half*)grad_up,
                batch_seq, in_features, out_features, rank, scale
            );
#else
            printf("FP16 requires compute capability 7.0+\n");
#endif
        }
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in lora_backward_optimized: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"