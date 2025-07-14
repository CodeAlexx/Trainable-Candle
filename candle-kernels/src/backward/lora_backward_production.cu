// Production-ready LoRA backward kernel with optimizations
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>

// Helper for type dispatch
template<typename T>
struct CudaType {
    using Type = T;
    static constexpr cudaDataType_t cublas_type = CUDA_R_32F;
};

template<>
struct CudaType<half> {
    using Type = half;
    static constexpr cudaDataType_t cublas_type = CUDA_R_16F;
};

// Production kernel using cuBLAS for stability
template<typename T>
void lora_backward_cublas(
    cublasHandle_t handle,
    const T* grad_out,    // [batch_seq, out_features]
    const T* input,       // [batch_seq, in_features]  
    const T* lora_down,   // [rank, in_features]
    const T* lora_up,     // [out_features, rank]
    T* grad_down,         // [rank, in_features]
    T* grad_up,           // [out_features, rank]
    T* workspace,         // [batch_seq, rank] for hidden
    int batch_seq,
    int in_features,
    int out_features,
    int rank,
    float scale,
    cudaStream_t stream
) {
    const float alpha = scale;
    const float beta = 0.0f;
    
    cublasSetStream(handle, stream);
    
    // Compute hidden = input @ lora_down^T for grad_up calculation
    // hidden[batch_seq, rank] = input[batch_seq, in_features] @ lora_down^T[in_features, rank]
    if constexpr (std::is_same_v<T, float>) {
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank, batch_seq, in_features,
            &alpha,
            lora_down, in_features,
            input, in_features,
            &beta,
            workspace, rank
        );
    } else {
        const __half alpha_h = __float2half(alpha);
        const __half beta_h = __float2half(beta);
        cublasHgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rank, batch_seq, in_features,
            &alpha_h,
            (const __half*)lora_down, in_features,
            (const __half*)input, in_features,
            &beta_h,
            (__half*)workspace, rank
        );
    }
    
    // Compute grad_up = grad_out^T @ hidden
    // grad_up[out_features, rank] = grad_out^T[out_features, batch_seq] @ hidden[batch_seq, rank]
    if constexpr (std::is_same_v<T, float>) {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, out_features, batch_seq,
            &alpha,
            workspace, rank,
            grad_out, out_features,
            &beta,
            grad_up, rank
        );
    } else {
        const __half alpha_h = __float2half(alpha);
        const __half beta_h = __float2half(beta);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, out_features, batch_seq,
            &alpha_h,
            (__half*)workspace, rank,
            (const __half*)grad_out, out_features,
            &beta_h,
            (__half*)grad_up, rank
        );
    }
    
    // Compute grad_hidden = grad_out @ lora_up for grad_down calculation
    // grad_hidden[batch_seq, rank] = grad_out[batch_seq, out_features] @ lora_up[out_features, rank]
    if constexpr (std::is_same_v<T, float>) {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, batch_seq, out_features,
            &alpha,
            lora_up, rank,
            grad_out, out_features,
            &beta,
            workspace, rank
        );
    } else {
        const __half alpha_h = __float2half(alpha);
        const __half beta_h = __float2half(beta);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, batch_seq, out_features,
            &alpha_h,
            (const __half*)lora_up, rank,
            (const __half*)grad_out, out_features,
            &beta_h,
            (__half*)workspace, rank
        );
    }
    
    // Compute grad_down = grad_hidden^T @ input
    // grad_down[rank, in_features] = grad_hidden^T[rank, batch_seq] @ input[batch_seq, in_features]
    if constexpr (std::is_same_v<T, float>) {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            in_features, rank, batch_seq,
            &alpha,
            input, in_features,
            workspace, rank,
            &beta,
            grad_down, in_features
        );
    } else {
        const __half alpha_h = __float2half(alpha);
        const __half beta_h = __float2half(beta);
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            in_features, rank, batch_seq,
            &alpha_h,
            (const __half*)input, in_features,
            (__half*)workspace, rank,
            &beta_h,
            (__half*)grad_down, in_features
        );
    }
}

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        return; \
    } \
} while(0)

// Global cuBLAS handle (initialized once)
static cublasHandle_t g_cublas_handle = nullptr;

extern "C" void init_cublas_handle() {
    if (g_cublas_handle == nullptr) {
        cublasCreate(&g_cublas_handle);
        // Enable tensor cores if available
        cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    }
}

extern "C" void destroy_cublas_handle() {
    if (g_cublas_handle != nullptr) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// Production launcher for float32
extern "C" void launch_lora_backward_f32_production(
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
    cudaStream_t cuda_stream = (cudaStream_t)stream;
    
    // Ensure cuBLAS is initialized
    init_cublas_handle();
    
    // Allocate workspace for intermediate results
    size_t workspace_size = batch_seq * rank * sizeof(float);
    float* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    
    // Clear output buffers
    CUDA_CHECK(cudaMemsetAsync(grad_down, 0, rank * in_features * sizeof(float), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(grad_up, 0, out_features * rank * sizeof(float), cuda_stream));
    
    // Run the optimized kernel
    lora_backward_cublas<float>(
        g_cublas_handle,
        grad_out, input, lora_down, lora_up,
        grad_down, grad_up, workspace,
        batch_seq, in_features, out_features, rank, scale,
        cuda_stream
    );
    
    // Clean up workspace
    CUDA_CHECK(cudaFree(workspace));
    
    // Check for errors
    if (!cuda_stream) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Production launcher for float16
extern "C" void launch_lora_backward_f16_production(
    const __half* grad_out,
    const __half* input,
    const __half* lora_down,
    const __half* lora_up,
    __half* grad_down,
    __half* grad_up,
    int batch_seq,
    int in_features,
    int out_features,
    int rank,
    float scale,
    void* stream
) {
    cudaStream_t cuda_stream = (cudaStream_t)stream;
    
    // Ensure cuBLAS is initialized
    init_cublas_handle();
    
    // Allocate workspace
    size_t workspace_size = batch_seq * rank * sizeof(__half);
    __half* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    
    // Clear output buffers
    CUDA_CHECK(cudaMemsetAsync(grad_down, 0, rank * in_features * sizeof(__half), cuda_stream));
    CUDA_CHECK(cudaMemsetAsync(grad_up, 0, out_features * rank * sizeof(__half), cuda_stream));
    
    // Run the optimized kernel
    lora_backward_cublas<__half>(
        g_cublas_handle,
        grad_out, input, lora_down, lora_up,
        grad_down, grad_up, workspace,
        batch_seq, in_features, out_features, rank, scale,
        cuda_stream
    );
    
    // Clean up
    CUDA_CHECK(cudaFree(workspace));
    
    if (!cuda_stream) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Cleanup function to be called at program exit
extern "C" void cleanup_lora_backward() {
    destroy_cublas_handle();
}