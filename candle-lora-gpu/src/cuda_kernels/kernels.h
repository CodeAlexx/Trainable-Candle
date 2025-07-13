#ifndef CANDLE_LORA_KERNELS_H
#define CANDLE_LORA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// LoRA backward kernels
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
);

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
    void* stream
);

// Normalization backward kernels
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
);

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
);

// Attention backward kernels
void launch_attention_qkv_backward(
    const void* grad_out,
    const void* q,
    const void* k,
    const void* v,
    const void* attn_weights,
    void* grad_q,
    void* grad_k,
    void* grad_v,
    int batch_size,
    int num_heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    float scale,
    int dtype,
    void* stream
);

#ifdef __cplusplus
}
#endif

#endif // CANDLE_LORA_KERNELS_H