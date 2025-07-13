//! candle-nn
//!
//! ## Other Crates
//!
//! Candle consists of a number of crates. This crate holds structs and functions
//! that allow you to build and train neural nets. You may wish
//! to look at the docs for the other crates which can be found here:
//!
//! - [candle-core](https://docs.rs/candle-core/). Core Datastructures and DataTypes.
//! - [candle-nn](https://docs.rs/candle-nn/). Building blocks for Neural Nets.
//! - [candle-datasets](https://docs.rs/candle-datasets/). Rust access to commonly used Datasets like MNIST.
//! - [candle-examples](https://docs.rs/candle-examples/). Examples of Candle in Use.
//! - [candle-onnx](https://docs.rs/candle-onnx/). Loading and using ONNX models.
//! - [candle-pyo3](https://docs.rs/candle-pyo3/). Access to Candle from Python.
//! - [candle-transformers](https://docs.rs/candle-transformers/). Candle implemntation of many published transformer models.
//!

pub mod activation;
pub mod adam_trainable;
pub mod backward;
pub mod batch_norm;
pub mod conv;
pub mod embedding;
pub mod encoding;
pub mod func;
pub mod gradient_checkpoint;
pub mod group_norm;
pub mod init;
pub mod kv_cache;
pub mod layer_norm;
pub mod linear;
pub mod linear_generic;
pub mod loss;
pub mod ops;
pub mod optim;
pub mod rnn;
pub mod rotary_emb;
pub mod sampling;
pub mod sequential;
pub mod trainable_tensor;
pub mod var_builder;
pub mod var_map;

pub use activation::{prelu, Activation, PReLU};
pub use adam_trainable::{AdamTrainable, Adam8bitTrainable};
pub use backward::{backward, mse_backward, linear_lora_backward, GradientTape};
pub use batch_norm::{batch_norm, BatchNorm, BatchNormConfig};
pub use conv::{
    conv1d, conv1d_no_bias, conv2d, conv2d_no_bias, conv_transpose1d, conv_transpose1d_no_bias,
    conv_transpose2d, conv_transpose2d_no_bias, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
pub use embedding::{embedding, Embedding};
pub use func::{func, func_t, Func, FuncT};
pub use gradient_checkpoint::{
    GradientCheckpointWrapper, SDXLCheckpointConfig, checkpoint_unet_block, 
    checkpointed_attention, checkpointed_resnet_block, checkpoint_transformer_blocks
};
pub use group_norm::{group_norm, GroupNorm};
pub use init::Init;
pub use layer_norm::{
    layer_norm, layer_norm_no_bias, rms_norm, LayerNorm, LayerNormConfig, RmsNorm,
};
pub use linear::{linear, linear_b, linear_no_bias, Linear};
pub use linear_generic::{
    linear as linear_inference, linear_training, linear_training_direct,
    Linear as LinearGeneric, LinearLoRA, VarBuilderTraining, VarBuilderTrainingPrefixed
};
pub use ops::Dropout;
pub use optim::{AdamW, Optimizer, ParamsAdamW, SGD};
pub use rnn::{gru, lstm, GRUConfig, LSTMConfig, GRU, LSTM, RNN};
pub use sequential::{seq, Sequential};
pub use trainable_tensor::{TrainableTensor, GradContext, LoRALinear, ModelWrapper, AdamOptimizer};
pub use var_builder::VarBuilder;
pub use var_map::VarMap;

pub use candle::{Module, ModuleT};
