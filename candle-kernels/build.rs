fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/binary_op_macros.cuh");
    
    // Build standard PTX kernels
    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/ptx.rs").unwrap();
    
    // Build backward kernels if feature enabled
    #[cfg(feature = "cuda-backward")]
    {
        compile_backward_kernels();
    }
}

#[cfg(feature = "cuda-backward")]
fn compile_backward_kernels() {
    use std::env;
    use std::path::PathBuf;
    
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let cuda_include = PathBuf::from(&cuda_path).join("include");
    
    println!("cargo:info=Building CUDA backward kernels");
    
    // Compile backward kernels
    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .flag("-arch=sm_70") // Volta minimum
        .flag("-gencode=arch=compute_70,code=sm_70")
        .flag("-gencode=arch=compute_75,code=sm_75") // Turing
        .flag("-gencode=arch=compute_80,code=sm_80") // Ampere
        .flag("-gencode=arch=compute_86,code=sm_86") // Ampere
        .flag("-gencode=arch=compute_89,code=sm_89") // Ada
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-Xcompiler=-fPIC")
        .include(&cuda_include)
        .include("src")
        .file("src/backward/lora_backward_production.cu")
        .compile("candle_backward_kernels");
    
    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
}