use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda_kernels/");
    
    // Only build CUDA kernels if cuda feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // Get CUDA path
        let cuda_path = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        
        // Build CUDA kernels
        cc::Build::new()
            .cuda(true)
            .cudart("shared")
            .flag("-arch=sm_75") // Minimum for Turing GPUs
            .flag("-gencode=arch=compute_75,code=sm_75")
            .flag("-gencode=arch=compute_80,code=sm_80")
            .flag("-gencode=arch=compute_86,code=sm_86")
            .flag("-O3")
            .flag("-use_fast_math")
            .file("src/cuda_kernels/lora_backward.cu")
            .file("src/cuda_kernels/norm_backward.cu")
            .file("src/cuda_kernels/attention_backward.cu")
            .compile("candle_lora_kernels");
        
        // Generate bindings
        let bindings = bindgen::Builder::default()
            .header("src/cuda_kernels/kernels.h")
            .clang_arg(format!("-I{}/include", cuda_path))
            .generate()
            .expect("Unable to generate bindings");
        
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}