# GPU_inference
# PyTorch Custom CUDA Extension: High-Performance Vectorized MatMul Kernel

An end-to-end implementation of custom CUDA C++ kernels integrated seamlessly into PyTorch via **Pybind11** and JIT Compilation (`torch.utils.cpp_extension`). 

This project demonstrates hardware-aware GPU optimization techniques—progressing from a naive global memory kernel to shared memory tiling and 128-bit (`float4`) memory coalescing—achieving significant speedups before benchmarking against vendor-tuned **NVIDIA cuBLAS** baselines.

---

## Key Technical Highlights

* **Zero-Copy PyTorch Binding:** Leverages `torch::Tensor::data_ptr<float>()` to bridge Python tensors directly with C++/CUDA device kernels without memory duplication overhead.
* **128-bit Vectorized Memory Coalescing:** Utilizes `float4` vector types to maximize VRAM memory bandwidth saturation and minimize memory transaction cycles.
* **Shared Memory Tiling (SRAM):** Implements 2D block tiling ($16 \times 16$) to eliminate global memory read bottlenecks and optimize L1/SRAM cache reuse.
* **Cross-Compiler Compatibility:** Resolved low-level MSVC (`cl.exe`) preprocessor standards conflicts with CUDA 13 via explicit compiler flags (`/Zc:preprocessor`).

---

## Architecture & Optimization Pipeline

| Kernel Variant | Latency ($N=1024$) | Throughput (TFLOPS) | Bottleneck / Optimization Mechanism |
| :--- | :--- | :--- | :--- |
| **Naive Kernel** | $3.360\text{ ms}$ | $0.638\text{ TFLOPS}$ | High VRAM read latency; uncoalesced global memory accesses. |
| **Shared Memory Tiling** | $2.683\text{ ms}$ | $0.800\text{ TFLOPS}$ | Reduced VRAM traffic via SRAM tiling ($16 \times 16$ tiles). |
| **Vectorized (`float4`)** | **$1.125\text{ ms}$** | **$1.860\text{ TFLOPS}$** | **128-bit Memory Alignment; $2\times$ speedup over tiled variant.** |
| **cuBLAS (Vendor Baseline)** | $0.321\text{ ms}$ | $6.678\text{ TFLOPS}$ | Hardware Tensor Cores activation + Register Tiling. |

> **Hardware Testbed:** NVIDIA GeForce RTX 4050 Laptop GPU (`sm_89`), CUDA 13.4, MSVC v14.51, PyTorch 2.x.

---
## Build & Execution Instructions
Prerequisites
-Windows 10/11 x64
-Visual Studio 2022 / 2026 Build Tools (cl.exe)
-CUDA Toolkit 12.x or 13.x
-PyTorch built with CUDA support

---
## Next Roadmap Steps
[ ] Implement WMMA (Warp Matrix Multiply and Accumulate) API for Tensor Core execution.
[ ] Profile Warp Occupancy and Memory Bandwidth saturation using NVIDIA Nsight Compute (ncu).
[ ] Develop INT8 / INT4 Weight-Only Quantization Kernel for Edge LLM Inference.
Author: Aslı
AI Systems & Infrastructure Engineering Research

---
## Code Snippet: 128-bit Vectorized Memory Fetch

Instead of issuing four individual 32-bit scalar loads, memory requests are coalesced into a single 128-bit transaction:

```cpp
__global__ void matmul_vectorized_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (row < N && col < N) {
        float4 c_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int k = 0; k < N; ++k) {
            float a_val = A[row * N + k];
            // 128-bit coalesced VRAM fetch
            float4 b_val = *reinterpret_cast<const float4*>(&B[k * N + col]);

            c_val.x += a_val * b_val.x;
            c_val.y += a_val * b_val.y;
            c_val.z += a_val * b_val.z;
            c_val.w += a_val * b_val.w;
        }

        // 128-bit vectorized store
        *reinterpret_cast<float4*>(&C[row * N + col]) = c_val;
    }
}
