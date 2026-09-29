#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 128-bit VRAM Paketleme (float4) kullanan CUDA Kernel'ımız
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
            float4 b_val = *reinterpret_cast<const float4*>(&B[k * N + col]);

            c_val.x += a_val * b_val.x;
            c_val.y += a_val * b_val.y;
            c_val.z += a_val * b_val.z;
            c_val.w += a_val * b_val.w;
        }

        *reinterpret_cast<float4*>(&C[row * N + col]) = c_val;
    }
}

// PyTorch C++ Arayüzü c++ wrapper
torch::Tensor matmul_vectorized_cuda(torch::Tensor A, torch::Tensor B) {
    // Tensor doğrulama kontrolü
    TORCH_CHECK(A.is_cuda(), "A matrisi gpu üzerinde olmalidir");
    TORCH_CHECK(B.is_cuda(), "B matrisi gpu üzerinde olmalidir");
    TORCH_CHECK(A.is_contiguous(), "A matrisi bellekte bitisik (contiguous) olmalidir");
    TORCH_CHECK(B.is_contiguous(), "B matrisi bellekte bitisik olmalidir");

    int N = A.size(0);

    // Çıktı için bellek ayrımı pytorch vram yonetimi
    auto C = torch::zeros({N, N}, A.options());

    // Pointer ları cuda kernal ına gönderme
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N / 4 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_vectorized_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    return C;
}

// Pybind11 Modül Bağlantısı
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_vectorized", &matmul_vectorized_cuda, "Vectorized CUDA MatMul (float4)");
}