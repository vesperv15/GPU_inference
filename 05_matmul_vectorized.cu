#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_SIZE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Hatasi: " << cudaGetErrorString(err) \
                    << " (Satir: " << __LINE__ << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// float4 128 bit vektörize tiled matmul kernel ı
__global__ void matmulVectorizedKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // Her thread 4 eleman işler

    float4 c_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k = 0; k < N; ++k) {
        float a_val = A[row * N + k];

        //b matrisinden 128 bit float4 vektörize okuma
        float4 b_val = *reinterpret_cast<const float4*>(&B[k * N + col]);

        c_val.x += a_val * b_val.x;
        c_val.y += a_val * b_val.y;
        c_val.z += a_val * b_val.z;
        c_val.w += a_val * b_val.w;
    }
    if (row < N && col < N) {
        // Sonucu vram e 128 bitlik paket olarak geri yazma
        *reinterpret_cast<float4*>(&C[row * N + col]) = c_val;
    }
}

int main() {
    int N = 1024;
    size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C(N * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    //her thread x ekseninde 4 eleman işlediği için blok enişiliği n/4 olur
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N / 4 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmulVectorizedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    double flops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;

    std::cout << "Matrix Size: " << N << "x" << N << std::endl;
    std::cout << "Vectorized (float4) Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;
    std::cout << "Sample Output C[0]: " << h_C[0] << " (Expected: " << 2.0f * N << ")" << std::endl;

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}