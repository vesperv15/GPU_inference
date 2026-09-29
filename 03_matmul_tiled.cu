#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Hatasi: " << cudaGetErrorString(err) \
                    << " (Satir: " << __LINE__ << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

//shared methodu kullanılarak tiled matmul kernel ı
__global__ void matmulTiledKernel (const float* A, const float* B, float* C, int N) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx= blockIdx.x; int by= blockIdx.y;
    int tx= threadIdx.x; int ty= threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    //matrisi tile width adımı ile parcalar halinde geziyoruz

    for ( int p =0; p< (N + TILE_WIDTH -1) / TILE_WIDTH; ++p){
        //vram den sram e veri yükleme coalesced memory access
        if ( row < N && (p * TILE_WIDTH + tx) < N)
            s_A[ty][tx] = A[row * N + p * TILE_WIDTH + tx];
        else
            s_A[ty][tx] = 0.0f;
        
        if ((p * TILE_WIDTH + ty) < N && col < N)
            s_B[ty][tx] = B[(p * TILE_WIDTH + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;
        // butun thread lerin yüklemeyi bitimesini bekle bariyer cakısma onlemek için
        
        __syncthreads();

        // carpım işlemini vram yerine sram de oku
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        //sıradakı kutu tile yuklenmeden once senkronize etmek için
        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = sum;
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

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH); 
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                    (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Matrix Size: " << N << "x" << N << std::endl;
    std::cout << "Tiled GPU Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Sample Output C[0]: " << h_C[0] << " (Expected: " << 2.0f * N << ")" << std::endl;

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}