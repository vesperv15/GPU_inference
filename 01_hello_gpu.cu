#include <iostream>
#include <cuda_runtime.h>

// global eki gpu kernel oldugunu gosterir fonk. cpu cagırır gpu uzerinde paralel olarak calısır

__global__ void vectorAddKernel( const float* A, const float* B, float* C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //gpu üzerindeki her thread izlek in küresel indeksi hesaplanır
    //dizi boyutunu asmamak için sınır kontrolu
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    int N = 1024;
    size_t bytes = N * sizeof(float);

    // cpu (host) bellek yönlendirme ayırma

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    //verileri doldur
    for (int i = 0; i< N; i++){
        h_A[i]= 1.0f;
        h_B[i]= 2.0f;
    }

    //device gpu bellek ayrımı yönlendirimi?
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    //veriyi cpu dan gpu ya kopyalama host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    //execution configuration ızgara ve blok düzeni 
    int threadsPerBlock = 256;
    int blockPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    //kernel cagırısı gridsize ve blocksize
    vectorAddKernel<<<blockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    //gpu işlemlerinin tamamlanması
    cudaDeviceSynchronize();
    //sonucu gpu dan cpuya geri kopyala device to host 
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    // dogrulama kontrolu
    std::cout << "ındex 0 result:" << h_C[0] << " expected : 3.0" << std::endl;
    std::cout << "ındex 1023 result:" << h_C[1023] << "expected: 3.0" << std::endl;

    //bellegi temizle
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}