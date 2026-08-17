#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    int N = 1024;
    size_t bytes = N * N * sizeof(float);

    // 初始化主机端矩阵
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 将数据拷贝到设备
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // cuBLAS库操作句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;
    
    // cuBLAS使用列主序，这里通过转置操作实现C = A * B (Row-Major)
    // 相当于 cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, ...);
    // 简化为标准调用：
    double start = wall_time();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaDeviceSynchronize(); // 等待所有GPU操作完成
    double end = wall_time();
    double elapsed = end - start;

    // 打印结果
    double gflops = (2.0 * N * N * N / elapsed) / 1e9;
    printf("cuBLAS (N=%d): time = %.5f sec, GFLOPS = %.2f\n", N, elapsed, gflops);

    // 清理
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}