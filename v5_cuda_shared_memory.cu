#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// CUDA Kernel：使用共享内存的分块矩阵乘法
// 每个线程块计算 C 中的一个 (BLOCK_SIZE x BLOCK_SIZE) 的子块
__global__ void matmul_cuda_kernel(float *A, float *B, float *C, int N) {
    // 声明共享内存，大小在编译时确定，用于存放当前块所需的数据块
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // 计算当前线程在全局矩阵中的行和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // 循环遍历所有块（沿着 K 维度分块）
    // num_tiles = N / BLOCK_SIZE
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // 1. 协作加载：将当前 K 维度块对应的 A 和 B 数据加载到共享内存中
        // 每个线程负责一个元素
        if (row < N && (tile * BLOCK_SIZE + threadIdx.x) < N) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < N && (tile * BLOCK_SIZE + threadIdx.y) < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads(); // 同步，等待块内所有线程都加载完成

        // 2. 计算点积：利用共享内存中的数据进行累加
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads(); // 同步，确保在当前块的数据被覆盖前，所有线程都完成了计算
    }

    // 3. 将计算结果写回全局内存
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024; // 矩阵维度，可以尝试 4096 以获得更佳效果
    size_t bytes = N * N * sizeof(float);

    // --- 分配主机内存并初始化 ---
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    // --- 分配设备内存 ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // --- 配置内核启动参数 ---
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // --- 启动CUDA内核并计时 ---
    double start = wall_time();
    matmul_cuda_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // 等待GPU完成计算
    double end = wall_time();
    double elapsed = end - start;

    // 可选：将结果拷贝回主机进行验证
    // cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 打印结果
    double gflops = (2.0 * N * N * N / elapsed) / 1e9;
    printf("CUDA Shared Memory (N=%d): time = %.5f sec, GFLOPS = %.2f\n", N, elapsed, gflops);

    // --- 释放内存 ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}