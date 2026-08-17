#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 32  // 分块大小，应能被缓存容纳（32×32×8=8192字节≈8KB）

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void matmul_blocked(int n, double *A, double *B, double *C) {
    int i, j, k, ii, jj, kk;
    
    // 清理C矩阵
    for (i = 0; i < n * n; i++) C[i] = 0.0;
    
    // 三层分块循环
    for (ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (kk = 0; kk < n; kk += BLOCK_SIZE) {
                // 在块内进行标准ikj乘法
                for (i = ii; i < ii + BLOCK_SIZE && i < n; i++) {
                    for (k = kk; k < kk + BLOCK_SIZE && k < n; k++) {
                        double aik = A[i * n + k];
                        for (j = jj; j < jj + BLOCK_SIZE && j < n; j++) {
                            C[i * n + j] += aik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n = 1024;
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    
    for (int i = 0; i < n * n; i++) A[i] = 1.0;
    for (int i = 0; i < n * n; i++) B[i] = 1.0;
    
    double start = wall_time();
    matmul_blocked(n, A, B, C);
    double end = wall_time();
    
    double elapsed = end - start;
    double gflops = (2.0 * n * n * n / elapsed) / 1e9;
    printf("v3_blocking: n=%d, BS=%d, time=%.3f sec, GFLOPS=%.2f\n", n, BLOCK_SIZE, elapsed, gflops);
    
    free(A); free(B); free(C);
    return 0;
}