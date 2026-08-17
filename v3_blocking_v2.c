#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// 标准分块矩阵乘法：循环顺序 ii → jj → kk，块内 i-k-j
void matmul_blocked(int n, double *A, double *B, double *C, int block_size) {
    int i, j, k, ii, jj, kk;
    // 清零 C
    for (i = 0; i < n * n; i++) C[i] = 0.0;

    for (ii = 0; ii < n; ii += block_size) {
        int imax = (ii + block_size < n) ? ii + block_size : n;
        for (jj = 0; jj < n; jj += block_size) {
            int jmax = (jj + block_size < n) ? jj + block_size : n;
            for (kk = 0; kk < n; kk += block_size) {
                int kmax = (kk + block_size < n) ? kk + block_size : n;
                // 块内使用 i-k-j 顺序（对缓存友好）
                for (i = ii; i < imax; i++) {
                    for (k = kk; k < kmax; k++) {
                        double aik = A[i * n + k];
                        // 手动展开内层 j 循环 4 次（可选，可提高指令级并行）
                        for (j = jj; j + 3 < jmax; j += 4) {
                            C[i * n + j]     += aik * B[k * n + j];
                            C[i * n + j + 1] += aik * B[k * n + j + 1];
                            C[i * n + j + 2] += aik * B[k * n + j + 2];
                            C[i * n + j + 3] += aik * B[k * n + j + 3];
                        }
                        // 处理剩余不足4个的元素
                        for (; j < jmax; j++) {
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
    int test_blocks[] = {32, 64, 96, 128, 256};
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));

    for (int i = 0; i < n * n; i++) A[i] = 1.0;
    for (int i = 0; i < n * n; i++) B[i] = 1.0;

//    printf("=== v3_blocking_v2 (blocked, ii-jj-kk order, i-k-j inner) ===\n");
    for (int t = 0; t < sizeof(test_blocks)/sizeof(test_blocks[0]); t++) {
        int bs = test_blocks[t];
        double start = wall_time();
        matmul_blocked(n, A, B, C, bs);
        double end = wall_time();
        double elapsed = end - start;
        double gflops = (2.0 * n * n * n / elapsed) / 1e9;
        printf("block_size = %3d : time = %.3f sec, GFLOPS = %.2f\n", bs, elapsed, gflops);
    }

    free(A); free(B); free(C);
    return 0;
}
