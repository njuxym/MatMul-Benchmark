#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void matmul_ikj_unrolled(int n, double *A, double *B, double *C) {
    int i, k, j;
    for (i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            double aik = A[i * n + k];
            // 4x unrolling: 一次展开4步，减少循环分支判断开销
            for (j = 0; j < n; j += 4) {
                C[i * n + j]     += aik * B[k * n + j];
                C[i * n + j + 1] += aik * B[k * n + j + 1];
                C[i * n + j + 2] += aik * B[k * n + j + 2];
                C[i * n + j + 3] += aik * B[k * n + j + 3];
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
    for (int i = 0; i < n * n; i++) C[i] = 0.0;
    
    double start = wall_time();
    matmul_ikj_unrolled(n, A, B, C);
    double end = wall_time();
    
    double elapsed = end - start;
    double gflops = (2.0 * n * n * n / elapsed) / 1e9;
    printf("v2_loop_unrolling: n=%d, time=%.3f sec, GFLOPS=%.2f\n", n, elapsed, gflops);
    
    free(A); free(B); free(C);
    return 0;
}