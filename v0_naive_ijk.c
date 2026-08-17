#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

void matmul_naive_ijk(int n, double *A, double *B, double *C) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
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
    matmul_naive_ijk(n, A, B, C);
    double end = wall_time();
    
    double elapsed = end - start;
    double gflops = (2.0 * n * n * n / elapsed) / 1e9;
    printf("v0_naive_ijk: n=%d, time=%.3f sec, GFLOPS=%.2f\n", n, elapsed, gflops);
    
    free(A); free(B); free(C);
    return 0;
}