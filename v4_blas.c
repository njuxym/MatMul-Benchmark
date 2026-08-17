#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>   // OpenBLAS/MKL

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
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
    // C ← α·A×B + β·C , 其中α=1.0, β=0.0
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A, n, B, n, 0.0, C, n);
    double end = wall_time();
    
    double elapsed = end - start;
    double gflops = (2.0 * n * n * n / elapsed) / 1e9;
    printf("v4_blas: n=%d, time=%.3f sec, GFLOPS=%.2f\n", n, elapsed, gflops);
    
    free(A); free(B); free(C);
    return 0;
}