#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// OpenMP 并行矩阵乘法 (ikj顺序 + 并行外层循环)
void matmul_openmp(int n, double *A, double *B, double *C, int num_threads) {
    int i, j, k;
    
    // 设置线程数
    omp_set_num_threads(num_threads);
    
    // 并行化外层 i 循环
    #pragma omp parallel for private(i, j, k) schedule(static)
    for (i = 0; i < n; i++) {
        for (k = 0; k < n; k++) {
            double aik = A[i * n + k];
            for (j = 0; j < n; j++) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int n = 1024;
    int num_threads = 4;  // 默认4线程，可通过命令行参数修改
    
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = calloc(n * n, sizeof(double));  // calloc自动置零
    
    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }
    
    double start = wall_time();
    matmul_openmp(n, A, B, C, num_threads);
    double end = wall_time();
    
    double elapsed = end - start;
    double gflops = (2.0 * n * n * n / elapsed) / 1e9;
    printf("OpenMP: n=%d, threads=%d, time=%.3f sec, GFLOPS=%.2f\n", 
           n, num_threads, elapsed, gflops);
    
    free(A); free(B); free(C);
    return 0;
}