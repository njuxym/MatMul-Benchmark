#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// 本地矩阵乘法（ikj顺序）
void local_matmul(int n_local, int n, double *A_local, double *B, double *C_local) {
    int i, j, k;
    for (i = 0; i < n_local; i++) {
        for (k = 0; k < n; k++) {
            double aik = A_local[i * n + k];
            for (j = 0; j < n; j++) {
                C_local[i * n + j] += aik * B[k * n + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 1024;  // 矩阵维度
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) n = atoi(argv[1]);
    
    // 计算每个进程的行数（按行分块）
    int rows_per_proc = n / size;
    int remainder = n % size;
    
    int n_local = rows_per_proc;
    if (rank < remainder) n_local++;
    
    // 分配本地矩阵
    double *A_local = malloc(n_local * n * sizeof(double));
    double *C_local = calloc(n_local * n, sizeof(double));
    double *B = NULL;
    double *C_full = NULL;
    
    if (rank == 0) {
        B = malloc(n * n * sizeof(double));
        C_full = malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) {
            B[i] = 1.0;
            C_full[i] = 0.0;
        }
    }
    
    // 初始化本地A矩阵
    for (int i = 0; i < n_local * n; i++) {
        A_local[i] = 1.0;
    }
    
    // 广播B矩阵给所有进程
    if (rank != 0) {
        B = malloc(n * n * sizeof(double));
    }
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 计时开始
    MPI_Barrier(MPI_COMM_WORLD);
    double start = wall_time();
    
    // 本地计算
    local_matmul(n_local, n, A_local, B, C_local);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end = wall_time();
    double local_elapsed = end - start;
    
    // 收集各进程的计算结果到0号进程
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc;
            if (i < remainder) rows++;
            recvcounts[i] = rows * n;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }
    
    MPI_Gatherv(C_local, n_local * n, MPI_DOUBLE,
                C_full, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // 0号进程输出结果
    if (rank == 0) {
        double elapsed = local_elapsed;  // 实际应该取所有进程的最大值
        double gflops = (2.0 * n * n * n / elapsed) / 1e9;
        printf("MPI: n=%d, procs=%d, time=%.3f sec, GFLOPS=%.2f\n", 
               n, size, elapsed, gflops);
        
        free(recvcounts);
        free(displs);
        free(B);
        free(C_full);
    } else {
        free(B);
    }
    
    free(A_local);
    free(C_local);
    
    MPI_Finalize();
    return 0;
}