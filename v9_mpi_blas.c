#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include <sys/time.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main(int argc, char *argv[]) {
    int n = 1024;   // 矩阵维度
    if (argc > 1) n = atoi(argv[1]);
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // === 1. 一维行划分：确定每个进程的行数 ===
    int rows_per_proc = n / size;
    int remainder = n % size;
    int n_local = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // 计算每个进程在全局矩阵中的起始行偏移（用于 MPI_Scatterv / Gatherv）
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * n;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // === 2. 分配本地内存 ===
    double *A_local = (double*)malloc(n_local * n * sizeof(double));
    double *C_local = (double*)calloc(n_local * n, sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    
    // === 3. 主进程初始化全局矩阵 ===
    double *A_full = NULL, *C_full = NULL;
    if (rank == 0) {
        A_full = (double*)malloc(n * n * sizeof(double));
        C_full = (double*)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) {
            A_full[i] = 1.0;   // 全1矩阵，方便验证
            C_full[i] = 0.0;
        }
        for (int i = 0; i < n * n; i++) {
            B[i] = 1.0;
        }
    }
    
    // === 4. 分发A矩阵的行块 ===
    MPI_Scatterv(rank == 0 ? A_full : NULL, sendcounts, displs, MPI_DOUBLE,
                 A_local, n_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // === 5. 广播B矩阵 ===
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // === 6. 本地计算：调用 BLAS ===
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    
    // C_local = A_local * B  (注意：A_local 尺寸为 n_local×n, B 为 n×n, 结果 C_local 为 n_local×n)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_local, n, n,
                1.0, A_local, n,
                B, n,
                0.0, C_local, n);
    
    double end = MPI_Wtime();
    double local_time = end - start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // === 7. 收集计算结果 ===
    int *recvcounts = NULL, *recvdispls = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        recvdispls = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * n;
            recvdispls[i] = offset;
            offset += recvcounts[i];
        }
    }
    MPI_Gatherv(C_local, n_local * n, MPI_DOUBLE,
                rank == 0 ? C_full : NULL, recvcounts, recvdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // === 8. 输出性能 ===
    if (rank == 0) {
        double gflops = (2.0 * n * n * n / max_time) / 1e9;
        printf("MPI+BLAS: n=%d, procs=%d, time=%.5f sec, GFLOPS=%.2f\n",
               n, size, max_time, gflops);
        // 可选：简单验证结果是否正确（全1矩阵相乘，结果每个元素应为 n）
        // printf("C[0]=%f (expected %d)\n", C_full[0], n);
        
        free(A_full); free(C_full);
        free(sendcounts); free(displs);
        free(recvcounts); free(recvdispls);
    }
    
    free(A_local); free(C_local); free(B);
    MPI_Finalize();
    return 0;
}
