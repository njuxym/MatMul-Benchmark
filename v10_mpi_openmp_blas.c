#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <cblas.h>

int main(int argc, char *argv[]) {
    int n = 1024;
    if (argc > 1) n = atoi(argv[1]);
    
    int rank, size;
    int provided;
    // 要求MPI支持多线程级别至少为 MPI_THREAD_FUNNELED
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) printf("警告: MPI 线程支持级别不足\n");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // === 1. 一维行划分 ===
    int rows_per_proc = n / size;
    int remainder = n % size;
    int n_local = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // 计算偏移量
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
    
    // === 2. 分配内存 ===
    double *A_local = (double*)malloc(n_local * n * sizeof(double));
    double *C_local = (double*)calloc(n_local * n, sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    
    double *A_full = NULL, *C_full = NULL;
    if (rank == 0) {
        A_full = (double*)malloc(n * n * sizeof(double));
        C_full = (double*)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) {
            A_full[i] = 1.0;
            B[i] = 1.0;
            C_full[i] = 0.0;
        }
    }
    
    // === 3. 分发 A 和 B ===
    MPI_Scatterv(rank == 0 ? A_full : NULL, sendcounts, displs, MPI_DOUBLE,
                 A_local, n_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // === 4. 混合并行计算 ===
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    
    // 块大小：每个 OpenMP 线程一次处理的行数（可以调节）
    int block_size = 32;
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int local_row_start = 0; local_row_start < n_local; local_row_start += block_size) {
        int rows_to_compute = (local_row_start + block_size <= n_local) ? block_size : (n_local - local_row_start);
        // 每个线程调用 BLAS 计算自己负责的行块
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows_to_compute, n, n,
                    1.0, &A_local[local_row_start * n], n,
                    B, n,
                    0.0, &C_local[local_row_start * n], n);
    }
    
    double end = MPI_Wtime();
    double local_time = end - start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // === 5. 收集结果 ===
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
    
    // === 6. 输出 ===
    if (rank == 0) {
        double gflops = (2.0 * n * n * n / max_time) / 1e9;
        printf("MPI+OpenMP+BLAS: n=%d, MPI_procs=%d, OMP_threads=%d, time=%.5f sec, GFLOPS=%.2f\n",
               n, size, omp_get_max_threads(), max_time, gflops);
        // 可选验证：printf("C[0]=%f\n", C_full[0]);
        
        free(A_full); free(C_full);
        free(sendcounts); free(displs);
        free(recvcounts); free(recvdispls);
    }
    
    free(A_local); free(C_local); free(B);
    MPI_Finalize();
    return 0;
}