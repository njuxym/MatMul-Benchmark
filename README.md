# MatMul-Benchmark

**MatMul-Benchmark** is a performance evaluation toolkit for dense matrix multiplication (GEMM) implementations, developed as part of a parallel computing course. It systematically implements and measures over ten different variants, ranging from naive triple loops to highly optimized hybrid parallel approaches combining MPI, OpenMP, and BLAS, as well as GPU-accelerated CUDA kernels.

The project provides a unified testing framework that clearly illustrates the impact of each optimization technique, including:

- Memory‑friendly loop ordering (i‑k‑j vs i‑j‑k)
- Loop unrolling and cache blocking
- Vendor‑optimized BLAS libraries (OpenBLAS)
- Shared‑memory parallelism (OpenMP)
- Distributed‑memory parallelism (MPI)
- GPU acceleration (CUDA with shared memory)
- Hybrid models (MPI+BLAS, MPI+OpenMP+BLAS)

All versions are compiled and run with a single script, outputting execution time and GFLOPS for easy comparison. This makes it an excellent hands‑on reference for learning high‑performance computing concepts and performance tuning.

---

## Implementations

| Version | Description | Key Technique |
|---------|-------------|---------------|
| `v0_naive_ijk` | Naive implementation with i-j-k loop order | Serial |
| `v1_ikj_ordered` | Cache-friendly i-k-j loop order | Serial |
| `v2_loop_unrolling` | 4-way loop unrolling | Serial |
| `v3_blocking_v2` | Blocked matrix multiplication with tunable block sizes | Serial + Cache Optimization |
| `v4_blas` | OpenBLAS library call | BLAS |
| `v5_cuda_shared_memory` | CUDA shared memory tiling | CUDA |
| `v7_openmp` | OpenMP parallelization | OpenMP |
| `v8_mpi` | Distributed computing with row-wise partitioning | MPI |
| `v9_mpi_blas` | MPI + BLAS hybrid | MPI + BLAS |
| `v10_mpi_openmp_blas` | MPI + OpenMP + BLAS hybrid | MPI + OpenMP + BLAS |

> **Note:** `v6_cublas.cu` is an optional GPU implementation using cuBLAS and can be added to the compilation script as needed.

---

## Dependencies

- GCC with OpenMP support
- MPI (OpenMPI or MPICH)
- OpenBLAS
- CUDA Toolkit (for GPU versions)

---

## Build & Run

Clone the repository and execute the compilation script:

```bash
chmod +x compile.sh
./compile.sh
