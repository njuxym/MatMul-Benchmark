#!/bin/bash
gcc -O2 v0_naive_ijk.c -o v0_naive_ijk
gcc -O2 v1_ikj_ordered.c -o v1_ikj_ordered
gcc -O2 v2_loop_unrolling.c -o v2_loop_unrolling
gcc -O2 v3_blocking_v2.c -o v3_blocking_v2
gcc -O2 v4_blas.c -o v4_blas -lopenblas
nvcc -O2 v5_cuda_shared_memory.cu -o v5_cuda_shared_memory
gcc -O2 -fopenmp v7_openmp.c -o v7_openmp
mpicc -O2 v8_mpi.c -o v8_mpi
mpicc -O2 v9_mpi_blas.c -o v9_mpi_blas -lopenblas
mpicc -O2 v10_mpi_openmp_blas.c -o v10_mpi_openmp_blas -fopenmp -lopenblas
#export OPENBLAS_NUM_THREADS=1

for version in v0_naive_ijk v1_ikj_ordered v2_loop_unrolling v3_blocking_v2 v4_blas v5_cuda_shared_memory; do
    echo "=== $version ==="
#    time ./$version
    ./$version
    echo ""
done

echo "=== v7_openmp ==="
./v7_openmp 1024 4
./v7_openmp 1024 8
./v7_openmp 1024 16
echo ""

echo "=== v8_mpi ==="
mpirun -np 4 ./v8_mpi 1024
mpirun -np 8 ./v8_mpi 1024
mpirun -np 16 ./v8_mpi 1024
echo ""

echo "=== v9_mpi_blas ==="
mpirun -np 2 ./v9_mpi_blas 1024
mpirun -np 4 ./v9_mpi_blas 2048
echo ""

echo "=== v10_mpi_openmp_blas ==="
export OMP_NUM_THREADS=2
mpirun -np 2 ./v10_mpi_openmp_blas 1024
mpirun -np 4 ./v10_mpi_openmp_blas 2048
echo ""
