#!/bin/bash

# compiles all the C code
# run this script before sumbitting the job with `pr_submit.sh --*`

module load CUDA
module load OpenMPI

mkdir compiled_code 2> /dev/null

gcc main.c -lm -fopenmp -lOpenCL -O2 -o compiled_code/main.out
if [ $? -ne 0 ]; then
    echo "Compilation of main.c failed. Exiting..."
    exit 1
fi

gcc main_ocl.c -lm -fopenmp -lOpenCL -O2 -o compiled_code/csr.out
if [ $? -ne 0 ]; then
    echo "Compilation of main_ocl.c failed. Exiting..."
    exit 1
fi

mpicc main_mpi.c -lm -fopenmp -lOpenCL -O2 -o compiled_code/mpi.out
if [ $? -ne 0 ]; then
    echo "Compilation of main_mpi.c failed. Exiting..."
    exit 1
fi