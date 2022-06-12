#!/bin/bash

# compile, sumbit with sbatch & finally monitor the job

./compile.sh
if ! [ $? -eq 0 ]; then
    echo "Compilation failed. Exiting..."
    exit 1
fi

sbatch pr_submit.sh --omp-ocl
watch squeue --me
