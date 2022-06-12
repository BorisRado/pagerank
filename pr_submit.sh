#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --time=0:0:50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/logs.txt
#SBATCH --gpus=1

# TO-DO: the MPI version will fail if --gpus=1

# runs the experiments:
#   when using the `--mpi` flag, run only the MPI algorithms
#   when using the `--omp-ocl` flag, run only the serial, OpenMP, OpenCL algorithms
#   when using the `--csr` flag, run only the CSR implementations
#   when `--all` flag is passed, run all the experiments
# the script requires no parameters apart from which implementations to run.
# change the graph and out_file inside this file
if [ "$#" -eq 0 ]; then
    echo "You need to provide which experiments to perform. Use --all to run all. Exiting..."
    exit 1;
fi

module load CUDA # needed for OpenCL
module load OpenMPI # needed for MPI

# all the experiments are performed with the same graph
GRAPH=data/web-Google_out.txt
# GRAPH=data/toy.txt
# GRAPH=data/collaboration_imdb_out.net
# GRAPH=data/karate_club_out.net
OUT_FILE="results/web-Google_out.txt"

# empty file if it already exists. This way the programs can append to it
rm "$OUT_FILE" 2> /dev/null
touch "$OUT_FILE"

# compile and run the serial, OpenMP, & OpenCL versions
if [[ $* == *--omp-ocl* ]] || [[ $* == *--all* ]]; then
    echo "Running OMP, OCL and serial versions..."
    srun compiled_code/main.out "$GRAPH" "$OUT_FILE"
    echo "______________________________________"
fi

# compile and run the CSR versions
if [[ $* == *--csr* ]] || [[ $* == *--all* ]]; then
    echo "Running CSR version..."
    srun compiled_code/csr.out "$GRAPH" "$OUT_FILE"
    echo "______________________________________"
fi


# compile and run the MPI version
if [[ $* == *--mpi* ]] || [[ $* == *--all* ]]; then
    echo "Running MPI version..."
    srun --mpi=pmix compiled_code/mpi.out "$GRAPH" "$OUT_FILE"
    echo "______________________________________"
fi

if [[ $* == *--test-nx* ]]; then
    echo "Testing results with networkx..."
    SECONDS=0
    ~/.venv/networkx/bin/python3 py_src/compare_pagerank.py $GRAPH $3
    echo "Computation with networkx took $SECONDS seconds"
fi