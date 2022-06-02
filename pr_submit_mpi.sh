#!/bin/bash
#SBATCH -N1
#SBATCH --reservation=fri
#SBATCH -n1
#SBATCH --ntasks=16
##SBATCH --cpus-per-task=16
#SBATCH --time=00:00:15
#SBATCH --output=logs/logs.txt

module load CUDA # needed for OpenCL
module load OpenMPI

GRAPH=data/web-Google_out.txt
# GRAPH=data/toy.txt
# GRAPH=data/collaboration_imdb_out.net
# GRAPH=data/karate_club_out.net

mpicc main_mpi.c -lm -fopenmp -lOpenCL -O2 -o a  
srun --mpi=pmix a $GRAPH results/web-Google_out.net