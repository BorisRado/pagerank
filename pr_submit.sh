#!/bin/bash

#SBATCH -N1
#SBATCH --reservation=fri
#SBATCH -n1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:15
#SBATCH --output=logs/logs.txt
#SBATCH --gpus=1

module load CUDA # needed for OpenCL

# GRAPH=data/web-Google_out.txt
# GRAPH=data/toy.txt
# GRAPH=data/collaboration_imdb_out.net
GRAPH=data/karate_club_out.net

srun ./run.sh main.c $GRAPH results/collaboration_imdb_out.net