#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:10
#SBATCH --output=logs/logs.txt

srun ./run.sh main.c data/collaboration_imdb_out.net results/collaboration_imdb_out.net