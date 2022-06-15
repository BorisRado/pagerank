#!/bin/bash

# compile, sumbit with sbatch & finally monitor the job

./compile.sh
if ! [ $? -eq 0 ]; then
    echo "Compilation failed. Exiting..."
    exit 1
fi

TMP_SUBMIT_FILE=pr_sumbit_tmp.sh
for i in $(seq 0 0); do # change this line to update the number of iterations
    sed "s/logs.txt/logs_$i.txt/g" pr_submit.sh > "$TMP_SUBMIT_FILE"
    sbatch "$TMP_SUBMIT_FILE" --omp-ocl
    if ! [ $? -eq 0 ]; then
        echo "Could not submit job. Exiting..."
        exit 1
    fi
done
rm "$TMP_SUBMIT_FILE"

watch squeue --me
