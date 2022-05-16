#!/bin/bash

# this script compiles the program passed as the first argument,
# runs the program on the graph, that is passed as the second
# argument, stores the result in the file passed as third argument,
# and finally invokes the python program that verifies the correctness
# of the results. Therefore, invoke the script as:
#       `./run.sh <c_program> <graph_file> <out_file>`

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit 1;
fi

# compile the program
gcc $1 -lm -fopenmp -O2

# run the program
./a.out $2 $3

SECONDS=0
~/.venv/networkx/bin/python3 py_src/compare_pagerank.py $2 $3
echo "Computation with networkx took $SECONDS seconds"