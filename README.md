# Pagerank

## How graphs are stored
The C code assumes the following:
* the first line of the file contains two integers: the first one states the number of nodes, the second one states the number of edges;
* if there are `n` nodes, all are in the range `0:n-1`

You can preprocess the graph in such a way with a convenience python script. Just issue the commamd 
```
python3 py_src/prepare_graph.py <file_name>
```

The currently sopported ways to store the graphs in memory are the following:
* COO*
* CSR*
* ELL*
* Non-matrix based approach: in this approach, we store the graph as a 2D array. This array is composed of `n` arrays, where `n` is the number of nodes of the graph. Then, each such sub-array is composed of `k+1` elements, where `k` is the number of neighbors of a node. The first element in the array is the number of neighbors of the node, i.e. `k`, the following elements are the neighbors.

\* The representation is still not complete. The matrix contains only 1.0 values, but they should be normalized based on the number of neighboring nodes.

## Running the examples
1. Add the graph to the `data` folder (create the folder if not present);
2. Prepare the graph with `python3 py_src/prepare_graph.py <graph_file>`;
3. Compile, e.g. `gcc main.c -lm -fopenmp -O2 -o main`
4. Run, `./main <graph_file> <output_file>`. Output file is the file where the results will be saved;
5. (*optional*) Verify the correctness of the results, `python3 py_src/compare_pagerank.py <graph_file> <output_file>`. Note that this command requires `networkx`, `numpy` and `scipy`. The script computes the true pagerank with networkx, and compares the results with the ones reported in `<output_file>`.

Alternatively, you may use the `run.sh` script in place of the steps $3$, $4$, and $5$.

### Setting up Python on HPC
Run the following commands:
```
python3 -m venv ~/.venv/networkx python=3.8
source ~/.venv/networkx/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Transferring graphs to HPC
Create a directory `data`, and from your local PC run `scp -i <key> -r data/* <username>@nsc-login1.ijs.si:~/hpc/pagerank/data`