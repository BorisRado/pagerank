# Pagerank

## How graphs are stored

### Preprocessing
The C code assumes the following:
* the first line of the file contains two integers: the first one states the number of nodes, the second one states the number of edges;
* if there are `n` nodes, all are in the range `0:n-1`

You can preprocess the graph in such a way with a convenience python script. Just issue the commamd 
```
python3 py_src/prepare_graph.py <file_name>
```
This, given a file `file.txt` that contains the non-formatted graph, will create a `file_out.txt` file will the graph in the correct format.

### Graph representations
The currently sopported ways to store the graphs in memory are the following:
* COO: COOrdinate format, stores data in three arrays for row, column and value of each datapoint. Not suitable for parallelization.
* CSR: Compressed Sparse Row format, stores data sorted by row in three arrays. The first two store column and value for each datapoint, whereas the third stores the pointers to the beginning of each row.
* ELL: ELLpack format, expands each row of CSR to same length and transposes the matrix, allowing the row pointers to be discarded. Stores only column and value for each datapoint, with additional integer for number of elements per row. 
* JDS: Jagged Diagonal Storage format, splits the matrix into submatrices with similar row lengths. Each submatrix is converted into a separate ELL format, allowing for better spatial efficiency.
* Non-matrix based approach: in this approach, we store the graph as a 2D array. Two separate methods are implemented in this category: in the first one, the array at index `i` contains the nodes, to which the node `i` points to. Similarly, in the second approach, the array at index `i` contains the nodes that point to node `i`. In both cases, we have two additional arrays, that state the in-degrees and out-degrees of all the nodes. Also, in both cases we store an additional array, which contains the nodes that have 0 out degree (the pagerank of these nodes is lost at every iteration, and having this array speeds up the execution of the program).

## Running the examples
1. Add the graph to the `data` folder (create the folder if not present);
2. Prepare the graph with `python3 py_src/prepare_graph.py <graph_file>`;
3. Compile, e.g. `gcc main.c -lm -fopenmp -O2 -o main`
4. Run, `./main <graph_file> <output_file>`. Output file is the file where the results will be saved;
5. (*optional*) Verify the correctness of the results, `python3 py_src/compare_pagerank.py <graph_file> <output_file>`. Note that this command requires `networkx`, `numpy` and `scipy`. The script computes the true pagerank with networkx, and compares the results with the ones reported in `<output_file>`.

Alternatively, you may use the `run.sh` script in place of the steps $3$, $4$, and $5$. On HPC, use `sbatch pr_submit.sh` (and set the `GRAPH` variable to the name of the file that contains the graph to be processed).

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

## How is the project structured?
In order to carry out all the experiments, compile and run the `main.c`, which invokes the code to read the graph, format it, and finally run the pagerank algorithm with multiple algorithms we developed. You may use `run.sh` locally or `sbatch pr_submit.sh` on HPC in order to compile and run (and optionally test the correctness of the results).

Apart from that, the code is logically divided into folders:
* `helpers`: contains some common code, that is used multiple times in the project. For example, writing data to a file, initializing OCL platforms and other required structures, swapping pointers, ...
* `kernels`: contains the OpenCL kernels that are used to compute the pagerank on GPUs;
* `pagerank_implementations`: contains the actual implementations that compute the pagerank;
* `py_src`: contain the python code used to format the graph into the provided format (`prepare_graph.py`) and to compute the true pagerank value with an established library, i.e. networkx (`compare_pageranks.py`);
* `readers`: contain functions that read the graph from the file system and format it in the required type;
