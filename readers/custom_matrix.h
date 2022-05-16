#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_custom_matrix(int ** graph, int nodes_count) {
    for(int i = 0; i < nodes_count; i++) {
        printf("%d (%d): ", i, graph[i][0]);

        for (int j = 1; j <= graph[i][0]; j++) {
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    }
}

int read_edges(char * file_name, int *** edges, int ** out_degrees,
            int ** in_degrees, int * nodes_count, int * edges_count) {
    /*
    Reads the graph at file_name and returns an 2D array containing one 
    entry for each edge present in the graph. `out_degrees` contains
    the number of outgoing edges from each node
    */
    FILE * fp;
    int * contiguous_space;
    int i;
    
    fp = fopen(file_name, "r");
    int from, to;
    
    // read number of nodes and edges
    if (fscanf(fp, "%d\t%d", nodes_count, edges_count) != 2) {
        printf("Error while reading first line...\n");
        return 1;
    }

    *out_degrees = (int *) calloc(*nodes_count, sizeof(int));
    *in_degrees = (int *) calloc(*nodes_count, sizeof(int));
    contiguous_space = (int *) malloc(2 * (*edges_count) * sizeof(int));
    *edges = (int **) malloc((*edges_count) * sizeof(int *));
    for (i = 0; i < (*edges_count); i++)
        (*edges)[i] = &contiguous_space[2 * i];

    i = 0;
    while (fscanf(fp, "%d\t%d", &from, &to) != EOF) {
        (*out_degrees)[from]++;
        (*in_degrees)[to]++;
        (*edges)[i][0] = from;
        (*edges)[i][1] = to;
        i++;
    }
    // for (i = 0; i < *nodes_count; i++)
    //     printf("here %d %d\n", i, (*out_degrees)[i]);

    fclose(fp);
    return 0;

}

int format_graph(int ** edges, int * out_degrees, int *** graph,
        int nodes_count, int edges_count) {
    /*
    Formats the edges contained in `edges` to the following format:
        [node]: neighbors_count | neighbor1 | neighbor2 | ...
    In other words, the array will be composed of `n` arrays. The array at index
    `i` will contain in the first entry the number of outgoing connections `out`,
    and then it will contain `out` indexes, that will state to which nodes the node
    `i` points to.

    Parameters:
        - (in) edges, list of edges obtained from the `read_edges` function
        - (in) out_degrees, array containing the number of outgoing edges for each node
        - (out) graph, the structure where the final graph will be saved
        - (in) nodes_count, edges_count
    Return value: 0 if everything ok, 1 otherwise
    */

    int * contiguous_space;
    int i;
    
    contiguous_space = (int*) malloc((edges_count + nodes_count) * sizeof(int));
    *graph = (int**) malloc(nodes_count * sizeof(int *));
    int CDF = 0;
    for (i = 0; i < nodes_count; i++) {
        (*graph)[i] = &contiguous_space[CDF];
        CDF = CDF + 1 + out_degrees[i];
        (*graph)[i][0] = out_degrees[i];
        out_degrees[i] = 0;
    }

    // omp seems not to improve the performances
    // int threads_count = omp_get_max_threads();
    // printf("Running %d threads\n", threads_count);
    // #pragma omp parallel for schedule(guided)
    int from, to;
    for (i = 0; i < edges_count; i++) {
        from = edges[i][0];
        to = edges[i][1];
        (*graph)[from][out_degrees[from] + 1] = to;
        out_degrees[from]++;
    }
    return 0;
}

int format_graph_2(int ** edges, int * out_degrees, int *** graph,
        int ** leaves, int nodes_count, int edges_count) {
    /*
    Formats the edges contained in `edges` to the following format:
        [node]: neighbor1 | neighbor2 | ...
    In other words, the array will be composed of `n+1` arrays. The array at index
    `i` will contain `out` indexes, that will state to which nodes the node
    `i` points to. The final array, the one at index `nodes_count`, will state 
    in the first entry how many nodes have 0 out degree, and then it will list such
    nodes

    Parameters:
        - (in) edges, list of edges obtained from the `read_edges` function
        - (in) out_degrees, array containing the number of outgoing edges for each node
        - (out) graph, the structure where the final graph will be saved
        - (in) nodes_count, edges_count
    Return value: 0 if everything ok, 1 otherwise
    */

    int * contiguous_space;
    int i;
    
    // count how many nodes have 0 out degree
    int leaves_count = 0;  // leaves are nodes with 0 out-degree
    for (i = 0; i < nodes_count; i++) {
        leaves_count += out_degrees[i] == 0;
    }

    contiguous_space = (int*) malloc((edges_count) * sizeof(int));
    *graph = (int**) malloc((nodes_count) * sizeof(int *));
    *leaves = (int *) malloc(leaves_count * sizeof(int));
    (*leaves)[0] = leaves_count;

    int CDF = 0;
    leaves_count = 0;
    for (i = 0; i < nodes_count; i++) {
        if (out_degrees[i] == 0) {
            (*leaves)[leaves_count + 1] = i;
            leaves_count++;
        }
        (*graph)[i] = &contiguous_space[CDF];
        CDF = CDF + out_degrees[i];
        out_degrees[i] = 0;
    }
    // for (i = 1; i <= (*leaves)[0]; i++) {
    //     printf("leaf: %d\n", (*leaves)[i]);
    // }
    // printf("ok\n");
    // fflush(stdout);

    // omp seems not to improve the performances
    // int threads_count = omp_get_max_threads();
    // printf("Running %d threads\n", threads_count);
    // #pragma omp parallel for schedule(guided)
    int from, to;
    for (i = 0; i < edges_count; i++) {
        from = edges[i][0];
        to = edges[i][1];
        (*graph)[from][out_degrees[from]] = to;
        out_degrees[from]++;
    }
    return 0;
}