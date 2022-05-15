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
    if (fp == NULL) {
        printf("ERROR while reading graph `%s`", file_name);
        exit(1);
    }

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

int format_graph_out(int ** edges, int * out_degrees, int * leaves_count, int ** leaves, 
        int *** graph, int nodes_count, int edges_count) {
    /*
    Formats the edges contained in `edges` to the following format:
        [node]: neighbor1 | neighbor2 | ...
    In other words, the array will be composed of `n` arrays, where `n` is the 
    number of nodes. The array at index `i` will be composed of `out` entries, `out`
    being the out degree of the node, that will state to which nodes the node
    `i` points to. `leaves_count` and `leaves` will contain the nodes that have no 
    outgoing edges (we call them leaves).

    Parameters:
        - (in) edges, list of edges obtained from the `read_edges` function
        - (in) out_degrees, array containing the number of outgoing edges for each node
        - (out) leaves_count, in this variable will be stored the number of 0-degree nodes
        - (out) leaves, will contain `leaves_count` entries, and will state which nodes have 0 out degree
        - (out) graph, the structure where the final graph will be saved
        - (in) nodes_count, edges_count
    Return value: 0 if everything ok, 1 otherwise
    */

    int * contiguous_space;
    int i;

    // count how many nodes have 0 out degree
    *leaves_count = 0;
    for (i = 0; i < nodes_count; i++) {
        *leaves_count += out_degrees[i] == 0;
    }
    *leaves = (int *) malloc(*leaves_count * sizeof(int));
    
    contiguous_space = (int*) malloc(edges_count * sizeof(int));
    *graph = (int**) malloc(nodes_count * sizeof(int *));
    int CDF = 0, _leaves_count = 0;

    for (i = 0; i < nodes_count; i++) {
        if (out_degrees[i] == 0) {
            (*leaves)[_leaves_count] = i;
            _leaves_count++;
        }
        (*graph)[i] = &contiguous_space[CDF];
        CDF = CDF + out_degrees[i];
        out_degrees[i] = 0;
    }

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

int format_graph_in(int ** edges, int * in_degrees, int * out_degrees, int * leaves_count, int ** leaves,
        int *** graph, int nodes_count, int edges_count) {
    
    /*
    Formats the edges contained in `edges` to the following format:
        [node]: neighbor1 | neighbor2 | ...
    In other words, the array will be composed of `n` arrays, where `n` is the 
    number of nodes. The array at index `i` will be composed of `in` entries, `in`
    being the in degree of the node, that will state which nodes have a connection
    node `i`. `leaves_count` and `leaves` will contain the nodes that have no 
    outgoing edges (we call them leaves).

    Parameters:
        - (in) edges, list of edges obtained from the `read_edges` function
        - (in) in_degrees, array containing the number of incoming edges for each node
        - (in) out_degrees, array containing the number of outgoing edges for each node
        - (out) leaves_count, in this variable will be stored the number of 0-out-degree nodes
        - (out) leaves, will contain `leaves_count` entries, and will state which nodes have 0 out degree
        - (out) graph, the structure where the final graph will be saved
        - (in) nodes_count, edges_count
    Return value: 0 if everything ok, 1 otherwise
    */

    int * contiguous_space;
    int i;

    // count how many nodes have 0 out degree
    *leaves_count = 0;
    for (i = 0; i < nodes_count; i++) {
        *leaves_count += out_degrees[i] == 0;
    }
    *leaves = (int *) malloc(*leaves_count * sizeof(int));
    
    contiguous_space = (int*) malloc(edges_count * sizeof(int));
    *graph = (int**) malloc(nodes_count * sizeof(int *));
    int CDF = 0, _leaves_count = 0;

    for (i = 0; i < nodes_count; i++) {
        if (out_degrees[i] == 0) {
            (*leaves)[_leaves_count] = i;
            _leaves_count++;
        }
        (*graph)[i] = &contiguous_space[CDF];
        CDF = CDF + in_degrees[i];
        in_degrees[i] = 0;
    }

    // int threads_count = omp_get_max_threads();
    // printf("Running %d threads\n", threads_count);
    // #pragma omp parallel for schedule(guided)
    int from, to;
    for (i = 0; i < edges_count; i++) {
        from = edges[i][0];
        to = edges[i][1];
        (*graph)[to][in_degrees[to]] = from;
        in_degrees[to]++;
    }
    return 0;
}