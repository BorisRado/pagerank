#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_custom_matrix(int ** graph, int nodes_count) {
    for(int i = 0; i < nodes_count; i++) {
        printf("%d (%d): ", i, graph[i][0]);
        fflush(stdout);

        for (int j = 1; j <= graph[i][0]; j++) {
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    }
}

int read_graph(char* file_name, int *** graph) {
    /*
    Reads a matrix in `graph` from file and stores it into an array of arrays as follows:
        [node]: neighbors_count | neighbor1 | neighbor2 | ...
    In other words, the array will be composed of `n` arrays. The array at index
    `i` will contain in the first entry, the number of outgoing connections `out`,
    and then it will contain `out` indexes, that will state to which nodes the node
    `i` points to.

    Parameters:
        - file_name, e.g. data/web-Google.txt
        - graph, address of a int** pointer, to which the results will be saved
    Return value: 0 if everything ok, 1 otherwise

    TO-DO: also implement the reverse representation: i.e. instead of
        `node: <list of nodes the nodes points to>`
    implement also
        `node: <list of nodes that point to the node>`
    */

    FILE * fp;
    size_t len = 0;
    int ** edges;
    int * node_neighbors;
    int * contiguous_space;
    int i, nodes_count, edges_count;
    
    fp = fopen(file_name, "r");
    int from, to;
    
    // read number of nodes and edges
    if (fscanf(fp, "%d\t%d", &nodes_count, &edges_count) != 2) {
        printf("Error while reading first line...\n");
        return 1;
    }

    node_neighbors = (int *) calloc(nodes_count, sizeof(int));
    contiguous_space = (int *) malloc(2 * edges_count * sizeof(int));
    edges = (int **) malloc(edges_count * sizeof(int *));
    for (i = 0; i < edges_count; i++)
        edges[i] = &contiguous_space[2 * i];

    i = 0;
    while (fscanf(fp, "%d\t%d", &from, &to) != EOF) {
        node_neighbors[from]++;
        edges[i][0] = from;
        edges[i][1] = to;
        i++;
    }
    fclose(fp);

    contiguous_space = (int*) malloc((edges_count + nodes_count) * sizeof(int));
    *graph = (int**) malloc(nodes_count * sizeof(int *));
    int CDF = 0;
    for (i = 0; i < nodes_count; i++) {
        (*graph)[i] = &contiguous_space[CDF];
        CDF = CDF + 1 + node_neighbors[i];
        (*graph)[i][0] = node_neighbors[i];
        node_neighbors[i] = 1;
    }

    // omp seems not to improve the performances
    // int threads_count = omp_get_max_threads();
    // printf("Running %d threads\n", threads_count);
    // #pragma omp parallel for schedule(guided)
    for (i = 0; i < edges_count; i++) {
        from = edges[i][0];
        to = edges[i][1];
        (*graph)[from][node_neighbors[from]] = to;
        node_neighbors[from]++;
    }

    free(node_neighbors);
    free(edges);

    return 0;
}