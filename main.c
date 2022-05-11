#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include <stdbool.h>
#include "readers/custom_matrix.h"
#include "readers/mtx_sparse.h"
#include "pagerank_implementations/pagerank_custom.h"
#include "helpers/file_helper.h"

int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Usage: ./a.out <graph_file_name> <out_file_name>\n");
        exit(1);
    }
    int nodes_count, edges_count;
    double start, end;

    start = omp_get_wtime();
    int ** edges;
    int * node_neighbors;
    read_edges(argv[1], &edges, &node_neighbors, &nodes_count, &edges_count);
    end = omp_get_wtime();
    printf("Matrix reading time: %.4f\n", end - start);

    int ** graph;
    start = omp_get_wtime();
    format_graph(edges, node_neighbors, &graph, nodes_count, edges_count);
    end = omp_get_wtime();
    printf("Matrix formatting time: %.4f\n", end - start);
    fflush(stdout);
    
    start = omp_get_wtime();
    float * pagerank = pagerank_custom_1(graph, nodes_count, 0.0000002);
    end = omp_get_wtime();
    printf("Pagerank computation time (serial): %.4f\n", end - start);

    write_to_file(argv[2], pagerank, nodes_count);

    // just test to read the COO matrix
    // struct mtx_COO coo;
    // mtx_COO_create_from_file(&coo, argv[1]);
}