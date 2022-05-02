#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
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
    int ** graph;
    read_graph(argv[1], &graph);
    end = omp_get_wtime();
    printf("Matrix reading time: %.4f\n", end - start);
    get_graph_size(argv[1], &nodes_count, &edges_count);
    
    start = omp_get_wtime();
    float * pagerank = pagerank_custom_1(graph, nodes_count, 0.0000002);
    end = omp_get_wtime();
    printf("Pagerank computation time: %.4f\n", end - start);

    write_to_file(argv[2], pagerank, nodes_count);

    // just test to read the COO matrix
    struct mtx_COO coo;
    mtx_COO_create_from_file(&coo, argv[1]);
    for (int i = 0; i < edges_count; i++) {
        printf("%d %d %f\n", coo.row[i], coo.col[i], coo.data[i]);
    }
}