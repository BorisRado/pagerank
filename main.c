#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include <stdbool.h>
#include "readers/custom_matrix.h"
#include "readers/mtx_sparse.h"
#include "pagerank_implementations/pagerank_custom.h"
#include "helpers/file_helper.h"

float * measure_time_custom_matrix_1(int ** edges, int * out_degrees, int nodes_count, int edges_count);
float * measure_time_custom_matrix_2(int ** edges, int * out_degrees, int nodes_count, int edges_count);

int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Usage: ./a.out <graph_file_name> <out_file_name>\n");
        exit(1);
    }
    int nodes_count, edges_count, i;
    double start, end;

    start = omp_get_wtime();
    int ** edges;
    int * out_degrees;
    int * in_degrees;
    read_edges(argv[1], &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count);
    end = omp_get_wtime();
    printf("Matrix reading time: %.4f\n", end - start);

    // compute pagerank with multiple strategies
    float * ref_pagerank = measure_time_custom_matrix_1(edges, out_degrees, nodes_count, edges_count);
    float * pagerank2 = measure_time_custom_matrix_2(edges, out_degrees, nodes_count, edges_count);

    // compare the obtained pageranks
    compare_vectors(ref_pagerank, pagerank2, nodes_count);

    // write the reference pagerank to file to be compared with the nx results
    write_to_file(argv[2], ref_pagerank, nodes_count);
}

float * measure_time_custom_matrix_1(int ** edges, int * out_degrees, int nodes_count, int edges_count) {
    double start, end;
    int ** graph;
    printf("\nCOMPUTING PAGERANK WITH CUSTOM_MATRIX_1\n");

    start = omp_get_wtime();
    format_graph(edges, out_degrees, &graph, nodes_count, edges_count);
    end = omp_get_wtime();
    printf("Matrix formatting time: %.4f\n", end - start);
    
    // with default init strategy
    start = omp_get_wtime();
    float * pagerank = pagerank_custom_1(graph, nodes_count, 0.0000002);
    end = omp_get_wtime();
    printf("Pagerank computation time (serial): %.4f\n\n", end - start);
    free(graph);
    return pagerank;

}

float * measure_time_custom_matrix_2(int ** edges, int * out_degrees, int nodes_count, int edges_count) {
    double start, end;
    int * leaves;
    int ** graph;
    printf("\nCOMPUTING PAGERANK WITH CUSTOM_MATRIX_2\n");

    start = omp_get_wtime();
    format_graph_2(edges, out_degrees, &graph, &leaves, nodes_count, edges_count);
    end = omp_get_wtime();
    printf("Matrix formatting time: %.4f\n", end - start);

    start = omp_get_wtime();
    float * pagerank = pagerank_custom_2(graph, out_degrees, leaves, nodes_count, 0.0000002);
    end = omp_get_wtime();
    printf("Pagerank computation time (serial): %.4f\n\n", end - start);
    return pagerank;
}