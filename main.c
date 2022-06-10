#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include <stdbool.h>
#include "readers/custom_matrix.h"
#include "readers/mtx_sparse.h"
#include "pagerank_implementations/pagerank_custom.h"
#include "helpers/file_helper.h"
#include "global_config.h"

float * measure_time_custom_matrix_out(int ** edges, int * out_degrees, int nodes_count, int edges_count);
float * measure_time_custom_matrix_in(int ** edges, int * in_degrees, int * out_degrees, int nodes_count, int edges_count);
float * measure_time_csr(int ** edges, int * in_degrees, int * out_degrees, int nodes_count, int edges_count);

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
    if(read_edges(argv[1], &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
        exit(1);
    
    end = omp_get_wtime();
    printf("Matrix reading time: %.4f\n", end - start);

    // compute pagerank with multiple strategies
    float * ref_pagerank = measure_time_custom_matrix_out(edges, out_degrees, nodes_count, edges_count);
    float * pagerank_in = measure_time_custom_matrix_in(edges, in_degrees, out_degrees, nodes_count, edges_count);

    // compare the obtained pageranks
    compare_vectors(ref_pagerank, pagerank_in, nodes_count);

    // write the reference pagerank to file to be compared with the nx results
    write_to_file(argv[2], ref_pagerank, nodes_count);
}

float * measure_time_custom_matrix_out(int ** edges, int * out_degrees, int nodes_count, int edges_count) {
    double start, end;
    int ** graph;
    int * leaves;
    int leaves_count;

    printf("\nCOMPUTING PAGERANK WITH CUSTOM_MATRIX_OUT\n");
    start = omp_get_wtime();
    format_graph_out(edges, out_degrees, &leaves_count, &leaves, &graph, nodes_count, edges_count);
    end = omp_get_wtime();
    printf("Matrix formatting time: %.4f\n", end - start);

    start = omp_get_wtime();
    float * pagerank = pagerank_custom_out(graph, out_degrees, leaves_count, leaves, nodes_count, EPSILON);
    end = omp_get_wtime();
    printf("Pagerank computation time (serial): %.4f\n\n", end - start);
    free(graph);
    return pagerank;

}

float * measure_time_custom_matrix_in(int ** edges, int * in_degrees, int * out_degrees, int nodes_count, int edges_count) {
    double start, end;
    int ** graph;
    int * leaves;
    int leaves_count;

    printf("\nCOMPUTING PAGERANK WITH CUSTOM_MATRIX_IN\n");
    start = omp_get_wtime();
    format_graph_in(edges, in_degrees, out_degrees, &leaves_count, &leaves, &graph, nodes_count, edges_count);
    end = omp_get_wtime();
    printf("Matrix formatting time: %.4f\n", end - start);
    
    start = omp_get_wtime();
    float * pagerank = pagerank_custom_in(graph, in_degrees, out_degrees, leaves_count, leaves, nodes_count, EPSILON, false);
    end = omp_get_wtime();
    printf("Pagerank computation time (serial): %.4f\n\n", end - start);

    // omp
    start = omp_get_wtime();
    float * pagerank_omp = pagerank_custom_in(graph, in_degrees, out_degrees, leaves_count, leaves, nodes_count, EPSILON, true);
    end = omp_get_wtime();
    printf("Pagerank computation time (OMP with %d threads): %.4f\n\n", omp_get_max_threads(), end - start);

    // ocl - pass `start` and `end` to function so not to measure compilation etc.
    float * pagerank_ocl_simple = pagerank_custom_in_ocl(graph, in_degrees, out_degrees, leaves_count,
                    leaves ,nodes_count, edges_count, EPSILON, &start, &end, "pagerank_step_simple");
    printf("Pagerank computation time (OCL, one thread per row): %.4f\n\n", omp_get_max_threads(), end - start);

    float * pagerank_ocl = pagerank_custom_in_ocl(graph, in_degrees, out_degrees, leaves_count,
                    leaves ,nodes_count, edges_count, EPSILON, &start, &end, "pagerank_step");
    printf("Pagerank computation time (OCL): %.4f\n\n", omp_get_max_threads(), end - start);

    compare_vectors(pagerank, pagerank_omp, nodes_count);
    compare_vectors(pagerank, pagerank_ocl, nodes_count);

    free(graph);
    return pagerank;

}