#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include "readers/custom_matrix.h"
#include "readers/mtx_sparse.h"
#include "pagerank_implementations/pagerank_custom.h"
#include "pagerank_implementations/pagerank_OCL.h"
#include "helpers/file_helper.h"


int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Usage: ./a.out <graph_file_name> <out_file_name>\n");
        exit(1);
    }

    // read edges from file
    int nodes_count, edges_count, i;
    int ** edges;
    int * out_degrees;
    int * in_degrees;
    if(read_edges(argv[1], &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
        exit(1);
    printf("Read edges.\n");

    // compute pagerank with custom out mtx (MPI)
    int ** graph;
    int * leaves;
    int leaves_count;
    format_graph_out(edges, out_degrees, &leaves_count, &leaves, &graph, nodes_count, edges_count);
    float * ref_pagerank = pagerank_custom_out(graph, out_degrees, leaves_count, leaves, nodes_count, EPSILON);
    printf("Computed custom out.\n");
    
    // compute pagerank with OCL CSR
    mtx_CSR mCSR;
    if(get_CSR_from_edges(&mCSR, &edges, &out_degrees, &nodes_count, &edges_count) != 0) {
        printf("Could not create CSR.\n");
        exit(1);
    }
    float * csr_pagerank = pagerank_CSR_vector(mCSR);
    printf("Computed OCL pagerank.\n");

    // compare the obtained pageranks
    compare_vectors(ref_pagerank, csr_pagerank, nodes_count);

    // free data
    free(edges);
    free(out_degrees);
    free(in_degrees);
    free(graph);
    mtx_CSR_free(&mCSR);

    return 0;
}
