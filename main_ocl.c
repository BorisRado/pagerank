#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include "readers/custom_matrix.h"
#include "readers/mtx_sparse.h"
#include "readers/mtx_hybrid.h"
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
    if (read_edges(argv[1], &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
        exit(1);
    printf("Read edges.\n");

    // compute pagerank with custom out mtx (MPI)
    int ** graph;
    int * leaves;
    int leaves_count;
    format_graph_out(edges, out_degrees, &leaves_count, &leaves, &graph, nodes_count, edges_count);
    
    float * ref_pagerank = pagerank_custom_out(graph, out_degrees, leaves_count, leaves, nodes_count, EPSILON);
    printf("Computed custom out.\n");
    // compute pagerank with OCL
    
    
    mtx_CSR mCSR;
    if (get_CSR_from_file(&mCSR, argv[1]) != 0) {
        printf("Could not create CSR.\n");
        exit(1);
    }
    
    
    /*mtx_ELL mELL;
    if (get_ELL_from_file(&mELL, argv[1]) != 0) {
        printf("Could not create ELL.\n");
        exit(1);
    }*/

    mtx_JDS mJDS;
    int * dangling;
    int num_pieces = 30; // if number is too low (<=25), can return wrong answer instead of OOM error
    if (get_JDS_from_file(&mJDS, &dangling, &num_pieces, argv[1]) != 0) {
        printf("Could not create JDS.\n");
        exit(1);
    }

    float start, end;

    float * csr_sca_pagerank = pagerank_CSR_scalar(mCSR, &start, &end);
    
    float * csr_vec_pagerank = pagerank_CSR_vector(mCSR, &start, &end);

    //float * ell_pagerank = pagerank_ELL(mELL, &start, &end);

    float * jds_pagerank = pagerank_JDS(mJDS, &dangling, &start, &end);

    // compare the obtained pageranks
    compare_vectors_detailed(ref_pagerank, csr_sca_pagerank, nodes_count);
    compare_vectors_detailed(ref_pagerank, csr_vec_pagerank, nodes_count);
    //compare_vectors_detailed(ref_pagerank, ell_pagerank, nodes_count);
    compare_vectors_detailed(ref_pagerank, jds_pagerank, nodes_count);
    
    // free data
    free(edges);
    free(out_degrees);
    free(in_degrees);
    free(graph);
    mtx_CSR_free(&mCSR);
    //mtx_ELL_free(&mELL);
    mtx_JDS_free(&mJDS);

    return 0;
}




