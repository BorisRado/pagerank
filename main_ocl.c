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

    double timer, start_gl, end_gl;

    // compute pagerank with OCL
    timer = omp_get_wtime();
    int nodes_count, edges_count, i;
    int ** edges;
    int * out_degrees;
    int * in_degrees;
    if (read_edges(argv[1], &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count)) {
        printf("Could not create custom format.\n");
        exit(1);
    }
    printf("Read edges.\n");
    int ** graph;
    int * leaves;
    int leaves_count;
    format_graph_in(edges, in_degrees, out_degrees, &leaves_count, &leaves, &graph, nodes_count, edges_count);
    timer = omp_get_wtime() - timer;
    printf("Custom format read time: %f.\n", timer);
    
    timer = omp_get_wtime();
    mtx_CSR mCSR;
    if (get_CSR_from_file(&mCSR, argv[1]) != 0) {
        printf("Could not create CSR.\n");
        exit(1);
    }
    timer = omp_get_wtime() - timer;
    printf("CSR matrix read time: %f.\n", timer);
    
    /*timer = omp_get_wtime();
    mtx_ELL mELL;
    if (get_ELL_from_file(&mELL, argv[1]) != 0) {
        printf("Could not create ELL.\n");
        exit(1);
    }
    timer = omp_get_wtime() - timer;
    printf("ELL matrix read time: %f.\n", timer);

    timer = omp_get_wtime();    
    mtx_JDS mJDS;
    int * dangling;
    int num_pieces = 3; // if number is too low (<=25), can return wrong answer instead of OOM error
    if (get_JDS_from_file(&mJDS, &dangling, &num_pieces, argv[1]) != 0) {
        printf("Could not create JDS.\n");
        exit(1);
    }
    timer = omp_get_wtime() - timer;
    printf("JDS matrix read time: %f.\n", timer);*/


    char kernel1[] = "pagerank_step_simple";
    char kernel2[] = "pagerank_step";
    char kernel3[] = "pagerank_step_expanded";
    
    timer = omp_get_wtime(); 
    float * custom_pagerank1 = pagerank_custom_in_ocl(graph, in_degrees, out_degrees, leaves_count, leaves, 
                nodes_count, edges_count, EPSILON, &start_gl, &end_gl, kernel1);
    timer = omp_get_wtime() - timer;
    printf("Custom kernel 1 total time: %f.\n", timer);

    timer = omp_get_wtime(); 
    float * custom_pagerank2 = pagerank_custom_in_ocl(graph, in_degrees, out_degrees, leaves_count, leaves, 
                nodes_count, edges_count, EPSILON, &start_gl, &end_gl, kernel2);
    timer = omp_get_wtime() - timer;
    printf("Custom kernel 2 total time: %f.\n", timer);

    timer = omp_get_wtime(); 
    float * custom_pagerank3 = pagerank_custom_in_ocl(graph, in_degrees, out_degrees, leaves_count, leaves, 
                nodes_count, edges_count, EPSILON, &start_gl, &end_gl, kernel3);
    timer = omp_get_wtime() - timer;
    printf("Custom kernel 3 total time: %f.\n", timer);

    timer = omp_get_wtime(); 
    float * csr_sca_pagerank = pagerank_CSR_scalar(mCSR);
    timer = omp_get_wtime() - timer;
    printf("CSR scalar OCL total time: %f.\n", timer);
    
    timer = omp_get_wtime(); 
    float * csr_vec_pagerank = pagerank_CSR_vector(mCSR);
    timer = omp_get_wtime() - timer;
    printf("CSR vector OCL total time: %f.\n", timer);

    /*timer = omp_get_wtime(); 
    float * ell_pagerank = pagerank_ELL(mELL);
    timer = omp_get_wtime() - timer;
    printf("ELL OCL total time: %f.\n", timer);

    timer = omp_get_wtime(); 
    float * jds_pagerank = pagerank_JDS(mJDS, &dangling);
    timer = omp_get_wtime() - timer;
    printf("JDS OCL total time: %f.\n", timer);*/
    

    // compare the obtained pageranks
    // compare_vectors_detailed(ref_pagerank, csr_sca_pagerank, nodes_count);
    // compare_vectors_detailed(ref_pagerank, csr_vec_pagerank, nodes_count);
    // compare_vectors_detailed(ref_pagerank, ell_pagerank, nodes_count);
    // compare_vectors_detailed(ref_pagerank, jds_pagerank, nodes_count);
    
    // free data
    mtx_CSR_free(&mCSR);
    //mtx_ELL_free(&mELL);
    //mtx_JDS_free(&mJDS);

    return 0;
}
