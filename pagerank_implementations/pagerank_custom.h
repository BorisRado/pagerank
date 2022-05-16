#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <CL/cl.h>
#include "../helpers/helper.h"
#include "../helpers/ocl_helper.h"

/*
This script contains implementations that use the custom matrix representation
*/

#define TELEPORTATION_PROBABILITY 0.85

void init_pagerank(float ** pagerank_old, float ** pagerank_new, int nodes_count) {
    *pagerank_old = (float*) malloc(nodes_count * sizeof(float));
    *pagerank_new = (float*) malloc(nodes_count * sizeof(float));

    float init_value = 1 / (float)nodes_count;
    for (int i = 0; i < nodes_count; i++)
        (*pagerank_old)[i] = init_value;

}

float * pagerank_custom_out(int ** graph, int * out_degrees, int leaves_count, int * leaves, int nodes_count, double epsilon) {
    float *pagerank_old, *pagerank_new;
    init_pagerank(&pagerank_old, &pagerank_new, nodes_count);
    
    int i, j;

    int iterations = 0;

    do {

        float leaked_pagerank = 0.;
        for (i = 0; i < leaves_count; i++) {
            leaked_pagerank += pagerank_old[leaves[i]]; 
        }
        leaked_pagerank = leaked_pagerank + (1 - leaked_pagerank) * (1 - TELEPORTATION_PROBABILITY);

        for (i = 0; i < nodes_count; i++)
            pagerank_new[i] = leaked_pagerank / (float)nodes_count;

        for (i = 0; i < nodes_count; i++) {
            float pagerank_contribution =
                TELEPORTATION_PROBABILITY * pagerank_old[i] / (float)out_degrees[i];
            for (j = 0; j < out_degrees[i]; j++){
                pagerank_new[graph[i][j]] += pagerank_contribution;
            }

        }
        swap_pointers(&pagerank_old, &pagerank_new);
        iterations++;
        if (iterations > 100) break;

    } while (get_norm_difference(pagerank_old, pagerank_new, nodes_count) > epsilon);
    printf("Total pagerank iterations: %d\n", iterations);
    swap_pointers(&pagerank_old, &pagerank_new);
    return pagerank_new;
}

float * pagerank_custom_in(int ** graph, int * in_degrees, int * out_degrees,
                int leaves_count, int * leaves, int nodes_count, double epsilon, bool parallel_for) {
    float *pagerank_old, *pagerank_new;
    init_pagerank(&pagerank_old, &pagerank_new, nodes_count);

    int i, j;

    int iterations = 0;

    do {

        float leaked_pagerank = 0.;
        for (i = 0; i < leaves_count; i++) {
            leaked_pagerank += pagerank_old[leaves[i]]; 
        }
        leaked_pagerank = leaked_pagerank + (1 - leaked_pagerank) * (1 - TELEPORTATION_PROBABILITY);
        float init_pagerank = leaked_pagerank / (float)nodes_count;

        #pragma omp parallel for if(parallel_for) schedule(guided) private(i,j) shared(out_degrees,graph,init_pagerank,nodes_count)
        for (i = 0; i < nodes_count; i++) {
            float i_pr = init_pagerank;
            for (j = 0; j < in_degrees[i]; j++){
                i_pr += TELEPORTATION_PROBABILITY * pagerank_old[graph[i][j]] / out_degrees[graph[i][j]];
            }
            pagerank_new[i] = i_pr;
        }
        swap_pointers(&pagerank_old, &pagerank_new);
        iterations++;
        if (iterations > 200) break;

    } while (get_norm_difference(pagerank_old, pagerank_new, nodes_count) > epsilon);
    printf("Total pagerank iterations: %d\n", iterations);
    swap_pointers(&pagerank_old, &pagerank_new);
    return pagerank_new;
}

float * pagerank_custom_in_ocl(int ** graph, int * in_degrees, int * out_degrees,
                int leaves_count, int * leaves, int nodes_count, double epsilon) {
    // TO-DO: still need to test this function... the nodes with GPU are currently down...
    
    // c code that uses the kernels implemented in `pr_custom_matrix_in.cl`
    cl_command_queue command_queue;
    cl_context context;
    cl_program program;

    ocl_init("kernels/pr_custom_matrix_in.cl", &command_queue, &context, &program);

    float *pagerank_old, *pagerank_new;
    init_pagerank(&pagerank_old, &pagerank_new, nodes_count);

    // Divide work
    size_t local_item_size = 128;
	size_t num_groups = 1;
    size_t global_item_size = num_groups*local_item_size;

    // allocate memory on device and transfer data from host
    cl_mem leaves_count_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(int), leaves_count, &clStatus);
    cl_mem leaves_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                leaves_count * sizeof(int), leaves, &clStatus);
    cl_mem pagerank_old_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                nodes_count * sizeof(float), pagerank_old, &clStatus);
    cl_mem leaked_pr_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                sizeof(float), NULL, &clStatus);

    cl_kernel kernel = clCreateKernel(program, "compute_leaked_pagerank", &clStatus);
    clStatus  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&leaves_count_d);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&leaves_d);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&pagerank_old_d);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&leaked_pr_d);
	clStatus |= clSetKernelArg(kernel, 4, local_item_size * sizeof(float), NULL);	    // allocate local memory on device

    // Execute kernel
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                    &global_item_size, &local_item_size, 0, NULL, NULL);

    // Copy results back to host
	float p;
    clStatus = clEnqueueReadBuffer(command_queue, leaked_pr_d, CL_TRUE, 0,
                    sizeof(float), &p, 0, NULL, NULL);

    ocl_destroy(command_queue, context, program);
    printf("Test concluded succesfully\n");
}
