#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <CL/cl.h>
#include "../helpers/helper.h"
#include "../helpers/ocl_helper.h"

/*
This script contains implementations that use the custom matrix representation
*/

void update_sizes(size_t new_groups, size_t new_local_item_size,
         size_t * local_item_size, size_t * global_item_size, size_t * num_groups) {
    *local_item_size = new_local_item_size;
    *global_item_size = new_local_item_size * new_groups;
    *num_groups = new_groups;
}

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
        printf("leaked... %f\n", leaked_pagerank);
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
    // TO-DO: complete this function... right now it just test some kernels, still need
    // to implement the kernel that actually performs a step
    
    // c code that uses the kernels implemented in `pr_custom_matrix_in.cl`
    cl_command_queue command_queue;
    cl_context context;
    cl_program program;

    int status = ocl_init("kernels/pr_custom_matrix_in.cl", &command_queue, &context, &program);
    if (status != 0) {
        printf("Initialization failed. Exiting OCL computation...\n");
        exit(1);
    }

    float *pagerank_old, *pagerank_new;
    init_pagerank(&pagerank_old, &pagerank_new, nodes_count);

    // Divide work
    size_t local_item_size, num_groups, global_item_size;
    update_sizes(1, 128, &local_item_size, &global_item_size, &num_groups);

    // allocate memory on device and transfer data from host
    cl_int clStatus = 0;
    cl_mem leaves_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                leaves_count * sizeof(int), leaves, &clStatus);
    cl_mem pagerank_old_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                nodes_count * sizeof(float), pagerank_old, &clStatus);
    cl_mem leaked_pr_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                1 * sizeof(float), NULL, &clStatus);
    cl_kernel kernel = clCreateKernel(program, "compute_leaked_pagerank", &clStatus);
    clStatus  = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&leaves_count);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&leaves_d);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&pagerank_old_d);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&leaked_pr_d);
	clStatus |= clSetKernelArg(kernel, 4, local_item_size * sizeof(float), NULL);	    // allocate local memory on device
    if (clStatus != CL_SUCCESS) {
        printf("Somethin went wrong while submitting arguments to the kernel (status %d). Exiting...\n", clStatus);
        exit(1);
    }

    // Execute kernel
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                    &global_item_size, &local_item_size, 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("Somethin went wrong while executing kernel (status %d). Exiting...\n", clStatus);
        exit(1);
    }

	float p;
    clStatus = clEnqueueReadBuffer(command_queue, leaked_pr_d, CL_TRUE, 0,
                    1 * sizeof(float), &p, 0, NULL, NULL);
    if (clStatus != 0) {
        printf("Something went wrong while reading value (%d). Exiting...", clStatus);
        exit(1);
    }
    printf("Leaked pagerank %f\n", p);

    // try to compute the norm
    for (int i = 0; i < nodes_count; i++)
        pagerank_new[i] = 0.0;
    
    update_sizes(2, 32, &local_item_size, &global_item_size, &num_groups);
    
    cl_mem pagerank_new_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                nodes_count * sizeof(float), pagerank_new, &clStatus);
    cl_mem wg_diffs_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                num_groups * sizeof(float), NULL, &clStatus);
    
    kernel = clCreateKernel(program, "compute_norm_difference_wg", &clStatus);
    clStatus  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&pagerank_old_d);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&pagerank_new_d);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&wg_diffs_d);
    clStatus |= clSetKernelArg(kernel, 3, num_groups * sizeof(float), NULL);
	clStatus |= clSetKernelArg(kernel, 4, sizeof(cl_int), &nodes_count);
    if (clStatus != CL_SUCCESS) {
        printf("Somethin went wrong while submitting arguments to the kernel (status %d). Exiting...\n", clStatus);
        exit(1);
    }
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                    &global_item_size, &local_item_size, 0, NULL, NULL);

    // sum up the partial results from the previous step
    cl_mem norm_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                sizeof(float), NULL, &clStatus);
    update_sizes(1, 32, &local_item_size, &global_item_size, &num_groups);
    kernel = clCreateKernel(program, "compute_norm_difference_fin", &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("Something went wrong while creating kernel %d. Exiting... \n", clStatus);
        exit(1);
    }
    clStatus  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&wg_diffs_d);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&norm_d);
    clStatus |= clSetKernelArg(kernel, 2, local_item_size * sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_int), &global_item_size);
    if (clStatus != CL_SUCCESS) {
        printf("Something went wrong while setting argument for the kernel %d. Exiting... \n", clStatus);
        exit(1);
    }
    
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                    &global_item_size, &local_item_size, 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("Something went wrong executing the kernel %d. Exiting... \n", clStatus);
        exit(1);
    }
    
    
    clStatus = clEnqueueReadBuffer(command_queue, norm_d, CL_TRUE, 0,
                    1 * sizeof(float), &p, 0, NULL, NULL);
    if (clStatus != 0) {
        printf("Something went wrong while reading value (%d). Exiting...", clStatus);
        exit(1);
    }
    printf("Final norm %f\n", p);
    ocl_destroy(command_queue, context, program);
    printf("Test concluded succesfully\n");
}
