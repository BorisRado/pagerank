#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <CL/cl.h>
#include <omp.h>
#include "../helpers/helper.h"
#include "../helpers/ocl_helper.h"
#include "../global_config.h"

/*
This script contains OpenCL and OpenMP implementations that use the custom matrix
representation. The MPI implementations are in `pagerank_custom_mpi.h`
*/

void update_sizes(size_t new_groups, size_t new_local_item_size,
         size_t * local_item_size, size_t * global_item_size, size_t * num_groups) {
    *local_item_size = new_local_item_size;
    *global_item_size = new_local_item_size * new_groups;
    *num_groups = new_groups;
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
        leaked_pagerank = leaked_pagerank + (1 - leaked_pagerank) * (1 - DAMPENING);

        for (i = 0; i < nodes_count; i++)
            pagerank_new[i] = leaked_pagerank / (float)nodes_count;

        for (i = 0; i < nodes_count; i++) {
            float pagerank_contribution =
                DAMPENING * pagerank_old[i] / (float)out_degrees[i];
            for (j = 0; j < out_degrees[i]; j++){
                pagerank_new[graph[i][j]] += pagerank_contribution;
            }

        }
        swap_pointers(&pagerank_old, &pagerank_new);
        iterations++;
        if (iterations > 100) break;

    } while (get_norm_difference(pagerank_old, pagerank_new, nodes_count, true) > epsilon);
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
        leaked_pagerank = leaked_pagerank + (1 - leaked_pagerank) * (1 - DAMPENING);
        // printf("leaked... %f\n", leaked_pagerank);
        float init_pagerank = leaked_pagerank / (float)nodes_count;

        #pragma omp parallel for if(parallel_for) schedule(guided) private(i,j) shared(out_degrees,graph,init_pagerank,nodes_count)
        for (i = 0; i < nodes_count; i++) {
            float i_pr = init_pagerank;
            for (j = 0; j < in_degrees[i]; j++){
                i_pr += DAMPENING * pagerank_old[graph[i][j]] / out_degrees[graph[i][j]];
            }
            pagerank_new[i] = i_pr;
        }

        swap_pointers(&pagerank_old, &pagerank_new);
        iterations++;
        if (iterations > MAX_ITER) break;
    } while (!(CHECK_CONVERGENCE && get_norm_difference(pagerank_old, pagerank_new, nodes_count, parallel_for) <= epsilon));
    printf("Total pagerank iterations: %d\n", iterations);
    swap_pointers(&pagerank_old, &pagerank_new);
    return pagerank_new;
}

float * pagerank_custom_in_ocl(int ** graph, int * in_degrees, int * out_degrees,
                int leaves_count, int * leaves, int nodes_count, int edges_count,
                double epsilon, double * start_global, double * end_global, char * pr_step_kernel) {
    // this function leverages the kernels implemented in `pr_custom_matrix_in.cl`
    bool expand_out_degrees = strstr(pr_step_kernel, "expand") != NULL;
    cl_command_queue command_queue;
    cl_context context;
    cl_program program;
    cl_event event;

    int status = ocl_init("kernels/pr_custom_matrix_in.cl", &command_queue, &context, &program);
    if (status != 0) {
        printf("Initialization failed. Exiting OCL computation...\n");
        exit(1);
    }

    double start, end;
    float *pagerank_old, *pagerank_new;
    init_pagerank(&pagerank_old, &pagerank_new, nodes_count);
    size_t local_item_size, num_groups, global_item_size;
    int threads_per_row = 8;

    // compile kernels
    cl_int clStatus = 0;
    cl_kernel kernel_leaked_pr = clCreateKernel(program, "compute_leaked_pagerank", &clStatus);
    cl_kernel kernel_pagerank_step = clCreateKernel(program, pr_step_kernel, &clStatus);
    cl_kernel kernel_norm_wg = clCreateKernel(program, "compute_norm_difference_wg", &clStatus);
    cl_kernel kernel_norm_fin = clCreateKernel(program, "compute_norm_difference_fin", &clStatus);

    *start_global = omp_get_wtime();
    int * CDF = (int *) malloc(nodes_count * sizeof(int));
    start = omp_get_wtime();
    CDF[0] = 0;
    for (int i = 1; i < nodes_count; i++) {
        CDF[i] = CDF[i-1] + in_degrees[i-1];
    }
    printf("%s - CDF time: %.5f\n", pr_step_kernel, omp_get_wtime() - start);

    // define parameters for executing the kernels (WI, WG)
    int kernel_leaked_pr_wi = 1024, kernel_leaked_pr_wg = 1,
            kernel_norm_wg_wi = 1024, kernel_norm_wg_wg = 32,
            kernel_norm_fin_wi = 32, kernel_norm_fin_wg = 1,
            kernel_pagerank_step_wi = 1024, kernel_pagerank_step_wg = 64;

    // transfer all the required data to the GPU
    start = omp_get_wtime();
    cl_mem graph_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                edges_count * sizeof(int), graph[0], &clStatus);
    cl_mem in_degrees_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                nodes_count * sizeof(int), in_degrees, &clStatus);
    cl_mem out_degrees_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                nodes_count * sizeof(int), out_degrees, &clStatus);
    cl_mem leaves_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                leaves_count * sizeof(int), leaves, &clStatus);
    cl_mem leaves_count_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(int), &leaves_count, &clStatus);
    cl_mem nodes_count_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(int), &nodes_count, &clStatus);
    cl_mem edges_count_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(int), &edges_count, &clStatus);
    cl_mem pagerank_old_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                nodes_count * sizeof(float), pagerank_old, &clStatus);
    cl_mem in_deg_CDF_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                nodes_count * sizeof(int), CDF, &clStatus);
    end = omp_get_wtime();
    printf("%s - Data transfer to GPU time: %.4f\n", pr_step_kernel, end - start);

    if (expand_out_degrees) {
        // expand out_degrees
        update_sizes(16, 256, &local_item_size, &global_item_size, &num_groups);
        cl_kernel kernel_expand_out_deg = clCreateKernel(program, "expand_out_degrees", &clStatus);
        cl_mem expanded_out_degrees_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    edges_count * sizeof(int), NULL, &clStatus);
        clStatus  = clSetKernelArg(kernel_expand_out_deg, 0, sizeof(cl_mem), (void *)&edges_count_d);
        clStatus |= clSetKernelArg(kernel_expand_out_deg, 1, sizeof(cl_mem), (void *)&out_degrees_d);
        clStatus |= clSetKernelArg(kernel_expand_out_deg, 2, sizeof(cl_mem), (void *)&graph_d);
        clStatus |= clSetKernelArg(kernel_expand_out_deg, 3, sizeof(cl_mem), (void *)&expanded_out_degrees_d);
        clStatus = clEnqueueNDRangeKernel(command_queue, kernel_expand_out_deg, 1, NULL,
                            &global_item_size, &local_item_size, 0, NULL, &event);
        float expand_time = print_ocl_time(event, command_queue, "Expand out degrees");
        printf("%s - Expanding out degrees time: %f\n", pr_step_kernel, expand_time);
        out_degrees_d = expanded_out_degrees_d;
    }
    
    // allocate additional data we will need
    cl_mem leaked_pr_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float), NULL, &clStatus);
    cl_mem pagerank_new_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                nodes_count * sizeof(float), NULL, &clStatus);
    cl_mem wg_diffs_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                kernel_norm_wg_wg * sizeof(float), NULL, &clStatus);
    cl_mem norm_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                sizeof(float), NULL, &clStatus);

    // set constant arguments to kernels (cannot fix pageranks because pointers change during iterations, nor local mem)
    clStatus  = clSetKernelArg(kernel_leaked_pr, 0, sizeof(cl_mem), (void *)&leaves_count_d);
    clStatus |= clSetKernelArg(kernel_leaked_pr, 1, sizeof(cl_mem), (void *)&leaves_d);

    clStatus  = clSetKernelArg(kernel_pagerank_step, 0, sizeof(cl_mem), (void *)&graph_d);
    clStatus |= clSetKernelArg(kernel_pagerank_step, 1, sizeof(cl_mem), (void *)&in_deg_CDF_d);
    clStatus |= clSetKernelArg(kernel_pagerank_step, 2, sizeof(cl_mem), (void *)&in_degrees_d);
    clStatus |= clSetKernelArg(kernel_pagerank_step, 3, sizeof(cl_mem), (void *)&out_degrees_d);
    clStatus |= clSetKernelArg(kernel_pagerank_step, 7, sizeof(cl_mem), (void *)&nodes_count_d);
    clStatus |= clSetKernelArg(kernel_pagerank_step, 8, sizeof(cl_int), (void *)&threads_per_row);

    // iterate the pagerank step
    float times_leaked_pr_kernel = 0.,
            times_pagerank_step_kernel = 0.,
            times_norm_wg_kernel = 0.,
            times_norm_fin_kernel = 0.;

    int iterations = 0;
    float norm;
    start = omp_get_wtime();
    do {
        // compute the pagerank that would be leaked in the next iteration
        update_sizes(kernel_leaked_pr_wg, kernel_leaked_pr_wi, &local_item_size, &global_item_size, &num_groups);
        clStatus |= clSetKernelArg(kernel_leaked_pr, 2, sizeof(cl_mem), (void *)&pagerank_old_d);
        clStatus |= clSetKernelArg(kernel_leaked_pr, 3, sizeof(cl_mem), (void *)&leaked_pr_d);
        clStatus |= clSetKernelArg(kernel_leaked_pr, 4, local_item_size * sizeof(float), NULL);	// allocate local memory on device
        // check_status(clStatus, "submitting args to kernel");
        clStatus = clEnqueueNDRangeKernel(command_queue, kernel_leaked_pr, 1, NULL,
                        &global_item_size, &local_item_size, 0, NULL, &event);
        // check_status(clStatus, "executing kernel");
        times_leaked_pr_kernel += print_ocl_time(event, command_queue, "leaked pagerank kernel");

        // pagerank step
        update_sizes(kernel_pagerank_step_wg, kernel_pagerank_step_wi, &local_item_size, &global_item_size, &num_groups);
        clStatus |= clSetKernelArg(kernel_pagerank_step, 4, sizeof(cl_mem), (void *)&pagerank_old_d);
        clStatus |= clSetKernelArg(kernel_pagerank_step, 5, sizeof(cl_mem), (void *)&pagerank_new_d);
        clStatus |= clSetKernelArg(kernel_pagerank_step, 6, sizeof(cl_mem), (void *)&leaked_pr_d);
        clStatus |= clSetKernelArg(kernel_pagerank_step, 9, local_item_size * sizeof(double), NULL);
        clStatus = clEnqueueNDRangeKernel(command_queue, kernel_pagerank_step, 1, NULL,
                        &global_item_size, &local_item_size, 0, NULL, &event);
        // check_status(clStatus, "executing kernel");
        times_pagerank_step_kernel += print_ocl_time(event, command_queue, "pagerank step kernel");

        // compute the norm - step 1
        update_sizes(kernel_norm_wg_wg, kernel_norm_wg_wi, &local_item_size, &global_item_size, &num_groups);
        clStatus  = clSetKernelArg(kernel_norm_wg, 0, sizeof(cl_mem), (void *)&pagerank_old_d);
        clStatus |= clSetKernelArg(kernel_norm_wg, 1, sizeof(cl_mem), (void *)&pagerank_new_d);
        clStatus |= clSetKernelArg(kernel_norm_wg, 2, sizeof(cl_mem), (void *)&wg_diffs_d);
        clStatus |= clSetKernelArg(kernel_norm_wg, 3, local_item_size * sizeof(float), NULL);
        clStatus |= clSetKernelArg(kernel_norm_wg, 4, sizeof(cl_mem), (void *)&nodes_count_d);
        // check_status(clStatus, "submitting args to kernel");
        clStatus = clEnqueueNDRangeKernel(command_queue, kernel_norm_wg, 1, NULL,
                        &global_item_size, &local_item_size, 0, NULL, &event);
        // check_status(clStatus, "executing kernel");
        times_norm_wg_kernel += print_ocl_time(event, command_queue, "norm step 1 kernel");

        // compute the norm - step 2
        update_sizes(kernel_norm_fin_wg, kernel_norm_fin_wi, &local_item_size, &global_item_size, &num_groups);
        clStatus  = clSetKernelArg(kernel_norm_fin, 0, sizeof(cl_mem), (void *)&wg_diffs_d);
        clStatus |= clSetKernelArg(kernel_norm_fin, 1, sizeof(cl_mem), (void *)&norm_d);
        clStatus |= clSetKernelArg(kernel_norm_fin, 2, local_item_size * sizeof(float), NULL);
        clStatus |= clSetKernelArg(kernel_norm_fin, 3, sizeof(cl_int), &global_item_size);
        // check_status(clStatus, "submitting args to kernel");
        clStatus = clEnqueueNDRangeKernel(command_queue, kernel_norm_fin, 1, NULL,
                        &global_item_size, &local_item_size, 0, NULL, &event);
        // check_status(clStatus, "executing kernel");
        times_norm_fin_kernel += print_ocl_time(event, command_queue, "norm final kernel");

        // read final norm value back to host
        clStatus = clEnqueueReadBuffer(command_queue, norm_d, CL_TRUE, 0,
                        1 * sizeof(float), &norm, 0, NULL, NULL);
        // check_status(clStatus, "reading value");

        iterations++;
        ocl_swap_pointers(&pagerank_new_d, &pagerank_old_d);
        if (iterations > MAX_ITER) break;
    } while (!(CHECK_CONVERGENCE && sqrt(norm) <= epsilon));
    end = omp_get_wtime();
    printf("Total number of iterations: %d\n", iterations);

    // read the final data to CPU - read pagerank_old_d because we just swapped new and old,
    // so the new values are actually in the _old vector
    clEnqueueReadBuffer(command_queue, pagerank_old_d, CL_TRUE, 0, 
                        nodes_count * sizeof(float), pagerank_new, 0, NULL, NULL);
    *end_global = omp_get_wtime();

    // print average times of the kernels
    printf("%s - Average time `Leaked pagerank kernel`: %.4f\n", pr_step_kernel, times_leaked_pr_kernel / iterations);
    printf("%s - Average time `Pagerank step kernel`: %.4f\n", pr_step_kernel, times_pagerank_step_kernel / iterations);
    printf("%s - Average time `Norm work group`: %.4f\n", pr_step_kernel, times_norm_wg_kernel / iterations);
    printf("%s - Average time `Norm final`: %.4f\n", pr_step_kernel, times_norm_fin_kernel / iterations);
    printf("%s - Average time per iteration: %.4f\n", pr_step_kernel, (end - start) / iterations);
    ocl_destroy(command_queue, context, program);
    ocl_release(13, graph_d,
            in_degrees_d,
            out_degrees_d,
            leaves_d,
            leaves_count_d,
            nodes_count_d,
            edges_count_d,
            pagerank_old_d,
            in_deg_CDF_d,
            leaked_pr_d,
            pagerank_new_d,
            wg_diffs_d,
            norm_d);
    return pagerank_new;
}