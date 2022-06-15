#include <omp.h>
#include <CL/cl.h>
#include "../global_config.h"


float * pagerank_CSR_vector(mtx_CSR mCSR, float * start, float * end) {
    // this function leverages the kernels implemented in `pr_custom_matrix_in.cl`
    cl_command_queue command_queue;
    cl_context context;
    cl_program program;
    cl_event event;

    int clStatus = ocl_init("kernels/CSR_vector.cl", &command_queue, &context, &program);
    if (clStatus != 0) {
        printf("Initialization failed. Exiting OCL computation.\n");
        exit(1);
    }

    *start = omp_get_wtime();
    float * pagerank_in  = (float*) malloc(mCSR.num_cols * sizeof(float));
    float * pagerank_out = (float*) malloc(mCSR.num_cols * sizeof(float));
    for (int i = 0; i < mCSR.num_cols; i++)
        pagerank_in[i] = 1. / mCSR.num_cols;

    // allocate memory on device and transfer data from host CSR
    cl_mem mCSRrowptr_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                   (mCSR.num_rows + 1) * sizeof(cl_int), NULL, &clStatus);
    cl_mem mCSRcol_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                    mCSR.num_nonzeros * sizeof(cl_int), NULL, &clStatus);
    cl_mem mCSRdata_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                    mCSR.num_nonzeros * sizeof(cl_float), NULL, &clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, mCSRrowptr_d, CL_TRUE, 0,						
                                   (mCSR.num_rows + 1) * sizeof(cl_int), mCSR.rowptr, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mCSRcol_d, CL_TRUE, 0,						
                                    mCSR.num_nonzeros * sizeof(cl_int), mCSR.col, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mCSRdata_d, CL_TRUE, 0,						
                                    mCSR.num_nonzeros * sizeof(cl_float), mCSR.data, 0, NULL, NULL);				

    // vectors
    cl_mem vecIn_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mCSR.num_cols * sizeof(cl_float), NULL, &clStatus);
    cl_mem vecOut_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                    mCSR.num_rows * sizeof(cl_float), NULL, &clStatus);

    // create kernel CSR and set arguments
    cl_kernel kernelCSR_multh = clCreateKernel(program, "mCSRmulth", &clStatus);
    clStatus  = clSetKernelArg(kernelCSR_multh, 0, sizeof(cl_mem), (void *)&mCSRrowptr_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 1, sizeof(cl_mem), (void *)&mCSRcol_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 2, sizeof(cl_mem), (void *)&mCSRdata_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 3, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 4, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |=	clSetKernelArg(kernelCSR_multh, 5, WORKGROUP_SIZE*sizeof(cl_float), NULL);
	clStatus |= clSetKernelArg(kernelCSR_multh, 6, sizeof(cl_int), (void *)&(mCSR.num_rows));

    size_t local_item_size = WORKGROUP_SIZE;

    // Divide work
	int num_groups = (WARP_SIZE * mCSR.num_rows - 1) / local_item_size + 1;
    size_t global_item_size_CSRpar = num_groups * local_item_size;

    // CSR multh write, execute, read
    int iterations = 0;
    float norm;
    double dtimeCSR_multh = omp_get_wtime();

    while (1) {
        // Schedule iteration on GPU
        clStatus |= clEnqueueWriteBuffer(command_queue, vecIn_d, CL_TRUE, 0,						
                                        mCSR.num_cols*sizeof(cl_float), pagerank_in, 0, NULL, NULL);				
        clStatus |= clEnqueueNDRangeKernel(command_queue, kernelCSR_multh, 1, NULL,						
                                        &global_item_size_CSRpar, &local_item_size, 0, NULL, NULL);	
        clStatus |= clEnqueueReadBuffer(command_queue, vecOut_d, CL_TRUE, 0,						
                                        mCSR.num_rows*sizeof(cl_float), pagerank_out, 0, NULL, NULL);
        
        // Update data
        float * tmp = pagerank_in;
        pagerank_in = pagerank_out;
        pagerank_out = tmp;
        iterations++;

        // Fix output vector
        for(int i = 0; i < mCSR.num_cols; i++)
            pagerank_in[i] = pagerank_in[i]*DAMPENING + (1. - DAMPENING)/mCSR.num_cols;

        // Check exit criteria
        if(CHECK_CONVERGENCE && get_norm_difference(pagerank_in, pagerank_out, mCSR.num_cols, false) <= EPSILON)
            break;
        
        if(MAX_ITER > 0 && iterations >= MAX_ITER)
            break;
    }

    dtimeCSR_multh = omp_get_wtime() - dtimeCSR_multh;

    // Normalize output
    double sum = 0.;
    for(int i = 0; i < mCSR.num_cols; i++)
        sum += pagerank_in[i];
    for(int i = 0; i < mCSR.num_cols; i++)
        pagerank_in[i] /= sum;
    *end = omp_get_wtime();

    // Report results
    printf("Total number of iterations: %d\n", iterations);
    printf("CSR - Average time per iteration: %.4f\n", dtimeCSR_multh / iterations);

    // Free memory structures
    clStatus = clReleaseKernel(kernelCSR_multh);
    clStatus = clReleaseMemObject(mCSRrowptr_d);
    clStatus = clReleaseMemObject(mCSRcol_d);
    clStatus = clReleaseMemObject(mCSRdata_d);
    clStatus = clReleaseMemObject(vecIn_d);
    clStatus = clReleaseMemObject(vecOut_d);
    
    ocl_destroy(command_queue, context, program);
    free(pagerank_out);
    return pagerank_in;
}
