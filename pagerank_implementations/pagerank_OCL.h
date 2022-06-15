#include <omp.h>
#include <CL/cl.h>
#include "../global_config.h"
#include "../readers/mtx_sparse.h"
#include "../helpers/ocl_helper.h"
#include "../helpers/helper.h"


float * pagerank_CSR_vector(mtx_CSR mCSR, float * start, float * end) {
    cl_command_queue command_queue;
    cl_context context;
    cl_program program;
    cl_event event;

    int clStatus = ocl_init("kernels/sparse_matrix.cl", &command_queue, &context, &program);
    if (clStatus != 0) {
        printf("Initialization failed. Exiting OCL computation.\n");
        exit(1);
    }

    *start = omp_get_wtime();

    /*
     * DATA ALLOCATION
     */

    // allocate pagerank vectors and compute initial values
    float * pagerank_in  = (float*) malloc(mCSR.num_cols * sizeof(float));
    float * pagerank_out = (float*) malloc(mCSR.num_cols * sizeof(float));
    for (int i = 0; i < mCSR.num_cols; i++)
        pagerank_in[i] = 1. / mCSR.num_cols;
    
    cl_mem vecIn_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mCSR.num_cols * sizeof(cl_float), NULL, &clStatus);
    cl_mem vecOut_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                    mCSR.num_cols * sizeof(cl_float), NULL, &clStatus);

    // allocate float for norm
    cl_mem norm_diff_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &clStatus);
    float zero = 0;
    float norm;

    // allocate CSR memory on device and transfer data from host CSR
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

    /*
     * CREATE KERNELS
     */

    cl_kernel fixPROutput = clCreateKernel(program, "fixPROutput", &clStatus);
    clStatus |= clSetKernelArg(fixPROutput, 0, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |= clSetKernelArg(fixPROutput, 1, sizeof(cl_int), (void *)&(mCSR.num_cols));

    cl_kernel normDiff = clCreateKernel(program, "normDiff", &clStatus);
    clStatus |= clSetKernelArg(normDiff, 0, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(normDiff, 1, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |= clSetKernelArg(normDiff, 2, sizeof(cl_int), (void *)&(mCSR.num_cols));
    clStatus |= clSetKernelArg(normDiff, 3, WORKGROUP_SIZE*sizeof(cl_float), NULL);
    clStatus |= clSetKernelArg(normDiff, 4, sizeof(cl_mem), (void *)&norm_diff_d);

    // create CSR kernel and set arguments
    cl_kernel kernelCSR_multh = clCreateKernel(program, "mCSRmulth", &clStatus);
    clStatus |= clSetKernelArg(kernelCSR_multh, 0, sizeof(cl_mem), (void *)&mCSRrowptr_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 1, sizeof(cl_mem), (void *)&mCSRcol_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 2, sizeof(cl_mem), (void *)&mCSRdata_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 3, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(kernelCSR_multh, 4, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |=	clSetKernelArg(kernelCSR_multh, 5, WORKGROUP_SIZE*sizeof(cl_float), NULL);
	clStatus |= clSetKernelArg(kernelCSR_multh, 6, sizeof(cl_int), (void *)&(mCSR.num_rows));


    /*
     * LAUNCH COMPUTATION
     */

    size_t local_item_size = WORKGROUP_SIZE;

    // Divide work
	int num_groups = (mCSR.num_rows - 1) / local_item_size + 1;
    size_t global_item_size_helpers = num_groups * local_item_size;

	num_groups = (WARP_SIZE * mCSR.num_rows - 1) / local_item_size + 1;
    size_t global_item_size_CSRpar = num_groups * local_item_size;

    // CSR multh write, execute, read
    int iterations = 0;
    double dtimeCSR_multh = omp_get_wtime();

    while (1) {
        iterations++;

        // Write pr_in
        clStatus |= clEnqueueWriteBuffer(command_queue, vecIn_d, CL_TRUE, 0,						
                                        mCSR.num_cols*sizeof(cl_float), pagerank_in, 0, NULL, NULL);	

        clStatus |= clEnqueueNDRangeKernel(command_queue, kernelCSR_multh, 1, NULL,						
                                        &global_item_size_CSRpar, &local_item_size, 0, NULL, NULL);
        clStatus |= clEnqueueNDRangeKernel(command_queue, fixPROutput, 1, NULL,						
                                        &global_item_size_helpers, &local_item_size, 0, NULL, NULL);
        
        // Transfer pr_out
        clStatus |= clEnqueueReadBuffer(command_queue, vecOut_d, CL_TRUE, 0,						
                                        mCSR.num_rows*sizeof(cl_float), pagerank_out, 0, NULL, NULL);

        // Check exit criteria
        if(MAX_ITER > 0 && iterations >= MAX_ITER)
            break;

        if(CHECK_CONVERGENCE) {
            clStatus |= clEnqueueWriteBuffer(command_queue, norm_diff_d, CL_TRUE, 0,						
                                        sizeof(cl_float), &zero, 0, NULL, NULL);
            clStatus |= clEnqueueNDRangeKernel(command_queue, normDiff, 1, NULL,						
                                        &global_item_size_helpers, &local_item_size, 0, NULL, NULL);
            clStatus |= clEnqueueReadBuffer(command_queue, norm_diff_d, CL_TRUE, 0,						
                                        sizeof(cl_float), &norm, 0, NULL, NULL);
            norm = sqrt(norm);
            if(norm <= EPSILON)
                break;
        }

        // Update data
        float * tmp = pagerank_in;
        pagerank_in = pagerank_out;
        pagerank_out = tmp;
    }

    dtimeCSR_multh = omp_get_wtime() - dtimeCSR_multh;

    // Normalize output
    double sum = 0.;
    for(int i = 0; i < mCSR.num_cols; i++)
        sum += pagerank_out[i];
    for(int i = 0; i < mCSR.num_cols; i++)
        pagerank_out[i] /= sum;
    
    *end = omp_get_wtime();

    // Report results
    printf("Total number of iterations: %d\n", iterations);
    printf("CSR - Average time per iteration: %.4f\n", dtimeCSR_multh / iterations);

    // Free memory structures
    clStatus = clReleaseKernel(fixPROutput);
    clStatus = clReleaseKernel(normDiff);
    clStatus = clReleaseKernel(kernelCSR_multh);

    clStatus = clReleaseMemObject(vecIn_d);
    clStatus = clReleaseMemObject(vecOut_d);
    clStatus = clReleaseMemObject(mCSRrowptr_d);
    clStatus = clReleaseMemObject(mCSRcol_d);
    clStatus = clReleaseMemObject(mCSRdata_d);
    clStatus = clReleaseMemObject(norm_diff_d);
    
    ocl_destroy(command_queue, context, program);
    free(pagerank_in);
    return pagerank_out;
}


float * pagerank_CSR_scalar(mtx_CSR mCSR, float * start, float * end) {
    cl_command_queue command_queue;
    cl_context context;
    cl_program program;
    cl_event event;

    int clStatus = ocl_init("kernels/sparse_matrix.cl", &command_queue, &context, &program);
    if (clStatus != 0) {
        printf("Initialization failed. Exiting OCL computation.\n");
        exit(1);
    }

    *start = omp_get_wtime();

    /*
     * DATA ALLOCATION
     */

    // allocate pagerank vectors and compute initial values
    float * pagerank_in  = (float*) malloc(mCSR.num_cols * sizeof(float));
    float * pagerank_out = (float*) malloc(mCSR.num_cols * sizeof(float));
    for (int i = 0; i < mCSR.num_cols; i++)
        pagerank_in[i] = 1. / mCSR.num_cols;
    
    cl_mem vecIn_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mCSR.num_cols * sizeof(cl_float), NULL, &clStatus);
    cl_mem vecOut_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                    mCSR.num_cols * sizeof(cl_float), NULL, &clStatus);
    
     // allocate float for norm
    cl_mem norm_diff_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &clStatus);
    float zero = 0;
    float norm;

    // allocate CSR memory on device and transfer data from host CSR
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

    /*
     * CREATE KERNELS
     */

    cl_kernel fixPROutput = clCreateKernel(program, "fixPROutput", &clStatus);
    clStatus |= clSetKernelArg(fixPROutput, 0, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |= clSetKernelArg(fixPROutput, 1, sizeof(cl_int), (void *)&(mCSR.num_cols));

    cl_kernel normDiff = clCreateKernel(program, "normDiff", &clStatus);
    clStatus |= clSetKernelArg(normDiff, 0, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(normDiff, 1, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |= clSetKernelArg(normDiff, 2, sizeof(cl_int), (void *)&(mCSR.num_cols));
    clStatus |= clSetKernelArg(normDiff, 3, WORKGROUP_SIZE*sizeof(cl_float), NULL);
    clStatus |= clSetKernelArg(normDiff, 4, sizeof(cl_mem), (void *)&norm_diff_d);

    // create CSR kernel and set arguments
    cl_kernel kernelCSR_basic = clCreateKernel(program, "mCSRbasic", &clStatus);
    clStatus |= clSetKernelArg(kernelCSR_basic, 0, sizeof(cl_mem), (void *)&mCSRrowptr_d);
    clStatus |= clSetKernelArg(kernelCSR_basic, 1, sizeof(cl_mem), (void *)&mCSRcol_d);
    clStatus |= clSetKernelArg(kernelCSR_basic, 2, sizeof(cl_mem), (void *)&mCSRdata_d);
    clStatus |= clSetKernelArg(kernelCSR_basic, 3, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(kernelCSR_basic, 4, sizeof(cl_mem), (void *)&vecOut_d);
	clStatus |= clSetKernelArg(kernelCSR_basic, 5, sizeof(cl_int), (void *)&(mCSR.num_rows));

    /*
     * LAUNCH COMPUTATION
     */

    size_t local_item_size = WORKGROUP_SIZE;

    // Divide work
	int num_groups = (mCSR.num_rows - 1) / local_item_size + 1;
    size_t global_item_size_helpers = num_groups * local_item_size;

    size_t global_item_size_CSR = num_groups * local_item_size;

    // CSR write, execute, read
    int iterations = 0;
    double dtimeCSR_basic = omp_get_wtime();

    while (1) {
        iterations++;

        // Write pr_in
        clStatus |= clEnqueueWriteBuffer(command_queue, vecIn_d, CL_TRUE, 0,						
                                        mCSR.num_cols*sizeof(cl_float), pagerank_in, 0, NULL, NULL);	

        clStatus |= clEnqueueNDRangeKernel(command_queue, kernelCSR_basic, 1, NULL,						
                                        &global_item_size_CSR, &local_item_size, 0, NULL, NULL);
        clStatus |= clEnqueueNDRangeKernel(command_queue, fixPROutput, 1, NULL,						
                                        &global_item_size_helpers, &local_item_size, 0, NULL, NULL);
        
        // Transfer pr_out
        clStatus |= clEnqueueReadBuffer(command_queue, vecOut_d, CL_TRUE, 0,						
                                        mCSR.num_rows*sizeof(cl_float), pagerank_out, 0, NULL, NULL);

        // Check exit criteria
        if(MAX_ITER > 0 && iterations >= MAX_ITER)
            break;

        if(CHECK_CONVERGENCE) {
            clStatus |= clEnqueueWriteBuffer(command_queue, norm_diff_d, CL_TRUE, 0,						
                                        sizeof(cl_float), &zero, 0, NULL, NULL);
            clStatus |= clEnqueueNDRangeKernel(command_queue, normDiff, 1, NULL,						
                                        &global_item_size_helpers, &local_item_size, 0, NULL, NULL);
            clStatus |= clEnqueueReadBuffer(command_queue, norm_diff_d, CL_TRUE, 0,						
                                        sizeof(cl_float), &norm, 0, NULL, NULL);
            norm = sqrt(norm);
            if(norm <= EPSILON)
                break;
        }

        // Update data
        float * tmp = pagerank_in;
        pagerank_in = pagerank_out;
        pagerank_out = tmp;
    }

    dtimeCSR_basic = omp_get_wtime() - dtimeCSR_basic;

    // Normalize output
    double sum = 0.;
    for(int i = 0; i < mCSR.num_cols; i++)
        sum += pagerank_out[i];
    for(int i = 0; i < mCSR.num_cols; i++)
        pagerank_out[i] /= sum;
    
    *end = omp_get_wtime();

    // Report results
    printf("Total number of iterations: %d\n", iterations);
    printf("CSR - Average time per iteration: %.4f\n", dtimeCSR_basic / iterations);

    // Free memory structures
    clStatus = clReleaseKernel(fixPROutput);
    clStatus = clReleaseKernel(normDiff);
    clStatus = clReleaseKernel(kernelCSR_basic);

    clStatus = clReleaseMemObject(vecIn_d);
    clStatus = clReleaseMemObject(vecOut_d);
    clStatus = clReleaseMemObject(mCSRrowptr_d);
    clStatus = clReleaseMemObject(mCSRcol_d);
    clStatus = clReleaseMemObject(mCSRdata_d);
    clStatus = clReleaseMemObject(norm_diff_d);
    
    ocl_destroy(command_queue, context, program);
    free(pagerank_in);
    return pagerank_out;
}


float * pagerank_ELL(mtx_ELL mELL, float * start, float * end) {
    cl_command_queue command_queue;
    cl_context context;
    cl_program program;
    cl_event event;

    int clStatus = ocl_init("kernels/sparse_matrix.cl", &command_queue, &context, &program);
    if (clStatus != 0) {
        printf("Initialization failed. Exiting OCL computation.\n");
        exit(1);
    }

    *start = omp_get_wtime();

    /*
     * DATA ALLOCATION
     */

    // allocate pagerank vectors and compute initial values
    float * pagerank_in  = (float*) malloc(mELL.num_cols * sizeof(float));
    float * pagerank_out = (float*) malloc(mELL.num_cols * sizeof(float));
    for (int i = 0; i < mELL.num_cols; i++)
        pagerank_in[i] = 1. / mELL.num_cols;
    
    cl_mem vecIn_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mELL.num_cols * sizeof(cl_float), NULL, &clStatus);
    cl_mem vecOut_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                    mELL.num_cols * sizeof(cl_float), NULL, &clStatus);

    // allocate float for norm
    cl_mem norm_diff_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &clStatus);
    float zero = 0;
    float norm;

    // allocate memory on device and transfer data from host ELL
    cl_mem mELLcol_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                    mELL.num_elements * sizeof(cl_int), NULL, &clStatus);
    cl_mem mELLdata_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                    mELL.num_elements * sizeof(cl_float), NULL, &clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, mELLcol_d, CL_TRUE, 0,						
                                    mELL.num_elements * sizeof(cl_int), mELL.col, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mELLdata_d, CL_TRUE, 0,						
                                    mELL.num_elements * sizeof(cl_float), mELL.data, 0, NULL, NULL);

    /*
     * CREATE KERNELS
     */

    cl_kernel fixPROutput = clCreateKernel(program, "fixPROutput", &clStatus);
    clStatus |= clSetKernelArg(fixPROutput, 0, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |= clSetKernelArg(fixPROutput, 1, sizeof(cl_int), (void *)&(mELL.num_cols));

    cl_kernel normDiff = clCreateKernel(program, "normDiff", &clStatus);
    clStatus |= clSetKernelArg(normDiff, 0, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(normDiff, 1, sizeof(cl_mem), (void *)&vecOut_d);
    clStatus |= clSetKernelArg(normDiff, 2, sizeof(cl_int), (void *)&(mELL.num_cols));
    clStatus |= clSetKernelArg(normDiff, 3, WORKGROUP_SIZE*sizeof(cl_float), NULL);
    clStatus |= clSetKernelArg(normDiff, 4, sizeof(cl_mem), (void *)&norm_diff_d);

    // create kernel ELL and set arguments
    cl_kernel kernelELL = clCreateKernel(program, "mELL", &clStatus);
    clStatus |= clSetKernelArg(kernelELL, 0, sizeof(cl_mem), (void *)&mELLcol_d);
    clStatus |= clSetKernelArg(kernelELL, 1, sizeof(cl_mem), (void *)&mELLdata_d);
    clStatus |= clSetKernelArg(kernelELL, 2, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(kernelELL, 3, sizeof(cl_mem), (void *)&vecOut_d);
	clStatus |= clSetKernelArg(kernelELL, 4, sizeof(cl_int), (void *)&(mELL.num_rows));
	clStatus |= clSetKernelArg(kernelELL, 5, sizeof(cl_int), (void *)&(mELL.num_elementsinrow));


    /*
     * LAUNCH COMPUTATION
     */

    size_t local_item_size = WORKGROUP_SIZE;

    // Divide work
	int num_groups = (mELL.num_rows - 1) / local_item_size + 1;
    size_t global_item_size_helpers = num_groups * local_item_size;

    num_groups = (mELL.num_rows - 1) / local_item_size + 1;
    size_t global_item_size_ELL = num_groups * local_item_size;

    // ELL write, execute, read
    int iterations = 0;
    double dtimeELL = omp_get_wtime();

    while (1) {
        iterations++;

        // Write pr_in
        clStatus |= clEnqueueWriteBuffer(command_queue, vecIn_d, CL_TRUE, 0,						
                                        mELL.num_cols*sizeof(cl_float), pagerank_in, 0, NULL, NULL);	

        clStatus |= clEnqueueNDRangeKernel(command_queue, kernelELL, 1, NULL,						
                                        &global_item_size_ELL, &local_item_size, 0, NULL, NULL);
        clStatus |= clEnqueueNDRangeKernel(command_queue, fixPROutput, 1, NULL,						
                                        &global_item_size_helpers, &local_item_size, 0, NULL, NULL);
        
        // Transfer pr_out
        clStatus |= clEnqueueReadBuffer(command_queue, vecOut_d, CL_TRUE, 0,						
                                        mELL.num_rows*sizeof(cl_float), pagerank_out, 0, NULL, NULL);

        // Check exit criteria
        if(MAX_ITER > 0 && iterations >= MAX_ITER)
            break;

        if(CHECK_CONVERGENCE) {
            clStatus |= clEnqueueWriteBuffer(command_queue, norm_diff_d, CL_TRUE, 0,						
                                        sizeof(cl_float), &zero, 0, NULL, NULL);
            clStatus |= clEnqueueNDRangeKernel(command_queue, normDiff, 1, NULL,						
                                        &global_item_size_helpers, &local_item_size, 0, NULL, NULL);
            clStatus |= clEnqueueReadBuffer(command_queue, norm_diff_d, CL_TRUE, 0,						
                                        sizeof(cl_float), &norm, 0, NULL, NULL);
            norm = sqrt(norm);
            if(norm <= EPSILON)
                break;
        }

        // Update data
        float * tmp = pagerank_in;
        pagerank_in = pagerank_out;
        pagerank_out = tmp;
    }

    dtimeELL = omp_get_wtime() - dtimeELL;

    // Normalize output
    double sum = 0.;
    for(int i = 0; i < mELL.num_cols; i++)
        sum += pagerank_out[i];
    for(int i = 0; i < mELL.num_cols; i++)
        pagerank_out[i] /= sum;
    
    *end = omp_get_wtime();

    // Report results
    printf("Total number of iterations: %d\n", iterations);
    printf("ELL - Average time per iteration: %.4f\n", dtimeELL / iterations);

    // Free memory structures
    clStatus = clReleaseKernel(fixPROutput);
    clStatus = clReleaseKernel(normDiff);
    clStatus = clReleaseKernel(kernelELL);

    clStatus = clReleaseMemObject(vecIn_d);
    clStatus = clReleaseMemObject(vecOut_d);
    clStatus = clReleaseMemObject(mELLcol_d);
    clStatus = clReleaseMemObject(mELLdata_d);
    clStatus = clReleaseMemObject(norm_diff_d);
    
    ocl_destroy(command_queue, context, program);
    free(pagerank_in);
    return pagerank_out;
}
