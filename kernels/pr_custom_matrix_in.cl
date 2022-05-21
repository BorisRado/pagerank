__kernel void compute_leaked_pagerank(
    int leaves_count,
    __global int * leaves,
    __global float * pagerank,
    __global float * leaked_pagerank_per_node,
    __local float * leaks
) {
    /**
     * sums the pagerank value of all the nodes in teh leaves array
     *   and returns the initial value that should be assigned to every node,
     *   using the formula:
     *           leaked_pagerank + (1 - leaked_pagerank) * (1 - TELEPORTATION_PROBABILITY)
     *   this value is returned in the `leaked_pagerank_per_node` variable
     */
    // if (get_group_id(0) > 0) return; // only one work group performs this operation

    // note: the code assumes that the local size is a power of 2
    int lid = get_local_id(0);
    float leak = 0.0;
    while (lid < leaves_count) {
        leak += pagerank[leaves[lid]];
        lid += get_global_size(0);
    }
    
    lid = get_local_id(0);
    leaks[lid] = leak;
    barrier(CLK_LOCAL_MEM_FENCE);

    // perform reduction over leaks
    int limit = get_global_size(0) / 2;
    while (lid < limit) {
        leaks[lid] += leaks[lid + limit];
        limit /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float leaked_pagerank_per_node_ = leaks[0] + (1 - leaks[0]) * (1 - 0.85); // TO-DO pass teleportation probability
        *leaked_pagerank_per_node = leaked_pagerank_per_node_;
        // printf("leaked: %f\n", *leaked_pagerank_per_node);
    }

}

__kernel void compute_norm_difference_wg(
    __global float * a, // e.g. old pagerank
    __global float * b, // e.g. new pagerank
    __global float * group_diffs, // array containing `get_num_groups` elements
    __local float * partial, // local item size
    int n // number of elements in a and b
) {
    /**
     * each work group computes the sum of square differences on a sectin of the data,
     * and writes the result to index `get_group_id(0)` of the `group_diffs` array.
     * In a second set, call the compute_norm_difference_fin function
     */
    
    // note: the code assumes that the local size is a power of 2

    int gid = get_global_id(0);
    int lid = get_local_id(0);

    float _diff = 0.0;
    while (gid < n) {
        _diff += pow(a[gid] - b[gid], 2);
        gid += get_global_size(0);
    }
    partial[lid] = _diff;

    // reduction
    for(int i = (get_local_size(0) >> 1); i > 0; i >>= 1) {
        if(lid < i)
            partial[lid] += partial[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        group_diffs[get_group_id(0)] = partial[0];
    }
}

__kernel void compute_norm_difference_fin(
    __global float * group_diffs,
    __global float * norm,
    __local float * partial,
    int n
) {
    /**
     * sums the `n` values that have been computed in the compute_norm_difference_wg function,
     * which are passed in the `group_diffs` array, and stores their sum in the `norm` variable 
     */
    
    // note: the code assumes that the local size is a power of 2. It should be set to the smallest
    // power of two that is larger or equal to the number of work groups in the previous step
    
    int lid = get_local_id(0);
    if (lid < n)
        partial[lid] = group_diffs[lid];
    else
        partial[lid] = 0.0;

    // reduction
    for(int i = (n >> 1); i > 0; i >>= 1) {
        if(lid < i) 
            partial[lid] += partial[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE); // TO-DO: speed-up if we have max 32 work groups in the previous step?
    }

    if (lid == 0)
        *norm = partial[0];

}

__kernel void pagerank_step(
    __global int * graph,
    __global int * in_deg_CDF, // used to correctly address the graph
    __global int * in_degrees,
    __global int * out_degrees,
    __global float * pagerank_old,
    __global float * pagerank_new,
    __global float * leaked_pagerank_addition_glob,
    int nodes_count,
    int threads_per_row
) {
    /**
     * this kernel performs one step of pagerank computation (one "matrix multiplication").
     * 
     * Parameters:
     *      * `threads_per_row`: how many threads compute concurrently the new pagerank value of some node
     *      * `leaked_pagerank_addition`: value as computed by the compute_leaked_pagerank kernel
     */

     // TO-DO: this function has not yet been tested

    int lid = get_local_id(0);
    int gid = get_global_id(0);
    float leaked_pagerank_addition = *leaked_pagerank_addition_glob / (float) nodes_count;

    float pagerank_contribution;
    int i;

    // if (lid == 0) {
    //     printf("start: %f\n", leaked_pagerank_addition);
    // }
    // if (lid == 0) {
    //     // printf("here\n");
    //     for(int i = 0; i < nodes_count; i++) {
    //         printf("%d: ", i);
    //         for (int j = 0; j < in_degrees[i]; j++) {
    //             printf("%d ", graph[in_deg_CDF[i] + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // simple version: each thread works on a line (i.e. processes the pagerank for one node)
    while (gid < nodes_count) {

        float i_pr = leaked_pagerank_addition;
        for (i = 0; i < in_degrees[gid]; i++){
            i_pr += 0.85 * pagerank_old[graph[in_deg_CDF[gid] + i]] / out_degrees[graph[in_deg_CDF[gid] + i]];
        }
        pagerank_new[gid] = i_pr;

        gid += get_global_size(0);
    }

}