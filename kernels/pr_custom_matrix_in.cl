__kernel void compute_leaked_pagerank(
    __global int zero_deg_nodes_count,
    __global int * zero_deg_nodes,
    __global float * pagerank,
    __global float leaked_pagerank_per_node,
    __local float * leaks
) {
    /**
     * sums the pagerank value of all the nodes in teh zero_deg_nodes array
     *   and returns the initial value that should be assigned to every node,
     *   using the formula:
     *           leaked_pagerank + (1 - leaked_pagerank) * (1 - TELEPORTATION_PROBABILITY)
     *   this value is returned in the `leaked_pagerank_per_node` variable
     */
    if (get_group_id(0) > 0) return; // only one work group performs this operation

    // TO-DO... still need to test this function
    // note: the code assumes that the local size is a power of 2
    int lid = get_local_id(0);
    float leak = 0.0;
    while (lid < zero_deg_nodes_count) {
        leak += pagerank[zero_deg_nodes[lid]];
        lid += get_global_size(0);
    }

    int lid = get_local_id(0);
    leaks[lid] = leak;

    // perform reduction over leaks
    int limit = get_global_size(0) / 2;
    while (lid < limit) {
        leaks[lid] = leaks[lid + limit];
        limit /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float leaked_pagerank_per_node_ = leaks[0] + (1 - leaks[0]) * (1 - TELEPORTATION_PROBABILITY);
        leaked_pagerank_per_node = leaked_pagerank_per_node_;
    }

}

__kernel void compute_pagerank(
    __global int ** graph,
    __global int * node_degrees,
    __global int nodes_count,
    __global float leaked_pagerank_addition,
    __global float * pagerank_old,
    __global float * pagerank_new,
    __global float teleportation_probability,
    __global int threads_per_row
) {
    /**
     * this kernel performs one step of pagerank computation (one "matrix multiplication").
     * 
     * Parameters:
     *      * `threads_per_row`: how many threads compute concurrently the new pagerank value of some node
     *      * `leaked_pagerank_addition`: value as computed by the compute_leaked_pagerank kernel
     */

     // TO-DO

    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int gsz = get_global_size(0);

    while (gid < nodes) {
        pagerank_new[gid] = leaked_pagerank_addition;
        gid += gsz;
    }

    gid = get_global_id(0);

    float pagerank_contribution;
    int node_degree, i, j;

    // simple version: each thread works on a line (i.e. processes the pagerank for one node)
    while (gid < nodes) {

        node_degree = node_degrees[gid];
        pagerank_contribution =
            TELEPORTATION_PROBABILITY * pagerank_old[i] / (float)out_degrees[i];
        for (i = 0; i <= node_degree; i++) {
            
            pagerank_new[graph[gid][i]] += pr_contribution;
        }

        gid += get_global_size(0);
    }

}