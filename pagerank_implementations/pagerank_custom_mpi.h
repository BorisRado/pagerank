#include <mpi.h>
#include <math.h>
#include <omp.h>
#include "../helpers/helper.h"
#include "../helpers/ocl_helper.h"
#include "../global_config.h"

float * pagerank_custom_in_mpi(int ** graph, int * in_degrees, int * out_degrees,
                int leaves_count, int * leaves, int nodes_count, double epsilon, 
                bool parallel_for, int my_id, int world_size) {

    int my_start = my_id * nodes_count / world_size;
    int my_end = (my_id + 1) * nodes_count / world_size;
    int my_node_count = my_end - my_start;

    // Needed in order to use MPI_Allgatherv()
    int* counts = (int* ) malloc(world_size * sizeof(int));
    int* displacements = (int* ) calloc(world_size, sizeof(int));
    MPI_Allgather(&my_node_count, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    for (size_t i = 1; i < world_size; i++){
        displacements[i] = displacements[i - 1] + counts[i - 1];
    }

    float *pagerank_old, *pagerank_new;
    init_pagerank(&pagerank_old, &pagerank_new, nodes_count);
    
    float *my_pagerank_old, *my_pagerank_new;
    init_pagerank(&my_pagerank_old, &my_pagerank_new, my_node_count);

    float init_pagerank;
    float norm_diff, my_norm_diff;

    int i, j;
    int iterations = 0;
    bool done = false;
    
    do{
        if (my_id == 0){
            float leaked_pagerank = 0.;
            for (i = 0; i < leaves_count; i++) {
                leaked_pagerank += pagerank_old[leaves[i]]; 
            }
            leaked_pagerank = leaked_pagerank + (1 - leaked_pagerank) * (1 - DAMPENING);
            init_pagerank = leaked_pagerank / (float)nodes_count;
        }

        MPI_Bcast(&init_pagerank, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Compute local pagerank
        #pragma omp parallel for if(parallel_for) schedule(guided) private(i,j) shared(out_degrees,graph, init_pagerank, my_start, my_end)
        for (i = my_start; i < my_end; i++) {
            float i_pr = init_pagerank;
            for (j = 0; j < in_degrees[i]; j++){
                i_pr += DAMPENING * pagerank_old[graph[i][j]] / out_degrees[graph[i][j]];
            }
            my_pagerank_new[i - my_start] = i_pr;
        }
        
        MPI_Allgatherv(my_pagerank_new, my_node_count, MPI_FLOAT, pagerank_old, 
                        counts, displacements, MPI_FLOAT, MPI_COMM_WORLD);

        my_norm_diff = get_norm_difference(my_pagerank_old, my_pagerank_new, my_node_count);
        MPI_Reduce(&my_norm_diff, &norm_diff, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


        iterations++;
        if (my_id == 0){
            if (CHECK_CONVERGENCE && norm_diff < epsilon || iterations > 200){
                done = true;
                //printf("Converged!\n");
            }
            
            if (MAX_ITER > 0 && iterations >= MAX_ITER){
                done = true;
                //printf("Max Iters!\n");
            }
        }

        MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        swap_pointers(&my_pagerank_old, &my_pagerank_new);
        
    }while(!done);


    if (my_id == 0)
        printf("Total pagerank iterations: %d\n", iterations);

    return pagerank_old;
}

