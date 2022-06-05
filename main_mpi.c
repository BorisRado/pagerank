#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include <stdbool.h>
#include "readers/custom_matrix.h"
#include "readers/mtx_sparse.h"
#include "pagerank_implementations/pagerank_custom.h"
#include "helpers/file_helper.h"
#include "global_config.h"

float * measure_time_custom_matrix_in_mpi(int ** edges, int * in_degrees, int * out_degrees, int nodes_count, int edges_count,
                                        int world_size, int my_id);

int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Usage: ./a.out <graph_file_name> <out_file_name>\n");
        exit(1);
    }

    int my_id, world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int ** edges;
    int * out_degrees;
    int * in_degrees;
    int nodes_count, edges_count, i;
    double start, end;

    if (my_id == 0){
        start = omp_get_wtime();
        if(read_edges(argv[1], &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
            exit(1);
        end = omp_get_wtime();
        printf("Matrix reading time: %.4f\n", end - start);
    }

    //Broadcast dimensions
    MPI_Bcast(&nodes_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edges_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Allocate memory
    if (my_id != 0){
        //printf("Received nodes_count=%d\n", nodes_count);
        //printf("Received edges_count=%d\n", edges_count);

        out_degrees = (int *) calloc(nodes_count, sizeof(int));
        in_degrees = (int *) calloc(nodes_count, sizeof(int));
        int * contiguous_space = (int *) malloc(2 * (edges_count) * sizeof(int));
        edges = (int **) malloc((edges_count) * sizeof(int *));
        for (i = 0; i < (edges_count); i++)
            (edges)[i] = &contiguous_space[2 * i];
    }

    MPI_Bcast(*edges, 2 * edges_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(out_degrees, nodes_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(in_degrees, nodes_count, MPI_INT, 0, MPI_COMM_WORLD);

    float* pagerank_mpi = measure_time_custom_matrix_in_mpi(edges, in_degrees, out_degrees, nodes_count, edges_count,
                                      world_size, my_id);

    MPI_Finalize();
}


float * measure_time_custom_matrix_in_mpi(int ** edges, int * in_degrees, int * out_degrees, int nodes_count, int edges_count, int world_size, int my_id) {
    double start, end;
    int ** graph;
    int * leaves;
    int leaves_count;

    start = omp_get_wtime();
    format_graph_in(edges, in_degrees, out_degrees, &leaves_count, &leaves, &graph, nodes_count, edges_count);
    end = omp_get_wtime();
    
    if (my_id == 0)
        printf("Matrix formatting time: %.4f\n", end - start);

    start = omp_get_wtime();
    float * pagerank_mpi = pagerank_custom_in_mpi(graph, in_degrees, out_degrees, leaves_count, leaves, nodes_count, EPSILON, 
                            my_id, world_size);
    end = omp_get_wtime();
    
    if (my_id == 0)
        printf("Pagerank computation time (MPI): %.4f\n\n", end - start);

    if (my_id == 0){
        start = omp_get_wtime();
        float * pagerank = pagerank_custom_in(graph, in_degrees, out_degrees, leaves_count, leaves, nodes_count, EPSILON, false);
        end = omp_get_wtime();
        printf("Pagerank computation time (serial): %.4f\n\n", end - start);

        compare_vectors(pagerank, pagerank_mpi, nodes_count);
    }

    free(graph);
    return pagerank_mpi;
}