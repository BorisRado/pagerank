#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include <stdbool.h>
#include <unistd.h>
#include "readers/custom_matrix.h"
#include "readers/mtx_sparse.h"
#include "pagerank_implementations/pagerank_custom_mpi.h"
#include "pagerank_implementations/pagerank_custom.h"
#include "helpers/file_helper.h"
#include "global_config.h"

#define MASTER 0

void measure_time_custom_matrix_in_mpi(int ** graph, int * in_degrees, int * out_degrees, int nodes_count, int leaves_count, 
                int * leaves, int my_id, int world_size);

int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Usage: ./a.out <graph_file_name> <out_file_name>\n");
        exit(1);
    }

    int my_id, world_size;

    float start, end;
    start = MPI_Wtime();
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    end = MPI_Wtime();
    if (my_id == 0) {
        printf("initializeing... %f\n", end - start);
    }

    int ** graph;
    int * out_degrees;
    int * in_degrees;
    int * leaves;
    int nodes_count, edges_count, i, leaves_count;

    if (my_id == MASTER){
        // the master node reads and formats the graph
        int ** edges;

        start = MPI_Wtime();
        if(read_edges(argv[1], &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
            exit(1);
        end = MPI_Wtime();
        printf("Matrix reading time: %.4f\n", end - start);

        // format graph
        start = MPI_Wtime();
        format_graph_in(edges, in_degrees, out_degrees, &leaves_count, &leaves, &graph, nodes_count, edges_count);
        end = MPI_Wtime();
        printf("Matrix formatting time: %.4f\n", end - start);
    }

    measure_time_custom_matrix_in_mpi(graph, in_degrees, out_degrees,
                nodes_count, leaves_count, leaves, my_id, world_size);

    MPI_Finalize();
}


void measure_time_custom_matrix_in_mpi(int ** graph, int * in_degrees, int * out_degrees, int nodes_count, int leaves_count, 
                int * leaves, int my_id, int world_size) {
    float start, end;
    start = MPI_Wtime();

    //Broadcast values required by all the nodes
    MPI_Bcast(&nodes_count, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&leaves_count, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (my_id != MASTER)
        leaves = (int *) malloc(leaves_count * sizeof(int));
    MPI_Bcast(leaves, leaves_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    // divide word and broadcast the graph
    int * counts_send_graph = (int *) calloc(world_size, sizeof(int));
    int * displacements_graph = (int *) calloc(world_size, sizeof(int));
    int * counts_send_nodes = (int *) malloc(world_size * sizeof(int));
    int * displacements_nodes = (int *) malloc(world_size * sizeof(int));

    // divide nodes
    int init_node, end_node, nodes, cdf_graph = 0, cdf_nodes = 0;
    for (int i = 0; i < world_size; i++) {
        init_node = i * nodes_count / world_size;
        end_node = (i + 1) * nodes_count / world_size;

        // set values for the nodes (in and out degrees)
        nodes = end_node - init_node;
        counts_send_nodes[i] = nodes;
        displacements_nodes[i] = cdf_nodes;
        cdf_nodes += nodes;

        // set values for the graph
        if (my_id == MASTER) {
            int node_total = 0;
            for (int node = init_node; node < end_node; node++)
                node_total += in_degrees[node];

            counts_send_graph[i] = node_total;
            displacements_graph[i] = cdf_graph;
            cdf_graph += node_total;
        }
    }

    MPI_Bcast(counts_send_graph, world_size, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(displacements_graph, world_size, MPI_INT, MASTER, MPI_COMM_WORLD);

    // scatter only the needed part of the graph
    int * my_graph_contiguous = (int *) malloc(counts_send_graph[my_id] * sizeof(int));
    int * my_in_degrees = (int *) malloc(counts_send_nodes[my_id] * sizeof(int));
    if (my_id != MASTER)
        out_degrees = (int *) malloc(nodes_count * sizeof(int));
    for (int i = 0; i < counts_send_graph[my_id]; i++)
        my_graph_contiguous[i] = 1;

    int my_nodes = counts_send_nodes[my_id];
    MPI_Scatterv(&graph[0][0], counts_send_graph, displacements_graph, MPI_INT, 
            my_graph_contiguous, counts_send_graph[my_id], MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(out_degrees, nodes_count, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(&in_degrees[0], counts_send_nodes, displacements_nodes, MPI_INT,
            my_in_degrees, my_nodes, MPI_INT, MASTER, MPI_COMM_WORLD);

    int ** my_graph = (int **) malloc(counts_send_nodes[my_id] * sizeof(int*));
    int cdf = 0;
    for (int i = 0; i < my_nodes; i++) {
        my_graph[i] = &my_graph_contiguous[cdf];
        cdf += my_in_degrees[i];
    }

    float * pagerank_mpi = pagerank_custom_in_mpi(my_graph, my_in_degrees, out_degrees, leaves_count, leaves,
                    nodes_count, EPSILON, true, my_id, world_size);
    end = MPI_Wtime();
    
    if (my_id == 0)
        printf("TOTAL MPI - Pagerank computation time (MPI): %.4f\n\n", end - start);

    if (my_id == 0) {
        // compute pagerank with an implementation we know works ok
        start = MPI_Wtime();
        float * pagerank = pagerank_custom_in(graph, in_degrees, out_degrees, leaves_count, leaves, nodes_count, EPSILON, false);
        end = MPI_Wtime();
        printf("Pagerank computation time (serial): %.4f\n\n", end - start);

        compare_vectors(pagerank, pagerank_mpi, nodes_count);
        free(graph);
    }
    free(my_graph_contiguous);
    free(my_graph);
}