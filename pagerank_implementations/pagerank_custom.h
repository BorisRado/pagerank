#include <stdio.h>
#include <stdlib.h>
#include "../helpers/helper.h"

/*
This script contains implementations that use the custom matrix representation
*/

#define TELEPORTATION_PROBABILITY 0.85

float * pagerank_custom_1(int ** graph, int nodes_count, double epsilon) {
    float * pagerank_old = (float*) malloc(nodes_count * sizeof(float));
    float * pagerank_new = (float*) malloc(nodes_count * sizeof(float));
    
    int i, j;
    float init_value = 1 / (float)nodes_count;

    for (i = 0; i < nodes_count; i++)
        pagerank_old[i] = init_value;

    do {

        float total_pagerank = 0.0;
        for (i = 0; i < nodes_count; i++)
            pagerank_new[i] = 0;

        // #pragma omp parallel private(i,j)
        for (i = 0; i < nodes_count; i++) {
            float pagerank_contribution =
                TELEPORTATION_PROBABILITY * pagerank_old[i] / (float)graph[i][0];
            for (j = 1; j <= graph[i][0]; j++){
                total_pagerank += pagerank_contribution;
                pagerank_new[graph[i][j]] += pagerank_contribution;
            }

        }

        float node_addition = (1 - total_pagerank) / nodes_count;
        for (i = 0 ; i < nodes_count; i++)
            pagerank_new[i] += node_addition;
        
        swap_pointers(&pagerank_old, &pagerank_new);

    } while (get_norm_difference(pagerank_old, pagerank_new, nodes_count) > epsilon);
    swap_pointers(&pagerank_old, &pagerank_new);
    return pagerank_new;
}