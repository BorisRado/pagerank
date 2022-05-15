#include <stdio.h>
#include <stdlib.h>
#include "../helpers/helper.h"

/*
This script contains implementations that use the custom matrix representation
*/

#define TELEPORTATION_PROBABILITY 0.85

float * pagerank_custom_out(int ** graph, int * out_degrees, int leaves_count, int * leaves, int nodes_count, double epsilon) {
    float * pagerank_old = (float*) malloc(nodes_count * sizeof(float));
    float * pagerank_new = (float*) malloc(nodes_count * sizeof(float));
    
    int i, j;

    float init_value = 1 / (float)nodes_count;
    for (i = 0; i < nodes_count; i++)
        pagerank_old[i] = init_value;

    int iterations = 0;

    do {

        float leaked_pagerank = 0.;
        for (i = 0; i < leaves_count; i++) {
            leaked_pagerank += pagerank_old[leaves[i]]; 
        }
        leaked_pagerank = leaked_pagerank + (1 - leaked_pagerank) * (1 - TELEPORTATION_PROBABILITY);

        for (i = 0; i < nodes_count; i++)
            pagerank_new[i] = leaked_pagerank / (float)nodes_count;

        // #pragma omp parallel private(i,j)
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
                int leaves_count, int * leaves, int nodes_count, double epsilon) {
    float * pagerank_old = (float*) malloc(nodes_count * sizeof(float));
    float * pagerank_new = (float*) malloc(nodes_count * sizeof(float));
    
    int i, j;

    float init_value = 1 / (float)nodes_count;
    for (i = 0; i < nodes_count; i++)
        pagerank_old[i] = init_value;

    int iterations = 0;

    do {

        float leaked_pagerank = 0.;
        for (i = 0; i < leaves_count; i++) {
            leaked_pagerank += pagerank_old[leaves[i]]; 
        }
        leaked_pagerank = leaked_pagerank + (1 - leaked_pagerank) * (1 - TELEPORTATION_PROBABILITY);

        for (i = 0; i < nodes_count; i++)
            pagerank_new[i] = leaked_pagerank / (float)nodes_count;

        // #pragma omp parallel private(i,j)
        for (i = 0; i < nodes_count; i++) {
            for (j = 0; j < in_degrees[i]; j++){
                pagerank_new[i] += TELEPORTATION_PROBABILITY * pagerank_old[graph[i][j]] / out_degrees[graph[i][j]];
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