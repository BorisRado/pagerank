#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float square(float val) {
    return val * val;
}

float get_norm_difference(float * pagerank_old,
            float * pagerank_new, int len) {
    double sum_of_squares = 0.0;

    for (int i = 0; i < len; i++) {
        sum_of_squares += square(pagerank_new[i] - pagerank_old[i]);
    }
    return sqrt(sum_of_squares);

}

void swap_pointers(float ** a, float ** b) {
    float * c = *a;
    *a = *b;
    *b = c;
}

void compare_vectors(float * a, float * b, int n) {
    for (int i = 0; i < n; i++) 
        if (fabsf(a[i] - b[i]) > 1e-6) {
            printf("Inconsistencies in the two vectors!!!\n");
            exit(1);
        }
    printf("No differences found in the two vectors\n");
    
}

int compare_vectors_detailed(float * a, float * b, int n) {
    int mistakes = 0;
    for (int i = 0; i < n; i++) 
        if (fabsf(a[i] - b[i]) > 1e-6) {
            printf("Inconsistency at [%d]: %.8f, %.8f.\n", i, a[i], b[i]);
            mistakes = 1;
        }
    if(mistakes == 0)
        printf("No differences found in the two vectors\n");
    return mistakes;
}
