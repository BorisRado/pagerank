#include <stdio.h>
#include <stdlib.h>

void get_graph_size(char * file_name, int * nodes_count, int * edges_count) {
    // reads first line of `file_name` file and inserts the number of nodes and
    // number of edges in the provided values

    FILE * fp = fopen(file_name, "r");

    if (fscanf(fp, "%d\t%d", nodes_count, edges_count) != 2)
        exit(1);

    fclose(fp);
}

void write_to_file(char * file_name, float * pagerank, int nodes_count) {
    // write the vector `pagerank` with `nodes_count` elements to file with name `file_name`
    FILE * fp = fopen(file_name, "w");

    if (fp == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i = 0; i < nodes_count; i++)
        fprintf(fp, "%.12f\n", pagerank[i]);
    
    fclose(fp);

}