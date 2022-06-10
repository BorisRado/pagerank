#ifndef FILE_HELPER
#define FILE_HELPER

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
    FILE * fp = fopen(file_name, "a");

    if (fp == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i = 0; i < nodes_count; i++)
        fprintf(fp, "%.12f\n", pagerank[i]);
    
    fclose(fp);
}

int read_edges(char * file_name, int *** edges, int ** out_degrees,
            int ** in_degrees, int * nodes_count, int * edges_count) {
    /*
        Reads the graph at file_name and returns an 2D array containing one 
    entry for each edge present in the graph. `out_degrees` contains
    the number of outgoing edges from each node.
        File should be formated as edge list with first line containing
    [node_count]\t[edge_count] and every line aferwards containing
    [arc_tail]\t[arc_head] for directed graphs (or endpoints for undirected).
        Returns 1 upon failure.
    */
   
    FILE * fp;
    int * contiguous_space;
    int i;
    
    fp = fopen(file_name, "r");
    if (fp == NULL) {
        printf("ERROR while reading graph `%s`\n", file_name);
        return 1;
    }

    int from, to;
    
    // read number of nodes and edges
    if (fscanf(fp, "%d\t%d", nodes_count, edges_count) != 2) {
        printf("Error while reading first line...\n");
        return 1;
    }

    // allocate space
    *out_degrees = (int *) calloc(*nodes_count, sizeof(int));
    *in_degrees = (int *) calloc(*nodes_count, sizeof(int));
    contiguous_space = (int *) malloc(2 * (*edges_count) * sizeof(int));
    *edges = (int **) malloc((*edges_count) * sizeof(int *));
    for (i = 0; i < (*edges_count); i++)
        (*edges)[i] = &contiguous_space[2 * i];

    // read lines
    i = 0;
    while (fscanf(fp, "%d\t%d", &from, &to) != EOF) {
        if(from < 0 || from > *nodes_count || to < 0 || to > *nodes_count)
            printf("ERROR: Line %d: %d\t%d\n", i, from, to);
        else {
            (*out_degrees)[from]++;
            (*in_degrees)[to]++;
            (*edges)[i][0] = from;
            (*edges)[i][1] = to;
            i++;
        }
    }

    fclose(fp);
    return 0;
}

#endif
