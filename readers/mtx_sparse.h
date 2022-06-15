#ifndef MTX_SPARSE
#define MTX_SPARSE

#include <stdio.h>
#include <stdlib.h>
#include "../helpers/file_helper.h"


/*
 * MATRIX DEFINITIONS
 */

struct mtx_COO  // COOrdinates
{
    int *row;
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

typedef struct mtx_COO mtx_COO;

struct mtx_CSR  // Compressed Sparse Row
{
    int *rowptr;
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

typedef struct mtx_CSR mtx_CSR;

struct mtx_ELL      // ELLiptic (developed by authors of ellipctic package)
{
    int *col;
    float *data;
    long long num_rows;
    int num_cols;
    int num_nonzeros;
    long long num_elements;
    int num_elementsinrow;    
};

typedef struct mtx_ELL mtx_ELL;


/*
 * QSORT HELPER
 */

int edge_compare(const void * a, const void * b) { 
    int * e1 = *(int **) a;
    int * e2 = *(int **) b;

    if (e1[1] < e2[1])
        return -1;
    else if (e1[1] > e2[1])
        return +1;
    else if (e1[0] < e2[0])
        return -1;
    else if (e1[0] > e2[0])
        return +1;
    else 
        return 0;
}


/*
 * MATRIX PRINT TO STDOUT
 */

void mtx_CSR_print(struct mtx_CSR *mCSR) {
    for(int i = 0; i < mCSR->num_rows; i++) {
        printf("ROW %d:\t", i);
        for(int j = mCSR->rowptr[i]; j < mCSR->rowptr[i+1]; j++) {
            printf("c%02d(%.3f), ", mCSR->col[j], mCSR->data[j]);
        }
        printf("\n");
    }
}

void mtx_ELL_print(struct mtx_ELL *mELL) {
    for(int i = 0; i < mELL->num_rows; i++) {
        printf("ROW %d:\t", i);
        for(int j = 0; j < mELL->num_elementsinrow; j++) {
            int ell_index = j * mELL->num_rows + i;
            if(mELL->data[ell_index] != 0)
                printf("c%02d(%.3f), ", mELL->col[ell_index], mELL->data[ell_index]);
        }
        printf("\n");
    }
}


/*
 * EDGE-MATRIX CONVERTERS
 */

int get_COO_from_edges(struct mtx_COO * mCOO, int *** edges, int ** out_degrees, int * nodes_count, int * edges_count) {
    
    // sort edges
    qsort(*edges, *edges_count, 2 * sizeof(int), edge_compare);

    mCOO->num_rows = (*nodes_count);
    mCOO->num_cols = (*nodes_count);
    mCOO->num_nonzeros = (*edges_count);

    // allocate COO matrix
    mCOO->data = (float *) malloc((*edges_count) * sizeof(float));
    mCOO->col = (int *) malloc((*edges_count) * sizeof(int));
    mCOO->row = (int *) malloc((*edges_count) * sizeof(int));

    if(mCOO->data == NULL || mCOO->col == NULL || mCOO->row == NULL)  {
        printf("Could not allocate space for COO matrix.\n");
        return 1;
    }

    // copy edges to COO structures and compute values
    for (int i = 0; i < *edges_count; i++) {
        mCOO->row[i] = (*edges)[i][1];
        mCOO->col[i] = (*edges)[i][0];               
           
        // If no outgoing links, return error (current edge is outgoing)
        if((*out_degrees)[(*edges)[i][0]] == 0) {
            printf("Inconsistency in data: outgoing edge for node %d with 0 out-degree.\n", (*edges)[i][0]);
            return 1;
        } else // Otherwise transition to one of outgoing links randomly                          
            mCOO->data[i] = 1./(*out_degrees)[(*edges)[i][0]];
    }

    return 0;
}

int get_CSR_from_edges(struct mtx_CSR *mCSR, int *** edges, int ** out_degrees, int * nodes_count, int * edges_count) {
    int row, prev_row, first_row;

    // sort edges
    qsort(*edges, *edges_count, 2 * sizeof(int), edge_compare);

    mCSR->num_nonzeros = (*edges_count);
    mCSR->num_rows = (*nodes_count);
    mCSR->num_cols = (*nodes_count);

    // allocate CSR matrix
    mCSR->data =  (float *) malloc((*edges_count) * sizeof(float));
    mCSR->col = (int *) malloc((*edges_count) * sizeof(int));
    mCSR->rowptr = (int *) calloc((*nodes_count) + 1, sizeof(int));

    if(mCSR->data == NULL || mCSR->col == NULL || mCSR->rowptr == NULL)  {
        printf("Could not allocate space for CSR matrix.\n");
        return 1;
    }

    mCSR->rowptr[0] = 0;
    mCSR->rowptr[(*nodes_count)] = (*edges_count);
    prev_row = 0;
    first_row = (*edges)[0][1];
    // copy edges to CSR structures and compute values
    for (int i = 0; i < *edges_count; i++) {
        mCSR->col[i] = (*edges)[i][0];

        // If no outgoing links, return error (current edge is outgoing)
        if((*out_degrees)[(*edges)[i][0]] == 0) {
            printf("Inconsistency in data: outgoing edge for node %d with 0 out-degree.\n", (*edges)[i][0]);
            return 1;
        } else // Otherwise transition to one of outgoing links randomly                          
            mCSR->data[i] = 1./(*out_degrees)[(*edges)[i][0]];

        row = (*edges)[i][1];
        if(row > prev_row) {
            int r = row;
            while(r > first_row && mCSR->rowptr[r] == 0)
                mCSR->rowptr[r--] = i;
            prev_row = row;
        }
    }

    return 0;
}

// UNTESTED: try with CSR converter first
int get_ELL_from_edges(struct mtx_ELL *mELL, int *** edges, int ** out_degrees, int * nodes_count, int * edges_count) {
    int row, prev_row, row_size;
    long long ell_index;

    // sort edges
    qsort(*edges, *edges_count, 2 * sizeof(int), edge_compare);

    mELL->num_nonzeros = (*edges_count);
    mELL->num_rows = (long long) (*nodes_count);
    mELL->num_cols = (*nodes_count);

    mELL->num_elementsinrow = 0;
    prev_row = 0;
    row_size = 0;
    // compute max el. per row
    for (int i = 0; i < (*edges_count); i++) {
        row = (*edges)[i][1];
        if(row > prev_row) {
            if(mELL->num_elementsinrow < row_size)
                mELL->num_elementsinrow = row_size;
            row_size = 1;
            prev_row = row;
        } else
            row_size++;
    }
    if(mELL->num_elementsinrow < row_size)
        mELL->num_elementsinrow = row_size;

    // allocate ELL matrix
    mELL->num_elements = mELL->num_rows * mELL->num_elementsinrow;
    mELL->data = (float *) calloc(mELL->num_elements, sizeof(float));
    mELL->col = (int *) calloc(mELL->num_elements, sizeof(int));

    if(mELL->data == NULL || mELL->col == NULL)  {
        printf("Could not allocate space for ELL matrix.\n");
        return 1;
    }

    prev_row = 0;
    row_size = 0;

    // copy edges to ELL structures and compute values
    for (int i = 0; i < (*edges_count); i++) {
        row  = (*edges)[i][1];
        if(row > prev_row) {
            row_size = 1;
            prev_row = row;
        } else
            row_size++;

        ell_index = (row_size - 1) * mELL->num_rows + row;

        mELL->col[ell_index] = (*edges)[i][0];

        // If no outgoing links, return error (current edge is outgoing)
        if((*out_degrees)[(*edges)[i][0]] == 0) {
            printf("Inconsistency in data: outgoing edge for node %d with 0 out-degree.\n", (*edges)[i][0]);
            return 1;
        } else // Otherwise transition to one of outgoing links randomly                          
            mELL->data[ell_index] = 1./(*out_degrees)[(*edges)[i][0]];

    }

    return 0;
}


/*
 * FILE-MATRIX WRAPPERS (read edges + convert) - contain memory leak (contiguous_space is not freed)
 */

int get_COO_from_file(struct mtx_COO * mCOO, char * file_name) {
    /*
    ** File should be formated as edge list with first line containing
    ** [node_count]\t[edge_count] and every line aferwards containing
    ** [arc_tail]\t[arc_head] for directed graphs (or endpoints for undirected)
    */

    // read edges
    int nodes_count, edges_count, i;
    int ** edges;
    int * out_degrees;
    int * in_degrees;
    if(read_edges(file_name, &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
        return 1;

    int status = get_COO_from_edges(mCOO, &edges, &out_degrees, &nodes_count, &edges_count);

    // NOTE: contiguous_space from read_edges is not freed (memory leak)
    free(edges);
    free(out_degrees);
    free(in_degrees);

    return status;
}

int get_CSR_from_file(struct mtx_CSR * mCSR, char * file_name) {
    /*
    ** File should be formated as edge list with first line containing
    ** [node_count]\t[edge_count] and every line aferwards containing
    ** [arc_tail]\t[arc_head] for directed graphs (or endpoints for undirected)
    */

    // read edges
    int nodes_count, edges_count, i;
    int ** edges;
    int * out_degrees;
    int * in_degrees;
    if(read_edges(file_name, &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
        return 1;

    int status = get_CSR_from_edges(mCSR, &edges, &out_degrees, &nodes_count, &edges_count);

    // NOTE: contiguous_space from read_edges is not freed (memory leak)
    free(edges);
    free(out_degrees);
    free(in_degrees);

    return status;
}

int get_ELL_from_file(struct mtx_ELL * mELL, char * file_name) {
    /*
    ** File should be formated as edge list with first line containing
    ** [node_count]\t[edge_count] and every line aferwards containing
    ** [arc_tail]\t[arc_head] for directed graphs (or endpoints for undirected)
    */

    // read edges
    int nodes_count, edges_count, i;
    int ** edges;
    int * out_degrees;
    int * in_degrees;
    if(read_edges(file_name, &edges, &out_degrees, &in_degrees, &nodes_count, &edges_count))
        return 1;

    int status = get_ELL_from_edges(mELL, &edges, &out_degrees, &nodes_count, &edges_count);

    // NOTE: contiguous_space from read_edges is not freed (memory leak)
    free(edges);
    free(out_degrees);
    free(in_degrees);

    return status;
}


/*
 * CONVERTERS BETWEEN MTX FORMATS
 */

int mtx_CSR_create_from_mtx_COO(struct mtx_CSR *mCSR, struct mtx_COO *mCOO) {
    mCSR->num_nonzeros = mCOO->num_nonzeros;
    mCSR->num_rows = mCOO->num_rows;
    mCSR->num_cols = mCOO->num_cols;

    // allocate matrix
    mCSR->data =  (float *)malloc(mCSR->num_nonzeros * sizeof(float));
    mCSR->col = (int *)malloc(mCSR->num_nonzeros * sizeof(int));
    mCSR->rowptr = (int *)calloc(mCSR->num_rows + 1, sizeof(int));
    if(mCSR->data == NULL || mCSR->col == NULL || mCSR->rowptr == NULL)  {
        printf("Could not allocate space for CSR matrix.\n");
        return 1;
    }

    mCSR->data[0] = mCOO->data[0];
    mCSR->col[0] = mCOO->col[0];
    mCSR->rowptr[0] = 0;
    mCSR->rowptr[mCSR->num_rows] = mCSR->num_nonzeros;
    for (int i = 1; i < mCSR->num_nonzeros; i++)
    {
        mCSR->data[i] = mCOO->data[i];
        mCSR->col[i] = mCOO->col[i];
        if (mCOO->row[i] > mCOO->row[i-1])
        {
            int r = mCOO->row[i];
            while (r > 0 && mCSR->rowptr[r] == 0)
                mCSR->rowptr[r--] = i;
        }
    }

    return 0;
}

int mtx_ELL_create_from_mtx_CSR(struct mtx_ELL *mELL, struct mtx_CSR *mCSR) {
    mELL->num_rows = (long long) mCSR->num_rows;
    mELL->num_cols = mCSR->num_cols;
    mELL->num_nonzeros = mCSR->num_nonzeros;

    mELL->num_elementsinrow = 0;
    // compute max el. per row
    for (int i = 0; i < mELL->num_rows; i++) {
        if (mELL->num_elementsinrow < mCSR->rowptr[i+1]-mCSR->rowptr[i]) 
            mELL->num_elementsinrow = mCSR->rowptr[i+1]-mCSR->rowptr[i];
    }
    mELL->num_elements = mELL->num_rows * mELL->num_elementsinrow;

    // allocate matrix
    mELL->data = (float *)calloc(mELL->num_elements, sizeof(float));
    mELL->col = (int *) calloc(mELL->num_elements, sizeof(int));
    if(mELL->data == NULL || mELL->col == NULL)  {
        printf("Could not allocate space for ELL matrix.\n");
        return 1;
    }

    // copy data to ELL structures
    for (int i = 0; i < mELL->num_rows; i++) {
        for (int j = mCSR->rowptr[i]; j < mCSR->rowptr[i+1]; j++) {            
            long long ELL_j = (j - mCSR->rowptr[i]) * mELL->num_rows + i;
            mELL->data[ELL_j] = mCSR->data[j];
            mELL->col[ELL_j] = mCSR->col[j];
        }
    }

    return 0;
}


/*
 * DEALLOCATION FUNCTIONS
 */

int mtx_COO_free(struct mtx_COO *mCOO)
{
    free(mCOO->data);
    free(mCOO->col);
    free(mCOO->row);

    return 0;
}

int mtx_CSR_free(struct mtx_CSR *mCSR) {
    free(mCSR->data);
    free(mCSR->col);
    free(mCSR->rowptr);

    return 0;
}

int mtx_ELL_free(struct mtx_ELL *mELL)
{
    free(mELL->col);
    free(mELL->data);

    return 0;
}




# endif
