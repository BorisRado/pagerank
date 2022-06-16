#ifndef MTX_HYBRID
#define MTX_HYBRID

#include "mtx_sparse.h"

struct mtx_JDS  // Jagged Diagonal Storage
{
    int num_cols;
    int num_nonzeros;
    long long num_elements;
    int num_pieces;
    mtx_ELL ** pieces;
    int ** row_ind;
};

typedef struct mtx_JDS mtx_JDS;


void mtx_JDS_print(struct mtx_JDS *mJDS) {
    for(int p = 0; p < mJDS->num_pieces; p++) {
        printf("PIECE %d\n", p);
        for(int i = 0; i < mJDS->pieces[p]->num_rows; i++) {
            printf("ROW %d:\t", mJDS->row_ind[p][i]);
            for(int j = 0; j < mJDS->pieces[p]->num_elementsinrow; j++) {
                int ell_index = j * mJDS->pieces[p]->num_rows + i;
                if(mJDS->pieces[p]->data[ell_index] != 0)
                    printf("c%02d(%.3f), ", mJDS->pieces[p]->col[ell_index], mJDS->pieces[p]->data[ell_index]);
            }
            printf("\n");
        }
    }
}


// generates up to <num_pieces> ELL matrices by grouping rows of similar lengths
// the interval between smallest and largest row is split into <num_pieces>
// consequently some pieces may be empty or very small, and empty pieces are removed
// it also ignores empty rows, and the indices of corresponding nodes are returned as <dangling>
int get_JDS_from_edges(struct mtx_JDS *mJDS, int** dangling, int* num_pieces, int *** edges, int ** out_degrees, int * nodes_count, int * edges_count) {
    int row, prev_row, row_size, min_row_len, max_row_len, row_len_interval, empty_pieces, empty_rows;
    int * row_len;
    
    // sort edges
    qsort(*edges, *edges_count, 2 * sizeof(int), edge_compare);

    // allocate space for JDS
    mJDS->num_pieces = (*num_pieces);
    mJDS->num_cols = (*nodes_count);
    mJDS->num_nonzeros = 0;
    mJDS->num_elements = 0;
    mJDS->pieces = (mtx_ELL **) malloc((*num_pieces) * sizeof(mtx_ELL *));
    mJDS->row_ind = (int **) malloc((*num_pieces)*sizeof(int *));
    (*dangling) = (int *) calloc((*nodes_count), sizeof(int));
    if(mJDS->pieces == NULL || mJDS->row_ind == NULL || (*dangling) == NULL)  {
        printf("Could not allocate space for JDS matrix.\n");
        return 1;
    }

    // compute row length distribution
    row_len = (int *) calloc((*nodes_count), sizeof(int));
    if(row_len == NULL)  {
        printf("Could not allocate space for JDS matrix.\n");
        return 1;
    }

    prev_row = 0;
    row_size = 0;
    min_row_len = (*nodes_count);
    max_row_len = 0;
    for (int i = 0; i < (*edges_count); i++) {
        row = (*edges)[i][1];
        if(row > prev_row) {
            row_len[prev_row] = row_size;
            if(row_size > 0) {
                if(max_row_len < row_size)
                    max_row_len = row_size;
                if(min_row_len > row_size)
                    min_row_len = row_size;
            }
            row_size = 1;
            prev_row = row;
        } else
            row_size++;
    }
    row_len[row] = row_size;
    if(row_size > 0) {
        if(max_row_len < row_size)
             max_row_len = row_size;
        if(min_row_len > row_size)
            min_row_len = row_size;
    }

    for(int i = 0; i < (*nodes_count); i++)
        (*dangling)[i] = (row_len[i] == 0);

    row_len_interval = max_row_len - min_row_len + 1;

    int min_row_limit_p, max_row_limit_p, max_row_len_p, row_count_p, el_count_p, r_ind;
    long long ell_index;
    int * row_ind_p;
    mtx_ELL * mELL_p;
    empty_pieces = 0;
    for(int p = 0; p < (*num_pieces); p++) {
        min_row_limit_p = min_row_len + (p*row_len_interval)/(*num_pieces);
        if(min_row_limit_p == 0)
            min_row_limit_p = 1; // ignore empty rows
        max_row_limit_p = min_row_len + ((p+1)*row_len_interval)/(*num_pieces) - 1;

        row_count_p = 0;
        el_count_p = 0;
        max_row_len_p = 0;
        // count number of rows, elements, and maximum row length for piece
        for(int i = 0; i < (*nodes_count); i++) {
            if(row_len[i] >= min_row_limit_p && row_len[i] <= max_row_limit_p) {
                row_count_p++;
                el_count_p += row_len[i];
                if(max_row_len_p < row_len[i])
                    max_row_len_p = row_len[i];
            }
        }


        mELL_p = (mtx_ELL *) malloc(sizeof(mtx_ELL));
        mELL_p->num_nonzeros = el_count_p;
        mELL_p->num_rows = row_count_p;
        mELL_p->num_cols = (*nodes_count);
        mELL_p->num_elementsinrow = max_row_len_p;
        mELL_p->num_elements = mELL_p->num_rows * mELL_p->num_elementsinrow;
        mJDS->pieces[p] = mELL_p;
        mJDS->num_nonzeros += el_count_p;
        mJDS->num_elements += mELL_p->num_elements;

        if(el_count_p == 0) { // empty section
            empty_pieces++;
            continue;
        }

        // allocate ELL piece and row vector
        mELL_p->data = (float *) calloc(mELL_p->num_elements, sizeof(float));
        mELL_p->col = (int *) calloc(mELL_p->num_elements, sizeof(int));
        row_ind_p = (int *) calloc(mELL_p->num_rows, sizeof(int));

        if(mELL_p->data == NULL || mELL_p->col == NULL || row_ind_p == NULL)  {
            printf("Could not allocate space for JDS matrix.\n");
            return 1;
        }
        mJDS->row_ind[p] = row_ind_p;

        
        prev_row = 0;
        row_size = 0;
        r_ind = -1;
        // copy edges from corresponding rows to ELL structures and compute values
        for (int i = 0; i < (*edges_count); i++) {
            row  = (*edges)[i][1];
            
            if(row_len[row] >= min_row_limit_p && row_len[row] <= max_row_limit_p) {

                if(row > prev_row) {
                    row_size = 1;
                    prev_row = row;
                    r_ind++; // Increments from -1 when switching to first relevant row
                    row_ind_p[r_ind] = row;
                } else
                    row_size++;

                if(r_ind == -1) {
                    r_ind = 0; // This fixes the piece with the first row (if any)
                    row_ind_p[r_ind] = row;
                }

                ell_index = (row_size - 1) * mELL_p->num_rows + r_ind;

                mELL_p->col[ell_index] = (*edges)[i][0];

                // If no outgoing links, return error (current edge is outgoing)
                if((*out_degrees)[(*edges)[i][0]] == 0) {
                    printf("Inconsistency in data: outgoing edge for node %d with 0 out-degree.\n", (*edges)[i][0]);
                    return 1;
                } else // Otherwise transition to one of outgoing links randomly                          
                    mELL_p->data[ell_index] = 1./(*out_degrees)[(*edges)[i][0]];

            } else if(row > prev_row) {
                row_size = 1;
                prev_row = row;
            } else
                row_size++;
        }
    }

    free(row_len);

    // remove empty pieces
    if(empty_pieces > 0) {

        mtx_ELL ** pieces2 = (mtx_ELL **) malloc(((*num_pieces) - empty_pieces) * sizeof(mtx_ELL *));
        int ** row_ind2 = (int **) malloc(((*num_pieces) - empty_pieces) * sizeof(int *));
        if(pieces2 == NULL || row_ind2 == NULL)  {
            printf("Could not allocate space for JDS matrix.\n");
            return 1;
        }

        int i = 0;
        for(int p = 0; p < mJDS->num_pieces; p++) {
            if(mJDS->pieces[p]->num_nonzeros > 0) {
                pieces2[i] = mJDS->pieces[p];
                row_ind2[i] = mJDS->row_ind[p];
                i++;
            }
        }

        free(mJDS->pieces);
        free(mJDS->row_ind);
        mJDS->num_pieces -= empty_pieces;
        mJDS->pieces = pieces2;
        mJDS->row_ind = row_ind2;

    }

    return 0;
}

int get_JDS_from_file(struct mtx_JDS * mJDS, int ** dangling, int * num_pieces, char * file_name) {
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

    int status = get_JDS_from_edges(mJDS, dangling, num_pieces, &edges, &out_degrees, &nodes_count, &edges_count);

    // NOTE: contiguous_space from read_edges is not freed (memory leak)
    free(edges);
    free(out_degrees);
    free(in_degrees);

    return status;
}


int mtx_JDS_free(struct mtx_JDS *mJDS)
{
    for(int p = 0; p < mJDS->num_pieces; p++) {
        free(mJDS->pieces[p]->col);
        free(mJDS->pieces[p]->data);
        free(mJDS->pieces[p]);
        free(mJDS->row_ind[p]);
    }
    free(mJDS->pieces);
    free(mJDS->row_ind);

    return 0;
}

#endif
