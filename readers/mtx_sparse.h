#include <stdio.h>
#include <stdlib.h>

struct mtx_COO  // COOrdinates
{
    int *row;
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

struct mtx_CSR  // Compressed Sparse Row
{
    int *rowptr;
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

struct mtx_ELL      // ELLiptic (developed by authors of ellipctic package)
{
    int *col;
    float *data;
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int num_elements;
    int num_elementsinrow;    
};

struct mtx_MM
{
    int row;
    int col;
    float data;
};

int mtx_COO_compare(const void * a, const void * b) { 
    struct mtx_MM aa = *(struct mtx_MM *)a;
    struct mtx_MM bb = *(struct mtx_MM *)b;

    if (aa.row < bb.row)
        return -1;
    else if (aa.row > bb.row)
        return +1;
    else if (aa.col < bb.col)
        return -1;
    else if (aa.col > bb.col)
        return +1;
    else 
        return 0;
}

int mtx_COO_create_from_file(struct mtx_COO * mCOO, char * file_name) {
    int nodes_count, edges_count;
    FILE * fp;

    fp = fopen(file_name, "r");
    if (fscanf(fp, "%d\t%d", &nodes_count, &edges_count) != 2)
        return 1;

    // allocate matrix
    struct mtx_MM *mMM = (struct mtx_MM *)malloc(edges_count * sizeof(struct mtx_MM));
    mCOO->data = (float *) malloc(edges_count * sizeof(float));
    mCOO->col = (int *) malloc(edges_count * sizeof(int));
    mCOO->row = (int *) malloc(edges_count * sizeof(int));

    // read data
    for (int i = 0; i < edges_count; i++) {
        if(fscanf(fp, "%d\t%d\n", &mMM[i].row, &mMM[i].col));
        mMM[i].data = 1.0; // think how to normalize later this value...
    }
    fclose(fp);

    // sort elements
    qsort(mMM, edges_count, sizeof(struct mtx_MM), mtx_COO_compare);

    // copy to mtx_COO structures (GPU friendly)
    for (int i = 0; i < edges_count; i++) {
        mCOO->data[i] = mMM[i].data;
        mCOO->row[i] = mMM[i].row;
        mCOO->col[i] = mMM[i].col;
    }

    free(mMM);

    return 0;
}

int mtx_COO_free(struct mtx_COO *mCOO)
{
    free(mCOO->data);
    free(mCOO->col);
    free(mCOO->row);

    return 0;
}

int mtx_CSR_create_from_mtx_COO(struct mtx_CSR *mCSR, struct mtx_COO *mCOO) {
    mCSR->num_nonzeros = mCOO->num_nonzeros;
    mCSR->num_rows = mCOO->num_rows;
    mCSR->num_cols = mCOO->num_cols;

    mCSR->data =  (float *)malloc(mCSR->num_nonzeros * sizeof(float));
    mCSR->col = (int *)malloc(mCSR->num_nonzeros * sizeof(int));
    mCSR->rowptr = (int *)calloc(mCSR->num_rows + 1, sizeof(int));
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

int mtx_CSR_free(struct mtx_CSR *mCSR) {
    free(mCSR->data);
    free(mCSR->col);
    free(mCSR->rowptr);

    return 0;
}

int mtx_ELL_create_from_mtx_CSR(struct mtx_ELL *mELL, struct mtx_CSR *mCSR) {
    mELL->num_nonzeros = mCSR->num_nonzeros;
    mELL->num_rows = mCSR->num_rows;
    mELL->num_cols = mCSR->num_cols;
    mELL->num_elementsinrow = 0;

    for (int i = 0; i < mELL->num_rows; i++)
        if (mELL->num_elementsinrow < mCSR->rowptr[i+1]-mCSR->rowptr[i]) 
            mELL->num_elementsinrow = mCSR->rowptr[i+1]-mCSR->rowptr[i];
    mELL->num_elements = mELL->num_rows * mELL->num_elementsinrow;
    mELL->data = (float *)calloc(mELL->num_elements, sizeof(float));
    mELL->col = (int *) calloc(mELL->num_elements, sizeof(int));    
    for (int i = 0; i < mELL->num_rows; i++)
    {
        for (int j = mCSR->rowptr[i]; j < mCSR->rowptr[i+1]; j++)
        {            
            int ELL_j = (j - mCSR->rowptr[i]) * mELL->num_rows + i;
            mELL->data[ELL_j] = mCSR->data[j];
            mELL->col[ELL_j] = mCSR->col[j];
        }
    }

    return 0;
}

int mtx_ELL_free(struct mtx_ELL *mELL)
{
    free(mELL->col);
    free(mELL->data);

    return 0;
}