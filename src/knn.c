
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <cblas.h>

#include "knn.h"
#include "utils.h"

#define BLOCK_SIZE 1000

knnresult make_knnresult(int m, int k) {
   knnresult ret;

   ret.m = m;
   ret.k = k;
   ret.nidx = (int*) malloc(m*k*sizeof(int));
   ret.ndist = (double*) malloc(m*k*sizeof(double));

   return ret;
}

void free_knnresult(knnresult result) {
    free(result.nidx);
    free(result.ndist);
}

void print_knnresult(knnresult result) {
    printf("m: %d\n", result.m);
    printf("k: %d\n", result.k);
    
    for (int i = 0; i < result.m; i++) {
        printf("\tpoint index: %d\n", i);
        for (int j = 0; j < result.k; j++) {
            printf("\t\tidx: %d, distance: %lf\n", result.nidx[i*result.k+j], result.ndist[i*result.k+j]);
        }
    }
}

// combines two knnresults in-place. \param to is extended to include the points from \param from
void combine_knnresults(knnresult* to, knnresult from) {
    assert(to->k == from.k);

    int m_total = to->m+from.m;
    to->nidx = realloc(to->nidx, m_total*to->k*sizeof(int));
    to->ndist = realloc(to->ndist, m_total*to->k*sizeof(double));
    memcpy(to->nidx+to->m*to->k, from.nidx, from.m*from.k*sizeof(int));
    memcpy(to->ndist+to->m*to->k, from.ndist, from.m*from.k*sizeof(double));
    to->m = m_total;
}

void merge_knnresults(knnresult* to, knnresult from) {
    assert(to->k == from.k);
    assert(to->m == from.m);
    int k = to->k;

    /*
    For each point i in [0, m):
    1) Concatenate i-th point's "from" and "to" k-NN to a temporary array
    2) quick-select that temporary array
    3) Write the k first elements of the array back to "to" knnresult
    */
    double* dist = (double*) malloc(2*k*sizeof(double));
    int* idx = (int*) malloc(2*k*sizeof(int));
    for (int i = 0; i < to->m; i++) {
        memcpy(dist, to->ndist+i*k, k*sizeof(double));
        memcpy(dist+k, from.ndist+i*k, k*sizeof(double));
        memcpy(idx, to->nidx+i*k, k*sizeof(int));
        memcpy(idx+k, from.nidx+i*k, k*sizeof(int));

        qselect(dist, idx, k, 0, 2*k);

        memcpy(to->ndist+i*k, dist, k*sizeof(double));
        memcpy(to->nidx+i*k, idx, k*sizeof(int));
    }
    free(dist);
    free(idx);
}

// private function that actually computes kNN of a single block
knnresult kNN_(double * X, double * Y, int n, int m, int d, int k, int offx) {
    knnresult result = make_knnresult(m,k);
    double* d_mat = dist_mat(X, Y, n, m, d);
    // tmp vars so that d_mat is not modified
    double* dist_cpy = (double*) malloc(n*sizeof(double));
    int* dist_idx = (int*) malloc(n*sizeof(int));

    for (int i = 0; i < m; i++) {
        // Initially, copy distances to result as-is
        for (int j = 0; j < n; j++) {
            dist_cpy[j] = d_mat[j*m+i]; // i-th column in the matrix
            dist_idx[j] = j+offx;
        }
        // qselect
        qselect(dist_cpy, dist_idx, k, 0, n);

        // copy k first elems to result
        memcpy(result.ndist+i*k, dist_cpy, k*sizeof(double));
        memcpy(result.nidx+i*k, dist_idx, k*sizeof(int));
    }

    free(dist_cpy);
    free(dist_idx);
    free(d_mat);

    return result;
}

knnresult kNN_off(double * X, double * Y, int n, int m, int d, int k, int offx) {
    int block_offset = 0;

    knnresult first_res;
    int blk_size = 0;
    if (block_offset+2*BLOCK_SIZE >= m) {
        // near end, m-offset block size
        blk_size = m-block_offset;
    } else {
        blk_size = BLOCK_SIZE;
    }
    first_res = kNN_(X, Y+block_offset*d, n, blk_size, d, k, offx);
    block_offset += blk_size;

    while (block_offset < m) {
        if (block_offset+2*BLOCK_SIZE >= m) {
            // near end, m-offset block size
            blk_size = m-block_offset;

        } else {
            blk_size = BLOCK_SIZE;
        }

        knnresult res = kNN_(X, Y+block_offset*d, n, blk_size, d, k, offx);
        // combine with first result
        combine_knnresults(&first_res, res);
        free_knnresult(res);

        block_offset += blk_size;
    }
    return first_res;
}

knnresult kNN(double * X, double * Y, int n, int m, int d, int k) {
    return kNN_off(X, Y, n, m, d, k, 0);
}

/// computes distance matrix
/// allocates n*m*sizeof(double)
/// caller is responsible for freeing allocated memory
double* dist_mat(double * X, double * Y, int n, int m, int d) {
    double* d_mat = (double*) malloc(n*m*sizeof(double));
    for (int i = 0; i < n*m; i++) d_mat[i] = 0;
    
    double* x_sq_norm = (double*) malloc(n*m*sizeof(double));
    for (int i = 0; i < n*m; i++) x_sq_norm[i] = 0;

    // X hadamard X
    double* x_sq = (double*) malloc(n*d*sizeof(double));
    for (int i = 0; i < n*d; i++) {
        x_sq[i] = X[i]*X[i];
    }

    // e*e^T aka sum-broadcast
    double* e_et = (double*) malloc(d*m*sizeof(double)); 
    for (int i = 0; i < d*m; i++) {
        e_et[i] = 1;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, d, 1, x_sq, d, e_et, m,
                0, x_sq_norm, m);
    free(x_sq);
    free(e_et);

    double* y_sq_norm = (double*) malloc(n*m*sizeof(double));
    for (int i = 0; i < n*m; i++) y_sq_norm[i] = 0;

    // Y hadamard Y transpose
    double* y_sq = (double*) malloc(d*m*sizeof(double));
    for (int i = 0; i < d*m; i++) {
        int row = i/m;
        int col = i%m;
        int i_t = col*d+row;
        y_sq[i] = Y[i_t]*Y[i_t];
    }

    // e^T*e aka broadcast-sum
    double* et_e = (double*) malloc(n*d*sizeof(double)); 
    for (int i = 0; i < n*d; i++) {
        et_e[i] = 1;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, d, 1, et_e, d, y_sq, m,
                0, y_sq_norm, m);
    free(y_sq);
    free(et_e);

    for (int i = 0; i < n*m; i++) {
        d_mat[i] = x_sq_norm[i] + y_sq_norm[i];
    }
    free(x_sq_norm);
    free(y_sq_norm);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d,
                1, d_mat, m);

    return d_mat;
}
